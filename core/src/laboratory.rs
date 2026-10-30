//! Reproducible native/Wasm laboratory contract: flat LCDM or EdS,
//! physical initial spectra, exact observation meshing, and discrete PM tangents.
use crate::{
    cosmological,
    fft::FftEngine,
    grid::Grid,
    growth::{growth_rate, Cosmology},
    ics::Preset,
    nbody::{cosmology_clocks, evolve_cosmology, PmCrossing, PmRun, PmState, PmTangent},
    observables, sheet, spectra, tangent, Cosmo,
};
use num_complex::Complex64;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::{
    ops::Deref,
    sync::{Arc, OnceLock},
};

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(default)]
pub struct LabRequest {
    pub dim: usize,
    pub n: usize,
    pub fraction: f64,
    pub order: usize,
    pub seed: u64,
    pub ic: String,
    pub smoothing: f64,
    pub steps: usize,
    pub mesh: usize,
    #[serde(rename = "forceMesh")]
    pub force_mesh: usize,
    #[serde(rename = "omegaM")]
    pub omega_m: f64,
    #[serde(rename = "omegaB")]
    pub omega_b: f64,
    pub h: f64,
    pub ns: f64,
    pub sigma8: f64,
    #[serde(rename = "boxSize")]
    pub box_size: f64,
    #[serde(rename = "initialSmoothing")]
    pub initial_smoothing: f64,
    #[serde(rename = "initialCutoff")]
    pub initial_cutoff: usize,
    pub epoch: String,
    #[serde(rename = "scaleFactor")]
    pub scale_factor: f64,
    /// Linear transfer fit for Gaussian initial modes: "smooth" (EH98
    /// non-oscillatory) or "bao" (EH98 with baryon acoustic oscillations).
    pub transfer: String,
    /// Optional explicit list of trajectory:estimator pairs (e.g. "4lpt:p0",
    /// "3lpt:p1", "pm:cic") requested by the interactive comparison matrix.
    #[serde(default)]
    pub estimators: Vec<String>,
    /// Optional extended-envelope flag unlocking larger 2D/3D browser runs
    /// (up to 2048 in 1D/2D and n<=128, mesh/forceMesh<=256 in 3D, steps<=1024).
    #[serde(default)]
    pub extended: bool,
}
impl Default for LabRequest {
    fn default() -> Self {
        Self {
            dim: 2,
            n: 32,
            fraction: 0.5,
            order: 4,
            seed: 7,
            ic: "waves".into(),
            smoothing: 0.03,
            steps: 64,
            mesh: 32,
            force_mesh: 32,
            omega_m: 0.315,
            omega_b: 0.049,
            h: 0.674,
            ns: 0.965,
            sigma8: 0.811,
            box_size: 300.,
            initial_smoothing: 14.,
            initial_cutoff: 3,
            epoch: "fraction".into(),
            scale_factor: 0.5,
            transfer: "smooth".into(),
            estimators: Vec::new(),
            extended: false,
        }
    }
}
impl LabRequest {
    /// Whether any PM trajectory/estimator is requested. Empty `estimators`
    /// (the default in standalone/test requests) keeps PM enabled; when the
    /// interactive matrix explicitly supplies `estimators`, PM is only needed
    /// if at least one `pm:*` entry is present.
    pub fn needs_pm(&self) -> bool {
        self.estimators.is_empty() || self.estimators.iter().any(|s| s.starts_with("pm:"))
    }

    /// Conservative allocation estimate including LPT tensors, tangents, FFT
    /// workspaces, result arrays and JSON serialization. Not a measured peak.
    pub fn estimated_peak_bytes(&self) -> usize {
        let size = self.n.saturating_pow(self.dim.min(3) as u32);
        let mesh = self
            .mesh
            .max(self.force_mesh)
            .saturating_pow(self.dim.min(3) as u32);
        let per_vertex = if self.dim < 3 { 750 } else { 2400 };
        let per_cell = if self.dim < 3 { 200 } else { 250 };
        size.saturating_mul(per_vertex)
            .saturating_add(mesh.saturating_mul(per_cell))
    }
    pub(crate) fn validate(&self) -> Result<(), String> {
        const GRIDS: &[usize] = &[16, 24, 32, 48, 64, 96, 128, 256, 512, 1024];
        const EXT_GRIDS: &[usize] = &[16, 24, 32, 48, 64, 96, 128, 256, 512, 1024, 2048];
        let grids = if self.extended { EXT_GRIDS } else { GRIDS };
        if !(1..=3).contains(&self.dim)
            || !grids.contains(&self.n)
            || !grids.contains(&self.mesh)
            || !grids.contains(&self.force_mesh)
            || !(1..=4).contains(&self.order)
        {
            return Err("Invalid dimension/order or grid; supported grids are 16,24,32,48,64,96,128,256,512,1024".into());
        }
        let cap = if self.extended {
            [0, 2048, 2048, 128][self.dim]
        } else {
            [0, 1024, 1024, 64][self.dim]
        };
        let mesh_cap = if self.extended {
            [0, 2048, 2048, 256][self.dim]
        } else {
            [0, 1024, 1024, 128][self.dim]
        };
        let over_budget = !self.extended && self.estimated_peak_bytes() > 1024 * 1024 * 1024;
        if self.n > cap
            || self.mesh > mesh_cap
            || self.force_mesh > mesh_cap
            || over_budget
        {
            return Err(format!("Requested {}D allocation exceeds the browser laboratory envelope (lattice <={}, meshes <={}, conservative estimate {:.0} MiB, budget 1024 MiB). Reduce a grid or use the Rust building blocks for a larger native calculation; requested grids were not changed.",self.dim,cap,mesh_cap,self.estimated_peak_bytes() as f64/1048576.));
        }
        if !self.fraction.is_finite() || !(0.05..=1.5).contains(&self.fraction) {
            return Err("Reference growth fraction must be in [0.05,1.5]".into());
        }
        if !["fraction", "scaleFactor"].contains(&self.epoch.as_str())
            || !self.scale_factor.is_finite()
            || !(0.01..=2.).contains(&self.scale_factor)
        {
            return Err("Epoch must be fraction or scaleFactor, with a in [0.01,2]".into());
        }
        let valid_steps = if self.extended {
            [32, 64, 128, 256, 512, 1024].contains(&self.steps)
        } else {
            [32, 64, 128, 256, 512].contains(&self.steps)
        };
        if !self.smoothing.is_finite()
            || !(0.0..=0.15).contains(&self.smoothing)
            || !valid_steps
        {
            return Err("Invalid observation smoothing or timesteps".into());
        }
        for (name, value, lo, hi) in [
            ("omegaM", self.omega_m, 0.05, 1.),
            ("omegaB", self.omega_b, 0., 0.2),
            ("h", self.h, 0.4, 1.),
            ("ns", self.ns, 0.8, 1.2),
            ("sigma8", self.sigma8, 0.1, 1.5),
            ("boxSize", self.box_size, 50., 2000.),
            ("initialSmoothing", self.initial_smoothing, 0., 200.),
        ] {
            if !value.is_finite() || value < lo || value > hi {
                return Err(format!("{name} must be finite in [{lo},{hi}]"));
            }
        }
        if self.omega_b / self.omega_m > 0.5 {
            return Err("Smooth EH transfer requires omegaB/omegaM <=0.5".into());
        }
        if self.ic != "waves" && self.ic != "gaussian" {
            return Err("Unknown initial-condition family".into());
        }
        if self.transfer != "smooth" && self.transfer != "bao" {
            return Err("Transfer fit must be smooth (EH98 no-wiggle) or bao (EH98 with acoustic oscillations)".into());
        }
        if self.seed > 9999 {
            return Err("Seed must be in [0,9999]".into());
        }
        Ok(())
    }
}
/// Construct the same Fourier realization at every supported refinement,
/// without building the CPU LPT expansion. This is the shared seed used by
/// native sessions and by the optional GPU LPT initializer.
pub fn initial_cosmo(r: &LabRequest) -> Cosmo {
    let mut c = Cosmo::new(r.dim, r.n, 1.0);
    c.keep_tensors = Some(true);
    c.set_cosmology(r.omega_m);
    if r.ic == "waves" {
        let mut modes = vec![([1, 0, 0], 0.5, 0.17)];
        if r.dim >= 2 {
            modes.extend([([0, 1, 0], 0.35, 0.43), ([1, 1, 0], 0.15, 0.7)]);
        }
        if r.dim == 3 {
            modes.extend([([0, 0, 1], 0.3, 0.87), ([1, 0, 1], 0.1, 0.5)]);
        }
        c.set_ic_preset(Preset::PlaneWaves { modes }, 0.0, 0.0);
    } else {
        let growth = crate::growth::Growth::new_lcdm(
            c.cosmo,
            &crate::growth::term_specs(1, r.dim),
            2.,
            4000,
        );
        let d1 = if c.cosmo.is_eds() {
            1.
        } else {
            growth.d_of_a(1.)
        };
        c.delta0_hat = cosmological::gaussian_hat(
            &c.grid,
            r.seed,
            r.initial_cutoff,
            r.box_size,
            r.omega_m,
            r.omega_b,
            r.h,
            r.ns,
            r.sigma8,
            r.initial_smoothing,
            d1,
            r.transfer == "bao",
        );
        c.sigma0 = c
            .delta0_hat
            .iter()
            .map(|v| v.norm_sqr())
            .sum::<f64>()
            .sqrt()
            / c.grid.size as f64;
    }
    c
}

/// Compatibility constructor for callers that need the CPU 4LPT fields.
pub fn initial_conditions(r: &LabRequest) -> Cosmo {
    let mut c = initial_cosmo(r);
    c.build_lpt(4);
    c
}
/// Double-precision evaluation of the inherited spatial terms. The terms
/// themselves retain upstream f32 precision, but time sums and amplitudes do not.
pub(crate) fn evaluate(c: &Cosmo, d: f64, order: usize, amp: f64) -> (Vec<f64>, Vec<f64>) {
    let l = c.lpt_ref();
    let mut x = vec![0.; c.grid.size * c.grid.dim];
    let mut v = x.clone();
    for id in 0..c.grid.size {
        let ijk = c.grid.unravel(id);
        for k in 0..c.grid.dim {
            x[id * c.grid.dim + k] = ijk[k] as f64 * c.grid.dx();
        }
    }
    for (i, t) in l.terms.iter().enumerate() {
        if t.spec.order > order {
            continue;
        }
        let (g, dg) = l.growth.g_and_dg(i, d);
        let scale = amp.powi(t.spec.order as i32);
        for id in 0..c.grid.size {
            for k in 0..c.grid.dim {
                x[id * c.grid.dim + k] += scale * g * t.psi[k][id] as f64;
                v[id * c.grid.dim + k] += scale * dg * t.psi[k][id] as f64;
            }
        }
    }
    (x, v)
}
fn density(g: &Grid, x: &[f64], mesh: usize) -> Vec<f32> {
    if g.dim == 1 {
        return observables::interval_density(g, x, mesh);
    }
    let pos: Vec<f32> = x.iter().map(|&v| v as f32).collect();
    if g.dim == 2 {
        sheet::sheet_density_2d_exact(g, &pos, None, mesh)
    } else {
        sheet::sheet_density_3d_exact(g, &pos, None, mesh)
    }
}
/// Logarithmic bin count for the isotropic spectra: 12 on small meshes, up to 48
/// on the largest, so that acoustic features remain visible where the mesh resolves them.
pub fn spectrum_bins(mesh: usize) -> usize {
    (mesh / 4).clamp(12, 48)
}
fn power(g: &Grid, rho: &[f32]) -> spectra::Spectrum {
    let mut eng = FftEngine::new(g);
    let contrast: Vec<f32> = rho.iter().map(|&v| v - 1.).collect();
    spectra::power_spectrum(g, &mut eng, &contrast, spectrum_bins(g.n))
}
/// LPT vertex positions at growth `d` and truncation `order` (amplitude = 1).
pub fn evaluate_lpt_positions(c: &Cosmo, d: f64, order: usize) -> Vec<f64> {
    evaluate(c, d, order, 1.0).0
}

/// The linear power of *this realization*, D^2 |delta0_k|^2, averaged over the same
/// isotropic bins as the measured fields, for bins lying entirely inside the radial
/// cutoff (where the mesh modes and the input modes coincide). Unit-box units.
pub fn realized_linear_power_bins(
    c: &Cosmo,
    eg: &Grid,
    d: f64,
    nbins: usize,
) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    realized_linear_bins_with_count(c, eg, d, nbins)
}

fn realized_linear_bins_with_count(
    c: &Cosmo,
    eg: &Grid,
    d: f64,
    nbins: usize,
) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let edges = spectra::bin_edges(eg, nbins);
    let lk0 = edges[0].ln();
    let dlk = (edges[nbins] / edges[0]).ln() / nbins as f64;
    let bin_of = |k: f64| -> Option<usize> {
        if k < edges[0] || k >= edges[nbins] {
            None
        } else {
            Some((((k.ln() - lk0) / dlk).floor() as usize).min(nbins - 1))
        }
    };
    let mut cnt = vec![0.0; nbins];
    let mut ksum = vec![0.0; nbins];
    for idx in 1..eg.size {
        let k = eg.k2(idx).sqrt();
        if let Some(b) = bin_of(k) {
            cnt[b] += 1.;
            ksum[b] += k;
        }
    }
    let g = &c.grid;
    let mut psum = vec![0.0; nbins];
    for idx in 1..g.size {
        let ijk = g.unravel(idx);
        if (0..g.dim).any(|a| g.n % 2 == 0 && ijk[a] == g.n / 2) {
            continue;
        }
        if let Some(b) = bin_of(g.k2(idx).sqrt()) {
            psum[b] += (d * c.delta0_hat[idx].norm() / g.size as f64).powi(2);
        }
    }
    let limit = g.knyq().min(eg.knyq());
    let mut ks = Vec::new();
    let mut ps = Vec::new();
    let mut counts = Vec::new();
    for b in 0..nbins {
        if cnt[b] > 0. && edges[b + 1] <= limit * (1. + 1e-9) {
            ks.push(ksum[b] / cnt[b]);
            ps.push(psum[b] / cnt[b]);
            counts.push(cnt[b]);
        }
    }
    (ks, ps, counts)
}
fn mean(v: &[f32]) -> f64 {
    v.iter().map(|&x| x as f64).sum::<f64>() / v.len() as f64
}
/// Execute a reproducible experiment. Allocation/parameter failures return errors
/// before allocating fields. Orientation failures produce a caution and retain results.
/// Live PDFs use Gaussian observation smoothing and raw mass weights. Power
/// spectra use the unsmoothed cell-average field. Derivative modes are first-axis
/// continuous P0 Fourier modes, at fixed time under an IC amplitude change.
pub struct PreparedLabRun {
    request: LabRequest,
    result: Value,
    modes: Vec<[i32; 4]>,
    h: f64,
    lpt_displacement: Vec<f32>,
    lpt_direction: Vec<f32>,
    pm_displacement: Vec<f32>,
    pm_direction: Vec<f32>,
    // Retain the f64 reference state for CPU fallback and finite differences.
    q: Vec<f64>,
    x: Vec<f64>,
    v: Vec<f64>,
    dpos: Vec<f64>,
    xplus: Vec<f64>,
    xminus: Vec<f64>,
    pmx: Vec<f64>,
    pmv: Vec<f64>,
    pmdpos: Vec<f64>,
    rs: Vec<f32>,
    rs_p1: Vec<f32>,
    pms: Vec<f32>,
    cics: Vec<f32>,
    tscs: Vec<f32>,
    custom_rhos: Vec<(String, Vec<f32>)>,
    omit_large_vertex_json: bool,
    omit_large_mesh_json: bool,
    lpt_displacement_expansion: OnceLock<(Vec<f32>, Vec<f32>)>,
    lpt_direction_expansion: OnceLock<(Vec<f32>, Vec<f32>)>,
    cache_status: Value,
    fd_channels: OnceLock<Vec<f64>>,
    fd_plus_expansion: OnceLock<(Vec<f32>, Vec<f32>)>,
    fd_minus_expansion: OnceLock<(Vec<f32>, Vec<f32>)>,
}

/// Shared f64 physical preparation before the final PM evolution. The PM seed,
/// tangent and clocks remain available when a GPU attempt fails, so fallback
/// can finish this exact run without rebuilding the realization or LPT fields.
pub struct PendingLabRun {
    request: LabRequest,
    c: Arc<Cosmo>,
    initial_source: String,
    geometry: Arc<PendingGeometry>,
    pm_reference_cache: OnceLock<Result<Arc<PmRun>, String>>,
    pm_reference_quantized_cache: OnceLock<Result<Arc<PmRun>, String>>,
    pm_reference_displacement_quantized_cache: OnceLock<Result<Arc<PmRun>, String>>,
    cache_status: Value,
}

/// Request-specific, expensive geometry and PM seed. Sharing this Arc allows
/// observation-only requests to update their manifest without copying fields.
pub struct PendingGeometry {
    a: f64,
    a_start: f64,
    d: f64,
    d_da: f64,
    sc: f64,
    minj: f64,
    mins: f64,
    x: Vec<f64>,
    v: Vec<f64>,
    dpos: Vec<f64>,
    xplus: Vec<f64>,
    xminus: Vec<f64>,
    pm_initial: PmState,
    pm_initial_tangent: PmTangent,
    pm_initial_min_jacobian: f64,
    pm_initial_quantized_min_jacobian: f64,
    pm_initial_displacement_quantized_min_jacobian: f64,
    pm_clocks: Vec<f64>,
}

impl Deref for PendingLabRun {
    type Target = PendingGeometry;
    fn deref(&self) -> &Self::Target {
        &self.geometry
    }
}

/// A deliberately small cache for interactive laboratory sessions. Each tier
/// retains one complete entry, so memory use is bounded and replacement is
/// predictable. Failed calculations are never inserted.
pub struct LabSession {
    initial: Option<(String, String, Arc<Cosmo>, usize)>,
    pending: Option<(String, Arc<PendingGeometry>, usize)>,
    pm: Option<(String, String, Arc<PmRun>, usize)>,
    observations: Option<(String, Arc<RawObservations>, usize)>,
    last_status: Value,
}

impl Default for LabSession {
    fn default() -> Self {
        Self {
            initial: None,
            pending: None,
            pm: None,
            observations: None,
            last_status: json!({"initial":{"status":"empty"},"pm":{"status":"empty"},"bytes":0}),
        }
    }
}

const SESSION_CACHE_LIMIT_BYTES: usize = 1536 * 1024 * 1024;

pub struct PreparedEstimatorUpdate {
    json: String,
    custom_rhos: Vec<(String, Vec<f32>)>,
}

impl PreparedEstimatorUpdate {
    pub fn json(&self) -> &str {
        &self.json
    }
    pub fn custom_rho_f32(&self, key: &str) -> Vec<f32> {
        self.custom_rhos
            .iter()
            .find(|(k, _)| k == key)
            .map(|(_, v)| v.clone())
            .unwrap_or_default()
    }
}

fn skipped_pm_run(pending: &PendingLabRun) -> PmRun {
    let a = pending.a;
    let scale_p = a.powi(3) * pending.c.cosmo.e(a);
    PmRun {
        state: PmState {
            x: pending.x.clone(),
            p: pending.v.iter().map(|&v| v * scale_p).collect(),
            a,
        },
        tangent: Some(PmTangent {
            dx: pending.dpos.clone(),
            dp: vec![0.0; pending.dpos.len()],
        }),
        min_jacobian: pending.mins,
        completed_steps: 0,
        crossed: false,
        first_crossing: None,
    }
}

struct RawObservations {
    rho: Vec<f32>,
    rho_p1: Vec<f32>,
    pmrho: Vec<f32>,
    cic: Vec<f32>,
    tsc: Vec<f32>,
    ps: spectra::Spectrum,
    ps_p1: spectra::Spectrum,
    psc: spectra::Spectrum,
    pst: spectra::Spectrum,
    psp: spectra::Spectrum,
    direct_modes: Vec<usize>,
    direct_powers: Vec<f64>,
    direct_pm_powers: Vec<f64>,
    direct_shell_k: Vec<f64>,
    direct_shell_lpt: Vec<f64>,
    direct_shell_pm: Vec<f64>,
    direct_shell_realized_k: Vec<f64>,
    direct_shell_realized: Vec<f64>,
    deposit_source: String,
}

impl RawObservations {
    fn estimated_bytes(&self) -> usize {
        (self.rho.len() + self.rho_p1.len() + self.pmrho.len() + self.cic.len() + self.tsc.len())
            * std::mem::size_of::<f32>()
            + (self.ps.p.len()
                + self.ps_p1.p.len()
                + self.psc.p.len()
                + self.pst.p.len()
                + self.psp.p.len())
                * std::mem::size_of::<f64>()
                * 3
            + (self.direct_modes.len()
                + self.direct_powers.len()
                + self.direct_pm_powers.len()
                + self.direct_shell_k.len()
                + self.direct_shell_lpt.len()
                + self.direct_shell_pm.len()
                + self.direct_shell_realized_k.len()
                + self.direct_shell_realized.len())
                * std::mem::size_of::<f64>()
            + self.deposit_source.capacity()
    }
}

fn meshfree_primitive_rays(dim: usize, grid_size: usize) -> &'static [[i32; 3]] {
    const RAYS_1D: [[i32; 3]; 1] = [[1, 0, 0]];
    const RAYS_2D_FULL: [[i32; 3]; 16] = [
        [1, 0, 0],
        [0, 1, 0],
        [1, 1, 0],
        [1, -1, 0],
        [2, 1, 0],
        [1, 2, 0],
        [2, -1, 0],
        [1, -2, 0],
        [3, 1, 0],
        [1, 3, 0],
        [3, -1, 0],
        [1, -3, 0],
        [3, 2, 0],
        [2, 3, 0],
        [3, -2, 0],
        [2, -3, 0],
    ];
    const RAYS_3D_FULL: [[i32; 3]; 13] = [
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
        [1, 1, 0],
        [1, -1, 0],
        [1, 0, 1],
        [1, 0, -1],
        [0, 1, 1],
        [0, 1, -1],
        [1, 1, 1],
        [1, 1, -1],
        [1, -1, 1],
        [-1, 1, 1],
    ];
    match dim {
        1 => &RAYS_1D,
        2 => &RAYS_2D_FULL,
        _ => {
            if grid_size > 32_768 {
                &RAYS_3D_FULL[..7]
            } else {
                &RAYS_3D_FULL
            }
        }
    }
}

/// Shell-averaged mesh-free exact simplex Fourier power spectra for LPT and PM
/// sheets across primitive lattice direction rays, binned into logarithmic |k| shells.
fn compute_meshfree_shell_spectra(
    pending: &PendingLabRun,
    pm_x: &[f64],
) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
    let r = &pending.request;
    let g = &pending.c.grid;
    let ref_n = r.mesh.max(r.n);
    let ref_grid = Grid::new(r.dim, ref_n, 1.0);
    let nbins = spectrum_bins(ref_n);
    let edges = spectra::bin_edges(&ref_grid, nbins);
    let lk0 = edges[0].ln();
    let dlk = (edges[nbins] / edges[0]).ln() / nbins as f64;
    let bin_of = |k: f64| -> Option<usize> {
        if k < edges[0] * (1.0 - 1e-9) || k > edges[nbins] * (1.0 + 1e-9) {
            None
        } else {
            let b = ((k.ln() - lk0) / dlk).floor().max(0.0) as usize;
            Some(b.min(nbins - 1))
        }
    };
    let base_sub_n = match r.dim {
        1 => 1024,
        2 => 128,
        _ => {
            if g.n <= 24 {
                24
            } else {
                16
            }
        }
    };
    let eff_base_n = if g.n > base_sub_n {
        let s = (1..=g.n)
            .find(|&s| g.n % s == 0 && g.n / s <= base_sub_n)
            .unwrap_or(1);
        g.n / s
    } else {
        g.n
    };
    let base_nyq_m = (eff_base_n / 2).max(1);
    let rays = meshfree_primitive_rays(r.dim, g.size);
    let mut cnt = vec![0.0_f64; nbins];
    let mut ksum = vec![0.0_f64; nbins];
    let mut lpt_sum = vec![0.0_f64; nbins];
    let mut pm_sum = vec![0.0_f64; nbins];
    let mut real_cnt = vec![0.0_f64; nbins];
    let mut real_ksum = vec![0.0_f64; nbins];
    let mut real_psum = vec![0.0_f64; nbins];
    let kf = g.kf();
    let mut accumulate_ray = |dir: [i32; 3], ray_modes: &[usize], sub_n: usize| {
        if ray_modes.is_empty() {
            return;
        }
        let dir_norm =
            ((dir[0] * dir[0] + dir[1] * dir[1] + dir[2] * dir[2]) as f64).sqrt();
        let f_lpt =
            observables::sheet_modes_along_ray(g, &pending.x, dir, ray_modes, sub_n);
        let f_pm = if r.needs_pm() {
            observables::sheet_modes_along_ray(g, pm_x, dir, ray_modes, sub_n)
        } else {
            f_lpt.clone()
        };
        for (idx_m, &m) in ray_modes.iter().enumerate() {
            let k = kf * (m as f64) * dir_norm;
            if let Some(b) = bin_of(k) {
                cnt[b] += 1.0;
                ksum[b] += k;
                lpt_sum[b] += f_lpt[idx_m].norm_sqr();
                pm_sum[b] += f_pm[idx_m].norm_sqr();
                if r.ic == "gaussian"
                    && (0..r.dim)
                        .all(|a| (m as i32 * dir[a]).unsigned_abs() as usize * 2 < g.n)
                {
                    let mut ijk = [0usize; 3];
                    for a in 0..r.dim {
                        ijk[a] = (m as i32 * dir[a]).rem_euclid(g.n as i32) as usize;
                    }
                    let grid_idx = g.ravel(ijk);
                    real_cnt[b] += 1.0;
                    real_ksum[b] += k;
                    real_psum[b] += (pending.d * pending.c.delta0_hat[grid_idx].norm()
                        / g.size as f64)
                        .powi(2);
                }
            }
        }
    };
    for &dir in rays {
        let dir_norm =
            ((dir[0] * dir[0] + dir[1] * dir[1] + dir[2] * dir[2]) as f64).sqrt();
        let dir_inf = dir[0]
            .unsigned_abs()
            .max(dir[1].unsigned_abs())
            .max(dir[2].unsigned_abs())
            .max(1) as usize;
        let m_bz_max = (base_nyq_m / dir_inf).max(1);
        let m_edge_max = ((edges[nbins] / (kf * dir_norm)).floor() as usize).max(1);
        let m_max = m_edge_max.min(m_bz_max);
        let mut ray_modes: Vec<usize> = (1..=m_max.min(32)).collect();
        if m_max > 32 {
            for b in 0..nbins {
                let m_lo = ((edges[b] / (kf * dir_norm)).ceil() as usize).max(1);
                let m_hi = ((edges[b + 1] / (kf * dir_norm)).floor() as usize).min(m_max);
                if m_lo <= m_hi {
                    ray_modes.push(m_lo);
                    ray_modes.push((m_lo + m_hi) / 2);
                    ray_modes.push(m_hi);
                }
            }
            ray_modes.push(m_max);
            ray_modes.sort_unstable();
            ray_modes.dedup();
        }
        accumulate_ray(dir, &ray_modes, base_sub_n);
    }
    if r.dim == 2 && g.n >= 256 {
        let fine_sub_n = 256usize;
        let fine_nyq_m = fine_sub_n / 2;
        for &dir in &rays[..4] {
            let dir_norm =
                ((dir[0] * dir[0] + dir[1] * dir[1] + dir[2] * dir[2]) as f64).sqrt();
            let m_edge_max = ((edges[nbins] / (kf * dir_norm)).floor() as usize).max(1);
            let m_max = m_edge_max.min(fine_nyq_m);
            if m_max > base_nyq_m {
                let mut hi_modes = Vec::new();
                for b in 0..nbins {
                    let m_mid =
                        (((edges[b] * edges[b + 1]).sqrt() / (kf * dir_norm)).round() as usize)
                            .clamp(1, m_max);
                    if m_mid > base_nyq_m {
                        hi_modes.push(m_mid);
                    }
                }
                hi_modes.push(m_max);
                hi_modes.sort_unstable();
                hi_modes.dedup();
                accumulate_ray(dir, &hi_modes, fine_sub_n);
            }
        }
    }
    let mut shell_k = Vec::new();
    let mut shell_lpt = Vec::new();
    let mut shell_pm = Vec::new();
    let mut shell_real_k = Vec::new();
    let mut shell_real_p = Vec::new();
    for b in 0..nbins {
        if cnt[b] > 0.0 {
            shell_k.push(ksum[b] / cnt[b]);
            shell_lpt.push(lpt_sum[b] / cnt[b]);
            shell_pm.push(pm_sum[b] / cnt[b]);
        }
        if real_cnt[b] > 0.0 {
            shell_real_k.push(real_ksum[b] / real_cnt[b]);
            shell_real_p.push(real_psum[b] / real_cnt[b]);
        }
    }
    (shell_k, shell_lpt, shell_pm, shell_real_k, shell_real_p)
}

/// Shell-averaged mesh-free exact simplex Fourier power spectrum for a single
/// trajectory `pos` at the same logarithmic |k| shells as `compute_meshfree_shell_spectra`.
fn compute_meshfree_shell_single(pending: &PendingLabRun, pos: &[f64]) -> Vec<f64> {
    let r = &pending.request;
    let g = &pending.c.grid;
    let ref_n = r.mesh.max(r.n);
    let ref_grid = Grid::new(r.dim, ref_n, 1.0);
    let nbins = spectrum_bins(ref_n);
    let edges = spectra::bin_edges(&ref_grid, nbins);
    let lk0 = edges[0].ln();
    let dlk = (edges[nbins] / edges[0]).ln() / nbins as f64;
    let bin_of = |k: f64| -> Option<usize> {
        if k < edges[0] * (1.0 - 1e-9) || k > edges[nbins] * (1.0 + 1e-9) {
            None
        } else {
            let b = ((k.ln() - lk0) / dlk).floor().max(0.0) as usize;
            Some(b.min(nbins - 1))
        }
    };
    let base_sub_n = match r.dim {
        1 => 1024,
        2 => 128,
        _ => {
            if g.n <= 24 {
                24
            } else {
                16
            }
        }
    };
    let eff_base_n = if g.n > base_sub_n {
        let s = (1..=g.n)
            .find(|&s| g.n % s == 0 && g.n / s <= base_sub_n)
            .unwrap_or(1);
        g.n / s
    } else {
        g.n
    };
    let base_nyq_m = (eff_base_n / 2).max(1);
    let rays = meshfree_primitive_rays(r.dim, g.size);
    let mut cnt = vec![0.0_f64; nbins];
    let mut p_sum = vec![0.0_f64; nbins];
    let kf = g.kf();
    let mut accumulate_ray = |dir: [i32; 3], ray_modes: &[usize], sub_n: usize| {
        if ray_modes.is_empty() {
            return;
        }
        let dir_norm =
            ((dir[0] * dir[0] + dir[1] * dir[1] + dir[2] * dir[2]) as f64).sqrt();
        let f_modes = observables::sheet_modes_along_ray(g, pos, dir, ray_modes, sub_n);
        for (idx_m, &m) in ray_modes.iter().enumerate() {
            let k = kf * (m as f64) * dir_norm;
            if let Some(b) = bin_of(k) {
                cnt[b] += 1.0;
                p_sum[b] += f_modes[idx_m].norm_sqr();
            }
        }
    };
    for &dir in rays {
        let dir_norm =
            ((dir[0] * dir[0] + dir[1] * dir[1] + dir[2] * dir[2]) as f64).sqrt();
        let dir_inf = dir[0]
            .unsigned_abs()
            .max(dir[1].unsigned_abs())
            .max(dir[2].unsigned_abs())
            .max(1) as usize;
        let m_bz_max = (base_nyq_m / dir_inf).max(1);
        let m_edge_max = ((edges[nbins] / (kf * dir_norm)).floor() as usize).max(1);
        let m_max = m_edge_max.min(m_bz_max);
        let mut ray_modes: Vec<usize> = (1..=m_max.min(32)).collect();
        if m_max > 32 {
            for b in 0..nbins {
                let m_lo = ((edges[b] / (kf * dir_norm)).ceil() as usize).max(1);
                let m_hi = ((edges[b + 1] / (kf * dir_norm)).floor() as usize).min(m_max);
                if m_lo <= m_hi {
                    ray_modes.push(m_lo);
                    ray_modes.push((m_lo + m_hi) / 2);
                    ray_modes.push(m_hi);
                }
            }
            ray_modes.push(m_max);
            ray_modes.sort_unstable();
            ray_modes.dedup();
        }
        accumulate_ray(dir, &ray_modes, base_sub_n);
    }
    if r.dim == 2 && g.n >= 256 {
        let fine_sub_n = 256usize;
        let fine_nyq_m = fine_sub_n / 2;
        for &dir in &rays[..4] {
            let dir_norm =
                ((dir[0] * dir[0] + dir[1] * dir[1] + dir[2] * dir[2]) as f64).sqrt();
            let m_edge_max = ((edges[nbins] / (kf * dir_norm)).floor() as usize).max(1);
            let m_max = m_edge_max.min(fine_nyq_m);
            if m_max > base_nyq_m {
                let mut hi_modes = Vec::new();
                for b in 0..nbins {
                    let m_mid =
                        (((edges[b] * edges[b + 1]).sqrt() / (kf * dir_norm)).round() as usize)
                            .clamp(1, m_max);
                    if m_mid > base_nyq_m {
                        hi_modes.push(m_mid);
                    }
                }
                hi_modes.push(m_max);
                hi_modes.sort_unstable();
                hi_modes.dedup();
                accumulate_ray(dir, &hi_modes, fine_sub_n);
            }
        }
    }
    let mut shell_p = Vec::new();
    for b in 0..nbins {
        if cnt[b] > 0.0 {
            shell_p.push(p_sum[b] / cnt[b]);
        }
    }
    shell_p
}

/// First-axis integer mode numbers m for the mesh-free simplex Fourier spectrum
/// k = (m k_f, 0, 0): starts with 1..=direct_mode_count(r), extends at least as
/// far in k as the gridded shell-averaged FFT spectrum, and uses at least as
/// many k points as the gridded shell average.
fn spectrum_axis_modes(r: &LabRequest, ps_k: &[f64]) -> Vec<usize> {
    let base_count = direct_mode_count(r);
    let kf = 2.0 * std::f64::consts::PI;
    let max_ps_m = ps_k
        .iter()
        .copied()
        .fold(0.0_f64, f64::max)
        .div_euclid(kf) as usize
        + 1;
    let max_grid_m = (((r.mesh.max(r.n) as f64) * 0.5 * (r.dim as f64).sqrt()).ceil() as usize).max(1);
    let m_max = max_grid_m.max(max_ps_m).max(base_count);
    let target_bins = spectrum_bins(r.mesh.max(r.n))
        .max(ps_k.len())
        .max(24);
    let linear_top = base_count.max(12).min(m_max);
    let mut modes: Vec<usize> = (1..=linear_top).collect();
    if m_max > linear_top {
        if m_max <= target_bins + 8 {
            modes.extend((linear_top + 1)..=m_max);
        } else {
            let log_count = target_bins;
            let lo = (linear_top as f64).ln();
            let hi = (m_max as f64).ln();
            for i in 1..=log_count {
                let m = (lo + (hi - lo) * (i as f64 / log_count as f64))
                    .exp()
                    .round() as usize;
                modes.push(m.clamp(linear_top + 1, m_max));
            }
            modes.push(m_max);
        }
    }
    modes.sort_unstable();
    modes.dedup();
    modes
}

/// CIC and TSC particle deposits of the LPT vertices treated as equal-mass
/// point particles: the conventional estimators that the exact sheet replaces.
fn particle_deposits(r: &LabRequest, g: &Grid, x: &[f64]) -> (Vec<f32>, Vec<f32>) {
    let pos: Vec<f32> = x.iter().map(|&v| v as f32).collect();
    (
        sheet::cic_density(r.dim, &pos, g.size, 1.0, r.mesh),
        sheet::tsc_density(r.dim, &pos, g.size, 1.0, r.mesh),
    )
}

fn raw_observations(pending: &PendingLabRun, pm: &PmRun) -> RawObservations {
    let r = &pending.request;
    let g = &pending.c.grid;
    let pm_active = pm.completed_steps > 0 && r.needs_pm();
    let (rho, rho_p1) = pending.lpt_density_p0_and_p1();
    let pmrho = if pm_active {
        density(g, &pm.state.x, r.mesh)
    } else {
        rho.clone()
    };
    let eg = Grid::new(r.dim, r.mesh, 1.0);
    let (cic, tsc) = particle_deposits(r, g, &pending.x);
    let ps = power(&eg, &rho);
    let ps_p1 = power(&eg, &rho_p1);
    let psc = power(&eg, &cic);
    let pst = power(&eg, &tsc);
    let psp = if pm_active {
        power(&eg, &pmrho)
    } else {
        ps.clone()
    };
    let direct_modes = spectrum_axis_modes(r, &ps.k);
    let direct_powers: Vec<f64> = observables::sheet_modes_first_axis(g, &pending.x, &direct_modes)
        .into_iter()
        .map(|f| f.norm_sqr())
        .collect();
    let direct_pm_powers = if pm_active {
        observables::sheet_modes_first_axis(g, &pm.state.x, &direct_modes)
            .into_iter()
            .map(|f| f.norm_sqr())
            .collect()
    } else {
        direct_powers.clone()
    };
    let (
        direct_shell_k,
        direct_shell_lpt,
        direct_shell_pm,
        direct_shell_realized_k,
        direct_shell_realized,
    ) = compute_meshfree_shell_spectra(pending, &pm.state.x);
    RawObservations {
        rho,
        rho_p1,
        pmrho,
        cic,
        tsc,
        ps,
        ps_p1,
        psc,
        pst,
        psp,
        direct_modes,
        direct_powers,
        direct_pm_powers,
        direct_shell_k,
        direct_shell_lpt,
        direct_shell_pm,
        direct_shell_realized_k,
        direct_shell_realized,
        deposit_source: "cpu-exact".into(),
    }
}

fn validate_observed_density(r: &LabRequest, name: &str, values: &[f32]) -> Result<(), String> {
    let expected = r
        .mesh
        .checked_pow(r.dim as u32)
        .ok_or_else(|| "Observation mesh size overflow".to_string())?;
    if values.len() != expected {
        return Err(format!(
            "{name} must contain {expected} unsmoothed density cells"
        ));
    }
    if values.iter().any(|x| !x.is_finite() || *x < 0.0) {
        return Err(format!(
            "{name} must contain finite nonnegative mass densities"
        ));
    }
    let mean = values.iter().map(|&x| x as f64).sum::<f64>() / expected as f64;
    if !mean.is_finite() || (mean - 1.0).abs() > 1e-6 {
        return Err(format!(
            "{name} mass mean must be within 1e-6 of unity (got {mean:.9})"
        ));
    }
    Ok(())
}

fn raw_observations_external(
    pending: &PendingLabRun,
    pm_x: &[f64],
    rho: &[f32],
    pmrho: &[f32],
    deposit_source: &str,
) -> Result<RawObservations, String> {
    validate_observed_density(&pending.request, "rho", rho)?;
    validate_observed_density(&pending.request, "pmrho", pmrho)?;
    if deposit_source.is_empty() {
        return Err("External density provenance must be nonempty".into());
    }
    let r = &pending.request;
    let g = &pending.c.grid;
    let rho_p1 = pending.lpt_density_p1();
    let (cic, tsc) = particle_deposits(r, g, &pending.x);
    let eg = Grid::new(r.dim, r.mesh, 1.0);
    let ps = power(&eg, rho);
    let direct_modes = spectrum_axis_modes(r, &ps.k);
    let direct_powers = observables::sheet_modes_first_axis(g, &pending.x, &direct_modes)
        .into_iter()
        .map(|f| f.norm_sqr())
        .collect();
    let direct_pm_powers = observables::sheet_modes_first_axis(g, pm_x, &direct_modes)
        .into_iter()
        .map(|f| f.norm_sqr())
        .collect();
    let (
        direct_shell_k,
        direct_shell_lpt,
        direct_shell_pm,
        direct_shell_realized_k,
        direct_shell_realized,
    ) = compute_meshfree_shell_spectra(pending, pm_x);
    Ok(RawObservations {
        rho: rho.to_vec(),
        ps_p1: power(&eg, &rho_p1),
        rho_p1,
        pmrho: pmrho.to_vec(),
        ps,
        psc: power(&eg, &cic),
        pst: power(&eg, &tsc),
        psp: power(&eg, pmrho),
        cic,
        tsc,
        direct_modes,
        direct_powers,
        direct_pm_powers,
        direct_shell_k,
        direct_shell_lpt,
        direct_shell_pm,
        direct_shell_realized_k,
        direct_shell_realized,
        deposit_source: deposit_source.into(),
    })
}

/// Compute the exact Rust reference deposit for an externally generated
/// geometry. Positions are the packed f32 values crossing the JS boundary.
pub fn exact_sheet_density(
    dim: usize,
    n: usize,
    mesh: usize,
    packed_positions: &[f32],
) -> Result<Vec<f32>, String> {
    if !(1..=3).contains(&dim) {
        return Err("Density reference dimension must be 1, 2, or 3".into());
    }
    let cap = [0, 2048, 2048, 256][dim];
    if n < 2 || n > cap || mesh == 0 || mesh > cap {
        return Err("Density reference grid exceeds the supported dimension limits".into());
    }
    let nv = n
        .checked_pow(dim as u32)
        .ok_or_else(|| "Density reference grid size overflow".to_string())?;
    let expected = nv
        .checked_mul(4)
        .ok_or_else(|| "Density reference position size overflow".to_string())?;
    if packed_positions.len() != expected {
        return Err(format!(
            "Packed density positions must contain {expected} f32 values"
        ));
    }
    if packed_positions.iter().any(|v| !v.is_finite()) {
        return Err("Packed density positions must be finite".into());
    }
    let mut positions = vec![0.0f64; nv * dim];
    for i in 0..nv {
        for a in 0..dim {
            positions[i * dim + a] = packed_positions[4 * i + a] as f64;
        }
    }
    Ok(density(&Grid::new(dim, n, 1.0), &positions, mesh))
}

fn initial_cache_key(r: &LabRequest, source: &str) -> String {
    json!({"dim":r.dim,"n":r.n,"ic":r.ic,"seed":r.seed,
        "omegaM":r.omega_m.to_bits(),"omegaB":r.omega_b.to_bits(),"h":r.h.to_bits(),
        "ns":r.ns.to_bits(),"sigma8":r.sigma8.to_bits(),"boxSize":r.box_size.to_bits(),
        "initialSmoothing":r.initial_smoothing.to_bits(),"initialCutoff":r.initial_cutoff,
        "provenance":source})
    .to_string()
}

fn pm_cache_key(pending: &PendingLabRun, arithmetic: &str) -> String {
    json!({"initial":initial_cache_key(&pending.request, &pending.initial_source),"a":pending.a.to_bits(),
        "steps":pending.request.steps,"forceMesh":pending.request.force_mesh,
        "omegaM":pending.request.omega_m.to_bits(),"arithmetic":arithmetic,
        "initialization":pending.initial_source})
    .to_string()
}

fn pending_cache_key(r: &LabRequest, source: &str) -> String {
    json!({"initial":initial_cache_key(r, source),"epoch":r.epoch,
        "fraction":r.fraction.to_bits(),"scaleFactor":r.scale_factor.to_bits(),
        "order":r.order,"steps":r.steps,"forceMesh":r.force_mesh,
        "omegaM":r.omega_m.to_bits()})
    .to_string()
}

fn observation_cache_key(
    pending: &PendingLabRun,
    arithmetic: &str,
    deposit_source: &str,
    pm_computed: bool,
) -> String {
    json!({"trajectory":pm_cache_key(pending, arithmetic),"order":pending.request.order,
        "mesh":pending.request.mesh,"depositSource":deposit_source,"pmComputed":pm_computed})
    .to_string()
}

fn cosmo_cache_bytes(c: &Cosmo) -> usize {
    let fft_plan_estimate = c
        .grid
        .n
        .saturating_mul((usize::BITS - c.grid.n.leading_zeros()) as usize)
        .saturating_mul(std::mem::size_of::<Complex64>())
        .saturating_mul(2);
    let mut bytes = c.delta0_hat.capacity() * std::mem::size_of::<Complex64>()
        + c.phi0.capacity() * std::mem::size_of::<f64>()
        + c.grid.n * std::mem::size_of::<Complex64>() // FFT scratch estimate
        + fft_plan_estimate;
    if let Some(lpt) = &c.lpt {
        for term in &lpt.terms {
            bytes += term.psi.iter().map(|v| v.capacity() * 4).sum::<usize>();
            bytes += term
                .m
                .as_ref()
                .map_or(0, |m| m.iter().map(|v| v.capacity() * 4).sum());
            bytes += term.div.capacity() * 4;
            bytes += term.curl.iter().map(|v| v.capacity() * 4).sum::<usize>();
        }
        bytes += match &lpt.growth {
            crate::growth::Growth::Eds { coef, orders } => {
                coef.capacity() * 8 + orders.capacity() * std::mem::size_of::<usize>()
            }
            crate::growth::Growth::Lcdm {
                lna, g, gp, gpp, ..
            } => {
                lna.capacity() * 8
                    + g.iter()
                        .chain(gp)
                        .chain(gpp)
                        .map(|v| v.capacity() * 8)
                        .sum::<usize>()
            }
        };
    }
    bytes
}

fn pending_geometry_bytes(g: &PendingGeometry) -> usize {
    let vec = |v: &Vec<f64>| v.capacity() * std::mem::size_of::<f64>();
    vec(&g.x)
        + vec(&g.v)
        + vec(&g.dpos)
        + vec(&g.xplus)
        + vec(&g.xminus)
        + vec(&g.pm_initial.x)
        + vec(&g.pm_initial.p)
        + vec(&g.pm_initial_tangent.dx)
        + vec(&g.pm_initial_tangent.dp)
        + g.pm_clocks.capacity() * std::mem::size_of::<f64>()
}

fn pm_cache_bytes(r: &LabRequest) -> usize {
    r.n.saturating_pow(r.dim as u32)
        .saturating_mul(r.dim.saturating_mul(8).saturating_mul(4).saturating_add(64))
}

impl LabSession {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn clear(&mut self) {
        self.initial = None;
        self.pending = None;
        self.pm = None;
        self.observations = None;
        self.last_status = json!({"initial":{"status":"cleared"},"pm":{"status":"cleared"},"observations":{"status":"cleared"},"bytes":0});
    }

    pub fn cache_status(&self) -> Value {
        self.last_status.clone()
    }

    fn retained_bytes(&self) -> usize {
        self.initial
            .as_ref()
            .map_or(0, |entry| entry.3)
            .saturating_add(self.pending.as_ref().map_or(0, |entry| entry.2))
            .saturating_add(self.pm.as_ref().map_or(0, |entry| entry.3))
            .saturating_add(self.observations.as_ref().map_or(0, |entry| entry.2))
    }

    fn retained_bytes_without_pm(&self) -> usize {
        self.initial
            .as_ref()
            .map_or(0, |entry| entry.3)
            .saturating_add(self.pending.as_ref().map_or(0, |entry| entry.2))
            .saturating_add(self.observations.as_ref().map_or(0, |entry| entry.2))
    }

    /// Evict a previous realization before a caller constructs an external
    /// initial state. GPU seed builders should call this before allocating the
    /// new Cosmo, then pass its Arc to `adopt_initial`.
    pub fn prepare_initial_slot(&mut self, r: &LabRequest, source: &str) -> Result<bool, String> {
        r.validate()?;
        validate_initial_source(source)?;
        let key = initial_cache_key(r, source);
        let hit = self
            .initial
            .as_ref()
            .is_some_and(|(old, old_source, _, _)| old == &key && old_source == source);
        if !hit {
            self.clear();
            self.last_status = json!({"initial":{"status":"miss","reason":"slot cleared for new initialization","provenance":source},"pm":{"status":"empty"},"pending":{"status":"empty"},"observations":{"status":"empty"},"bytes":0,"limitBytes":SESSION_CACHE_LIMIT_BYTES});
        }
        Ok(hit)
    }

    pub fn prepare_pending(&mut self, r: &LabRequest) -> Result<PendingLabRun, String> {
        self.prepare_pending_source(r, "cpu-f64")
    }

    pub fn prepare_pending_source(
        &mut self,
        r: &LabRequest,
        source: &str,
    ) -> Result<PendingLabRun, String> {
        r.validate()?;
        validate_initial_source(source)?;
        let key = initial_cache_key(r, source);
        let existing = self
            .initial
            .as_ref()
            .and_then(|(cached_key, cached_source, c, bytes)| {
                (cached_key == &key && cached_source == source).then(|| (Arc::clone(c), *bytes))
            });
        if let Some((c, bytes)) = existing {
            return self.prepare_from_initial(
                r,
                c,
                source,
                json!({"status":"hit","bytes":bytes,"provenance":source}),
                true,
            );
        }
        if source == "gpu-f32" {
            return Err("No adopted gpu-f32 initialization is cached for this request".into());
        }
        // Release the previous dataset before allocating another Cosmo.
        self.clear();
        let c = Arc::new(initial_conditions(r));
        self.prepare_from_initial(r, c, source, json!({"status":"miss","reason":"initialization key changed or empty","provenance":source}), false)
    }

    pub fn adopt_initial(
        &mut self,
        r: &LabRequest,
        c: Arc<Cosmo>,
        source: &str,
    ) -> Result<PendingLabRun, String> {
        r.validate()?;
        validate_initial_source(source)?;
        if c.grid.dim != r.dim
            || c.grid.n != r.n
            || (c.cosmo.omega_m - r.omega_m).abs() > 1e-12
            || c.lpt_ref().order < 4
        {
            return Err("Adopted initial state does not match the validated request or lacks order-four LPT fields".into());
        }
        let key = initial_cache_key(r, source);
        let cached = self
            .initial
            .as_ref()
            .and_then(|(old, old_source, cached, bytes)| {
                (old == &key && old_source == source).then(|| (Arc::clone(cached), *bytes))
            });
        if let Some((cached, bytes)) = cached {
            return self.prepare_from_initial(r, cached, source, json!({"status":"hit","bytes":bytes,"reason":"adopted initialization","provenance":source}), true);
        }
        self.clear();
        self.prepare_from_initial(
            r,
            c,
            source,
            json!({"status":"miss","reason":"adopted initialization","provenance":source}),
            false,
        )
    }

    fn prepare_from_initial(
        &mut self,
        r: &LabRequest,
        c: Arc<Cosmo>,
        source: &str,
        initial_status: Value,
        initial_hit: bool,
    ) -> Result<PendingLabRun, String> {
        let initial_key = initial_cache_key(r, source);
        let geometry_key = pending_cache_key(r, source);
        let cosmo_bytes = cosmo_cache_bytes(&c);
        let geometry_hit = self.pending.as_ref().and_then(|(key, geometry, bytes)| {
            (key == &geometry_key).then(|| (Arc::clone(geometry), *bytes))
        });
        let mut pending = if let Some((geometry, bytes)) = geometry_hit {
            pending_from_geometry(
                r,
                Arc::clone(&c),
                source,
                geometry,
                json!({"initial":initial_status,"pending":{"status":"hit","bytes":bytes,"key":"epoch/order/steps/force"}}),
            )
        } else {
            self.pending = None;
            let computed = prepare_pending_from_initial_source(r, &c, source)?;
            let geometry = Arc::clone(&computed.geometry);
            let bytes = pending_geometry_bytes(&geometry);
            let total = cosmo_bytes
                .saturating_add(bytes)
                .saturating_add(self.pm.as_ref().map_or(0, |entry| entry.3))
                .saturating_add(self.observations.as_ref().map_or(0, |entry| entry.2));
            if total <= SESSION_CACHE_LIMIT_BYTES {
                self.pending = Some((geometry_key, geometry, bytes));
            }
            let mut computed = computed;
            computed.cache_status["initial"] = initial_status;
            computed.cache_status["pending"] = if total <= SESSION_CACHE_LIMIT_BYTES {
                json!({"status":"miss","reason":"computed","bytes":bytes})
            } else {
                json!({"status":"skipped","reason":"session memory budget","bytes":bytes})
            };
            computed
        };
        if !initial_hit {
            let total = cosmo_bytes
                .saturating_add(self.pending.as_ref().map_or(0, |entry| entry.2))
                .saturating_add(self.pm.as_ref().map_or(0, |entry| entry.3))
                .saturating_add(self.observations.as_ref().map_or(0, |entry| entry.2));
            if total <= SESSION_CACHE_LIMIT_BYTES {
                self.initial = Some((initial_key, source.into(), Arc::clone(&c), cosmo_bytes));
                pending.cache_status["initial"]["bytes"] = json!(cosmo_bytes);
            } else {
                pending.cache_status["initial"] = json!({"status":"skipped","reason":"session memory budget","bytes":cosmo_bytes,"provenance":source});
            }
        }
        let total = self
            .initial
            .as_ref()
            .map_or(0, |entry| entry.3)
            .saturating_add(self.pending.as_ref().map_or(0, |entry| entry.2))
            .saturating_add(self.pm.as_ref().map_or(0, |entry| entry.3))
            .saturating_add(self.observations.as_ref().map_or(0, |entry| entry.2));
        pending.cache_status["bytes"] = json!(total.min(SESSION_CACHE_LIMIT_BYTES));
        pending.cache_status["limitBytes"] = json!(SESSION_CACHE_LIMIT_BYTES);
        self.last_status = pending.cache_status.clone();
        Ok(pending)
    }

    /// Return a prepared result when a completed PM trajectory matches the
    /// current request and arithmetic. Smoothing and observation mesh remain
    /// request-specific and are assembled from the retained trajectory.
    pub fn try_cached(
        &mut self,
        r: &LabRequest,
        arithmetic: &str,
    ) -> Result<Option<PreparedLabRun>, String> {
        self.try_cached_source(r, arithmetic, "cpu-f64")
    }

    pub fn try_cached_source(
        &mut self,
        r: &LabRequest,
        arithmetic: &str,
        initial_source: &str,
    ) -> Result<Option<PreparedLabRun>, String> {
        self.try_cached_observed_source(r, arithmetic, initial_source, "cpu-exact")
    }

    /// Reassemble a cached trajectory only when its unsmoothed deposit
    /// provenance also matches. External deposits must already be retained;
    /// this path never silently computes a CPU exact deposit for them.
    pub fn try_cached_observed_source(
        &mut self,
        r: &LabRequest,
        arithmetic: &str,
        initial_source: &str,
        deposit_source: &str,
    ) -> Result<Option<PreparedLabRun>, String> {
        if !["f64", "f32"].contains(&arithmetic) {
            return Err("PM arithmetic must be f64 or f32".into());
        }
        if deposit_source.is_empty() {
            return Err("Density deposit provenance must be nonempty".into());
        }
        r.validate()?;
        validate_initial_source(initial_source)?;
        if initial_source == "gpu-f32"
            && !self.initial.as_ref().is_some_and(|(key, source, _, _)| {
                key == &initial_cache_key(r, initial_source) && source == initial_source
            })
        {
            self.last_status = json!({"initial":{"status":"miss","reason":"gpu-f32 initialization must be adopted before preparation","provenance":initial_source},"pm":{"status":"miss","reason":"initialization unavailable","arithmetic":arithmetic},"bytes":0,"limitBytes":SESSION_CACHE_LIMIT_BYTES});
            return Ok(None);
        }
        let pending = self.prepare_pending_source(r, initial_source)?;
        let key = pm_cache_key(&pending, arithmetic);
        let cached = self
            .pm
            .as_ref()
            .and_then(|(cached_key, cached_arithmetic, pm, bytes)| {
                (cached_key == &key && cached_arithmetic == arithmetic)
                    .then(|| (Arc::clone(pm), *bytes))
            });
        if let Some((pm, bytes)) = cached {
            let pm_computed = pm.completed_steps > 0 && r.needs_pm();
            if deposit_source != "cpu-exact"
                && !self
                    .observations
                    .as_ref()
                    .is_some_and(|(cached_key, _, _)| {
                        cached_key
                            == &observation_cache_key(
                                &pending,
                                arithmetic,
                                deposit_source,
                                pm_computed,
                            )
                    })
            {
                self.last_status = json!({"initial":{"status":"hit"},"pending":{"status":"hit"},"pm":{"status":"hit","arithmetic":arithmetic,"bytes":bytes},"observations":{"status":"miss","reason":"requested external deposit is not cached","depositSource":deposit_source},"bytes":self.retained_bytes().min(SESSION_CACHE_LIMIT_BYTES),"limitBytes":SESSION_CACHE_LIMIT_BYTES});
                return Ok(None);
            }
            let mut hit = pending;
            hit.cache_status["pm"] = json!({"status":"hit","arithmetic":arithmetic,"bytes":bytes});
            self.last_status = hit.cache_status.clone();
            return self
                .assemble_cached(&mut hit, Arc::as_ref(&pm), arithmetic, deposit_source, None)
                .map(Some);
        }
        if !r.needs_pm() && deposit_source == "cpu-exact" {
            let pm = skipped_pm_run(&pending);
            let mut hit = pending;
            hit.cache_status["pm"] = json!({"status":"skipped","reason":"no PM trajectory selected","arithmetic":arithmetic,"bytes":0});
            self.last_status = hit.cache_status.clone();
            let mut prepared = self.assemble_cached(&mut hit, &pm, arithmetic, deposit_source, None)?;
            self.refresh_status(&mut prepared);
            return Ok(Some(prepared));
        }
        let mut miss = pending;
        miss.cache_status["pm"] =
            json!({"status":"miss","reason":"trajectory key changed","arithmetic":arithmetic});
        self.last_status = miss.cache_status.clone();
        Ok(None)
    }

    /// Incrementally evaluate only the requested `r.estimators` (`1lpt..4lpt`, `pm`
    /// × `p0`, `p1`, `ngp`, `cic`, `tsc`) using the already-retained `initial`
    /// (which stores all 1–4LPT tensors) and `pm` (when `r.needs_pm()` is true),
    /// without re-running LPT FFTs, PM evolution, RSD analysis, or direct/FD modes.
    pub fn compute_estimators_cached(
        &mut self,
        r: &LabRequest,
        arithmetic: &str,
        initial_source: &str,
    ) -> Result<Option<PreparedEstimatorUpdate>, String> {
        if !["f64", "f32"].contains(&arithmetic) {
            return Err("PM arithmetic must be f64 or f32".into());
        }
        r.validate()?;
        validate_initial_source(initial_source)?;
        if !self.initial.as_ref().is_some_and(|(key, source, _, _)| {
            key == &initial_cache_key(r, initial_source) && source == initial_source
        }) {
            return Ok(None);
        }
        let pending = self.prepare_pending_source(r, initial_source)?;
        let key = pm_cache_key(&pending, arithmetic);
        let cached_pm = self
            .pm
            .as_ref()
            .and_then(|(cached_key, cached_arithmetic, pm, _)| {
                (cached_key == &key && cached_arithmetic == arithmetic).then(|| Arc::clone(pm))
            });
        if r.needs_pm() && cached_pm.is_none() {
            return Ok(None);
        }
        let pm = cached_pm.unwrap_or_else(|| Arc::new(skipped_pm_run(&pending)));
        let obs_key = observation_cache_key(&pending, arithmetic, "cpu-exact", pm.completed_steps > 0 && r.needs_pm());
        let raw_opt = self
            .observations
            .as_ref()
            .and_then(|(k, raw, _)| (k == &obs_key).then(|| Arc::clone(raw)));
        let (custom_series, custom_rhos) =
            compute_custom_series_standalone(&pending, Arc::as_ref(&pm), raw_opt.as_deref());
        Ok(Some(PreparedEstimatorUpdate {
            json: json!({ "customSeries": custom_series }).to_string(),
            custom_rhos,
        }))
    }

    /// Return the absolute PM positions for the pending trajectory when its
    /// arithmetic-specific result is already retained. This lets callers
    /// build a new observation-mesh deposit without rerunning PM evolution.
    pub fn cached_pm_positions(
        &self,
        pending: &PendingLabRun,
        arithmetic: &str,
    ) -> Option<Vec<f32>> {
        self.pm
            .as_ref()
            .and_then(|(key, cached_arithmetic, pm, _)| {
                (cached_arithmetic == arithmetic && key == &pm_cache_key(pending, arithmetic))
                    .then(|| packed_vec4(&pending.c.grid, &pm.state.x, false))
            })
    }

    /// Finalize an already retained trajectory with CPU-exact observations.
    /// This reuses the original PM result and never rematerializes its state.
    pub fn finish_cached(
        &mut self,
        pending: &mut PendingLabRun,
        arithmetic: &str,
    ) -> Result<Option<PreparedLabRun>, String> {
        let Some((cached_arithmetic, pm, bytes)) =
            self.pm
                .as_ref()
                .and_then(|(key, cached_arithmetic, pm, bytes)| {
                    (cached_arithmetic == arithmetic && key == &pm_cache_key(pending, arithmetic))
                        .then(|| (cached_arithmetic.clone(), Arc::clone(pm), *bytes))
                })
        else {
            return Ok(None);
        };
        pending.cache_status["pm"] =
            json!({"status":"hit","arithmetic":cached_arithmetic,"bytes":bytes});
        self.last_status = pending.cache_status.clone();
        let mut prepared =
            self.assemble_cached(pending, Arc::as_ref(&pm), arithmetic, "cpu-exact", None)?;
        self.refresh_status(&mut prepared);
        Ok(Some(prepared))
    }

    /// Finalize a retained trajectory using freshly supplied unsmoothed
    /// densities. Supplied fields always replace matching-key raw observations.
    pub fn finish_cached_observed(
        &mut self,
        pending: &mut PendingLabRun,
        arithmetic: &str,
        rho: &[f32],
        pmrho: &[f32],
        deposit_source: &str,
    ) -> Result<PreparedLabRun, String> {
        if !["f64", "f32"].contains(&arithmetic) {
            return Err("PM arithmetic must be f64 or f32".into());
        }
        if deposit_source.is_empty() {
            return Err("Density deposit provenance must be nonempty".into());
        }
        validate_observed_density(&pending.request, "rho", rho)?;
        validate_observed_density(&pending.request, "pmrho", pmrho)?;
        let Some((pm, bytes)) = self
            .pm
            .as_ref()
            .and_then(|(key, cached_arithmetic, pm, bytes)| {
                (cached_arithmetic == arithmetic && key == &pm_cache_key(pending, arithmetic))
                    .then(|| (Arc::clone(pm), *bytes))
            })
        else {
            return Err("No matching cached PM trajectory is available".into());
        };
        pending.cache_status["pm"] = json!({"status":"hit","arithmetic":arithmetic,"bytes":bytes});
        self.last_status = pending.cache_status.clone();
        let mut prepared = self.assemble_cached(
            pending,
            Arc::as_ref(&pm),
            arithmetic,
            deposit_source,
            Some((rho, pmrho)),
        )?;
        self.refresh_status(&mut prepared);
        Ok(prepared)
    }

    pub fn finish_cpu(&mut self, pending: &mut PendingLabRun) -> Result<PreparedLabRun, String> {
        let key = pm_cache_key(pending, "f64");
        let (pm, hit, skipped) = match self.pm.as_ref() {
            Some((cached_key, arithmetic, pm, _)) if cached_key == &key && arithmetic == "f64" => {
                (Arc::clone(pm), true, false)
            }
            _ if !pending.request.needs_pm() => (Arc::new(skipped_pm_run(pending)), false, true),
            _ => (pending.pm_reference_arc()?, false, false),
        };
        let bytes = if skipped { 0 } else { pm_cache_bytes(&pending.request) };
        pending.cache_status["pm"] = if hit {
            json!({"status":"hit","arithmetic":"f64","bytes":bytes})
        } else if skipped {
            json!({"status":"skipped","reason":"no PM trajectory selected","arithmetic":"f64","bytes":0})
        } else {
            json!({"status":"miss","reason":"computed","arithmetic":"f64","bytes":bytes})
        };
        if !hit && !skipped {
            // A stale trajectory must not consume budget while the replacement
            // raw observations and PM state are considered.
            self.pm = None;
        }
        self.last_status = pending.cache_status.clone();
        let mut prepared =
            self.assemble_cached(pending, Arc::as_ref(&pm), "f64", "cpu-exact", None)?;
        let retained = !skipped
            && bytes.saturating_add(self.retained_bytes_without_pm()) <= SESSION_CACHE_LIMIT_BYTES
            && !hit;
        if retained {
            self.pm = Some((key, "f64".into(), Arc::clone(&pm), bytes));
        } else if !hit && !skipped {
            self.last_status["pm"] = json!({"status":"skipped","reason":"session memory budget","arithmetic":"f64","bytes":bytes});
        }
        if self.last_status["pm"]["status"] == "skipped" {
            prepared.cache_status["pm"] = self.last_status["pm"].clone();
        }
        self.refresh_status(&mut prepared);
        Ok(prepared)
    }

    pub fn finish_gpu_pm(
        &mut self,
        pending: &mut PendingLabRun,
        x: &[f32],
        p: &[f32],
        dx: &[f32],
        dp: &[f32],
        drift_minima: &[f64],
    ) -> Result<PreparedLabRun, String> {
        let pm = Arc::new(gpu_pm_result(pending, x, p, dx, dp, drift_minima)?);
        let key = pm_cache_key(pending, "f32");
        let bytes = pm_cache_bytes(&pending.request);
        pending.cache_status["pm"] =
            json!({"status":"miss","reason":"computed","arithmetic":"f32","bytes":bytes});
        self.pm = None;
        // This is a newly evolved PM result, so any raw observation keyed to
        // the same trajectory name must be rebuilt from this exact state.
        self.observations = None;
        self.last_status = pending.cache_status.clone();
        let mut prepared =
            self.assemble_cached(pending, Arc::as_ref(&pm), "f32", "cpu-exact", None)?;
        let retained =
            bytes.saturating_add(self.retained_bytes_without_pm()) <= SESSION_CACHE_LIMIT_BYTES;
        if retained {
            self.pm = Some((key, "f32".into(), Arc::clone(&pm), bytes));
        } else {
            self.last_status["pm"] = json!({"status":"skipped","reason":"session memory budget","arithmetic":"f32","bytes":bytes});
        }
        if self.last_status["pm"]["status"] == "skipped" {
            prepared.cache_status["pm"] = self.last_status["pm"].clone();
        }
        self.refresh_status(&mut prepared);
        Ok(prepared)
    }

    pub fn finish_cpu_observed(
        &mut self,
        pending: &mut PendingLabRun,
        rho: &[f32],
        pmrho: &[f32],
        deposit_source: &str,
    ) -> Result<PreparedLabRun, String> {
        let key = pm_cache_key(pending, "f64");
        let (pm, hit) = match self.pm.as_ref() {
            Some((cached_key, arithmetic, pm, _)) if cached_key == &key && arithmetic == "f64" => {
                (Arc::clone(pm), true)
            }
            _ if !pending.request.needs_pm() => (Arc::new(skipped_pm_run(pending)), false),
            _ => (pending.pm_reference_arc()?, false),
        };
        self.finish_with_observed_pm(pending, pm, hit, "f64", rho, pmrho, deposit_source)
    }

    pub fn finish_gpu_pm_observed(
        &mut self,
        pending: &mut PendingLabRun,
        x: &[f32],
        p: &[f32],
        dx: &[f32],
        dp: &[f32],
        drift_minima: &[f64],
        rho: &[f32],
        pmrho: &[f32],
        deposit_source: &str,
    ) -> Result<PreparedLabRun, String> {
        let pm = Arc::new(gpu_pm_result(pending, x, p, dx, dp, drift_minima)?);
        self.finish_with_observed_pm(pending, pm, false, "f32", rho, pmrho, deposit_source)
    }

    fn finish_with_observed_pm(
        &mut self,
        pending: &mut PendingLabRun,
        pm: Arc<PmRun>,
        hit: bool,
        arithmetic: &str,
        rho: &[f32],
        pmrho: &[f32],
        deposit_source: &str,
    ) -> Result<PreparedLabRun, String> {
        if deposit_source.is_empty() {
            return Err("Density deposit provenance must be nonempty".into());
        }
        // Validate before any observation or PM entry is inserted.
        validate_observed_density(&pending.request, "rho", rho)?;
        validate_observed_density(&pending.request, "pmrho", pmrho)?;
        let skipped = pm.completed_steps == 0;
        let key = pm_cache_key(pending, arithmetic);
        let bytes = if skipped { 0 } else { pm_cache_bytes(&pending.request) };
        pending.cache_status["pm"] = if hit {
            json!({"status":"hit","arithmetic":arithmetic,"bytes":bytes})
        } else if skipped {
            json!({"status":"skipped","reason":"no PM trajectory selected","arithmetic":arithmetic,"bytes":0})
        } else {
            json!({"status":"miss","reason":"computed","arithmetic":arithmetic,"bytes":bytes})
        };
        if !hit && !skipped {
            self.pm = None;
        }
        self.last_status = pending.cache_status.clone();
        let mut prepared = self.assemble_cached(
            pending,
            Arc::as_ref(&pm),
            arithmetic,
            deposit_source,
            Some((rho, pmrho)),
        )?;
        let retained = !skipped
            && bytes.saturating_add(self.retained_bytes_without_pm()) <= SESSION_CACHE_LIMIT_BYTES
            && !hit;
        if retained {
            self.pm = Some((key, arithmetic.into(), Arc::clone(&pm), bytes));
        } else if !hit && !skipped {
            self.last_status["pm"] = json!({"status":"skipped","reason":"session memory budget","arithmetic":arithmetic,"bytes":bytes});
        }
        if self.last_status["pm"]["status"] == "skipped" {
            prepared.cache_status["pm"] = self.last_status["pm"].clone();
        }
        self.refresh_status(&mut prepared);
        Ok(prepared)
    }

    fn assemble_cached(
        &mut self,
        pending: &mut PendingLabRun,
        pm: &PmRun,
        arithmetic: &str,
        deposit_source: &str,
        external_fields: Option<(&[f32], &[f32])>,
    ) -> Result<PreparedLabRun, String> {
        let pm_computed = pm.completed_steps > 0 && pending.request.needs_pm();
        let key = observation_cache_key(pending, arithmetic, deposit_source, pm_computed);
        let (raw, hit) = match self.observations.as_ref() {
            Some((cached_key, raw, _)) if external_fields.is_none() && cached_key == &key => {
                (Arc::clone(raw), true)
            }
            _ => {
                let raw = if let Some((rho, pmrho)) = external_fields {
                    raw_observations_external(pending, &pm.state.x, rho, pmrho, deposit_source)?
                } else if deposit_source == "cpu-exact" {
                    raw_observations(pending, pm)
                } else {
                    return Err(format!(
                        "No retained raw observations for deposit source {deposit_source}"
                    ));
                };
                (Arc::new(raw), false)
            }
        };
        let bytes = raw.estimated_bytes();
        // Replacing a raw observation entry releases the old allocation before
        // checking whether the new entry fits the combined session budget.
        if !hit {
            self.observations = None;
        }
        pending.cache_status["observations"] = if hit {
            json!({"status":"hit","bytes":self.observations.as_ref().map_or(0,|v|v.2),"smoothing":"recomputed","depositSource":raw.deposit_source})
        } else {
            json!({"status":"miss","reason":"trajectory/order/mesh/deposit-source key changed","bytes":bytes,"smoothing":"recomputed","depositSource":deposit_source})
        };
        self.last_status = pending.cache_status.clone();
        let prepared = assemble_prepared(pending, pm, arithmetic, Some(&raw))?;
        let total = self
            .initial
            .as_ref()
            .map_or(0, |v| v.3)
            .saturating_add(self.pending.as_ref().map_or(0, |v| v.2))
            .saturating_add(self.pm.as_ref().map_or(0, |v| v.3))
            .saturating_add(bytes);
        if !hit && total <= SESSION_CACHE_LIMIT_BYTES {
            self.observations = Some((key, raw, bytes));
        }
        self.last_status["observations"] = if hit {
            json!({"status":"hit","bytes":bytes,"smoothing":"recomputed","depositSource":deposit_source})
        } else if total <= SESSION_CACHE_LIMIT_BYTES {
            json!({"status":"miss","reason":"computed","bytes":bytes,"smoothing":"recomputed","depositSource":deposit_source})
        } else {
            json!({"status":"skipped","reason":"session memory budget","bytes":bytes})
        };
        self.last_status["bytes"] = json!(total.min(SESSION_CACHE_LIMIT_BYTES));
        Ok(prepared)
    }

    fn refresh_status(&mut self, prepared: &mut PreparedLabRun) {
        let bytes = self.retained_bytes();
        self.last_status["bytes"] = json!(bytes.min(SESSION_CACHE_LIMIT_BYTES));
        self.last_status["limitBytes"] = json!(SESSION_CACHE_LIMIT_BYTES);
        prepared.cache_status = self.last_status.clone();
        prepared.result["cache"] = self.last_status.clone();
    }
}

fn packed_vec4(grid: &Grid, values: &[f64], displacement: bool) -> Vec<f32> {
    let mut out = vec![0.0f32; grid.size * 4];
    for i in 0..grid.size {
        for a in 0..grid.dim {
            let q = if displacement {
                grid.unravel(i)[a] as f64 * grid.dx()
            } else {
                0.0
            };
            out[4 * i + a] = (values[i * grid.dim + a] - q) as f32;
        }
    }
    out
}

fn pending_from_geometry(
    r: &LabRequest,
    c: Arc<Cosmo>,
    source: &str,
    geometry: Arc<PendingGeometry>,
    cache_status: Value,
) -> PendingLabRun {
    PendingLabRun {
        request: r.clone(),
        c,
        initial_source: source.into(),
        geometry,
        pm_reference_cache: OnceLock::new(),
        pm_reference_quantized_cache: OnceLock::new(),
        pm_reference_displacement_quantized_cache: OnceLock::new(),
        cache_status,
    }
}

/// Prepare the common realization and retain the initial PM state for either backend.
pub fn prepare_pending(r: &LabRequest) -> Result<PendingLabRun, String> {
    r.validate()?;
    prepare_pending_from_initial_source(r, &Arc::new(initial_conditions(r)), "cpu-f64")
}

/// Build request-specific physical state from a validated, shared realization.
/// The input must already contain the order-four LPT fields.
pub fn prepare_pending_from_initial(
    r: &LabRequest,
    c: &Arc<Cosmo>,
) -> Result<PendingLabRun, String> {
    prepare_pending_from_initial_source(r, c, "cpu-f64")
}

pub fn prepare_pending_from_initial_source(
    r: &LabRequest,
    c: &Arc<Cosmo>,
    source: &str,
) -> Result<PendingLabRun, String> {
    r.validate()?;
    validate_initial_source(source)?;
    prepare_pending_with_cosmo(
        r,
        Arc::clone(c),
        source,
        json!({
            "initial": {"status":"miss", "reason":"not using a LabSession cache","provenance":source}
        }),
    )
}

fn validate_initial_source(source: &str) -> Result<(), String> {
    if ["cpu-f64", "gpu-f32"].contains(&source) {
        Ok(())
    } else {
        Err(format!("Unsupported initialization source: {source}"))
    }
}

fn prepare_pending_with_cosmo(
    r: &LabRequest,
    c: Arc<Cosmo>,
    initial_source: &str,
    cache_status: Value,
) -> Result<PendingLabRun, String> {
    let g = &c.grid;
    let sc = (1..=4)
        .map(|o| c.shell_crossing(o))
        .fold(f64::INFINITY, f64::min);
    if (!sc.is_finite() || sc <= 0.) && r.epoch == "fraction" {
        return Err("Cannot identify a finite crossing time for this realization".into());
    }
    let growth = &c.lpt_ref().growth;
    let a = if r.epoch == "scaleFactor" {
        r.scale_factor
    } else {
        let target_d = r.fraction * sc;
        if target_d > growth.d_max() {
            return Err("Requested fraction exceeds reachable LCDM growth (table extends to a=20). Select a physical scale factor, EdS, or another realization.".into());
        }
        growth.a_of_d(target_d)
    };
    let d = growth.d_of_a(a);
    let d_da = growth_rate(growth, a) * d / a;
    let a_start = (a * 0.05).min(0.03).max(0.001);
    let d_start = growth.d_of_a(a_start);
    let d_da_start = growth_rate(growth, a_start) * d_start / a_start;
    let (x, v_d) = evaluate(&c, d, r.order, 1.0);
    let v: Vec<f64> = v_d.iter().map(|v| v * d_da).collect();
    let dpos = tangent::lpt_amplitude_jvp(c.lpt_ref(), d, r.order).0;
    let h = 1e-4;
    let (xplus, _) = evaluate(&c, d, r.order, 1. + h);
    let (xminus, _) = evaluate(&c, d, r.order, 1. - h);
    let mut minj = f64::INFINITY;
    let mut mins = f64::INFINITY;
    // Resolved time-history check; positivity only at sampled times is not a
    // continuum guarantee. PM additionally checks each full linear drift.
    let history_steps = if g.size > 65_536 { 2 } else { 16 };
    for t in 0..=history_steps {
        let dt = d * t as f64 / history_steps as f64;
        if t == 0 {
            minj = minj.min(1.);
            mins = mins.min(1.);
            continue;
        }
        minj = minj.min(c.lpt_ref().min_jacobian(dt, r.order));
        if t == history_steps {
            mins = mins.min(observables::minimum_simplex_jacobian(g, &x));
        } else {
            let (xt, _) = evaluate(&c, dt, r.order, 1.);
            mins = mins.min(observables::minimum_simplex_jacobian(g, &xt));
        }
    }
    // Common growing initial condition uses all four orders, irrespective of
    // the order selected for the LPT comparison at the output.
    let (xi, vi) = evaluate(&c, d_start, 4, 1.0);
    let p: Vec<f64> = vi
        .iter()
        .map(|v| a_start.powi(3) * c.cosmo.e(a_start) * d_da_start * v)
        .collect();
    let pm_initial_min_jacobian = observables::minimum_simplex_jacobian(g, &xi);
    let (
        dx,
        dp,
        pm_initial_quantized_min_jacobian,
        pm_initial_displacement_quantized_min_jacobian,
    ) = if g.size > 16_384 {
        if !pm_initial_min_jacobian.is_finite() {
            return Err("PM initial Jacobian is nonfinite".into());
        }
        (
            dpos.clone(),
            vec![0.0; dpos.len()],
            pm_initial_min_jacobian,
            pm_initial_min_jacobian,
        )
    } else {
        let (dx, dv) = tangent::lpt_amplitude_jvp(c.lpt_ref(), d_start, 4);
        let dp = dv
            .iter()
            .map(|v| a_start.powi(3) * c.cosmo.e(a_start) * d_da_start * v)
            .collect();
        let mut xi_f32 = vec![0.0f32; g.size * 4];
        for i in 0..g.size {
            for axis in 0..g.dim {
                xi_f32[i * 4 + axis] = xi[i * g.dim + axis] as f32;
            }
        }
        let pm_initial_quantized_min_jacobian =
            observables::minimum_simplex_jacobian(g, &unpack_vec4_f32(g.dim, g.size, &xi_f32));
        let mut xi_displacement_f32 = vec![0.0f32; g.size * 4];
        for i in 0..g.size {
            let q = g.unravel(i);
            for axis in 0..g.dim {
                let psi = xi[i * g.dim + axis] - q[axis] as f64 * g.dx();
                xi_displacement_f32[i * 4 + axis] = psi as f32;
            }
        }
        // This minimum describes the absolute f32 state materialized by the GPU
        // orientation/output path. The force kernel itself consumes ψ directly.
        let mut xi_displacement_quantized = vec![0.0; g.size * g.dim];
        for i in 0..g.size {
            let q = g.unravel(i);
            for axis in 0..g.dim {
                xi_displacement_quantized[i * g.dim + axis] =
                    (q[axis] as f32 / g.n as f32 + xi_displacement_f32[i * 4 + axis]) as f64;
            }
        }
        let pm_initial_displacement_quantized_min_jacobian =
            observables::minimum_simplex_jacobian(g, &xi_displacement_quantized);
        if !pm_initial_min_jacobian.is_finite() || !pm_initial_quantized_min_jacobian.is_finite() {
            return Err("PM initial Jacobian is nonfinite".into());
        }
        if !pm_initial_displacement_quantized_min_jacobian.is_finite() {
            return Err("PM displacement-quantized initial Jacobian is nonfinite".into());
        }
        (
            dx,
            dp,
            pm_initial_quantized_min_jacobian,
            pm_initial_displacement_quantized_min_jacobian,
        )
    };
    let mut pm_clocks = Vec::with_capacity(r.steps * 5);
    let u_start = a_start.sqrt();
    let u_end = a.sqrt();
    for s in 0..r.steps {
        let u0 = u_start + (u_end - u_start) * s as f64 / r.steps as f64;
        let u1 = u_start + (u_end - u_start) * (s + 1) as f64 / r.steps as f64;
        let umid = 0.5 * (u0 + u1);
        let a0 = if s == 0 { a_start } else { u0 * u0 };
        let a1 = if s + 1 == r.steps { a } else { u1 * u1 };
        let amid = umid * umid;
        let kick0 = cosmology_clocks(Cosmology { omega_m: r.omega_m }, a0, amid).0;
        let drift = cosmology_clocks(Cosmology { omega_m: r.omega_m }, a0, a1).1;
        let kick1 = cosmology_clocks(Cosmology { omega_m: r.omega_m }, amid, a1).0;
        pm_clocks.extend([kick0, drift, kick1, a0, a1]);
    }
    Ok(PendingLabRun {
        request: r.clone(),
        c,
        initial_source: initial_source.into(),
        geometry: Arc::new(PendingGeometry {
            a,
            a_start,
            d,
            d_da,
            sc,
            minj,
            mins,
            x,
            v,
            dpos,
            xplus,
            xminus,
            pm_initial: PmState {
                x: xi,
                p,
                a: a_start,
            },
            pm_initial_tangent: PmTangent { dx, dp },
            pm_initial_min_jacobian,
            pm_initial_quantized_min_jacobian,
            pm_initial_displacement_quantized_min_jacobian,
            pm_clocks,
        }),
        pm_reference_cache: OnceLock::new(),
        pm_reference_quantized_cache: OnceLock::new(),
        pm_reference_displacement_quantized_cache: OnceLock::new(),
        cache_status,
    })
}

fn direct_mode_count(r: &LabRequest) -> usize {
    if r.n.saturating_pow(r.dim.min(3) as u32) >= 262_144 {
        2
    } else {
        (r.n / 4).min(6)
    }
}

/// Compatibility CPU path.
pub fn prepare(r: &LabRequest) -> Result<PreparedLabRun, String> {
    prepare_pending(r)?.finish_cpu()
}

impl PendingLabRun {
    pub fn request(&self) -> &LabRequest {
        &self.request
    }
    pub fn pm_clocks(&self) -> &[f64] {
        &self.pm_clocks
    }
    pub fn pm_initial_min_jacobian(&self) -> f64 {
        self.pm_initial_min_jacobian
    }
    /// GPU input displacement ψ=x−q, with subtraction performed in f64 before
    /// the weak displacement is rounded to f32.
    pub fn pm_displacement(&self) -> Vec<f32> {
        let grid = Grid::new(self.request.dim, self.request.n, 1.0);
        packed_vec4(&grid, &self.pm_initial.x, true)
    }
    /// Absolute pending LPT sheet positions in the packed f32 geometry used
    /// by the external overlap deposit, with inactive vec4 lanes set to zero.
    pub fn lpt_positions(&self) -> Vec<f32> {
        let grid = Grid::new(self.request.dim, self.request.n, 1.0);
        packed_vec4(&grid, &self.x, false)
    }
    /// Exact conservative sheet deposit with P1 (piecewise-linear) density
    /// interpolation from the vertex inverse Jacobians w_i = 1/|J(q_i)|.
    pub fn lpt_density_p1(&self) -> Vec<f32> {
        let g = &self.c.grid;
        let mesh = self.request.mesh;
        if g.dim == 1 {
            return observables::interval_density(g, &self.x, mesh);
        }
        let pos: Vec<f32> = self.x.iter().map(|&v| v as f32).collect();
        let mut j = vec![0.0f64; g.size];
        self.c.lpt_ref().jacobian(self.d, self.request.order, &mut j);
        let w: Vec<f32> = j.iter().map(|&v| (1.0 / v.abs().max(1e-6)) as f32).collect();
        if g.dim == 2 {
            sheet::sheet_density_2d_exact(g, &pos, Some(&w), mesh)
        } else {
            sheet::sheet_density_3d_exact(g, &pos, Some(&w), mesh)
        }
    }
    pub fn lpt_density_p0_and_p1(&self) -> (Vec<f32>, Vec<f32>) {
        let g = &self.c.grid;
        let mesh = self.request.mesh;
        if g.dim == 1 {
            let rho = observables::interval_density(g, &self.x, mesh);
            return (rho.clone(), rho);
        }
        let pos: Vec<f32> = self.x.iter().map(|&v| v as f32).collect();
        let mut j = vec![0.0f64; g.size];
        self.c.lpt_ref().jacobian(self.d, self.request.order, &mut j);
        let w: Vec<f32> = j.iter().map(|&v| (1.0 / v.abs().max(1e-6)) as f32).collect();
        if g.dim == 2 {
            sheet::sheet_density_2d_exact_p0_p1(g, &pos, &w, mesh)
        } else {
            sheet::sheet_density_3d_exact_p0_p1(g, &pos, &w, mesh)
        }
    }
    pub fn metadata(&self) -> Value {
        json!({"dim":self.request.dim,"n":self.request.n,"forceMesh":self.request.force_mesh,"initialSource":self.initial_source,
            "box":1.0,"omegaM":self.request.omega_m,"aStart":self.a_start,"aEnd":self.a,
            "steps":self.request.steps,"modeCount":direct_mode_count(&self.request),
            "pmArithmetic":"f32","pmInitialMinimumJacobianF64":self.pm_initial_min_jacobian,
            "pmInitialMinimumJacobianF32":self.pm_initial_quantized_min_jacobian,
            "pmInitialMinimumJacobianDisplacementF32":self.pm_initial_displacement_quantized_min_jacobian})
    }
    pub fn pm_reference(&self) -> Result<Vec<f64>, String> {
        self.pack_pm_reference(self.pm_reference_run()?)
    }
    /// CPU PM oracle from the f32-rounded initial state, with the reference
    /// solver and exact f64 clocks retained to isolate input quantization.
    pub fn pm_reference_quantized(&self) -> Result<Vec<f64>, String> {
        self.pack_pm_reference(self.pm_reference_quantized_run()?)
    }
    /// CPU PM oracle from f32-rounded displacement ψ and f32-rounded p/dx/dp.
    /// The lattice q is restored in f64 before evolution, using exact f64 clocks.
    pub fn pm_reference_displacement_quantized(&self) -> Result<Vec<f64>, String> {
        self.pack_pm_reference(self.pm_reference_displacement_quantized_run()?)
    }
    fn pack_pm_reference(&self, run: &PmRun) -> Result<Vec<f64>, String> {
        let dim = self.request.dim;
        let nv = self.request.n.pow(dim as u32);
        let mut out = vec![0.0; nv * 16];
        for (block, values) in [
            (0, run.state.x.as_slice()),
            (1, run.state.p.as_slice()),
            (2, run.tangent.as_ref().unwrap().dx.as_slice()),
            (3, run.tangent.as_ref().unwrap().dp.as_slice()),
        ] {
            for i in 0..nv {
                for a in 0..dim {
                    out[block * nv * 4 + i * 4 + a] = values[i * dim + a];
                }
            }
        }
        Ok(out)
    }
    fn pm_reference_run(&self) -> Result<&PmRun, String> {
        self.pm_reference_cache
            .get_or_init(|| {
                self.evolve_pm(self.pm_initial.clone(), self.pm_initial_tangent.clone())
                    .map(Arc::new)
            })
            .as_ref()
            .map(Arc::as_ref)
            .map_err(Clone::clone)
    }
    fn pm_reference_arc(&self) -> Result<Arc<PmRun>, String> {
        self.pm_reference_cache
            .get_or_init(|| {
                self.evolve_pm(self.pm_initial.clone(), self.pm_initial_tangent.clone())
                    .map(Arc::new)
            })
            .as_ref()
            .map(Arc::clone)
            .map_err(Clone::clone)
    }
    fn pm_reference_quantized_run(&self) -> Result<&PmRun, String> {
        self.pm_reference_quantized_cache
            .get_or_init(|| {
                let (x, p, dx, dp) = self.pm_input();
                let dim = self.request.dim;
                let nv = self.request.n.pow(dim as u32);
                let compact = |v: &[f32]| {
                    let mut out = vec![0.0; nv * dim];
                    for i in 0..nv {
                        for a in 0..dim {
                            out[i * dim + a] = v[i * 4 + a] as f64;
                        }
                    }
                    out
                };
                self.evolve_pm(
                    PmState {
                        x: compact(&x),
                        p: compact(&p),
                        a: self.a_start,
                    },
                    PmTangent {
                        dx: compact(&dx),
                        dp: compact(&dp),
                    },
                )
                .map(Arc::new)
            })
            .as_ref()
            .map(Arc::as_ref)
            .map_err(Clone::clone)
    }
    fn pm_reference_displacement_quantized_run(&self) -> Result<&PmRun, String> {
        self.pm_reference_displacement_quantized_cache
            .get_or_init(|| {
                let grid = Grid::new(self.request.dim, self.request.n, 1.0);
                let displacement = self.pm_displacement();
                let p = self.pm_initial_component(1);
                let dx = self.pm_initial_component(2);
                let dp = self.pm_initial_component(3);
                let dim = self.request.dim;
                let nv = self.request.n.pow(dim as u32);
                let compact = |v: &[f32]| {
                    let mut out = vec![0.0; nv * dim];
                    for i in 0..nv {
                        for a in 0..dim {
                            out[i * dim + a] = v[i * 4 + a] as f64;
                        }
                    }
                    out
                };
                let mut x = vec![0.0; nv * dim];
                for i in 0..nv {
                    let q = grid.unravel(i);
                    for a in 0..dim {
                        x[i * dim + a] = q[a] as f64 * grid.dx() + displacement[i * 4 + a] as f64;
                    }
                }
                self.evolve_pm(
                    PmState {
                        x,
                        p: compact(&p),
                        a: self.a_start,
                    },
                    PmTangent {
                        dx: compact(&dx),
                        dp: compact(&dp),
                    },
                )
                .map(Arc::new)
            })
            .as_ref()
            .map(Arc::as_ref)
            .map_err(Clone::clone)
    }
    fn evolve_pm(&self, initial: PmState, tangent: PmTangent) -> Result<PmRun, String> {
        let sheet = Grid::new(self.request.dim, self.request.n, 1.0);
        let force_grid = Grid::new(self.request.dim, self.request.force_mesh, 1.0);
        evolve_cosmology(
            &force_grid,
            &sheet,
            initial,
            if sheet.size > 65_536 {
                None
            } else {
                Some(tangent)
            },
            self.a,
            self.request.steps,
            Cosmology {
                omega_m: self.request.omega_m,
            },
            true,
        )
    }
    pub fn pm_input(&self) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
        (
            self.pm_initial_component(0),
            self.pm_initial_component(1),
            self.pm_initial_component(2),
            self.pm_initial_component(3),
        )
    }
    pub fn pm_initial_component(&self, component: usize) -> Vec<f32> {
        let dim = self.request.dim;
        let nv = self.request.n.pow(dim as u32);
        let values = match component {
            0 => &self.pm_initial.x,
            1 => &self.pm_initial.p,
            2 => &self.pm_initial_tangent.dx,
            3 => &self.pm_initial_tangent.dp,
            _ => return Vec::new(),
        };
        let mut out = vec![0.0f32; nv * 4];
        for i in 0..nv {
            for a in 0..dim {
                out[i * 4 + a] = values[i * dim + a] as f32;
            }
        }
        out
    }
    pub fn finish_cpu(&self) -> Result<PreparedLabRun, String> {
        if !self.request.needs_pm() {
            let pm = skipped_pm_run(self);
            return assemble_prepared(self, &pm, "f64", None);
        }
        let pm = self.pm_reference_run()?;
        assemble_prepared(self, pm, "f64", None)
    }
    pub fn finish_gpu_pm(
        &self,
        x: &[f32],
        p: &[f32],
        dx: &[f32],
        dp: &[f32],
        drift_minima: &[f64],
    ) -> Result<PreparedLabRun, String> {
        let dim = self.request.dim;
        let nv = self.request.n.pow(dim as u32);
        let expected = nv * 4;
        for (name, values) in [("x", x), ("p", p), ("dx", dx), ("dp", dp)] {
            if values.len() != expected {
                return Err(format!("GPU PM {name} length must be {expected}"));
            }
            if values.iter().any(|v| !v.is_finite()) {
                return Err(format!("GPU PM {name} contains a nonfinite value"));
            }
        }
        if drift_minima.len() != self.request.steps {
            return Err(format!(
                "GPU PM needs {} verified drift minima",
                self.request.steps
            ));
        }
        if drift_minima.iter().any(|v| !v.is_finite()) {
            return Err("GPU PM drift minima must be finite".into());
        }
        let mut pmx = vec![0.0; nv * dim];
        let mut p64 = pmx.clone();
        let mut dx64 = pmx.clone();
        let mut dp64 = pmx.clone();
        for i in 0..nv {
            for a in 0..dim {
                let j = i * dim + a;
                let k = i * 4 + a;
                pmx[j] = x[k] as f64;
                p64[j] = p[k] as f64;
                dx64[j] = dx[k] as f64;
                dp64[j] = dp[k] as f64;
            }
        }
        // The active GPU protocol carries displacement ψ, so its materialized
        // initial state is the displacement-quantized branch. Keep the legacy
        // absolute-f32 diagnostic available in metadata, but don't let that
        // unused representation create a crossing warning.
        let min_initial = self
            .pm_initial_min_jacobian
            .min(self.pm_initial_displacement_quantized_min_jacobian);
        let mut minimum = min_initial;
        let mut first_crossing = if minimum <= 0.0 {
            Some(PmCrossing {
                step: 0,
                a_start: self.a_start,
                a_end: self.a_start,
                minimum_jacobian: minimum,
            })
        } else {
            None
        };
        for (i, &jac) in drift_minima.iter().enumerate() {
            minimum = minimum.min(jac);
            if jac <= 0.0 && first_crossing.is_none() {
                first_crossing = Some(PmCrossing {
                    step: i + 1,
                    a_start: self.pm_clocks[5 * i + 3],
                    a_end: self.pm_clocks[5 * i + 4],
                    minimum_jacobian: jac,
                });
            }
        }
        let pm = PmRun {
            state: PmState {
                x: pmx,
                p: p64,
                a: self.a,
            },
            tangent: Some(PmTangent { dx: dx64, dp: dp64 }),
            min_jacobian: minimum,
            completed_steps: self.request.steps,
            crossed: first_crossing.is_some(),
            first_crossing,
        };
        assemble_prepared(self, &pm, "f32", None)
    }
}

fn unpack_vec4_f32(dim: usize, nv: usize, packed: &[f32]) -> Vec<f64> {
    let mut out = vec![0.0; nv * dim];
    for i in 0..nv {
        for a in 0..dim {
            out[i * dim + a] = packed[i * 4 + a] as f64;
        }
    }
    out
}

fn gpu_pm_result(
    pending: &PendingLabRun,
    x: &[f32],
    p: &[f32],
    dx: &[f32],
    dp: &[f32],
    drift_minima: &[f64],
) -> Result<PmRun, String> {
    let dim = pending.request.dim;
    let nv = pending.request.n.pow(dim as u32);
    let expected = nv * 4;
    for (name, values) in [("x", x), ("p", p), ("dx", dx), ("dp", dp)] {
        if values.len() != expected {
            return Err(format!("GPU PM {name} length must be {expected}"));
        }
        if values.iter().any(|v| !v.is_finite()) {
            return Err(format!("GPU PM {name} contains a nonfinite value"));
        }
    }
    if drift_minima.len() != pending.request.steps {
        return Err(format!(
            "GPU PM needs {} verified drift minima",
            pending.request.steps
        ));
    }
    if drift_minima.iter().any(|v| !v.is_finite()) {
        return Err("GPU PM drift minima must be finite".into());
    }
    let mut pmx = vec![0.0; nv * dim];
    let mut p64 = pmx.clone();
    let mut dx64 = pmx.clone();
    let mut dp64 = pmx.clone();
    for i in 0..nv {
        for a in 0..dim {
            let j = i * dim + a;
            let k = i * 4 + a;
            pmx[j] = x[k] as f64;
            p64[j] = p[k] as f64;
            dx64[j] = dx[k] as f64;
            dp64[j] = dp[k] as f64;
        }
    }
    let mut minimum = pending
        .pm_initial_min_jacobian
        .min(pending.pm_initial_displacement_quantized_min_jacobian);
    let mut first_crossing = if minimum <= 0.0 {
        Some(PmCrossing {
            step: 0,
            a_start: pending.a_start,
            a_end: pending.a_start,
            minimum_jacobian: minimum,
        })
    } else {
        None
    };
    for (i, &jac) in drift_minima.iter().enumerate() {
        minimum = minimum.min(jac);
        if jac <= 0.0 && first_crossing.is_none() {
            first_crossing = Some(PmCrossing {
                step: i + 1,
                a_start: pending.pm_clocks[5 * i + 3],
                a_end: pending.pm_clocks[5 * i + 4],
                minimum_jacobian: jac,
            });
        }
    }
    Ok(PmRun {
        state: PmState {
            x: pmx,
            p: p64,
            a: pending.a,
        },
        tangent: Some(PmTangent { dx: dx64, dp: dp64 }),
        min_jacobian: minimum,
        completed_steps: pending.request.steps,
        crossed: first_crossing.is_some(),
        first_crossing,
    })
}

fn compute_custom_series_standalone(
    pending: &PendingLabRun,
    pm: &PmRun,
    raw_opt: Option<&RawObservations>,
) -> (serde_json::Map<String, Value>, Vec<(String, Vec<f32>)>) {
    let r = &pending.request;
    let c = &pending.c;
    let d = pending.d;
    let x = &pending.x;
    let g = &c.grid;
    let pmx = &pm.state.x;
    let eg = Grid::new(r.dim, r.mesh, 1.0);
    let probabilities = [0.001, 0.01, 0.1, 0.5, 0.9, 0.99, 0.999];
    let dy = 3.6 / 80.0;
    let omit_large_mesh_json = eg.size > 262_144;
    let empty_f32: &[f32] = &[];
    let physical_volume = r.box_size.powi(r.dim as i32);
    let physical_k = |v: &[f64]| v.iter().map(|k| k / r.box_size).collect::<Vec<_>>();
    let physical_p = |v: &[f64]| v.iter().map(|p| p * physical_volume).collect::<Vec<_>>();
    let cell_scale = r.box_size / r.mesh as f64;
    let particle_scale = r.box_size / r.n as f64;
    let r_min = (0.25 * cell_scale).max(0.05);
    let r_max = (0.35 * r.box_size).max(r_min * 2.0);
    let n_radii = 36usize;
    let mut radii: Vec<f64> = (0..n_radii)
        .map(|i| r_min * (r_max / r_min).powf(i as f64 / (n_radii - 1) as f64))
        .collect();
    for special in [8.0, r.initial_smoothing, cell_scale, particle_scale] {
        if special >= r_min && special <= r_max {
            radii.push(special);
        }
    }
    radii.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    radii.dedup_by(|a, b| (*a - *b).abs() <= 1e-4 * b.abs().max(1e-12));
    let spectrum_variance =
        |sp: &spectra::Spectrum, radius_mpc: f64, gaussian_window: bool| -> f64 {
            let r_unit = radius_mpc / r.box_size;
            sp.k.iter()
                .zip(&sp.p)
                .zip(&sp.nmodes)
                .map(|((&k, &p), &nm)| {
                    let w = cosmological::filter_window(r.dim, k * r_unit, gaussian_window);
                    nm * p * w * w
                })
                .sum::<f64>()
        };
    let mut custom_series = serde_json::Map::new();
    let mut custom_rhos: Vec<(String, Vec<f32>)> = Vec::new();
    if r.estimators.is_empty() {
        return (custom_series, custom_rhos);
    }
    let active_lpt_tag = format!("{}lpt", r.order);
    for (traj, ord_opt) in [
        ("1lpt", Some(1usize)),
        ("2lpt", Some(2usize)),
        ("3lpt", Some(3usize)),
        ("4lpt", Some(4usize)),
        ("pm", None),
    ] {
        let prefix = format!("{traj}:");
        let wanted_ests: Vec<&str> = r
            .estimators
            .iter()
            .filter_map(|s| s.strip_prefix(&prefix))
            .filter(|&e| matches!(e, "p0" | "p1" | "ngp" | "cic" | "tsc"))
            .collect();
        if wanted_ests.is_empty() {
            continue;
        }
        let x_owned: Option<Vec<f64>> = match ord_opt {
            Some(ord) if ord != r.order => Some(evaluate(c, d, ord, 1.0).0),
            _ => None,
        };
        let x_traj: &[f64] = if traj == "pm" {
            pmx
        } else if let Some(ref xo) = x_owned {
            xo.as_slice()
        } else {
            x
        };
        let pos_f32: Vec<f32> = x_traj.iter().map(|&v| v as f32).collect();
        let can_reuse_active = raw_opt.is_some() && traj == active_lpt_tag;
        let can_reuse_pm = raw_opt.is_some() && traj == "pm" && pm.completed_steps > 0;
        let need_p0_fresh = wanted_ests.contains(&"p0") && !can_reuse_active && !can_reuse_pm;
        let need_p1_fresh = wanted_ests.contains(&"p1") && !can_reuse_active;
        let w_p1: Option<Vec<f32>> = if wanted_ests.contains(&"p1") {
            if let Some(ord) = ord_opt {
                let mut j = vec![0.0f64; g.size];
                c.lpt_ref().jacobian(d, ord, &mut j);
                Some(j.iter().map(|&v| (1.0 / v.abs().max(1e-6)) as f32).collect())
            } else {
                Some(observables::vertex_inverse_jacobians(g, x_traj))
            }
        } else {
            None
        };
        let (mut pre_p0, mut pre_p1): (Option<Vec<f32>>, Option<Vec<f32>>) =
            if need_p0_fresh && need_p1_fresh && g.dim >= 2 {
                let w_ref = w_p1.as_ref().unwrap();
                let (r0, r1) = if g.dim == 2 {
                    sheet::sheet_density_2d_exact_p0_p1(g, &pos_f32, w_ref, r.mesh)
                } else {
                    sheet::sheet_density_3d_exact_p0_p1(g, &pos_f32, w_ref, r.mesh)
                };
                (Some(r0), Some(r1))
            } else {
                (None, None)
            };
        for &est in &wanted_ests {
            let key = format!("{traj}:{est}");
            if custom_series.contains_key(&key) {
                continue;
            }
            let (raw_owned, smooth_owned, ps_owned): (
                Option<Vec<f32>>,
                Option<Vec<f32>>,
                Option<spectra::Spectrum>,
            );
            let (raw_ref, smooth_ref, ps_ref): (&[f32], &[f32], &spectra::Spectrum) =
                if let Some(raw) = raw_opt.filter(|_| can_reuse_active && matches!(est, "p0" | "p1" | "cic" | "tsc")) {
                    let (r_slice, p_ref) = match est {
                        "p0" => (raw.rho.as_slice(), &raw.ps),
                        "p1" => (raw.rho_p1.as_slice(), &raw.ps_p1),
                        "cic" => (raw.cic.as_slice(), &raw.psc),
                        "tsc" => (raw.tsc.as_slice(), &raw.pst),
                        _ => unreachable!(),
                    };
                    smooth_owned = Some(observables::smooth_density(&eg, r_slice, r.smoothing));
                    (r_slice, smooth_owned.as_ref().unwrap().as_slice(), p_ref)
                } else if let Some(raw) = raw_opt.filter(|_| can_reuse_pm && est == "p0") {
                    smooth_owned = Some(observables::smooth_density(&eg, &raw.pmrho, r.smoothing));
                    (raw.pmrho.as_slice(), smooth_owned.as_ref().unwrap().as_slice(), &raw.psp)
                } else {
                    let r_vec = match est {
                        "p0" => pre_p0.take().unwrap_or_else(|| density(g, x_traj, r.mesh)),
                        "p1" => pre_p1.take().unwrap_or_else(|| {
                            let w_ref = w_p1.as_ref().unwrap();
                            if g.dim == 1 {
                                observables::interval_density_p1(g, x_traj, w_ref, r.mesh)
                            } else if g.dim == 2 {
                                sheet::sheet_density_2d_exact(g, &pos_f32, Some(w_ref), r.mesh)
                            } else {
                                sheet::sheet_density_3d_exact(g, &pos_f32, Some(w_ref), r.mesh)
                            }
                        }),
                        "ngp" => sheet::ngp_density(r.dim, &pos_f32, g.size, 1.0, r.mesh),
                        "cic" => sheet::cic_density(r.dim, &pos_f32, g.size, 1.0, r.mesh),
                        "tsc" => sheet::tsc_density(r.dim, &pos_f32, g.size, 1.0, r.mesh),
                        _ => unreachable!(),
                    };
                    let s_vec = observables::smooth_density(&eg, &r_vec, r.smoothing);
                    let p_spec = power(&eg, &r_vec);
                    raw_owned = Some(r_vec);
                    smooth_owned = Some(s_vec);
                    ps_owned = Some(p_spec);
                    (
                        raw_owned.as_ref().unwrap().as_slice(),
                        smooth_owned.as_ref().unwrap().as_slice(),
                        ps_owned.as_ref().unwrap(),
                    )
                };
            let (logrho_est, vol_est, mass_est, var_est, dy_est, summary_est) =
                if matches!(est, "p0" | "p1") {
                    let w_simplex_p1 = if est == "p1" {
                        w_p1.as_deref()
                    } else {
                        None
                    };
                    let (lr_e, v_e, m_e, _, var_e, dy_e, sum_e) =
                        observables::simplex_density_pdf_and_summary(
                            g,
                            x_traj,
                            w_simplex_p1,
                            80,
                            &probabilities,
                        );
                    (lr_e, v_e, m_e, var_e, dy_e, sum_e)
                } else {
                    let (pdf_raw_nat, pdf_smooth_nat);
                    let (pdf_raw, pdf_smooth): (&[f32], &[f32]) =
                        if r.mesh > r.n && matches!(est, "ngp" | "cic" | "tsc") {
                            pdf_raw_nat = match est {
                                "ngp" => sheet::ngp_density(r.dim, &pos_f32, g.size, 1.0, r.n),
                                "cic" => sheet::cic_density(r.dim, &pos_f32, g.size, 1.0, r.n),
                                "tsc" => sheet::tsc_density(r.dim, &pos_f32, g.size, 1.0, r.n),
                                _ => unreachable!(),
                            };
                            pdf_smooth_nat =
                                observables::smooth_density(&c.grid, &pdf_raw_nat, r.smoothing);
                            (&pdf_raw_nat, &pdf_smooth_nat)
                        } else {
                            (raw_ref, smooth_ref)
                        };
                    let (lr_e, v_e, m_e, _, var_e) =
                        observables::density_pdf(pdf_smooth, pdf_raw, 80);
                    let sum_e =
                        observables::density_summary(pdf_smooth, pdf_raw, &probabilities);
                    (lr_e, v_e, m_e, var_e, dy, sum_e)
                };
            let direct_shell = if est == "p0" {
                if let Some(raw) = raw_opt.filter(|_| can_reuse_active) {
                    Some(physical_p(&raw.direct_shell_lpt))
                } else if let Some(raw) = raw_opt.filter(|_| can_reuse_pm) {
                    Some(physical_p(&raw.direct_shell_pm))
                } else {
                    Some(physical_p(&compute_meshfree_shell_single(pending, x_traj)))
                }
            } else {
                None
            };
            let var_tophat: Vec<f64> = radii
                .iter()
                .map(|&rad| spectrum_variance(ps_ref, rad, false))
                .collect();
            let var_gaussian: Vec<f64> = radii
                .iter()
                .map(|&rad| spectrum_variance(ps_ref, rad, true))
                .collect();
            let var_unsmoothed = spectrum_variance(ps_ref, 0.0, false);
            let rho_json_slice = if omit_large_mesh_json {
                empty_f32
            } else {
                smooth_ref
            };
            custom_rhos.push((key.clone(), smooth_ref.to_vec()));
            custom_series.insert(
                key.clone(),
                json!({
                    "key": key,
                    "traj": traj,
                    "est": est,
                    "rho": rho_json_slice,
                    "massMean": mean(raw_ref),
                    "power": physical_p(&ps_ref.p),
                    "counts": ps_ref.nmodes,
                    "fineK": physical_k(&ps_ref.fine_k),
                    "finePower": physical_p(&ps_ref.fine_p),
                    "fineCounts": ps_ref.fine_nmodes,
                    "directShell": direct_shell,
                    "variance": {
                        "unsmoothed": var_unsmoothed,
                        "tophat": var_tophat,
                        "gaussian": var_gaussian,
                    },
                    "pdf": {
                        "logrho": logrho_est,
                        "dy": dy_est,
                        "volume": vol_est,
                        "mass": mass_est,
                        "variance": var_est,
                        "capturedVolume": vol_est.iter().sum::<f64>() * dy_est,
                        "capturedMass": mass_est.iter().sum::<f64>() * dy_est,
                        "summary": summary_est,
                    }
                }),
            );
        }
    }
    (custom_series, custom_rhos)
}

fn assemble_prepared(
    pending: &PendingLabRun,
    pm: &PmRun,
    pm_arithmetic: &str,
    raw_cached: Option<&RawObservations>,
) -> Result<PreparedLabRun, String> {
    let r = &pending.request;
    let c = &pending.c;
    let a = pending.a;
    let a_start = pending.a_start;
    let d = pending.d;
    let d_da = pending.d_da;
    let sc = pending.sc;
    let minj = pending.minj;
    let mins = pending.mins;
    let x = &pending.x;
    let v = &pending.v;
    let dpos = &pending.dpos;
    let xplus = &pending.xplus;
    let xminus = &pending.xminus;
    let g = &c.grid;
    let growth = &c.lpt_ref().growth;
    let h = 1e-4;
    let pmx = &pm.state.x;
    let pmv: Vec<f64> = pm
        .state
        .p
        .iter()
        .map(|p| p / (a.powi(3) * c.cosmo.e(a)))
        .collect();
    let owned_raw;
    let raw = if let Some(raw) = raw_cached {
        raw
    } else {
        owned_raw = raw_observations(pending, &pm);
        &owned_raw
    };
    let rho = &raw.rho;
    let rho_p1 = &raw.rho_p1;
    let pmrho = &raw.pmrho;
    let cic = &raw.cic;
    let eg = Grid::new(r.dim, r.mesh, 1.0);
    let rs = observables::smooth_density(&eg, &rho, r.smoothing);
    let rs_p1 = observables::smooth_density(&eg, &rho_p1, r.smoothing);
    let pms = observables::smooth_density(&eg, &pmrho, r.smoothing);
    let cics = observables::smooth_density(&eg, cic, r.smoothing);
    let (cic_pdf_raw_owned, cic_pdf_smooth_owned);
    let (cic_pdf_raw, cic_pdf_smooth): (&[f32], &[f32]) = if r.mesh > r.n {
        let pos: Vec<f32> = x.iter().map(|&v| v as f32).collect();
        cic_pdf_raw_owned = sheet::cic_density(r.dim, &pos, c.grid.size, 1.0, r.n);
        cic_pdf_smooth_owned =
            observables::smooth_density(&c.grid, &cic_pdf_raw_owned, r.smoothing);
        (&cic_pdf_raw_owned, &cic_pdf_smooth_owned)
    } else {
        (cic.as_slice(), cics.as_slice())
    };
    let probabilities = [0.001, 0.01, 0.1, 0.5, 0.9, 0.99, 0.999];
    let w_lpt_active: Vec<f32> = {
        let mut j = vec![0.0f64; g.size];
        c.lpt_ref().jacobian(d, r.order, &mut j);
        j.iter().map(|&v| (1.0 / v.abs().max(1e-6)) as f32).collect()
    };
    let (
        (logrho, vol, mass, pdfmean, var, dy_p0, summary),
        (p1_logrho, p1vol, p1mass, _, p1var, dy_p1, p1_summary),
        (pm_logrho, pmvol, pmmass, _, pmvar, dy_pm, pm_summary),
    ) = {
        let p0_pdf = observables::simplex_density_pdf_and_summary(g, &x, None, 80, &probabilities);
        let p1_pdf = observables::simplex_density_pdf_and_summary(
            g,
            &x,
            Some(&w_lpt_active),
            80,
            &probabilities,
        );
        let pm_pdf = if pm.completed_steps > 0 && r.needs_pm() {
            observables::simplex_density_pdf_and_summary(g, pmx, None, 80, &probabilities)
        } else {
            p0_pdf.clone()
        };
        (p0_pdf, p1_pdf, pm_pdf)
    };
    let (cic_logrho, cicvol, cicmass, _, cicvar) =
        observables::density_pdf(cic_pdf_smooth, cic_pdf_raw, 80);
    let dy_cic = 3.6 / 80.0;
    let cic_summary =
        observables::density_summary(cic_pdf_smooth, cic_pdf_raw, &probabilities);
    let tscs = observables::smooth_density(&eg, &raw.tsc, r.smoothing);
    let ps = &raw.ps;
    let ps_p1 = &raw.ps_p1;
    let psc = &raw.psc;
    let pst = &raw.pst;
    let psp = &raw.psp;
    let mut ks = Vec::new();
    let mut powers = Vec::new();
    let mut derivatives = Vec::new();
    let mut fd = Vec::new();
    let mut errs = Vec::new();
    let mut pmpowers = Vec::new();
    let mut pmderivs = Vec::new();
    let mut mode_records = Vec::new();
    for m in 1..=direct_mode_count(r) {
        let k = [g.kf() * m as f64, 0., 0.];
        ks.push(k[0]);
        powers.push(0.);
        derivatives.push(0.);
        fd.push(0.);
        errs.push(0.);
        pmpowers.push(0.);
        pmderivs.push(0.);
        mode_records.push([m as i32, 0, 0, 0]);
    }
    let spec_modes = &raw.direct_modes;
    let spec_ks: Vec<f64> = spec_modes.iter().map(|&m| g.kf() * m as f64).collect();
    let spec_powers = &raw.direct_powers;
    let spec_pm_powers = &raw.direct_pm_powers;
    let pmdpos = pm
        .tangent
        .as_ref()
        .map(|t| t.dx.clone())
        .unwrap_or_else(|| dpos.clone());
    let pm_initial_minimum_used = if pm_arithmetic == "f64" {
        pending.pm_initial_min_jacobian
    } else {
        pending
            .pm_initial_min_jacobian
            .min(pending.pm_initial_displacement_quantized_min_jacobian)
    };
    let pm_initial_minimum_representation = if pm_arithmetic == "f64" {
        "f64"
    } else {
        "min(f64, displacement-f32-materialized)"
    };
    let q: Vec<f64> = (0..g.size)
        .flat_map(|i| {
            let ijk = g.unravel(i);
            (0..r.dim).map(move |d| ijk[d] as f64 * g.dx())
        })
        .collect();
    let position_rms = x
        .iter()
        .zip(pmx)
        .map(|(a, b)| (a - b).powi(2))
        .sum::<f64>()
        .sqrt()
        / (x.len() as f64).sqrt();
    let displacement_rms = x
        .iter()
        .zip(&q)
        .map(|(a, b)| (a - b).powi(2))
        .sum::<f64>()
        .sqrt()
        / (x.len() as f64).sqrt();
    let power_error =
        ps.p.iter()
            .zip(&psp.p)
            .take(4)
            .filter(|(_, p)| **p > 1e-14)
            .map(|(s, p)| (s / p - 1.).abs())
            .fold(0., f64::max);
    let physical_volume = r.box_size.powi(r.dim as i32);
    let physical_k = |v: &[f64]| v.iter().map(|k| k / r.box_size).collect::<Vec<_>>();
    let physical_p = |v: &[f64]| v.iter().map(|p| p * physical_volume).collect::<Vec<_>>();
    let d1 = if c.cosmo.is_eds() { 1. } else { growth.d_of_a(1.) };
    let linear_spectrum = if r.ic == "gaussian" {
        Some(cosmological::LinearSpectrum::new(
            r.omega_m,
            r.omega_b,
            r.h,
            r.ns,
            r.sigma8,
            r.transfer == "bao",
        ))
    } else {
        None
    };
    // The filtered linear input spectrum at the same physical wavenumbers,
    // scaled to the output epoch; empty for the deterministic wave benchmark.
    let (
        linear_k,
        linear_p,
        fine_linear_k,
        fine_linear_p,
        direct_linear_k,
        direct_linear_p,
        direct_shell_linear_k,
        direct_shell_linear_p,
    ) = if let Some(ref spectrum) = linear_spectrum {
        let input = |k_unit: f64| {
            let k = k_unit / r.box_size;
            (
                k,
                spectrum.filtered_power(
                    k,
                    r.dim,
                    r.box_size,
                    r.initial_smoothing,
                    d / d1,
                    r.initial_cutoff,
                ),
            )
        };
        let band: Vec<(f64, f64)> = ps
            .k
            .iter()
            .map(|&k| input(k))
            .filter(|(_, p)| *p > 0.)
            .collect();
        let fine_band: Vec<(f64, f64)> = ps
            .fine_k
            .iter()
            .map(|&k| input(k))
            .filter(|(_, p)| *p > 0.)
            .collect();
        let axis: Vec<(f64, f64)> = spec_ks
            .iter()
            .map(|&k| input(k))
            .filter(|(_, p)| *p > 0.)
            .collect();
        let shell: Vec<(f64, f64)> = raw
            .direct_shell_k
            .iter()
            .map(|&k| input(k))
            .filter(|(_, p)| *p > 0.)
            .collect();
        (
            band.iter().map(|v| v.0).collect::<Vec<_>>(),
            band.iter().map(|v| v.1).collect::<Vec<_>>(),
            fine_band.iter().map(|v| v.0).collect::<Vec<_>>(),
            fine_band.iter().map(|v| v.1).collect::<Vec<_>>(),
            axis.iter().map(|v| v.0).collect::<Vec<_>>(),
            axis.iter().map(|v| v.1).collect::<Vec<_>>(),
            shell.iter().map(|v| v.0).collect::<Vec<_>>(),
            shell.iter().map(|v| v.1).collect::<Vec<_>>(),
        )
    } else {
        (
            Vec::new(),
            Vec::new(),
            Vec::new(),
            Vec::new(),
            Vec::new(),
            Vec::new(),
            Vec::new(),
            Vec::new(),
        )
    };
    let (
        realized_k,
        realized_p,
        realized_counts,
        fine_realized_k,
        fine_realized_p,
        fine_realized_counts,
        direct_realized_k,
        direct_realized_p,
    ) = if r.ic == "gaussian" {
        let (ks_r, ps_r, cnts_r) =
            realized_linear_bins_with_count(c, &eg, d, spectrum_bins(eg.n));
        let (fks_r, fps_r, fcnts_r) =
            realized_linear_bins_with_count(c, &eg, d, eg.n.clamp(32, 256));
        let mut dk = Vec::new();
        let mut dp = Vec::new();
        for (&m, &k) in spec_modes.iter().zip(&spec_ks) {
            if m >= g.n / 2 {
                break;
            }
            let mut ijk = [0usize; 3];
            ijk[0] = m;
            let idx = g.ravel(ijk);
            dk.push(k);
            dp.push((d * c.delta0_hat[idx].norm() / g.size as f64).powi(2));
        }
        (
            physical_k(&ks_r),
            physical_p(&ps_r),
            cnts_r,
            physical_k(&fks_r),
            physical_p(&fps_r),
            fcnts_r,
            physical_k(&dk),
            physical_p(&dp),
        )
    } else {
        (
            Vec::new(),
            Vec::new(),
            Vec::new(),
            Vec::new(),
            Vec::new(),
            Vec::new(),
            Vec::new(),
            Vec::new(),
        )
    };
    let psc_nat_owned = if r.mesh > r.n {
        Some(power(&c.grid, cic_pdf_raw))
    } else {
        None
    };
    let psc_nat = psc_nat_owned.as_ref().unwrap_or(psc);
    let ps_realized_owned = if r.ic == "gaussian" {
        let mut sp = spectra::power_spectrum_hat(
            &c.grid,
            &c.delta0_hat,
            &c.delta0_hat,
            spectrum_bins(c.grid.n),
            None,
        );
        for p in &mut sp.p {
            *p *= d * d;
        }
        Some(sp)
    } else {
        None
    };
    let cell_scale = r.box_size / r.mesh as f64;
    let particle_scale = r.box_size / r.n as f64;
    let r_min = (0.25 * cell_scale).max(0.05);
    let r_max = (0.35 * r.box_size).max(r_min * 2.0);
    let n_radii = 36usize;
    let mut radii: Vec<f64> = (0..n_radii)
        .map(|i| r_min * (r_max / r_min).powf(i as f64 / (n_radii - 1) as f64))
        .collect();
    for special in [8.0, r.initial_smoothing, cell_scale, particle_scale] {
        if special >= r_min && special <= r_max {
            radii.push(special);
        }
    }
    radii.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    radii.dedup_by(|a, b| (*a - *b).abs() <= 1e-4 * b.abs().max(1e-12));
    let spectrum_variance = |sp: &spectra::Spectrum, radius_mpc: f64, gaussian_window: bool| -> f64 {
        let r_unit = radius_mpc / r.box_size;
        sp.k.iter()
            .zip(&sp.p)
            .zip(&sp.nmodes)
            .map(|((&k, &p), &nm)| {
                let w = cosmological::filter_window(r.dim, k * r_unit, gaussian_window);
                nm * p * w * w
            })
            .sum::<f64>()
    };
    let build_variance_curves = |gaussian_window: bool| -> Value {
        let curve = |sp: &spectra::Spectrum| -> Vec<f64> {
            radii
                .iter()
                .map(|&rad| spectrum_variance(sp, rad, gaussian_window))
                .collect()
        };
        let realized_curve = ps_realized_owned
            .as_ref()
            .map(curve)
            .unwrap_or_default();
        let linear_curve = linear_spectrum
            .as_ref()
            .map(|spec| {
                radii
                    .iter()
                    .map(|&rad| {
                        spec.filtered_variance(
                            r.dim,
                            r.initial_smoothing,
                            d / d1,
                            rad,
                            gaussian_window,
                        )
                    })
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default();
        json!({
            "sheet": curve(ps),
            "sheetP1": curve(ps_p1),
            "pm": curve(psp),
            "cic": curve(psc),
            "cicNatural": curve(psc_nat),
            "tsc": curve(pst),
            "realized": realized_curve,
            "linear": linear_curve,
        })
    };
    let variance_scale = json!({
        "radii": radii,
        "cellScale": cell_scale,
        "particleScale": particle_scale,
        "initialSmoothing": r.initial_smoothing,
        "unsmoothed": {
            "sheet": spectrum_variance(ps, 0.0, false),
            "sheetP1": spectrum_variance(ps_p1, 0.0, false),
            "pm": spectrum_variance(psp, 0.0, false),
            "cic": spectrum_variance(psc, 0.0, false),
            "cicNatural": spectrum_variance(psc_nat, 0.0, false),
            "tsc": spectrum_variance(pst, 0.0, false),
            "realized": ps_realized_owned.as_ref().map(|sp| spectrum_variance(sp, 0.0, false)),
            "linear": linear_spectrum.as_ref().map(|spec| spec.filtered_variance(r.dim, r.initial_smoothing, d / d1, 0.0, false)),
        },
        "tophat": build_variance_curves(false),
        "gaussian": build_variance_curves(true),
    });
    let initial_spectrum = if r.transfer == "bao" {
        "EH98 CDM+baryon transfer with baryon acoustic oscillations and Silk damping; no radiation in late-time growth or massive neutrinos"
    } else {
        "EH98 smooth baryon-suppressed transfer; no BAO, radiation in late-time growth, or massive neutrinos"
    };
    let safe = minj > 0. && mins > 0. && !pm.crossed;
    let crossing_diagnostic = if pm_arithmetic == "f64" {
        "f64 CPU polynomial extrema for every PM drift"
    } else {
        "f32 GPU polynomial extrema; f64 CPU recomputation when the GPU minimum is <=1e-3; f64 and f32 initial-sheet checks"
    };
    let first_pm_crossing = pm.first_crossing.as_ref().map(|f| {
        json!({
            "step":f.step,"aStart":f.a_start,"aEnd":f.a_end,"minimumJacobian":f.minimum_jacobian
        })
    });
    let a_sc = if sc.is_finite() && sc > 0. && sc <= growth.d_max() {
        Some(growth.a_of_d(sc))
    } else {
        None
    };
    let reason = if safe {
        "Positive resolved LPT orientation and PM drift orientation"
    } else {
        "Shell crossing detected: results remain visible; the single-stream fluid interpretation and controlled pre-crossing LPT validity no longer apply. Deposited density sums positive stream masses."
    };
    let dy = 3.6 / 80.;
    let omit_large_vertex_json = g.size > 262_144;
    let omit_large_mesh_json = eg.size > 262_144;
    let empty_f64: &[f64] = &[];
    let empty_f32: &[f32] = &[];
    let q_json = if omit_large_vertex_json { empty_f64 } else { q.as_slice() };
    let v_json = if omit_large_vertex_json { empty_f64 } else { v.as_slice() };
    let pmv_json = if omit_large_vertex_json { empty_f64 } else { pmv.as_slice() };
    let rs_json = if omit_large_mesh_json { empty_f32 } else { rs.as_slice() };
    let rs_p1_json = if omit_large_mesh_json { empty_f32 } else { rs_p1.as_slice() };
    let pms_json = if omit_large_mesh_json { empty_f32 } else { pms.as_slice() };
    let cics_json = if omit_large_mesh_json { empty_f32 } else { cics.as_slice() };
    let tscs_json = if omit_large_mesh_json { empty_f32 } else { tscs.as_slice() };
    let lpt_final_inverted_mass = observables::inverted_mass_fraction(g, &x);
    let pm_final_inverted_mass = observables::inverted_mass_fraction(g, pmx);
    let mut custom_series = serde_json::Map::new();
    let mut custom_rhos: Vec<(String, Vec<f32>)> = Vec::new();
    if !r.estimators.is_empty() {
        let active_lpt_tag = format!("{}lpt", r.order);
        for (traj, ord_opt) in [
            ("1lpt", Some(1usize)),
            ("2lpt", Some(2usize)),
            ("3lpt", Some(3usize)),
            ("4lpt", Some(4usize)),
            ("pm", None),
        ] {
            let prefix = format!("{traj}:");
            let wanted_ests: Vec<&str> = r
                .estimators
                .iter()
                .filter_map(|s| s.strip_prefix(&prefix))
                .filter(|&e| matches!(e, "p0" | "p1" | "ngp" | "cic" | "tsc"))
                .collect();
            if wanted_ests.is_empty() {
                continue;
            }
            let x_owned: Option<Vec<f64>> = match ord_opt {
                Some(ord) if ord != r.order => Some(evaluate(c, d, ord, 1.0).0),
                _ => None,
            };
            let x_traj: &[f64] = if traj == "pm" {
                pmx
            } else if let Some(ref xo) = x_owned {
                xo.as_slice()
            } else {
                &x
            };
            let pos_f32: Vec<f32> = x_traj.iter().map(|&v| v as f32).collect();
            let need_p0_fresh = wanted_ests.contains(&"p0")
                && traj != active_lpt_tag
                && traj != "pm";
            let need_p1_fresh = wanted_ests.contains(&"p1") && traj != active_lpt_tag;
            let w_p1: Option<Vec<f32>> = if wanted_ests.contains(&"p1") {
                if let Some(ord) = ord_opt {
                    let mut j = vec![0.0f64; g.size];
                    c.lpt_ref().jacobian(d, ord, &mut j);
                    Some(j.iter().map(|&v| (1.0 / v.abs().max(1e-6)) as f32).collect())
                } else {
                    Some(observables::vertex_inverse_jacobians(g, x_traj))
                }
            } else {
                None
            };
            let (mut pre_p0, mut pre_p1): (Option<Vec<f32>>, Option<Vec<f32>>) =
                if need_p0_fresh && need_p1_fresh && g.dim >= 2 {
                    let w_ref = w_p1.as_ref().unwrap();
                    let (r0, r1) = if g.dim == 2 {
                        sheet::sheet_density_2d_exact_p0_p1(g, &pos_f32, w_ref, r.mesh)
                    } else {
                        sheet::sheet_density_3d_exact_p0_p1(g, &pos_f32, w_ref, r.mesh)
                    };
                    (Some(r0), Some(r1))
                } else {
                    (None, None)
                };
            for &est in &wanted_ests {
                let key = format!("{traj}:{est}");
                if custom_series.contains_key(&key) {
                    continue;
                }
                let (raw_owned, smooth_owned, ps_owned): (
                    Option<Vec<f32>>,
                    Option<Vec<f32>>,
                    Option<spectra::Spectrum>,
                );
                let (raw_ref, smooth_ref, ps_ref): (&[f32], &[f32], &spectra::Spectrum) =
                    if traj == active_lpt_tag && est == "p0" {
                        (rho.as_slice(), rs.as_slice(), ps)
                    } else if traj == active_lpt_tag && est == "p1" {
                        (rho_p1.as_slice(), rs_p1.as_slice(), ps_p1)
                    } else if traj == active_lpt_tag && est == "cic" {
                        (cic.as_slice(), cics.as_slice(), psc)
                    } else if traj == active_lpt_tag && est == "tsc" {
                        (raw.tsc.as_slice(), tscs.as_slice(), pst)
                    } else if traj == "pm" && est == "p0" {
                        (pmrho.as_slice(), pms.as_slice(), psp)
                    } else {
                        let r_vec = match est {
                            "p0" => pre_p0.take().unwrap_or_else(|| density(g, x_traj, r.mesh)),
                            "p1" => pre_p1.take().unwrap_or_else(|| {
                                let w_ref = w_p1.as_ref().unwrap();
                                if g.dim == 1 {
                                    observables::interval_density_p1(g, x_traj, w_ref, r.mesh)
                                } else if g.dim == 2 {
                                    sheet::sheet_density_2d_exact(g, &pos_f32, Some(w_ref), r.mesh)
                                } else {
                                    sheet::sheet_density_3d_exact(g, &pos_f32, Some(w_ref), r.mesh)
                                }
                            }),
                            "ngp" => sheet::ngp_density(r.dim, &pos_f32, g.size, 1.0, r.mesh),
                            "cic" => sheet::cic_density(r.dim, &pos_f32, g.size, 1.0, r.mesh),
                            "tsc" => sheet::tsc_density(r.dim, &pos_f32, g.size, 1.0, r.mesh),
                            _ => unreachable!(),
                        };
                        let s_vec = observables::smooth_density(&eg, &r_vec, r.smoothing);
                        let p_spec = power(&eg, &r_vec);
                        raw_owned = Some(r_vec);
                        smooth_owned = Some(s_vec);
                        ps_owned = Some(p_spec);
                        (
                            raw_owned.as_ref().unwrap().as_slice(),
                            smooth_owned.as_ref().unwrap().as_slice(),
                            ps_owned.as_ref().unwrap(),
                        )
                    };
                let (logrho_est, vol_est, mass_est, var_est, dy_est, summary_est) =
                    if matches!(est, "p0" | "p1") {
                        if traj == active_lpt_tag && est == "p0" {
                            (
                                logrho.clone(),
                                vol.clone(),
                                mass.clone(),
                                var,
                                dy_p0,
                                summary.clone(),
                            )
                        } else if traj == active_lpt_tag && est == "p1" {
                            (
                                p1_logrho.clone(),
                                p1vol.clone(),
                                p1mass.clone(),
                                p1var,
                                dy_p1,
                                p1_summary.clone(),
                            )
                        } else if traj == "pm" && est == "p0" {
                            (
                                pm_logrho.clone(),
                                pmvol.clone(),
                                pmmass.clone(),
                                pmvar,
                                dy_pm,
                                pm_summary.clone(),
                            )
                        } else {
                            let w_simplex_p1 = if est == "p1" {
                                w_p1.as_deref()
                            } else {
                                None
                            };
                            let (lr_e, v_e, m_e, _, var_e, dy_e, sum_e) =
                                observables::simplex_density_pdf_and_summary(
                                    g,
                                    x_traj,
                                    w_simplex_p1,
                                    80,
                                    &probabilities,
                                );
                            (lr_e, v_e, m_e, var_e, dy_e, sum_e)
                        }
                    } else {
                        let (pdf_raw_nat, pdf_smooth_nat);
                        let (pdf_raw, pdf_smooth): (&[f32], &[f32]) =
                            if r.mesh > r.n && matches!(est, "ngp" | "cic" | "tsc") {
                                pdf_raw_nat = match est {
                                    "ngp" => sheet::ngp_density(r.dim, &pos_f32, g.size, 1.0, r.n),
                                    "cic" => sheet::cic_density(r.dim, &pos_f32, g.size, 1.0, r.n),
                                    "tsc" => sheet::tsc_density(r.dim, &pos_f32, g.size, 1.0, r.n),
                                    _ => unreachable!(),
                                };
                                pdf_smooth_nat =
                                    observables::smooth_density(&c.grid, &pdf_raw_nat, r.smoothing);
                                (&pdf_raw_nat, &pdf_smooth_nat)
                            } else {
                                (raw_ref, smooth_ref)
                            };
                        let (lr_e, v_e, m_e, _, var_e) =
                            observables::density_pdf(pdf_smooth, pdf_raw, 80);
                        let sum_e =
                            observables::density_summary(pdf_smooth, pdf_raw, &probabilities);
                        (lr_e, v_e, m_e, var_e, dy, sum_e)
                    };
                let direct_shell = if est == "p0" {
                    if traj == active_lpt_tag {
                        Some(physical_p(&raw.direct_shell_lpt))
                    } else if traj == "pm" {
                        Some(physical_p(&raw.direct_shell_pm))
                    } else {
                        Some(physical_p(&compute_meshfree_shell_single(pending, x_traj)))
                    }
                } else {
                    None
                };
                let var_tophat: Vec<f64> = radii
                    .iter()
                    .map(|&rad| spectrum_variance(ps_ref, rad, false))
                    .collect();
                let var_gaussian: Vec<f64> = radii
                    .iter()
                    .map(|&rad| spectrum_variance(ps_ref, rad, true))
                    .collect();
                let var_unsmoothed = spectrum_variance(ps_ref, 0.0, false);
                let rho_json_slice = if omit_large_mesh_json {
                    empty_f32
                } else {
                    smooth_ref
                };
                custom_rhos.push((key.clone(), smooth_ref.to_vec()));
                custom_series.insert(
                    key.clone(),
                    json!({
                        "key": key,
                        "traj": traj,
                        "est": est,
                        "rho": rho_json_slice,
                        "massMean": mean(raw_ref),
                        "power": physical_p(&ps_ref.p),
                        "counts": ps_ref.nmodes,
                        "fineK": physical_k(&ps_ref.fine_k),
                        "finePower": physical_p(&ps_ref.fine_p),
                        "fineCounts": ps_ref.fine_nmodes,
                        "directShell": direct_shell,
                        "variance": {
                            "unsmoothed": var_unsmoothed,
                            "tophat": var_tophat,
                            "gaussian": var_gaussian,
                        },
                        "pdf": {
                            "logrho": logrho_est,
                            "dy": dy_est,
                            "volume": vol_est,
                            "mass": mass_est,
                            "variance": var_est,
                            "capturedVolume": vol_est.iter().sum::<f64>() * dy_est,
                            "capturedMass": mass_est.iter().sum::<f64>() * dy_est,
                            "summary": summary_est,
                        }
                    }),
                );
            }
        }
    }
    let f_growth = growth_rate(growth, a);
    let e_hubble = c.cosmo.e(a);
    let rsd = observables::redshift_space_analysis(
        g,
        &x,
        &v,
        a,
        f_growth,
        e_hubble,
        r.box_size,
        r.mesh,
        r.smoothing,
    );
    let result = json!({"dim":r.dim,"n":r.n,"mesh":r.mesh,"forceMesh":r.force_mesh,"box":1.,"boxSize":r.box_size,"a":a,"aStart":a_start,"D":d,"dDa":d_da,"redshift":1./a-1.,"growthAtOne":growth.d_of_a(1.),"shellCrossing":sc,"aShellCrossing":a_sc,
        "request":r,"engineVersion":env!("CARGO_PKG_VERSION"),"schemaVersion":2,"estimatedPeakBytes":r.estimated_peak_bytes(),
        "crossingReference":"Minimum ZA growth threshold and reachable sampled-vertex 2--4LPT crossing estimates; not a guarantee for the affine mesh or PM trajectory",
        "cosmology":{"omegaM":r.omega_m,"omegaB":r.omega_b,"h":r.h,"ns":r.ns,"sigma8":r.sigma8},
        "guard":{"ok":safe,"crossed":!safe,"pmCrossed":pm.crossed,"firstPmCrossing":first_pm_crossing,"minimumJacobian":minj,"minimumSimplexJacobian":mins.min(pm.min_jacobian),"pmMinimumSimplexJacobian":pm.min_jacobian,"pmInitialMinimumJacobianF64":pending.pm_initial_min_jacobian,"pmInitialMinimumJacobianF32":pending.pm_initial_quantized_min_jacobian,"pmInitialMinimumJacobianDisplacementF32":pending.pm_initial_displacement_quantized_min_jacobian,"pmInitialMinimumJacobianUsed":pm_initial_minimum_used,"pmInitialMinimumJacobianRepresentation":pm_initial_minimum_representation,"lptFinalInvertedMass":lpt_final_inverted_mass,"pmFinalInvertedMass":pm_final_inverted_mass,"reason":reason},
        "q":q_json,"x":[],"v":v_json,"pmX":[],"pmV":pmv_json,"rho":rs_json,"rhoP1":rs_p1_json,"pmRho":pms_json,"cicRho":cics_json,"tscRho":tscs_json,
        "customSeries":custom_series,
        "rsd":rsd,
        "spectra":{"k":physical_k(&ps.k),"sheet":physical_p(&ps.p),"sheetP1":physical_p(&ps_p1.p),"cic":physical_p(&psc.p),"tsc":physical_p(&pst.p),"pm":physical_p(&psp.p),"counts":ps.nmodes,"fineK":physical_k(&ps.fine_k),"fineCounts":ps.fine_nmodes,"fineSheet":physical_p(&ps.fine_p),"fineSheetP1":physical_p(&ps_p1.fine_p),"fineCic":physical_p(&psc.fine_p),"fineTsc":physical_p(&pst.fine_p),"finePm":physical_p(&psp.fine_p),"directK":physical_k(&spec_ks),"direct":physical_p(spec_powers),"directPM":physical_p(spec_pm_powers),"directShellK":physical_k(&raw.direct_shell_k),"directShell":physical_p(&raw.direct_shell_lpt),"directShellPM":physical_p(&raw.direct_shell_pm),"directShellRealizedK":physical_k(&raw.direct_shell_realized_k),"directShellRealized":physical_p(&raw.direct_shell_realized),"directShellLinearK":direct_shell_linear_k,"directShellLinear":direct_shell_linear_p,"linearK":linear_k,"linear":linear_p,"fineLinearK":fine_linear_k,"fineLinear":fine_linear_p,"directLinearK":direct_linear_k,"directLinear":direct_linear_p,"realizedK":realized_k,"realized":realized_p,"realizedCounts":realized_counts,"fineRealizedK":fine_realized_k,"fineRealized":fine_realized_p,"fineRealizedCounts":fine_realized_counts,"directRealizedK":direct_realized_k,"directRealized":direct_realized_p},
        "varianceScale":variance_scale,
        "pdf":{"logrho":logrho,"p1Logrho":p1_logrho,"pmLogrho":pm_logrho,"cicLogrho":cic_logrho,"dy":dy_p0,"p1Dy":dy_p1,"pmDy":dy_pm,"cicDy":dy_cic,"volume":vol,"mass":mass,"p1Volume":p1vol,"p1Mass":p1mass,"pmVolume":pmvol,"pmMass":pmmass,"cicVolume":cicvol,"cicMass":cicmass,"mean":pdfmean,"variance":var,"p1Variance":p1var,"pmVariance":pmvar,"cicVariance":cicvar,"summary":summary,"p1Summary":p1_summary,"pmSummary":pm_summary,"cicSummary":cic_summary,"capturedVolume":vol.iter().sum::<f64>()*dy_p0,"capturedMass":mass.iter().sum::<f64>()*dy_p0,"p1CapturedVolume":p1vol.iter().sum::<f64>()*dy_p1,"p1CapturedMass":p1mass.iter().sum::<f64>()*dy_p1,"pmCapturedVolume":pmvol.iter().sum::<f64>()*dy_pm,"pmCapturedMass":pmmass.iter().sum::<f64>()*dy_pm,"cicCapturedVolume":cicvol.iter().sum::<f64>()*dy_cic,"cicCapturedMass":cicmass.iter().sum::<f64>()*dy_cic},
        "gradient":{"k":physical_k(&ks),"power":physical_p(&powers),"tangent":physical_p(&derivatives),"finiteDifference":physical_p(&fd),"relativeError":errs,"pmTangent":physical_p(&pmderivs),"parameter":"initial amplitude A at A=1, fixed epoch","finiteDifferenceStep":h},
        "diagnostics":{"mass":mean(&rho),"p1Mass":mean(&rho_p1),"pmMass":mean(&pmrho),"cicMass":mean(&raw.cic),"tscMass":mean(&raw.tsc),"positionRms":position_rms,"positionRelativeRms":position_rms/displacement_rms.max(1e-30),"powerRelativeError":power_error},
        "conventions":{"liveCosmology":if c.cosmo.is_eds(){"EdS"}else{"flat LCDM"},"initialBandCutoff":r.initial_cutoff,
        "modelGeometry":if r.dim==3 {"3D periodic matter field"} else {"Translationally invariant subspace of 3D cosmological equations; not a random 3D slice"},
        "initialSpectrum":initial_spectrum,"transfer":r.transfer,
        "initialNormalization":"Gaussian spectrum sigma8 at z=0 before Gaussian filtering and radial cutoff; no realization RMS rescaling. Deterministic waves use fixed delta0 amplitudes independent of sigma8.",
        "initialFilter":"Gaussian exp(-k^2 R_initial^2/2) plus fixed radial integer-mode cutoff",
        "modeGenerator":"PSC Fourier mode hash v2, SplitMix64 and xoshiro256** Box-Muller, exact Hermitian pairing",
        "positionUnits":"x/L","velocityUnits":"d(x/L)/da","wavenumberUnits":"h/Mpc","powerUnits":format!("(Mpc/h)^{}",r.dim),"observationFilter":"Gaussian","powerField":"unsmoothed cell averages","pdfMassWeights":"unsmoothed cell mass","pmObservation":"exact P0 sheet attached to PM vertices","pmArithmetic":pm_arithmetic,"initialSource":pending.initial_source,"crossingDiagnostic":crossing_diagnostic},"cache":pending.cache_status});
    let grid = &c.grid;
    Ok(PreparedLabRun {
        request: r.clone(),
        result,
        modes: mode_records,
        h,
        lpt_displacement: packed_vec4(grid, &x, true),
        lpt_direction: packed_vec4(grid, &dpos, false),
        pm_displacement: packed_vec4(grid, pmx, true),
        pm_direction: packed_vec4(grid, &pmdpos, false),
        q,
        x: x.clone(),
        v: v.clone(),
        dpos: dpos.clone(),
        xplus: xplus.clone(),
        xminus: xminus.clone(),
        pmx: pmx.clone(),
        pmv,
        pmdpos,
        rs,
        rs_p1,
        pms,
        cics,
        tscs,
        custom_rhos,
        omit_large_vertex_json,
        omit_large_mesh_json,
        lpt_displacement_expansion: OnceLock::new(),
        lpt_direction_expansion: OnceLock::new(),
        cache_status: pending.cache_status.clone(),
        fd_channels: OnceLock::new(),
        fd_plus_expansion: OnceLock::new(),
        fd_minus_expansion: OnceLock::new(),
    })
}

impl PreparedLabRun {
    pub fn dim(&self) -> usize {
        self.request.dim
    }
    pub fn n(&self) -> usize {
        self.request.n
    }
    pub fn modes(&self) -> &[[i32; 4]] {
        &self.modes
    }
    pub fn q_f64(&self) -> &[f64] {
        &self.q
    }
    pub fn x_f64(&self) -> &[f64] {
        &self.x
    }
    pub fn v_f64(&self) -> &[f64] {
        &self.v
    }
    pub fn pm_x_f64(&self) -> &[f64] {
        &self.pmx
    }
    pub fn pm_v_f64(&self) -> &[f64] {
        &self.pmv
    }
    pub fn rho_f32(&self) -> &[f32] {
        &self.rs
    }
    pub fn rho_p1_f32(&self) -> &[f32] {
        &self.rs_p1
    }
    pub fn pm_rho_f32(&self) -> &[f32] {
        &self.pms
    }
    pub fn cic_rho_f32(&self) -> &[f32] {
        &self.cics
    }
    pub fn tsc_rho_f32(&self) -> &[f32] {
        &self.tscs
    }
    pub fn custom_rho_f32(&self, key: &str) -> Vec<f32> {
        self.custom_rhos
            .iter()
            .find(|(k, _)| k == key)
            .map(|(_, v)| v.clone())
            .unwrap_or_default()
    }
    pub fn lpt_displacement(&self) -> &[f32] {
        &self.lpt_displacement
    }
    pub fn lpt_direction(&self) -> &[f32] {
        &self.lpt_direction
    }
    fn lpt_expansion(&self, direction: bool) -> &(Vec<f32>, Vec<f32>) {
        let cache = if direction {
            &self.lpt_direction_expansion
        } else {
            &self.lpt_displacement_expansion
        };
        cache.get_or_init(|| {
            let grid = Grid::new(self.request.dim, self.request.n, 1.0);
            let mut hi = vec![0.0f32; grid.size * 4];
            let mut lo = vec![0.0f32; grid.size * 4];
            for i in 0..grid.size {
                let q = grid.unravel(i);
                for axis in 0..grid.dim {
                    let value = if direction {
                        self.dpos[i * grid.dim + axis]
                    } else {
                        self.x[i * grid.dim + axis] - q[axis] as f64 * grid.dx()
                    };
                    let high = value as f32;
                    hi[i * 4 + axis] = high;
                    lo[i * 4 + axis] = (value - high as f64) as f32;
                }
            }
            (hi, lo)
        })
    }
    pub fn lpt_displacement_hi(&self) -> &[f32] {
        &self.lpt_expansion(false).0
    }
    pub fn lpt_displacement_lo(&self) -> &[f32] {
        &self.lpt_expansion(false).1
    }
    pub fn lpt_direction_hi(&self) -> &[f32] {
        &self.lpt_expansion(true).0
    }
    pub fn lpt_direction_lo(&self) -> &[f32] {
        &self.lpt_expansion(true).1
    }
    pub fn pm_displacement(&self) -> &[f32] {
        &self.pm_displacement
    }
    pub fn pm_direction(&self) -> &[f32] {
        &self.pm_direction
    }
    pub fn direct_cpu(&self) -> Vec<f64> {
        self.direct_with_fields(&self.x, &self.dpos, &self.pmx, &self.pmdpos)
    }

    /// Recompute selected LPT channels from retained f64 positions. Each requested
    /// index addresses a mode record and returns [F.re,F.im,dF.re,dF.im].
    pub fn direct_lpt_cpu_modes(&self, indices: &[u32]) -> Result<Vec<f64>, String> {
        let grid = Grid::new(self.request.dim, self.request.n, 1.0);
        let mut out = Vec::with_capacity(indices.len() * 4);
        for &index in indices {
            let mode = self.modes.get(index as usize).ok_or_else(|| {
                format!(
                    "Direct mode index {index} is out of range (mode count {})",
                    self.modes.len()
                )
            })?;
            let k = [grid.kf() * mode[0] as f64, 0.0, 0.0];
            let (f, df) = observables::sheet_mode_jvp(&grid, &self.x, &self.dpos, k);
            out.extend([f.re, f.im, df.re, df.im]);
        }
        Ok(out)
    }

    /// CPU arithmetic on the exact f32-packed displacement/direction inputs sent
    /// to the GPU, separating input quantization from GPU kernel arithmetic.
    pub fn direct_quantized_cpu(&self) -> Vec<f64> {
        let grid = Grid::new(self.request.dim, self.request.n, 1.0);
        let unpack = |packed: &[f32], displacement: bool| {
            let mut values = vec![0.0; grid.size * grid.dim];
            for i in 0..grid.size {
                let ijk = grid.unravel(i);
                for a in 0..grid.dim {
                    values[i * grid.dim + a] = packed[4 * i + a] as f64
                        + if displacement {
                            ijk[a] as f64 * grid.dx()
                        } else {
                            0.0
                        };
                }
            }
            values
        };
        let x = unpack(&self.lpt_displacement, true);
        let dx = unpack(&self.lpt_direction, false);
        let pmx = unpack(&self.pm_displacement, true);
        let pmdx = unpack(&self.pm_direction, false);
        self.direct_with_fields(&x, &dx, &pmx, &pmdx)
    }

    fn direct_with_fields(&self, x: &[f64], dx: &[f64], pmx: &[f64], pmdx: &[f64]) -> Vec<f64> {
        let grid = Grid::new(self.request.dim, self.request.n, 1.0);
        let need_pm = self.request.needs_pm();
        let mut out = Vec::with_capacity(self.modes.len() * 8);
        for mode in &self.modes {
            let k = [grid.kf() * mode[0] as f64, 0.0, 0.0];
            let (f, df) = observables::sheet_mode_jvp(&grid, x, dx, k);
            let (pmf, pmdf) = if need_pm {
                observables::sheet_mode_jvp(&grid, pmx, pmdx, k)
            } else {
                (f, df)
            };
            out.extend([f.re, f.im, df.re, df.im, pmf.re, pmf.im, pmdf.re, pmdf.im]);
        }
        out
    }

    /// Finite-difference power derivatives from the retained f64 positions.
    /// Values are raw-box power derivatives, cached so a GPU fallback can reuse them.
    pub fn finite_difference_cpu(&self) -> &[f64] {
        self.fd_channels.get_or_init(|| {
            let grid = Grid::new(self.request.dim, self.request.n, 1.0);
            if grid.size > 65_536 && self.modes.iter().all(|m| m[0] > 0 && m[1] == 0 && m[2] == 0) {
                let ms: Vec<usize> = self.modes.iter().map(|m| m[0] as usize).collect();
                let f_plus = observables::sheet_modes_first_axis(&grid, &self.xplus, &ms);
                let f_minus = observables::sheet_modes_first_axis(&grid, &self.xminus, &ms);
                return f_plus
                    .into_iter()
                    .zip(f_minus)
                    .map(|(fp, fm)| (fp.norm_sqr() - fm.norm_sqr()) / (2.0 * self.h))
                    .collect();
            }
            self.modes
                .iter()
                .map(|mode| {
                    let k = [grid.kf() * mode[0] as f64, 0.0, 0.0];
                    (observables::sheet_mode(&grid, &self.xplus, k).norm_sqr()
                        - observables::sheet_mode(&grid, &self.xminus, k).norm_sqr())
                        / (2.0 * self.h)
                })
                .collect()
        })
    }

    fn finite_difference_expansion(&self, plus: bool) -> &(Vec<f32>, Vec<f32>) {
        let grid = Grid::new(self.request.dim, self.request.n, 1.0);
        let values = if plus { &self.xplus } else { &self.xminus };
        let cache = if plus {
            &self.fd_plus_expansion
        } else {
            &self.fd_minus_expansion
        };
        cache.get_or_init(|| {
            let mut hi = vec![0.0f32; grid.size * 4];
            let mut lo = vec![0.0f32; grid.size * 4];
            for i in 0..grid.size {
                let q = grid.unravel(i);
                for axis in 0..grid.dim {
                    let displacement = values[i * grid.dim + axis] - q[axis] as f64 * grid.dx();
                    let high = displacement as f32;
                    hi[i * 4 + axis] = high;
                    lo[i * 4 + axis] = (displacement - high as f64) as f32;
                }
            }
            (hi, lo)
        })
    }

    /// High part of the f32 expansion of x(A+h)−q, packed as zero-padded vec4.
    pub fn finite_difference_plus_hi(&self) -> &[f32] {
        &self.finite_difference_expansion(true).0
    }
    /// Low residual of the f32 expansion of x(A+h)−q, packed as zero-padded vec4.
    pub fn finite_difference_plus_lo(&self) -> &[f32] {
        &self.finite_difference_expansion(true).1
    }
    /// High part of the f32 expansion of x(A−h)−q, packed as zero-padded vec4.
    pub fn finite_difference_minus_hi(&self) -> &[f32] {
        &self.finite_difference_expansion(false).0
    }
    /// Low residual of the f32 expansion of x(A−h)−q, packed as zero-padded vec4.
    pub fn finite_difference_minus_lo(&self) -> &[f32] {
        &self.finite_difference_expansion(false).1
    }

    pub fn metadata(&self) -> Value {
        json!({"dim":self.request.dim,"n":self.request.n,"modeCount":self.modes.len(),"cache":self.cache_status,
            "modes":self.modes.iter().map(|m| [m[0],m[1],m[2]]).collect::<Vec<_>>(),"h":self.h,
            "channelScalarsPerMode":8})
    }

    pub fn finish(&self, channels: &[f64], execution: &str) -> Result<Value, String> {
        self.finish_with_finite_difference(channels, self.finite_difference_cpu(), execution)
    }

    /// Finalize with caller-supplied raw-box finite-difference power derivatives.
    /// This path does not invoke the retained CPU finite-difference calculation.
    pub fn finish_with_finite_difference(
        &self,
        channels: &[f64],
        finite_difference: &[f64],
        execution: &str,
    ) -> Result<Value, String> {
        let expected = self.modes.len() * 8;
        if channels.len() != expected {
            return Err(format!(
                "Expected {expected} direct channel scalars, got {}",
                channels.len()
            ));
        }
        if channels.iter().any(|v| !v.is_finite()) {
            return Err("Direct channel values must all be finite".into());
        }
        if finite_difference.len() != self.modes.len() {
            return Err(format!(
                "Expected {} finite-difference channel scalars, got {}",
                self.modes.len(),
                finite_difference.len()
            ));
        }
        if finite_difference.iter().any(|v| !v.is_finite()) {
            return Err("Finite-difference channel values must all be finite".into());
        }
        let execution: Value = serde_json::from_str(execution)
            .map_err(|e| format!("Invalid execution metadata JSON: {e}"))?;
        let volume = self.request.box_size.powi(self.request.dim as i32);
        let mut powers = Vec::with_capacity(self.modes.len());
        let mut derivatives = Vec::with_capacity(self.modes.len());
        let mut pmpowers = Vec::with_capacity(self.modes.len());
        let mut pmderivs = Vec::with_capacity(self.modes.len());
        let mut errors = Vec::with_capacity(self.modes.len());
        let mut fd_values = Vec::with_capacity(self.modes.len());
        for (i, c) in channels.chunks_exact(8).enumerate() {
            let f = Complex64::new(c[0], c[1]);
            let df = Complex64::new(c[2], c[3]);
            let pmf = Complex64::new(c[4], c[5]);
            let pmdf = Complex64::new(c[6], c[7]);
            let p = f.norm_sqr();
            let dp = 2.0 * (f.conj() * df).re;
            let pmp = pmf.norm_sqr();
            let pmdp = 2.0 * (pmf.conj() * pmdf).re;
            let fdv = finite_difference[i];
            fd_values.push(fdv * volume);
            powers.push(p * volume);
            derivatives.push(dp * volume);
            pmpowers.push(pmp * volume);
            pmderivs.push(pmdp * volume);
            errors.push((dp - fdv).abs() / dp.abs().max(1e-9));
        }
        let mut out = self.result.clone();
        if !self.omit_large_vertex_json {
            out["x"] = json!(self.x);
            out["pmX"] = json!(self.pmx);
        }
        if let Some(arr) = out["spectra"]["direct"].as_array_mut() {
            for (dst, &val) in arr.iter_mut().zip(&powers) {
                *dst = json!(val);
            }
        } else {
            out["spectra"]["direct"] = json!(powers);
        }
        if let Some(arr) = out["spectra"]["directPM"].as_array_mut() {
            for (dst, &val) in arr.iter_mut().zip(&pmpowers) {
                *dst = json!(val);
            }
        } else {
            out["spectra"]["directPM"] = json!(pmpowers);
        }
        out["gradient"]["power"] = json!(powers);
        out["gradient"]["tangent"] = json!(derivatives);
        out["gradient"]["finiteDifference"] = json!(fd_values);
        out["gradient"]["pmTangent"] = json!(pmderivs);
        out["gradient"]["relativeError"] = json!(errors);
        out["execution"] = execution;
        Ok(out)
    }
}

pub fn run(r: &LabRequest) -> Result<Value, String> {
    let prepared = prepare(r)?;
    let channels = prepared.direct_cpu();
    let mut out = prepared.finish(&channels, &json!({"backend":"cpu"}).to_string())?;
    if prepared.omit_large_vertex_json {
        out["q"] = json!(prepared.q);
        out["x"] = json!(prepared.x);
        out["v"] = json!(prepared.v);
        out["pmX"] = json!(prepared.pmx);
        out["pmV"] = json!(prepared.pmv);
    }
    if prepared.omit_large_mesh_json {
        out["rho"] = json!(prepared.rs);
        out["rhoP1"] = json!(prepared.rs_p1);
        out["pmRho"] = json!(prepared.pms);
        out["cicRho"] = json!(prepared.cics);
        out["tscRho"] = json!(prepared.tscs);
        for (k, v) in &prepared.custom_rhos {
            if let Some(entry) = out["customSeries"].get_mut(k) {
                entry["rho"] = json!(v);
            }
        }
    }
    Ok(out)
}
/// Stable JSON facade used by WebAssembly and native integration tests.
pub fn run_json(request: &str) -> String {
    let result = serde_json::from_str::<Value>(request)
        .map_err(|e| format!("Invalid laboratory JSON: {e}"))
        .and_then(|v| {
            if let Some(version) = v.get("schemaVersion").and_then(Value::as_u64) {
                if version != 2 && version != 3 {return Err(format!("Unsupported experiment schema {version}; use the recorded repository revision"));}
            }
            if v.get("state").and_then(|s|s.get("view")).and_then(Value::as_str)==Some("published") {
                return Err("An archival replay manifest preserves packaged evidence; it cannot be recomputed by the live engine".into());
            }
            // Native callers may supply a request, a complete response, or the
            // browser's exported experiment manifest. Prefer the executed
            // request to display state, and never replay a stale plot silently.
            let inputs = v.get("result").and_then(|r|r.get("request"))
                .or_else(|| v.get("request"))
                .or_else(|| v.get("state"))
                .unwrap_or(&v).clone();
            serde_json::from_value::<LabRequest>(inputs).map_err(|e|format!("Invalid experiment inputs: {e}"))
        })
        .and_then(|r| run(&r));
    match result {
        Ok(v) => v.to_string(),
        Err(e) => json!({"error":e}).to_string(),
    }
}
