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
use std::sync::OnceLock;

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
        }
    }
}
impl LabRequest {
    /// Conservative allocation estimate including LPT tensors, tangents, FFT
    /// workspaces, result arrays and JSON serialization. Not a measured peak.
    pub fn estimated_peak_bytes(&self) -> usize {
        let size = self.n.saturating_pow(self.dim.min(3) as u32);
        let mesh = self
            .mesh
            .max(self.force_mesh)
            .saturating_pow(self.dim.min(3) as u32);
        size.saturating_mul(2400)
            .saturating_add(mesh.saturating_mul(250))
    }
    fn validate(&self) -> Result<(), String> {
        const GRIDS: &[usize] = &[16, 24, 32, 48, 64, 96, 128, 256, 512, 1024];
        if !(1..=3).contains(&self.dim)
            || !GRIDS.contains(&self.n)
            || !GRIDS.contains(&self.mesh)
            || !GRIDS.contains(&self.force_mesh)
            || !(1..=4).contains(&self.order)
        {
            return Err("Invalid dimension/order or grid; supported grids are 16,24,32,48,64,96,128,256,512,1024".into());
        }
        let cap = [0, 1024, 256, 64][self.dim];
        let mesh_cap = [0, 1024, 256, 128][self.dim];
        if self.n > cap
            || self.mesh > mesh_cap
            || self.force_mesh > mesh_cap
            || self.estimated_peak_bytes() > 1024 * 1024 * 1024
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
        if !self.smoothing.is_finite()
            || !(0.0..=0.15).contains(&self.smoothing)
            || ![32, 64, 128, 256, 512].contains(&self.steps)
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
        if self.initial_cutoff == 0 || self.initial_cutoff > 32 {
            return Err("Initial radial mode cutoff must be in [1,32]".into());
        }
        if self.ic == "gaussian" && self.n <= 8 * self.initial_cutoff {
            return Err("Gaussian 4LPT requires lattice n > 8*initialCutoff so the fourth-order products remain below Nyquist. Refine the lattice or lower the explicit cutoff.".into());
        }
        if self.ic != "waves" && self.ic != "gaussian" {
            return Err("Unknown initial-condition family".into());
        }
        if self.seed > 9999 {
            return Err("Seed must be in [0,9999]".into());
        }
        Ok(())
    }
}
/// Construct the same Fourier realization at every supported refinement.
pub fn initial_conditions(r: &LabRequest) -> Cosmo {
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
        );
        c.sigma0 = c
            .delta0_hat
            .iter()
            .map(|v| v.norm_sqr())
            .sum::<f64>()
            .sqrt()
            / c.grid.size as f64;
    }
    c.build_lpt(4);
    c
}
/// Double-precision evaluation of the inherited spatial terms. The terms
/// themselves retain upstream f32 precision, but time sums and amplitudes do not.
fn evaluate(c: &Cosmo, d: f64, order: usize, amp: f64) -> (Vec<f64>, Vec<f64>) {
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
fn power(g: &Grid, rho: &[f32]) -> spectra::Spectrum {
    let mut eng = FftEngine::new(g);
    let contrast: Vec<f32> = rho.iter().map(|&v| v - 1.).collect();
    spectra::power_spectrum(g, &mut eng, &contrast, 12)
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
    x: Vec<f64>,
    dpos: Vec<f64>,
    xplus: Vec<f64>,
    xminus: Vec<f64>,
    pmx: Vec<f64>,
    pmdpos: Vec<f64>,
    fd_channels: OnceLock<Vec<f64>>,
}

/// Shared f64 physical preparation before the final PM evolution. The PM seed,
/// tangent and clocks remain available when a GPU attempt fails, so fallback
/// can finish this exact run without rebuilding the realization or LPT fields.
pub struct PendingLabRun {
    request: LabRequest,
    c: Cosmo,
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
    pm_reference_cache: OnceLock<Result<PmRun, String>>,
    pm_reference_quantized_cache: OnceLock<Result<PmRun, String>>,
    pm_reference_displacement_quantized_cache: OnceLock<Result<PmRun, String>>,
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

/// Prepare the common realization and retain the initial PM state for either backend.
pub fn prepare_pending(r: &LabRequest) -> Result<PendingLabRun, String> {
    r.validate()?;
    let c = initial_conditions(r);
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
    for t in 0..=16 {
        let dt = d * t as f64 / 16.;
        if t == 0 {
            minj = minj.min(1.);
            mins = mins.min(1.);
            continue;
        }
        minj = minj.min(c.lpt_ref().min_jacobian(dt, r.order));
        let (xt, _) = evaluate(&c, dt, r.order, 1.);
        mins = mins.min(observables::minimum_simplex_jacobian(g, &xt));
    }
    // Common growing initial condition uses all four orders, irrespective of
    // the order selected for the LPT comparison at the output.
    let (xi, vi) = evaluate(&c, d_start, 4, 1.0);
    let (dx, dv) = tangent::lpt_amplitude_jvp(c.lpt_ref(), d_start, 4);
    let p: Vec<f64> = vi
        .iter()
        .map(|v| a_start.powi(3) * c.cosmo.e(a_start) * d_da_start * v)
        .collect();
    let dp = dv
        .iter()
        .map(|v| a_start.powi(3) * c.cosmo.e(a_start) * d_da_start * v)
        .collect();
    let pm_initial_min_jacobian = observables::minimum_simplex_jacobian(g, &xi);
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
    let mut pm_clocks = Vec::with_capacity(r.steps * 5);
    for s in 0..r.steps {
        let a0 = a_start + (a - a_start) * s as f64 / r.steps as f64;
        let a1 = a_start + (a - a_start) * (s + 1) as f64 / r.steps as f64;
        let amid = 0.5 * (a0 + a1);
        let kick0 = cosmology_clocks(Cosmology { omega_m: r.omega_m }, a0, amid).0;
        let drift = cosmology_clocks(Cosmology { omega_m: r.omega_m }, a0, a1).1;
        let kick1 = cosmology_clocks(Cosmology { omega_m: r.omega_m }, amid, a1).0;
        pm_clocks.extend([kick0, drift, kick1, a0, a1]);
    }
    Ok(PendingLabRun {
        request: r.clone(),
        c,
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
        pm_reference_cache: OnceLock::new(),
        pm_reference_quantized_cache: OnceLock::new(),
        pm_reference_displacement_quantized_cache: OnceLock::new(),
    })
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
    pub fn metadata(&self) -> Value {
        json!({"dim":self.request.dim,"n":self.request.n,"forceMesh":self.request.force_mesh,
            "box":1.0,"omegaM":self.request.omega_m,"aStart":self.a_start,"aEnd":self.a,
            "steps":self.request.steps,"modeCount":(self.request.n/4).min(6),
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
            })
            .as_ref()
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
            })
            .as_ref()
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
            })
            .as_ref()
            .map_err(Clone::clone)
    }
    fn evolve_pm(&self, initial: PmState, tangent: PmTangent) -> Result<PmRun, String> {
        let force_grid = Grid::new(self.request.dim, self.request.force_mesh, 1.0);
        let sheet = Grid::new(self.request.dim, self.request.n, 1.0);
        evolve_cosmology(
            &force_grid,
            &sheet,
            initial,
            Some(tangent),
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
        let pm = self.pm_reference_run()?.clone();
        assemble_prepared(self, pm, "f64")
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
        assemble_prepared(self, pm, "f32")
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

fn assemble_prepared(
    pending: &PendingLabRun,
    pm: PmRun,
    pm_arithmetic: &str,
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
    let rho = density(g, &x, r.mesh);
    let pmrho = density(g, pmx, r.mesh);
    let eg = Grid::new(r.dim, r.mesh, 1.0);
    let rs = observables::smooth_density(&eg, &rho, r.smoothing);
    let pms = observables::smooth_density(&eg, &pmrho, r.smoothing);
    let (logrho, vol, mass, pdfmean, var) = observables::density_pdf(&rs, &rho, 80);
    let (_, pmvol, pmmass, _, pmvar) = observables::density_pdf(&pms, &pmrho, 80);
    let probabilities = [0.001, 0.01, 0.1, 0.5, 0.9, 0.99, 0.999];
    let summary = observables::density_summary(&rs, &rho, &probabilities);
    let pm_summary = observables::density_summary(&pms, &pmrho, &probabilities);
    let cic = sheet::cic_density(
        r.dim,
        &x.iter().map(|&v| v as f32).collect::<Vec<_>>(),
        g.size,
        1.0,
        r.mesh,
    );
    let ps = power(&eg, &rho);
    let psc = power(&eg, &cic);
    let psp = power(&eg, &pmrho);
    let mut ks = Vec::new();
    let mut powers = Vec::new();
    let mut derivatives = Vec::new();
    let mut fd = Vec::new();
    let mut errs = Vec::new();
    let mut pmpowers = Vec::new();
    let mut pmderivs = Vec::new();
    let mut mode_records = Vec::new();
    for m in 1..=(r.n / 4).min(6) {
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
    let pmdpos = pm.tangent.as_ref().unwrap().dx.clone();
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
    let result = json!({"dim":r.dim,"n":r.n,"mesh":r.mesh,"forceMesh":r.force_mesh,"box":1.,"boxSize":r.box_size,"a":a,"aStart":a_start,"D":d,"dDa":d_da,"redshift":1./a-1.,"growthAtOne":growth.d_of_a(1.),"shellCrossing":sc,"aShellCrossing":a_sc,
        "request":r,"engineVersion":env!("CARGO_PKG_VERSION"),"schemaVersion":2,"estimatedPeakBytes":r.estimated_peak_bytes(),
        "crossingReference":"Minimum ZA growth threshold and reachable sampled-vertex 2--4LPT crossing estimates; not a guarantee for the affine mesh or PM trajectory",
        "cosmology":{"omegaM":r.omega_m,"omegaB":r.omega_b,"h":r.h,"ns":r.ns,"sigma8":r.sigma8},
        "guard":{"ok":safe,"crossed":!safe,"pmCrossed":pm.crossed,"firstPmCrossing":first_pm_crossing,"minimumJacobian":minj,"minimumSimplexJacobian":mins.min(pm.min_jacobian),"pmMinimumSimplexJacobian":pm.min_jacobian,"pmInitialMinimumJacobianF64":pending.pm_initial_min_jacobian,"pmInitialMinimumJacobianF32":pending.pm_initial_quantized_min_jacobian,"pmInitialMinimumJacobianDisplacementF32":pending.pm_initial_displacement_quantized_min_jacobian,"pmInitialMinimumJacobianUsed":pm_initial_minimum_used,"pmInitialMinimumJacobianRepresentation":pm_initial_minimum_representation,"lptFinalInvertedMass":observables::inverted_mass_fraction(g,&x),"pmFinalInvertedMass":observables::inverted_mass_fraction(g,pmx),"reason":reason},
        "q":q,"x":[],"v":v,"pmX":[],"pmV":pmv,"rho":rs,"pmRho":pms,
        "spectra":{"k":physical_k(&ps.k),"sheet":physical_p(&ps.p),"cic":physical_p(&psc.p),"pm":physical_p(&psp.p),"counts":ps.nmodes,"directK":physical_k(&ks),"direct":physical_p(&powers),"directPM":physical_p(&pmpowers)},
        "pdf":{"logrho":logrho,"volume":vol,"mass":mass,"pmVolume":pmvol,"pmMass":pmmass,"mean":pdfmean,"variance":var,"pmVariance":pmvar,"summary":summary,"pmSummary":pm_summary,"capturedVolume":vol.iter().sum::<f64>()*dy,"capturedMass":mass.iter().sum::<f64>()*dy,"pmCapturedVolume":pmvol.iter().sum::<f64>()*dy,"pmCapturedMass":pmmass.iter().sum::<f64>()*dy},
        "gradient":{"k":physical_k(&ks),"power":physical_p(&powers),"tangent":physical_p(&derivatives),"finiteDifference":physical_p(&fd),"relativeError":errs,"pmTangent":physical_p(&pmderivs),"parameter":"initial amplitude A at A=1, fixed epoch","finiteDifferenceStep":h},
        "diagnostics":{"mass":mean(&rho),"pmMass":mean(&pmrho),"positionRms":position_rms,"positionRelativeRms":position_rms/displacement_rms.max(1e-30),"powerRelativeError":power_error},
        "conventions":{"liveCosmology":if c.cosmo.is_eds(){"EdS"}else{"flat LCDM"},"initialBandCutoff":r.initial_cutoff,
        "modelGeometry":if r.dim==3 {"3D periodic matter field"} else {"Translationally invariant subspace of 3D cosmological equations; not a random 3D slice"},
        "initialSpectrum":"EH98 smooth baryon-suppressed transfer; no BAO, radiation in late-time growth, or massive neutrinos",
        "initialNormalization":"Gaussian spectrum sigma8 at z=0 before Gaussian filtering and radial cutoff; no realization RMS rescaling. Deterministic waves use fixed delta0 amplitudes independent of sigma8.",
        "initialFilter":"Gaussian exp(-k^2 R_initial^2/2) plus fixed radial integer-mode cutoff",
        "modeGenerator":"PSC Fourier mode hash v2, SplitMix64 and xoshiro256** Box-Muller, exact Hermitian pairing",
        "positionUnits":"x/L","velocityUnits":"d(x/L)/da","wavenumberUnits":"h/Mpc","powerUnits":format!("(Mpc/h)^{}",r.dim),"observationFilter":"Gaussian","powerField":"unsmoothed cell averages","pdfMassWeights":"unsmoothed cell mass","pmObservation":"exact P0 sheet attached to PM vertices","pmArithmetic":pm_arithmetic,"crossingDiagnostic":crossing_diagnostic}});
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
        x: x.clone(),
        dpos: dpos.clone(),
        xplus: xplus.clone(),
        xminus: xminus.clone(),
        pmx: pmx.clone(),
        pmdpos,
        fd_channels: OnceLock::new(),
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
    pub fn lpt_displacement(&self) -> &[f32] {
        &self.lpt_displacement
    }
    pub fn lpt_direction(&self) -> &[f32] {
        &self.lpt_direction
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
        let mut out = Vec::with_capacity(self.modes.len() * 8);
        for mode in &self.modes {
            let k = [grid.kf() * mode[0] as f64, 0.0, 0.0];
            let (f, df) = observables::sheet_mode_jvp(&grid, x, dx, k);
            let (pmf, pmdf) = observables::sheet_mode_jvp(&grid, pmx, pmdx, k);
            out.extend([f.re, f.im, df.re, df.im, pmf.re, pmf.im, pmdf.re, pmdf.im]);
        }
        out
    }

    /// Finite-difference power derivatives from the retained f64 positions.
    /// Values are raw-box power derivatives, cached so a GPU fallback can reuse them.
    pub fn finite_difference_cpu(&self) -> &[f64] {
        self.fd_channels.get_or_init(|| {
            let grid = Grid::new(self.request.dim, self.request.n, 1.0);
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

    pub fn metadata(&self) -> Value {
        json!({"dim":self.request.dim,"n":self.request.n,"modeCount":self.modes.len(),
            "modes":self.modes.iter().map(|m| [m[0],m[1],m[2]]).collect::<Vec<_>>(),"h":self.h,
            "channelScalarsPerMode":8})
    }

    pub fn finish(&self, channels: &[f64], execution: &str) -> Result<Value, String> {
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
        let execution: Value = serde_json::from_str(execution)
            .map_err(|e| format!("Invalid execution metadata JSON: {e}"))?;
        let volume = self.request.box_size.powi(self.request.dim as i32);
        let mut powers = Vec::with_capacity(self.modes.len());
        let mut derivatives = Vec::with_capacity(self.modes.len());
        let mut pmpowers = Vec::with_capacity(self.modes.len());
        let mut pmderivs = Vec::with_capacity(self.modes.len());
        let mut errors = Vec::with_capacity(self.modes.len());
        let fd_cpu = self.finite_difference_cpu();
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
            let fdv = fd_cpu[i];
            fd_values.push(fdv * volume);
            powers.push(p * volume);
            derivatives.push(dp * volume);
            pmpowers.push(pmp * volume);
            pmderivs.push(pmdp * volume);
            errors.push((dp - fdv).abs() / dp.abs().max(1e-9));
        }
        let mut out = self.result.clone();
        out["x"] = json!(self.x);
        out["pmX"] = json!(self.pmx);
        out["spectra"]["direct"] = json!(powers);
        out["spectra"]["directPM"] = json!(pmpowers);
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
    prepared.finish(&channels, &json!({"backend":"cpu"}).to_string())
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
