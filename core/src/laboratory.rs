//! Reproducible native/Wasm laboratory contract: flat LCDM or EdS,
//! physical initial spectra, exact observation meshing, and discrete PM tangents.
use crate::{
    cosmological,
    fft::FftEngine,
    grid::Grid,
    growth::{growth_rate, Cosmology},
    ics::Preset,
    nbody::{evolve_cosmology, PmState, PmTangent},
    observables, sheet, spectra, tangent, Cosmo,
};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

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
pub fn run(r: &LabRequest) -> Result<Value, String> {
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
    let force_grid = Grid::new(r.dim, r.force_mesh, 1.0);
    let pm = evolve_cosmology(
        &force_grid,
        g,
        PmState {
            x: xi,
            p,
            a: a_start,
        },
        Some(PmTangent { dx, dp }),
        a,
        r.steps,
        Cosmology { omega_m: r.omega_m },
        true,
    )?;
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
    let (dpos, _) = tangent::lpt_amplitude_jvp(c.lpt_ref(), d, r.order);
    let h = 1e-4;
    let (xplus, _) = evaluate(&c, d, r.order, 1. + h);
    let (xminus, _) = evaluate(&c, d, r.order, 1. - h);
    let zeros = vec![0.; x.len()];
    let mut ks = Vec::new();
    let mut powers = Vec::new();
    let mut derivatives = Vec::new();
    let mut fd = Vec::new();
    let mut errs = Vec::new();
    let mut pmpowers = Vec::new();
    let mut pmderivs = Vec::new();
    for m in 1..=(r.n / 4).min(6) {
        let k = [g.kf() * m as f64, 0., 0.];
        let (mode, dm) = observables::sheet_mode_jvp(g, &x, &dpos, k);
        let vp = observables::sheet_mode_jvp(g, &xplus, &zeros, k)
            .0
            .norm_sqr();
        let vm = observables::sheet_mode_jvp(g, &xminus, &zeros, k)
            .0
            .norm_sqr();
        let deriv = 2. * (mode.conj() * dm).re;
        let diff = (vp - vm) / (2. * h);
        ks.push(k[0]);
        powers.push(mode.norm_sqr());
        derivatives.push(deriv);
        fd.push(diff);
        errs.push((deriv - diff).abs() / deriv.abs().max(1e-9));
        let (pmode, pd) = observables::sheet_mode_jvp(g, pmx, &pm.tangent.as_ref().unwrap().dx, k);
        pmpowers.push(pmode.norm_sqr());
        pmderivs.push(2. * (pmode.conj() * pd).re);
    }
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
    Ok(
        json!({"dim":r.dim,"n":r.n,"mesh":r.mesh,"forceMesh":r.force_mesh,"box":1.,"boxSize":r.box_size,"a":a,"aStart":a_start,"D":d,"dDa":d_da,"redshift":1./a-1.,"growthAtOne":growth.d_of_a(1.),"shellCrossing":sc,"aShellCrossing":a_sc,
        "request":r,"engineVersion":"0.2.0","schemaVersion":2,"estimatedPeakBytes":r.estimated_peak_bytes(),
        "crossingReference":"Minimum ZA growth threshold and reachable sampled-vertex 2--4LPT crossing estimates; not a guarantee for the affine mesh or PM trajectory",
        "cosmology":{"omegaM":r.omega_m,"omegaB":r.omega_b,"h":r.h,"ns":r.ns,"sigma8":r.sigma8},
        "guard":{"ok":safe,"crossed":!safe,"pmCrossed":pm.crossed,"firstPmCrossing":first_pm_crossing,"minimumJacobian":minj,"minimumSimplexJacobian":mins.min(pm.min_jacobian),"pmMinimumSimplexJacobian":pm.min_jacobian,"lptFinalInvertedMass":observables::inverted_mass_fraction(g,&x),"pmFinalInvertedMass":observables::inverted_mass_fraction(g,pmx),"reason":reason},
        "q":q,"x":x,"v":v,"pmX":pmx,"pmV":pmv,"rho":rs,"pmRho":pms,
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
        "positionUnits":"x/L","velocityUnits":"d(x/L)/da","wavenumberUnits":"h/Mpc","powerUnits":format!("(Mpc/h)^{}",r.dim),"observationFilter":"Gaussian","powerField":"unsmoothed cell averages","pdfMassWeights":"unsmoothed cell mass","pmObservation":"exact P0 sheet attached to PM vertices"}}),
    )
}
/// Stable JSON facade used by WebAssembly and native integration tests.
pub fn run_json(request: &str) -> String {
    let result = serde_json::from_str::<Value>(request)
        .map_err(|e| format!("Invalid laboratory JSON: {e}"))
        .and_then(|v| {
            if let Some(version) = v.get("schemaVersion").and_then(Value::as_u64) {
                if version != 2 {return Err(format!("Unsupported experiment schema {version}; use the recorded repository revision"));}
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
