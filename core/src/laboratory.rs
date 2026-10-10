//! Checked native/Wasm laboratory contract. Live runs are EdS teaching
//! experiments, distinct from the archived LCDM validation data.
use crate::{
    fft::FftEngine,
    grid::Grid,
    ics::{PkShape, Preset},
    nbody::{evolve_eds, PmState, PmTangent},
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
        }
    }
}
impl LabRequest {
    fn validate(&self) -> Result<(), String> {
        if !(1..=3).contains(&self.dim)
            || ![16, 24, 32].contains(&self.n)
            || !(1..=4).contains(&self.order)
        {
            return Err(
                "Dimension, vertex count or LPT order is outside the laboratory range".into(),
            );
        }
        if !self.fraction.is_finite() || !(0.1..=0.85).contains(&self.fraction) {
            return Err("Pre-crossing time fraction must be in [0.1, 0.85]".into());
        }
        if !self.smoothing.is_finite()
            || !(0.0..=0.15).contains(&self.smoothing)
            || ![32, 64, 128].contains(&self.steps)
            || ![32, 48, 64].contains(&self.mesh)
            || ![16, 24, 32, 48, 64].contains(&self.force_mesh)
        {
            return Err("Invalid smoothing, timesteps or mesh resolution".into());
        }
        if self.dim == 3 && (self.n > 24 || self.mesh > 32 || self.force_mesh > 32) {
            return Err("3D laboratory cap: vertices <=24, observation/force mesh <=32".into());
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
fn initial_conditions(r: &LabRequest) -> Cosmo {
    let mut c = Cosmo::new(r.dim, r.n, 1.0);
    c.keep_tensors = Some(true);
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
        c.set_ic_gaussian(PkShape::PowerLaw { n: -1.0 }, 0.08, r.seed, 0.35);
        let cutoff = (r.n - 1) / 8;
        for i in 0..c.grid.size {
            let ijk = c.grid.unravel(i);
            if (0..r.dim).any(|a| c.grid.ifreq(ijk[a]).unsigned_abs() as usize > cutoff) {
                c.delta0_hat[i] = num_complex::Complex64::new(0., 0.);
            }
        }
        let rms = c
            .delta0_hat
            .iter()
            .map(|v| v.norm_sqr())
            .sum::<f64>()
            .sqrt()
            / c.grid.size as f64;
        for z in &mut c.delta0_hat {
            *z *= 0.35 / rms;
        }
        c.sigma0 = 0.35;
    }
    c.build_lpt(4);
    c
}
/// Double-precision evaluation of the inherited spatial terms. The terms
/// themselves retain upstream f32 precision, but time sums and amplitudes do not.
fn evaluate(c: &Cosmo, a: f64, order: usize, amp: f64) -> (Vec<f64>, Vec<f64>) {
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
        let (g, dg) = l.growth.g_and_dg(i, a);
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
/// Execute a checked pre-shell-crossing experiment. Invalid requests return
/// errors before allocating fields; invalid trajectory guards return no plots.
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
    if !sc.is_finite() || sc <= 0. {
        return Err("Cannot identify a finite crossing time for this realization".into());
    }
    let a = r.fraction * sc;
    let a_start = (a * 0.05).min(0.03);
    let (x, v) = evaluate(&c, a, r.order, 1.0);
    let mut minj = f64::INFINITY;
    let mut mins = f64::INFINITY;
    // Resolved time-history check; positivity only at sampled times is not a
    // continuum guarantee. PM additionally checks each full linear drift.
    for t in 0..=16 {
        let at = a * t as f64 / 16.;
        minj = minj.min(c.lpt_ref().min_jacobian(at, r.order));
        let (xt, _) = evaluate(&c, at, r.order, 1.);
        mins = mins.min(observables::minimum_simplex_jacobian(g, &xt));
    }
    if minj <= 0. || mins <= 0. {
        return Ok(
            json!({"guard":{"ok":false,"minimumJacobian":minj,"minimumSimplexJacobian":mins,"reason":"Resolved LPT sheet loses positive orientation"},"a":a,"shellCrossing":sc}),
        );
    }
    // Common growing initial condition uses all four orders, irrespective of
    // the order selected for the LPT comparison at the output.
    let (xi, vi) = evaluate(&c, a_start, 4, 1.0);
    let (dx, dv) = tangent::lpt_amplitude_jvp(c.lpt_ref(), a_start, 4);
    let p: Vec<f64> = vi.iter().map(|v| a_start.powf(1.5) * v).collect();
    let dp = dv.iter().map(|v| a_start.powf(1.5) * v).collect();
    let force_grid = Grid::new(r.dim, r.force_mesh, 1.0);
    let pm = evolve_eds(
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
    );
    let pm = match pm {
        Ok(p) => p,
        Err(e) => {
            return Ok(
                json!({"guard":{"ok":false,"minimumJacobian":minj,"minimumSimplexJacobian":mins,"reason":e},"a":a,"shellCrossing":sc}),
            )
        }
    };
    let pmx = &pm.state.x;
    let pmv: Vec<f64> = pm.state.p.iter().map(|p| p / a.powf(1.5)).collect();
    let rho = density(g, &x, r.mesh);
    let pmrho = density(g, pmx, r.mesh);
    let eg = Grid::new(r.dim, r.mesh, 1.0);
    let rs = observables::smooth_density(&eg, &rho, r.smoothing);
    let pms = observables::smooth_density(&eg, &pmrho, r.smoothing);
    let (logrho, vol, mass, pdfmean, var) = observables::density_pdf(&rs, &rho, 80);
    let (_, pmvol, pmmass, _, pmvar) = observables::density_pdf(&pms, &pmrho, 80);
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
    let (dpos, _) = tangent::lpt_amplitude_jvp(c.lpt_ref(), a, r.order);
    let h = 1e-4;
    let (xplus, _) = evaluate(&c, a, r.order, 1. + h);
    let (xminus, _) = evaluate(&c, a, r.order, 1. - h);
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
    let dy = 3.6 / 80.;
    Ok(
        json!({"dim":r.dim,"n":r.n,"mesh":r.mesh,"forceMesh":r.force_mesh,"box":1.,"a":a,"aStart":a_start,"shellCrossing":sc,
        "guard":{"ok":true,"minimumJacobian":minj,"minimumSimplexJacobian":mins.min(pm.min_jacobian),"pmMinimumSimplexJacobian":pm.min_jacobian,"reason":"Positive resolved orientation at sampled LPT times and throughout each discrete PM drift"},
        "q":q,"x":x,"v":v,"pmX":pmx,"pmV":pmv,"rho":rs,"pmRho":pms,
        "spectra":{"k":ps.k,"sheet":ps.p,"cic":psc.p,"pm":psp.p,"counts":ps.nmodes,"directK":ks,"direct":powers,"directPM":pmpowers},
        "pdf":{"logrho":logrho,"volume":vol,"mass":mass,"pmVolume":pmvol,"pmMass":pmmass,"mean":pdfmean,"variance":var,"pmVariance":pmvar,"capturedVolume":vol.iter().sum::<f64>()*dy,"capturedMass":mass.iter().sum::<f64>()*dy},
        "gradient":{"k":ks,"power":powers,"tangent":derivatives,"finiteDifference":fd,"relativeError":errs,"pmTangent":pmderivs,"parameter":"initial amplitude A at A=1, fixed epoch","finiteDifferenceStep":h},
        "diagnostics":{"mass":mean(&rho),"pmMass":mean(&pmrho),"positionRms":position_rms,"positionRelativeRms":position_rms/displacement_rms.max(1e-30),"powerRelativeError":power_error},
        "conventions":{"liveCosmology":"EdS","initialBandCutoff":(r.n-1)/8,"observationFilter":"Gaussian","powerField":"unsmoothed cell averages","pdfMassWeights":"unsmoothed cell mass","pmObservation":"exact P0 sheet attached to PM vertices"}}),
    )
}
/// Stable JSON facade used by WebAssembly and native integration tests.
pub fn run_json(request: &str) -> String {
    let result = serde_json::from_str::<LabRequest>(request)
        .map_err(|e| format!("Invalid laboratory JSON: {e}"))
        .and_then(|r| run(&r));
    match result {
        Ok(v) => v.to_string(),
        Err(e) => json!({"error":e}).to_string(),
    }
}
