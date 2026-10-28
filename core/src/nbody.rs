//! Periodic particle-mesh dynamics and its analytic directional derivative.
//!
//! Flat ΛCDM, H0=1: `a` is time and `p=a² dx/dt=a³ E(a) dx/da`.
//! We solve ∇²φ=3Ω_m δ/(2a) with an undeconvolved spectral Green function
//! and derivative-safe Fourier gradients. Matched node-centered CIC deposition
//! and gathering define the finite-resolution force. KDK freezes the scaled
//! force -3Ω_m∇∇⁻²δ/2 in each kick. Its kick and drift clocks are integrated
//! accurately in ln(a), with closed forms in EdS.
//! Positions remain unwrapped. Tangents differentiate the actual discrete
//! solver, including both CIC operations, away from CIC cell boundaries.
//! Crossing may either terminate a run or be recorded while the PM particles
//! continue; continuing does not make finite-order LPT a post-crossing solution.

use crate::{fft::FftEngine, grid::Grid, growth::Cosmology};
use num_complex::Complex64 as C64;

#[derive(Clone, Debug)]
pub struct PmState {
    pub x: Vec<f64>,
    pub p: Vec<f64>,
    pub a: f64,
}
#[derive(Clone, Debug)]
pub struct PmTangent {
    pub dx: Vec<f64>,
    pub dp: Vec<f64>,
}
/// Bracket of the first discrete drift that loses positive sheet orientation.
/// A step of zero indicates the input sheet was already crossed. This records
/// a diagnostic bracket, not an interpolated physical shell-crossing time.
#[derive(Clone, Debug)]
pub struct PmCrossing {
    pub step: usize,
    pub a_start: f64,
    pub a_end: f64,
    pub minimum_jacobian: f64,
}
#[derive(Clone, Debug)]
pub struct PmRun {
    pub state: PmState,
    pub tangent: Option<PmTangent>,
    pub min_jacobian: f64,
    pub completed_steps: usize,
    pub crossed: bool,
    pub first_crossing: Option<PmCrossing>,
}

/// TSC (piecewise-quadratic B-spline M_3) stencil and directional weight derivatives,
/// periodic at mesh nodes. C^1 continuous with position-independent second moment.
fn stencil(grid: &Grid, x: &[f64], dx: Option<&[f64]>) -> ([(usize, f64, f64); 27], usize) {
    let mut base = [0usize; 3];
    let mut w1d = [[0.0; 3]; 3];
    let mut dw1d = [[0.0; 3]; 3];
    let inv_dx = 1.0 / grid.dx();
    let nf = grid.n as f64;
    for a in 0..grid.dim {
        let u = (x[a] * inv_dx + 0.5).rem_euclid(nf);
        let fl = u.floor();
        base[a] = (fl as usize) % grid.n;
        let d = (u - fl) - 0.5;
        let m = 0.5 - d;
        let p = 0.5 + d;
        w1d[a] = [0.5 * m * m, 0.75 - d * d, 0.5 * p * p];
        if let Some(v) = dx {
            let s = v[a] * inv_dx;
            dw1d[a] = [(d - 0.5) * s, -2.0 * d * s, (d + 0.5) * s];
        }
    }
    let count = 3usize.pow(grid.dim as u32);
    let mut out = [(0usize, 0.0, 0.0); 27];
    let n = grid.n;
    for idx in 0..count {
        let c0 = idx % 3;
        let c1 = (idx / 3) % 3;
        let c2 = idx / 9;
        let cs = [c0, c1, c2];
        let mut ijk = [0usize; 3];
        let mut factors = [1.0; 3];
        for a in 0..grid.dim {
            ijk[a] = (base[a] + n + cs[a] - 1) % n;
            factors[a] = w1d[a][cs[a]];
        }
        let w = match grid.dim {
            1 => factors[0],
            2 => factors[0] * factors[1],
            _ => factors[0] * factors[1] * factors[2],
        };
        let mut dw = 0.0;
        if dx.is_some() {
            for a in 0..grid.dim {
                let mut term = dw1d[a][cs[a]];
                for b in 0..grid.dim {
                    if b != a {
                        term *= factors[b];
                    }
                }
                dw += term;
            }
        }
        out[idx] = (grid.ravel(ijk), w, dw);
    }
    (out, count)
}

/// Equal-mass particle density ρ/ρ̄ and its JVP. Density has mean one, JVP mean zero.
/// `x` and `dx` are interleaved, and particle number need not equal mesh size.
pub fn cic_density_jvp(mesh: &Grid, x: &[f64], dx: Option<&[f64]>) -> (Vec<f64>, Vec<f64>) {
    assert!(!x.is_empty() && x.len() % mesh.dim == 0);
    if let Some(dx) = dx {
        assert_eq!(dx.len(), x.len());
    }
    let np = x.len() / mesh.dim;
    let norm = mesh.size as f64 / np as f64;
    let mut rho = vec![0.0; mesh.size];
    let mut drho = if dx.is_some() {
        vec![0.0; mesh.size]
    } else {
        Vec::new()
    };
    for i in 0..np {
        let j = i * mesh.dim;
        let (st, count) = stencil(mesh, &x[j..j + mesh.dim], dx.map(|v| &v[j..j + mesh.dim]));
        if dx.is_some() {
            for &(node, w, dw) in &st[..count] {
                rho[node] += norm * w;
                drho[node] += norm * dw;
            }
        } else {
            for &(node, w, _) in &st[..count] {
                rho[node] += norm * w;
            }
        }
    }
    (rho, drho)
}

fn sheet_side(dim: usize, np: usize) -> usize {
    match dim {
        1 => np,
        2 => (np as f64).sqrt().round() as usize,
        _ => (np as f64).cbrt().round() as usize,
    }
}

fn inv_tsc_window_sq(mesh: &Grid, i: usize, sheet_n: usize) -> f64 {
    let ijk = mesh.unravel(i);
    let inv_n = std::f64::consts::PI / mesh.n as f64;
    let mut w = 1.0;
    for d in 0..mesh.dim {
        let m = mesh.ifreq(ijk[d]);
        if mesh.n > sheet_n && (m.unsigned_abs() as usize) * 2 >= sheet_n {
            return 0.0;
        }
        if m != 0 {
            let theta = (m as f64) * inv_n;
            let sinc = theta.sin() / theta;
            let s2 = sinc * sinc;
            w *= s2 * s2 * s2;
        }
    }
    1.0 / w
}

fn mesh_force(
    mesh: &Grid,
    eng: &mut FftEngine,
    rho: &[f64],
    omega_m: f64,
    sheet_n: usize,
) -> Vec<Vec<f64>> {
    let hat = eng.forward_real(rho);
    (0..mesh.dim)
        .map(|a| {
            let fh = hat
                .iter()
                .enumerate()
                .map(|(i, z)| {
                    let k2 = mesh.k2(i);
                    if k2 == 0.0 {
                        C64::new(0.0, 0.0)
                    } else {
                        let deconv = inv_tsc_window_sq(mesh, i, sheet_n);
                        C64::new(0.0, 1.5 * omega_m * deconv * mesh.kvec(i)[a] / k2) * z
                    }
                })
                .collect();
            eng.inverse_to_real(fh)
        })
        .collect()
}

/// EdS scaled acceleration F=-3∇∇⁻²δ/2 and its complete discrete JVP.
pub fn force_jvp(mesh: &Grid, x: &[f64], dx: Option<&[f64]>) -> (Vec<f64>, Vec<f64>) {
    force_cosmology_jvp(mesh, x, dx, Cosmology::eds())
}

/// Flat ΛCDM scaled acceleration F=-3Ω_m∇∇⁻²δ/2 and its complete JVP.
/// The matter parameter is the present-day Ω_m, held fixed in the derivative.
pub fn force_cosmology_jvp(
    mesh: &Grid,
    x: &[f64],
    dx: Option<&[f64]>,
    cosmo: Cosmology,
) -> (Vec<f64>, Vec<f64>) {
    assert!(cosmo.omega_m.is_finite() && cosmo.omega_m > 0.0 && cosmo.omega_m <= 1.0);
    let mut eng = FftEngine::new(mesh);
    force_jvp_with_engine(mesh, &mut eng, x, dx, cosmo.omega_m)
}

fn force_jvp_with_engine(
    mesh: &Grid,
    eng: &mut FftEngine,
    x: &[f64],
    dx: Option<&[f64]>,
    omega_m: f64,
) -> (Vec<f64>, Vec<f64>) {
    let np = x.len() / mesh.dim;
    let sheet_n = sheet_side(mesh.dim, np);
    let (rho, drho) = cic_density_jvp(mesh, x, dx);
    let f = mesh_force(mesh, eng, &rho, omega_m, sheet_n);
    let mut out = vec![0.0; x.len()];
    if dx.is_some() {
        let df = mesh_force(mesh, eng, &drho, omega_m, sheet_n);
        let mut dout = vec![0.0; x.len()];
        for i in 0..x.len() / mesh.dim {
            let j = i * mesh.dim;
            let (st, count) = stencil(mesh, &x[j..j + mesh.dim], dx.map(|v| &v[j..j + mesh.dim]));
            for &(node, w, dw) in &st[..count] {
                for a in 0..mesh.dim {
                    out[j + a] += w * f[a][node];
                    dout[j + a] += w * df[a][node] + dw * f[a][node];
                }
            }
        }
        (out, dout)
    } else {
        for i in 0..x.len() / mesh.dim {
            let j = i * mesh.dim;
            let (st, count) = stencil(mesh, &x[j..j + mesh.dim], None);
            for &(node, w, _) in &st[..count] {
                for a in 0..mesh.dim {
                    out[j + a] += w * f[a][node];
                }
            }
        }
        (out, Vec::new())
    }
}

fn min_jacobian(sheet: &Grid, x: &[f64]) -> f64 {
    if sheet.dim == 1 {
        (0..sheet.n)
            .map(|i| {
                let next = if i + 1 == sheet.n {
                    x[0] + sheet.l
                } else {
                    x[i + 1]
                };
                (next - x[i]) / sheet.dx()
            })
            .fold(f64::INFINITY, f64::min)
    } else {
        crate::observables::minimum_simplex_jacobian(sheet, x)
    }
}

// The determinant along a drift is a polynomial of degree <= dim. Check its
// interior stationary points too: positive endpoints alone can miss a fold.
pub fn minimum_drift_jacobian(sheet: &Grid, start: &[f64], end: &[f64]) -> f64 {
    fn determinant(e: &[[f64; 3]; 3], d: usize) -> f64 {
        match d {
            1 => e[0][0],
            2 => e[0][0] * e[1][1] - e[0][1] * e[1][0],
            _ => {
                e[0][0] * (e[1][1] * e[2][2] - e[1][2] * e[2][1])
                    - e[0][1] * (e[1][0] * e[2][2] - e[1][2] * e[2][0])
                    + e[0][2] * (e[1][0] * e[2][1] - e[1][1] * e[2][0])
            }
        }
    }
    let d = sheet.dim;
    let mut minimum = f64::INFINITY;
    crate::observables::for_each_simplex(sheet, |verts, _| {
        let mut e = [[0.0; 3]; 3];
        let mut de = e;
        let mut q = e;
        let (origin, lift0) = verts[0];
        let ijk0 = sheet.unravel(origin);
        for j in 0..d {
            let (id, lift) = verts[j + 1];
            let ijk = sheet.unravel(id);
            for a in 0..d {
                e[j][a] = start[id * d + a] + lift[a] - start[origin * d + a] - lift0[a];
                de[j][a] = end[id * d + a] - start[id * d + a] - end[origin * d + a]
                    + start[origin * d + a];
                q[j][a] = (ijk[a] as f64 - ijk0[a] as f64) * sheet.dx() + lift[a] - lift0[a];
            }
        }
        let denominator = determinant(&q, d);
        let mut coeff = [0.0; 4];
        for bits in 0..(1usize << d) {
            let mut edges = e;
            for j in 0..d {
                if bits & (1 << j) != 0 {
                    edges[j] = de[j];
                }
            }
            coeff[bits.count_ones() as usize] += determinant(&edges, d) / denominator;
        }
        let value = |t: f64| coeff[0] + t * (coeff[1] + t * (coeff[2] + t * coeff[3]));
        minimum = minimum.min(value(0.0)).min(value(1.0));
        let (aa, bb, cc) = (3.0 * coeff[3], 2.0 * coeff[2], coeff[1]);
        if aa.abs() <= 1e-14 * bb.abs().max(cc.abs()).max(1e-300) {
            if bb != 0.0 {
                let t = -cc / bb;
                if t > 0.0 && t < 1.0 {
                    minimum = minimum.min(value(t));
                }
            }
        } else {
            let discriminant = bb * bb - 4.0 * aa * cc;
            if discriminant >= 0.0 {
                for t in [
                    (-bb - discriminant.sqrt()) / (2.0 * aa),
                    (-bb + discriminant.sqrt()) / (2.0 * aa),
                ] {
                    if t > 0.0 && t < 1.0 {
                        minimum = minimum.min(value(t));
                    }
                }
            }
        }
    });
    minimum
}

/// KDK clocks `(kick, drift)` for a flat ΛCDM interval:
/// kick=∫ da/(a²E(a)), drift=∫ da/(a³E(a)). The EdS forms are evaluated
/// analytically. Otherwise composite eight-point Gauss-Legendre quadrature
/// uses ln(a) segments of width at most 0.5, avoiding early-time singularities.
pub fn cosmology_clocks(cosmo: Cosmology, a0: f64, a1: f64) -> (f64, f64) {
    assert!(a0.is_finite() && a1.is_finite() && a0 > 0.0 && a1 >= a0);
    assert!(cosmo.omega_m.is_finite() && cosmo.omega_m > 0.0 && cosmo.omega_m <= 1.0);
    if a0 == a1 {
        return (0.0, 0.0);
    }
    if cosmo.omega_m == 1.0 {
        return (
            2.0 * (a1.sqrt() - a0.sqrt()),
            2.0 * (1.0 / a0.sqrt() - 1.0 / a1.sqrt()),
        );
    }
    const NODES: [f64; 4] = [
        0.1834346424956498,
        0.5255324099163290,
        0.7966664774136267,
        0.9602898564975363,
    ];
    const WEIGHTS: [f64; 4] = [
        0.3626837833783620,
        0.3137066458778873,
        0.2223810344533745,
        0.1012285362903763,
    ];
    let u0 = a0.ln();
    let u1 = a1.ln();
    let segments = ((u1 - u0) / 0.5).ceil().max(1.0) as usize;
    let width = (u1 - u0) / segments as f64;
    let mut kick = 0.0;
    let mut drift = 0.0;
    for segment in 0..segments {
        let mid = u0 + width * (segment as f64 + 0.5);
        for j in 0..4 {
            for sign in [-1.0, 1.0] {
                let u = mid + sign * width * 0.5 * NODES[j];
                // Equivalent to exp(-u)/E and exp(-2u)/E, rearranged
                // to avoid forming Ω_m/a³ at very early times.
                let (ki, di) = if u <= 0.0 {
                    let denominator =
                        (cosmo.omega_m + (1.0 - cosmo.omega_m) * (3.0 * u).exp()).sqrt();
                    (
                        (0.5 * u).exp() / denominator,
                        (-0.5 * u).exp() / denominator,
                    )
                } else {
                    let denominator =
                        (cosmo.omega_m * (-3.0 * u).exp() + 1.0 - cosmo.omega_m).sqrt();
                    ((-u).exp() / denominator, (-2.0 * u).exp() / denominator)
                };
                kick += width * 0.5 * WEIGHTS[j] * ki;
                drift += width * 0.5 * WEIGHTS[j] * di;
            }
        }
    }
    (kick, drift)
}

/// Strict EdS compatibility wrapper. Rejects initially crossed sheets and any
/// linear drift whose simplex determinant becomes nonpositive.
pub fn evolve_eds(
    mesh: &Grid,
    sheet: &Grid,
    state: PmState,
    tangent: Option<PmTangent>,
    a_end: f64,
    steps: usize,
) -> Result<PmRun, String> {
    evolve_cosmology(
        mesh,
        sheet,
        state,
        tangent,
        a_end,
        steps,
        Cosmology::eds(),
        false,
    )
}

/// Integrate a supplied canonical state with equal scale-factor KDK steps.
/// `sheet` gives Lagrangian connectivity and `mesh` sets force resolution.
/// `allow_crossing=true` continues the PM dynamics after simplex orientation
/// loss, recording the first crossing bracket and the trajectory minimum.
/// Otherwise crossing is rejected, as in `evolve_eds`. Each drift determinant
/// is checked at its endpoints and polynomial extrema, once per step.
/// The optional JVP holds cosmology, masses, times, mesh and step count fixed;
/// it does not differentiate the crossing diagnostic or its policy.
pub fn evolve_cosmology(
    mesh: &Grid,
    sheet: &Grid,
    mut state: PmState,
    mut tangent: Option<PmTangent>,
    a_end: f64,
    steps: usize,
    cosmo: Cosmology,
    allow_crossing: bool,
) -> Result<PmRun, String> {
    if mesh.dim != sheet.dim
        || state.x.len() != sheet.size * sheet.dim
        || state.p.len() != state.x.len()
    {
        return Err("PM dimensions or particle arrays do not match sheet".into());
    }
    if mesh.n < 2
        || !mesh.l.is_finite()
        || mesh.l <= 0.0
        || mesh.l != sheet.l
        || !state.a.is_finite()
        || state.a <= 0.0
        || !a_end.is_finite()
        || a_end < state.a
        || steps == 0
    {
        return Err("PM requires positive scale factors, matching boxes and steps > 0".into());
    }
    if !cosmo.omega_m.is_finite() || cosmo.omega_m <= 0.0 || cosmo.omega_m > 1.0 {
        return Err("PM flat cosmology requires 0 < Omega_m <= 1".into());
    }
    if let Some(t) = &tangent {
        if t.dx.len() != state.x.len()
            || t.dp.len() != state.x.len()
            || t.dx.iter().chain(&t.dp).any(|v| !v.is_finite())
        {
            return Err("PM tangent arrays are invalid".into());
        }
    }
    if state.x.iter().chain(&state.p).any(|v| !v.is_finite()) {
        return Err("PM state must be finite".into());
    }
    let mut minimum = min_jacobian(sheet, &state.x);
    if !minimum.is_finite() {
        return Err("PM initial Jacobian is nonfinite".into());
    }
    let mut first_crossing = if minimum <= 0.0 {
        if !allow_crossing {
            return Err("PM initial sheet has crossed".into());
        }
        Some(PmCrossing {
            step: 0,
            a_start: state.a,
            a_end: state.a,
            minimum_jacobian: minimum,
        })
    } else {
        None
    };
    let a_start = state.a;
    // The final kick leaves x and dx unchanged, so its force and force JVP
    // also serve the next step's initial kick. Keep one FFT engine for the run.
    let mut eng = FftEngine::new(mesh);
    let (mut f, mut df) = force_jvp_with_engine(
        mesh,
        &mut eng,
        &state.x,
        tangent.as_ref().map(|t| t.dx.as_slice()),
        cosmo.omega_m,
    );
    let u_start = a_start.sqrt();
    let u_end = a_end.sqrt();
    for s in 0..steps {
        let u0 = u_start + (u_end - u_start) * s as f64 / steps as f64;
        let u1 = u_start + (u_end - u_start) * (s + 1) as f64 / steps as f64;
        let umid = 0.5 * (u0 + u1);
        let a0 = if s == 0 { a_start } else { u0 * u0 };
        let a1 = if s + 1 == steps { a_end } else { u1 * u1 };
        let amid = umid * umid;
        let k0 = cosmology_clocks(cosmo, a0, amid).0;
        let k1 = cosmology_clocks(cosmo, amid, a1).0;
        let drift = cosmology_clocks(cosmo, a0, a1).1;
        if !k0.is_finite() || !k1.is_finite() || !drift.is_finite() {
            return Err("PM clock overflowed".into());
        }
        let check_drift = !allow_crossing || sheet.size <= 16_384 || s == 0 || s + 1 == steps;
        let before_drift = if check_drift {
            state.x.clone()
        } else {
            Vec::new()
        };
        for j in 0..state.x.len() {
            state.p[j] += k0 * f[j];
            state.x[j] += drift * state.p[j];
            if let Some(t) = &mut tangent {
                t.dp[j] += k0 * df[j];
                t.dx[j] += drift * t.dp[j];
            }
        }
        if state.x.iter().chain(&state.p).any(|v| !v.is_finite())
            || tangent
                .as_ref()
                .is_some_and(|t| t.dx.iter().chain(&t.dp).any(|v| !v.is_finite()))
        {
            return Err("PM state or tangent overflowed during drift".into());
        }
        if check_drift {
            let jac = minimum_drift_jacobian(sheet, &before_drift, &state.x);
            minimum = minimum.min(jac);
            if !jac.is_finite() {
                return Err("PM drift Jacobian is nonfinite".into());
            }
            if jac <= 0.0 {
                if !allow_crossing {
                    return Err(format!("PM sheet crossed during drift ending at a={a1:.6}; increase start safety or reduce target"));
                }
                if first_crossing.is_none() {
                    first_crossing = Some(PmCrossing {
                        step: s + 1,
                        a_start: a0,
                        a_end: a1,
                        minimum_jacobian: jac,
                    });
                }
            }
        }
        (f, df) = force_jvp_with_engine(
            mesh,
            &mut eng,
            &state.x,
            tangent.as_ref().map(|t| t.dx.as_slice()),
            cosmo.omega_m,
        );
        for j in 0..state.x.len() {
            state.p[j] += k1 * f[j];
            if let Some(t) = &mut tangent {
                t.dp[j] += k1 * df[j];
            }
        }
        if state.p.iter().any(|v| !v.is_finite())
            || tangent
                .as_ref()
                .is_some_and(|t| t.dp.iter().any(|v| !v.is_finite()))
        {
            return Err("PM momentum or tangent overflowed during kick".into());
        }
        state.a = a1;
    }
    Ok(PmRun {
        state,
        tangent,
        min_jacobian: minimum,
        completed_steps: steps,
        crossed: first_crossing.is_some(),
        first_crossing,
    })
}
