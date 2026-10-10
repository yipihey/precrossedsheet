//! Periodic particle-mesh dynamics and its analytic directional derivative.
//!
//! EdS, H0=1: `a` is time, `p=a² dx/dt=a^(3/2) dx/da`.
//! We solve ∇²φ=3δ/(2a) with an undeconvolved spectral Green function and
//! derivative-safe Fourier gradients. Matched node-centered CIC deposition
//! and gathering define the finite-resolution force. KDK freezes the scaled
//! force -3∇∇⁻²δ/2 in each kick and integrates its EdS time coefficient exactly.
//! Positions remain unwrapped. Tangents differentiate the actual discrete
//! solver, including both CIC operations, away from CIC cell boundaries.

use crate::{fft::FftEngine, grid::Grid};
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
#[derive(Clone, Debug)]
pub struct PmRun {
    pub state: PmState,
    pub tangent: Option<PmTangent>,
    pub min_jacobian: f64,
    pub completed_steps: usize,
}

/// CIC stencil and directional weight derivatives, periodic at mesh nodes.
fn stencil(grid: &Grid, x: &[f64], dx: Option<&[f64]>) -> Vec<(usize, f64, f64)> {
    let mut base = [0usize; 3];
    let mut f = [0.0; 3];
    for a in 0..grid.dim {
        let u = x[a].rem_euclid(grid.l) / grid.dx();
        base[a] = (u.floor() as usize) % grid.n;
        f[a] = u - u.floor();
    }
    (0..1usize << grid.dim)
        .map(|bits| {
            let mut ijk = base;
            let mut factors = [1.0; 3];
            for a in 0..grid.dim {
                if bits & (1 << a) != 0 {
                    ijk[a] = (base[a] + 1) % grid.n;
                    factors[a] = f[a];
                } else {
                    factors[a] = 1.0 - f[a];
                }
            }
            let w = factors[..grid.dim].iter().product();
            let mut dw = 0.0;
            if let Some(v) = dx {
                for a in 0..grid.dim {
                    let sign = if bits & (1 << a) != 0 { 1.0 } else { -1.0 };
                    let mut term = sign * v[a] / grid.dx();
                    for b in 0..grid.dim {
                        if b != a {
                            term *= factors[b];
                        }
                    }
                    dw += term;
                }
            }
            (grid.ravel(ijk), w, dw)
        })
        .collect()
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
    let mut drho = vec![0.0; mesh.size];
    for i in 0..np {
        let j = i * mesh.dim;
        for (node, w, dw) in stencil(mesh, &x[j..j + mesh.dim], dx.map(|v| &v[j..j + mesh.dim])) {
            rho[node] += norm * w;
            drho[node] += norm * dw;
        }
    }
    (rho, drho)
}

fn mesh_force(mesh: &Grid, eng: &mut FftEngine, rho: &[f64]) -> Vec<Vec<f64>> {
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
                        C64::new(0.0, 1.5 * mesh.kvec(i)[a] / k2) * z
                    }
                })
                .collect();
            eng.inverse_to_real(fh)
        })
        .collect()
}

/// Scaled acceleration F=-3∇∇⁻²δ/2 at particles and its complete discrete JVP.
pub fn force_jvp(mesh: &Grid, x: &[f64], dx: Option<&[f64]>) -> (Vec<f64>, Vec<f64>) {
    let (rho, drho) = cic_density_jvp(mesh, x, dx);
    let mut eng = FftEngine::new(mesh);
    let f = mesh_force(mesh, &mut eng, &rho);
    let df = if dx.is_some() {
        mesh_force(mesh, &mut eng, &drho)
    } else {
        vec![vec![0.0; mesh.size]; mesh.dim]
    };
    let mut out = vec![0.0; x.len()];
    let mut dout = vec![0.0; x.len()];
    for i in 0..x.len() / mesh.dim {
        let j = i * mesh.dim;
        for (node, w, dw) in stencil(mesh, &x[j..j + mesh.dim], dx.map(|v| &v[j..j + mesh.dim])) {
            for a in 0..mesh.dim {
                out[j + a] += w * f[a][node];
                dout[j + a] += w * df[a][node] + dw * f[a][node];
            }
        }
    }
    (out, dout)
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
fn minimum_drift_jacobian(sheet: &Grid, start: &[f64], end: &[f64]) -> f64 {
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

/// Integrate a supplied state to `a_end` with equal scale-factor KDK steps.
/// `sheet` identifies particle Lagrangian connectivity; `mesh` sets force resolution.
/// Rejects an initially crossed sheet or any linear KDK drift whose simplex
/// signed Jacobian becomes nonpositive, checking the determinant polynomial at
/// endpoints and interior extrema. This guard applies to the discrete trajectory;
/// physical accuracy still requires mesh and step-convergence checks.
/// Gradients hold cosmology, times, particle masses, mesh and step count fixed.
pub fn evolve_eds(
    mesh: &Grid,
    sheet: &Grid,
    mut state: PmState,
    mut tangent: Option<PmTangent>,
    a_end: f64,
    steps: usize,
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
    if minimum <= 0.0 || !minimum.is_finite() {
        return Err("PM initial sheet has crossed".into());
    }
    let a_start = state.a;
    for s in 0..steps {
        let a0 = a_start + (a_end - a_start) * s as f64 / steps as f64;
        let a1 = a_start + (a_end - a_start) * (s + 1) as f64 / steps as f64;
        let amid = 0.5 * (a0 + a1);
        let k0 = 2.0 * (amid.sqrt() - a0.sqrt());
        let k1 = 2.0 * (a1.sqrt() - amid.sqrt());
        let drift = 2.0 * (1.0 / a0.sqrt() - 1.0 / a1.sqrt());
        let (f, df) = force_jvp(mesh, &state.x, tangent.as_ref().map(|t| t.dx.as_slice()));
        let before_drift = state.x.clone();
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
        let jac = minimum_drift_jacobian(sheet, &before_drift, &state.x);
        minimum = minimum.min(jac);
        if jac <= 0.0 || !jac.is_finite() {
            return Err(format!("PM sheet crossed during drift ending at a={a1:.6}; increase start safety or reduce target"));
        }
        let (f, df) = force_jvp(mesh, &state.x, tangent.as_ref().map(|t| t.dx.as_slice()));
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
    })
}
