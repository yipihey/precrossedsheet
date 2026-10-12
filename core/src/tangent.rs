//! Analytic JVPs for nLPT and the smooth fixed-mass simplex Fourier observation.
//! IC directions hold cosmology and time fixed; they do not differentiate random
//! generation, RMS normalization, shell-crossing estimates, or density clamps.

use crate::{
    direct::{exp_divided_difference, exp_divided_difference_with_scratch},
    fft::FftEngine,
    grid::Grid,
    growth::{term_specs, Cosmology, Growth, Parents},
    lpt::{mu2, mu3, Lpt},
};
use num_complex::Complex64 as C64;

/// Reusable scratch for repeated fixed-mass simplex transforms. The largest
/// supported simplex has four vertices and one repeated knot per derivative.
pub(crate) struct SimplexJvpWorkspace {
    phases: [C64; 4],
    knots: [C64; 5],
    dd_table: Vec<C64>,
    dd_series: Vec<C64>,
}

impl SimplexJvpWorkspace {
    pub(crate) fn new() -> Self {
        Self {
            phases: [C64::new(0.0, 0.0); 4],
            knots: [C64::new(0.0, 0.0); 5],
            dd_table: Vec::with_capacity(28),
            dd_series: Vec::with_capacity(128),
        }
    }
}

/// Exact fixed-mass P0 simplex transform and position directional derivative.
/// The barycentric integral is `mass * dim! * exp[s0,...,sd]`, s=-ik·x.
/// Differentiating a knot repeats that knot in the divided difference. No
/// Eulerian volume derivative enters: each Lagrangian simplex mass is fixed.
/// Valid even for degenerate simplices and coincident vertex phases.
pub fn simplex_mass_transform_jvp(
    v: &[[f64; 3]],
    dv: &[[f64; 3]],
    dim: usize,
    k: [f64; 3],
    mass: f64,
) -> (C64, C64) {
    let mut workspace = SimplexJvpWorkspace::new();
    simplex_mass_transform_jvp_with_scratch(v, dv, dim, k, mass, &mut workspace)
}

/// Workspace form used when evaluating many adjacent simplices. It keeps the
/// public value and tangent formula identical while avoiding per-element heap
/// allocations for phases, repeated knots, and divided-difference scratch.
pub(crate) fn simplex_mass_transform_jvp_with_scratch(
    v: &[[f64; 3]],
    dv: &[[f64; 3]],
    dim: usize,
    k: [f64; 3],
    mass: f64,
    workspace: &mut SimplexJvpWorkspace,
) -> (C64, C64) {
    assert!((1..=3).contains(&dim) && v.len() == dim + 1 && dv.len() == v.len());
    let count = v.len();
    for i in 0..count {
        workspace.phases[i] = C64::new(
            0.0,
            -(0..dim).map(|a| k[a] * v[i][a]).sum::<f64>(),
        );
    }
    let scale = mass * (1..=dim).product::<usize>() as f64;
    let value = scale
        * exp_divided_difference_with_scratch(
            &workspace.phases[..count],
            &mut workspace.dd_table,
            &mut workspace.dd_series,
        );
    let mut deriv = C64::new(0.0, 0.0);
    for i in 0..count {
        workspace.knots[..count].copy_from_slice(&workspace.phases[..count]);
        workspace.knots[count] = workspace.phases[i];
        let ds = C64::new(0.0, -(0..dim).map(|a| k[a] * dv[i][a]).sum::<f64>());
        deriv += scale
            * ds
            * exp_divided_difference_with_scratch(
                &workspace.knots[..count + 1],
                &mut workspace.dd_table,
                &mut workspace.dd_series,
            );
    }
    (value, deriv)
}

/// Per-simplex mass-normalized P1 barycentric observation with optional weight
/// directions. Weights must have positive mean. Normalizing by their mean makes
/// the zero-mode mass exactly `mass`, including its zero derivative.
pub fn simplex_mass_transform_p1_jvp(
    v: &[[f64; 3]],
    dv: &[[f64; 3]],
    weights: &[f64],
    dweights: &[f64],
    dim: usize,
    k: [f64; 3],
    mass: f64,
) -> (C64, C64) {
    assert_eq!(v.len(), dim + 1);
    assert_eq!(dv.len(), v.len());
    assert_eq!(weights.len(), v.len());
    assert_eq!(dweights.len(), v.len());
    let mean = weights.iter().sum::<f64>() / weights.len() as f64;
    let dm = dweights.iter().sum::<f64>() / weights.len() as f64;
    assert!(mean > 0.0);
    let s: Vec<C64> = v
        .iter()
        .map(|p| C64::new(0.0, -(0..dim).map(|a| k[a] * p[a]).sum::<f64>()))
        .collect();
    let ds: Vec<C64> = dv
        .iter()
        .map(|p| C64::new(0.0, -(0..dim).map(|a| k[a] * p[a]).sum::<f64>()))
        .collect();
    let scale = mass * (1..=dim).product::<usize>() as f64;
    let mut value = C64::new(0.0, 0.0);
    let mut deriv = value;
    for j in 0..s.len() {
        let mut knots = s.clone();
        knots.push(s[j]);
        let dd = exp_divided_difference(&knots);
        let w = weights[j] / mean;
        let dw = dweights[j] / mean - weights[j] * dm / (mean * mean);
        value += scale * w * dd;
        deriv += scale * dw * dd;
        for i in 0..knots.len() {
            let mut repeated = knots.clone();
            repeated.push(knots[i]);
            deriv += scale
                * w
                * ds[if i == s.len() { j } else { i }]
                * exp_divided_difference(&repeated);
        }
    }
    (value, deriv)
}

/// Analytic logarithmic amplitude derivative δ0→exp(θ)δ0. Each order n is
/// homogeneous of degree n, including merged EdS terms and unmerged ΛCDM terms.
/// Returns interleaved position JVP and velocity (`dx/dD`) JVP at fixed D.
pub fn lpt_amplitude_jvp(lpt: &Lpt, d: f64, order: usize) -> (Vec<f64>, Vec<f64>) {
    let dim = lpt.grid.dim;
    let mut dx = vec![0.0; lpt.grid.size * dim];
    let mut dv = dx.clone();
    for (i, term) in lpt.terms.iter().enumerate() {
        if term.spec.order > order.min(lpt.order).max(1) {
            continue;
        }
        let (g, dg) = lpt.growth.g_and_dg(i, d);
        let n = term.spec.order as f64;
        for j in 0..lpt.grid.size {
            for a in 0..dim {
                dx[j * dim + a] += n * g * term.psi[a][j] as f64;
                dv[j * dim + a] += n * dg * term.psi[a][j] as f64;
            }
        }
    }
    (dx, dv)
}

/// Double-precision unmerged nLPT fields and a general IC directional derivative.
/// Keeps spectral derivative tensors during the recurrence. Uses the same
/// μ2, μ3 and curl sources and Growth table as `Lpt`, avoiding f32 quantization.
pub struct LptJvp {
    pub grid: Grid,
    pub growth: Growth,
    pub orders: Vec<usize>,
    pub psi: Vec<Vec<Vec<f64>>>,
    pub dpsi: Vec<Vec<Vec<f64>>>,
}

#[cfg(test)]
mod simplex_workspace_tests {
    use super::{simplex_mass_transform_jvp_with_scratch, SimplexJvpWorkspace};
    use num_complex::Complex64 as C64;

    #[test]
    fn repeated_knot_workspace_matches_closed_form_without_capacity_growth() {
        let k = [3.2, -1.7, 0.8];
        let mass = 0.37;
        let v = [[0.13, 0.27, 0.31]; 4];
        let dv = [
            [0.02, -0.01, 0.03],
            [-0.04, 0.05, 0.01],
            [0.03, 0.02, -0.02],
            [-0.01, 0.04, 0.06],
        ];
        let dim = 3;
        let phase = -(0..dim).map(|a| k[a] * v[0][a]).sum::<f64>();
        let expected_value = mass * C64::new(0.0, phase).exp();
        let mean_slope = -(0..4)
            .map(|i| (0..dim).map(|a| k[a] * dv[i][a]).sum::<f64>())
            .sum::<f64>()
            / 4.0;
        let expected_tangent = C64::new(0.0, mean_slope) * expected_value;

        let mut workspace = SimplexJvpWorkspace::new();
        let (value, tangent) = simplex_mass_transform_jvp_with_scratch(
            &v, &dv, dim, k, mass, &mut workspace,
        );
        assert!((value - expected_value).norm() < 2e-15);
        assert!((tangent - expected_tangent).norm() < 2e-15);
        let capacities = (
            workspace.dd_table.capacity(),
            workspace.dd_series.capacity(),
        );
        for _ in 0..32 {
            let (value, tangent) = simplex_mass_transform_jvp_with_scratch(
                &v, &dv, dim, k, mass, &mut workspace,
            );
            assert!((value - expected_value).norm() < 2e-15);
            assert!((tangent - expected_tangent).norm() < 2e-15);
            assert_eq!(
                (
                    workspace.dd_table.capacity(),
                    workspace.dd_series.capacity(),
                ),
                capacities,
                "repeated simplex evaluations should reuse divided-difference storage"
            );
        }
    }
}

fn solve(grid: &Grid, hat: &[Vec<C64>], curl: bool) -> Vec<Vec<C64>> {
    let mut out = vec![vec![C64::new(0.0, 0.0); grid.size]; grid.dim];
    if curl && grid.dim == 1 {
        return out;
    }
    for i in 0..grid.size {
        let k2 = grid.k2(i);
        if k2 == 0.0 {
            continue;
        }
        let k = grid.kvec(i);
        if !curl {
            for a in 0..grid.dim {
                out[a][i] = C64::new(0.0, -k[a] / k2) * hat[0][i];
            }
        } else if grid.dim == 2 {
            out[0][i] = C64::new(0.0, k[1] / k2) * hat[0][i];
            out[1][i] = C64::new(0.0, -k[0] / k2) * hat[0][i];
        } else {
            for a in 0..3 {
                let b = (a + 1) % 3;
                let c = (a + 2) % 3;
                out[a][i] = C64::new(0.0, 1.0 / k2) * (k[b] * hat[c][i] - k[c] * hat[b][i]);
            }
        }
    }
    out
}
fn curl_product(a: &[f64], b: &[f64], dim: usize, component: usize) -> f64 {
    if dim == 1 {
        return 0.0;
    }
    let (j, k) = if dim == 2 {
        (0, 1)
    } else {
        ((component + 1) % 3, (component + 2) % 3)
    };
    (0..dim)
        .map(|l| a[l * dim + j] * b[l * dim + k] - a[l * dim + k] * b[l * dim + j])
        .sum()
}
impl LptJvp {
    /// Build arbitrary directional derivative for Fourier fields in the existing
    /// unnormalized FFT convention; input fields must be Hermitian and zero mean.
    pub fn new(
        grid: &Grid,
        delta: &[C64],
        direction: &[C64],
        order: usize,
        cosmo: Cosmology,
    ) -> Self {
        assert_eq!(delta.len(), grid.size);
        assert_eq!(direction.len(), grid.size);
        let specs = term_specs(order, grid.dim);
        let growth = if cosmo.is_eds() {
            Growth::new_eds(&specs)
        } else {
            Growth::new_lcdm(cosmo, &specs, 20.0, 4000)
        };
        let mut eng = FftEngine::new(grid);
        let d = grid.dim;
        let mut psi = Vec::new();
        let mut dpsi = Vec::new();
        let mut m: Vec<Vec<Vec<f64>>> = Vec::new();
        let mut dm: Vec<Vec<Vec<f64>>> = Vec::new();
        for spec in &specs {
            let (h, dh) = if matches!(spec.parents, Parents::Linear) {
                let s = vec![delta.iter().map(|v| -v).collect()];
                let ds = vec![direction.iter().map(|v| -v).collect()];
                (solve(grid, &s, false), solve(grid, &ds, false))
            } else {
                let curl = matches!(spec.parents, Parents::Curl(..));
                let nc = if curl && d == 3 { 3 } else { 1 };
                let mut s = vec![vec![0.0; grid.size]; nc];
                let mut ds = s.clone();
                for i in 0..grid.size {
                    let tensor = |j: usize| -> Vec<f64> { m[j].iter().map(|v| v[i]).collect() };
                    let dtensor = |j: usize| -> Vec<f64> { dm[j].iter().map(|v| v[i]).collect() };
                    match spec.parents {
                        Parents::Mu2(a, b) => {
                            let (a, b, da, db) = (tensor(a), tensor(b), dtensor(a), dtensor(b));
                            s[0][i] = mu2(&a, &b, d);
                            ds[0][i] = mu2(&da, &b, d) + mu2(&a, &db, d);
                        }
                        Parents::Mu3(a, b, c) => {
                            let (a, b, c, da, db, dc) = (
                                tensor(a),
                                tensor(b),
                                tensor(c),
                                dtensor(a),
                                dtensor(b),
                                dtensor(c),
                            );
                            s[0][i] = mu3(&a, &b, &c, d);
                            ds[0][i] =
                                mu3(&da, &b, &c, d) + mu3(&a, &db, &c, d) + mu3(&a, &b, &dc, d);
                        }
                        Parents::Curl(a, b) => {
                            let (a, b, da, db) = (tensor(a), tensor(b), dtensor(a), dtensor(b));
                            for c in 0..nc {
                                s[c][i] = curl_product(&a, &b, d, c);
                                ds[c][i] =
                                    curl_product(&da, &b, d, c) + curl_product(&a, &db, d, c);
                            }
                        }
                        _ => unreachable!(),
                    }
                }
                let sh: Vec<_> = s.iter().map(|v| eng.forward_real(v)).collect();
                let dsh: Vec<_> = ds.iter().map(|v| eng.forward_real(v)).collect();
                (solve(grid, &sh, curl), solve(grid, &dsh, curl))
            };
            let mut mt = Vec::new();
            let mut dmt = Vec::new();
            let mut pt = Vec::new();
            let mut dpt = Vec::new();
            for a in 0..d {
                pt.push(eng.inverse_to_real(h[a].clone()));
                dpt.push(eng.inverse_to_real(dh[a].clone()));
                for b in 0..d {
                    mt.push(
                        eng.inverse_to_real(
                            h[a].iter()
                                .enumerate()
                                .map(|(i, v)| C64::new(0.0, grid.kvec(i)[b]) * v)
                                .collect(),
                        ),
                    );
                    dmt.push(
                        eng.inverse_to_real(
                            dh[a]
                                .iter()
                                .enumerate()
                                .map(|(i, v)| C64::new(0.0, grid.kvec(i)[b]) * v)
                                .collect(),
                        ),
                    );
                }
            }
            psi.push(pt);
            dpsi.push(dpt);
            m.push(mt);
            dm.push(dmt);
        }
        Self {
            grid: grid.clone(),
            growth,
            orders: specs.iter().map(|s| s.order).collect(),
            psi,
            dpsi,
        }
    }
    /// Interleaved `(x, dx/dD, directional x, directional dx/dD)` at fixed D.
    pub fn evaluate(&self, d: f64, order: usize) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
        let dim = self.grid.dim;
        let len = self.grid.size * dim;
        let mut x = vec![0.0; len];
        let mut v = x.clone();
        let mut dx = x.clone();
        let mut dv = x.clone();
        for i in 0..self.grid.size {
            let ijk = self.grid.unravel(i);
            for a in 0..dim {
                x[i * dim + a] = ijk[a] as f64 * self.grid.dx();
            }
        }
        for t in 0..self.orders.len() {
            if self.orders[t] > order {
                continue;
            }
            let (g, dg) = self.growth.g_and_dg(t, d);
            for i in 0..self.grid.size {
                for a in 0..dim {
                    let j = i * dim + a;
                    x[j] += g * self.psi[t][a][i];
                    v[j] += dg * self.psi[t][a][i];
                    dx[j] += g * self.dpsi[t][a][i];
                    dv[j] += dg * self.dpsi[t][a][i];
                }
            }
        }
        (x, v, dx, dv)
    }
}
