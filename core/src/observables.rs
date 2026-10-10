//! Conservative pre-crossing sheet observables for the teaching laboratory.
//! Fixed simplex masses, unwrapped periodic connectivity, cell-average fields,
//! and PDFs of log10(rho) are kept distinct from the displacement dynamics.
use crate::{fft::FftEngine, grid::Grid};
use num_complex::Complex64;

/// Visit each periodic interval, triangle, or Kuhn tetrahedron.
/// Vertices are `(flat_vertex_index, periodic_lift)`; masses use mean rho = 1.
pub fn for_each_simplex(grid: &Grid, mut visit: impl FnMut(&[(usize, [f64; 3])], f64)) {
    let d = grid.dim;
    let n = grid.n;
    let cell = grid.dx().powi(d as i32);
    const TETS: [[usize; 4]; 6] = [
        [0, 1, 3, 7],
        [0, 1, 5, 7],
        [0, 2, 3, 7],
        [0, 2, 6, 7],
        [0, 4, 5, 7],
        [0, 4, 6, 7],
    ];
    for id in 0..grid.size {
        let ijk = grid.unravel(id);
        let mut vertices = Vec::with_capacity(1 << d);
        for b in 0..(1 << d) {
            let mut q = ijk;
            let mut lift = [0.0; 3];
            for a in 0..d {
                q[a] += (b >> a) & 1;
                if q[a] >= n {
                    q[a] -= n;
                    lift[a] = grid.l;
                }
            }
            vertices.push((grid.ravel(q), lift));
        }
        match d {
            1 => visit(&[vertices[0], vertices[1]], cell),
            2 => {
                visit(&[vertices[0], vertices[1], vertices[3]], cell / 2.0);
                visit(&[vertices[0], vertices[3], vertices[2]], cell / 2.0);
            }
            _ => {
                for t in TETS {
                    visit(
                        &[
                            vertices[t[0]],
                            vertices[t[1]],
                            vertices[t[2]],
                            vertices[t[3]],
                        ],
                        cell / 6.0,
                    )
                }
            }
        }
    }
}
fn det(v: &[[f64; 3]], d: usize) -> f64 {
    let mut e = [[0.0; 3]; 3];
    for i in 0..d {
        for a in 0..d {
            e[i][a] = v[i + 1][a] - v[0][a];
        }
    }
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
/// Smallest signed volume ratio of the piecewise-affine sheet. A positive
/// spectral vertex Jacobian alone is insufficient to establish this property.
pub fn minimum_simplex_jacobian(grid: &Grid, pos: &[f64]) -> f64 {
    assert_eq!(pos.len(), grid.size * grid.dim);
    let mut minimum = f64::INFINITY;
    for_each_simplex(grid, |verts, _| {
        let mut x = Vec::new();
        let mut q = Vec::new();
        for &(id, lift) in verts {
            let ijk = grid.unravel(id);
            let mut p = [0.; 3];
            let mut r = [0.; 3];
            for a in 0..grid.dim {
                p[a] = pos[id * grid.dim + a] + lift[a];
                r[a] = ijk[a] as f64 * grid.dx() + lift[a];
            }
            x.push(p);
            q.push(r);
        }
        minimum = minimum.min(det(&x, grid.dim) / det(&q, grid.dim));
    });
    minimum
}
/// Exact mass overlap for the pre-crossing 1D affine map (periodic lifts).
/// The 2D and 3D counterparts are the inherited R3D clipping routines.
pub fn interval_density(grid: &Grid, pos: &[f64], ne: usize) -> Vec<f32> {
    assert_eq!(grid.dim, 1);
    let dx = grid.l / ne as f64;
    let mut rho = vec![0.; ne];
    for i in 0..grid.n {
        let lo = pos[i];
        let hi = pos[(i + 1) % grid.n] + if i + 1 == grid.n { grid.l } else { 0. };
        assert!(hi > lo, "crossed interval");
        let density = grid.dx() / (hi - lo);
        let first = (lo / dx).floor() as i64;
        let last = (hi / dx).ceil() as i64;
        for j in first..last {
            let overlap = (hi.min((j + 1) as f64 * dx) - lo.max(j as f64 * dx)).max(0.);
            rho[j.rem_euclid(ne as i64) as usize] += density * overlap / dx;
        }
    }
    rho.into_iter().map(|x| x as f32).collect()
}
/// Gaussian smoothing of an Eulerian cell-average field. R is in box units;
/// this observation filter is distinct from the initial-condition cutoff.
pub fn smooth_density(grid: &Grid, rho: &[f32], r: f64) -> Vec<f32> {
    if r == 0. {
        return rho.to_vec();
    }
    let mut eng = FftEngine::new(grid);
    let mut hat = eng.forward_real_f32(rho);
    for (i, h) in hat.iter_mut().enumerate() {
        *h *= (-0.5 * grid.k2(i) * r * r).exp();
    }
    eng.inverse_to_real_f32(hat)
}
/// PDF ordinates are density per unit y=log10(rho), without renormalizing
/// truncated tails. Mass weights are the *unsmoothed* cell densities: for
/// R>0 this is the distribution of smoothed environments around matter.
pub fn density_pdf(
    rho: &[f32],
    raw_mass: &[f32],
    bins: usize,
) -> (Vec<f64>, Vec<f64>, Vec<f64>, f64, f64) {
    assert_eq!(rho.len(), raw_mass.len());
    let (lo, hi) = (-1.8, 1.8);
    let dy = (hi - lo) / bins as f64;
    let mut pv = vec![0.; bins];
    let mut pm = vec![0.; bins];
    let total_mass: f64 = raw_mass.iter().map(|&r| r as f64).sum();
    for (&r, &w) in rho.iter().zip(raw_mass) {
        if r <= 0. {
            continue;
        }
        let y = (r as f64).log10();
        let b = ((y - lo) / dy).floor() as isize;
        if b >= 0 && (b as usize) < bins {
            pv[b as usize] += 1. / (rho.len() as f64 * dy);
            pm[b as usize] += w as f64 / (total_mass * dy);
        }
    }
    let mean = rho.iter().map(|&r| r as f64).sum::<f64>() / rho.len() as f64;
    let var = rho.iter().map(|&r| (r as f64 - mean).powi(2)).sum::<f64>() / rho.len() as f64;
    (
        (0..bins).map(|b| lo + (b as f64 + 0.5) * dy).collect(),
        pv,
        pm,
        mean,
        var,
    )
}
/// Continuous P0 sheet mode and directional derivative for fixed element
/// masses. `jvp` maps normalized vertex phases, so no volume derivative is
/// needed: density and volume factors cancel in each element's mass.
pub fn sheet_mode_jvp(
    grid: &Grid,
    pos: &[f64],
    dpos: &[f64],
    k: [f64; 3],
) -> (Complex64, Complex64) {
    let mut value = Complex64::new(0., 0.);
    let mut tangent = Complex64::new(0., 0.);
    for_each_simplex(grid, |verts, mass| {
        let mut x = Vec::new();
        let mut dx = Vec::new();
        for &(id, lift) in verts {
            let mut p = [0.; 3];
            let mut dp = [0.; 3];
            for a in 0..grid.dim {
                p[a] = pos[id * grid.dim + a] + lift[a];
                dp[a] = dpos[id * grid.dim + a];
            }
            x.push(p);
            dx.push(dp);
        }
        let (v, t) = crate::tangent::simplex_mass_transform_jvp(&x, &dx, grid.dim, k, mass);
        value += v;
        tangent += t;
    });
    (value, tangent)
}
