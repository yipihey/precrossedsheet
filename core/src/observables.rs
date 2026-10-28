//! Conservative sheet observables for the numerical laboratory.
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
        let mut vertices = [(0usize, [0.0; 3]); 8];
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
            vertices[b] = (grid.ravel(q), lift);
        }
        match d {
            1 => visit(&vertices[..2], cell),
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
        let mut x = [[0.0; 3]; 4];
        let mut q = [[0.0; 3]; 4];
        for (vi, &(id, lift)) in verts.iter().enumerate() {
            let ijk = grid.unravel(id);
            for a in 0..grid.dim {
                x[vi][a] = pos[id * grid.dim + a] + lift[a];
                q[vi][a] = ijk[a] as f64 * grid.dx() + lift[a];
            }
        }
        let n = verts.len();
        minimum = minimum.min(det(&x[..n], grid.dim) / det(&q[..n], grid.dim));
    });
    minimum
}
/// Fraction of initial mass in negatively oriented affine elements at this
/// epoch. This is a final signed-orientation statistic, not an ever-crossed
/// union, Eulerian volume fraction, or local stream multiplicity.
pub fn inverted_mass_fraction(grid: &Grid, pos: &[f64]) -> f64 {
    let mut crossed = 0.;
    for_each_simplex(grid, |verts, mass| {
        let mut x = [[0.0; 3]; 4];
        let mut q = [[0.0; 3]; 4];
        for (vi, &(id, lift)) in verts.iter().enumerate() {
            let ijk = grid.unravel(id);
            for a in 0..grid.dim {
                x[vi][a] = pos[id * grid.dim + a] + lift[a];
                q[vi][a] = ijk[a] as f64 * grid.dx() + lift[a];
            }
        }
        let n = verts.len();
        if det(&x[..n], grid.dim) / det(&q[..n], grid.dim) < 0. {
            crossed += mass;
        }
    });
    crossed / grid.l.powi(grid.dim as i32)
}
/// Vertex inverse-Jacobian weights w_i = mean(1 / |J_S|) over incident simplices
/// for P1 continuous sheet interpolation on any vertex position field (e.g. PM).
pub fn vertex_inverse_jacobians(grid: &Grid, pos: &[f64]) -> Vec<f32> {
    assert_eq!(pos.len(), grid.size * grid.dim);
    let mut w_sum = vec![0.0_f64; grid.size];
    let mut w_cnt = vec![0_u32; grid.size];
    for_each_simplex(grid, |verts, _| {
        let mut x = [[0.0; 3]; 4];
        let mut q = [[0.0; 3]; 4];
        for (vi, &(id, lift)) in verts.iter().enumerate() {
            let ijk = grid.unravel(id);
            for a in 0..grid.dim {
                x[vi][a] = pos[id * grid.dim + a] + lift[a];
                q[vi][a] = ijk[a] as f64 * grid.dx() + lift[a];
            }
        }
        let n = verts.len();
        let j = (det(&x[..n], grid.dim) / det(&q[..n], grid.dim))
            .abs()
            .max(1e-6);
        let inv_j = 1.0 / j;
        for &(id, _) in verts {
            w_sum[id] += inv_j;
            w_cnt[id] += 1;
        }
    });
    w_sum
        .into_iter()
        .zip(w_cnt)
        .map(|(s, c)| if c > 0 { (s / c as f64) as f32 } else { 1.0 })
        .collect()
}

/// Exact mass overlap for the 1D affine map (periodic lifts), including folds.
/// The 2D and 3D counterparts are the inherited R3D clipping routines.
pub fn interval_density(grid: &Grid, pos: &[f64], ne: usize) -> Vec<f32> {
    assert_eq!(grid.dim, 1);
    let dx = grid.l / ne as f64;
    let mut rho = vec![0.; ne];
    for i in 0..grid.n {
        let a = pos[i];
        let b = pos[(i + 1) % grid.n] + if i + 1 == grid.n { grid.l } else { 0. };
        let (lo, hi) = (a.min(b), a.max(b));
        if hi == lo {
            // A collapsed interval is a point mass; choose the containing cell.
            let j = (lo / dx).floor() as i64;
            rho[j.rem_euclid(ne as i64) as usize] += grid.dx() / dx;
            continue;
        }
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

/// Exact P1 (piecewise-linear vertex weight) mass overlap for the 1D affine map.
pub fn interval_density_p1(grid: &Grid, pos: &[f64], w: &[f32], ne: usize) -> Vec<f32> {
    assert_eq!(grid.dim, 1);
    let dx = grid.l / ne as f64;
    let mut rho = vec![0.; ne];
    for i in 0..grid.n {
        let a = pos[i];
        let b = pos[(i + 1) % grid.n] + if i + 1 == grid.n { grid.l } else { 0. };
        let wa = (w[i] as f64).max(0.0);
        let wb = (w[(i + 1) % grid.n] as f64).max(0.0);
        let (lo, hi, w_lo, w_hi) = if a <= b {
            (a, b, wa, wb)
        } else {
            (b, a, wb, wa)
        };
        if hi == lo {
            let j = (lo / dx).floor() as i64;
            rho[j.rem_euclid(ne as i64) as usize] += grid.dx() / dx;
            continue;
        }
        let span = hi - lo;
        let density = grid.dx() / span;
        let w_mean = 0.5 * (w_lo + w_hi);
        let first = (lo / dx).floor() as i64;
        let last = (hi / dx).ceil() as i64;
        for j in first..last {
            let sub_lo = lo.max(j as f64 * dx);
            let sub_hi = hi.min((j + 1) as f64 * dx);
            if sub_hi > sub_lo {
                let overlap = sub_hi - sub_lo;
                let shape = if w_mean > 0.0 {
                    let lam = (0.5 * (sub_lo + sub_hi) - lo) / span;
                    ((1.0 - lam) * w_lo + lam * w_hi) / w_mean
                } else {
                    1.0
                };
                rho[j.rem_euclid(ne as i64) as usize] += density * overlap * shape / dx;
            }
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

/// Full-field moments and inverse empirical CDFs, independent of histogram
/// clipping. Mass CDFs weight the smoothed environment by raw cell mass.
pub fn density_summary(rho: &[f32], raw_mass: &[f32], probabilities: &[f64]) -> serde_json::Value {
    assert_eq!(rho.len(), raw_mass.len());
    let count = rho.len() as f64;
    let mean = rho.iter().map(|&r| r as f64).sum::<f64>() / count;
    let central = |order: i32| {
        rho.iter()
            .map(|&r| (r as f64 - mean).powi(order))
            .sum::<f64>()
            / count
    };
    let variance = central(2);
    let s3 = if variance > 0. {
        Some(central(3) / variance.powi(2))
    } else {
        None
    };
    let s4 = if variance > 0. {
        Some((central(4) - 3. * variance.powi(2)) / variance.powi(3))
    } else {
        None
    };
    let mut sorted: Vec<usize> = (0..rho.len()).collect();
    sorted.sort_unstable_by(|&i, &j| rho[i].total_cmp(&rho[j]));
    let total = raw_mass.iter().map(|&r| r as f64).sum::<f64>();
    let quantile = |p: f64, mass: bool| {
        let target = p * if mass { total } else { count };
        let mut cumulative = 0.;
        for &i in &sorted {
            cumulative += if mass { raw_mass[i] as f64 } else { 1. };
            if cumulative >= target {
                return rho[i] as f64;
            }
        }
        rho[*sorted.last().unwrap()] as f64
    };
    const FINE_GRID_BINS: usize = 320;
    let (f_lo, f_hi) = (-1.8, 1.8);
    let f_dy = (f_hi - f_lo) / FINE_GRID_BINS as f64;
    let mut f_pv = vec![0.0_f64; FINE_GRID_BINS];
    let mut f_pm = vec![0.0_f64; FINE_GRID_BINS];
    let tot_m = total.max(1e-30);
    for (&r, &w) in rho.iter().zip(raw_mass) {
        if r <= 0.0 {
            continue;
        }
        let y = (r as f64).log10();
        let b = ((y - f_lo) / f_dy).floor() as isize;
        if b >= 0 && (b as usize) < FINE_GRID_BINS {
            f_pv[b as usize] += 1.0 / (count * f_dy);
            f_pm[b as usize] += (w as f64) / (tot_m * f_dy);
        }
    }
    let f_logrho: Vec<f64> = (0..FINE_GRID_BINS)
        .map(|b| f_lo + (b as f64 + 0.5) * f_dy)
        .collect();
    serde_json::json!({"mean":mean,"variance":variance,"S3":s3,"S4":s4,
        "probabilities":probabilities,
        "volumeQuantiles":probabilities.iter().map(|&p|quantile(p,false)).collect::<Vec<_>>(),
        "massQuantiles":probabilities.iter().map(|&p|quantile(p,true)).collect::<Vec<_>>(),
        "finePdf":{"logrho":f_logrho,"dy":f_dy,"volume":f_pv,"mass":f_pm}})
}

/// Compute the continuous sheet volume-weighted and mass-weighted log10(rho)
/// PDF and full-field summary statistics directly from the Lagrangian simplices
/// (intervals in 1D, triangles in 2D, Kuhn tetrahedra in 3D) without depositing
/// onto an Eulerian grid.
///
/// When `vertex_weights` is `None`, evaluates the exact piecewise-constant (P0)
/// simplex densities `r_T = 1 / |J_T|` with volume weight `V_T = m_T |J_T|` and
/// mass weight `m_T`. When `vertex_weights` is `Some(w)`, evaluates the P1
/// corner densities `r_{T,a} = r_T * w_a / mean(w_T)` with volume weight
/// `V_T / (d + 1)` and mass weight `m_T * w_a / ((d + 1) * mean(w_T))`.
pub fn simplex_density_pdf_and_summary(
    grid: &Grid,
    pos: &[f64],
    vertex_weights: Option<&[f32]>,
    bins: usize,
    probabilities: &[f64],
) -> (Vec<f64>, Vec<f64>, Vec<f64>, f64, f64, f64, serde_json::Value) {
    assert_eq!(pos.len(), grid.size * grid.dim);
    let d = grid.dim;
    let q_det = grid.dx().powi(d as i32);
    let mut min_y = f64::INFINITY;
    let mut max_y = f64::NEG_INFINITY;
    let mut total_vol = 0.0_f64;
    let mut total_mass = 0.0_f64;

    // Symmetric barycentric lattice on the d-simplex (including the vertices
    // themselves as well as edge/interior nodes) so that P1 linear interpolation
    // reaches the true nodal extrema while conserving simplex volume and mass.
    let p1_bary_1d: &[[f64; 4]] = &[
        [1.0, 0.0, 0.0, 0.0],
        [2.0 / 3.0, 1.0 / 3.0, 0.0, 0.0],
        [1.0 / 3.0, 2.0 / 3.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
    ];
    let p1_bary_2d: &[[f64; 4]] = &[
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [2.0 / 3.0, 1.0 / 3.0, 0.0, 0.0],
        [2.0 / 3.0, 0.0, 1.0 / 3.0, 0.0],
        [1.0 / 3.0, 2.0 / 3.0, 0.0, 0.0],
        [0.0, 2.0 / 3.0, 1.0 / 3.0, 0.0],
        [1.0 / 3.0, 0.0, 2.0 / 3.0, 0.0],
        [0.0, 1.0 / 3.0, 2.0 / 3.0, 0.0],
        [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0, 0.0],
    ];
    let p1_bary_3d: &[[f64; 4]] = &[
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
        [0.5, 0.5, 0.0, 0.0],
        [0.5, 0.0, 0.5, 0.0],
        [0.5, 0.0, 0.0, 0.5],
        [0.0, 0.5, 0.5, 0.0],
        [0.0, 0.5, 0.0, 0.5],
        [0.0, 0.0, 0.5, 0.5],
    ];
    let p1_bary = match d {
        1 => p1_bary_1d,
        2 => p1_bary_2d,
        _ => p1_bary_3d,
    };

    // Pass 1: streaming extrema, total Eulerian simplex volume, and total mass.
    for_each_simplex(grid, |verts, mass| {
        let mut x = [[0.0; 3]; 4];
        for (vi, &(id, lift)) in verts.iter().enumerate() {
            for a in 0..d {
                x[vi][a] = pos[id * d + a] + lift[a];
            }
        }
        let n = verts.len();
        let abs_j = (det(&x[..n], d) / q_det).abs().max(1e-8);
        let r_t = 1.0 / abs_j;
        let v_t = mass * abs_j;
        if let Some(w) = vertex_weights {
            let mut w_v = [0.0_f64; 4];
            let mut w_sum = 0.0_f64;
            let mut w_min = f64::INFINITY;
            let mut w_max = f64::NEG_INFINITY;
            for (vi, &(id, _)) in verts.iter().enumerate() {
                let wa = (w[id] as f64).max(1e-8);
                w_v[vi] = wa;
                w_sum += wa;
                if wa < w_min {
                    w_min = wa;
                }
                if wa > w_max {
                    w_max = wa;
                }
            }
            let w_bar = w_sum / n as f64;
            let r_lo = r_t.min(w_min);
            let r_hi = r_t.max(w_max);
            let mut alpha = 1.0_f64;
            for vi in 0..n {
                let dw = w_v[vi] - w_bar;
                if dw > 1e-12 {
                    alpha = alpha.min((r_hi - r_t) / dw);
                } else if dw < -1e-12 {
                    alpha = alpha.min((r_lo - r_t) / dw);
                }
            }
            alpha = alpha.clamp(0.0, 1.0);
            let k_pts = p1_bary.len() as f64;
            for bary in p1_bary {
                let mut dw_q = 0.0_f64;
                for vi in 0..n {
                    dw_q += bary[vi] * (w_v[vi] - w_bar);
                }
                let r_q = (r_t + alpha * dw_q).max(1e-8);
                let shape = r_q / r_t;
                let y = r_q.log10();
                if y < min_y {
                    min_y = y;
                }
                if y > max_y {
                    max_y = y;
                }
                total_vol += v_t / k_pts;
                total_mass += mass * shape / k_pts;
            }
        } else {
            let y = r_t.log10();
            if y < min_y {
                min_y = y;
            }
            if y > max_y {
                max_y = y;
            }
            total_vol += v_t;
            total_mass += mass;
        }
    });

    let total_vol = total_vol.max(1e-30);
    let total_mass = total_mass.max(1e-30);
    let mean = total_mass / total_vol;

    let (lo, hi) = if !min_y.is_finite() || !max_y.is_finite() || (max_y - min_y).abs() < 1e-9 {
        (-1.8, 1.8)
    } else {
        let pad = ((max_y - min_y) * 0.015).max(1e-3);
        let lo_bound = (min_y - pad).max(-8.0);
        let hi_bound = (max_y + pad).min(8.0);
        (lo_bound, hi_bound)
    };
    let dy = (hi - lo) / bins as f64;

    const FINE_BINS: usize = 640;
    let fine_dy = (hi - lo) / FINE_BINS as f64;
    let mut fine_pv = vec![0.0_f64; FINE_BINS];
    let mut fine_pm = vec![0.0_f64; FINE_BINS];

    const CDF_BINS: usize = 8192;
    let cdf_lo = (min_y - 1e-4).min(lo);
    let cdf_hi = (max_y + 1e-4).max(hi).max(cdf_lo + 1e-3);
    let cdf_dy = (cdf_hi - cdf_lo) / CDF_BINS as f64;

    let mut pv = vec![0.0_f64; bins];
    let mut pm = vec![0.0_f64; bins];
    let mut cdf_v = vec![0.0_f64; CDF_BINS];
    let mut cdf_m = vec![0.0_f64; CDF_BINS];
    let mut m2 = 0.0_f64;
    let mut m3 = 0.0_f64;
    let mut m4 = 0.0_f64;

    // Pass 2: streaming histogram, central moments, and fine CDF bins.
    for_each_simplex(grid, |verts, mass| {
        let mut x = [[0.0; 3]; 4];
        for (vi, &(id, lift)) in verts.iter().enumerate() {
            for a in 0..d {
                x[vi][a] = pos[id * d + a] + lift[a];
            }
        }
        let n = verts.len();
        let abs_j = (det(&x[..n], d) / q_det).abs().max(1e-8);
        let r_t = 1.0 / abs_j;
        let v_t = mass * abs_j;
        let mut accum = |r: f64, v: f64, m: f64| {
            let y = r.log10();
            let b = ((y - lo) / dy).floor() as isize;
            if b >= 0 && (b as usize) < bins {
                pv[b as usize] += v / (total_vol * dy);
                pm[b as usize] += m / (total_mass * dy);
            }
            let fb = ((y - lo) / fine_dy).floor() as isize;
            if fb >= 0 && (fb as usize) < FINE_BINS {
                fine_pv[fb as usize] += v / (total_vol * fine_dy);
                fine_pm[fb as usize] += m / (total_mass * fine_dy);
            }
            let cb = (((y - cdf_lo) / cdf_dy).floor() as isize).clamp(0, CDF_BINS as isize - 1) as usize;
            cdf_v[cb] += v;
            cdf_m[cb] += m;
            let dr = r - mean;
            let dr2 = dr * dr;
            let vw = v / total_vol;
            m2 += vw * dr2;
            m3 += vw * dr2 * dr;
            m4 += vw * dr2 * dr2;
        };
        if let Some(w) = vertex_weights {
            let mut w_v = [0.0_f64; 4];
            let mut w_sum = 0.0_f64;
            let mut w_min = f64::INFINITY;
            let mut w_max = f64::NEG_INFINITY;
            for (vi, &(id, _)) in verts.iter().enumerate() {
                let wa = (w[id] as f64).max(1e-8);
                w_v[vi] = wa;
                w_sum += wa;
                if wa < w_min {
                    w_min = wa;
                }
                if wa > w_max {
                    w_max = wa;
                }
            }
            let w_bar = w_sum / n as f64;
            let r_lo = r_t.min(w_min);
            let r_hi = r_t.max(w_max);
            let mut alpha = 1.0_f64;
            for vi in 0..n {
                let dw = w_v[vi] - w_bar;
                if dw > 1e-12 {
                    alpha = alpha.min((r_hi - r_t) / dw);
                } else if dw < -1e-12 {
                    alpha = alpha.min((r_lo - r_t) / dw);
                }
            }
            alpha = alpha.clamp(0.0, 1.0);
            let k_pts = p1_bary.len() as f64;
            for bary in p1_bary {
                let mut dw_q = 0.0_f64;
                for vi in 0..n {
                    dw_q += bary[vi] * (w_v[vi] - w_bar);
                }
                let r_q = (r_t + alpha * dw_q).max(1e-8);
                let shape = r_q / r_t;
                accum(r_q, v_t / k_pts, mass * shape / k_pts);
            }
        } else {
            accum(r_t, v_t, mass);
        }
    });

    let variance = m2;
    let s3 = if variance > 1e-24 {
        Some(m3 / variance.powi(2))
    } else {
        None
    };
    let s4 = if variance > 1e-24 {
        Some((m4 - 3.0 * variance.powi(2)) / variance.powi(3))
    } else {
        None
    };

    let quantile_from_cdf = |p: f64, is_mass: bool| -> f64 {
        let tot = if is_mass { total_mass } else { total_vol };
        let hist = if is_mass { &cdf_m } else { &cdf_v };
        let target = p * tot;
        let mut cum = 0.0_f64;
        for (i, &w) in hist.iter().enumerate() {
            let next = cum + w;
            if next >= target && w > 0.0 {
                let frac = ((target - cum) / w).clamp(0.0, 1.0);
                let y = cdf_lo + (i as f64 + frac) * cdf_dy;
                return 10.0_f64.powf(y);
            }
            cum = next;
        }
        10.0_f64.powf(max_y)
    };

    let fine_logrho: Vec<f64> = (0..FINE_BINS)
        .map(|b| lo + (b as f64 + 0.5) * fine_dy)
        .collect();
    let summary = serde_json::json!({
        "mean": mean,
        "variance": variance,
        "S3": s3,
        "S4": s4,
        "probabilities": probabilities,
        "volumeQuantiles": probabilities.iter().map(|&p| quantile_from_cdf(p, false)).collect::<Vec<_>>(),
        "massQuantiles": probabilities.iter().map(|&p| quantile_from_cdf(p, true)).collect::<Vec<_>>(),
        "finePdf": {
            "logrho": fine_logrho,
            "dy": fine_dy,
            "volume": fine_pv,
            "mass": fine_pm,
        },
    });
    let logrho: Vec<f64> = (0..bins).map(|b| lo + (b as f64 + 0.5) * dy).collect();
    (logrho, pv, pm, mean, variance, dy, summary)
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
    let mut x = [[0.0; 3]; 4];
    let mut dx = [[0.0; 3]; 4];
    let mut workspace = crate::tangent::SimplexJvpWorkspace::new();
    for_each_simplex(grid, |verts, mass| {
        for (i, &(id, lift)) in verts.iter().enumerate() {
            for a in 0..grid.dim {
                x[i][a] = pos[id * grid.dim + a] + lift[a];
                dx[i][a] = dpos[id * grid.dim + a];
            }
        }
        let count = verts.len();
        let (v, t) = crate::tangent::simplex_mass_transform_jvp_with_scratch(
            &x[..count],
            &dx[..count],
            grid.dim,
            k,
            mass,
            &mut workspace,
        );
        value += v;
        tangent += t;
    });
    (value, tangent)
}

/// Value-only continuous P0 mode. Reuses all divided-difference workspaces
/// across simplices, which is the CPU finite-difference path for prepared runs.
pub fn sheet_mode(grid: &Grid, pos: &[f64], k: [f64; 3]) -> Complex64 {
    assert_eq!(pos.len(), grid.size * grid.dim);
    let mut value = Complex64::new(0.0, 0.0);
    let mut phases = Vec::with_capacity(4);
    let mut table = Vec::with_capacity(28);
    let mut series = Vec::with_capacity(25);
    for_each_simplex(grid, |verts, mass| {
        phases.clear();
        for &(id, lift) in verts {
            let phase = (0..grid.dim)
                .map(|a| k[a] * (pos[id * grid.dim + a] + lift[a]))
                .sum::<f64>();
            phases.push(Complex64::new(0.0, -phase));
        }
        let factorial = (1..=grid.dim).product::<usize>() as f64;
        value += (mass * factorial)
            * crate::direct::exp_divided_difference_with_scratch(&phases, &mut table, &mut series);
    });
    value
}

#[inline]
fn cpow_u(mut base: Complex64, mut exp: usize) -> Complex64 {
    let mut acc = Complex64::new(1.0, 0.0);
    while exp > 0 {
        if (exp & 1) != 0 {
            acc *= base;
        }
        exp >>= 1;
        if exp > 0 {
            base *= base;
        }
    }
    acc
}

#[inline]
fn taylor_coeffs_first_axis(dx_row: &[f64; 4], i: usize, level: usize) -> [f64; 13] {
    const TERMS: usize = 12;
    let d1 = dx_row[i + 1];
    let mut g = [0.0f64; TERMS + 1];
    g[0] = 1.0;
    for p in 1..=TERMS {
        g[p] = g[p - 1] * d1 / (p + 1) as f64;
    }
    for a in 2..=level {
        let da = dx_row[i + a];
        g[0] /= a as f64;
        for p in 1..=TERMS {
            g[p] = (g[p] + da * g[p - 1]) / (p + a) as f64;
        }
    }
    g
}

#[inline]
fn eval_taylor_horner(e_base: Complex64, k: f64, k2: f64, c: &[f64; 13]) -> Complex64 {
    let re = c[0]
        - k2 * (c[2] - k2 * (c[4] - k2 * (c[6] - k2 * (c[8] - k2 * (c[10] - k2 * c[12])))));
    let im = -k * (c[1] - k2 * (c[3] - k2 * (c[5] - k2 * (c[7] - k2 * (c[9] - k2 * c[11])))));
    e_base * Complex64::new(re, im)
}

/// Evaluate continuous P0 simplex Fourier modes along the first axis k = (m * k_f, 0, 0)
/// for a sorted slice of positive integer mode numbers `modes`.
pub fn sheet_modes_first_axis(grid: &Grid, pos: &[f64], modes: &[usize]) -> Vec<Complex64> {
    let cap = if grid.dim == 2 { 256 } else { 32 };
    sheet_modes_along_ray(grid, pos, [1, 0, 0], modes, cap)
}

/// Evaluate continuous P0 simplex Fourier modes along an integer lattice ray
/// `k = m * k_f * (dir[0], dir[1], dir[2])` for a sorted slice of positive integer
/// harmonics `modes`. When `grid.n > max_sub_n` and `modes.len() > 6`, evaluates
/// on a connected isotropic sub-lattice of resolution `max_sub_n` per axis.
pub fn sheet_modes_along_ray(
    grid: &Grid,
    pos: &[f64],
    dir: [i32; 3],
    modes: &[usize],
    max_sub_n: usize,
) -> Vec<Complex64> {
    assert_eq!(pos.len(), grid.size * grid.dim);
    let nm = modes.len();
    if nm == 0 {
        return Vec::new();
    }
    let kf = grid.kf();
    let ks: Vec<f64> = modes.iter().map(|&m| kf * m as f64).collect();
    let k2s: Vec<f64> = ks.iter().map(|&k| k * k).collect();
    let inv_ks: Vec<f64> = ks.iter().map(|&k| 1.0 / k).collect();
    let dms: Vec<usize> = modes
        .iter()
        .enumerate()
        .map(|(i, &m)| if i == 0 { m } else { m - modes[i - 1] })
        .collect();
    let sub_stride = if nm > 6 && max_sub_n > 0 && grid.n > max_sub_n {
        (1..=grid.n)
            .find(|&s| grid.n % s == 0 && grid.n / s <= max_sub_n)
            .unwrap_or(1)
    } else {
        1
    };
    let mut z1 = vec![Complex64::new(0.0, 0.0); grid.size];
    let cnt = grid.dim + 1;
    let factorial = (1..=grid.dim).product::<usize>() as f64;
    let mut out = vec![Complex64::new(0.0, 0.0); nm];
    let df0 = dir[0] as f64;
    let df1 = dir[1] as f64;
    let df2 = dir[2] as f64;
    let mut visit_simplex = |verts: &[(usize, [f64; 3])], mass: f64| {
        let scale = mass * factorial;
        let mut x = [0.0f64; 4];
        let mut z = [Complex64::new(1.0, 0.0); 4];
        for (a, &(id, lift)) in verts.iter().enumerate() {
            let base = id * grid.dim;
            let s = match grid.dim {
                1 => df0 * (pos[base] + lift[0]),
                2 => df0 * (pos[base] + lift[0]) + df1 * (pos[base + 1] + lift[1]),
                _ => {
                    df0 * (pos[base] + lift[0])
                        + df1 * (pos[base + 1] + lift[1])
                        + df2 * (pos[base + 2] + lift[2])
                }
            };
            x[a] = s;
            let mut z_id = z1[id];
            if z_id.re == 0.0 && z_id.im == 0.0 {
                let s0 = match grid.dim {
                    1 => df0 * pos[base],
                    2 => df0 * pos[base] + df1 * pos[base + 1],
                    _ => df0 * pos[base] + df1 * pos[base + 1] + df2 * pos[base + 2],
                };
                let (sv, cv) = (kf * s0).sin_cos();
                z_id = Complex64::new(cv, -sv);
                z1[id] = z_id;
            }
            z[a] = z_id;
        }
        for a in 1..cnt {
            let xv = x[a];
            let zv = z[a];
            let mut b = a;
            while b > 0 && x[b - 1] > xv {
                x[b] = x[b - 1];
                z[b] = z[b - 1];
                b -= 1;
            }
            x[b] = xv;
            z[b] = zv;
        }
        let mut dx = [[0.0f64; 4]; 4];
        let mut inv_dx = [[0.0f64; 4]; 4];
        for i in 0..cnt {
            for j in (i + 1)..cnt {
                let d = x[j] - x[i];
                dx[i][j] = d;
                if d > 0.0 {
                    inv_dx[i][j] = 1.0 / d;
                }
            }
        }
        let c_full = taylor_coeffs_first_axis(&dx[0], 0, cnt - 1);
        let max_span = dx[0][cnt - 1];
        let mut e = [Complex64::new(1.0, 0.0); 4];
        let mut others_initialized = false;
        for mi in 0..nm {
            let dm = dms[mi];
            let k = ks[mi];
            let k2 = k2s[mi];
            if k * max_span < 0.5 {
                if dm == 1 {
                    e[0] *= z[0];
                } else if dm == 2 {
                    e[0] *= z[0] * z[0];
                } else {
                    e[0] *= cpow_u(z[0], dm);
                }
                let val = eval_taylor_horner(e[0], k, k2, &c_full);
                out[mi] += scale * val;
            } else {
                if !others_initialized {
                    let m = modes[mi];
                    e[0] = cpow_u(z[0], m);
                    for a in 1..cnt {
                        e[a] = cpow_u(z[a], m);
                    }
                    others_initialized = true;
                } else if dm == 1 {
                    for a in 0..cnt {
                        e[a] *= z[a];
                    }
                } else if dm == 2 {
                    for a in 0..cnt {
                        let z2 = z[a] * z[a];
                        e[a] *= z2;
                    }
                } else {
                    for a in 0..cnt {
                        e[a] *= cpow_u(z[a], dm);
                    }
                }
                let inv_k = inv_ks[mi];
                let mut table = [[Complex64::new(0.0, 0.0); 4]; 4];
                for i in 0..cnt {
                    table[0][i] = e[i];
                }
                for level in 1..cnt {
                    for i in 0..(cnt - level) {
                        let span = k * dx[i][i + level];
                        table[level][i] = if span >= 0.5 {
                            let d_val = table[level - 1][i + 1] - table[level - 1][i];
                            let inv_span = inv_k * inv_dx[i][i + level];
                            Complex64::new(-d_val.im * inv_span, d_val.re * inv_span)
                        } else {
                            let c_sub = taylor_coeffs_first_axis(&dx[i], i, level);
                            eval_taylor_horner(e[i], k, k2, &c_sub)
                        };
                    }
                }
                out[mi] += scale * table[cnt - 1][0];
            }
        }
    };
    if sub_stride == 1 {
        for_each_simplex(grid, visit_simplex);
    } else {
        let d = grid.dim;
        let n_sub = grid.n / sub_stride;
        let cell = (grid.l / n_sub as f64).powi(d as i32);
        const TETS: [[usize; 4]; 6] = [
            [0, 1, 3, 7],
            [0, 1, 5, 7],
            [0, 2, 3, 7],
            [0, 2, 6, 7],
            [0, 4, 5, 7],
            [0, 4, 6, 7],
        ];
        let k_end = if d == 3 { n_sub } else { 1 };
        for i in 0..n_sub {
            for j in 0..n_sub {
                for k in 0..k_end {
                    let ijk = [i, j, k];
                    let mut vertices = [(0usize, [0.0; 3]); 8];
                    for b in 0..(1 << d) {
                        let mut q = [0usize; 3];
                        let mut lift = [0.0; 3];
                        for a in 0..d {
                            let step = ijk[a] + ((b >> a) & 1);
                            if step >= n_sub {
                                q[a] = 0;
                                lift[a] = grid.l;
                            } else {
                                q[a] = step * sub_stride;
                            }
                        }
                        vertices[b] = (grid.ravel(q), lift);
                    }
                    if d == 2 {
                        visit_simplex(&[vertices[0], vertices[1], vertices[3]], cell / 2.0);
                        visit_simplex(&[vertices[0], vertices[3], vertices[2]], cell / 2.0);
                    } else {
                        for t in TETS {
                            visit_simplex(
                                &[
                                    vertices[t[0]],
                                    vertices[t[1]],
                                    vertices[t[2]],
                                    vertices[t[3]],
                                ],
                                cell / 6.0,
                            );
                        }
                    }
                }
            }
        }
    }
    out
}

/// Evaluate exact simplicial redshift-space distortions (RSD), velocity-divergence
/// and velocity-gradient Gaussian simplex packets, anisotropic power spectrum
/// multipoles (`P0`, `P2`, line-of-sight `P_||`, transverse `P_perp`), and
/// direct redshift-space simplex PDFs and signed volume distributions.
pub fn redshift_space_analysis(
    grid: &Grid,
    pos: &[f64],
    vel: &[f64],
    a: f64,
    f_growth: f64,
    e_hubble: f64,
    box_size: f64,
    mesh: usize,
    smoothing: f64,
) -> serde_json::Value {
    assert_eq!(pos.len(), grid.size * grid.dim);
    assert_eq!(vel.len(), grid.size * grid.dim);
    let d = grid.dim;
    let los_axis = d - 1;
    let v_box_kms = 100.0 * a * e_hubble * box_size;

    // For very large grids, extract a connected periodic sub-lattice so exact R3D
    // clipping, Gaussian packet deposition, and anisotropic FFTs remain interactive.
    let max_n = match d {
        1 => 512,
        2 => 128,
        _ => 32,
    };
    let stride = if grid.n > max_n {
        (1..=grid.n)
            .find(|&s| grid.n % s == 0 && grid.n / s <= max_n)
            .unwrap_or(1)
    } else {
        1
    };
    let n_rsd = grid.n / stride;
    let g_rsd = Grid::new(d, n_rsd, grid.l);
    let mesh_rsd = match d {
        1 => mesh.min(512),
        2 => mesh.min(128),
        _ => mesh.min(32),
    };
    let eg_rsd = Grid::new(d, mesh_rsd, grid.l);

    let mut pos_r = vec![0.0_f64; g_rsd.size * d];
    let mut vel_r = vec![0.0_f64; g_rsd.size * d];
    let mut pos_s = vec![0.0_f64; g_rsd.size * d];
    let mut u_los_sq_sum = 0.0_f64;

    for id_sub in 0..g_rsd.size {
        let ijk_sub = g_rsd.unravel(id_sub);
        let ijk_full = [ijk_sub[0] * stride, ijk_sub[1] * stride, ijk_sub[2] * stride];
        let id_full = grid.ravel(ijk_full);
        for axis in 0..d {
            let x_val = pos[id_full * d + axis];
            let v_val = vel[id_full * d + axis];
            pos_r[id_sub * d + axis] = x_val;
            vel_r[id_sub * d + axis] = v_val;
            pos_s[id_sub * d + axis] = if axis == los_axis {
                let u_los = a * v_val;
                u_los_sq_sum += u_los * u_los;
                x_val + u_los
            } else {
                x_val
            };
        }
    }
    let rms_u_los_box = (u_los_sq_sum / g_rsd.size as f64).sqrt();
    let rms_u_los_kms = rms_u_los_box * v_box_kms;

    // 1. Exact conservative R3D simplex deposit for real space and redshift space.
    let (rho_real_raw, rho_rsd_sheet_raw) = if d == 1 {
        (
            interval_density(&g_rsd, &pos_r, mesh_rsd),
            interval_density(&g_rsd, &pos_s, mesh_rsd),
        )
    } else if d == 2 {
        let pr_f32: Vec<f32> = pos_r.iter().map(|&v| v as f32).collect();
        let ps_f32: Vec<f32> = pos_s.iter().map(|&v| v as f32).collect();
        (
            crate::sheet::sheet_density_2d_exact(&g_rsd, &pr_f32, None, mesh_rsd),
            crate::sheet::sheet_density_2d_exact(&g_rsd, &ps_f32, None, mesh_rsd),
        )
    } else {
        let pr_f32: Vec<f32> = pos_r.iter().map(|&v| v as f32).collect();
        let ps_f32: Vec<f32> = pos_s.iter().map(|&v| v as f32).collect();
        (
            crate::sheet::sheet_density_3d_exact(&g_rsd, &pr_f32, None, mesh_rsd),
            crate::sheet::sheet_density_3d_exact(&g_rsd, &ps_f32, None, mesh_rsd),
        )
    };

    // 2. Simplex pass: exact velocity divergence, velocity-gradient Gaussian packets,
    // Monte-Carlo sampled Gaussian realization, and signed Jacobian distributions.
    let q_det = g_rsd.dx().powi(d as i32);
    let dx_mesh = eg_rsd.dx();
    let cell_vol = dx_mesh.powi(d as i32);
    let mut rho_rsd_gauss_raw = vec![0.0_f32; eg_rsd.size];
    let mut sample_positions = Vec::with_capacity(g_rsd.size * 4 * d);
    let mut vel_conv_map = vec![0.0_f32; eg_rsd.size];
    let mut vel_conv_wt = vec![0.0_f32; eg_rsd.size];
    let mut vel_width_map = vec![0.0_f32; eg_rsd.size];

    let mut min_j_real = f64::INFINITY;
    let mut min_j_rsd = f64::INFINITY;
    let mut inv_mass_real = 0.0_f64;
    let mut inv_mass_rsd = 0.0_f64;
    let mut inv_mass_rsd_only = 0.0_f64;
    let mut sigma_u_sq_sum = 0.0_f64;
    let mut max_sigma_u = 0.0_f64;
    let mut total_mass = 0.0_f64;

    const J_BINS: usize = 72;
    let j_lo = -1.2_f64;
    let j_hi = 2.8_f64;
    let dj = (j_hi - j_lo) / J_BINS as f64;
    let mut j_hist_real = vec![0.0_f64; J_BINS];
    let mut j_hist_rsd = vec![0.0_f64; J_BINS];

    let mut vertex_j_real_sum = vec![0.0_f64; g_rsd.size];
    let mut vertex_j_rsd_sum = vec![0.0_f64; g_rsd.size];
    let mut vertex_j_cnt = vec![0_u32; g_rsd.size];
    let mut simplex_idx = 0_u64;

    let next_u01 = |state: &mut u64| -> f64 {
        *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = *state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z = z ^ (z >> 31);
        ((z >> 11) as f64) * (1.0 / ((1u64 << 53) as f64))
    };

    for_each_simplex(&g_rsd, |verts, mass| {
        simplex_idx += 1;
        total_mass += mass;
        let n_v = verts.len();
        let mut x = [[0.0_f64; 3]; 4];
        let mut s = [[0.0_f64; 3]; 4];
        let mut u = [[0.0_f64; 3]; 4];
        let mut x_bar = [0.0_f64; 3];
        let mut s_bar = [0.0_f64; 3];
        let mut u_bar = [0.0_f64; 3];

        for (vi, &(id, lift)) in verts.iter().enumerate() {
            for axis in 0..d {
                let xv = pos_r[id * d + axis] + lift[axis];
                let uv = a * vel_r[id * d + axis];
                let sv = if axis == los_axis { xv + uv } else { xv };
                x[vi][axis] = xv;
                u[vi][axis] = uv;
                s[vi][axis] = sv;
                x_bar[axis] += xv / n_v as f64;
                u_bar[axis] += uv / n_v as f64;
                s_bar[axis] += sv / n_v as f64;
            }
        }

        let j_r = det(&x[..n_v], d) / q_det;
        let j_s = det(&s[..n_v], d) / q_det;
        if j_r < min_j_real {
            min_j_real = j_r;
        }
        if j_s < min_j_rsd {
            min_j_rsd = j_s;
        }
        if j_r <= 0.0 {
            inv_mass_real += mass;
        }
        if j_s <= 0.0 {
            inv_mass_rsd += mass;
            if j_r > 0.0 {
                inv_mass_rsd_only += mass;
            }
        }

        for &(id, _) in verts {
            vertex_j_real_sum[id] += j_r;
            vertex_j_rsd_sum[id] += j_s;
            vertex_j_cnt[id] += 1;
        }

        let b_r = ((j_r - j_lo) / dj).floor() as isize;
        if b_r >= 0 && (b_r as usize) < J_BINS {
            j_hist_real[b_r as usize] += mass;
        }
        let b_s = ((j_s - j_lo) / dj).floor() as isize;
        if b_s >= 0 && (b_s as usize) < J_BINS {
            j_hist_rsd[b_s as usize] += mass;
        }

        // Exact cofactor expansion for volume-weighted velocity divergence J_T * (div_x u)_T.
        let mut j_div_u = 0.0_f64;
        for repl_axis in 0..d {
            let mut x_mod = x;
            for vi in 0..n_v {
                x_mod[vi][repl_axis] = u[vi][repl_axis];
            }
            j_div_u += det(&x_mod[..n_v], d) / q_det;
        }
        let theta_t = j_div_u / j_r.abs().max(0.05);
        let conv_t = -theta_t;

        // Spatial inertia and velocity-gradient variance across the simplex vertices.
        let mut var_x = [0.0_f64; 3];
        let mut var_u = [0.0_f64; 3];
        let mut frob_u_sq = 0.0_f64;
        for vi in 0..n_v {
            for axis in 0..d {
                let dx_a = x[vi][axis] - x_bar[axis];
                let du_a = u[vi][axis] - u_bar[axis];
                var_x[axis] += (dx_a * dx_a) / ((n_v * (n_v + 1)) as f64);
                var_u[axis] += (du_a * du_a) / (n_v as f64);
                frob_u_sq += (du_a * du_a) / ((d * n_v) as f64);
            }
        }
        let conv_boost = g_rsd.dx() * conv_t.max(0.0) * 0.35;
        let sigma_u_los = (var_u[los_axis] + frob_u_sq + conv_boost * conv_boost).sqrt();
        sigma_u_sq_sum += mass * sigma_u_los * sigma_u_los;
        if sigma_u_los > max_sigma_u {
            max_sigma_u = sigma_u_los;
        }

        // Record velocity convergence and dispersion width at the real-space centroid cell.
        let mut c_ijk = [0usize; 3];
        for axis in 0..d {
            let u_coord = (x_bar[axis].rem_euclid(eg_rsd.l) / dx_mesh).floor() as usize;
            c_ijk[axis] = u_coord.min(mesh_rsd - 1);
        }
        let c_idx = eg_rsd.ravel(c_ijk);
        vel_conv_map[c_idx] += (mass as f32) * (conv_t as f32);
        vel_width_map[c_idx] += (mass as f32) * ((sigma_u_los * v_box_kms) as f32);
        vel_conv_wt[c_idx] += mass as f32;

        // Fitted d-dimensional Gaussian packet parameters in redshift space.
        let mut sigma_s = [0.0_f64; 3];
        for axis in 0..d {
            let base_sq = var_x[axis] + (0.45 * dx_mesh).powi(2);
            sigma_s[axis] = if axis == los_axis {
                (base_sq + 2.25 * sigma_u_los * sigma_u_los)
                    .sqrt()
                    .clamp(0.42 * dx_mesh, 0.22 * eg_rsd.l)
            } else {
                base_sq.sqrt().clamp(0.42 * dx_mesh, 0.08 * eg_rsd.l)
            };
        }

        // Deposit normalized separable Gaussian packet onto eg_rsd.
        let mut w_axes: [Vec<(usize, f64)>; 3] = [Vec::new(), Vec::new(), Vec::new()];
        for axis in 0..d {
            let sig = sigma_s[axis];
            let rad = ((2.75 * sig / dx_mesh).ceil() as isize).clamp(1, (mesh_rsd as isize / 2).max(1));
            let center_cell = (s_bar[axis] / dx_mesh).floor() as isize;
            let mut sum_w = 0.0_f64;
            let mut list = Vec::with_capacity((2 * rad + 1) as usize);
            for off in -rad..=rad {
                let cell_i = center_cell + off;
                let cell_center = (cell_i as f64 + 0.5) * dx_mesh;
                let dist = cell_center - s_bar[axis];
                let w = (-0.5 * (dist / sig).powi(2)).exp();
                if w > 1e-5 {
                    let wrapped = cell_i.rem_euclid(mesh_rsd as isize) as usize;
                    list.push((wrapped, w));
                    sum_w += w;
                }
            }
            if sum_w > 0.0 {
                for item in &mut list {
                    item.1 /= sum_w;
                }
            } else {
                let wrapped = center_cell.rem_euclid(mesh_rsd as isize) as usize;
                list.push((wrapped, 1.0));
            }
            w_axes[axis] = list;
        }
        let m_norm = (mass / cell_vol) as f32;
        if d == 1 {
            for &(i0, w0) in &w_axes[0] {
                rho_rsd_gauss_raw[i0] += m_norm * (w0 as f32);
            }
        } else if d == 2 {
            for &(i0, w0) in &w_axes[0] {
                for &(i1, w1) in &w_axes[1] {
                    rho_rsd_gauss_raw[i0 * mesh_rsd + i1] += m_norm * ((w0 * w1) as f32);
                }
            }
        } else {
            for &(i0, w0) in &w_axes[0] {
                for &(i1, w1) in &w_axes[1] {
                    for &(i2, w2) in &w_axes[2] {
                        rho_rsd_gauss_raw[(i0 * mesh_rsd + i1) * mesh_rsd + i2] +=
                            m_norm * ((w0 * w1 * w2) as f32);
                    }
                }
            }
        }

        // Draw 2 deterministic Gaussian samples from N(s_bar, diag(sigma_s^2)) for this simplex.
        let mut rng_state = simplex_idx.wrapping_mul(0x517C_C1B7_2722_0A95) ^ 0xA5A5_5A5A_F0F0_0F0F;
        for _ in 0..2 {
            for axis in 0..d {
                let u1 = next_u01(&mut rng_state).max(1e-12);
                let u2 = next_u01(&mut rng_state);
                let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
                let samp = (s_bar[axis] + sigma_s[axis] * z).rem_euclid(eg_rsd.l);
                sample_positions.push(samp as f32);
            }
        }
    });

    for idx in 0..eg_rsd.size {
        if vel_conv_wt[idx] > 0.0 {
            vel_conv_map[idx] /= vel_conv_wt[idx];
            vel_width_map[idx] /= vel_conv_wt[idx];
        }
    }

    let n_samples = sample_positions.len() / d;
    let rho_rsd_sampled_raw = crate::sheet::cic_density(
        d,
        &sample_positions,
        n_samples.max(1),
        eg_rsd.l,
        mesh_rsd,
    );

    let rho_real_smooth = smooth_density(&eg_rsd, &rho_real_raw, smoothing);
    let rho_rsd_sheet_smooth = smooth_density(&eg_rsd, &rho_rsd_sheet_raw, smoothing);
    let rho_rsd_gauss_smooth = smooth_density(&eg_rsd, &rho_rsd_gauss_raw, smoothing);
    let rho_rsd_sampled_smooth = smooth_density(&eg_rsd, &rho_rsd_sampled_raw, smoothing);

    // 3. Anisotropic Fourier multipoles: Monopole P0(k), Quadrupole P2(k),
    // transverse P_perp(k), line-of-sight P_||(k), and exact discrete-lattice Kaiser expectation.
    let mut fft = FftEngine::new(&eg_rsd);
    let hat_real = fft.forward_real_f32(&rho_real_raw);
    let hat_sheet = fft.forward_real_f32(&rho_rsd_sheet_raw);
    let hat_gauss = fft.forward_real_f32(&rho_rsd_gauss_raw);
    let hat_samp = fft.forward_real_f32(&rho_rsd_sampled_raw);

    let nbins = 28usize;
    let edges = crate::spectra::bin_edges(&eg_rsd, nbins);
    let lk0 = edges[0].ln();
    let dlk = (edges[nbins] / edges[0]).ln() / nbins as f64;
    let norm = eg_rsd.l.powi(d as i32) / (eg_rsd.size as f64).powi(2);

    let mut k_sum = vec![0.0_f64; nbins];
    let mut cnt = vec![0.0_f64; nbins];
    let mut p0_real = vec![0.0_f64; nbins];
    let mut p0_sheet = vec![0.0_f64; nbins];
    let mut p0_gauss = vec![0.0_f64; nbins];
    let mut p0_samp = vec![0.0_f64; nbins];
    let mut p0_kaiser = vec![0.0_f64; nbins];
    let mut p2_sheet = vec![0.0_f64; nbins];
    let mut p2_gauss = vec![0.0_f64; nbins];
    let mut p2_kaiser = vec![0.0_f64; nbins];
    let mut p_los_sheet = vec![0.0_f64; nbins];
    let mut p_los_gauss = vec![0.0_f64; nbins];
    let mut p_los_kaiser = vec![0.0_f64; nbins];
    let mut cnt_los = vec![0.0_f64; nbins];
    let mut p_perp_sheet = vec![0.0_f64; nbins];
    let mut p_perp_gauss = vec![0.0_f64; nbins];
    let mut p_perp_real = vec![0.0_f64; nbins];
    let mut cnt_perp = vec![0.0_f64; nbins];

    for idx in 1..eg_rsd.size {
        let ijk = eg_rsd.unravel(idx);
        let mut k2 = 0.0_f64;
        let mut k_los = 0.0_f64;
        for axis in 0..d {
            let ka = eg_rsd.k1(ijk[axis]);
            k2 += ka * ka;
            if axis == los_axis {
                k_los = ka;
            }
        }
        let k = k2.sqrt();
        if k < edges[0] || k >= edges[nbins] {
            continue;
        }
        let bi = (((k.ln() - lk0) / dlk).floor() as usize).min(nbins - 1);
        let mu2 = if k2 > 0.0 { (k_los * k_los) / k2 } else { 0.0 };
        let leg2 = if d == 2 {
            2.0 * (2.0 * mu2 - 1.0)
        } else {
            2.5 * (3.0 * mu2 - 1.0)
        };
        let kaiser_boost = (1.0 + f_growth * mu2).powi(2);

        let pr = hat_real[idx].norm_sqr() * norm;
        let ps = hat_sheet[idx].norm_sqr() * norm;
        let pg = hat_gauss[idx].norm_sqr() * norm;
        let pmc = hat_samp[idx].norm_sqr() * norm;
        let pk = pr * kaiser_boost;

        k_sum[bi] += k;
        cnt[bi] += 1.0;
        p0_real[bi] += pr;
        p0_sheet[bi] += ps;
        p0_gauss[bi] += pg;
        p0_samp[bi] += pmc;
        p0_kaiser[bi] += pk;
        p2_sheet[bi] += leg2 * ps;
        p2_gauss[bi] += leg2 * pg;
        p2_kaiser[bi] += leg2 * pk;

        if mu2 >= 0.5 {
            p_los_sheet[bi] += ps;
            p_los_gauss[bi] += pg;
            p_los_kaiser[bi] += pk;
            cnt_los[bi] += 1.0;
        } else {
            p_perp_sheet[bi] += ps;
            p_perp_gauss[bi] += pg;
            p_perp_real[bi] += pr;
            cnt_perp[bi] += 1.0;
        }
    }

    let k_scale = 1.0 / box_size.max(1e-12);
    let p_scale = box_size.powi(d as i32);
    let mut k_out = Vec::new();
    let mut p0_real_out = Vec::new();
    let mut p0_sheet_out = Vec::new();
    let mut p0_gauss_out = Vec::new();
    let mut p0_samp_out = Vec::new();
    let mut p0_kaiser_out = Vec::new();
    let mut p2_sheet_out = Vec::new();
    let mut p2_gauss_out = Vec::new();
    let mut p2_kaiser_out = Vec::new();
    let mut p_los_sheet_out = Vec::new();
    let mut p_los_gauss_out = Vec::new();
    let mut p_los_kaiser_out = Vec::new();
    let mut p_perp_sheet_out = Vec::new();
    let mut p_perp_gauss_out = Vec::new();
    let mut p_perp_real_out = Vec::new();

    for i in 0..nbins {
        if cnt[i] > 0.0 {
            let c = cnt[i];
            let pr_m = (p0_real[i] / c) * p_scale;
            let ps_m = (p0_sheet[i] / c) * p_scale;
            let pg_m = (p0_gauss[i] / c) * p_scale;
            let pk_m = (p0_kaiser[i] / c) * p_scale;
            k_out.push((k_sum[i] / c) * k_scale);
            p0_real_out.push(pr_m);
            p0_sheet_out.push(ps_m);
            p0_gauss_out.push(pg_m);
            p0_samp_out.push((p0_samp[i] / c) * p_scale);
            p0_kaiser_out.push(pk_m);
            p2_sheet_out.push((p2_sheet[i] / c) * p_scale);
            p2_gauss_out.push((p2_gauss[i] / c) * p_scale);
            p2_kaiser_out.push((p2_kaiser[i] / c) * p_scale);
            let cl = cnt_los[i];
            p_los_sheet_out.push(if cl > 0.0 { (p_los_sheet[i] / cl) * p_scale } else { ps_m });
            p_los_gauss_out.push(if cl > 0.0 { (p_los_gauss[i] / cl) * p_scale } else { pg_m });
            p_los_kaiser_out.push(if cl > 0.0 { (p_los_kaiser[i] / cl) * p_scale } else { (1.0 + f_growth).powi(2) * pr_m });
            let cp = cnt_perp[i];
            p_perp_sheet_out.push(if cp > 0.0 { (p_perp_sheet[i] / cp) * p_scale } else { pr_m });
            p_perp_gauss_out.push(if cp > 0.0 { (p_perp_gauss[i] / cp) * p_scale } else { pr_m });
            p_perp_real_out.push(if cp > 0.0 { (p_perp_real[i] / cp) * p_scale } else { pr_m });
        }
    }

    // 4. Direct simplex PDFs for real-space sheet and redshifted sheet, plus Gaussian grid PDF.
    let probabilities = [0.001, 0.01, 0.1, 0.5, 0.9, 0.99, 0.999];
    let (lr_real, vol_real, mass_real, _, var_real, dy_real, sum_real) =
        simplex_density_pdf_and_summary(&g_rsd, &pos_r, None, 80, &probabilities);
    let (lr_rsd, vol_rsd, mass_rsd, _, var_rsd, dy_rsd, sum_rsd) =
        simplex_density_pdf_and_summary(&g_rsd, &pos_s, None, 80, &probabilities);
    let (lr_gauss, vol_gauss, mass_gauss, _, var_gauss) =
        density_pdf(&rho_rsd_gauss_smooth, &rho_rsd_gauss_raw, 80);
    let sum_gauss = density_summary(&rho_rsd_gauss_smooth, &rho_rsd_gauss_raw, &probabilities);
    let dy_gauss = 3.6 / 80.0;

    let j_axis: Vec<f64> = (0..J_BINS).map(|i| j_lo + (i as f64 + 0.5) * dj).collect();
    let tot_m = total_mass.max(1e-30);
    for i in 0..J_BINS {
        j_hist_real[i] /= tot_m * dj;
        j_hist_rsd[i] /= tot_m * dj;
    }

    // 5. Wireframe subset (capped at n_wire <= 48 in 2D, 16 in 3D, 128 in 1D) for sheet visualization.
    let max_wire_n = match d {
        1 => 128,
        2 => 48,
        _ => 16,
    };
    let wire_stride = if g_rsd.n > max_wire_n {
        (g_rsd.n / max_wire_n).max(1)
    } else {
        1
    };
    let n_wire = g_rsd.n / wire_stride;
    let g_wire = Grid::new(d, n_wire, g_rsd.l);
    let mut x_wire_real = Vec::with_capacity(g_wire.size * d);
    let mut x_wire_rsd = Vec::with_capacity(g_wire.size * d);
    let mut v_wire_los = Vec::with_capacity(g_wire.size);
    let mut j_wire_real = Vec::with_capacity(g_wire.size);
    let mut j_wire_rsd = Vec::with_capacity(g_wire.size);

    for wid in 0..g_wire.size {
        let ijk_w = g_wire.unravel(wid);
        let ijk_r = [
            ijk_w[0] * wire_stride,
            ijk_w[1] * wire_stride,
            ijk_w[2] * wire_stride,
        ];
        let rid = g_rsd.ravel(ijk_r);
        for axis in 0..d {
            x_wire_real.push(pos_r[rid * d + axis] as f32);
            x_wire_rsd.push(pos_s[rid * d + axis] as f32);
        }
        v_wire_los.push((a * vel_r[rid * d + los_axis]) as f32);
        let cnt_v = vertex_j_cnt[rid].max(1) as f64;
        j_wire_real.push((vertex_j_real_sum[rid] / cnt_v) as f32);
        j_wire_rsd.push((vertex_j_rsd_sum[rid] / cnt_v) as f32);
    }

    let max_sample_pts = 1200usize;
    let sample_step = (n_samples / max_sample_pts).max(1);
    let mut sample_cloud = Vec::new();
    for si in (0..n_samples).step_by(sample_step) {
        for axis in 0..d {
            sample_cloud.push(sample_positions[si * d + axis]);
        }
    }

    let kaiser_mono_factor = match d {
        1 => (1.0 + f_growth).powi(2),
        2 => 1.0 + f_growth + 0.375 * f_growth * f_growth,
        _ => 1.0 + (2.0 / 3.0) * f_growth + 0.2 * f_growth * f_growth,
    };
    let kaiser_quad_factor = match d {
        1 => 0.0,
        2 => f_growth + 0.5 * f_growth * f_growth,
        _ => (4.0 / 3.0) * f_growth + (4.0 / 7.0) * f_growth * f_growth,
    };

    let rms_sigma_u_box = (sigma_u_sq_sum / tot_m).sqrt();

    serde_json::json!({
        "mesh": mesh_rsd,
        "nSimplexGrid": n_rsd,
        "losAxis": los_axis,
        "summary": {
            "growthRateF": f_growth,
            "hubbleScaleKmS": v_box_kms,
            "kaiserLosFactor": (1.0 + f_growth).powi(2),
            "kaiserMonopoleFactor": kaiser_mono_factor,
            "kaiserQuadrupoleFactor": kaiser_quad_factor,
            "minRealSimplexJacobian": min_j_real,
            "minRsdSimplexJacobian": min_j_rsd,
            "realInvertedMassFraction": inv_mass_real / tot_m,
            "rsdInvertedMassFraction": inv_mass_rsd / tot_m,
            "rsdPreCrossingInvertedMass": inv_mass_rsd_only / tot_m,
            "rmsVelocityLosBox": rms_u_los_box,
            "rmsVelocityLosKmS": rms_u_los_kms,
            "rmsSimplexVelocityWidthBox": rms_sigma_u_box,
            "rmsSimplexVelocityWidthKmS": rms_sigma_u_box * v_box_kms,
            "maxSimplexVelocityWidthKmS": max_sigma_u * v_box_kms,
        },
        "maps": {
            "realRho": rho_real_smooth,
            "rsdSheetRho": rho_rsd_sheet_smooth,
            "rsdGaussRho": rho_rsd_gauss_smooth,
            "rsdSampledRho": rho_rsd_sampled_smooth,
            "velocityConvergence": vel_conv_map,
            "simplexVelocityWidthKmS": vel_width_map,
        },
        "spectra": {
            "k": k_out,
            "realMonopole": p0_real_out,
            "rsdSheetMonopole": p0_sheet_out,
            "rsdGaussMonopole": p0_gauss_out,
            "rsdSampledMonopole": p0_samp_out,
            "kaiserMonopole": p0_kaiser_out,
            "rsdSheetQuadrupole": p2_sheet_out,
            "rsdGaussQuadrupole": p2_gauss_out,
            "kaiserQuadrupole": p2_kaiser_out,
            "rsdSheetLos": p_los_sheet_out,
            "rsdGaussLos": p_los_gauss_out,
            "kaiserLos": p_los_kaiser_out,
            "rsdSheetTransverse": p_perp_sheet_out,
            "rsdGaussTransverse": p_perp_gauss_out,
            "realTransverse": p_perp_real_out,
        },
        "pdf": {
            "real": {
                "logrho": lr_real,
                "dy": dy_real,
                "volume": vol_real,
                "mass": mass_real,
                "variance": var_real,
                "capturedVolume": vol_real.iter().sum::<f64>() * dy_real,
                "capturedMass": mass_real.iter().sum::<f64>() * dy_real,
                "summary": sum_real,
            },
            "rsdSheet": {
                "logrho": lr_rsd,
                "dy": dy_rsd,
                "volume": vol_rsd,
                "mass": mass_rsd,
                "variance": var_rsd,
                "capturedVolume": vol_rsd.iter().sum::<f64>() * dy_rsd,
                "capturedMass": mass_rsd.iter().sum::<f64>() * dy_rsd,
                "summary": sum_rsd,
            },
            "rsdGauss": {
                "logrho": lr_gauss,
                "dy": dy_gauss,
                "volume": vol_gauss,
                "mass": mass_gauss,
                "variance": var_gauss,
                "capturedVolume": vol_gauss.iter().sum::<f64>() * dy_gauss,
                "capturedMass": mass_gauss.iter().sum::<f64>() * dy_gauss,
                "summary": sum_gauss,
            },
            "jacobian": {
                "j": j_axis,
                "dj": dj,
                "realMassPdf": j_hist_real,
                "rsdMassPdf": j_hist_rsd,
            }
        },
        "wireframe": {
            "n": n_wire,
            "xReal": x_wire_real,
            "xRsd": x_wire_rsd,
            "uLos": v_wire_los,
            "jReal": j_wire_real,
            "jRsd": j_wire_rsd,
            "gaussSamples": sample_cloud,
        }
    })
}
