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
    serde_json::json!({"mean":mean,"variance":variance,"S3":s3,"S4":s4,
        "probabilities":probabilities,
        "volumeQuantiles":probabilities.iter().map(|&p|quantile(p,false)).collect::<Vec<_>>(),
        "massQuantiles":probabilities.iter().map(|&p|quantile(p,true)).collect::<Vec<_>>()})
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
/// k = m * k_f * (dir[0], dir[1], dir[2]) for a sorted slice of positive integer
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
    let sub_stride = if nm > 6 && max_sub_n > 0 && grid.n > max_sub_n && grid.n % max_sub_n == 0 {
        grid.n / max_sub_n
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
