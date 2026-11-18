//! Dual-smoothing Lagrangian UV response and the leading sound-speed counterterm.
//!
//! Two Gaussian initial filters of the same seed share long-wavelength phases. The
//! difference of their nonlinear displacements at low *k* is the integrated response
//! of the modes between the two cutoffs. Following Mirbabayi & Zaldarriaga (2016),
//! the stress \(\tau\) sources a counterterm displacement; the leading operator used
//! here is \(\widehat{\boldsymbol\psi}_{\rm ct}=-\alpha k^2\widehat{\boldsymbol\psi}_{\rm IR}\)
//! (unit-box \(k\) scaled so \(\alpha\equiv c_s^2\) is in \((h^{-1}\mathrm{Mpc})^2\)).
//! Residual vertex scatter after the deterministic fit is summarized as a stochastic
//! displacement power that can seed an ensemble of consistent noise realizations.
use crate::{
    cosmological::LinearSpectrum,
    fft::FftEngine,
    grid::Grid,
    laboratory::{self, LabRequest},
    observables, rng::Rng, sheet, spectra, Cosmo,
};
use num_complex::Complex64 as C64;
use serde_json::{json, Value};

/// Least-squares fit of ΔP(k) ≈ −2 cₛ² k² P_lin(k) on bins with 0 < k ≤ k_max.
#[derive(Clone, Debug)]
pub struct SoundSpeedFit {
    /// Counterterm coefficient cₛ² in (Mpc/h)² under the convention
    /// P_UV(k) ≈ P_IR(k) − 2 cₛ² k² P_lin^IR(k).
    pub cs2: f64,
    /// Number of isotropic bins entering the fit.
    pub n_bins: usize,
    /// Maximum physical wavenumber retained (h/Mpc).
    pub k_max: f64,
    /// Weighted residual RMS of ΔP + 2 cₛ² k² P_lin over the fit window.
    pub residual_rms: f64,
    /// Weighted mean of (ΔP / P_lin) over the fit window.
    pub mean_relative_delta: f64,
}

/// Fit cₛ² from paired spectra. All arrays are physical: k in h/Mpc, P in (Mpc/h)^d.
pub fn fit_sound_speed(
    k: &[f64],
    p_uv: &[f64],
    p_ir: &[f64],
    p_lin: &[f64],
    nmodes: &[f64],
    k_max: f64,
) -> Option<SoundSpeedFit> {
    let n = k.len().min(p_uv.len()).min(p_ir.len()).min(p_lin.len());
    if n == 0 || !(k_max > 0.) {
        return None;
    }
    let mut sw = 0.0;
    let mut sxx = 0.0;
    let mut sxy = 0.0;
    let mut s_rel = 0.0;
    let mut n_bins = 0usize;
    for i in 0..n {
        let ki = k[i];
        let pl = p_lin[i];
        if !(ki > 0. && ki <= k_max && pl > 0. && p_uv[i].is_finite() && p_ir[i].is_finite()) {
            continue;
        }
        let w = if i < nmodes.len() && nmodes[i] > 0. {
            nmodes[i]
        } else {
            1.0
        };
        // Model: ΔP = P_UV − P_IR = −2 cₛ² k² P_lin  ⇒  y = −ΔP/(2 P_lin) = cₛ² k²
        let delta = p_uv[i] - p_ir[i];
        let y = -delta / (2.0 * pl);
        let x = ki * ki;
        sw += w;
        sxx += w * x * x;
        sxy += w * x * y;
        s_rel += w * (delta / pl);
        n_bins += 1;
    }
    if n_bins < 2 || !(sxx > 0.) {
        return None;
    }
    let cs2 = sxy / sxx;
    let mut rss = 0.0;
    for i in 0..n {
        let ki = k[i];
        let pl = p_lin[i];
        if !(ki > 0. && ki <= k_max && pl > 0.) {
            continue;
        }
        let w = if i < nmodes.len() && nmodes[i] > 0. {
            nmodes[i]
        } else {
            1.0
        };
        let pred = -2.0 * cs2 * ki * ki * pl;
        let resid = (p_uv[i] - p_ir[i]) - pred;
        rss += w * resid * resid;
    }
    Some(SoundSpeedFit {
        cs2,
        n_bins,
        k_max,
        residual_rms: (rss / sw.max(1e-300)).sqrt(),
        mean_relative_delta: s_rel / sw.max(1e-300),
    })
}

/// Continuum linear RMS of the Gaussian-filtered density (no extra top-hat), at growth
/// ratio `growth_ratio` relative to the z=0 σ₈ normalization.
pub fn continuum_filtered_sigma(r: &LabRequest, initial_smoothing: f64, growth_ratio: f64) -> f64 {
    let spectrum =
        LinearSpectrum::new(r.omega_m, r.omega_b, r.h, r.ns, r.sigma8, r.transfer == "bao");
    spectrum
        .filtered_variance(r.dim, initial_smoothing, growth_ratio, 0.0, true)
        .max(0.0)
        .sqrt()
}

/// Find an IR Gaussian radius whose continuum linear σ is `sigma_ratio` times the UV σ.
pub fn match_sigma_ratio(
    r: &LabRequest,
    uv_smoothing: f64,
    sigma_ratio: f64,
    growth_ratio: f64,
) -> Option<f64> {
    if !(uv_smoothing >= 0. && sigma_ratio > 0. && sigma_ratio < 1.) {
        return None;
    }
    let sigma_uv = continuum_filtered_sigma(r, uv_smoothing, growth_ratio);
    if !(sigma_uv > 0.) {
        return None;
    }
    let target = sigma_uv * sigma_ratio;
    let mut lo = uv_smoothing;
    let mut hi = (uv_smoothing * 8.0).max(uv_smoothing + 1.0).min(200.0);
    if continuum_filtered_sigma(r, hi, growth_ratio) > target {
        for _ in 0..8 {
            hi = (hi * 1.5).min(200.0);
            if continuum_filtered_sigma(r, hi, growth_ratio) <= target {
                break;
            }
            if hi >= 199.0 {
                return None;
            }
        }
    }
    for _ in 0..48 {
        let mid = 0.5 * (lo + hi);
        let s = continuum_filtered_sigma(r, mid, growth_ratio);
        if s > target {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    Some(0.5 * (lo + hi))
}

fn simplex_det(v: &[[f64; 3]], d: usize) -> f64 {
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

/// Count and mass fractions of negatively oriented affine elements.
pub fn inverted_fractions(grid: &Grid, pos: &[f64]) -> (f64, f64) {
    let mut crossed_mass = 0.0_f64;
    let mut total_mass = 0.0_f64;
    let mut crossed_n = 0.0_f64;
    let mut total_n = 0.0_f64;
    observables::for_each_simplex(grid, |verts, mass| {
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
        let j = simplex_det(&x[..n], grid.dim) / simplex_det(&q[..n], grid.dim);
        total_mass += mass;
        total_n += 1.0;
        if j < 0.0 {
            crossed_mass += mass;
            crossed_n += 1.0;
        }
    });
    (
        crossed_n / total_n.max(1e-300),
        crossed_mass / total_mass.max(1e-300),
    )
}

fn density_p1(g: &Grid, x: &[f64], mesh: usize) -> Vec<f32> {
    let w = observables::vertex_inverse_jacobians(g, x);
    if g.dim == 1 {
        return observables::interval_density_p1(g, x, &w, mesh);
    }
    let pos: Vec<f32> = x.iter().map(|&v| v as f32).collect();
    if g.dim == 2 {
        sheet::sheet_density_2d_exact(g, &pos, Some(&w), mesh)
    } else {
        sheet::sheet_density_3d_exact(g, &pos, Some(&w), mesh)
    }
}

fn spectrum_physical(rho: &[f32], mesh_grid: &Grid, box_size: f64) -> spectra::Spectrum {
    let mut eng = FftEngine::new(mesh_grid);
    let contrast: Vec<f32> = rho.iter().map(|&v| v - 1.).collect();
    let mut sp = spectra::power_spectrum(
        mesh_grid,
        &mut eng,
        &contrast,
        laboratory::spectrum_bins(mesh_grid.n),
    );
    let vol = box_size.powi(mesh_grid.dim as i32);
    for k in &mut sp.k {
        *k /= box_size;
    }
    for p in &mut sp.p {
        *p *= vol;
    }
    for k in &mut sp.fine_k {
        *k /= box_size;
    }
    for p in &mut sp.fine_p {
        *p *= vol;
    }
    sp
}

fn realized_linear_physical(
    c: &Cosmo,
    eg: &Grid,
    d: f64,
    box_size: f64,
) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let nbins = laboratory::spectrum_bins(eg.n);
    let (k, p, n) = laboratory::realized_linear_power_bins(c, eg, d, nbins);
    let vol = box_size.powi(eg.dim as i32);
    (
        k.into_iter().map(|v| v / box_size).collect(),
        p.into_iter().map(|v| v * vol).collect(),
        n,
    )
}

/// Leading displacement counterterm in Fourier space: ψ_ct = −α k_phys² ψ.
/// `alpha` is cₛ² in (Mpc/h)²; positions/displacements are in unit-box coordinates.
pub fn apply_displacement_counterterm(
    grid: &Grid,
    eng: &mut FftEngine,
    psi: &[f64],
    alpha: f64,
    box_size: f64,
) -> Vec<f64> {
    assert_eq!(psi.len(), grid.size * grid.dim);
    let scale = alpha / box_size.powi(2); // −α k_phys² = −(α/L²) k_unit²
    let mut out = vec![0.0; psi.len()];
    for a in 0..grid.dim {
        let mut component = vec![0.0; grid.size];
        for id in 0..grid.size {
            component[id] = psi[id * grid.dim + a];
        }
        let mut hat = eng.forward_real(&component);
        for (idx, v) in hat.iter_mut().enumerate() {
            let k2 = grid.k2(idx);
            *v *= -scale * k2;
        }
        let real = eng.inverse_to_real(hat);
        for id in 0..grid.size {
            out[id * grid.dim + a] = real[id];
        }
    }
    out
}

fn displacement_from_positions(grid: &Grid, x: &[f64]) -> Vec<f64> {
    let mut psi = vec![0.0; x.len()];
    for id in 0..grid.size {
        let ijk = grid.unravel(id);
        for a in 0..grid.dim {
            let q = ijk[a] as f64 * grid.dx();
            // Minimal-image displacement in the unit box.
            let mut d = x[id * grid.dim + a] - q;
            if d > 0.5 {
                d -= 1.0;
            }
            if d < -0.5 {
                d += 1.0;
            }
            psi[id * grid.dim + a] = d;
        }
    }
    psi
}

fn vector_rms(v: &[f64]) -> f64 {
    if v.is_empty() {
        return 0.0;
    }
    (v.iter().map(|x| x * x).sum::<f64>() / v.len() as f64).sqrt()
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

/// Fit α ≡ cₛ² so Δψ = ψ_UV − ψ_IR ≈ −α k_phys² ψ_IR on modes with k_phys ≤ k_max.
#[derive(Clone, Debug)]
pub struct DisplacementSoundSpeedFit {
    pub cs2: f64,
    pub k_max: f64,
    pub n_modes: usize,
    /// RMS of ψ_UV − ψ_IR (unit box) before the counterterm.
    pub rms_before: f64,
    /// RMS of ψ_UV − (ψ_IR + ψ_ct) after the fit.
    pub rms_after: f64,
    /// Cosine similarity between Δψ and ψ_ct over all components.
    pub correlation: f64,
    /// Fraction of Δψ variance explained by ψ_ct.
    pub variance_explained: f64,
}

pub fn fit_displacement_sound_speed(
    grid: &Grid,
    eng: &mut FftEngine,
    psi_uv: &[f64],
    psi_ir: &[f64],
    box_size: f64,
    k_max: f64,
) -> Option<DisplacementSoundSpeedFit> {
    assert_eq!(psi_uv.len(), grid.size * grid.dim);
    assert_eq!(psi_ir.len(), psi_uv.len());
    let k_max_unit = k_max * box_size;
    let mut sxx = 0.0;
    let mut sxy = 0.0;
    let mut n_modes = 0usize;
    // Δψ ≈ −α k_phys² ψ_IR = −(α/L²) k_unit² ψ_IR  ⇒  y = −Δψ_hat, x = (k_unit²/L²) ψ_IR_hat
    for a in 0..grid.dim {
        let mut du = vec![0.0; grid.size];
        let mut di = vec![0.0; grid.size];
        for id in 0..grid.size {
            du[id] = psi_uv[id * grid.dim + a] - psi_ir[id * grid.dim + a];
            di[id] = psi_ir[id * grid.dim + a];
        }
        let hat_du = eng.forward_real(&du);
        let hat_di = eng.forward_real(&di);
        for idx in 1..grid.size {
            let k = grid.k2(idx).sqrt();
            if k > k_max_unit || k == 0.0 {
                continue;
            }
            let k2_phys = (k / box_size).powi(2);
            let x = hat_di[idx] * k2_phys;
            let y = -hat_du[idx];
            // Hermitian: weight positive half only via full sum of Re[x* conj y] with 1/2 for pairs.
            sxx += x.norm_sqr();
            sxy += (x.conj() * y).re;
            n_modes += 1;
        }
    }
    if n_modes < 4 || !(sxx > 0.) {
        return None;
    }
    let cs2 = sxy / sxx;
    let psi_ct = apply_displacement_counterterm(grid, eng, psi_ir, cs2, box_size);
    let mut delta = vec![0.0; psi_uv.len()];
    let mut resid = vec![0.0; psi_uv.len()];
    for i in 0..psi_uv.len() {
        delta[i] = psi_uv[i] - psi_ir[i];
        resid[i] = psi_uv[i] - (psi_ir[i] + psi_ct[i]);
    }
    let rms_before = vector_rms(&delta);
    let rms_after = vector_rms(&resid);
    let corr_den = vector_rms(&delta) * vector_rms(&psi_ct);
    let correlation = if corr_den > 0. {
        dot(&delta, &psi_ct) / (delta.len() as f64 * corr_den)
    } else {
        0.0
    };
    let var_d = delta.iter().map(|v| v * v).sum::<f64>();
    let var_r = resid.iter().map(|v| v * v).sum::<f64>();
    let variance_explained = if var_d > 0. { 1.0 - var_r / var_d } else { 0.0 };
    Some(DisplacementSoundSpeedFit {
        cs2,
        k_max,
        n_modes,
        rms_before,
        rms_after,
        correlation,
        variance_explained,
    })
}

/// Shell-averaged power of a vector field's components (unit-box), returned as physical k, P.
fn vector_field_power(
    grid: &Grid,
    eng: &mut FftEngine,
    field: &[f64],
    box_size: f64,
    nbins: usize,
) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let edges = spectra::bin_edges(grid, nbins);
    let mut ksum = vec![0.0; nbins];
    let mut psum = vec![0.0; nbins];
    let mut cnt = vec![0.0; nbins];
    let lk0 = edges[0].ln();
    let dlk = (edges[nbins] / edges[0]).ln() / nbins as f64;
    let vol = box_size.powi(grid.dim as i32);
    // Physical power of physical displacement: ψ_phys = ψ_unit L, P_phys = P_unit L² * L^d
    // Unit-box auto-spectrum of one component uses norm L_unit^d / N^{2d} with L_unit=1.
    let norm = 1.0 / (grid.size as f64).powi(2);
    for a in 0..grid.dim {
        let mut component = vec![0.0; grid.size];
        for id in 0..grid.size {
            component[id] = field[id * grid.dim + a];
        }
        let hat = eng.forward_real(&component);
        for idx in 1..grid.size {
            let k = grid.k2(idx).sqrt();
            if k < edges[0] || k >= edges[nbins] {
                continue;
            }
            let bi = (((k.ln() - lk0) / dlk).floor() as usize).min(nbins - 1);
            ksum[bi] += k;
            psum[bi] += hat[idx].norm_sqr() * norm;
            cnt[bi] += 1.0;
        }
    }
    let mut k = Vec::new();
    let mut p = Vec::new();
    let mut n = Vec::new();
    for i in 0..nbins {
        if cnt[i] > 0. {
            k.push((ksum[i] / cnt[i]) / box_size);
            // Summed over components already; convert unit-box displacement power to physical.
            p.push((psum[i] / cnt[i]) * vol * box_size.powi(2));
            n.push(cnt[i]);
        }
    }
    (k, p, n)
}

/// Draw `n_realizations` Gaussian stochastic displacements with shell power matched to
/// `template` (the residual after the deterministic counterterm), restricted to k ≤ k_max.
pub fn draw_stochastic_displacement_ensemble(
    grid: &Grid,
    eng: &mut FftEngine,
    residual: &[f64],
    box_size: f64,
    k_max: f64,
    n_realizations: usize,
    seed: u64,
) -> Vec<Vec<f64>> {
    let (k_shell, p_shell, _) =
        vector_field_power(grid, eng, residual, box_size, laboratory::spectrum_bins(grid.n));
    let k_max_unit = k_max * box_size;
    let mut out = Vec::with_capacity(n_realizations);
    for r in 0..n_realizations {
        let mut rng = Rng::new(seed.wrapping_add(17 + 104729 * r as u64));
        let mut field = vec![0.0; residual.len()];
        for a in 0..grid.dim {
            let mut hat = vec![C64::new(0.0, 0.0); grid.size];
            for idx in 1..grid.size {
                let ijk = grid.unravel(idx);
                let mut conj_ijk = [0usize; 3];
                for b in 0..grid.dim {
                    conj_ijk[b] = if ijk[b] == 0 { 0 } else { grid.n - ijk[b] };
                }
                let j = grid.ravel(conj_ijk);
                if j < idx {
                    continue; // already filled as conjugate of a lower index
                }
                let k_unit = grid.k2(idx).sqrt();
                if k_unit == 0.0 || k_unit > k_max_unit {
                    continue;
                }
                let k_phys = k_unit / box_size;
                let p = nearest_power(k_phys, &k_shell, &p_shell).max(0.0);
                // Physical P → unit-box component variance: undo vol * L² and split over dim.
                let p_unit =
                    p / (box_size.powi(grid.dim as i32) * box_size.powi(2) * grid.dim as f64);
                let sigma = (0.5 * p_unit).max(0.0).sqrt() * grid.size as f64;
                let z = if j == idx {
                    C64::new(rng.normal() * sigma * std::f64::consts::SQRT_2, 0.0)
                } else {
                    C64::new(rng.normal(), rng.normal()) * sigma
                };
                hat[idx] = z;
                if j != idx {
                    hat[j] = z.conj();
                }
            }
            let real = eng.inverse_to_real(hat);
            for id in 0..grid.size {
                field[id * grid.dim + a] = real[id];
            }
        }
        out.push(field);
    }
    out
}

/// Hard Fourier low-pass of a vector field: keep modes with |k|_unit ≤ k_max_unit.
fn lowpass_vector_field(
    grid: &Grid,
    eng: &mut FftEngine,
    field: &[f64],
    k_max_unit: f64,
) -> Vec<f64> {
    let mut out = vec![0.0; field.len()];
    for a in 0..grid.dim {
        let mut component = vec![0.0; grid.size];
        for id in 0..grid.size {
            component[id] = field[id * grid.dim + a];
        }
        let mut hat = eng.forward_real(&component);
        for (idx, v) in hat.iter_mut().enumerate() {
            let k = grid.k2(idx).sqrt();
            if idx != 0 && k > k_max_unit {
                *v = C64::new(0.0, 0.0);
            }
        }
        let real = eng.inverse_to_real(hat);
        for id in 0..grid.size {
            out[id * grid.dim + a] = real[id];
        }
    }
    out
}

fn nearest_power(k: f64, ks: &[f64], ps: &[f64]) -> f64 {
    if ks.is_empty() || ps.is_empty() || !(k > 0.) {
        return 0.0;
    }
    let mut best = 0usize;
    let mut best_d = f64::INFINITY;
    for (i, &ki) in ks.iter().enumerate() {
        if !(ki > 0.) {
            continue;
        }
        let d = (ki.ln() - k.ln()).abs();
        if d < best_d {
            best_d = d;
            best = i;
        }
    }
    ps[best.min(ps.len() - 1)]
}

/// Zero Fourier modes with physical |k| > `k_cut` (h/Mpc). Soft modes are left untouched.
fn apply_sharp_k_physical(c: &mut Cosmo, k_cut: f64, box_size: f64) {
    let k_cut_unit = k_cut * box_size;
    let k2_cut = k_cut_unit * k_cut_unit;
    for i in 0..c.grid.size {
        if c.grid.k2(i) > k2_cut {
            c.delta0_hat[i] = C64::new(0.0, 0.0);
        }
    }
    c.sigma0 = c
        .delta0_hat
        .iter()
        .map(|v| v.norm_sqr())
        .sum::<f64>()
        .sqrt()
        / c.grid.size as f64;
    c.lpt = None;
}

fn realized_hat_sigma(c: &Cosmo) -> f64 {
    c.delta0_hat
        .iter()
        .map(|v| v.norm_sqr())
        .sum::<f64>()
        .sqrt()
        / c.grid.size as f64
}

fn scale_delta0(c: &mut Cosmo, scale: f64) {
    if (scale - 1.0).abs() <= 1e-14 {
        return;
    }
    for v in c.delta0_hat.iter_mut() {
        *v *= scale;
    }
    c.sigma0 *= scale;
    c.lpt = None;
}

/// Mid-plane slice of one Cartesian displacement component (flat row-major).
fn displacement_component_slice(grid: &Grid, psi: &[f64], component: usize) -> (usize, usize, Vec<f64>) {
    let n = grid.n;
    let dim = grid.dim;
    let a = component.min(dim.saturating_sub(1));
    match dim {
        1 => {
            let mut out = Vec::with_capacity(n);
            for i in 0..n {
                out.push(psi[i * dim + a]);
            }
            (n, 1, out)
        }
        2 => {
            let mut out = Vec::with_capacity(n * n);
            for i in 0..n {
                for j in 0..n {
                    out.push(psi[(i * n + j) * dim + a]);
                }
            }
            (n, n, out)
        }
        _ => {
            let mid = n / 2;
            let mut out = Vec::with_capacity(n * n);
            for i in 0..n {
                for j in 0..n {
                    let id = (i * n + j) * n + mid;
                    out.push(psi[id * dim + a]);
                }
            }
            (n, n, out)
        }
    }
}

fn kmax_grid(k_max: f64) -> Vec<f64> {
    // Log-ish ladder around the requested fit cutoff for operator stability.
    let anchors = [
        0.40 * k_max,
        0.55 * k_max,
        0.70 * k_max,
        0.85 * k_max,
        k_max,
        1.15 * k_max,
        1.35 * k_max,
        1.60 * k_max,
        2.00 * k_max,
    ];
    let mut out = Vec::new();
    for &k in &anchors {
        if k > 0.01 && k <= 0.5 {
            if out.last().map(|&p: &f64| (k - p).abs() > 1e-6).unwrap_or(true) {
                out.push(k);
            }
        }
    }
    if out.is_empty() {
        out.push(k_max.max(0.02));
    }
    out
}

/// Dual-smoothing P1 sheet comparison across LPT orders 1–4 with a sound-speed fit.
///
/// - `uv_smoothing`: Gaussian IC radius expected near ∼1% inverted simplices at a=1.
/// - `ir_smoothing`: optional safer radius; when `None`, chosen so σ_IR ≈ σ_UV/3.
/// - `match_linear_variance`: primary IR amplitude — when true, scale IR δ₀ so continuum
///   linear σ matches UV. Both native and matched IR sheets are always evolved so the
///   UI can show side-by-side density ratios.
/// - `pair_fixed_soft`: sharp-k split of one unsmoothed white-noise seed with cuts
///   k_UV = 1/R_UV and k_IR = 1/R_IR so every mode with |k| ≤ k_IR is bitwise identical.
/// - Observation mesh should be finer than the Lagrangian lattice (P1 exact deposit).
pub fn run_dual_smoothing_counterterms(
    base: &LabRequest,
    uv_smoothing: f64,
    ir_smoothing: Option<f64>,
    k_max: f64,
    match_linear_variance: bool,
    pair_fixed_soft: bool,
) -> Result<Value, String> {
    if base.ic != "gaussian" {
        return Err(
            "Dual-smoothing counterterm comparison requires Gaussian initial conditions".into(),
        );
    }
    if !(uv_smoothing >= 0. && uv_smoothing <= 200.) {
        return Err("UV smoothing must lie in [0, 200] Mpc/h".into());
    }
    let growth_ratio = 1.0;
    let ir = match ir_smoothing {
        Some(r) => {
            if !(r >= uv_smoothing && r <= 200.) {
                return Err("IR smoothing must satisfy UV ≤ IR ≤ 200 Mpc/h".into());
            }
            r
        }
        None => match_sigma_ratio(base, uv_smoothing, 1.0 / 3.0, growth_ratio).ok_or_else(|| {
            "Could not match an IR filter with one-third the UV linear σ".to_string()
        })?,
    };

    let sigma_uv_cont = continuum_filtered_sigma(base, uv_smoothing, growth_ratio);
    let sigma_ir_cont = continuum_filtered_sigma(base, ir, growth_ratio);
    let match_scale = if sigma_ir_cont > 0. {
        sigma_uv_cont / sigma_ir_cont
    } else {
        1.0
    };

    let k_uv_cut = if uv_smoothing > 0. {
        1.0 / uv_smoothing
    } else {
        1e6
    };
    let k_ir_cut = if ir > 0. { 1.0 / ir } else { 1e6 };

    // --- Build UV + native IR (+ matched IR clone) ---
    let (mut c_uv, mut c_ir_native) = if pair_fixed_soft {
        let mut req0 = base.clone();
        req0.initial_smoothing = 0.0;
        req0.order = 4;
        req0.estimators = vec![];
        req0.validate()?;
        let mut uv = laboratory::initial_cosmo(&req0);
        let mut ir_c = laboratory::initial_cosmo(&req0);
        apply_sharp_k_physical(&mut uv, k_uv_cut, base.box_size);
        apply_sharp_k_physical(&mut ir_c, k_ir_cut, base.box_size);
        uv.build_lpt(4);
        ir_c.build_lpt(4);
        (uv, ir_c)
    } else {
        let mut req_uv = base.clone();
        req_uv.initial_smoothing = uv_smoothing;
        req_uv.order = 4;
        req_uv.estimators = vec![];
        req_uv.validate()?;
        let mut req_ir = base.clone();
        req_ir.initial_smoothing = ir;
        req_ir.order = 4;
        req_ir.estimators = vec![];
        req_ir.validate()?;
        (
            laboratory::initial_conditions(&req_uv),
            laboratory::initial_conditions(&req_ir),
        )
    };

    // Matched-IR: same native realization, δ₀ scaled so continuum σ matches UV.
    let mut c_ir_matched = laboratory::initial_cosmo(&{
        let mut r = base.clone();
        r.initial_smoothing = if pair_fixed_soft { 0.0 } else { ir };
        r.order = 4;
        r.estimators = vec![];
        r
    });
    c_ir_matched.delta0_hat = c_ir_native.delta0_hat.clone();
    c_ir_matched.sigma0 = c_ir_native.sigma0;
    c_ir_matched.cosmo = c_ir_native.cosmo.clone();
    c_ir_matched.keep_tensors = c_ir_native.keep_tensors;
    scale_delta0(&mut c_ir_matched, match_scale);
    c_ir_matched.build_lpt(4);

    let sigma_uv_realized = realized_hat_sigma(&c_uv);
    let sigma_ir_native_realized = realized_hat_sigma(&c_ir_native);
    let sigma_ir_matched_realized = realized_hat_sigma(&c_ir_matched);
    let amplitude_scale = if match_linear_variance {
        match_scale
    } else {
        1.0
    };
    // Prefer continuum σ for Gaussian-filter bookkeeping (stable vs discrete
    // sampling); for sharp-k pair-fixed, report realized discrete σ.
    let (sigma_uv, sigma_ir_native, sigma_ir_matched, sigma_ir_primary) = if pair_fixed_soft {
        let primary = if match_linear_variance {
            sigma_ir_matched_realized
        } else {
            sigma_ir_native_realized
        };
        (
            sigma_uv_realized,
            sigma_ir_native_realized,
            sigma_ir_matched_realized,
            primary,
        )
    } else {
        let primary = if match_linear_variance {
            sigma_uv_cont
        } else {
            sigma_ir_cont
        };
        (sigma_uv_cont, sigma_ir_cont, sigma_uv_cont, primary)
    };

    let a = if base.epoch == "scaleFactor" {
        base.scale_factor
    } else {
        let sc = (1..=4)
            .map(|o| c_uv.shell_crossing(o))
            .fold(f64::INFINITY, f64::min);
        if !sc.is_finite() || sc <= 0. {
            return Err("Cannot identify a finite crossing time for the UV realization".into());
        }
        c_uv.lpt_ref().growth.a_of_d(base.fraction * sc)
    };
    let d_uv = c_uv.lpt_ref().growth.d_of_a(a);
    let d_ir_n = c_ir_native.lpt_ref().growth.d_of_a(a);
    let d_ir_m = c_ir_matched.lpt_ref().growth.d_of_a(a);
    let eg = Grid::new(base.dim, base.mesh, 1.0);

    let (k_lin_uv, p_lin_uv, _) = realized_linear_physical(&c_uv, &eg, d_uv, base.box_size);
    let (k_lin_ir_n, p_lin_ir_n, n_lin_ir) =
        realized_linear_physical(&c_ir_native, &eg, d_ir_n, base.box_size);
    let (k_lin_ir_m, p_lin_ir_m, _) =
        realized_linear_physical(&c_ir_matched, &eg, d_ir_m, base.box_size);

    let c_ir_primary: &Cosmo = if match_linear_variance {
        &c_ir_matched
    } else {
        &c_ir_native
    };
    let d_ir = if match_linear_variance { d_ir_m } else { d_ir_n };
    let (k_lin_ir, p_lin_ir) = if match_linear_variance {
        (k_lin_ir_m.clone(), p_lin_ir_m.clone())
    } else {
        (k_lin_ir_n.clone(), p_lin_ir_n.clone())
    };

    let k_sweep = kmax_grid(k_max);
    let mut orders = Vec::new();
    let mut slice_preview: Option<Value> = None;

    for order in 1..=4 {
        let x_uv = laboratory::evaluate_lpt_positions(&c_uv, d_uv, order);
        let x_ir = laboratory::evaluate_lpt_positions(c_ir_primary, d_ir, order);
        let x_ir_n = laboratory::evaluate_lpt_positions(&c_ir_native, d_ir_n, order);
        let x_ir_m = laboratory::evaluate_lpt_positions(&c_ir_matched, d_ir_m, order);

        let (inv_n_uv, inv_m_uv) = inverted_fractions(&c_uv.grid, &x_uv);
        let (inv_n_ir, inv_m_ir) = inverted_fractions(&c_ir_primary.grid, &x_ir);
        let rho_uv = density_p1(&c_uv.grid, &x_uv, base.mesh);
        let rho_ir = density_p1(&c_ir_primary.grid, &x_ir, base.mesh);
        let rho_ir_n = density_p1(&c_ir_native.grid, &x_ir_n, base.mesh);
        let rho_ir_m = density_p1(&c_ir_matched.grid, &x_ir_m, base.mesh);
        let sp_uv = spectrum_physical(&rho_uv, &eg, base.box_size);
        let sp_ir = spectrum_physical(&rho_ir, &eg, base.box_size);
        let sp_ir_n = spectrum_physical(&rho_ir_n, &eg, base.box_size);
        let sp_ir_m = spectrum_physical(&rho_ir_m, &eg, base.box_size);

        let p_uv_on_ir: Vec<f64> = sp_ir
            .k
            .iter()
            .map(|&k| nearest_power(k, &sp_uv.k, &sp_uv.p))
            .collect();
        let p_lin_on_ir: Vec<f64> = sp_ir
            .k
            .iter()
            .map(|&k| nearest_power(k, &k_lin_ir, &p_lin_ir))
            .collect();
        let p_uv_on_n: Vec<f64> = sp_ir_n
            .k
            .iter()
            .map(|&k| nearest_power(k, &sp_uv.k, &sp_uv.p))
            .collect();
        let p_lin_uv_on_n: Vec<f64> = sp_ir_n
            .k
            .iter()
            .map(|&k| nearest_power(k, &k_lin_uv, &p_lin_uv))
            .collect();
        let p_lin_n_on_n: Vec<f64> = sp_ir_n
            .k
            .iter()
            .map(|&k| nearest_power(k, &k_lin_ir_n, &p_lin_ir_n))
            .collect();
        let p_uv_on_m: Vec<f64> = sp_ir_m
            .k
            .iter()
            .map(|&k| nearest_power(k, &sp_uv.k, &sp_uv.p))
            .collect();
        let p_lin_uv_on_m: Vec<f64> = sp_ir_m
            .k
            .iter()
            .map(|&k| nearest_power(k, &k_lin_uv, &p_lin_uv))
            .collect();
        let p_lin_m_on_m: Vec<f64> = sp_ir_m
            .k
            .iter()
            .map(|&k| nearest_power(k, &k_lin_ir_m, &p_lin_ir_m))
            .collect();

        let fit = fit_sound_speed(
            &sp_ir.k,
            &p_uv_on_ir,
            &sp_ir.p,
            &p_lin_on_ir,
            &sp_ir.nmodes,
            k_max,
        );
        let cs2_pk = fit.as_ref().map(|f| f.cs2).unwrap_or(0.0);
        let p_ir_plus_ctr: Vec<f64> = sp_ir
            .k
            .iter()
            .zip(sp_ir.p.iter())
            .zip(p_lin_on_ir.iter())
            .map(|((&k, &pir), &pl)| pir - 2.0 * cs2_pk * k * k * pl)
            .collect();

        let ratio_native: Vec<f64> = sp_ir_n
            .k
            .iter()
            .enumerate()
            .map(|(i, _)| {
                if sp_ir_n.p[i] > 0. {
                    p_uv_on_n[i] / sp_ir_n.p[i]
                } else {
                    0.0
                }
            })
            .collect();
        let ratio_matched: Vec<f64> = sp_ir_m
            .k
            .iter()
            .enumerate()
            .map(|(i, _)| {
                if sp_ir_m.p[i] > 0. {
                    p_uv_on_m[i] / sp_ir_m.p[i]
                } else {
                    0.0
                }
            })
            .collect();
        let lin_ratio_native: Vec<f64> = sp_ir_n
            .k
            .iter()
            .enumerate()
            .map(|(i, _)| {
                if p_lin_n_on_n[i] > 0. {
                    p_lin_uv_on_n[i] / p_lin_n_on_n[i]
                } else {
                    0.0
                }
            })
            .collect();
        let lin_ratio_matched: Vec<f64> = sp_ir_m
            .k
            .iter()
            .enumerate()
            .map(|(i, _)| {
                if p_lin_m_on_m[i] > 0. {
                    p_lin_uv_on_m[i] / p_lin_m_on_m[i]
                } else {
                    0.0
                }
            })
            .collect();
        let filter_corr_native: Vec<f64> = ratio_native
            .iter()
            .zip(lin_ratio_native.iter())
            .map(|(&r, &lr)| if lr > 0. { r / lr } else { 0.0 })
            .collect();
        let filter_corr_matched: Vec<f64> = ratio_matched
            .iter()
            .zip(lin_ratio_matched.iter())
            .map(|(&r, &lr)| if lr > 0. { r / lr } else { 0.0 })
            .collect();

        // Vertex-level displacement comparison (Mirbabayi–Zaldarriaga bookkeeping).
        let psi_uv = displacement_from_positions(&c_uv.grid, &x_uv);
        let psi_ir = displacement_from_positions(&c_ir_primary.grid, &x_ir);
        let mut eng_l = FftEngine::new(&c_uv.grid);
        let disp_fit = fit_displacement_sound_speed(
            &c_uv.grid,
            &mut eng_l,
            &psi_uv,
            &psi_ir,
            base.box_size,
            k_max,
        );
        let cs2_disp = disp_fit.as_ref().map(|f| f.cs2).unwrap_or(0.0);
        let psi_ct =
            apply_displacement_counterterm(&c_uv.grid, &mut eng_l, &psi_ir, cs2_disp, base.box_size);
        let mut resid = vec![0.0; psi_uv.len()];
        let mut delta = vec![0.0; psi_uv.len()];
        let mut psi_ir_ct = vec![0.0; psi_uv.len()];
        for i in 0..psi_uv.len() {
            delta[i] = psi_uv[i] - psi_ir[i];
            resid[i] = psi_uv[i] - (psi_ir[i] + psi_ct[i]);
            psi_ir_ct[i] = psi_ir[i] + psi_ct[i];
        }
        let (k_res, p_res, n_res) = vector_field_power(
            &c_uv.grid,
            &mut eng_l,
            &resid,
            base.box_size,
            laboratory::spectrum_bins(c_uv.grid.n),
        );
        let (k_del, p_del, _) = vector_field_power(
            &c_uv.grid,
            &mut eng_l,
            &delta,
            base.box_size,
            laboratory::spectrum_bins(c_uv.grid.n),
        );
        let ensemble = draw_stochastic_displacement_ensemble(
            &c_uv.grid,
            &mut eng_l,
            &resid,
            base.box_size,
            k_max,
            16,
            base.seed.wrapping_add(order as u64 * 10007),
        );
        let resid_lp = lowpass_vector_field(&c_uv.grid, &mut eng_l, &resid, k_max * base.box_size);
        let mut covered = 0.0_f64;
        let mut total = 0.0_f64;
        for id in 0..c_uv.grid.size {
            for a in 0..c_uv.grid.dim {
                let idx = id * c_uv.grid.dim + a;
                let target = resid_lp[idx];
                let mut var = 0.0_f64;
                for e in &ensemble {
                    let d = e[idx];
                    var += d * d;
                }
                var /= ensemble.len().max(1) as f64;
                let sig = var.sqrt().max(1e-15);
                if target.abs() <= sig {
                    covered += 1.0;
                }
                total += 1.0;
            }
        }
        let ensemble_coverage = covered / total.max(1.0);

        // k_max stability sweep — refit only (no re-LPT).
        let mut sweep = Vec::new();
        for &km in &k_sweep {
            let df =
                fit_displacement_sound_speed(&c_uv.grid, &mut eng_l, &psi_uv, &psi_ir, base.box_size, km);
            let pf = fit_sound_speed(
                &sp_ir.k,
                &p_uv_on_ir,
                &sp_ir.p,
                &p_lin_on_ir,
                &sp_ir.nmodes,
                km,
            );
            sweep.push(json!({
                "kMax": km,
                "cs2Disp": df.as_ref().map(|f| f.cs2),
                "cs2Pk": pf.as_ref().map(|f| f.cs2),
                "rmsReduction": df.as_ref().map(|f| {
                    if f.rms_before > 0. { 1.0 - f.rms_after / f.rms_before } else { 0.0 }
                }),
                "varianceExplained": df.as_ref().map(|f| f.variance_explained),
                "nModes": df.as_ref().map(|f| f.n_modes),
                "nBins": pf.as_ref().map(|f| f.n_bins),
            }));
        }

        if order == 4 {
            let (nx, ny, uv_sl) = displacement_component_slice(&c_uv.grid, &psi_uv, 0);
            let (_, _, ir_ct_sl) = displacement_component_slice(&c_uv.grid, &psi_ir_ct, 0);
            let mut draw_slices = Vec::new();
            for (di, draw) in ensemble.iter().take(3).enumerate() {
                let mut full = psi_ir_ct.clone();
                for i in 0..full.len() {
                    full[i] += draw[i];
                }
                let (_, _, sl) = displacement_component_slice(&c_uv.grid, &full, 0);
                draw_slices.push(json!({ "index": di, "psiX": sl }));
            }
            slice_preview = Some(json!({
                "order": 4,
                "component": "psi_x",
                "plane": if c_uv.grid.dim >= 3 { "z=n/2" } else { "full" },
                "nx": nx,
                "ny": ny,
                "uv": uv_sl,
                "irPlusCt": ir_ct_sl,
                "draws": draw_slices,
                "note": "2D mid-plane of the first Cartesian displacement component. Draws = IR+CT+stochastic residual.",
            }));
        }

        orders.push(json!({
            "order": order,
            "uv": {
                "invertedElementFraction": inv_n_uv,
                "invertedMassFraction": inv_m_uv,
                "k": sp_uv.k,
                "power": sp_uv.p,
                "nmodes": sp_uv.nmodes,
                "fineK": sp_uv.fine_k,
                "finePower": sp_uv.fine_p,
                "fineNmodes": sp_uv.fine_nmodes,
            },
            "ir": {
                "invertedElementFraction": inv_n_ir,
                "invertedMassFraction": inv_m_ir,
                "k": sp_ir.k,
                "power": sp_ir.p,
                "nmodes": sp_ir.nmodes,
                "fineK": sp_ir.fine_k,
                "finePower": sp_ir.fine_p,
                "fineNmodes": sp_ir.fine_nmodes,
            },
            "deltaPower": sp_ir.k.iter().enumerate().map(|(i, _)| p_uv_on_ir[i] - sp_ir.p[i]).collect::<Vec<_>>(),
            "irPlusCounterterm": p_ir_plus_ctr,
            "amplitudeCompare": {
                "native": {
                    "k": sp_ir_n.k,
                    "ratio": ratio_native,
                    "linearRatio": lin_ratio_native,
                    "filterCorrectedRatio": filter_corr_native,
                },
                "matched": {
                    "k": sp_ir_m.k,
                    "ratio": ratio_matched,
                    "linearRatio": lin_ratio_matched,
                    "filterCorrectedRatio": filter_corr_matched,
                },
            },
            "kMaxSweep": sweep,
            "fit": fit.as_ref().map(|f| json!({
                "cs2": f.cs2,
                "nBins": f.n_bins,
                "kMax": f.k_max,
                "residualRms": f.residual_rms,
                "meanRelativeDelta": f.mean_relative_delta,
            })),
            "displacement": {
                "fit": disp_fit.as_ref().map(|f| json!({
                    "cs2": f.cs2,
                    "kMax": f.k_max,
                    "nModes": f.n_modes,
                    "rmsBefore": f.rms_before,
                    "rmsAfter": f.rms_after,
                    "rmsReduction": if f.rms_before > 0. { 1.0 - f.rms_after / f.rms_before } else { 0.0 },
                    "correlation": f.correlation,
                    "varianceExplained": f.variance_explained,
                })),
                "operator": "psi_ct(k) = -cs2 * k_phys^2 * psi_IR(k)",
                "deltaPower": { "k": k_del, "power": p_del },
                "residualPower": { "k": k_res, "power": p_res, "nmodes": n_res },
                "stochastic": {
                    "nRealizations": ensemble.len(),
                    "ensembleCoverageWithin1Sigma": ensemble_coverage,
                    "note": "Gaussian draws matched to residual displacement shell power for k<=kMax; deterministic CT is the ensemble mean.",
                },
            },
        }));
    }

    // Silence unused mut on c_uv after construction path.
    let _ = &mut c_uv;
    let _ = &mut c_ir_native;

    Ok(json!({
        "kind": "eftCounterterms",
        "request": {
            "dim": base.dim,
            "n": base.n,
            "mesh": base.mesh,
            "seed": base.seed,
            "boxSize": base.box_size,
            "epoch": base.epoch,
            "scaleFactor": a,
            "fraction": base.fraction,
            "transfer": base.transfer,
            "omegaM": base.omega_m,
            "omegaB": base.omega_b,
            "h": base.h,
            "ns": base.ns,
            "sigma8": base.sigma8,
            "initialCutoff": base.initial_cutoff,
            "uvSmoothing": uv_smoothing,
            "irSmoothing": ir,
            "kMaxFit": k_max,
            "matchLinearVariance": match_linear_variance,
            "pairFixedSoft": pair_fixed_soft,
        },
        "filters": {
            "uvSmoothing": uv_smoothing,
            "irSmoothing": ir,
            "sigmaLinUv": sigma_uv,
            "sigmaLinUvContinuum": sigma_uv_cont,
            "sigmaLinUvRealized": sigma_uv_realized,
            "sigmaLinIrNative": sigma_ir_native,
            "sigmaLinIrMatched": sigma_ir_matched,
            "sigmaLinIrNativeRealized": sigma_ir_native_realized,
            "sigmaLinIrMatchedRealized": sigma_ir_matched_realized,
            "sigmaLinIr": sigma_ir_primary,
            "sigmaRatio": if sigma_uv > 0. { sigma_ir_primary / sigma_uv } else { 0.0 },
            "matchLinearVariance": match_linear_variance,
            "amplitudeScale": amplitude_scale,
            "matchScale": match_scale,
            "pairFixedSoft": pair_fixed_soft,
            "kUvCut": if pair_fixed_soft { Some(k_uv_cut) } else { None },
            "kIrCut": if pair_fixed_soft { Some(k_ir_cut) } else { None },
            "note": if pair_fixed_soft {
                "Sharp-k pair-fixed soft modes: shared white noise with k_cut = 1/R. Modes |k|≤k_IR are identical in UV and IR."
            } else if match_linear_variance {
                "Primary IR δ0 scaled by σ_UV/σ_IR(native) before LPT. Native and matched density ratios both returned for side-by-side comparison."
            } else {
                "Primary uses native filtered amplitudes. Matched-σ density ratios also returned for side-by-side comparison."
            },
        },
        "linear": {
            "uv": { "k": k_lin_uv, "power": p_lin_uv },
            "ir": { "k": k_lin_ir, "power": p_lin_ir, "nmodes": n_lin_ir },
            "irNative": { "k": k_lin_ir_n, "power": p_lin_ir_n },
            "irMatched": { "k": k_lin_ir_m, "power": p_lin_ir_m },
        },
        "orders": orders,
        "slicePreview": slice_preview,
        "convention": "Displacement EFT (Mirbabayi & Zaldarriaga): psi_UV ≈ psi_IR - cs2 k^2 psi_IR on modes k<=kMax, plus a Gaussian stochastic residual ensemble matched to leftover displacement power. Density P(k) fit retained as a companion diagnostic. Both native and matched-σ IR ratios are returned; kMaxSweep refits without re-LPT.",
    }))
}
