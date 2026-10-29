//! Dual-smoothing Lagrangian UV response and the leading sound-speed counterterm.
//!
//! Two Gaussian initial filters of the same seed share long-wavelength phases. The
//! difference of their nonlinear P1 sheet power at low *k* is the integrated response
//! of the modes between the two cutoffs. The usual one-loop Lagrangian / Eulerian EFT
//! counterterm −2 cₛ² k² P_lin absorbs that difference on large scales.
use crate::{
    cosmological::LinearSpectrum,
    fft::FftEngine,
    grid::Grid,
    laboratory::{self, LabRequest},
    observables, sheet, spectra, Cosmo,
};
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

/// Dual-smoothing P1 sheet comparison across LPT orders 1–4 with a sound-speed fit.
///
/// - `uv_smoothing`: Gaussian IC radius expected near ∼1% inverted simplices at a=1.
/// - `ir_smoothing`: optional safer radius; when `None`, chosen so σ_IR ≈ σ_UV/3.
/// - Observation mesh should be finer than the Lagrangian lattice (P1 exact deposit).
pub fn run_dual_smoothing_counterterms(
    base: &LabRequest,
    uv_smoothing: f64,
    ir_smoothing: Option<f64>,
    k_max: f64,
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

    let sigma_uv = continuum_filtered_sigma(base, uv_smoothing, growth_ratio);
    let sigma_ir = continuum_filtered_sigma(base, ir, growth_ratio);

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

    let c_uv = laboratory::initial_conditions(&req_uv);
    let c_ir = laboratory::initial_conditions(&req_ir);
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
    let d_ir = c_ir.lpt_ref().growth.d_of_a(a);
    let eg = Grid::new(base.dim, base.mesh, 1.0);

    let (k_lin_ir, p_lin_ir, n_lin_ir) = realized_linear_physical(&c_ir, &eg, d_ir, base.box_size);
    let (k_lin_uv, p_lin_uv, _) = realized_linear_physical(&c_uv, &eg, d_uv, base.box_size);

    let mut orders = Vec::new();
    for order in 1..=4 {
        let x_uv = laboratory::evaluate_lpt_positions(&c_uv, d_uv, order);
        let x_ir = laboratory::evaluate_lpt_positions(&c_ir, d_ir, order);
        let (inv_n_uv, inv_m_uv) = inverted_fractions(&c_uv.grid, &x_uv);
        let (inv_n_ir, inv_m_ir) = inverted_fractions(&c_ir.grid, &x_ir);
        let rho_uv = density_p1(&c_uv.grid, &x_uv, base.mesh);
        let rho_ir = density_p1(&c_ir.grid, &x_ir, base.mesh);
        let sp_uv = spectrum_physical(&rho_uv, &eg, base.box_size);
        let sp_ir = spectrum_physical(&rho_ir, &eg, base.box_size);

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
        let fit = fit_sound_speed(
            &sp_ir.k,
            &p_uv_on_ir,
            &sp_ir.p,
            &p_lin_on_ir,
            &sp_ir.nmodes,
            k_max,
        );
        let cs2 = fit.as_ref().map(|f| f.cs2).unwrap_or(0.0);
        let p_ir_plus_ctr: Vec<f64> = sp_ir
            .k
            .iter()
            .zip(sp_ir.p.iter())
            .zip(p_lin_on_ir.iter())
            .map(|((&k, &pir), &pl)| pir - 2.0 * cs2 * k * k * pl)
            .collect();

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
            "fit": fit.as_ref().map(|f| json!({
                "cs2": f.cs2,
                "nBins": f.n_bins,
                "kMax": f.k_max,
                "residualRms": f.residual_rms,
                "meanRelativeDelta": f.mean_relative_delta,
            })),
        }));
    }

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
        },
        "filters": {
            "uvSmoothing": uv_smoothing,
            "irSmoothing": ir,
            "sigmaLinUv": sigma_uv,
            "sigmaLinIr": sigma_ir,
            "sigmaRatio": if sigma_uv > 0. { sigma_ir / sigma_uv } else { 0.0 },
        },
        "linear": {
            "uv": { "k": k_lin_uv, "power": p_lin_uv },
            "ir": { "k": k_lin_ir, "power": p_lin_ir, "nmodes": n_lin_ir },
        },
        "orders": orders,
        "convention": "P_UV(k) ≈ P_IR(k) - 2 cs2 k^2 P_lin^IR(k); cs2 in (Mpc/h)^2; P1 exact sheet deposit; no PM.",
    }))
}
