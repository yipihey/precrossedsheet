//! Experimental 2LPT→3LPT closure diagnostics (Neural Sheet).
//!
//! This module does **not** replace the exact nLPT solver. It exposes the
//! EdS cubic source decomposition into local deformation-tensor invariants
//! plus the known spectral Poisson/Helmholtz maps, for cheap-closure research.

use crate::fft::FftEngine;
use crate::growth::{term_specs, Cosmology, Growth, Parents, TermSpec};
use crate::lpt::{mu2, mu3, Lpt, Term};
use num_complex::Complex64 as C64;

/// One node in the unmerged LPT term dependency graph.
#[derive(Clone, Debug)]
pub struct TermNode {
    pub index: usize,
    pub order: usize,
    pub label: String,
    pub kind: &'static str,
    pub parents: Vec<usize>,
    pub eds_coef: f64,
}

/// Enumerate unmerged terms through `max_order` with EdS coefficients.
pub fn dependency_nodes(max_order: usize, dim: usize) -> Vec<TermNode> {
    let specs = term_specs(max_order, dim);
    let growth = Growth::new_eds(&specs);
    let coef = match &growth {
        Growth::Eds { coef, .. } => coef.clone(),
        _ => unreachable!(),
    };
    specs
        .into_iter()
        .enumerate()
        .map(|(index, s)| {
            let (kind, parents) = match s.parents {
                Parents::Linear => ("linear", vec![]),
                Parents::Mu2(a, b) => ("mu2", vec![a, b]),
                Parents::Mu3(a, b, c) => ("mu3", vec![a, b, c]),
                Parents::Curl(a, b) => ("curl", vec![a, b]),
            };
            TermNode {
                index,
                order: s.order,
                label: s.label,
                kind,
                parents,
                eds_coef: coef[index],
            }
        })
        .collect()
}

/// Count raw (unmerged) terms and how many FFTs a naive build performs per term.
///
/// Each non-linear longitudinal term: 1 forward (source) + `d` inverse (Ψ) + `d²`
/// inverse (M). Each transverse term: `ncurl` forwards + `d` + `d²` inverses.
/// Linear: no source FFT; `d` inverses for Ψ and `d²` for M (from Ψ̂).
#[derive(Clone, Debug)]
pub struct CostModel {
    pub dim: usize,
    pub max_order: usize,
    pub n_terms_unmerged: usize,
    pub n_terms_merged_eds: usize,
    pub n_mu2: usize,
    pub n_mu3: usize,
    pub n_curl: usize,
    /// Approximate complex FFT count to build all terms with tensors kept.
    pub approx_fft_count: usize,
    /// Gradient-tensor field count if all terms keep M (dim² per term).
    pub tensor_field_count: usize,
}

pub fn cost_model(max_order: usize, dim: usize) -> CostModel {
    let nodes = dependency_nodes(max_order, dim);
    let ncurl = if dim == 2 { 1 } else if dim >= 3 { 3 } else { 0 };
    let mut n_mu2 = 0;
    let mut n_mu3 = 0;
    let mut n_curl = 0;
    let mut ffts = 0;
    for n in &nodes {
        let d = dim;
        match n.kind {
            "linear" => {
                // Ψ̂ from δ̂ (no FFT); then d IFFTs for Ψ and d² for M.
                ffts += d + d * d;
            }
            "mu2" => {
                n_mu2 += 1;
                ffts += 1 + d + d * d;
            }
            "mu3" => {
                n_mu3 += 1;
                ffts += 1 + d + d * d;
            }
            "curl" => {
                n_curl += 1;
                ffts += ncurl + d + d * d;
            }
            _ => {}
        }
    }
    CostModel {
        dim,
        max_order,
        n_terms_unmerged: nodes.len(),
        n_terms_merged_eds: max_order,
        n_mu2,
        n_mu3,
        n_curl,
        approx_fft_count: ffts,
        tensor_field_count: nodes.len() * dim * dim,
    }
}

/// Cauchy curl bilinear used by the exact recurrence: T_i = ε_ijk M^a_lj M^b_lk.
pub fn curl_product(a: &[f64], b: &[f64], dim: usize, component: usize) -> f64 {
    if dim < 2 {
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

/// Exact EdS cubic longitudinal source in terms of merged M⁽¹⁾, M⁽²⁾:
/// ∇·Ψ⁽³⁾ = (−10/9) μ₂(M¹,M²) + (−1/3) μ₃(M¹,M¹,M¹)  (μ₃ term in 3D only).
pub fn eds_div3_from_merged(m1: &[f64], m2: &[f64], dim: usize) -> f64 {
    let mut s = -10.0 / 9.0 * mu2(m1, m2, dim);
    if dim >= 3 {
        s -= (1.0 / 3.0) * mu3(m1, m1, m1, dim);
    }
    s
}

/// Exact EdS cubic transverse source component from merged tensors:
/// (∇×Ψ⁽³⁾)_i = (1/3) curl_product(M¹, M²).
pub fn eds_curl3_from_merged(m1: &[f64], m2: &[f64], dim: usize, component: usize) -> f64 {
    (1.0 / 3.0) * curl_product(m1, m2, dim, component)
}

fn tensor_slice(term: &crate::lpt::Term, idx: usize, dim: usize, out: &mut [f64]) {
    let m = term.m.as_ref().expect("keep_tensors required for closure diagnostics");
    for c in 0..dim * dim {
        out[c] = m[c][idx] as f64;
    }
}

/// Diagnostics comparing exact merged-order-3 sources to invariant reconstructions.
#[derive(Clone, Debug)]
pub struct InvariantBasisReport {
    pub dim: usize,
    pub n: usize,
    pub rms_div3: f64,
    pub rms_div3_exact_basis: f64,
    pub rel_div3_exact_basis: f64,
    pub rms_curl3: f64,
    pub rms_curl3_exact_basis: f64,
    pub rel_curl3_exact_basis: f64,
    /// Best least-squares residual using only local cubic scalars of M¹.
    pub rel_div3_m1_local_fit: f64,
    pub m1_local_fit_coefs: [f64; 3],
    pub cosmology: Cosmology,
}

fn rms(v: &[f64]) -> f64 {
    (v.iter().map(|x| x * x).sum::<f64>() / v.len().max(1) as f64).sqrt()
}

fn lstsq_3(cols: [&[f64]; 3], target: &[f64]) -> ([f64; 3], f64) {
    // Normal equations for 3 columns.
    let mut ata = [[0.0; 3]; 3];
    let mut atb = [0.0; 3];
    let n = target.len();
    for i in 0..n {
        let r = [cols[0][i], cols[1][i], cols[2][i]];
        for a in 0..3 {
            atb[a] += r[a] * target[i];
            for b in 0..3 {
                ata[a][b] += r[a] * r[b];
            }
        }
    }
    // Solve 3×3 with simple Gaussian elimination.
    let mut a = ata;
    let mut b = atb;
    for k in 0..3 {
        let mut piv = k;
        for i in k + 1..3 {
            if a[i][k].abs() > a[piv][k].abs() {
                piv = i;
            }
        }
        if a[piv][k].abs() < 1e-30 {
            return ([0.0; 3], rms(target));
        }
        a.swap(k, piv);
        b.swap(k, piv);
        let diag = a[k][k];
        for j in k..3 {
            a[k][j] /= diag;
        }
        b[k] /= diag;
        for i in 0..3 {
            if i == k {
                continue;
            }
            let f = a[i][k];
            for j in k..3 {
                a[i][j] -= f * a[k][j];
            }
            b[i] -= f * b[k];
        }
    }
    let coef = b;
    let mut resid = vec![0.0; n];
    for i in 0..n {
        resid[i] = target[i] - (coef[0] * cols[0][i] + coef[1] * cols[1][i] + coef[2] * cols[2][i]);
    }
    (coef, rms(&resid))
}

/// Require an EdS `Lpt` built with `keep_tensors=true` and `order >= 3`.
pub fn assess_invariant_basis(lpt: &Lpt) -> Result<InvariantBasisReport, String> {
    if lpt.order < 3 {
        return Err("need order >= 3".into());
    }
    if !lpt.growth.is_eds() {
        return Err("Phase-A basis check uses EdS merged fields".into());
    }
    if lpt.terms.len() < 3 || lpt.terms.iter().any(|t| t.m.is_none()) {
        return Err("need merged EdS terms with gradient tensors kept".into());
    }
    let dim = lpt.grid.dim;
    let n = lpt.grid.size;
    // Merged EdS: terms[0]=1LPT, terms[1]=2LPT, terms[2]=3LPT.
    let t1 = &lpt.terms[0];
    let t2 = &lpt.terms[1];
    let t3 = &lpt.terms[2];

    let mut m1 = vec![0.0; 9];
    let mut m2 = vec![0.0; 9];
    let mut div_exact = vec![0.0; n];
    let mut div_basis = vec![0.0; n];
    let mut div_err = vec![0.0; n];
    let mut b_mu2 = vec![0.0; n];
    let mut b_mu3 = vec![0.0; n];
    let mut b_tr3 = vec![0.0; n];
    let mut b_tr_tr2 = vec![0.0; n];
    let mut b_det = vec![0.0; n];

    for i in 0..n {
        tensor_slice(t1, i, dim, &mut m1);
        tensor_slice(t2, i, dim, &mut m2);
        let d = &m1[..dim * dim];
        let e = &m2[..dim * dim];
        let tr = (0..dim).map(|a| d[a * dim + a]).sum::<f64>();
        let tr2 = {
            let mut s = 0.0;
            for a in 0..dim {
                for b in 0..dim {
                    s += d[a * dim + b] * d[b * dim + a];
                }
            }
            s
        };
        div_exact[i] = if !t3.div.is_empty() {
            t3.div[i] as f64
        } else {
            0.0
        };
        div_basis[i] = eds_div3_from_merged(d, e, dim);
        div_err[i] = div_exact[i] - div_basis[i];
        b_mu2[i] = mu2(d, d, dim);
        b_mu3[i] = mu3(d, d, d, dim);
        b_tr3[i] = tr.powi(3);
        b_tr_tr2[i] = tr * tr2;
        b_det[i] = b_mu3[i]; // μ3(M,M,M)=det M in 3D; in 2D μ3=0 and μ2(M,M)=det
    }

    let rms_div = rms(&div_exact);
    let rms_div_err = rms(&div_err);
    let rel_div = if rms_div > 0.0 {
        rms_div_err / rms_div
    } else {
        0.0
    };

    // Local M¹-only cubic fit: α μ2(M,M) + β (tr M)³ + γ (tr M)(tr M²)
    // (μ3 is linearly dependent with these in 3D via Cayley–Hamilton; keep μ2, tr³, tr·tr2).
    let (coefs, rms_m1) = if dim == 2 {
        // In 2D μ3≡0; cubic scalars of one 2×2 matrix: use μ2(M,M)=det, (tr)³, tr·tr2.
        lstsq_3([&b_mu2, &b_tr3, &b_tr_tr2], &div_exact)
    } else {
        lstsq_3([&b_mu2, &b_tr3, &b_tr_tr2], &div_exact)
    };
    let rel_m1 = if rms_div > 0.0 { rms_m1 / rms_div } else { 0.0 };

    let ncurl = if dim == 2 { 1 } else { 3 };
    let mut curl_exact = vec![0.0; n * ncurl];
    let mut curl_err = vec![0.0; n * ncurl];
    for i in 0..n {
        tensor_slice(t1, i, dim, &mut m1);
        tensor_slice(t2, i, dim, &mut m2);
        for c in 0..ncurl {
            let ex = if c < t3.curl.len() {
                t3.curl[c][i] as f64
            } else {
                0.0
            };
            let pred = eds_curl3_from_merged(&m1[..dim * dim], &m2[..dim * dim], dim, c);
            curl_exact[i * ncurl + c] = ex;
            curl_err[i * ncurl + c] = ex - pred;
        }
    }
    let rms_c = rms(&curl_exact);
    let rms_ce = rms(&curl_err);
    let rel_c = if rms_c > 0.0 { rms_ce / rms_c } else { 0.0 };

    let _ = b_det;
    Ok(InvariantBasisReport {
        dim,
        n: lpt.grid.n,
        rms_div3: rms_div,
        rms_div3_exact_basis: rms_div_err,
        rel_div3_exact_basis: rel_div,
        rms_curl3: rms_c,
        rms_curl3_exact_basis: rms_ce,
        rel_curl3_exact_basis: rel_c,
        rel_div3_m1_local_fit: rel_m1,
        m1_local_fit_coefs: coefs,
        cosmology: lpt.cosmo,
    })
}

/// Ablation switches for the explicit EdS cubic closure C₃.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CubicClosureOpts {
    /// Include (−1/3)μ₃(M¹,M¹,M¹) in the longitudinal source (3D only).
    pub include_mu3: bool,
    /// Include the transverse Cauchy source.
    pub include_curl: bool,
    /// Keep spectral gradient tensors of the cubic term.
    pub keep_tensors: bool,
}

impl Default for CubicClosureOpts {
    fn default() -> Self {
        Self {
            include_mu3: true,
            include_curl: true,
            keep_tensors: true,
        }
    }
}

impl CubicClosureOpts {
    pub fn full() -> Self {
        Self::default()
    }
    pub fn no_mu3() -> Self {
        Self {
            include_mu3: false,
            ..Self::default()
        }
    }
    pub fn no_curl() -> Self {
        Self {
            include_curl: false,
            ..Self::default()
        }
    }
}

/// Pointwise EdS cubic sources with optional ablations.
pub fn eds_cubic_sources(
    m1: &[f64],
    m2: &[f64],
    dim: usize,
    opts: CubicClosureOpts,
) -> (f64, Vec<f64>) {
    let mut div = -10.0 / 9.0 * mu2(m1, m2, dim);
    if opts.include_mu3 && dim >= 3 {
        div -= (1.0 / 3.0) * mu3(m1, m1, m1, dim);
    }
    let ncurl = if dim == 2 { 1 } else if dim >= 3 { 3 } else { 0 };
    let mut curl = vec![0.0; ncurl];
    if opts.include_curl {
        for c in 0..ncurl {
            curl[c] = eds_curl3_from_merged(m1, m2, dim, c);
        }
    }
    (div, curl)
}

/// Product-rule tangents of EdS cubic sources (for JVPs through C₃).
pub fn eds_cubic_source_jvp(
    m1: &[f64],
    m2: &[f64],
    dm1: &[f64],
    dm2: &[f64],
    dim: usize,
    opts: CubicClosureOpts,
) -> (f64, Vec<f64>) {
    let mut ddiv = -10.0 / 9.0 * (mu2(dm1, m2, dim) + mu2(m1, dm2, dim));
    if opts.include_mu3 && dim >= 3 {
        // μ₃ is totally symmetric ⇒ d μ₃(M,M,M) = 3 μ₃(dM,M,M).
        ddiv -= mu3(dm1, m1, m1, dim);
    }
    let ncurl = if dim == 2 { 1 } else if dim >= 3 { 3 } else { 0 };
    let mut dcurl = vec![0.0; ncurl];
    if opts.include_curl {
        for c in 0..ncurl {
            dcurl[c] = (1.0 / 3.0)
                * (curl_product(dm1, m2, dim, c) + curl_product(m1, dm2, dim, c));
        }
    }
    (ddiv, dcurl)
}

fn clone_term(t: &Term) -> Term {
    Term {
        spec: TermSpec {
            order: t.spec.order,
            parents: t.spec.parents.clone(),
            label: t.spec.label.clone(),
        },
        psi: t.psi.clone(),
        m: t.m.clone(),
        div: t.div.clone(),
        curl: t.curl.clone(),
    }
}

/// Deep-copy an `Lpt` so callers can mutate a paired `FftEngine` afterwards.
pub fn clone_lpt(lpt: &Lpt) -> Lpt {
    Lpt {
        grid: lpt.grid.clone(),
        order: lpt.order,
        terms: lpt.terms.iter().map(clone_term).collect(),
        growth: match &lpt.growth {
            Growth::Eds { coef, orders } => Growth::Eds {
                coef: coef.clone(),
                orders: orders.clone(),
            },
            Growth::Lcdm {
                cosmo,
                lna,
                g,
                gp,
                gpp,
            } => Growth::Lcdm {
                cosmo: *cosmo,
                lna: lna.clone(),
                g: g.clone(),
                gp: gp.clone(),
                gpp: gpp.clone(),
            },
        },
        cosmo: lpt.cosmo,
    }
}

fn spectral_gradients(
    eng: &mut FftEngine,
    grid: &crate::grid::Grid,
    psihat: &[Vec<C64>],
) -> Vec<Vec<f32>> {
    let d = grid.dim;
    let mut m = vec![Vec::new(); d * d];
    for a in 0..d {
        for b in 0..d {
            let dh: Vec<C64> = psihat[a]
                .iter()
                .enumerate()
                .map(|(idx, &v)| C64::new(0.0, grid.kvec(idx)[b]) * v)
                .collect();
            m[a * d + b] = eng.inverse_to_real_f32(dh);
        }
    }
    m
}

/// Approximate incremental FFT count for one cubic Helmholtz solve (combined longitudinal).
pub fn cubic_incremental_fft_count(dim: usize, opts: CubicClosureOpts) -> usize {
    let d = dim;
    let ncurl = if !opts.include_curl {
        0
    } else if dim == 2 {
        1
    } else {
        3
    };
    // 1 forward for combined div + d IFFT Ψ + (optional) d² IFFT M
    // + ncurl forwards + d IFFT Ψ_⊥ + (optional) d² IFFT M_⊥, then add into same term
    // Implementation solves longitudinal and transverse separately then adds Ψ.
    let long = 1 + d + if opts.keep_tensors { d * d } else { 0 };
    let trans = if ncurl == 0 {
        0
    } else {
        ncurl + d + if opts.keep_tensors { d * d } else { 0 }
    };
    long + trans
}

/// Extend an EdS merged 2LPT solution with the explicit cubic closure C₃.
///
/// Requires `lpt2.order >= 2`, EdS cosmology, and gradient tensors on terms 0 and 1.
/// Returns a merged EdS `Lpt` with `order = 3` whose cubic term is built from
/// `μ₂(M¹,M²)`, optional `μ₃(M¹³)`, and optional curl, then exact spectral Helmholtz.
pub fn eds_extend_cubic(
    lpt2: &Lpt,
    eng: &mut FftEngine,
    opts: CubicClosureOpts,
) -> Result<Lpt, String> {
    if lpt2.order < 2 {
        return Err("eds_extend_cubic needs order >= 2".into());
    }
    if !lpt2.growth.is_eds() || !lpt2.cosmo.is_eds() {
        return Err("eds_extend_cubic is EdS-only (ΛCDM keeps unmerged g_τ)".into());
    }
    if lpt2.terms.len() < 2 {
        return Err("need merged terms for orders 1 and 2".into());
    }
    if lpt2.terms[0].m.is_none() || lpt2.terms[1].m.is_none() {
        return Err("keep_tensors required on 1LPT and 2LPT".into());
    }
    let dim = lpt2.grid.dim;
    if dim < 2 {
        return Err("cubic closure targets 2D/3D".into());
    }
    let n = lpt2.grid.size;
    let ncurl = if dim == 2 { 1 } else { 3 };

    let mut div = vec![0.0f64; n];
    let mut curl = vec![vec![0.0f64; n]; ncurl];
    let mut m1 = vec![0.0; 9];
    let mut m2 = vec![0.0; 9];
    for i in 0..n {
        tensor_slice(&lpt2.terms[0], i, dim, &mut m1);
        tensor_slice(&lpt2.terms[1], i, dim, &mut m2);
        let (d, c) = eds_cubic_sources(&m1[..dim * dim], &m2[..dim * dim], dim, opts);
        div[i] = d;
        for (k, ck) in c.iter().enumerate() {
            curl[k][i] = *ck;
        }
    }

    // Longitudinal Helmholtz on the combined cubic divergence.
    let shat = eng.forward_real(&div);
    let psihat_l = lpt2.solve_longitudinal(&shat);
    let mut psi = Vec::with_capacity(dim);
    for a in 0..dim {
        psi.push(eng.inverse_to_real_f32(psihat_l[a].clone()));
    }
    let mut m = if opts.keep_tensors {
        Some(spectral_gradients(eng, &lpt2.grid, &psihat_l))
    } else {
        None
    };

    // Transverse Helmholtz; add into the same cubic displacement.
    let mut curl_f32 = vec![];
    if opts.include_curl {
        let that: Vec<Vec<C64>> = curl.iter().map(|c| eng.forward_real(c)).collect();
        let psihat_t = lpt2.solve_transverse(&that);
        for a in 0..dim {
            let pt = eng.inverse_to_real_f32(psihat_t[a].clone());
            for i in 0..n {
                psi[a][i] += pt[i];
            }
        }
        if opts.keep_tensors {
            let mt = spectral_gradients(eng, &lpt2.grid, &psihat_t);
            if let Some(ref mut mref) = m {
                for c in 0..dim * dim {
                    for i in 0..n {
                        mref[c][i] += mt[c][i];
                    }
                }
            }
        }
        curl_f32 = curl
            .iter()
            .map(|c| c.iter().map(|&v| v as f32).collect())
            .collect();
    }

    let term3 = Term {
        spec: TermSpec {
            order: 3,
            parents: Parents::Mu2(0, 1),
            label: "3:C3".into(),
        },
        psi,
        m,
        div: div.iter().map(|&v| v as f32).collect(),
        curl: curl_f32,
    };

    let mut terms: Vec<Term> = lpt2.terms.iter().take(2).map(clone_term).collect();
    terms.push(term3);
    Ok(Lpt {
        grid: lpt2.grid.clone(),
        order: 3,
        terms,
        growth: Growth::Eds {
            coef: vec![1.0, 1.0, 1.0],
            orders: vec![1, 2, 3],
        },
        cosmo: lpt2.cosmo,
    })
}

fn field_rel_rms(a: &[f32], b: &[f32]) -> f64 {
    let mut num = 0.0;
    let mut den = 0.0;
    for (x, y) in a.iter().zip(b.iter()) {
        let dx = *x as f64 - *y as f64;
        num += dx * dx;
        den += (*y as f64) * (*y as f64);
    }
    if den > 0.0 {
        (num / den).sqrt()
    } else {
        num.sqrt()
    }
}

fn field_max_abs_diff(a: &[f32], b: &[f32]) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (*x as f64 - *y as f64).abs())
        .fold(0.0, f64::max)
}

/// Compare cubic term (and optional full displacement) of two order-3 EdS LPTs.
#[derive(Clone, Debug)]
pub struct CubicParityReport {
    pub rel_rms_psi3: f64,
    pub max_abs_psi3: f64,
    pub rel_rms_div3: f64,
    pub rel_rms_curl3: f64,
    pub rel_rms_m3: f64,
    pub rel_rms_x_at_d: f64,
    pub d_compare: f64,
}

pub fn compare_cubic_terms(reference: &Lpt, candidate: &Lpt, d: f64) -> Result<CubicParityReport, String> {
    if reference.terms.len() < 3 || candidate.terms.len() < 3 {
        return Err("need order-3 merged terms".into());
    }
    let dim = reference.grid.dim;
    let n = reference.grid.size;
    let mut rel_psi: f64 = 0.0;
    let mut max_psi: f64 = 0.0;
    for a in 0..dim {
        rel_psi = rel_psi.max(field_rel_rms(
            &candidate.terms[2].psi[a],
            &reference.terms[2].psi[a],
        ));
        max_psi = max_psi.max(field_max_abs_diff(
            &candidate.terms[2].psi[a],
            &reference.terms[2].psi[a],
        ));
    }
    let rel_div = field_rel_rms(&candidate.terms[2].div, &reference.terms[2].div);
    let mut rel_curl: f64 = 0.0;
    let nc = reference.terms[2].curl.len().max(candidate.terms[2].curl.len());
    for c in 0..nc {
        let z = vec![0.0f32; n];
        let a = candidate.terms[2].curl.get(c).unwrap_or(&z);
        let b = reference.terms[2].curl.get(c).unwrap_or(&z);
        rel_curl = rel_curl.max(field_rel_rms(a, b));
    }
    let mut rel_m: f64 = 0.0;
    if let (Some(ma), Some(mb)) = (&candidate.terms[2].m, &reference.terms[2].m) {
        for c in 0..dim * dim {
            rel_m = rel_m.max(field_rel_rms(&ma[c], &mb[c]));
        }
    }
    let mut xref = vec![0.0f32; n * dim];
    let mut xcan = vec![0.0f32; n * dim];
    reference.positions(d, 3, &mut xref);
    candidate.positions(d, 3, &mut xcan);
    Ok(CubicParityReport {
        rel_rms_psi3: rel_psi,
        max_abs_psi3: max_psi,
        rel_rms_div3: rel_div,
        rel_rms_curl3: rel_curl,
        rel_rms_m3: rel_m,
        rel_rms_x_at_d: field_rel_rms(&xcan, &xref),
        d_compare: d,
    })
}

/// λ-homogeneity: rms(Ψ³[λδ]/λ³ − Ψ³[δ]) / rms(Ψ³[δ]).
pub fn psi3_homogeneity_rel(psi_unit: &[Vec<f32>], psi_scaled: &[Vec<f32>], lambda: f64) -> f64 {
    let scale = lambda.powi(3);
    let mut num = 0.0;
    let mut den = 0.0;
    for (a, b) in psi_unit.iter().zip(psi_scaled.iter()) {
        for (u, s) in a.iter().zip(b.iter()) {
            let e = *s as f64 / scale - *u as f64;
            num += e * e;
            den += (*u as f64) * (*u as f64);
        }
    }
    if den > 0.0 {
        (num / den).sqrt()
    } else {
        0.0
    }
}

/// First shell-crossing diagnostics: time and Lagrangian argmin of J near crossing.
#[derive(Clone, Debug)]
pub struct CrossingReport {
    pub d_cross: f64,
    pub argmin_idx: usize,
    pub j_at_cross: f64,
}

pub fn crossing_report(lpt: &Lpt, order: usize) -> CrossingReport {
    let d_cross = lpt.shell_crossing(order);
    let d = if d_cross.is_finite() {
        d_cross
    } else {
        1.0
    };
    let mut j = vec![0.0; lpt.grid.size];
    lpt.jacobian(d, order, &mut j);
    let (argmin_idx, &j_at_cross) = j
        .iter()
        .enumerate()
        .min_by(|(_, a), (_, b)| a.total_cmp(b))
        .unwrap();
    CrossingReport {
        d_cross,
        argmin_idx,
        j_at_cross,
    }
}

/// Field / deformation comparison of a candidate order-3 LPT vs a reference at growth `d`.
#[derive(Clone, Debug)]
pub struct AblationReport {
    pub label: &'static str,
    pub rel_rms_psi3: f64,
    pub rel_rms_m_total: f64,
    pub rel_rms_j: f64,
    pub d_cross: f64,
    pub d_cross_ref: f64,
    pub rel_d_cross: f64,
    pub argmin_match: bool,
    pub false_early_crossing: bool,
}

pub fn ablation_vs_reference(
    reference: &Lpt,
    candidate: &Lpt,
    label: &'static str,
    d: f64,
) -> AblationReport {
    let dim = reference.grid.dim;
    let n = reference.grid.size;
    let mut rel_psi: f64 = 0.0;
    for a in 0..dim {
        rel_psi = rel_psi.max(field_rel_rms(
            &candidate.terms[2].psi[a],
            &reference.terms[2].psi[a],
        ));
    }
    let mref = reference.displacement_gradient(d, 3);
    let mcan = candidate.displacement_gradient(d, 3);
    let mut rel_m: f64 = 0.0;
    for c in 0..dim * dim {
        let a: Vec<f32> = mcan[c].iter().map(|&v| v as f32).collect();
        let b: Vec<f32> = mref[c].iter().map(|&v| v as f32).collect();
        rel_m = rel_m.max(field_rel_rms(&a, &b));
    }
    let mut jref = vec![0.0; n];
    let mut jcan = vec![0.0; n];
    reference.jacobian(d, 3, &mut jref);
    candidate.jacobian(d, 3, &mut jcan);
    let jref_f: Vec<f32> = jref.iter().map(|&v| v as f32).collect();
    let jcan_f: Vec<f32> = jcan.iter().map(|&v| v as f32).collect();
    let rel_j = field_rel_rms(&jcan_f, &jref_f);
    let cref = crossing_report(reference, 3);
    let ccan = crossing_report(candidate, 3);
    let rel_d = if cref.d_cross.is_finite() && cref.d_cross > 0.0 {
        (ccan.d_cross - cref.d_cross).abs() / cref.d_cross
    } else {
        f64::NAN
    };
    AblationReport {
        label,
        rel_rms_psi3: rel_psi,
        rel_rms_m_total: rel_m,
        rel_rms_j: rel_j,
        d_cross: ccan.d_cross,
        d_cross_ref: cref.d_cross,
        rel_d_cross: rel_d,
        argmin_match: ccan.argmin_idx == cref.argmin_idx,
        false_early_crossing: ccan.d_cross < cref.d_cross * 0.98,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ics::PkShape;
    use crate::Cosmo;

    #[test]
    fn term_counts_match_book() {
        let n3 = dependency_nodes(3, 3);
        assert_eq!(n3.len(), 5, "1 + 1 + 3 raw terms through 3LPT in 3D");
        let kinds: Vec<_> = n3.iter().map(|t| t.kind).collect();
        assert_eq!(kinds, ["linear", "mu2", "mu2", "mu3", "curl"]);
        assert!((n3[1].eds_coef + 3.0 / 7.0).abs() < 1e-12);
        assert!((n3[2].eds_coef - 10.0 / 21.0).abs() < 1e-12);
        assert!((n3[3].eds_coef + 1.0 / 3.0).abs() < 1e-12);
        assert!((n3[4].eds_coef + 1.0 / 7.0).abs() < 1e-12);
        assert_eq!(dependency_nodes(4, 3).len(), 13);
        assert_eq!(dependency_nodes(4, 2).len(), 9);
    }

    #[test]
    fn eds_3lpt_sources_match_invariant_basis() {
        let mut c = Cosmo::new(3, 24, 1.0);
        c.keep_tensors = Some(true);
        c.set_ic_gaussian(PkShape::PowerLaw { n: -1.0 }, 0.08, 7, 1.0);
        c.build_lpt(3);
        let report = assess_invariant_basis(c.lpt_ref()).unwrap();
        assert!(
            report.rel_div3_exact_basis < 5e-6,
            "longitudinal basis residual {}",
            report.rel_div3_exact_basis
        );
        assert!(
            report.rel_curl3_exact_basis < 5e-6,
            "transverse basis residual {}",
            report.rel_curl3_exact_basis
        );
        // Purely local M¹ cubics cannot reproduce μ2(M¹,M²) (nonlocal in M¹).
        assert!(
            report.rel_div3_m1_local_fit > 0.2,
            "M1-only local fit unexpectedly good: {}",
            report.rel_div3_m1_local_fit
        );
    }

    #[test]
    fn cost_grows_with_order() {
        let c2 = cost_model(2, 3);
        let c3 = cost_model(3, 3);
        let c4 = cost_model(4, 3);
        assert!(c3.approx_fft_count > c2.approx_fft_count);
        assert!(c4.approx_fft_count > c3.approx_fft_count);
        assert_eq!(c3.n_terms_unmerged, 5);
        assert_eq!(c4.n_terms_unmerged, 13);
    }

    #[test]
    fn eds_extend_cubic_matches_native_3lpt() {
        let mut c = Cosmo::new(3, 24, 1.0);
        c.keep_tensors = Some(true);
        c.set_ic_gaussian(PkShape::PowerLaw { n: -1.0 }, 0.08, 7, 1.0);
        c.build_lpt(3);
        let native = clone_lpt(c.lpt_ref());

        let mut c2 = Cosmo::new(3, 24, 1.0);
        c2.keep_tensors = Some(true);
        c2.set_ic_gaussian(PkShape::PowerLaw { n: -1.0 }, 0.08, 7, 1.0);
        c2.build_lpt(2);
        let lpt2 = clone_lpt(c2.lpt_ref());
        let built = eds_extend_cubic(&lpt2, &mut c2.eng, CubicClosureOpts::full()).unwrap();
        let parity = compare_cubic_terms(&native, &built, 0.4).unwrap();
        assert!(
            parity.rel_rms_psi3 < 1e-5,
            "Ψ³ rel rms {}",
            parity.rel_rms_psi3
        );
        assert!(
            parity.rel_rms_div3 < 1e-5,
            "div³ rel rms {}",
            parity.rel_rms_div3
        );
        assert!(
            parity.rel_rms_curl3 < 1e-5,
            "curl³ rel rms {}",
            parity.rel_rms_curl3
        );
        assert!(
            parity.rel_rms_x_at_d < 1e-5,
            "x rel rms {}",
            parity.rel_rms_x_at_d
        );
    }

    #[test]
    fn eds_extend_cubic_2d_and_homogeneity() {
        let mut c = Cosmo::new(2, 48, 1.0);
        c.keep_tensors = Some(true);
        c.set_ic_gaussian(PkShape::PowerLaw { n: -1.0 }, 0.08, 3, 1.0);
        c.build_lpt(2);
        let lpt2 = clone_lpt(c.lpt_ref());
        let full = eds_extend_cubic(&lpt2, &mut c.eng, CubicClosureOpts::full()).unwrap();

        let mut c2 = Cosmo::new(2, 48, 1.0);
        c2.keep_tensors = Some(true);
        c2.set_ic_gaussian(PkShape::PowerLaw { n: -1.0 }, 0.08, 3, 1.0);
        let lambda = 1.7f64;
        for v in c2.delta0_hat.iter_mut() {
            *v *= lambda;
        }
        c2.lpt = None;
        c2.build_lpt(2);
        let lpt2s = clone_lpt(c2.lpt_ref());
        let scaled = eds_extend_cubic(&lpt2s, &mut c2.eng, CubicClosureOpts::full()).unwrap();
        let rel = psi3_homogeneity_rel(&full.terms[2].psi, &scaled.terms[2].psi, lambda);
        assert!(rel < 1e-5, "homogeneity residual {rel}");
    }
}
