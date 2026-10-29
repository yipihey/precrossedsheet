//! Experimental 2LPT→3LPT closure diagnostics (Neural Sheet).
//!
//! This module does **not** replace the exact nLPT solver. It exposes the
//! EdS cubic source decomposition into local deformation-tensor invariants
//! plus the known spectral Poisson/Helmholtz maps, for cheap-closure research.

use crate::growth::{term_specs, Cosmology, Growth, Parents};
use crate::lpt::{mu2, mu3, Lpt};

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
}
