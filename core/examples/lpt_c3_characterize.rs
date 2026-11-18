//! Rigorous characterization of the explicit EdS 2LPT→3LPT closure C₃.
//!
//! ```sh
//! cargo run --release --manifest-path core/Cargo.toml --example lpt_c3_characterize
//! ```
use hcc_core::ics::PkShape;
use hcc_core::lpt_closure::{
    ablation_vs_reference, clone_lpt, compare_cubic_terms, cost_model, crossing_report,
    cubic_incremental_fft_count, eds_cubic_source_jvp, eds_extend_cubic, psi3_homogeneity_rel,
    CubicClosureOpts,
};
use hcc_core::tangent::LptJvp;
use hcc_core::Cosmo;
use hcc_core::growth::Cosmology;
use serde_json::{json, Value};
use std::time::Instant;

fn setup(dim: usize, n: usize, seed: u64, order: usize) -> Cosmo {
    let mut c = Cosmo::new(dim, n, 1.0);
    c.keep_tensors = Some(true);
    c.set_ic_gaussian(PkShape::PowerLaw { n: -1.0 }, 0.08, seed, 1.0);
    c.build_lpt(order);
    c
}

fn timed_ms<T>(f: impl FnOnce() -> T) -> (T, f64) {
    let t0 = Instant::now();
    let v = f();
    (v, t0.elapsed().as_secs_f64() * 1e3)
}

fn median(xs: &mut [f64]) -> f64 {
    xs.sort_by(|a, b| a.total_cmp(b));
    xs[xs.len() / 2]
}

fn bench_builds(dim: usize, n: usize, seed: u64, repeats: usize) -> Value {
    let mut t2 = Vec::new();
    let mut t3 = Vec::new();
    let mut t_c3 = Vec::new();
    for r in 0..repeats {
        let mut c = Cosmo::new(dim, n, 1.0);
        c.keep_tensors = Some(true);
        c.set_ic_gaussian(PkShape::PowerLaw { n: -1.0 }, 0.08, seed + r as u64, 1.0);
        let (_, ms2) = timed_ms(|| c.build_lpt(2));
        t2.push(ms2);

        let mut c3 = Cosmo::new(dim, n, 1.0);
        c3.keep_tensors = Some(true);
        c3.set_ic_gaussian(PkShape::PowerLaw { n: -1.0 }, 0.08, seed + r as u64, 1.0);
        let (_, ms3) = timed_ms(|| c3.build_lpt(3));
        t3.push(ms3);

        let mut cc = Cosmo::new(dim, n, 1.0);
        cc.keep_tensors = Some(true);
        cc.set_ic_gaussian(PkShape::PowerLaw { n: -1.0 }, 0.08, seed + r as u64, 1.0);
        cc.build_lpt(2);
        let lpt2 = clone_lpt(cc.lpt_ref());
        let (_, msc) = timed_ms(|| {
            eds_extend_cubic(&lpt2, &mut cc.eng, CubicClosureOpts::full()).unwrap();
        });
        // Wall for 2LPT + C₃ = build2 + extend (extend-only also reported)
        t_c3.push(ms2 + msc);
    }
    // Separate extend-only median
    let mut text = Vec::new();
    for r in 0..repeats {
        let mut cc = Cosmo::new(dim, n, 1.0);
        cc.keep_tensors = Some(true);
        cc.set_ic_gaussian(PkShape::PowerLaw { n: -1.0 }, 0.08, seed + r as u64, 1.0);
        cc.build_lpt(2);
        let lpt2 = clone_lpt(cc.lpt_ref());
        let (_, msc) = timed_ms(|| {
            eds_extend_cubic(&lpt2, &mut cc.eng, CubicClosureOpts::full()).unwrap();
        });
        text.push(msc);
    }
    let c2 = cost_model(2, dim);
    let c3 = cost_model(3, dim);
    let inc = cubic_incremental_fft_count(dim, CubicClosureOpts::full());
    // Rough tensor memory: f32 fields × 4 bytes
    let bytes_term = |order_terms: usize| {
        let ncells = n.pow(dim as u32);
        // psi(d) + M(d²) + div + curl(~3)
        let stride = dim + dim * dim + 1 + 3;
        order_terms * ncells * stride * 4
    };
    json!({
        "dim": dim, "n": n, "repeats": repeats,
        "medianMs": {
            "native2": median(&mut t2),
            "native3": median(&mut t3),
            "native2PlusC3": median(&mut t_c3),
            "c3ExtendOnly": median(&mut text),
        },
        "approxFftCount": {
            "native2": c2.approx_fft_count,
            "native3": c3.approx_fft_count,
            "c3Incremental": inc,
            "native2PlusC3": c2.approx_fft_count + inc,
        },
        "approxPackedBytes": {
            "native2": bytes_term(c2.n_terms_merged_eds),
            "native3": bytes_term(c3.n_terms_merged_eds),
            "note": "merged EdS term packing estimate (psi+M+sources); not RSS"
        }
    })
}

fn homogeneity_suite(dim: usize, n: usize, seed: u64) -> Value {
    let base = setup(dim, n, seed, 2);
    let lpt2 = clone_lpt(base.lpt_ref());
    let mut eng_owner = setup(dim, n, seed, 2);
    let full = eds_extend_cubic(&lpt2, &mut eng_owner.eng, CubicClosureOpts::full()).unwrap();
    let mut rows = Vec::new();
    for &lambda in &[0.5f64, 0.8, 1.25, 2.0] {
        let mut c = Cosmo::new(dim, n, 1.0);
        c.keep_tensors = Some(true);
        c.set_ic_gaussian(PkShape::PowerLaw { n: -1.0 }, 0.08, seed, 1.0);
        for v in c.delta0_hat.iter_mut() {
            *v *= lambda;
        }
        c.lpt = None;
        c.build_lpt(2);
        let l2 = clone_lpt(c.lpt_ref());
        let scaled = eds_extend_cubic(&l2, &mut c.eng, CubicClosureOpts::full()).unwrap();
        let rel = psi3_homogeneity_rel(&full.terms[2].psi, &scaled.terms[2].psi, lambda);
        rows.push(json!({"lambda": lambda, "relRmsPsi3OverLambda3": rel}));
    }
    json!({"dim": dim, "n": n, "rows": rows})
}

fn ablation_suite(dim: usize, n: usize, seed: u64) -> Value {
    let cref = setup(dim, n, seed, 3);
    let native = clone_lpt(cref.lpt_ref());
    let d_cross = native.shell_crossing(3);
    let d = 0.5 * d_cross.min(2.0);

    let mut c2 = setup(dim, n, seed, 2);
    let l2 = clone_lpt(c2.lpt_ref());
    let full = eds_extend_cubic(&l2, &mut c2.eng, CubicClosureOpts::full()).unwrap();
    let no_mu3 = eds_extend_cubic(&l2, &mut c2.eng, CubicClosureOpts::no_mu3()).unwrap();
    let no_curl = eds_extend_cubic(&l2, &mut c2.eng, CubicClosureOpts::no_curl()).unwrap();

    let parity = compare_cubic_terms(&native, &full, d).unwrap();
    let ablations = [
        ablation_vs_reference(&native, &full, "full_C3", d),
        ablation_vs_reference(&native, &no_mu3, "no_mu3", d),
        ablation_vs_reference(&native, &no_curl, "no_curl", d),
    ];
    let c2only = setup(dim, n, seed, 2);
    let cross2 = crossing_report(c2only.lpt_ref(), 2);
    let cross3 = crossing_report(&native, 3);
    json!({
        "dim": dim, "n": n, "dEval": d, "dCrossNative3": d_cross,
        "parityFullC3VsNative": {
            "relRmsPsi3": parity.rel_rms_psi3,
            "maxAbsPsi3": parity.max_abs_psi3,
            "relRmsDiv3": parity.rel_rms_div3,
            "relRmsCurl3": parity.rel_rms_curl3,
            "relRmsM3": parity.rel_rms_m3,
            "relRmsX": parity.rel_rms_x_at_d,
        },
        "ablations": ablations.iter().map(|a| json!({
            "label": a.label,
            "relRmsPsi3": a.rel_rms_psi3,
            "relRmsMTotal": a.rel_rms_m_total,
            "relRmsJ": a.rel_rms_j,
            "dCross": a.d_cross,
            "dCrossRef": a.d_cross_ref,
            "relDCross": a.rel_d_cross,
            "argminMatch": a.argmin_match,
            "falseEarlyCrossing": a.false_early_crossing,
        })).collect::<Vec<_>>(),
        "order2Vs3Crossing": {
            "dCross2": cross2.d_cross,
            "dCross3": cross3.d_cross,
            "argmin2": cross2.argmin_idx,
            "argmin3": cross3.argmin_idx,
            "relD": (cross2.d_cross - cross3.d_cross).abs() / cross3.d_cross,
        },
    })
}

fn jvp_suite(dim: usize, n: usize, seed: u64) -> Value {
    let mut c = setup(dim, n, seed, 2);
    let delta = c.delta0_hat.clone();
    // Amplitude direction = δ itself ⇒ analytic cubic tangent is 3 Ψ³ at D=1 shape.
    let l2 = clone_lpt(c.lpt_ref());
    let full = eds_extend_cubic(&l2, &mut c.eng, CubicClosureOpts::full()).unwrap();
    let eps = 1e-4;
    let mut cp = Cosmo::new(dim, n, 1.0);
    cp.keep_tensors = Some(true);
    cp.set_ic_gaussian(PkShape::PowerLaw { n: -1.0 }, 0.08, seed, 1.0);
    for v in cp.delta0_hat.iter_mut() {
        *v *= 1.0 + eps;
    }
    cp.lpt = None;
    cp.build_lpt(2);
    let lp = clone_lpt(cp.lpt_ref());
    let plus = eds_extend_cubic(&lp, &mut cp.eng, CubicClosureOpts::full()).unwrap();

    let mut cm = Cosmo::new(dim, n, 1.0);
    cm.keep_tensors = Some(true);
    cm.set_ic_gaussian(PkShape::PowerLaw { n: -1.0 }, 0.08, seed, 1.0);
    for v in cm.delta0_hat.iter_mut() {
        *v *= 1.0 - eps;
    }
    cm.lpt = None;
    cm.build_lpt(2);
    let lm = clone_lpt(cm.lpt_ref());
    let minus = eds_extend_cubic(&lm, &mut cm.eng, CubicClosureOpts::full()).unwrap();

    // FD of Ψ³ vs 3 Ψ³ (amplitude JVP)
    let mut num = 0.0;
    let mut den = 0.0;
    for a in 0..dim {
        for i in 0..full.grid.size {
            let fd = (plus.terms[2].psi[a][i] as f64 - minus.terms[2].psi[a][i] as f64) / (2.0 * eps);
            let an = 3.0 * full.terms[2].psi[a][i] as f64;
            let e = fd - an;
            num += e * e;
            den += an * an;
        }
    }
    let amp_rel = if den > 0.0 { (num / den).sqrt() } else { 0.0 };

    // Amplitude IC direction: native LptJvp (θ in e^θ δ) matches d/dλ|λ=1 of λδ.
    let dir = delta.clone();
    let jvp = LptJvp::new(&c.grid, &delta, &dir, 3, Cosmology::eds());
    let (_x, _v, dx_native, _) = jvp.evaluate(0.5, 3);

    // FD of C₃ positions under δ → (1±ε)δ
    let mut xp = vec![0.0f32; plus.grid.size * dim];
    let mut xm = vec![0.0f32; minus.grid.size * dim];
    plus.positions(0.5, 3, &mut xp);
    minus.positions(0.5, 3, &mut xm);
    let mut num = 0.0;
    let mut den = 0.0;
    for i in 0..dx_native.len() {
        let fd = (xp[i] as f64 - xm[i] as f64) / (2.0 * eps);
        // Amplitude direction: native JVP is d/dθ at θ=0 of exp(θ)δ, while FD is d/dλ at λ=1 of λδ.
        // For λ = e^θ, d/dλ = (1/λ) d/dθ; at λ=1 they match for amplitude.
        let e = fd - dx_native[i];
        num += e * e;
        den += dx_native[i] * dx_native[i];
    }
    let pos_rel = if den > 0.0 { (num / den).sqrt() } else { 0.0 };

    // Source-level product rule smoke: rebuild d(div) at one cell from tensors
    let m1 = full.terms[0].m.as_ref().unwrap();
    let m2 = full.terms[1].m.as_ref().unwrap();
    let mut ok = true;
    let idx = n / 2 * full.grid.n.pow((dim - 1) as u32) + n / 2;
    if idx < full.grid.size {
        let mut a = vec![0.0; dim * dim];
        let mut b = vec![0.0; dim * dim];
        for c in 0..dim * dim {
            a[c] = m1[c][idx] as f64;
            b[c] = m2[c][idx] as f64;
        }
        // Under δ→λδ: M¹→λM¹, M²→λ²M² ⇒ at λ=1, dM¹=M¹ and dM²=2M².
        let dm2: Vec<f64> = b.iter().map(|v| 2.0 * v).collect();
        let (dd, _) =
            eds_cubic_source_jvp(&a, &b, &a, &dm2, dim, CubicClosureOpts::full());
        // Amplitude: d/dλ|₁ sources = 3 × sources
        let (s, _) =
            hcc_core::lpt_closure::eds_cubic_sources(&a, &b, dim, CubicClosureOpts::full());
        if (dd - 3.0 * s).abs() > 1e-9 * (1.0 + s.abs()) {
            ok = false;
        }
    }

    json!({
        "dim": dim, "n": n,
        "amplitudePsi3JvpVsFdRelRms": amp_rel,
        "amplitudePosJvpNative3VsFdC3RelRms": pos_rel,
        "sourceProductRuleAmplitudeOk": ok,
        "eps": eps,
    })
}

fn main() {
    let cases = [(2usize, 64usize), (3usize, 24usize), (3usize, 32usize)];
    let mut out = json!({
        "parityAndAblation": [],
        "homogeneity": [],
        "jvp": [],
        "bench": [],
        "higherOrderNote": {
            "summary": "Explicit C₃ collapses 3 unmerged cubic terms into one combined longitudinal + one transverse Helmholtz. At 4LPT the unmerged count jumps to 13 with ~176 FFTs (tensors on); an analogous distilled operator would need a growing invariant/spectral basis or learned remainder. Structure helps through 3; beyond that term combinatorics dominate unless a compact nonlocal basis is found.",
            "cost3d": (2..=4).map(|o| {
                let c = cost_model(o, 3);
                json!({"order": o, "unmerged": c.n_terms_unmerged, "approxFfts": c.approx_fft_count})
            }).collect::<Vec<_>>(),
        }
    });
    for &(dim, n) in &cases {
        println!("characterizing dim={dim} n={n} ...");
        out["parityAndAblation"]
            .as_array_mut()
            .unwrap()
            .push(ablation_suite(dim, n, 7));
        out["homogeneity"]
            .as_array_mut()
            .unwrap()
            .push(homogeneity_suite(dim, n, 7));
        out["jvp"].as_array_mut().unwrap().push(jvp_suite(dim, n, 7));
        out["bench"]
            .as_array_mut()
            .unwrap()
            .push(bench_builds(dim, n, 7, 5));
    }
    println!("{}", serde_json::to_string_pretty(&out).unwrap());
}
