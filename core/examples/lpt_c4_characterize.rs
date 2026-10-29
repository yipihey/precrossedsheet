//! Characterization of the explicit EdS 3LPT→4LPT closure C₄.
//!
//! ```sh
//! cargo run --release --manifest-path core/Cargo.toml --example lpt_c4_characterize
//! ```
use hcc_core::growth::Cosmology;
use hcc_core::ics::PkShape;
use hcc_core::lpt_closure::{
    ablation_vs_reference_order, clone_lpt, compare_order_term, cost_model, crossing_report,
    cubic_incremental_fft_count, eds_extend_cubic, eds_extend_quartic, eds_quartic_source_jvp,
    psi_order_homogeneity_rel, quartic_incremental_fft_count, CubicClosureOpts, QuarticClosureOpts,
};
use hcc_core::tangent::LptJvp;
use hcc_core::Cosmo;
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

fn bench(dim: usize, n: usize, seed: u64, repeats: usize) -> Value {
    let mut t3 = Vec::new();
    let mut t4 = Vec::new();
    let mut t3c4 = Vec::new();
    let mut t2c3c4 = Vec::new();
    let mut text4 = Vec::new();
    for r in 0..repeats {
        let s = seed + r as u64;
        let mut c3 = Cosmo::new(dim, n, 1.0);
        c3.keep_tensors = Some(true);
        c3.set_ic_gaussian(PkShape::PowerLaw { n: -1.0 }, 0.08, s, 1.0);
        let (_, ms3) = timed_ms(|| c3.build_lpt(3));
        t3.push(ms3);

        let mut c4 = Cosmo::new(dim, n, 1.0);
        c4.keep_tensors = Some(true);
        c4.set_ic_gaussian(PkShape::PowerLaw { n: -1.0 }, 0.08, s, 1.0);
        let (_, ms4) = timed_ms(|| c4.build_lpt(4));
        t4.push(ms4);

        let mut cc = Cosmo::new(dim, n, 1.0);
        cc.keep_tensors = Some(true);
        cc.set_ic_gaussian(PkShape::PowerLaw { n: -1.0 }, 0.08, s, 1.0);
        cc.build_lpt(3);
        let l3 = clone_lpt(cc.lpt_ref());
        let (_, msc4) = timed_ms(|| {
            eds_extend_quartic(&l3, &mut cc.eng, QuarticClosureOpts::full()).unwrap();
        });
        text4.push(msc4);
        t3c4.push(ms3 + msc4);

        let mut c2 = Cosmo::new(dim, n, 1.0);
        c2.keep_tensors = Some(true);
        c2.set_ic_gaussian(PkShape::PowerLaw { n: -1.0 }, 0.08, s, 1.0);
        let (_, ms2) = timed_ms(|| c2.build_lpt(2));
        let l2 = clone_lpt(c2.lpt_ref());
        let (via3, msc3) = timed_ms(|| eds_extend_cubic(&l2, &mut c2.eng, CubicClosureOpts::full()).unwrap());
        let (_, msc4b) = timed_ms(|| {
            eds_extend_quartic(&via3, &mut c2.eng, QuarticClosureOpts::full()).unwrap();
        });
        t2c3c4.push(ms2 + msc3 + msc4b);
    }
    let c2 = cost_model(2, dim);
    let c3 = cost_model(3, dim);
    let c4 = cost_model(4, dim);
    let inc3 = cubic_incremental_fft_count(dim, CubicClosureOpts::full());
    let inc4 = quartic_incremental_fft_count(dim, QuarticClosureOpts::full());
    let bytes = |terms: usize| {
        let ncells = n.pow(dim as u32);
        let stride = dim + dim * dim + 1 + 3;
        terms * ncells * stride * 4
    };
    json!({
        "dim": dim, "n": n, "repeats": repeats,
        "medianMs": {
            "native3": median(&mut t3),
            "native4": median(&mut t4),
            "native3PlusC4": median(&mut t3c4),
            "native2PlusC3C4": median(&mut t2c3c4),
            "c4ExtendOnly": median(&mut text4),
        },
        "approxFftCount": {
            "native3": c3.approx_fft_count,
            "native4": c4.approx_fft_count,
            "c4Incremental": inc4,
            "native3PlusC4": c3.approx_fft_count + inc4,
            "native2PlusC3C4": c2.approx_fft_count + inc3 + inc4,
        },
        "approxPackedBytes": {
            "native3": bytes(3),
            "native4": bytes(4),
        },
        "speedupVsNative4": {
            "native3PlusC4": median(&mut { let mut t=t4.clone(); let a=median(&mut t); let b=median(&mut t3c4); vec![a/b] }),
        }
    })
}

fn parity_ablation(dim: usize, n: usize, seed: u64) -> Value {
    let cref = setup(dim, n, seed, 4);
    let native = clone_lpt(cref.lpt_ref());
    let d_cross = native.shell_crossing(4);
    let d = 0.5 * d_cross.min(2.0);

    let mut c3 = setup(dim, n, seed, 3);
    let l3 = clone_lpt(c3.lpt_ref());
    let full = eds_extend_quartic(&l3, &mut c3.eng, QuarticClosureOpts::full()).unwrap();
    let no13 = eds_extend_quartic(&l3, &mut c3.eng, QuarticClosureOpts::no_mu2_13()).unwrap();
    let no22 = eds_extend_quartic(&l3, &mut c3.eng, QuarticClosureOpts::no_mu2_22()).unwrap();
    let no_mu3 = eds_extend_quartic(&l3, &mut c3.eng, QuarticClosureOpts::no_mu3_112()).unwrap();
    let no_curl = eds_extend_quartic(&l3, &mut c3.eng, QuarticClosureOpts::no_curl()).unwrap();

    let parity = compare_order_term(&native, &full, 4, d).unwrap();
    let abs = [
        ablation_vs_reference_order(&native, &full, "full_C4", d, 4),
        ablation_vs_reference_order(&native, &no13, "no_mu2_13", d, 4),
        ablation_vs_reference_order(&native, &no22, "no_mu2_22", d, 4),
        ablation_vs_reference_order(&native, &no_mu3, "no_mu3_112", d, 4),
        ablation_vs_reference_order(&native, &no_curl, "no_curl", d, 4),
    ];
    let c3only = setup(dim, n, seed, 3);
    let cross3 = crossing_report(c3only.lpt_ref(), 3);
    let cross4 = crossing_report(&native, 4);

    // 2+C3+C4 chain parity
    let mut c2 = setup(dim, n, seed, 2);
    let l2 = clone_lpt(c2.lpt_ref());
    let via3 = eds_extend_cubic(&l2, &mut c2.eng, CubicClosureOpts::full()).unwrap();
    let via4 = eds_extend_quartic(&via3, &mut c2.eng, QuarticClosureOpts::full()).unwrap();
    let chain = compare_order_term(&native, &via4, 4, d).unwrap();

    json!({
        "dim": dim, "n": n, "dEval": d, "dCrossNative4": d_cross,
        "parityFullC4VsNative": {
            "relRmsPsi4": parity.rel_rms_psi3,
            "maxAbsPsi4": parity.max_abs_psi3,
            "relRmsDiv4": parity.rel_rms_div3,
            "relRmsCurl4": parity.rel_rms_curl3,
            "relRmsM4": parity.rel_rms_m3,
            "relRmsX": parity.rel_rms_x_at_d,
        },
        "parity2C3C4VsNative": {
            "relRmsPsi4": chain.rel_rms_psi3,
            "relRmsX": chain.rel_rms_x_at_d,
        },
        "ablations": abs.iter().map(|a| json!({
            "label": a.label,
            "relRmsPsi4": a.rel_rms_psi3,
            "relRmsMTotal": a.rel_rms_m_total,
            "relRmsJ": a.rel_rms_j,
            "dCross": a.d_cross,
            "relDCross": a.rel_d_cross,
            "argminMatch": a.argmin_match,
            "falseEarlyCrossing": a.false_early_crossing,
        })).collect::<Vec<_>>(),
        "order3Vs4Crossing": {
            "dCross3": cross3.d_cross,
            "dCross4": cross4.d_cross,
            "relD": (cross3.d_cross - cross4.d_cross).abs() / cross4.d_cross,
            "argminSame": cross3.argmin_idx == cross4.argmin_idx,
        }
    })
}

fn homogeneity(dim: usize, n: usize, seed: u64) -> Value {
    let base = setup(dim, n, seed, 3);
    let l3 = clone_lpt(base.lpt_ref());
    let mut eng = setup(dim, n, seed, 3);
    let full = eds_extend_quartic(&l3, &mut eng.eng, QuarticClosureOpts::full()).unwrap();
    let mut rows = Vec::new();
    for &lambda in &[0.5f64, 0.8, 1.25, 2.0] {
        let mut c = Cosmo::new(dim, n, 1.0);
        c.keep_tensors = Some(true);
        c.set_ic_gaussian(PkShape::PowerLaw { n: -1.0 }, 0.08, seed, 1.0);
        for v in c.delta0_hat.iter_mut() {
            *v *= lambda;
        }
        c.lpt = None;
        c.build_lpt(3);
        let ls = clone_lpt(c.lpt_ref());
        let scaled = eds_extend_quartic(&ls, &mut c.eng, QuarticClosureOpts::full()).unwrap();
        let rel = psi_order_homogeneity_rel(&full.terms[3].psi, &scaled.terms[3].psi, lambda, 4);
        rows.push(json!({"lambda": lambda, "relRmsPsi4OverLambda4": rel}));
    }
    json!({"dim": dim, "n": n, "rows": rows})
}

fn jvp_suite(dim: usize, n: usize, seed: u64) -> Value {
    let mut c = setup(dim, n, seed, 3);
    let delta = c.delta0_hat.clone();
    let l3 = clone_lpt(c.lpt_ref());
    let full = eds_extend_quartic(&l3, &mut c.eng, QuarticClosureOpts::full()).unwrap();
    let eps = 1e-4;
    let make = |scale: f64| {
        let mut cx = Cosmo::new(dim, n, 1.0);
        cx.keep_tensors = Some(true);
        cx.set_ic_gaussian(PkShape::PowerLaw { n: -1.0 }, 0.08, seed, 1.0);
        for v in cx.delta0_hat.iter_mut() {
            *v *= scale;
        }
        cx.lpt = None;
        cx.build_lpt(3);
        let l = clone_lpt(cx.lpt_ref());
        eds_extend_quartic(&l, &mut cx.eng, QuarticClosureOpts::full()).unwrap()
    };
    let plus = make(1.0 + eps);
    let minus = make(1.0 - eps);
    let mut num = 0.0;
    let mut den = 0.0;
    for a in 0..dim {
        for i in 0..full.grid.size {
            let fd = (plus.terms[3].psi[a][i] as f64 - minus.terms[3].psi[a][i] as f64) / (2.0 * eps);
            let an = 4.0 * full.terms[3].psi[a][i] as f64;
            num += (fd - an) * (fd - an);
            den += an * an;
        }
    }
    let amp_rel = if den > 0.0 { (num / den).sqrt() } else { 0.0 };

    let jvp = LptJvp::new(&c.grid, &delta, &delta, 4, Cosmology::eds());
    let (_x, _v, dx_native, _) = jvp.evaluate(0.4, 4);
    let mut xp = vec![0.0f32; plus.grid.size * dim];
    let mut xm = vec![0.0f32; minus.grid.size * dim];
    plus.positions(0.4, 4, &mut xp);
    minus.positions(0.4, 4, &mut xm);
    let mut num = 0.0;
    let mut den = 0.0;
    for i in 0..dx_native.len() {
        let fd = (xp[i] as f64 - xm[i] as f64) / (2.0 * eps);
        num += (fd - dx_native[i]) * (fd - dx_native[i]);
        den += dx_native[i] * dx_native[i];
    }
    let pos_rel = if den > 0.0 { (num / den).sqrt() } else { 0.0 };

    // Amplitude product rule at one cell: dM^n = n M^n
    let m1 = full.terms[0].m.as_ref().unwrap();
    let m2 = full.terms[1].m.as_ref().unwrap();
    let m3 = full.terms[2].m.as_ref().unwrap();
    let idx = full.grid.size / 2;
    let mut a = vec![0.0; dim * dim];
    let mut b = vec![0.0; dim * dim];
    let mut e = vec![0.0; dim * dim];
    for c in 0..dim * dim {
        a[c] = m1[c][idx] as f64;
        b[c] = m2[c][idx] as f64;
        e[c] = m3[c][idx] as f64;
    }
    let dm2: Vec<f64> = b.iter().map(|v| 2.0 * v).collect();
    let dm3: Vec<f64> = e.iter().map(|v| 3.0 * v).collect();
    let (dd, _) = eds_quartic_source_jvp(&a, &b, &e, &a, &dm2, &dm3, dim, QuarticClosureOpts::full());
    let (s, _) = hcc_core::lpt_closure::eds_quartic_sources(&a, &b, &e, dim, QuarticClosureOpts::full());
    let ok = (dd - 4.0 * s).abs() <= 1e-9 * (1.0 + s.abs());

    json!({
        "dim": dim, "n": n,
        "amplitudePsi4JvpVsFdRelRms": amp_rel,
        "amplitudePosJvpNative4VsFdC4RelRms": pos_rel,
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
        "scalingNote": {
            "summary": "C₄ collapses 8 unmerged order-4 terms into one combined longitudinal + one transverse Helmholtz after 3LPT. FFT model: native4=176 vs native3+C4=94 vs 2+C3+C4=81 (3D, tensors on). Exact enumeration remains cheap through 4; by ~6–8LPT partition growth likely exceeds a single fused Helmholtz unless further algebraic compaction is found.",
            "cost3d": (2..=6).filter(|&o| o <= 4 || true).filter_map(|o| {
                if o > 4 {
                    // term_specs supports arbitrary order; cost_model uses it
                    let c = cost_model(o, 3);
                    Some(json!({"order": o, "unmerged": c.n_terms_unmerged, "approxFfts": c.approx_fft_count}))
                } else {
                    let c = cost_model(o, 3);
                    Some(json!({"order": o, "unmerged": c.n_terms_unmerged, "approxFfts": c.approx_fft_count}))
                }
            }).collect::<Vec<_>>(),
        }
    });
    // Cleaner cost table 2..=8
    out["scalingNote"]["cost3d"] = json!((2..=8).map(|o| {
        let c = cost_model(o, 3);
        json!({"order": o, "unmerged": c.n_terms_unmerged, "approxFfts": c.approx_fft_count, "tensorFields": c.tensor_field_count})
    }).collect::<Vec<_>>());

    for &(dim, n) in &cases {
        println!("characterizing C4 dim={dim} n={n} ...");
        out["parityAndAblation"]
            .as_array_mut()
            .unwrap()
            .push(parity_ablation(dim, n, 7));
        out["homogeneity"]
            .as_array_mut()
            .unwrap()
            .push(homogeneity(dim, n, 7));
        out["jvp"].as_array_mut().unwrap().push(jvp_suite(dim, n, 7));
        out["bench"]
            .as_array_mut()
            .unwrap()
            .push(bench(dim, n, 7, 5));
    }
    println!("{}", serde_json::to_string_pretty(&out).unwrap());
}
