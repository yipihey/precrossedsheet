use hcc_core::{
    eft::{
        continuum_filtered_sigma, fit_sound_speed, match_sigma_ratio, run_dual_smoothing_counterterms,
    },
    laboratory::LabRequest,
};

#[test]
fn sound_speed_fit_recovers_injected_counterterm() {
    // Synthetic: P_UV = P_IR − 2 α k² P_lin with α = 12 (Mpc/h)².
    let alpha = 12.0_f64;
    let k: Vec<f64> = (1..12).map(|i| 0.02 * i as f64).collect();
    let p_lin: Vec<f64> = k.iter().map(|&ki| 1.0e3 / (1.0 + (ki / 0.05).powi(2))).collect();
    let p_ir: Vec<f64> = p_lin
        .iter()
        .zip(k.iter())
        .map(|(&pl, &ki)| pl * (1.0 + 0.15 * ki))
        .collect();
    let p_uv: Vec<f64> = p_ir
        .iter()
        .zip(k.iter())
        .zip(p_lin.iter())
        .map(|((&pir, &ki), &pl)| pir - 2.0 * alpha * ki * ki * pl)
        .collect();
    let nmodes = vec![8.0; k.len()];
    let fit = fit_sound_speed(&k, &p_uv, &p_ir, &p_lin, &nmodes, 0.18).expect("fit");
    assert!(
        (fit.cs2 - alpha).abs() / alpha < 1e-10,
        "recovered cs2={} vs injected {}",
        fit.cs2,
        alpha
    );
    assert!(fit.n_bins >= 5, "expected several fit bins, got {}", fit.n_bins);
    assert!(fit.residual_rms < 1e-8, "residual {}", fit.residual_rms);
}

#[test]
fn ir_filter_matches_one_third_linear_sigma() {
    let r = LabRequest {
        dim: 3,
        ic: "gaussian".into(),
        box_size: 300.,
        ..Default::default()
    };
    let uv = 6.0;
    let sigma_uv = continuum_filtered_sigma(&r, uv, 1.0);
    assert!(sigma_uv > 0.4 && sigma_uv < 0.9, "σ_UV={sigma_uv}");
    let ir = match_sigma_ratio(&r, uv, 1.0 / 3.0, 1.0).expect("IR radius");
    let sigma_ir = continuum_filtered_sigma(&r, ir, 1.0);
    let ratio = sigma_ir / sigma_uv;
    assert!(
        (ratio - 1.0 / 3.0).abs() < 0.01,
        "σ_IR/σ_UV={ratio}, IR={ir}, UV={uv}"
    );
    assert!(ir > uv, "IR filter should be smoother: IR={ir}, UV={uv}");
}

#[test]
fn dual_smoothing_p1_sheet_fits_finite_sound_speed_in_2d() {
    // Draft laboratory contract: two Gaussian smoothings, 1–4LPT, fine P1 deposit.
    // UV near the mild-crossing edge; IR at ∼⅓ the linear σ. Browser-scale 2D grid.
    let base = LabRequest {
        dim: 2,
        n: 32,
        mesh: 64,
        force_mesh: 32,
        ic: "gaussian".into(),
        seed: 42,
        box_size: 300.,
        initial_cutoff: 4,
        epoch: "scaleFactor".into(),
        scale_factor: 1.0,
        fraction: 1.0,
        order: 4,
        steps: 32,
        smoothing: 0.0,
        transfer: "smooth".into(),
        estimators: vec![],
        ..Default::default()
    };
    let uv = 5.0;
    let out = run_dual_smoothing_counterterms(&base, uv, None, 0.08).expect("dual run");
    assert_eq!(out["kind"], "eftCounterterms");
    let ratio = out["filters"]["sigmaRatio"].as_f64().unwrap();
    assert!(
        (ratio - 1.0 / 3.0).abs() < 0.02,
        "sigma ratio {}",
        ratio
    );

    let orders = out["orders"].as_array().unwrap();
    assert_eq!(orders.len(), 4);
    let mut prev_inv = -1.0;
    for entry in orders {
        let order = entry["order"].as_u64().unwrap();
        let inv_uv = entry["uv"]["invertedElementFraction"].as_f64().unwrap();
        let inv_ir = entry["ir"]["invertedElementFraction"].as_f64().unwrap();
        assert!(
            inv_ir <= inv_uv + 1e-12,
            "order {order}: IR crossing {inv_ir} should not exceed UV {inv_uv}"
        );
        if order == 4 {
            // Mild UV crossing; IR nearly uncrossed. Exact fraction is realization-dependent.
            assert!(
                inv_uv < 0.08,
                "4LPT UV inverted fraction unexpectedly large: {inv_uv}"
            );
            assert!(
                inv_ir < 0.005,
                "4LPT IR inverted fraction unexpectedly large: {inv_ir}"
            );
        }
        let fit = &entry["fit"];
        assert!(fit.is_object(), "order {order} missing fit");
        let cs2 = fit["cs2"].as_f64().unwrap();
        assert!(cs2.is_finite(), "order {order} cs2 not finite");
        assert!(
            fit["nBins"].as_u64().unwrap() >= 2,
            "order {order} needs fit bins"
        );
        let k = entry["ir"]["k"].as_array().unwrap();
        let p = entry["ir"]["power"].as_array().unwrap();
        assert!(k.len() >= 4 && k.len() == p.len());
        assert!(
            inv_uv + 1e-15 >= prev_inv,
            "crossing should not drop strongly with order"
        );
        prev_inv = inv_uv;
        let _ = prev_inv;
    }
}

#[test]
fn dual_smoothing_3d_tetrahedra_near_percent_crossing_and_cs2_fit() {
    // Draft laboratory contract on a 32³ sheet / 64³ P1 mesh: UV Rf=5 Mpc/h yields
    // ∼1% inverted Kuhn tetrahedra at a=1 for this seed; IR is auto-matched to σ/3.
    let base = LabRequest {
        dim: 3,
        n: 32,
        mesh: 64,
        force_mesh: 32,
        ic: "gaussian".into(),
        seed: 7,
        box_size: 300.,
        initial_cutoff: 4,
        epoch: "scaleFactor".into(),
        scale_factor: 1.0,
        fraction: 1.0,
        order: 4,
        steps: 32,
        smoothing: 0.0,
        transfer: "smooth".into(),
        estimators: vec![],
        ..Default::default()
    };
    let out = run_dual_smoothing_counterterms(&base, 5.0, None, 0.08).expect("3d dual");
    let o4 = &out["orders"][3];
    let inv = o4["uv"]["invertedElementFraction"].as_f64().unwrap();
    assert!(
        (0.004..0.03).contains(&inv),
        "UV tetrahedron inverted fraction expected near ∼1%: {inv}"
    );
    let inv_ir = o4["ir"]["invertedElementFraction"].as_f64().unwrap();
    assert!(inv_ir < 1e-4, "IR inverted fraction {inv_ir}");
    let cs2 = o4["fit"]["cs2"].as_f64().unwrap();
    assert!(cs2.is_finite() && cs2 < 0.0 && cs2 > -5.0e3, "cs2={cs2}");
    let ratio = out["filters"]["sigmaRatio"].as_f64().unwrap();
    assert!((ratio - 1.0 / 3.0).abs() < 0.02);
    assert!((out["filters"]["uvSmoothing"].as_f64().unwrap() - 5.0).abs() < 1e-12);
}
