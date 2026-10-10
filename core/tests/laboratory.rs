use hcc_core::{
    grid::Grid,
    laboratory::{run, run_json, LabRequest},
    observables,
};
#[test]
fn checked_lab_conserves_mass_and_has_fixed_time_gradients_in_all_dimensions() {
    for dim in 1..=3 {
        let r = LabRequest {
            dim,
            n: if dim == 3 { 24 } else { 32 },
            force_mesh: if dim == 3 { 24 } else { 32 },
            fraction: 0.35,
            ..Default::default()
        };
        let d = run(&r).unwrap();
        assert_eq!(d["guard"]["ok"], true, "dim {dim}: {}", d["guard"]);
        for key in ["mass", "pmMass"] {
            assert!(
                (d["diagnostics"][key].as_f64().unwrap() - 1.).abs() < 1e-6,
                "dim {dim} {key}"
            );
        }
        let err = d["gradient"]["relativeError"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_f64().unwrap())
            .fold(0., f64::max);
        assert!(err < 1e-5, "dim {dim}: amplitude power tangent err {err}");
        assert!(d["pdf"]["capturedVolume"].as_f64().unwrap() > 0.999);
        assert!(d["guard"]["minimumSimplexJacobian"].as_f64().unwrap() > 0.);
    }
}
#[test]
fn requests_reject_unbounded_allocations_and_invalid_parameters() {
    for s in [
        r#"{"dim":0}"#,
        r#"{"n":100000}"#,
        r#"{"fraction":-1}"#,
        r#"{"dim":3,"n":128}"#,
        r#"{"smoothing":-1}"#,
        r#"{"forceMesh":9999}"#,
        r#"{"ic":"unknown"}"#,
    ] {
        let d: serde_json::Value = serde_json::from_str(&run_json(s)).unwrap();
        assert!(
            d["error"].is_string(),
            "{s}: expectederror,guard={}",
            d["guard"]
        );
        assert!(d.get("rho").is_none());
    }
    assert!(
        serde_json::from_str::<serde_json::Value>(&run_json("bad JSON")).unwrap()["error"]
            .is_string()
    );
}
#[test]
fn periodic_intervals_have_exact_mass_and_constant_translation_mode() {
    let g = Grid::new(1, 16, 1.);
    let x: Vec<f64> = (0..16).map(|i| i as f64 / 16. + 0.123).collect();
    let rho = observables::interval_density(&g, &x, 48);
    assert!(rho.iter().all(|&r| (r - 1.).abs() < 1e-6));
    let (dc, ddc) = observables::sheet_mode_jvp(&g, &x, &vec![0.5; 16], [0.; 3]);
    assert!((dc.re - 1.).abs() < 1e-12 && ddc.norm() < 1e-12);
    let (k, dk) =
        observables::sheet_mode_jvp(&g, &x, &vec![0.5; 16], [2. * std::f64::consts::PI, 0., 0.]);
    assert!(k.norm() < 1e-12 && dk.norm() < 1e-12);
}
#[test]
fn log_density_pdfs_conserve_probability_and_use_raw_mass_after_smoothing() {
    let values = [0.5, 0.5, 1.5, 1.5];
    let raw = [0.2, 0.4, 1.6, 1.8];
    let (_, v, m, mean, var) = observables::density_pdf(&values, &raw, 80);
    let dy = 3.6 / 80.;
    assert!((v.iter().sum::<f64>() * dy - 1.).abs() < 1e-12);
    assert!((m.iter().sum::<f64>() * dy - 1.).abs() < 1e-12);
    assert!((mean - 1.).abs() < 1e-12 && (var - 0.25).abs() < 1e-12);
    let mass_low = m.iter().take(40).sum::<f64>() * dy;
    assert!(
        (mass_low - 0.15).abs() < 1e-7,
        "must weight with raw mass, not smoothed values"
    );
}

#[test]
fn crossed_laboratory_keeps_fields_and_conserves_stream_mass() {
    for dim in 1..=3 {
        let r = LabRequest {
            dim,
            n: 16,
            force_mesh: 16,
            mesh: 32,
            fraction: 1.2,
            omega_m: 1.,
            steps: 128,
            ..Default::default()
        };
        let d = run(&r).unwrap();
        assert_eq!(d["guard"]["ok"], false, "dimension {dim}");
        assert_eq!(d["rho"].as_array().unwrap().len(), 32usize.pow(dim as u32));
        for key in ["mass", "pmMass"] {
            assert!(
                (d["diagnostics"][key].as_f64().unwrap() - 1.).abs() < 2e-6,
                "{dim} {key}"
            );
        }
        assert!(d["spectra"]["direct"]
            .as_array()
            .unwrap()
            .iter()
            .all(|v| v.as_f64().unwrap().is_finite()));
        assert_eq!(d["request"]["fraction"], 1.2);
    }
}
#[test]
fn physical_epoch_does_not_depend_on_crossing_reference_and_large_grids_are_accepted() {
    let r = LabRequest {
        dim: 2,
        n: 128,
        force_mesh: 128,
        mesh: 128,
        steps: 32,
        epoch: "scaleFactor".into(),
        scale_factor: 0.25,
        ..Default::default()
    };
    let d = run(&r).unwrap();
    assert_eq!(d["a"], 0.25);
    assert_eq!(d["rho"].as_array().unwrap().len(), 128 * 128);
    assert!((d["diagnostics"]["mass"].as_f64().unwrap() - 1.).abs() < 1e-6);
    assert!(d["D"].as_f64().unwrap() > 0.24 && d["D"].as_f64().unwrap() < 0.25);
    assert!(
        d["spectra"]["directK"][0].as_f64().unwrap() - 2. * std::f64::consts::PI / 300. < 1e-12
    );
}
#[test]
fn collapsed_and_inverted_periodic_intervals_keep_positive_mass() {
    let g = Grid::new(1, 16, 1.);
    let x: Vec<f64> = (0..16)
        .map(|i| {
            let q = i as f64 / 16.;
            q - 0.3 * (2. * std::f64::consts::PI * q).sin()
        })
        .collect();
    let rho = observables::interval_density(&g, &x, 64);
    assert!(rho.iter().all(|&r| r >= 0.));
    assert!((rho.iter().map(|&r| r as f64).sum::<f64>() / 64. - 1.).abs() < 1e-7);
    let collapsed = observables::interval_density(&g, &vec![0.; 16], 64);
    assert!((collapsed.iter().map(|&r| r as f64).sum::<f64>() / 64. - 1.).abs() < 1e-7);
}

#[test]
fn full_field_quantiles_use_raw_mass_and_moments_ignore_histogram_clipping() {
    let d = observables::density_summary(&[0.5, 0.5, 1.5, 1.5], &[0.2, 0.4, 1.6, 1.8], &[0.5]);
    assert_eq!(d["volumeQuantiles"][0], 0.5);
    assert_eq!(d["massQuantiles"][0], 1.5);
    assert_eq!(d["variance"], 0.25);
    assert_eq!(d["S3"], 0.);
    assert_eq!(d["S4"], -8.);
    let g = Grid::new(1, 16, 1.);
    let flipped: Vec<f64> = (0..16).map(|i| -(i as f64) / 16.).collect();
    assert_eq!(observables::inverted_mass_fraction(&g, &flipped), 15. / 16.);
}
#[test]
fn complete_response_and_manifest_replay_the_executed_request() {
    let request = LabRequest {
        dim: 1,
        n: 16,
        force_mesh: 16,
        steps: 32,
        epoch: "scaleFactor".into(),
        scale_factor: 0.25,
        ..Default::default()
    };
    let d = run(&request).unwrap();
    let manifest = serde_json::json!({"schemaVersion":2,"state":{"n":64},"request":{"n":64},"result":d});
    let replay: serde_json::Value = serde_json::from_str(&run_json(&manifest.to_string())).unwrap();
    assert_eq!(replay["request"]["n"], 16);
    assert_eq!(replay["rho"], manifest["result"]["rho"]);
    assert_eq!(replay["a"], 0.25);
    let invalid: serde_json::Value =
        serde_json::from_str(&run_json(r#"{"schemaVersion":99}"#)).unwrap();
    assert!(invalid["error"].is_string());
}

#[test]
fn physical_epoch_allows_no_finite_crossing_reference() {
    let r = LabRequest {
        dim: 1,
        n: 32,
        force_mesh: 32,
        ic: "gaussian".into(),
        box_size: 50.,
        initial_smoothing: 200.,
        epoch: "scaleFactor".into(),
        scale_factor: 0.25,
        ..Default::default()
    };
    let d = run(&r).unwrap();
    assert_eq!(d["guard"]["ok"], true);
    assert!(d["shellCrossing"].is_null());
    assert!(d["aShellCrossing"].is_null());
    assert_eq!(d["a"], 0.25);
    assert!((d["diagnostics"]["mass"].as_f64().unwrap() - 1.).abs() < 1e-7);
}
