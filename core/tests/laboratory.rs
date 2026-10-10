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
        assert_eq!(d["guard"]["ok"], true, "dim {dim}: {d}");
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
fn requests_reject_unbounded_allocations_and_non_pre_crossing_states() {
    for s in [
        r#"{"dim":0}"#,
        r#"{"n":100000}"#,
        r#"{"fraction":1.1}"#,
        r#"{"dim":3,"mesh":64}"#,
        r#"{"smoothing":-1}"#,
        r#"{"forceMesh":9999}"#,
        r#"{"ic":"unknown"}"#,
    ] {
        let d: serde_json::Value = serde_json::from_str(&run_json(s)).unwrap();
        assert!(d["error"].is_string(), "{s}: {d}");
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
