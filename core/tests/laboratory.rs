use hcc_core::{
    grid::Grid,
    laboratory::{exact_sheet_density, prepare_pending, run, run_json, LabRequest, LabSession},
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
fn lab_session_reuses_physical_trajectory_and_raw_observations_with_exact_keys() {
    let request = LabRequest {
        dim: 2,
        n: 16,
        mesh: 16,
        force_mesh: 16,
        steps: 32,
        ..Default::default()
    };
    let mut session = LabSession::new();
    let mut pending = session.prepare_pending(&request).unwrap();
    assert_eq!(pending.metadata()["pmArithmetic"], "f32");
    assert_eq!(session.cache_status()["initial"]["status"], "miss");
    assert_eq!(session.cache_status()["pending"]["status"], "miss");
    let first = session.finish_cpu(&mut pending).unwrap();
    let channels = first.direct_cpu();
    let first_result = first.finish(&channels, r#"{"backend":"cpu"}"#).unwrap();
    assert_eq!(first_result["cache"]["pm"]["arithmetic"], "f64");
    let cache = &first_result["cache"];
    let retained_stages = ["initial", "pending", "pm", "observations"]
        .iter()
        .map(|stage| cache[*stage]["bytes"].as_u64().unwrap_or(0))
        .sum::<u64>();
    assert!(retained_stages > 0);
    assert_eq!(
        cache["bytes"].as_u64().unwrap(),
        retained_stages,
        "cache={cache}"
    );

    let changed_smoothing = LabRequest {
        smoothing: 0.07,
        ..request.clone()
    };
    let cached = session
        .try_cached(&changed_smoothing, "f64")
        .unwrap()
        .unwrap();
    let cached_result = cached
        .finish(&cached.direct_cpu(), r#"{"backend":"cpu"}"#)
        .unwrap();
    assert_eq!(cached_result["request"]["smoothing"], 0.07);
    assert_eq!(cached_result["cache"]["initial"]["status"], "hit");
    assert_eq!(cached_result["cache"]["pending"]["status"], "hit");
    assert_eq!(cached_result["cache"]["pm"]["status"], "hit");
    assert_eq!(cached_result["cache"]["observations"]["status"], "hit");
    assert!(cached
        .lpt_displacement_hi()
        .iter()
        .zip(cached.lpt_displacement())
        .all(|(hi, x)| hi == x));
    let positions = cached_result["x"].as_array().unwrap();
    for particle in 0..16 * 16 {
        let ix = particle / 16;
        let iy = particle % 16;
        for axis in 0..2 {
            let q = (if axis == 0 { ix } else { iy }) as f64 / 16.0;
            let packed = particle * 4 + axis;
            let expected = positions[particle * 2 + axis].as_f64().unwrap() - q;
            let reconstructed = cached.lpt_displacement_hi()[packed] as f64
                + cached.lpt_displacement_lo()[packed] as f64;
            assert!((reconstructed - expected).abs() < 1e-10);
        }
    }
    assert!(cached
        .lpt_direction_hi()
        .iter()
        .zip(cached.lpt_direction())
        .all(|(hi, x)| hi == x));
    assert!(cached.lpt_direction_lo().iter().all(|x| x.is_finite()));

    let changed_observation = LabRequest {
        order: 3,
        mesh: 24,
        ..request.clone()
    };
    let recalculated = session
        .try_cached(&changed_observation, "f64")
        .unwrap()
        .unwrap();
    let recalculated_result = recalculated
        .finish(&recalculated.direct_cpu(), r#"{"backend":"cpu"}"#)
        .unwrap();
    assert_eq!(recalculated_result["request"]["order"], 3);
    assert_eq!(recalculated_result["request"]["mesh"], 24);
    assert_eq!(recalculated_result["cache"]["pm"]["status"], "hit");
    assert_eq!(recalculated_result["cache"]["pending"]["status"], "miss");
    assert_eq!(
        recalculated_result["cache"]["observations"]["status"],
        "miss"
    );

    let same_epoch = LabRequest {
        epoch: "scaleFactor".into(),
        scale_factor: first_result["a"].as_f64().unwrap(),
        ..request.clone()
    };
    let same_epoch_cached = session.try_cached(&same_epoch, "f64").unwrap().unwrap();
    let same_epoch_result = same_epoch_cached
        .finish(&same_epoch_cached.direct_cpu(), r#"{"backend":"cpu"}"#)
        .unwrap();
    assert_eq!(same_epoch_result["request"]["epoch"], "scaleFactor");
    assert_eq!(same_epoch_result["cache"]["initial"]["status"], "hit");
    assert_eq!(same_epoch_result["cache"]["pending"]["status"], "miss");
    assert_eq!(same_epoch_result["cache"]["pm"]["status"], "hit");

    // The PM path is part of the key: an f32 request cannot consume an f64
    // trajectory. An invalid request also leaves the completed entry intact.
    assert!(session
        .try_cached(&changed_smoothing, "f32")
        .unwrap()
        .is_none());
    assert!(session
        .try_cached(
            &LabRequest {
                force_mesh: 24,
                ..request.clone()
            },
            "f64"
        )
        .unwrap()
        .is_none());
    assert!(session
        .prepare_pending(&LabRequest {
            n: 999,
            ..request.clone()
        })
        .is_err());
    assert!(session.try_cached(&request, "f64").unwrap().is_some());
    assert!(session.prepare_initial_slot(&request, "cpu-f64").unwrap());

    let mut gpu_only = LabSession::new();
    assert!(gpu_only
        .try_cached_source(&request, "f64", "gpu-f32")
        .unwrap()
        .is_none());
    assert!(!gpu_only.prepare_initial_slot(&request, "gpu-f32").unwrap());
    assert_eq!(gpu_only.cache_status()["initial"]["status"], "miss");
    assert!(gpu_only
        .prepare_pending_source(&request, "gpu-f32")
        .is_err());
    assert_eq!(gpu_only.cache_status()["initial"]["status"], "miss");

    let changed_seed = LabRequest {
        seed: request.seed + 1,
        ic: "gaussian".into(),
        initial_cutoff: 1,
        epoch: "scaleFactor".into(),
        scale_factor: 0.5,
        ..request.clone()
    };
    let _pending = session.prepare_pending(&changed_seed).unwrap();
    assert_eq!(session.cache_status()["initial"]["status"], "miss");

    session.clear();
    assert_eq!(session.cache_status()["bytes"], 0);
    assert!(session.try_cached(&request, "f64").unwrap().is_none());
}

#[test]
fn external_density_finish_matches_exact_reference_and_rejects_invalid_mass_without_poisoning_cache(
) {
    let request = LabRequest {
        dim: 2,
        n: 16,
        mesh: 16,
        force_mesh: 16,
        steps: 32,
        ..Default::default()
    };
    let mut session = LabSession::new();
    let mut pending = session.prepare_pending(&request).unwrap();
    let lpt = pending.lpt_positions();
    let rho = exact_sheet_density(2, 16, 16, &lpt).unwrap();
    let pm_reference = pending.pm_reference().unwrap();
    let pm_positions = pm_reference[..16 * 16 * 4]
        .iter()
        .map(|&x| x as f32)
        .collect::<Vec<_>>();
    let pmrho = exact_sheet_density(2, 16, 16, &pm_positions).unwrap();

    let mut bad_length = rho.clone();
    bad_length.pop();
    let err = session
        .finish_cpu_observed(
            &mut pending,
            &bad_length,
            &pmrho,
            "gpu-f32-geometry-overlap",
        )
        .err()
        .expect("short external density must be rejected");
    assert!(err.contains("rho must contain"));
    let mut bad_value = rho.clone();
    bad_value[0] = f32::NAN;
    let err = session
        .finish_cpu_observed(&mut pending, &bad_value, &pmrho, "gpu-f32-geometry-overlap")
        .err()
        .expect("nonfinite external density must be rejected");
    assert!(err.contains("finite nonnegative"));
    let mut bad_mass = rho.clone();
    bad_mass[0] += 0.01;
    let err = session
        .finish_cpu_observed(&mut pending, &bad_mass, &pmrho, "gpu-f32-geometry-overlap")
        .err()
        .expect("nonconserved external density must be rejected");
    assert!(err.contains("mean must be within 1e-6"));
    let external = session
        .finish_cpu_observed(&mut pending, &rho, &pmrho, "gpu-f32-geometry-overlap")
        .unwrap();
    let external_json = external
        .finish(&external.direct_cpu(), r#"{"backend":"cpu"}"#)
        .unwrap();
    assert_eq!(
        external_json["cache"]["observations"]["depositSource"],
        "gpu-f32-geometry-overlap"
    );
    assert_eq!(external_json["cache"]["observations"]["status"], "miss");
    let externally_cached = session
        .try_cached_observed_source(&request, "f64", "cpu-f64", "gpu-f32-geometry-overlap")
        .unwrap()
        .expect("matching external deposit should be reusable");
    let externally_cached_json = externally_cached
        .finish(&externally_cached.direct_cpu(), r#"{"backend":"cpu"}"#)
        .unwrap();
    assert_eq!(
        externally_cached_json["cache"]["observations"]["depositSource"],
        "gpu-f32-geometry-overlap"
    );

    let mut cpu_session = LabSession::new();
    let mut cpu_pending = cpu_session.prepare_pending(&request).unwrap();
    let cpu = cpu_session.finish_cpu(&mut cpu_pending).unwrap();
    let cpu_json = cpu
        .finish(&cpu.direct_cpu(), r#"{"backend":"cpu"}"#)
        .unwrap();
    for key in ["rho", "pmRho", "spectra", "pdf", "diagnostics"] {
        assert_eq!(
            external_json[key], cpu_json[key],
            "external exact deposit differs at {key}"
        );
    }
    assert!(cpu_session
        .try_cached_observed_source(&request, "f64", "cpu-f64", "gpu-f32-geometry-overlap",)
        .unwrap()
        .is_none());

    assert!(exact_sheet_density(2, 16, 16, &lpt[..lpt.len() - 1]).is_err());
    let mut nonfinite = lpt;
    nonfinite[0] = f32::INFINITY;
    assert!(exact_sheet_density(2, 16, 16, &nonfinite).is_err());
}

#[test]
fn cached_pm_positions_rebuild_new_mesh_and_never_reuse_stale_external_deposits() {
    let request = LabRequest {
        dim: 2,
        n: 16,
        mesh: 16,
        force_mesh: 16,
        steps: 32,
        ..Default::default()
    };
    let mut session = LabSession::new();
    let mut pending = session.prepare_pending(&request).unwrap();
    let cpu = session.finish_cpu(&mut pending).unwrap();
    let _ = cpu
        .finish(&cpu.direct_cpu(), r#"{"backend":"cpu"}"#)
        .unwrap();

    let mut pending = session.prepare_pending(&request).unwrap();
    let cached = session
        .finish_cached(&mut pending, "f64")
        .unwrap()
        .expect("matching PM should finalize from the retained result");
    let cached = cached
        .finish(&cached.direct_cpu(), r#"{"backend":"cpu"}"#)
        .unwrap();
    assert_eq!(cached["cache"]["pm"]["status"], "hit");
    assert_eq!(cached["cache"]["observations"]["status"], "hit");

    let changed_mesh = LabRequest {
        mesh: 24,
        ..request.clone()
    };
    let mut pending = session.prepare_pending(&changed_mesh).unwrap();
    let pm_positions = session
        .cached_pm_positions(&pending, "f64")
        .expect("matching PM state should be available without evolving it again");
    let rho = exact_sheet_density(2, 16, 24, &pending.lpt_positions()).unwrap();
    let pmrho = exact_sheet_density(2, 16, 24, &pm_positions).unwrap();
    let prepared = session
        .finish_cached_observed(
            &mut pending,
            "f64",
            &rho,
            &pmrho,
            "gpu-f32-geometry-overlap",
        )
        .unwrap();
    let result = prepared
        .finish(&prepared.direct_cpu(), r#"{"backend":"cpu"}"#)
        .unwrap();
    assert_eq!(result["cache"]["pm"]["status"], "hit");
    assert_eq!(result["cache"]["observations"]["status"], "miss");
    let observation_grid = Grid::new(2, 24, 1.0);
    assert_eq!(
        result["rho"],
        serde_json::to_value(observables::smooth_density(
            &observation_grid,
            &rho,
            changed_mesh.smoothing,
        ))
        .unwrap()
    );
    assert_eq!(
        result["pmRho"],
        serde_json::to_value(observables::smooth_density(
            &observation_grid,
            &pmrho,
            changed_mesh.smoothing,
        ))
        .unwrap()
    );

    // Same observation key, deliberately different but conserved deposits.
    // The supplied values must replace the previous cached fields.
    let uniform = vec![1.0f32; 24 * 24];
    let mut pending = session.prepare_pending(&changed_mesh).unwrap();
    let prepared = session
        .finish_cached_observed(
            &mut pending,
            "f64",
            &uniform,
            &uniform,
            "gpu-f32-geometry-overlap",
        )
        .unwrap();
    let result = prepared
        .finish(&prepared.direct_cpu(), r#"{"backend":"cpu"}"#)
        .unwrap();
    assert_eq!(result["rho"], serde_json::to_value(&uniform).unwrap());
    assert_eq!(result["pmRho"], serde_json::to_value(&uniform).unwrap());

    let mut invalid = uniform.clone();
    invalid[0] += 0.01;
    assert!(session
        .finish_cached_observed(
            &mut pending,
            "f64",
            &invalid,
            &uniform,
            "gpu-f32-geometry-overlap",
        )
        .is_err());
    let still_cached = session
        .try_cached_observed_source(&changed_mesh, "f64", "cpu-f64", "gpu-f32-geometry-overlap")
        .unwrap()
        .expect("invalid payload must leave the previous valid observations intact");
    let still_cached = still_cached
        .finish(&still_cached.direct_cpu(), r#"{"backend":"cpu"}"#)
        .unwrap();
    assert_eq!(still_cached["rho"], serde_json::to_value(&uniform).unwrap());
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
    let manifest =
        serde_json::json!({"schemaVersion":2,"state":{"n":64},"request":{"n":64},"result":d});
    let replay: serde_json::Value = serde_json::from_str(&run_json(&manifest.to_string())).unwrap();
    assert_eq!(replay["request"]["n"], 16);
    assert_eq!(replay["rho"], manifest["result"]["rho"]);
    assert_eq!(replay["a"], 0.25);
    let invalid: serde_json::Value =
        serde_json::from_str(&run_json(r#"{"schemaVersion":99}"#)).unwrap();
    assert!(invalid["error"].is_string());
}

#[test]
fn v3_manifest_replays_physics_request_and_ignores_ui_execution_policy() {
    let request = LabRequest {
        dim: 1,
        n: 16,
        force_mesh: 16,
        steps: 32,
        epoch: "scaleFactor".into(),
        scale_factor: 0.25,
        ..Default::default()
    };
    let expected = run(&request).unwrap();
    let manifest = serde_json::json!({
        "manifestFormat":"precrossedsheet-experiment",
        "schemaVersion":3,
        "state":{"dim":3,"n":64,"mesh":128,"forceMesh":64,"epoch":"scaleFactor","scaleFactor":1.2,"backend":"gpu","view":"gradient"},
        "request":request,
        "execution":{"requestedBackend":"gpu","executedBackend":"gpu","policy":"ignored by native physical replay"}
    });
    let replay: serde_json::Value = serde_json::from_str(&run_json(&manifest.to_string())).unwrap();
    assert!(replay.get("error").is_none(), "{}", replay["error"]);
    assert_eq!(
        replay["schemaVersion"], 2,
        "physical response schema remains v2"
    );
    assert_eq!(replay["engineVersion"], env!("CARGO_PKG_VERSION"));
    assert_eq!(
        replay["request"]["n"], request.n,
        "manifest request wins over stale UI state"
    );
    assert_eq!(replay["a"], expected["a"]);
    assert_eq!(replay["x"], expected["x"]);
    assert_eq!(replay["spectra"]["direct"], expected["spectra"]["direct"]);
    assert_eq!(
        replay["execution"]["backend"], "cpu",
        "execution policy is not a physical input"
    );
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

#[test]
fn prepared_run_finishes_identically_and_validates_direct_channels() {
    let request = LabRequest {
        dim: 2,
        n: 16,
        force_mesh: 16,
        steps: 32,
        epoch: "scaleFactor".into(),
        scale_factor: 0.3,
        ..Default::default()
    };
    let prepared = hcc_core::laboratory::prepare(&request).unwrap();
    assert_eq!(prepared.metadata()["modeCount"], 4);
    let channels = prepared.direct_cpu();
    assert_eq!(channels.len(), 32);
    let fd_first = prepared.finite_difference_cpu();
    let fd_ptr = fd_first.as_ptr();
    let finished = prepared.finish(&channels, r#"{"backend":"cpu"}"#).unwrap();
    assert_eq!(prepared.finite_difference_cpu().as_ptr(), fd_ptr);
    let grid = Grid::new(request.dim, request.n, 1.0);
    let plus_hi = prepared.finite_difference_plus_hi();
    let plus_lo = prepared.finite_difference_plus_lo();
    let minus_hi = prepared.finite_difference_minus_hi();
    let minus_lo = prepared.finite_difference_minus_lo();
    for packed in [plus_hi, plus_lo, minus_hi, minus_lo] {
        assert_eq!(packed.len(), grid.size * 4);
        assert!(packed.iter().all(|v| v.is_finite()));
        for i in 0..grid.size {
            for axis in grid.dim..4 {
                assert_eq!(packed[i * 4 + axis], 0.0);
            }
        }
    }
    assert!(
        plus_lo.iter().chain(minus_lo).any(|value| *value != 0.0),
        "the packed low parts must carry endpoint precision"
    );
    let reconstruct = |hi: &[f32], lo: Option<&[f32]>| {
        let mut x = vec![0.0; grid.size * grid.dim];
        for i in 0..grid.size {
            let q = grid.unravel(i);
            for axis in 0..grid.dim {
                x[i * grid.dim + axis] = q[axis] as f64 * grid.dx()
                    + hi[i * 4 + axis] as f64
                    + lo.map_or(0.0, |values| values[i * 4 + axis] as f64);
            }
        }
        x
    };
    let plus = reconstruct(plus_hi, Some(plus_lo));
    let minus = reconstruct(minus_hi, Some(minus_lo));
    let plus_single = reconstruct(plus_hi, None);
    let minus_single = reconstruct(minus_hi, None);
    let mut worst_expanded_error = 0.0f64;
    let mut worst_single_error = 0.0f64;
    for (i, mode) in prepared.modes().iter().enumerate() {
        let k = [grid.kf() * mode[0] as f64, 0.0, 0.0];
        let fd_expanded = (observables::sheet_mode(&grid, &plus, k).norm_sqr()
            - observables::sheet_mode(&grid, &minus, k).norm_sqr())
            / (2.0 * 1e-4);
        let fd_single = (observables::sheet_mode(&grid, &plus_single, k).norm_sqr()
            - observables::sheet_mode(&grid, &minus_single, k).norm_sqr())
            / (2.0 * 1e-4);
        let scale = fd_first[i].abs().max(1e-9);
        let expanded_error = (fd_expanded - fd_first[i]).abs() / scale;
        let single_error = (fd_single - fd_first[i]).abs() / scale;
        worst_expanded_error = worst_expanded_error.max(expanded_error);
        worst_single_error = worst_single_error.max(single_error);
    }
    assert!(
        worst_expanded_error <= 2e-7,
        "expanded f32 endpoints differ from retained f64 FD: {worst_expanded_error}"
    );
    assert!(
        worst_single_error > worst_expanded_error * 4.0,
        "hi/lo endpoints should improve over single-f32 positions: expanded={worst_expanded_error}, single={worst_single_error}"
    );
    let supplied_fd = fd_first.iter().map(|v| v + 0.125).collect::<Vec<_>>();
    let supplied_finish = prepared
        .finish_with_finite_difference(&channels, &supplied_fd, r#"{"backend":"external-fd"}"#)
        .unwrap();
    let supplied_scaled = supplied_finish["gradient"]["finiteDifference"]
        .as_array()
        .unwrap();
    for (actual, raw) in supplied_scaled.iter().zip(&supplied_fd) {
        assert_eq!(
            actual.as_f64().unwrap(),
            raw * request.box_size.powi(request.dim as i32)
        );
    }
    assert_eq!(supplied_finish["execution"]["backend"], "external-fd");
    let compatibility = run(&request).unwrap();
    for path in [
        "spectra.direct",
        "spectra.directPM",
        "gradient.power",
        "gradient.tangent",
        "gradient.finiteDifference",
        "gradient.relativeError",
    ] {
        let pointer = format!("/{}", path.replace('.', "/"));
        assert_eq!(
            finished.pointer(&pointer),
            compatibility.pointer(&pointer),
            "{path}"
        );
    }
    assert!(prepared
        .finish(&channels[..channels.len() - 1], "{}")
        .is_err());
    assert!(prepared
        .finish_with_finite_difference(&channels, &fd_first[..fd_first.len() - 1], "{}")
        .is_err());
    let mut bad_fd = fd_first.to_vec();
    bad_fd[0] = f64::INFINITY;
    assert!(prepared
        .finish_with_finite_difference(&channels, &bad_fd, "{}")
        .is_err());
    let mut bad = channels.clone();
    bad[0] = f64::NAN;
    assert!(prepared.finish(&bad, "{}").is_err());
    assert!(prepared.finish(&channels, "{").is_err());
}

#[test]
fn pending_run_reuses_common_state_after_gpu_validation_failure() {
    let request = LabRequest {
        dim: 2,
        n: 16,
        force_mesh: 16,
        steps: 32,
        epoch: "scaleFactor".into(),
        scale_factor: 0.3,
        ..Default::default()
    };
    let pending = prepare_pending(&request).unwrap();
    assert_eq!(pending.pm_clocks().len(), 5 * request.steps);
    let reference = pending.pm_reference().unwrap();
    let quantized_reference = pending.pm_reference_quantized().unwrap();
    assert_eq!(reference.len(), 16 * request.n.pow(request.dim as u32));
    assert_eq!(quantized_reference.len(), reference.len());
    assert!(reference
        .iter()
        .chain(&quantized_reference)
        .all(|v| v.is_finite()));
    let (x, p, dx, dp) = pending.pm_input();
    assert_eq!(x.len(), request.n.pow(request.dim as u32) * 4);
    assert_eq!(p.len(), x.len());
    assert_eq!(dx.len(), x.len());
    assert_eq!(dp.len(), x.len());
    let bad = vec![0.0; request.steps - 1];
    assert!(pending.finish_gpu_pm(&x, &p, &dx, &dp, &bad).is_err());
    let mut nonfinite = x.clone();
    nonfinite[0] = f32::NAN;
    assert!(pending
        .finish_gpu_pm(&nonfinite, &p, &dx, &dp, &vec![1.0; request.steps])
        .is_err());
    // The same retained pending state can still complete the exact CPU route.
    let prepared = pending.finish_cpu().unwrap();
    let channels = prepared.direct_cpu();
    let finished = prepared.finish(&channels, r#"{"backend":"cpu"}"#).unwrap();
    assert_eq!(finished, run(&request).unwrap());
}

#[test]
fn pm_displacement_packing_preserves_weak_modes_and_has_quantized_cpu_oracle() {
    let request = LabRequest {
        dim: 3,
        n: 16,
        force_mesh: 32,
        steps: 32,
        omega_m: 1.0,
        epoch: "scaleFactor".into(),
        scale_factor: 0.25,
        ..Default::default()
    };
    let pending = prepare_pending(&request).unwrap();
    let displacement = pending.pm_displacement();
    let absolute = pending.pm_initial_component(0);
    let grid = Grid::new(request.dim, request.n, 1.0);

    // Subtracting q after rounding absolute x loses low displacement bits.
    // Packing ψ=x−q in f64 first should retain a much smaller relative error.
    let mut worst_relative_separation = 0.0f64;
    for i in 0..grid.size {
        let q = grid.unravel(i);
        for axis in 0..grid.dim {
            let j = i * 4 + axis;
            let q = q[axis] as f64 * grid.dx();
            let psi = displacement[j] as f64;
            if q > 0.5 && psi.abs() > 1e-7 {
                let abs_then_subtract = absolute[j] as f64 - q;
                let separation = (abs_then_subtract - psi).abs();
                let displacement_ulp_scale = f32::EPSILON as f64 * psi.abs();
                worst_relative_separation =
                    worst_relative_separation.max(separation / displacement_ulp_scale);
            }
        }
    }
    assert!(
        worst_relative_separation > 8.0,
        "displacement packing should preserve weak-mode bits lost by absolute f32 positions; ratio={worst_relative_separation}"
    );

    let metadata = pending.metadata();
    assert!(metadata["pmInitialMinimumJacobianDisplacementF32"]
        .as_f64()
        .unwrap()
        .is_finite());
    let oracle = pending.pm_reference_displacement_quantized().unwrap();
    assert_eq!(oracle.len(), 16 * request.n.pow(request.dim as u32));
    assert!(oracle.iter().all(|v| v.is_finite()));
}

#[test]
fn pending_gpu_result_records_first_verified_drift_bracket() {
    let request = LabRequest {
        dim: 2,
        n: 16,
        force_mesh: 16,
        steps: 32,
        epoch: "scaleFactor".into(),
        scale_factor: 0.3,
        ..Default::default()
    };
    let pending = prepare_pending(&request).unwrap();
    let (x, p, dx, dp) = pending.pm_input();
    let mut minima = vec![0.5; request.steps];
    minima[4] = -0.125;
    minima[9] = -0.25;
    let prepared = pending.finish_gpu_pm(&x, &p, &dx, &dp, &minima).unwrap();
    let channels = prepared.direct_cpu();
    let finished = prepared
        .finish(&channels, r#"{"backend":"gpu","pmArithmetic":"f32"}"#)
        .unwrap();
    assert_eq!(finished["guard"]["pmCrossed"], true);
    assert_eq!(finished["guard"]["firstPmCrossing"]["step"], 5);
    let clocks = pending.pm_clocks();
    assert_eq!(
        finished["guard"]["firstPmCrossing"]["aStart"],
        clocks[5 * 4 + 3]
    );
    assert_eq!(
        finished["guard"]["firstPmCrossing"]["aEnd"],
        clocks[5 * 4 + 4]
    );
    assert_eq!(finished["conventions"]["pmArithmetic"], "f32");
    assert_eq!(finished["guard"]["pmMinimumSimplexJacobian"], -0.25);
    let metadata = pending.metadata();
    let initial_used = metadata["pmInitialMinimumJacobianF64"]
        .as_f64()
        .unwrap()
        .min(
            metadata["pmInitialMinimumJacobianDisplacementF32"]
                .as_f64()
                .unwrap(),
        );
    assert_eq!(
        finished["guard"]["pmInitialMinimumJacobianDisplacementF32"],
        metadata["pmInitialMinimumJacobianDisplacementF32"]
    );
    assert_eq!(
        finished["guard"]["pmInitialMinimumJacobianUsed"],
        initial_used
    );
    assert_eq!(
        finished["guard"]["pmInitialMinimumJacobianRepresentation"],
        "min(f64, displacement-f32-materialized)"
    );
}

#[test]
fn pm_drift_minimum_detects_reversed_orientation() {
    let grid = Grid::new(2, 2, 1.0);
    let start: Vec<f64> = (0..4)
        .flat_map(|i| {
            let q = grid.unravel(i);
            [q[0] as f64 / 2.0, q[1] as f64 / 2.0]
        })
        .collect();
    let end = start.iter().map(|v| -*v).collect::<Vec<_>>();
    let minimum = hcc_core::nbody::minimum_drift_jacobian(&grid, &start, &end);
    assert!(minimum < 0.0, "the affine drift reverses orientation");
}

#[test]
fn packed_gpu_inputs_are_quantized_displacements_and_match_cpu_modes() {
    let request = LabRequest {
        dim: 2,
        n: 16,
        force_mesh: 16,
        steps: 32,
        epoch: "scaleFactor".into(),
        scale_factor: 0.3,
        ..Default::default()
    };
    let prepared = hcc_core::laboratory::prepare(&request).unwrap();
    let grid = Grid::new(request.dim, request.n, 1.0);
    let channels = prepared.direct_cpu();
    let quantized_channels = prepared.direct_quantized_cpu();
    assert_eq!(quantized_channels.len(), channels.len());
    for (i, mode) in prepared.modes().iter().enumerate() {
        let mut x = vec![0.0; grid.size * grid.dim];
        let mut dx = x.clone();
        let mut pmx = x.clone();
        let mut pmdx = x.clone();
        for vertex in 0..grid.size {
            let q = grid.unravel(vertex);
            for axis in 0..grid.dim {
                let packed = 4 * vertex + axis;
                let flat = grid.dim * vertex + axis;
                // The packed data is ψ=x−q and Δx, both rounded once to f32.
                x[flat] = q[axis] as f64 * grid.dx() + prepared.lpt_displacement()[packed] as f64;
                dx[flat] = prepared.lpt_direction()[packed] as f64;
                pmx[flat] = q[axis] as f64 * grid.dx() + prepared.pm_displacement()[packed] as f64;
                pmdx[flat] = prepared.pm_direction()[packed] as f64;
            }
        }
        let k = [grid.kf() * mode[0] as f64, 0.0, 0.0];
        let (f, df) = observables::sheet_mode_jvp(&grid, &x, &dx, k);
        let (pmf, pmdf) = observables::sheet_mode_jvp(&grid, &pmx, &pmdx, k);
        let got = &channels[8 * i..8 * i + 8];
        let packed_channels = [f.re, f.im, df.re, df.im, pmf.re, pmf.im, pmdf.re, pmdf.im];
        for j in 0..8 {
            assert!(got[j].is_finite() && packed_channels[j].is_finite());
            assert!(
                (quantized_channels[8 * i + j] - packed_channels[j]).abs() < 1e-13,
                "quantized mode {}, channel {j}: API {} vs manual {}",
                mode[0],
                quantized_channels[8 * i + j],
                packed_channels[j]
            );
            assert!(
                (got[j] - packed_channels[j]).abs() < 2e-4,
                "mode {}, channel {j}: f64 {} vs packed-f32 {}",
                mode[0],
                got[j],
                packed_channels[j]
            );
        }
        let (value, _) = observables::sheet_mode_jvp(&grid, &x, &dx, k);
        assert!((observables::sheet_mode(&grid, &x, k) - value).norm() < 1e-13);
    }
}

#[test]
fn selected_lpt_mode_recovery_matches_full_cpu_channels_and_checks_indices() {
    let request = LabRequest {
        dim: 2,
        n: 16,
        force_mesh: 16,
        steps: 32,
        epoch: "scaleFactor".into(),
        scale_factor: 0.3,
        ..Default::default()
    };
    let prepared = hcc_core::laboratory::prepare(&request).unwrap();
    let full = prepared.direct_cpu();
    let selected = prepared.direct_lpt_cpu_modes(&[3, 0, 3]).unwrap();
    assert_eq!(selected.len(), 12);
    for (out_mode, full_mode) in [(0usize, 3usize), (1, 0), (2, 3)] {
        assert_eq!(
            &selected[4 * out_mode..4 * out_mode + 4],
            &full[8 * full_mode..8 * full_mode + 4],
        );
    }
    assert!(prepared
        .direct_lpt_cpu_modes(&[prepared.metadata()["modeCount"].as_u64().unwrap() as u32])
        .is_err());
}

#[test]
fn particle_deposits_conserve_mass_and_leave_a_uniform_lattice_uniform() {
    use hcc_core::sheet::{cic_density, tsc_density};
    for dim in 1..=3 {
        let n = if dim == 3 { 8 } else { 16 };
        let g = Grid::new(dim, n, 1.0);
        let dx = g.dx();
        let pos: Vec<f32> = (0..g.size)
            .flat_map(|i| {
                let ijk = g.unravel(i);
                (0..dim).map(move |d| (ijk[d] as f64 * dx) as f32)
            })
            .collect();
        for mesh in [n, 2 * n] {
            for rho in [cic_density(dim, &pos, g.size, 1.0, mesh), tsc_density(dim, &pos, g.size, 1.0, mesh)] {
                let mean = rho.iter().map(|&v| v as f64).sum::<f64>() / rho.len() as f64;
                assert!((mean - 1.).abs() < 1e-6, "dim {dim} mesh {mesh}");
                if mesh == n {
                    assert!(rho.iter().all(|&v| (v - 1.).abs() < 1e-5), "dim {dim}: a lattice coincident with the mesh must deposit uniformly");
                }
            }
        }
    }
    // One particle: TSC spreads over three cells per axis with symmetric quadratic weights.
    let rho = tsc_density(1, &[0.5 + 0.0625 / 8.], 1, 1.0, 8);
    let weights: Vec<f64> = rho.iter().map(|&v| v as f64 / 8.).collect();
    assert!((weights.iter().sum::<f64>() - 1.).abs() < 1e-6);
    // x = 0.5078 is nearest the center of cell 4 (0.5625), then cell 3 (0.4375), then cell 5.
    assert!(weights[4] > weights[3] && weights[3] > weights[5] && weights[5] > 0.);
    assert!(weights.iter().enumerate().all(|(i, &w)| (3..=5).contains(&i) || w == 0.));
}

#[test]
fn gaussian_runs_report_particle_deposits_and_the_filtered_linear_input() {
    use hcc_core::cosmological::LinearSpectrum;
    use std::f64::consts::PI;
    let request = LabRequest {
        dim: 2,
        n: 32,
        mesh: 32,
        force_mesh: 32,
        steps: 32,
        ic: "gaussian".into(),
        initial_cutoff: 3,
        epoch: "scaleFactor".into(),
        scale_factor: 0.4,
        transfer: "bao".into(),
        ..Default::default()
    };
    let d = run(&request).unwrap();
    assert_eq!(d["conventions"]["transfer"], "bao");
    for key in ["cicMass", "tscMass", "mass"] {
        assert!((d["diagnostics"][key].as_f64().unwrap() - 1.).abs() < 1e-6, "{key}");
    }
    let count = d["spectra"]["k"].as_array().unwrap().len();
    assert_eq!(d["spectra"]["tsc"].as_array().unwrap().len(), count);
    assert_eq!(d["cicRho"].as_array().unwrap().len(), 32 * 32);
    assert_eq!(d["tscRho"].as_array().unwrap().len(), 32 * 32);
    let linear_k: Vec<f64> = d["spectra"]["linearK"].as_array().unwrap().iter().map(|v| v.as_f64().unwrap()).collect();
    let linear: Vec<f64> = d["spectra"]["linear"].as_array().unwrap().iter().map(|v| v.as_f64().unwrap()).collect();
    assert!(!linear_k.is_empty() && linear_k.len() == linear.len());
    let spectrum = LinearSpectrum::new(0.315, 0.049, 0.674, 0.965, 0.811, true);
    let growth_ratio = d["D"].as_f64().unwrap() / d["growthAtOne"].as_f64().unwrap();
    for (k, p) in linear_k.iter().zip(&linear) {
        let expected = spectrum.filtered_power(*k, 2, 300., 14., growth_ratio, 3);
        assert!((p / expected - 1.).abs() < 1e-9);
    }
    let axis: Vec<f64> = d["spectra"]["directLinearK"].as_array().unwrap().iter().map(|v| v.as_f64().unwrap()).collect();
    let direct_k_late: Vec<f64> = d["spectra"]["directK"].as_array().unwrap().iter().map(|v| v.as_f64().unwrap()).collect();
    let shell_k_late: Vec<f64> = d["spectra"]["k"].as_array().unwrap().iter().map(|v| v.as_f64().unwrap()).collect();
    assert!(
        direct_k_late.len() >= count,
        "directK should have at least as many k points as the gridded shell average ({} vs {})",
        direct_k_late.len(),
        count
    );
    assert!(
        *direct_k_late.last().unwrap() >= *shell_k_late.last().unwrap(),
        "directK should extend at least as far as the gridded shell average ({} vs {})",
        direct_k_late.last().unwrap(),
        shell_k_late.last().unwrap()
    );
    assert!(axis.len() >= 6, "first-axis linear modes are populated across the direct spectrum range");
    // The realized linear power of this seed: in the linear regime (1LPT, tiny growth) it equals
    // the measured sheet power mode by mode and bin by bin, up to the cell window and tiny
    // second-order terms; at later times it is the reference the nonlinear curves are read against.
    // n=64 keeps the piecewise-linear element error of the m=3 mode below one percent.
    let early = run(&LabRequest { order: 1, scale_factor: 0.02, n: 64, mesh: 64, force_mesh: 64, ..request.clone() }).unwrap();
    let f = |v: &serde_json::Value| -> Vec<f64> { v.as_array().unwrap().iter().map(|x| x.as_f64().unwrap()).collect() };
    let (direct, direct_realized, direct_k) = (f(&early["spectra"]["direct"]), f(&early["spectra"]["directRealized"]), f(&early["spectra"]["directRealizedK"]));
    assert!(direct.len() >= f(&early["spectra"]["k"]).len());
    assert!(direct_realized.len() >= 6);
    for i in 0..3 {
        assert!((direct_k[i] - 2. * PI * (i + 1) as f64 / 300.).abs() < 1e-12);
        assert!((direct[i] / direct_realized[i] - 1.).abs() < 0.02, "mode {}: {} vs {}", i + 1, direct[i], direct_realized[i]);
    }
    let (sheet, k, realized, realized_k) = (f(&early["spectra"]["sheet"]), f(&early["spectra"]["k"]), f(&early["spectra"]["realized"]), f(&early["spectra"]["realizedK"]));
    assert!(!realized.is_empty());
    for (kr, pr) in realized_k.iter().zip(&realized).filter(|(kr, _)| **kr <= 2. * PI * 3. / 300. + 1e-9) {
        let i = k.iter().position(|kk| (kk - kr).abs() < 1e-12).expect("realized bins share the measured bin centers");
        assert!((sheet[i] / pr - 1.).abs() < 0.04, "bin k={kr}: sheet {} vs realized {pr}", sheet[i]);
    }
    let realized_late: Vec<f64> = f(&d["spectra"]["realized"]);
    assert!(
        realized_late.iter().zip(&linear).zip(&linear_k)
            .filter(|(_, k)| **k <= 2. * PI * 3. / 300. + 1e-9)
            .all(|((r, l), _)| *r > 0.05 * l && *r < 20. * l),
        "the realization scatters around its expectation"
    );
    // The deterministic benchmark has no linear input spectrum, and unknown fits are rejected.
    let waves = run(&LabRequest { dim: 1, n: 32, steps: 32, ..Default::default() }).unwrap();
    assert!(waves["spectra"]["linear"].as_array().unwrap().is_empty());
    assert_eq!(waves["conventions"]["transfer"], "smooth");
    assert!(run(&LabRequest { transfer: "wiggly".into(), ..Default::default() }).is_err());
    let parsed: LabRequest = serde_json::from_str(r#"{"dim":2,"n":16}"#).unwrap();
    assert_eq!(parsed.transfer, "smooth", "older manifests without the field keep the smooth fit");
}

#[test]
fn unsmoothed_sheet_pdfs_evaluate_directly_on_simplices_beyond_grid_limits() {
    let req = LabRequest {
        dim: 2,
        n: 64,
        mesh: 16,
        force_mesh: 64,
        order: 4,
        epoch: "scaleFactor".into(),
        scale_factor: 1.35,
        ic: "gaussian".into(),
        transfer: "bao".into(),
        initial_cutoff: 16,
        box_size: 1000.0,
        initial_smoothing: 6.0,
        smoothing: 0.0,
        steps: 32,
        seed: 7,
        estimators: vec![
            "4lpt:p0".into(),
            "4lpt:p1".into(),
            "4lpt:cic".into(),
            "pm:p0".into(),
        ],
        ..Default::default()
    };
    let d = run(&req).unwrap();
    let p0 = &d["customSeries"]["4lpt:p0"]["pdf"];
    let p1 = &d["customSeries"]["4lpt:p1"]["pdf"];
    let cic = &d["customSeries"]["4lpt:cic"]["pdf"];

    let max_populated_rho = |pdf: &serde_json::Value| -> f64 {
        let lr = pdf["logrho"].as_array().unwrap();
        let m = pdf["mass"].as_array().unwrap();
        lr.iter()
            .zip(m.iter())
            .filter(|(_, mv)| mv.as_f64().unwrap() > 0.0)
            .map(|(y, _)| 10.0_f64.powf(y.as_f64().unwrap()))
            .fold(0.0_f64, f64::max)
    };

    let p0_max = max_populated_rho(p0);
    let p1_max = max_populated_rho(p1);
    let cic_max = max_populated_rho(cic);

    assert!(
        p0_max > cic_max * 5.0,
        "direct simplex P0 PDF should extend far beyond the coarse Eulerian grid CIC limit ({p0_max} vs {cic_max})"
    );
    assert!(
        p1_max > cic_max * 5.0,
        "direct simplex P1 PDF should extend far beyond the coarse Eulerian grid CIC limit ({p1_max} vs {cic_max})"
    );
    assert!((p0["capturedMass"].as_f64().unwrap() - 1.0).abs() < 1e-4);
    assert!((p0["capturedVolume"].as_f64().unwrap() - 1.0).abs() < 1e-4);
    assert!((p1["capturedMass"].as_f64().unwrap() - 1.0).abs() < 1e-4);
    assert!((p1["capturedVolume"].as_f64().unwrap() - 1.0).abs() < 1e-4);

    let req_smoothed = LabRequest {
        smoothing: 0.03,
        ..req
    };
    let d_s = run(&req_smoothed).unwrap();
    let p0_s_max = max_populated_rho(&d_s["customSeries"]["4lpt:p0"]["pdf"]);
    let p1_s_max = max_populated_rho(&d_s["customSeries"]["4lpt:p1"]["pdf"]);
    assert!(
        (p0_s_max - p0_max).abs() < 1e-9 && (p1_s_max - p1_max).abs() < 1e-9,
        "sheet PDFs must always evaluate directly on simplices and never revert to the narrow grid range"
    );
}

#[test]
fn redshift_space_sheet_and_gaussian_packets_capture_kaiser_and_crossing() {
    // 1. Linear regime: verify the redshifted sheet monopole matches the exact discrete Kaiser expectation.
    let linear_req = LabRequest {
        dim: 2,
        n: 64,
        mesh: 64,
        force_mesh: 64,
        order: 1,
        epoch: "scaleFactor".into(),
        scale_factor: 0.05,
        ic: "gaussian".into(),
        transfer: "smooth".into(),
        initial_cutoff: 6,
        box_size: 300.0,
        initial_smoothing: 14.0,
        smoothing: 0.0,
        steps: 32,
        seed: 7,
        ..Default::default()
    };
    let d_lin = run(&linear_req).unwrap();
    let rsd_lin = &d_lin["rsd"];
    let p0_sheet = rsd_lin["spectra"]["rsdSheetMonopole"].as_array().unwrap();
    let p0_kaiser = rsd_lin["spectra"]["kaiserMonopole"].as_array().unwrap();
    let p0_real = rsd_lin["spectra"]["realMonopole"].as_array().unwrap();
    let ratio_sheet = p0_sheet[0].as_f64().unwrap() / p0_real[0].as_f64().unwrap();
    let ratio_kaiser = p0_kaiser[0].as_f64().unwrap() / p0_real[0].as_f64().unwrap();
    assert!(
        (ratio_sheet / ratio_kaiser - 1.0).abs() < 0.03,
        "low-k redshift-space sheet monopole should match the exact discrete Kaiser expectation ({ratio_sheet} vs {ratio_kaiser})"
    );

    // 2. Pre-crossing real-space epoch where velocity gradients drive redshift-space shell crossing.
    let pre_req = LabRequest {
        dim: 2,
        n: 64,
        mesh: 64,
        force_mesh: 64,
        order: 4,
        omega_m: 1.0,
        omega_b: 0.0,
        epoch: "fraction".into(),
        fraction: 0.72,
        ic: "waves".into(),
        smoothing: 0.0,
        steps: 32,
        ..Default::default()
    };
    let d_pre = run(&pre_req).unwrap();
    let sum_pre = &d_pre["rsd"]["summary"];
    let min_j_real = sum_pre["minRealSimplexJacobian"].as_f64().unwrap();
    let min_j_rsd = sum_pre["minRsdSimplexJacobian"].as_f64().unwrap();
    assert!(
        min_j_real > 0.0 && min_j_rsd < min_j_real,
        "redshift-space minimum simplex Jacobian ({min_j_rsd}) must be smaller than real-space ({min_j_real}) and cross zero earlier"
    );
    let gauss_rho: Vec<f64> = d_pre["rsd"]["maps"]["rsdGaussRho"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_f64().unwrap())
        .collect();
    let gauss_mean = gauss_rho.iter().sum::<f64>() / gauss_rho.len() as f64;
    assert!(
        (gauss_mean - 1.0).abs() < 1e-4,
        "simplex velocity-gradient Gaussian packet deposit must conserve mass (mean={gauss_mean})"
    );
}
