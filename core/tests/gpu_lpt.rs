use hcc_core::{
    fft::FftEngine,
    gpu_lpt::GpuLptSeed,
    laboratory::{initial_conditions, LabRequest, LabSession},
    lpt::Lpt,
    tangent,
};

#[test]
fn raw_lpt_boundary_preserves_growth_merge_and_actual_amplitude_direction() {
    for dim in [2, 3] {
        for omega_m in [1.0, 0.315] {
            let r = LabRequest {
                dim,
                n: 16,
                mesh: 16,
                force_mesh: 16,
                omega_m,
                epoch: "scaleFactor".into(),
                scale_factor: 0.25,
                ..Default::default()
            };
            let c = initial_conditions(&r);
            let mut eng = FftEngine::new(&c.grid);
            let raw = Lpt::new_unmerged(&c.grid, &mut eng, &c.delta0_hat, 4, c.cosmo, true);
            let rebuilt =
                Lpt::from_packed_fields(&c.grid, 4, c.cosmo, &raw.packed_fields()).unwrap();
            assert_eq!(rebuilt.terms.len(), c.lpt_ref().terms.len());
            for order in 1..=4 {
                for d in [0.1, 0.25, 0.6] {
                    let original = tangent::lpt_amplitude_jvp(c.lpt_ref(), d, order);
                    let adopted = tangent::lpt_amplitude_jvp(&rebuilt, d, order);
                    assert_eq!(
                        original, adopted,
                        "dim={dim} Om={omega_m} order={order} D={d}"
                    );
                    let mut a = vec![0.0; c.grid.size * dim];
                    let mut b = a.clone();
                    c.lpt_ref().positions(d, order, &mut a);
                    rebuilt.positions(d, order, &mut b);
                    assert_eq!(a, b);
                    assert_eq!(
                        c.lpt_ref().min_jacobian(d, order),
                        rebuilt.min_jacobian(d, order)
                    );
                }
            }
            let packed = raw.packed_fields();
            assert!(
                Lpt::from_packed_fields(&c.grid, 4, c.cosmo, &packed[..packed.len() - 1]).is_err()
            );
            let mut bad = packed.clone();
            bad[0] = f32::NAN;
            assert!(Lpt::from_packed_fields(&c.grid, 4, c.cosmo, &bad).is_err());
            let seed = GpuLptSeed::new(&r).unwrap();
            let seed_meta = seed.metadata();
            let field_stride = dim + dim * dim + 3;
            let particles = 16usize.pow(dim as u32);
            assert_eq!(seed_meta["fieldStride"], field_stride);
            assert_eq!(seed_meta["termScalars"], particles * field_stride);
            assert_eq!(
                seed_meta["totalScalars"],
                seed_meta["termCount"].as_u64().unwrap() as usize * particles * field_stride
            );
            let mut session = LabSession::new();
            assert!(seed
                .finish_gpu_with_session(&mut session, &packed[..packed.len() - 1])
                .is_err());
            assert_eq!(
                session.cache_status()["bytes"],
                0,
                "malformed output must not adopt the seed"
            );
            assert!(
                seed.delta_hat().is_ok(),
                "a rejected payload must not consume the seed"
            );
            let mut nonfinite = packed.clone();
            nonfinite[0] = f32::INFINITY;
            assert!(seed.finish_gpu(&nonfinite).is_err());
            assert!(
                seed.delta_hat().is_ok(),
                "nonfinite output must not consume the seed"
            );
            let pending = seed.finish_gpu_with_session(&mut session, &packed).unwrap();
            assert_eq!(pending.metadata()["dim"], dim);
            assert_eq!(session.cache_status()["initial"]["provenance"], "gpu-f32");
            assert!(
                seed.delta_hat().is_err(),
                "consumed seeds cannot alias a second construction"
            );
            let direct = GpuLptSeed::new(&r).unwrap().finish_gpu(&packed).unwrap();
            assert_eq!(direct.metadata()["initialSource"], "gpu-f32");
        }
    }
    assert!(GpuLptSeed::new(&LabRequest {
        dim: 1,
        n: 16,
        ..Default::default()
    })
    .is_err());
    assert!(GpuLptSeed::new(&LabRequest {
        dim: 2,
        n: 24,
        ..Default::default()
    })
    .is_err());
}
