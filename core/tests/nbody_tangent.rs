use hcc_core::{
    fft::FftEngine,
    grid::Grid,
    growth::Cosmology,
    lpt::Lpt,
    nbody::{
        cic_density_jvp, cosmology_clocks, evolve_cosmology, evolve_eds, force_cosmology_jvp,
        force_jvp, PmState, PmTangent,
    },
    tangent::{
        lpt_amplitude_jvp, simplex_mass_transform_jvp, simplex_mass_transform_p1_jvp, LptJvp,
    },
};
use num_complex::Complex64 as C64;

fn lattice(g: &Grid, offset: f64) -> Vec<f64> {
    (0..g.size)
        .flat_map(|i| {
            let ijk = g.unravel(i);
            (0..g.dim).map(move |a| (ijk[a] as f64 + offset) * g.dx())
        })
        .collect()
}
fn rms(x: &[f64]) -> f64 {
    (x.iter().map(|v| v * v).sum::<f64>() / x.len() as f64).sqrt()
}
fn difference(a: &[f64], b: &[f64]) -> Vec<f64> {
    a.iter().zip(b).map(|(x, y)| x - y).collect()
}

#[test]
fn cic_mass_and_tangent_conserve_in_all_dimensions() {
    for dim in 1..=3 {
        let g = Grid::new(dim, 8, 1.0);
        let mut x = lattice(&g, 0.37);
        let dx: Vec<_> = (0..x.len())
            .map(|i| 0.03 * ((i * 17 + 3) as f64).sin())
            .collect();
        for (i, v) in x.iter_mut().enumerate() {
            *v += 0.015 * ((i * 11 + 1) as f64).cos();
        }
        let (rho, drho) = cic_density_jvp(&g, &x, Some(&dx));
        assert!((rho.iter().sum::<f64>() - g.size as f64).abs() < 1e-10);
        assert!(drho.iter().sum::<f64>().abs() < 1e-10);
        assert!(rho.iter().all(|v| *v >= 0.0));
        let eps = 1e-6;
        let xp: Vec<_> = x.iter().zip(&dx).map(|(v, d)| v + eps * d).collect();
        let xm: Vec<_> = x.iter().zip(&dx).map(|(v, d)| v - eps * d).collect();
        let rp = cic_density_jvp(&g, &xp, None).0;
        let rm = cic_density_jvp(&g, &xm, None).0;
        for i in 0..g.size {
            assert!(((rp[i] - rm[i]) / (2.0 * eps) - drho[i]).abs() < 2e-8);
        }
    }
}
#[test]
fn uniform_particles_remain_stationary() {
    for dim in 1..=3 {
        let g = Grid::new(dim, 8, 1.0);
        let x = lattice(&g, 0.37);
        let p = vec![0.0; x.len()];
        let r = evolve_eds(
            &g,
            &g,
            PmState {
                x: x.clone(),
                p,
                a: 0.1,
            },
            None,
            0.5,
            12,
        )
        .unwrap();
        assert!(rms(&difference(&r.state.x, &x)) < 1e-14);
        assert!(rms(&r.state.p) < 1e-14);
        assert!((r.min_jacobian - 1.0).abs() < 1e-12);
    }
}
fn plane_run(steps: usize) -> (Grid, Vec<f64>, hcc_core::nbody::PmRun) {
    let g = Grid::new(1, 128, 1.0);
    let q = lattice(&g, 0.0);
    let a0: f64 = 0.1;
    let vel: Vec<_> = q
        .iter()
        .map(|x| -0.002 * (g.kf() * x).sin() / g.kf())
        .collect();
    let x: Vec<_> = q.iter().zip(&vel).map(|(q, v)| q + a0 * v).collect();
    let p: Vec<_> = vel.iter().map(|v| a0.powf(1.5) * v).collect();
    let r = evolve_eds(&g, &g, PmState { x, p, a: a0 }, None, 0.4, steps).unwrap();
    (g, q, r)
}
#[test]
fn plane_mode_follows_eds_growth_and_converges_in_step_count() {
    let (g, q, r) = plane_run(128);
    let disp = difference(&r.state.x, &q);
    let expected: Vec<_> = q
        .iter()
        .map(|x| -0.4 * 0.002 * (g.kf() * x).sin() / g.kf())
        .collect();
    assert!(
        rms(&difference(&disp, &expected)) / rms(&expected) < 0.004,
        "growth mismatch"
    );
    let x16 = plane_run(16).2.state.x;
    let x32 = plane_run(32).2.state.x;
    let x64 = plane_run(64).2.state.x;
    let e16 = rms(&difference(&x16, &r.state.x));
    let e32 = rms(&difference(&x32, &r.state.x));
    let e64 = rms(&difference(&x64, &r.state.x));
    assert!(
        e16 / e32 > 3.5 && e32 / e64 > 3.5,
        "errors {e16} {e32} {e64}"
    );
}
#[test]
fn complete_force_and_kdk_jvp_match_finite_difference() {
    for dim in 1..=3 {
        let g = Grid::new(dim, 8, 1.0);
        let mut x = lattice(&g, 0.37);
        let mut p = vec![0.0; x.len()];
        let dx: Vec<_> = (0..x.len())
            .map(|i| 0.013 * ((i * 17 + 3) as f64).sin())
            .collect();
        let dp: Vec<_> = (0..x.len())
            .map(|i| 0.0002 * ((i * 13 + 7) as f64).cos())
            .collect();
        for i in 0..x.len() {
            x[i] += 0.008 * ((i * 11 + 1) as f64).cos();
            p[i] = 0.0001 * ((i * 3 + 5) as f64).cos();
        }
        let (_, df) = force_jvp(&g, &x, Some(&dx));
        let eps = 1e-5;
        let plus = |sign: f64| PmState {
            x: x.iter().zip(&dx).map(|(v, d)| v + sign * eps * d).collect(),
            p: p.iter().zip(&dp).map(|(v, d)| v + sign * eps * d).collect(),
            a: 0.2,
        };
        let fp = force_jvp(&g, &plus(1.).x, None).0;
        let fm = force_jvp(&g, &plus(-1.).x, None).0;
        let fd: Vec<_> = fp
            .iter()
            .zip(fm)
            .map(|(a, b)| (a - b) / (2. * eps))
            .collect();
        assert!(rms(&difference(&df, &fd)) < 2e-8);
        let primal = evolve_eds(
            &g,
            &g,
            PmState {
                x: x.clone(),
                p: p.clone(),
                a: 0.2,
            },
            Some(PmTangent {
                dx: dx.clone(),
                dp: dp.clone(),
            }),
            0.28,
            7,
        )
        .unwrap();
        let rp = evolve_eds(&g, &g, plus(1.), None, 0.28, 7).unwrap();
        let rm = evolve_eds(&g, &g, plus(-1.), None, 0.28, 7).unwrap();
        let t = primal.tangent.unwrap();
        let fd: Vec<_> = rp
            .state
            .x
            .iter()
            .zip(rm.state.x)
            .map(|(a, b)| (a - b) / (2. * eps))
            .collect();
        assert!(rms(&difference(&t.dx, &fd)) < 3e-8);
        let fd: Vec<_> = rp
            .state
            .p
            .iter()
            .zip(rm.state.p)
            .map(|(a, b)| (a - b) / (2. * eps))
            .collect();
        assert!(rms(&difference(&t.dp, &fd)) < 3e-8);
    }
}
#[test]
fn pre_crossing_guard_rejects_inverted_initial_and_drift_endpoint() {
    let g = Grid::new(1, 8, 1.0);
    let x = lattice(&g, 0.37);
    let mut bad = x.clone();
    bad[2] = bad[1] - 0.01;
    assert!(evolve_eds(
        &g,
        &g,
        PmState {
            x: bad,
            p: vec![0.; 8],
            a: 0.2
        },
        None,
        0.3,
        1
    )
    .is_err());
    let mut p = vec![0.; 8];
    p[2] = -1.;
    assert!(evolve_eds(&g, &g, PmState { x, p, a: 0.2 }, None, 0.3, 1).is_err());
}
fn ic_fields(g: &Grid) -> (Vec<C64>, Vec<C64>) {
    let f: Vec<_> = (0..g.size)
        .map(|i| {
            let j = g.unravel(i);
            let x = j[0] as f64 * g.dx();
            let y = j[1] as f64 * g.dx();
            let z = j[2] as f64 * g.dx();
            0.3 * (g.kf() * x).cos()
                + 0.19 * (g.kf() * (x + y)).sin()
                + if g.dim == 3 {
                    0.12 * (g.kf() * (y + z)).cos()
                } else {
                    0.
                }
        })
        .collect();
    let df: Vec<_> = (0..g.size)
        .map(|i| {
            let j = g.unravel(i);
            let x = j[0] as f64 * g.dx();
            let y = j[1] as f64 * g.dx();
            let z = j[2] as f64 * g.dx();
            0.17 * (g.kf() * y).sin()
                + 0.13 * (g.kf() * (x - y)).cos()
                + if g.dim == 3 {
                    0.08 * (g.kf() * (x + z)).sin()
                } else {
                    0.
                }
        })
        .collect();
    let mut eng = FftEngine::new(g);
    (eng.forward_real(&f), eng.forward_real(&df))
}
#[test]
fn arbitrary_ic_jvp_matches_fourth_order_finite_difference_eds_and_lcdm() {
    for dim in 2..=3 {
        for cosmo in [Cosmology::eds(), Cosmology { omega_m: 0.3 }] {
            let g = Grid::new(dim, 8, 1.0);
            let (delta, dir) = ic_fields(&g);
            let l = LptJvp::new(&g, &delta, &dir, 4, cosmo);
            let (x, v, dx, dv) = l.evaluate(0.5, 4);
            let eps = 1e-5;
            let dp: Vec<_> = delta.iter().zip(&dir).map(|(a, b)| a + eps * b).collect();
            let dm: Vec<_> = delta.iter().zip(&dir).map(|(a, b)| a - eps * b).collect();
            let zero = vec![C64::new(0., 0.); g.size];
            let plus = LptJvp::new(&g, &dp, &zero, 4, cosmo).evaluate(0.5, 4);
            let minus = LptJvp::new(&g, &dm, &zero, 4, cosmo).evaluate(0.5, 4);
            let fd: Vec<_> = plus
                .0
                .iter()
                .zip(minus.0)
                .map(|(a, b)| (a - b) / (2. * eps))
                .collect();
            assert!(rms(&difference(&dx, &fd)) < 1e-10);
            let fd: Vec<_> = plus
                .1
                .iter()
                .zip(minus.1)
                .map(|(a, b)| (a - b) / (2. * eps))
                .collect();
            assert!(rms(&difference(&dv, &fd)) < 1e-10);
            let mut eng = FftEngine::new(&g);
            let reference = Lpt::new(&g, &mut eng, &delta, 4, cosmo, true);
            let mut xr = vec![0f32; x.len()];
            let mut vr = xr.clone();
            reference.positions(0.5, 4, &mut xr);
            reference.velocities(0.5, 4, &mut vr);
            assert!(
                rms(&difference(
                    &x,
                    &xr.iter().map(|v| *v as f64).collect::<Vec<_>>()
                )) < 3e-8
            );
            assert!(
                rms(&difference(
                    &v,
                    &vr.iter().map(|v| *v as f64).collect::<Vec<_>>()
                )) < 3e-8
            );
        }
    }
}
#[test]
fn homogeneous_amplitude_jvp_matches_general_direction() {
    for dim in 2..=3 {
        let g = Grid::new(dim, 8, 1.0);
        let (delta, _) = ic_fields(&g);
        let general = LptJvp::new(&g, &delta, &delta, 4, Cosmology::eds()).evaluate(0.5, 4);
        let mut eng = FftEngine::new(&g);
        let l = Lpt::new(&g, &mut eng, &delta, 4, Cosmology::eds(), true);
        let (dx, dv) = lpt_amplitude_jvp(&l, 0.5, 4);
        assert!(rms(&difference(&dx, &general.2)) < 3e-9);
        assert!(rms(&difference(&dv, &general.3)) < 1e-8);
    }
}
#[test]
fn simplex_jvp_mass_degeneracies_translation_and_p1_finite_difference() {
    for dim in 1..=3 {
        let v: Vec<_> = (0..=dim)
            .map(|i| {
                [
                    0.2 + 0.1 * i as f64,
                    0.1 + 0.03 * (i * i) as f64,
                    0.3 - 0.02 * (i * i * i) as f64,
                ]
            })
            .collect();
        let dv: Vec<_> = (0..=dim)
            .map(|i| {
                [
                    0.05 * (i as f64 + 1.).sin(),
                    -0.03 * (i as f64 + 2.).cos(),
                    0.02,
                ]
            })
            .collect();
        let k = [6.1, -3.7, 2.4];
        let mass = 0.17;
        let eps = 1e-5;
        let (_, jvp) = simplex_mass_transform_jvp(&v, &dv, dim, k, mass);
        let mut vp = v.clone();
        let mut vm = v.clone();
        for i in 0..v.len() {
            for a in 0..dim {
                vp[i][a] += eps * dv[i][a];
                vm[i][a] -= eps * dv[i][a];
            }
        }
        let fd = (simplex_mass_transform_jvp(&vp, &dv, dim, k, mass).0
            - simplex_mass_transform_jvp(&vm, &dv, dim, k, mass).0)
            / (2. * eps);
        assert!((fd - jvp).norm() < 1e-9);
        let (zero, dzero) = simplex_mass_transform_jvp(&v, &dv, dim, [0.; 3], mass);
        assert!((zero.re - mass).abs() < 1e-14 && dzero.norm() == 0.);
        let shift = [0.07, -0.02, 0.03];
        let trans = vec![shift; v.len()];
        let (value, deriv) = simplex_mass_transform_jvp(&v, &trans, dim, k, mass);
        let phase = C64::new(0., -(0..dim).map(|a| k[a] * shift[a]).sum::<f64>());
        assert!((deriv - phase * value).norm() < 1e-12);
        let collapsed = vec![[0.2, 0.2, 0.2]; v.len()];
        let (cv, cd) = simplex_mass_transform_jvp(&collapsed, &trans, dim, k, mass);
        assert!((cd - phase * cv).norm() < 1e-12);
        let w: Vec<_> = (0..v.len()).map(|i| 1. + 0.2 * i as f64).collect();
        let dw: Vec<_> = (0..v.len()).map(|i| 0.03 * (i as f64 + 1.).cos()).collect();
        let (_, dj) = simplex_mass_transform_p1_jvp(&v, &dv, &w, &dw, dim, k, mass);
        let wp: Vec<_> = w.iter().zip(&dw).map(|(a, b)| a + eps * b).collect();
        let wm: Vec<_> = w.iter().zip(&dw).map(|(a, b)| a - eps * b).collect();
        let fd = (simplex_mass_transform_p1_jvp(&vp, &dv, &wp, &dw, dim, k, mass).0
            - simplex_mass_transform_p1_jvp(&vm, &dv, &wm, &dw, dim, k, mass).0)
            / (2. * eps);
        assert!((fd - dj).norm() < 1e-9);
        let (z, dz) = simplex_mass_transform_p1_jvp(&v, &dv, &w, &dw, dim, [0.; 3], mass);
        assert!((z.re - mass).abs() < 1e-14 && dz.norm() < 1e-14);
    }
}

#[test]
fn drift_guard_detects_interior_fold_with_positive_endpoint_jacobians() {
    // A smooth radial twist preserves orientation at both endpoints. Its
    // straight-line interpolation collapses the inner disk halfway through.
    let g = Grid::new(2, 256, 1.0);
    let x = lattice(&g, 0.37);
    let mut end = x.clone();
    for i in 0..g.size {
        let u = x[2 * i] - 0.5;
        let v = x[2 * i + 1] - 0.5;
        let r = u.hypot(v);
        let angle = if r < 0.15 {
            std::f64::consts::PI
        } else if r > 0.45 {
            0.0
        } else {
            std::f64::consts::PI * 0.5 * (1.0 + (std::f64::consts::PI * (r - 0.15) / 0.3).cos())
        };
        end[2 * i] = 0.5 + angle.cos() * u - angle.sin() * v;
        end[2 * i + 1] = 0.5 + angle.sin() * u + angle.cos() * v;
    }
    let end_min = hcc_core::observables::minimum_simplex_jacobian(&g, &end);
    assert!(end_min > 0.0, "endpoint min {end_min}");
    let (a0, a1) = (0.2_f64, 0.3_f64);
    let drift = 2.0 * (1.0 / a0.sqrt() - 1.0 / a1.sqrt());
    let kick = 2.0 * ((0.5 * (a0 + a1)).sqrt() - a0.sqrt());
    let force = force_jvp(&g, &x, None).0;
    let p = end
        .iter()
        .zip(&x)
        .zip(force)
        .map(|((end, start), force)| (end - start) / drift - kick * force)
        .collect();
    assert!(evolve_eds(&g, &g, PmState { x, p, a: a0 }, None, a1, 1).is_err());
}

#[test]
fn one_dimensional_general_lpt_reduces_to_exact_zeldovich() {
    let g = Grid::new(1, 16, 1.0);
    let mut eng = FftEngine::new(&g);
    let delta = eng.forward_real(
        &(0..g.n)
            .map(|i| 0.3 * (g.kf() * i as f64 * g.dx()).cos())
            .collect::<Vec<_>>(),
    );
    let direction = eng.forward_real(
        &(0..g.n)
            .map(|i| 0.2 * (2.0 * g.kf() * i as f64 * g.dx()).sin())
            .collect::<Vec<_>>(),
    );
    let high = LptJvp::new(&g, &delta, &direction, 4, Cosmology::eds()).evaluate(0.5, 4);
    let za = LptJvp::new(&g, &delta, &direction, 1, Cosmology::eds()).evaluate(0.5, 1);
    assert!(rms(&difference(&high.0, &za.0)) < 1e-15);
    assert!(rms(&difference(&high.1, &za.1)) < 1e-15);
    assert!(rms(&difference(&high.2, &za.2)) < 1e-15);
    assert!(rms(&difference(&high.3, &za.3)) < 1e-15);
}

/// Jointly refine the particle lattice and its force mesh. A finer force mesh
/// alone need not improve a sparsely sampled particle representation. Here the
/// reference is fourth-order LPT at a fixed pre-crossing epoch, not an assertion
/// that either finite-resolution PM or fourth-order LPT is the exact solution.
#[test]
fn mixed_waves_joint_particle_mesh_refinement_converges_toward_lpt() {
    use hcc_core::{ics::Preset, observables::sheet_mode_jvp, Cosmo};
    fn evaluate(c: &Cosmo, a: f64) -> (Vec<f64>, Vec<f64>) {
        let g = &c.grid;
        let mut x = lattice(g, 0.0);
        let mut v = vec![0.0; x.len()];
        for (t, term) in c.lpt_ref().terms.iter().enumerate() {
            let (growth, derivative) = c.lpt_ref().growth.g_and_dg(t, a);
            for i in 0..g.size {
                for axis in 0..g.dim {
                    x[i * g.dim + axis] += growth * term.psi[axis][i] as f64;
                    v[i * g.dim + axis] += derivative * term.psi[axis][i] as f64;
                }
            }
        }
        (x, v)
    }
    let mut displacement_errors = Vec::new();
    let mut fundamental_power_ratios = Vec::new();
    for n in [16, 32, 64] {
        let mut c = Cosmo::new(2, n, 1.0);
        c.keep_tensors = Some(true);
        c.set_ic_preset(
            Preset::PlaneWaves {
                modes: vec![
                    ([1, 0, 0], 0.5, 0.17),
                    ([0, 1, 0], 0.35, 0.43),
                    ([1, 1, 0], 0.15, 0.7),
                ],
            },
            0.0,
            0.0,
        );
        c.build_lpt(4);
        let a_start = 0.03_f64;
        let a_end = 0.7;
        let (x0, v0) = evaluate(&c, a_start);
        let p0 = v0.iter().map(|v| a_start.powf(1.5) * v).collect();
        let pm = evolve_eds(
            &c.grid,
            &c.grid,
            PmState {
                x: x0,
                p: p0,
                a: a_start,
            },
            None,
            a_end,
            128,
        )
        .unwrap();
        let (lpt_x, _) = evaluate(&c, a_end);
        assert!(hcc_core::observables::minimum_simplex_jacobian(&c.grid, &lpt_x) > 0.0);
        assert!(pm.min_jacobian > 0.0);
        let q = lattice(&c.grid, 0.0);
        let displacement_error =
            rms(&difference(&pm.state.x, &lpt_x)) / rms(&difference(&lpt_x, &q));
        let zero = vec![0.0; lpt_x.len()];
        let k = [c.grid.kf(), 0.0, 0.0];
        let lpt_mode = sheet_mode_jvp(&c.grid, &lpt_x, &zero, k).0;
        let pm_mode = sheet_mode_jvp(&c.grid, &pm.state.x, &zero, k).0;
        let power_ratio = pm_mode.norm_sqr() / lpt_mode.norm_sqr();
        println!("n=forceMesh={n}: displacementRelativeRms={displacement_error:.9}, fundamentalPowerRatio={power_ratio:.9}");
        displacement_errors.push(displacement_error);
        fundamental_power_ratios.push(power_ratio);
    }
    // Require substantial refinement improvements rather than incidental small
    // changes. Leave room for legitimate small compiler and FFT differences.
    for errors in displacement_errors.windows(2) {
        assert!(
            errors[1] < 0.6 * errors[0],
            "displacement residuals: {displacement_errors:?}"
        );
    }
    assert!(
        displacement_errors[0] > 0.05 && displacement_errors[2] < 0.025,
        "displacement residuals: {displacement_errors:?}"
    );
    let power_errors: Vec<f64> = fundamental_power_ratios
        .iter()
        .map(|ratio| (ratio - 1.0).abs())
        .collect();
    for errors in power_errors.windows(2) {
        assert!(
            errors[1] < 0.5 * errors[0],
            "fundamental ratios: {fundamental_power_ratios:?}"
        );
    }
    assert!(
        power_errors[0] > 0.05 && power_errors[2] < 0.01,
        "fundamental ratios: {fundamental_power_ratios:?}"
    );
}

// An independent composite Simpson reference uses E(a) directly rather than
// the production quadrature's rearranged integrands or Gauss-Legendre nodes.
fn reference_clocks(cosmo: Cosmology, a0: f64, a1: f64) -> (f64, f64) {
    let intervals = 20000;
    let width = (a1.ln() - a0.ln()) / intervals as f64;
    let mut kick = 0.0;
    let mut drift = 0.0;
    for j in 0..=intervals {
        let a = (a0.ln() + j as f64 * width).exp();
        let weight = if j == 0 || j == intervals {
            1.0
        } else if j % 2 == 0 {
            2.0
        } else {
            4.0
        };
        kick += weight / (a * cosmo.e(a));
        drift += weight / (a * a * cosmo.e(a));
    }
    (kick * width / 3.0, drift * width / 3.0)
}

#[test]
fn lcdm_clocks_match_independent_quadrature_and_eds_closed_forms() {
    let (kick, drift) = cosmology_clocks(Cosmology::eds(), 0.04, 0.25);
    assert!((kick - 0.6).abs() < 1e-15);
    assert!((drift - 6.0).abs() < 1e-14);
    for omega_m in [0.3, 0.7, 0.999999, 1.0] {
        let cosmo = Cosmology { omega_m };
        for (a0, a1) in [(0.0001, 0.05), (0.03, 1.0), (0.1, 10.0)] {
            let actual = cosmology_clocks(cosmo, a0, a1);
            let reference = reference_clocks(cosmo, a0, a1);
            assert!(
                (actual.0 / reference.0 - 1.0).abs() < 8e-13,
                "kick Omega_m={omega_m}: {actual:?} vs {reference:?}"
            );
            assert!(
                (actual.1 / reference.1 - 1.0).abs() < 8e-13,
                "drift Omega_m={omega_m}: {actual:?} vs {reference:?}"
            );
            let mid = (a0 * a1).sqrt();
            let left = cosmology_clocks(cosmo, a0, mid);
            let right = cosmology_clocks(cosmo, mid, a1);
            assert!(((left.0 + right.0) / actual.0 - 1.0).abs() < 2e-14);
            assert!(((left.1 + right.1) / actual.1 - 1.0).abs() < 2e-14);
        }
    }
}

#[test]
fn uniform_bulk_motion_uses_canonical_lcdm_drift_and_eds_wrapper_matches() {
    let g = Grid::new(2, 8, 1.0);
    let x = lattice(&g, 0.37);
    let p: Vec<_> = (0..x.len())
        .map(|j| if j % 2 == 0 { 0.004 } else { -0.003 })
        .collect();
    for cosmo in [Cosmology::eds(), Cosmology { omega_m: 0.3 }] {
        let initial = PmState {
            x: x.clone(),
            p: p.clone(),
            a: 0.03,
        };
        let run = evolve_cosmology(&g, &g, initial.clone(), None, 1.0, 32, cosmo, false).unwrap();
        let drift = reference_clocks(cosmo, initial.a, run.state.a).1;
        let expected: Vec<_> = x.iter().zip(&p).map(|(x, p)| x + drift * p).collect();
        assert!(rms(&difference(&run.state.x, &expected)) < 2e-14);
        assert!(rms(&difference(&run.state.p, &p)) < 2e-15);
        assert!(!run.crossed && run.first_crossing.is_none());
        assert!((run.min_jacobian - 1.0).abs() < 2e-13);
        if cosmo.is_eds() {
            let old = evolve_eds(&g, &g, initial, None, 1.0, 32).unwrap();
            assert_eq!(old.state.x, run.state.x);
            assert_eq!(old.state.p, run.state.p);
        }
    }
}

#[test]
fn lcdm_plane_mode_tracks_linear_growth() {
    use hcc_core::growth::{growth_rate, term_specs, Growth};
    let g = Grid::new(1, 256, 1.0);
    let cosmo = Cosmology { omega_m: 0.3 };
    let growth = Growth::new_lcdm(cosmo, &term_specs(1, 1), 20.0, 4000);
    let (a0, a1) = (0.03_f64, 1.0);
    let (d0, d1) = (growth.d_of_a(a0), growth.d_of_a(a1));
    let dd_da = growth_rate(&growth, a0) * d0 / a0;
    let q = lattice(&g, 0.0);
    let shape: Vec<_> = q
        .iter()
        .map(|x| -0.002 * (g.kf() * x).sin() / g.kf())
        .collect();
    let x = q.iter().zip(&shape).map(|(q, s)| q + d0 * s).collect();
    let p = shape
        .iter()
        .map(|s| a0.powi(3) * cosmo.e(a0) * dd_da * s)
        .collect();
    let run =
        evolve_cosmology(&g, &g, PmState { x, p, a: a0 }, None, a1, 256, cosmo, false).unwrap();
    let expected: Vec<_> = shape.iter().map(|s| d1 * s).collect();
    let actual = difference(&run.state.x, &q);
    let error = rms(&difference(&actual, &expected)) / rms(&expected);
    println!("LCDM linear plane mode relative RMS residual: {error:.9}");
    assert!(error < 0.006, "LCDM linear growth residual {error}");
}

#[test]
fn lcdm_force_and_kdk_jvp_match_fixed_cosmology_finite_difference() {
    let cosmo = Cosmology { omega_m: 0.3 };
    for dim in 1..=3 {
        let g = Grid::new(dim, 8, 1.0);
        let mut x = lattice(&g, 0.37);
        let mut p = vec![0.0; x.len()];
        let dx: Vec<_> = (0..x.len())
            .map(|i| 0.013 * ((i * 17 + 3) as f64).sin())
            .collect();
        let dp: Vec<_> = (0..x.len())
            .map(|i| 0.0002 * ((i * 13 + 7) as f64).cos())
            .collect();
        for i in 0..x.len() {
            x[i] += 0.008 * ((i * 11 + 1) as f64).cos();
            p[i] = 0.0001 * ((i * 3 + 5) as f64).cos();
        }
        let (force, df) = force_cosmology_jvp(&g, &x, Some(&dx), cosmo);
        let eds = force_jvp(&g, &x, Some(&dx));
        assert!(
            rms(&difference(
                &force,
                &eds.0.iter().map(|f| cosmo.omega_m * f).collect::<Vec<_>>()
            )) < 1e-14
        );
        let eps = 1e-5;
        let perturbed = |sign: f64| PmState {
            x: x.iter().zip(&dx).map(|(x, d)| x + sign * eps * d).collect(),
            p: p.iter().zip(&dp).map(|(p, d)| p + sign * eps * d).collect(),
            a: 0.2,
        };
        let fp = force_cosmology_jvp(&g, &perturbed(1.0).x, None, cosmo).0;
        let fm = force_cosmology_jvp(&g, &perturbed(-1.0).x, None, cosmo).0;
        let fd: Vec<_> = fp
            .iter()
            .zip(&fm)
            .map(|(p, m)| (p - m) / (2.0 * eps))
            .collect();
        assert!(rms(&difference(&df, &fd)) < 2e-8);
        let primal = evolve_cosmology(
            &g,
            &g,
            PmState {
                x: x.clone(),
                p: p.clone(),
                a: 0.2,
            },
            Some(PmTangent {
                dx: dx.clone(),
                dp: dp.clone(),
            }),
            0.4,
            11,
            cosmo,
            false,
        )
        .unwrap();
        let plus = evolve_cosmology(&g, &g, perturbed(1.0), None, 0.4, 11, cosmo, false).unwrap();
        let minus = evolve_cosmology(&g, &g, perturbed(-1.0), None, 0.4, 11, cosmo, false).unwrap();
        let t = primal.tangent.unwrap();
        let fd: Vec<_> = plus
            .state
            .x
            .iter()
            .zip(&minus.state.x)
            .map(|(p, m)| (p - m) / (2.0 * eps))
            .collect();
        assert!(rms(&difference(&t.dx, &fd)) < 3e-8);
        let fd: Vec<_> = plus
            .state
            .p
            .iter()
            .zip(&minus.state.p)
            .map(|(p, m)| (p - m) / (2.0 * eps))
            .collect();
        assert!(rms(&difference(&t.dp, &fd)) < 3e-8);
    }
}

#[test]
fn crossing_policy_continues_and_records_first_drift_or_crossed_input() {
    let g = Grid::new(1, 8, 1.0);
    let x = lattice(&g, 0.37);
    let mut p = vec![0.0; g.size];
    p[2] = -1.0;
    let state = PmState {
        x: x.clone(),
        p,
        a: 0.2,
    };
    for cosmo in [Cosmology::eds(), Cosmology { omega_m: 0.3 }] {
        assert!(evolve_cosmology(&g, &g, state.clone(), None, 0.3, 3, cosmo, false).is_err());
        let run = evolve_cosmology(&g, &g, state.clone(), None, 0.3, 3, cosmo, true).unwrap();
        assert!(run.crossed && run.min_jacobian < 0.0);
        assert_eq!(run.completed_steps, 3);
        assert_eq!(run.state.a, 0.3);
        assert!(run
            .state
            .x
            .iter()
            .chain(&run.state.p)
            .all(|v| v.is_finite()));
        let crossing = run.first_crossing.unwrap();
        assert_eq!(crossing.step, 1);
        assert_eq!(crossing.a_start, 0.2);
        assert!((crossing.a_end - (0.2 + 0.1 / 3.0)).abs() < 1e-14);
        assert!(crossing.minimum_jacobian <= 0.0);
    }
    let mut crossed_x = x;
    crossed_x[2] = crossed_x[1] - 0.01;
    let run = evolve_cosmology(
        &g,
        &g,
        PmState {
            x: crossed_x,
            p: vec![0.0; g.size],
            a: 0.2,
        },
        None,
        0.25,
        3,
        Cosmology { omega_m: 0.3 },
        true,
    )
    .unwrap();
    assert!(run.crossed && run.min_jacobian < 0.0);
    let first = run.first_crossing.unwrap();
    assert_eq!(first.step, 0);
    assert_eq!(first.a_start, 0.2);
    assert_eq!(first.a_end, 0.2);
}

#[test]
fn lcdm_jvp_remains_valid_when_crossing_is_diagnostic_only() {
    let g = Grid::new(1, 8, 1.0);
    let cosmo = Cosmology { omega_m: 0.3 };
    let x = lattice(&g, 0.37);
    let mut p = vec![0.0; g.size];
    p[2] = -1.0;
    let dx: Vec<_> = (0..g.size)
        .map(|i| 0.013 * (i as f64 + 0.7).sin())
        .collect();
    let dp: Vec<_> = (0..g.size)
        .map(|i| 0.0002 * (i as f64 + 0.3).cos())
        .collect();
    let run = evolve_cosmology(
        &g,
        &g,
        PmState {
            x: x.clone(),
            p: p.clone(),
            a: 0.2,
        },
        Some(PmTangent {
            dx: dx.clone(),
            dp: dp.clone(),
        }),
        0.3,
        3,
        cosmo,
        true,
    )
    .unwrap();
    assert!(run.crossed);
    let eps = 1e-6;
    let perturbed = |sign: f64| PmState {
        x: x.iter().zip(&dx).map(|(x, d)| x + sign * eps * d).collect(),
        p: p.iter().zip(&dp).map(|(p, d)| p + sign * eps * d).collect(),
        a: 0.2,
    };
    let plus = evolve_cosmology(&g, &g, perturbed(1.0), None, 0.3, 3, cosmo, true).unwrap();
    let minus = evolve_cosmology(&g, &g, perturbed(-1.0), None, 0.3, 3, cosmo, true).unwrap();
    assert!(plus.crossed && minus.crossed);
    let tangent = run.tangent.unwrap();
    let fdx: Vec<_> = plus
        .state
        .x
        .iter()
        .zip(&minus.state.x)
        .map(|(p, m)| (p - m) / (2.0 * eps))
        .collect();
    let fdp: Vec<_> = plus
        .state
        .p
        .iter()
        .zip(&minus.state.p)
        .map(|(p, m)| (p - m) / (2.0 * eps))
        .collect();
    assert!(rms(&difference(&tangent.dx, &fdx)) < 3e-8);
    assert!(rms(&difference(&tangent.dp, &fdp)) < 3e-8);
}
