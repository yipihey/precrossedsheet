use hcc_core::{
    cosmological::{gaussian_hat, top_hat, transfer, variance},
    grid::Grid,
};
#[test]
fn smooth_transfer_has_correct_limits_and_sigma8_integral_converges() {
    assert_eq!(transfer(0., 0.315, 0.049, 0.674), 1.);
    assert!((transfer(1e-7, 0.315, 0.049, 0.674) - 1.).abs() < 1e-8);
    assert!(transfer(0.2, 0.315, 0.049, 0.674) < transfer(0.2, 0.315, 0., 0.674));
    assert!((top_hat(1e-6) - 1.).abs() < 1e-12);
    let v = variance(0.315, 0.049, 0.674, 0.965, 8., 4096);
    let refined = variance(0.315, 0.049, 0.674, 0.965, 8., 16384);
    assert!((v / refined - 1.).abs() < 1e-6);
    let amp = 0.811_f64.powi(2) / v;
    assert!((amp * refined - 0.811_f64.powi(2)).abs() < 1e-6);
}
#[test]
fn gaussian_modes_are_hermitian_and_identical_under_lattice_refinement() {
    for dim in 1..=3 {
        let a = Grid::new(dim, 32, 1.);
        let b = Grid::new(dim, 64, 1.);
        let generate = |g: &Grid| {
            gaussian_hat(
                g, 2650, 3, 300., 0.315, 0.049, 0.674, 0.965, 0.811, 14., 0.7878, false,
            )
        };
        let ha = generate(&a);
        let hb = generate(&b);
        for i in 0..a.size {
            let ijk = a.unravel(i);
            let mut enlarged = [0; 3];
            let mut opposite = [0; 3];
            for d in 0..dim {
                let m = a.ifreq(ijk[d]);
                enlarged[d] = m.rem_euclid(64) as usize;
                opposite[d] = (-m).rem_euclid(32) as usize;
            }
            assert_eq!(ha[i], ha[a.ravel(opposite)].conj());
            assert!((ha[i] / a.size as f64 - hb[b.ravel(enlarged)] / b.size as f64).norm() < 1e-14);
        }
        assert_eq!(ha[0].norm(), 0.);
        let mut changed = generate(&a);
        let different = gaussian_hat(
            &a, 2651, 3, 300., 0.315, 0.049, 0.674, 0.965, 0.811, 14., 0.7878, false,
        );
        assert!(changed
            .iter()
            .zip(different)
            .any(|(x, y)| (*x - y).norm() > 0.));
        changed.clear();
    }
}

#[test]
fn bao_transfer_oscillates_about_the_smooth_fit_and_shares_its_limits() {
    use hcc_core::cosmological::{transfer_bao, transfer_model, variance_model, LinearSpectrum};
    let (om, ob, h) = (0.315, 0.049, 0.674);
    assert_eq!(transfer_bao(0., om, ob, h), 1.);
    let ratio = |k: f64| transfer_bao(k, om, ob, h) / transfer(k, om, ob, h);
    // Both fits share the large-scale limit and agree to a few percent on small scales.
    assert!((ratio(1e-4) - 1.).abs() < 1e-3);
    assert!((ratio(1e-3) - 1.).abs() < 1e-3);
    let mut k = 0.01;
    while k < 50. {
        assert!((ratio(k) - 1.).abs() < 0.05, "k={k}: ratio {}", ratio(k));
        k *= 1.05;
    }
    // Acoustic oscillations: several sign changes of the ratio across the BAO band,
    // with the first trough near k~0.05 and the first crest near k~0.08 h/Mpc.
    let mut crossings = 0;
    let mut previous = 0.0f64;
    let mut k = 0.03;
    while k < 0.5 {
        let r = ratio(k) - 1.;
        if previous != 0. && r.signum() != previous.signum() {
            crossings += 1;
        }
        previous = r;
        k *= 1.01;
    }
    assert!(crossings >= 6, "only {crossings} sign changes");
    let extremum = |lo: f64, hi: f64, want_min: bool| {
        let mut best = if want_min { f64::INFINITY } else { f64::NEG_INFINITY };
        let mut k = lo;
        while k <= hi {
            let r = ratio(k);
            best = if want_min { best.min(r) } else { best.max(r) };
            k += 0.0005;
        }
        best
    };
    assert!(extremum(0.035, 0.065, true) < 0.985);
    assert!(extremum(0.065, 0.095, false) > 1.02);
    // The sigma8 normalization is nearly the same because the wiggles average out.
    let vs = variance_model(om, ob, h, 0.965, 8., 4096, false);
    let vb = variance_model(om, ob, h, 0.965, 8., 4096, true);
    assert!((vb / vs - 1.).abs() < 0.01);
    assert_eq!(transfer_model(0.1, om, ob, h, true), transfer_bao(0.1, om, ob, h));
    assert_eq!(transfer_model(0.1, om, ob, h, false), transfer(0.1, om, ob, h));
    // LinearSpectrum fixes the amplitude so the top-hat variance equals sigma8^2 for either fit.
    for bao in [false, true] {
        let s = LinearSpectrum::new(om, ob, h, 0.965, 0.811, bao);
        let v = variance_model(om, ob, h, 0.965, 8., 4096, bao) * s.amplitude;
        assert!((v - 0.811_f64.powi(2)).abs() < 1e-9);
        assert_eq!(s.power(0.), 0.);
        assert!((s.power(0.1) / (s.amplitude * 0.1_f64.powf(0.965) * transfer_model(0.1, om, ob, h, bao).powi(2)) - 1.).abs() < 1e-12);
    }
}

#[test]
fn filtered_linear_spectrum_follows_the_stated_formula_and_vanishes_above_the_cutoff() {
    use hcc_core::cosmological::LinearSpectrum;
    use std::f64::consts::PI;
    let s = LinearSpectrum::new(0.315, 0.049, 0.674, 0.965, 0.811, true);
    let (l, rf, cutoff) = (300., 14., 3);
    for m in 1..=cutoff {
        let k = 2. * PI * m as f64 / l;
        for dim in 1..=3 {
            let expected = 0.6_f64.powi(2) * s.power(k) * (-(k * rf).powi(2)).exp() / l.powi(3 - dim as i32);
            let got = s.filtered_power(k, dim, l, rf, 0.6, cutoff);
            assert!((got / expected - 1.).abs() < 1e-12, "dim {dim} m {m}");
        }
    }
    assert_eq!(s.filtered_power(2. * PI * 4. / l, 3, l, rf, 0.6, cutoff), 0.);
    assert_eq!(s.filtered_power(0., 3, l, rf, 0.6, cutoff), 0.);
}
