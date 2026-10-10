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
                g, 2650, 3, 300., 0.315, 0.049, 0.674, 0.965, 0.811, 14., 0.7878,
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
            &a, 2651, 3, 300., 0.315, 0.049, 0.674, 0.965, 0.811, 14., 0.7878,
        );
        assert!(changed
            .iter()
            .zip(different)
            .any(|(x, y)| (*x - y).norm() > 0.));
        changed.clear();
    }
}
