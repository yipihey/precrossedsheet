use num_complex::Complex64;

const WIDTHS: [f64; 4] = [0.1, 0.2, 0.3, 0.4];
const MASSES: [f64; 4] = [0.25; 4];
const ORDER_A: [usize; 4] = [0, 1, 2, 3];
const ORDER_B: [usize; 4] = [0, 2, 1, 3];

fn intervals(order: &[usize; 4]) -> [(f64, f64); 4] {
    let mut result = [(0.0, 0.0); 4];
    let mut left = 0.0;
    for &element in order {
        let right = left + WIDTHS[element];
        result[element] = (left, right);
        left = right;
    }
    assert!((left - 1.0).abs() < 1e-14);
    result
}

fn periodic_overlap(a: f64, b: f64, c: f64, d: f64) -> f64 {
    (-2..=2)
        .map(|period| {
            let shift = period as f64;
            (b.min(d + shift) - a.max(c + shift)).max(0.0)
        })
        .sum()
}

/// K_TU(s) = length(T intersect (U - s)) on the unit circle.
fn joint_volume(order: &[usize; 4], separation: f64) -> [[f64; 4]; 4] {
    let spans = intervals(order);
    let s = separation.rem_euclid(1.0);
    std::array::from_fn(|t| {
        std::array::from_fn(|u| {
            let (a, b) = spans[t];
            let (c, d) = spans[u];
            periodic_overlap(a, b, c - s, d - s)
        })
    })
}

fn correlation(order: &[usize; 4], separation: f64) -> f64 {
    let joint = joint_volume(order, separation);
    let mut mean_product = 0.0;
    for t in 0..4 {
        for u in 0..4 {
            mean_product += (MASSES[t] / WIDTHS[t]) * (MASSES[u] / WIDTHS[u]) * joint[t][u];
        }
    }
    mean_product
}

/// Independent spatial midpoint quadrature: sample the piecewise-constant
/// density directly instead of constructing pairwise interval overlaps.
fn density_at(order: &[usize; 4], x: f64) -> f64 {
    let spans = intervals(order);
    let x = x.rem_euclid(1.0);
    let element = spans
        .iter()
        .position(|&(left, right)| x >= left && x < right)
        .unwrap_or(3);
    MASSES[element] / WIDTHS[element]
}

fn spatial_quadrature(order: &[usize; 4], separation: f64, samples: usize) -> f64 {
    (0..samples)
        .map(|i| {
            let x = (i as f64 + 0.5) / samples as f64;
            density_at(order, x) * density_at(order, x + separation)
        })
        .sum::<f64>()
        / samples as f64
}

/// Fourier autocorrelation from the exact Fourier transform of each interval
/// window. Keeping the pair indices explicit lets the test detect accidentally
/// dropping same-element (T=U) contributions.
fn interval_fourier_correlation(
    order: &[usize; 4],
    separation: f64,
    max_mode: usize,
    include_same_element: bool,
) -> f64 {
    let spans = intervals(order);
    let mut total = 0.0;
    for t in 0..4 {
        for u in 0..4 {
            if include_same_element || t != u {
                // k=0 interval coefficients are the interval widths, so their
                // weighted contribution is exactly m_T m_U.
                total += MASSES[t] * MASSES[u];
            }
        }
    }
    let imaginary = Complex64::new(0.0, 1.0);
    for mode in 1..=max_mode {
        let k = mode as f64;
        let coefficients: [Complex64; 4] = std::array::from_fn(|t| {
            let (left, right) = spans[t];
            let lower = Complex64::from_polar(1.0, -std::f64::consts::TAU * k * left);
            let upper = Complex64::from_polar(1.0, -std::f64::consts::TAU * k * right);
            (MASSES[t] / WIDTHS[t]) * (lower - upper) / (imaginary * (std::f64::consts::TAU * k))
        });
        let mut pair_sum = Complex64::new(0.0, 0.0);
        for t in 0..4 {
            for u in 0..4 {
                if include_same_element || t != u {
                    pair_sum += coefficients[t] * coefficients[u].conj();
                }
            }
        }
        let phase = Complex64::from_polar(1.0, -std::f64::consts::TAU * k * separation);
        total += 2.0 * (pair_sum * phase).re;
    }
    total
}

#[test]
fn periodic_joint_volume_marginals_and_mass_pair_weights_are_distinct() {
    let separation = 0.25;
    let expected_xi_a = -79.0 / 2304.0;
    let expected_xi_b = -169.0 / 2304.0;
    let mut xis = Vec::new();

    for order in [&ORDER_A, &ORDER_B] {
        let joint = joint_volume(order, separation);
        let total: f64 = joint.iter().flatten().sum();
        assert!((total - 1.0).abs() < 1e-13);

        for t in 0..4 {
            let row = joint[t].iter().sum::<f64>();
            let column = (0..4).map(|u| joint[u][t]).sum::<f64>();
            assert!((row - WIDTHS[t]).abs() < 1e-13, "row {t}: {row}");
            assert!((column - WIDTHS[t]).abs() < 1e-13, "column {t}: {column}");
        }

        let mean_product = correlation(order, separation);
        let xi = mean_product - 1.0;
        xis.push(xi);
        let expected = if *order == ORDER_A {
            expected_xi_a
        } else {
            expected_xi_b
        };
        assert!((xi - expected).abs() < 2e-13, "order {order:?}: {xi}");

        let mass_pair: [[f64; 4]; 4] = std::array::from_fn(|t| {
            std::array::from_fn(|u| {
                (MASSES[t] / WIDTHS[t]) * (MASSES[u] / WIDTHS[u]) * joint[t][u] / mean_product
            })
        });
        assert!((mass_pair.iter().flatten().sum::<f64>() - 1.0).abs() < 1e-13);
        let row_zero = mass_pair[0].iter().sum::<f64>();
        assert!((row_zero - 0.25).abs() > 0.01);
        if *order == ORDER_A {
            assert!((row_zero - 24.0 / 89.0).abs() < 1e-13);
        }
        // The one-point mass marginal remains m_T=1/4 in both arrangements.
        assert!(MASSES.iter().all(|mass| (*mass - 0.25).abs() < 1e-15));

        let quadrature = spatial_quadrature(order, separation, 100_003);
        assert!(
            (quadrature - mean_product).abs() < 3e-5,
            "spatial quadrature {quadrature} != overlap {mean_product}"
        );
    }
    assert!((xis[0] - xis[1]).abs() > 0.03);
}

#[test]
fn zero_lag_and_separation_average_obey_periodic_density_identities() {
    for order in [&ORDER_A, &ORDER_B] {
        // At zero lag only identical intervals overlap. This retains all four
        // self-pairs and gives xi(0)=sum_T m_T^2/w_T - 1 = 29/96.
        let zero_lag_xi = correlation(order, 0.0) - 1.0;
        assert!((zero_lag_xi - 29.0 / 96.0).abs() < 1e-13);

        // Integrate the exact overlap correlation over all periodic shifts.
        // Midpoint sampling is independent of the spatial quadrature above.
        let samples = 4096;
        let average = (0..samples)
            .map(|i| correlation(order, (i as f64 + 0.5) / samples as f64))
            .sum::<f64>()
            / samples as f64;
        assert!(
            (average - 1.0).abs() < 2e-7,
            "average correlation {average}"
        );
    }
}

#[test]
fn interval_window_fourier_autocorrelation_matches_overlap_with_self_pairs() {
    for order in [&ORDER_A, &ORDER_B] {
        let separation = 0.25;
        let exact = correlation(order, separation);
        let fourier = interval_fourier_correlation(order, separation, 2048, true);
        assert!(
            (fourier - exact).abs() < 1e-7,
            "Fourier {fourier}, overlap {exact}"
        );

        let without_self_pairs = interval_fourier_correlation(order, separation, 2048, false);
        assert!(
            (fourier - without_self_pairs).abs() > 0.05,
            "same-element contribution was not material"
        );
    }
}
