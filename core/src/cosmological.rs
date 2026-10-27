//! Physical CDM+baryon initial spectra and resolution-stable Fourier draws.
//! Lengths are Mpc/h and wavenumbers h/Mpc. Two Eisenstein & Hu (1998) fits
//! are available: the non-oscillatory form (scale-dependent baryon suppression,
//! no acoustic oscillations) and the full form with baryon acoustic oscillations
//! and Silk damping. Neither includes massive neutrinos or radiation in the
//! late-time growth.
use crate::{grid::Grid, rng::Rng};
use num_complex::Complex64 as C64;
use std::f64::consts::PI;

/// Eisenstein & Hu (1998), equations 26 and 28--31; T_CMB=2.7255 K.
pub fn transfer(k: f64, omega_m: f64, omega_b: f64, h: f64) -> f64 {
    if k == 0. {
        return 1.;
    }
    let omhh = omega_m * h * h;
    let obhh = omega_b * h * h;
    let fb = omega_b / omega_m;
    let sound = 44.5 * (9.83 / omhh).ln() / (1. + 10. * obhh.powf(0.75)).sqrt();
    let alpha = 1. - 0.328 * (431. * omhh).ln() * fb + 0.38 * (22.3 * omhh).ln() * fb * fb;
    let gamma = omega_m * h * (alpha + (1. - alpha) / (1. + (0.43 * k * h * sound).powi(4)));
    let q = k * (2.7255_f64 / 2.7).powi(2) / gamma;
    let l = (2. * std::f64::consts::E + 1.8 * q).ln();
    l / (l + (14.2 + 731. / (1. + 62.5 * q)) * q * q)
}
/// Eisenstein & Hu (1998), equations 2--24: the full CDM+baryon transfer
/// function including baryon acoustic oscillations and Silk damping.
/// `k` is in h/Mpc; the fit's internal wavenumbers are in Mpc^-1.
pub fn transfer_bao(k: f64, omega_m: f64, omega_b: f64, h: f64) -> f64 {
    if k == 0. {
        return 1.;
    }
    let theta = 2.7255_f64 / 2.7;
    let omhh = omega_m * h * h;
    let obhh = omega_b * h * h;
    let fb = omega_b / omega_m;
    let fc = 1. - fb;
    let kk = k * h; // Mpc^-1
    // Equations 2--7: equality, drag epoch, sound horizon and Silk scale.
    let z_eq = 2.50e4 * omhh / theta.powi(4);
    let k_eq = 7.46e-2 * omhh / theta.powi(2);
    let b1 = 0.313 * omhh.powf(-0.419) * (1. + 0.607 * omhh.powf(0.674));
    let b2 = 0.238 * omhh.powf(0.223);
    let z_d = 1291. * omhh.powf(0.251) / (1. + 0.659 * omhh.powf(0.828)) * (1. + b1 * obhh.powf(b2));
    let r_of = |z: f64| 31.5 * obhh / theta.powi(4) / (z / 1e3);
    let r_d = r_of(z_d);
    let r_eq = r_of(z_eq);
    let s = 2. / (3. * k_eq)
        * (6. / r_eq).sqrt()
        * (((1. + r_d).sqrt() + (r_d + r_eq).sqrt()) / (1. + r_eq.sqrt())).ln();
    let k_silk = 1.6 * obhh.powf(0.52) * omhh.powf(0.73) * (1. + (10.4 * omhh).powf(-0.95));
    // Equations 10--12, 17--20: the cold-matter part.
    let q = kk / (13.41 * k_eq);
    let a1 = (46.9 * omhh).powf(0.670) * (1. + (32.1 * omhh).powf(-0.532));
    let a2 = (12.0 * omhh).powf(0.424) * (1. + (45.0 * omhh).powf(-0.582));
    let alpha_c = a1.powf(-fb) * a2.powf(-fb.powi(3));
    let bb1 = 0.944 / (1. + (458. * omhh).powf(-0.708));
    let bb2 = (0.395 * omhh).powf(-0.0266);
    let beta_c = 1. / (1. + bb1 * (fc.powf(bb2) - 1.));
    let t0 = |alpha: f64, beta: f64| {
        let c = 14.2 / alpha + 386. / (1. + 69.9 * q.powf(1.08));
        let l = (std::f64::consts::E + 1.8 * beta * q).ln();
        l / (l + c * q * q)
    };
    let f = 1. / (1. + (kk * s / 5.4).powi(4));
    let t_c = f * t0(1., beta_c) + (1. - f) * t0(alpha_c, beta_c);
    // Equations 13--15, 21--24: the baryon part with its oscillations.
    let y = (1. + z_eq) / (1. + z_d);
    let sq = (1. + y).sqrt();
    let g = y * (-6. * sq + (2. + 3. * y) * ((sq + 1.) / (sq - 1.)).ln());
    let alpha_b = 2.07 * k_eq * s * (1. + r_d).powf(-0.75) * g;
    let beta_b = 0.5 + fb + (3. - 2. * fb) * ((17.2 * omhh).powi(2) + 1.).sqrt();
    let beta_node = 8.41 * omhh.powf(0.435);
    let s_tilde = s / (1. + (beta_node / (kk * s)).powi(3)).cbrt();
    let x = kk * s_tilde;
    let j0 = if x.abs() < 1e-4 { 1. - x * x / 6. } else { x.sin() / x };
    let t_b = (t0(1., 1.) / (1. + (kk * s / 5.2).powi(2))
        + alpha_b / (1. + (beta_b / (kk * s)).powi(3)) * (-(kk / k_silk).powf(1.4)).exp())
        * j0;
    fb * t_b + fc * t_c
}
/// Select the EH98 fit: `bao=false` is the non-oscillatory form, `bao=true` the full form.
pub fn transfer_model(k: f64, omega_m: f64, omega_b: f64, h: f64, bao: bool) -> f64 {
    if bao {
        transfer_bao(k, omega_m, omega_b, h)
    } else {
        transfer(k, omega_m, omega_b, h)
    }
}
/// Spherical top-hat Fourier window, with a stable small-argument series.
pub fn top_hat(x: f64) -> f64 {
    if x.abs() < 0.01 {
        1. - x * x / 10. + x.powi(4) / 280. - x.powi(6) / 15120.
    } else {
        3. * (x.sin() - x * x.cos()) / x.powi(3)
    }
}
/// Dimension-appropriate top-hat Fourier window W_d(kR):
/// d=1: sin(x)/x; d=2: 2 J_1(x)/x; d=3: 3(sin x - x cos x)/x^3.
pub fn top_hat_dim(x: f64, dim: usize) -> f64 {
    let ax = x.abs();
    match dim {
        1 => {
            if ax < 0.01 {
                1. - x * x / 6. + x.powi(4) / 120. - x.powi(6) / 5040.
            } else {
                x.sin() / x
            }
        }
        2 => {
            if ax <= 8.0 {
                let h2 = 0.25 * x * x;
                let mut term = 1.0;
                let mut sum = 1.0;
                for m in 1..=24 {
                    term *= -h2 / (m as f64 * (m + 1) as f64);
                    sum += term;
                }
                sum
            } else {
                // Hankel asymptotic expansion for 2 J_1(x) / x on x > 8
                let inv8x = 1.0 / (8.0 * ax);
                let inv8x2 = inv8x * inv8x;
                // mu = 4*1^2 = 4; a_k = prod_{j=1}^k (4 - (2j-1)^2) / k!
                let a1 = 3.0;
                let a2 = 3.0 * (-5.0) / 2.0;
                let a3 = a2 * (-21.0) / 3.0;
                let a4 = a3 * (-45.0) / 4.0;
                let a5 = a4 * (-77.0) / 5.0;
                let a6 = a5 * (-117.0) / 6.0;
                let p = 1.0 - a2 * inv8x2 + a4 * inv8x2 * inv8x2 - a6 * inv8x2.powi(3);
                let q = a1 * inv8x - a3 * inv8x * inv8x2 + a5 * inv8x * inv8x2 * inv8x2;
                let chi = ax - 0.75 * PI;
                let j1 = (2.0 / (PI * ax)).sqrt() * (p * chi.cos() - q * chi.sin());
                2.0 * j1 / ax
            }
        }
        _ => top_hat(x),
    }
}
/// Integral sigma_R^2 for the unfiltered continuum 3D spectrum A k^ns T^2
/// with the non-oscillatory transfer. Simpson integration in ln k,
/// 10^-5..100 h/Mpc, independent of the box/grid.
pub fn variance(omega_m: f64, omega_b: f64, h: f64, ns: f64, radius: f64, intervals: usize) -> f64 {
    variance_model(omega_m, omega_b, h, ns, radius, intervals, false)
}
/// The same top-hat variance integral for either transfer fit in 3D.
pub fn variance_model(
    omega_m: f64,
    omega_b: f64,
    h: f64,
    ns: f64,
    radius: f64,
    intervals: usize,
    bao: bool,
) -> f64 {
    variance_model_dim(omega_m, omega_b, h, ns, radius, intervals, bao, 3)
}
/// Top-hat variance integral in `dim` dimensions (d = 1, 2, 3) for unit
/// amplitude P_d(k) = k^{3 - d + ns} T(k)^2, so that the dimensionless
/// power per ln k, Delta^2(k) ~ k^{3 + ns} T(k)^2, matches 3D and the
/// d-dimensional top-hat variance at R = 8 Mpc/h is normalized to sigma8^2.
pub fn variance_model_dim(
    omega_m: f64,
    omega_b: f64,
    h: f64,
    ns: f64,
    radius: f64,
    intervals: usize,
    bao: bool,
    dim: usize,
) -> f64 {
    assert!(intervals > 0 && intervals % 2 == 0);
    let lo = 1e-5_f64.ln();
    let hi = 100_f64.ln();
    let step = (hi - lo) / intervals as f64;
    let mut sum = 0.;
    for i in 0..=intervals {
        let k = (lo + i as f64 * step).exp();
        let weight = if i == 0 || i == intervals {
            1.
        } else if i % 2 == 0 {
            2.
        } else {
            4.
        };
        sum += weight
            * k.powf(3. + ns)
            * transfer_model(k, omega_m, omega_b, h, bao).powi(2)
            * top_hat_dim(k * radius, dim).powi(2);
    }
    let norm = match dim {
        1 => 3. * PI,
        2 => 6. * PI,
        _ => 6. * PI * PI,
    };
    sum * step / norm
}
/// The unfiltered z=0 linear spectrum in d dimensions:
/// P_d(k) = A_d k^{3 - d + ns} T(k)^2 in (Mpc/h)^d, normalized so that the
/// d-dimensional top-hat variance at R = 8 Mpc/h equals sigma8^2.
#[derive(Clone, Debug)]
pub struct LinearSpectrum {
    pub omega_m: f64,
    pub omega_b: f64,
    pub h: f64,
    pub ns: f64,
    pub bao: bool,
    /// 3D amplitude A_3 = sigma8^2 / sigma_{8,3D}^2[A=1].
    pub amplitude: f64,
    /// Dimension-specific amplitudes [A_1, A_2, A_3] matching sigma8^2 in d-D top-hats.
    pub amplitudes: [f64; 3],
}
/// Isotropic Fourier filter window W(k R) for either a Gaussian filter
/// exp(-(k R)^2 / 2) (`gaussian_window = true`) or a d-dimensional real-space
/// top-hat sphere/disk/interval (`gaussian_window = false`).
pub fn filter_window(dim: usize, x: f64, gaussian_window: bool) -> f64 {
    if gaussian_window {
        (-0.5 * x.powi(2)).exp()
    } else {
        top_hat_dim(x, dim.clamp(1, 3))
    }
}

impl LinearSpectrum {
    pub fn new(omega_m: f64, omega_b: f64, h: f64, ns: f64, sigma8: f64, bao: bool) -> Self {
        let s8_2 = sigma8.powi(2);
        let amplitudes = [
            s8_2 / variance_model_dim(omega_m, omega_b, h, ns, 8., 4096, bao, 1),
            s8_2 / variance_model_dim(omega_m, omega_b, h, ns, 8., 4096, bao, 2),
            s8_2 / variance_model_dim(omega_m, omega_b, h, ns, 8., 4096, bao, 3),
        ];
        Self { omega_m, omega_b, h, ns, bao, amplitude: amplitudes[2], amplitudes }
    }
    /// P_{3,0}(k) for k in h/Mpc.
    pub fn power(&self, k: f64) -> f64 {
        self.power_dim(k, 3)
    }
    /// P_{d,0}(k) = A_d k^{3 - d + ns} T(k)^2 in (Mpc/h)^d, with d-D top-hat sigma8 normalization.
    pub fn power_dim(&self, k: f64, dim: usize) -> f64 {
        if k <= 0. {
            return 0.;
        }
        let d = dim.clamp(1, 3);
        self.amplitudes[d - 1]
            * k.powf((3 - d) as f64 + self.ns)
            * transfer_model(k, self.omega_m, self.omega_b, self.h, self.bao).powi(2)
    }
    /// The laboratory's filtered, growth-scaled input spectrum in d dimensions:
    /// (D/D1)^2 P_{d,0}(k) exp(-k^2 R^2), in (Mpc/h)^d.
    pub fn filtered_power(&self, k: f64, dim: usize, _box_size: f64, initial_smoothing: f64, growth_ratio: f64, _cutoff: usize) -> f64 {
        if k <= 0. {
            return 0.;
        }
        growth_ratio.powi(2) * self.power_dim(k, dim) * (-(k * initial_smoothing).powi(2)).exp()
    }
    /// Filtered linear continuum variance sigma^2(R) at physical radius `radius` (Mpc/h)
    /// for either a d-dimensional real-space top-hat (`gaussian_window = false`) or
    /// a Gaussian filter exp(-k^2 R^2 / 2) (`gaussian_window = true`).
    pub fn filtered_variance(
        &self,
        dim: usize,
        initial_smoothing: f64,
        growth_ratio: f64,
        radius: f64,
        gaussian_window: bool,
    ) -> f64 {
        let intervals = 1024usize;
        let lo = 1e-5_f64.ln();
        let hi = 100_f64.ln();
        let step = (hi - lo) / intervals as f64;
        let d = dim.clamp(1, 3);
        let mut sum = 0.0;
        for i in 0..=intervals {
            let k = (lo + i as f64 * step).exp();
            let weight = if i == 0 || i == intervals {
                1.0
            } else if i % 2 == 0 {
                2.0
            } else {
                4.0
            };
            let w = if gaussian_window {
                (-0.5 * (k * radius).powi(2)).exp()
            } else {
                top_hat_dim(k * radius, d)
            };
            sum += weight
                * k.powf(3.0 + self.ns)
                * transfer_model(k, self.omega_m, self.omega_b, self.h, self.bao).powi(2)
                * (-(k * initial_smoothing).powi(2)).exp()
                * w.powi(2);
        }
        let norm = match d {
            1 => 3.0 * PI,
            2 => 6.0 * PI,
            _ => 6.0 * PI * PI,
        };
        growth_ratio.powi(2) * self.amplitudes[d - 1] * sum * step / norm
    }
}
fn mix(mut z: u64) -> u64 {
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
    z ^ (z >> 31)
}
/// Hash a canonical signed integer wavevector and seed; independent of grid size.
pub fn mode_seed(seed: u64, mode: [i64; 3]) -> u64 {
    let mut out = mix(seed ^ 0x5053435300000002);
    for m in mode {
        out = mix(out ^ mix((m as u64).wrapping_add(0x9e3779b97f4a7c15)));
    }
    out
}
/// Fourier coefficients in a unit-coordinate grid. The d-dimensional z=0
/// spectrum P_d(k) sets the coefficient variance P_d(k)/L^d so that the
/// d-dimensional top-hat variance at R = 8 Mpc/h equals sigma8^2.
/// `growth_at_one` converts z=0 normalization to the early-time D~a convention.
pub fn gaussian_hat(
    grid: &Grid,
    seed: u64,
    cutoff: usize,
    box_size: f64,
    omega_m: f64,
    omega_b: f64,
    h: f64,
    ns: f64,
    sigma8: f64,
    initial_smoothing: f64,
    growth_at_one: f64,
    bao: bool,
) -> Vec<C64> {
    let spectrum = LinearSpectrum::new(omega_m, omega_b, h, ns, sigma8, bao);
    let mut out = vec![C64::new(0., 0.); grid.size];
    let _ = cutoff;
    for i in 1..grid.size {
        let ijk = grid.unravel(i);
        if (0..grid.dim).any(|a| grid.n % 2 == 0 && ijk[a] == grid.n / 2) {
            continue;
        }
        let mut mode = [0_i64; 3];
        for a in 0..grid.dim {
            mode[a] = grid.ifreq(ijk[a]);
        }
        let m2 = mode.iter().map(|m| m * m).sum::<i64>();
        let positive = *mode.iter().find(|&&m| m != 0).unwrap() > 0;
        let canonical = if positive { mode } else { mode.map(|m| -m) };
        let mut rng = Rng::new(mode_seed(seed, canonical));
        let k = 2. * PI * (m2 as f64).sqrt() / box_size;
        let p = spectrum.power_dim(k, grid.dim) * (-(k * initial_smoothing).powi(2)).exp();
        let scale = grid.size as f64 * (p / (2. * box_size.powi(grid.dim as i32))).sqrt() / growth_at_one;
        let z = C64::new(rng.normal(), rng.normal()) * scale;
        out[i] = if positive { z } else { z.conj() };
    }
    out
}
