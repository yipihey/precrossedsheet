//! Physical smooth-CDM initial spectrum and resolution-stable Fourier draws.
//! Lengths are Mpc/h and wavenumbers h/Mpc. The EH98 non-oscillatory fit
//! includes scale-dependent baryon suppression, not BAO or massive neutrinos.
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
/// Spherical top-hat Fourier window, with a stable small-argument series.
pub fn top_hat(x: f64) -> f64 {
    if x.abs() < 0.01 {
        1. - x * x / 10. + x.powi(4) / 280. - x.powi(6) / 15120.
    } else {
        3. * (x.sin() - x * x.cos()) / x.powi(3)
    }
}
/// Integral sigma_R^2 for the unfiltered continuum 3D spectrum A k^ns T^2.
/// Simpson integration in ln k, 10^-5..100 h/Mpc, independent of the box/grid.
pub fn variance(omega_m: f64, omega_b: f64, h: f64, ns: f64, radius: f64, intervals: usize) -> f64 {
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
            * transfer(k, omega_m, omega_b, h).powi(2)
            * top_hat(k * radius).powi(2);
    }
    sum * step / (6. * PI * PI)
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
/// Fourier coefficients in a unit-coordinate grid. The full 3D z=0 spectrum
/// sets the coefficient variance P3(k)/L^3 in every dimension. In d<3 this
/// selects a translationally invariant subspace, not a random 3D slice.
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
) -> Vec<C64> {
    let amplitude = sigma8.powi(2) / variance(omega_m, omega_b, h, ns, 8., 4096);
    let mut out = vec![C64::new(0., 0.); grid.size];
    for i in 1..grid.size {
        let ijk = grid.unravel(i);
        let mut mode = [0_i64; 3];
        for a in 0..grid.dim {
            mode[a] = grid.ifreq(ijk[a]);
        }
        let m2 = mode.iter().map(|m| m * m).sum::<i64>();
        if m2 > (cutoff * cutoff) as i64 {
            continue;
        }
        let positive = *mode.iter().find(|&&m| m != 0).unwrap() > 0;
        let canonical = if positive { mode } else { mode.map(|m| -m) };
        let mut rng = Rng::new(mode_seed(seed, canonical));
        let k = 2. * PI * (m2 as f64).sqrt() / box_size;
        let p = amplitude
            * k.powf(ns)
            * transfer(k, omega_m, omega_b, h).powi(2)
            * (-(k * initial_smoothing).powi(2)).exp();
        let scale = grid.size as f64 * (p / (2. * box_size.powi(3))).sqrt() / growth_at_one;
        let z = C64::new(rng.normal(), rng.normal()) * scale;
        out[i] = if positive { z } else { z.conj() };
    }
    out
}
