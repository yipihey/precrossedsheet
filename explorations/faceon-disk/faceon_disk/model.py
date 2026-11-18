"""Minimal star-plus-dust disk.

Stars and dust are exponential atmospheres with separate scale heights and the
same radial pattern. The face-on surface brightness depends on the dust column
and the ratio of the scale heights. Scaling both heights together leaves the
face-on image unchanged and changes the galaxy once it is inclined.

Extinction is absorption only, with the Cardelli, Clayton, and Mathis (1989)
curve at R_V = 3.1. There is no scattering and no emission from the dust.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

# Effective wavelengths in nm, close to the SDSS response centroids in
# Fukugita et al. (1996). The curve is evaluated at these wavelengths.
BANDS = (
    ("g", 477.0),
    ("r", 623.1),
    ("i", 762.5),
)


def ccm_extinction(wavelength_nm: float, r_v: float = 3.1) -> float:
    """A_λ / A_V from Cardelli, Clayton, and Mathis (1989).

    The infrared piece covers 0.3 ≤ x ≤ 1.1 µm^-1, which includes DECam z.
    The optical piece covers 1.1 < x ≤ 3.3 µm^-1.
    """
    x = 1000.0 / wavelength_nm  # µm^-1
    if 0.3 <= x <= 1.1:
        power = x**1.61
        return float(0.574 * power - 0.527 * power / r_v)
    if not 1.1 < x <= 3.3:
        raise ValueError(f"wavelength {wavelength_nm} nm is outside the CCM pieces used here")
    y = x - 1.82
    a = (
        1.0
        + 0.17699 * y
        - 0.50447 * y**2
        - 0.02427 * y**3
        + 0.72085 * y**4
        + 0.01979 * y**5
        - 0.77530 * y**6
        + 0.32999 * y**7
    )
    b = (
        1.41338 * y
        + 2.28305 * y**2
        + 1.07233 * y**3
        - 5.38434 * y**4
        - 0.62251 * y**5
        + 5.30260 * y**6
        - 2.09002 * y**7
    )
    return float(a + b / r_v)


EXTINCTION = np.array([ccm_extinction(w) for _, w in BANDS], dtype=np.float64)


@dataclass(frozen=True)
class Disk:
    """Axisymmetric exponential disk plus a fixed logarithmic spiral.

    ``I0`` is the unattenuated face-on central surface brightness in the middle
    band. ``color_g`` and ``color_red`` are the intrinsic f_ν of the blue and
    red bands relative to that middle band, before dust. An unattenuated disk
    with both colors equal to one is white in f_ν. ``tau_V0`` is the total
    vertical V-band optical depth through the center.

    ``pattern_map`` replaces the analytic spiral for the stars when it is set.
    It is the stellar pattern on a face-on grid. The dust then stays the smooth
    exponential. With no map, the analytic spiral is shared by stars and dust.
    """

    h_R: float = 3.2
    I0: float = 1.0
    tau_V0: float = 2.4
    h_star: float = 0.45
    h_dust: float = 0.10
    spiral_amplitude: float = 0.38
    pitch_deg: float = 18.0
    arm_count: int = 2
    color_g: float = 1.0
    color_red: float = 1.0
    pattern_map: np.ndarray | None = None
    pattern_extent: float = 14.0

    def log_params(self) -> np.ndarray:
        return np.log(
            np.array(
                [self.h_R, self.I0, self.tau_V0, self.h_star, self.h_dust, self.color_g, self.color_red],
                dtype=np.float64,
            )
        )

    def with_log_params(self, theta: np.ndarray) -> "Disk":
        h_R, I0, tau, h_star, h_dust, color_g, color_red = np.exp(np.asarray(theta, dtype=np.float64))
        return replace(
            self,
            h_R=float(h_R),
            I0=float(I0),
            tau_V0=float(tau),
            h_star=float(h_star),
            h_dust=float(h_dust),
            color_g=float(color_g),
            color_red=float(color_red),
        )

    def with_pattern(self, pattern_map: np.ndarray, extent: float) -> "Disk":
        return replace(self, pattern_map=np.asarray(pattern_map, dtype=np.float64), pattern_extent=float(extent))


def pattern(R: np.ndarray, phi: np.ndarray, disk: Disk) -> np.ndarray:
    """Radial exponential times a logarithmic spiral of azimuthal mean one."""
    radial = np.exp(-R / disk.h_R)
    if disk.spiral_amplitude == 0.0:
        return radial
    pitch = np.deg2rad(disk.pitch_deg)
    fade = np.clip(R / (0.18 * disk.h_R), 0.0, 1.0)
    phase = disk.arm_count * (phi - np.log(np.maximum(R, 1e-6 * disk.h_R) / disk.h_R) / np.tan(pitch))
    return radial * (1.0 + disk.spiral_amplitude * fade * np.cos(phase))


def spatial_pattern(x: np.ndarray, y: np.ndarray, disk: Disk) -> np.ndarray:
    """Stellar face-on pattern. A map, when set, replaces the spiral."""
    if disk.pattern_map is None:
        return pattern(np.hypot(x, y), np.arctan2(y, x), disk)
    return _sample_pattern(x, y, disk.pattern_map, disk.pattern_extent)


def dust_pattern(x: np.ndarray, y: np.ndarray, disk: Disk) -> np.ndarray:
    """Dust pattern. A painted stellar map does not thicken the dust in the arms.

    The analytic spiral is shared by stars and dust. An image-derived map is the
    stellar light only; the dust stays the smooth exponential atmosphere.
    """
    if disk.pattern_map is None:
        return spatial_pattern(x, y, disk)
    return np.exp(-np.hypot(x, y) / disk.h_R)


def _sample_pattern(x: np.ndarray, y: np.ndarray, image: np.ndarray, extent: float) -> np.ndarray:
    """Bilinear sample of a face-on map.

    The map uses the same pixel convention as ``faceon_image`` at azimuth 0.
    Row 0 is sky Y = +extent. Galactic (x, y) equals (−X, −Y) in that frame,
    so a feature stored in the map rotates on the sky when the camera's azimuth changes.
    """
    ny, nx = image.shape
    sky_x = -np.asarray(x, dtype=np.float64)
    sky_y = -np.asarray(y, dtype=np.float64)
    fx = (sky_x + extent) / (2.0 * extent) * nx - 0.5
    fy = (extent - sky_y) / (2.0 * extent) * ny - 0.5
    fx = np.clip(fx, 0.0, nx - 1.001)
    fy = np.clip(fy, 0.0, ny - 1.001)
    x0 = np.floor(fx).astype(np.int64)
    y0 = np.floor(fy).astype(np.int64)
    wx = fx - x0
    wy = fy - y0
    x1 = np.minimum(x0 + 1, nx - 1)
    y1 = np.minimum(y0 + 1, ny - 1)
    top = (1.0 - wx) * image[y0, x0] + wx * image[y0, x1]
    bottom = (1.0 - wx) * image[y1, x0] + wx * image[y1, x1]
    return (1.0 - wy) * top + wy * bottom


def band_scales(disk: Disk) -> np.ndarray:
    """Intrinsic f_ν multipliers for the blue, middle, and red bands."""
    return np.array([disk.color_g, 1.0, disk.color_red], dtype=np.float64)


def column_factors(tau: np.ndarray, alpha: float, n_z: int = 2400, steep: float = 4.5) -> np.ndarray:
    """Face-on I / I_thin for total vertical optical depths ``tau``.

    Heights are in units of the dust scale height, so ``alpha = h_star / h_dust``.
    The observer sits at +z. ``tau`` is already the band optical depth.
    """
    tau = np.asarray(tau, dtype=np.float64)
    alpha = float(alpha)
    z_max = 14.0 * max(alpha, 1.0)
    u = np.linspace(-1.0, 1.0, n_z)
    z = z_max * np.sinh(steep * u) / np.sinh(steep)
    z = z[::-1]
    z_mid = 0.5 * (z[:-1] + z[1:])
    dz = z[:-1] - z[1:]
    w_star = np.exp(-np.abs(z_mid) / alpha) / (2.0 * alpha) * dz
    w_dust = np.exp(-np.abs(z_mid)) * 0.5 * dz
    transmission = np.ones_like(tau)
    accumulated = np.zeros_like(tau)
    for w_s, w_d in zip(w_star, w_dust):
        dtau = np.minimum(tau * w_d, 80.0)
        atten = np.exp(-dtau)
        cell = np.empty_like(dtau)
        small = dtau < 1e-4
        cell[small] = 1.0 - 0.5 * dtau[small] + dtau[small] ** 2 / 6.0
        cell[~small] = (1.0 - atten[~small]) / dtau[~small]
        accumulated += w_s * cell * transmission
        transmission *= atten
    return accumulated


class AttenuationGrid:
    """Bilinear table of face-on attenuation over scale-height ratio and optical depth."""

    def __init__(self, n_alpha: int = 80, n_tau: int = 128, n_z: int = 1600):
        self.ln_alpha = np.linspace(np.log(0.12), np.log(50.0), n_alpha)
        self.ln_tau = np.linspace(np.log(1e-4), np.log(25.0), n_tau)
        tau = np.exp(self.ln_tau)
        self.factor = np.empty((n_alpha, n_tau), dtype=np.float64)
        for i, ln_a in enumerate(self.ln_alpha):
            self.factor[i] = column_factors(tau, float(np.exp(ln_a)), n_z=n_z)

    def at(self, alpha: float, tau: np.ndarray) -> np.ndarray:
        tau = np.asarray(tau, dtype=np.float64)
        out = np.ones_like(tau)
        positive = tau > 1e-6
        if not np.any(positive):
            return out
        ln_a = np.log(alpha)
        ia = (ln_a - self.ln_alpha[0]) / (self.ln_alpha[1] - self.ln_alpha[0])
        ia = float(np.clip(ia, 0.0, len(self.ln_alpha) - 1.001))
        i0 = int(np.floor(ia))
        wa = ia - i0
        ln_t = np.log(np.clip(tau[positive], 1e-4, 25.0))
        it = (ln_t - self.ln_tau[0]) / (self.ln_tau[1] - self.ln_tau[0])
        it = np.clip(it, 0.0, len(self.ln_tau) - 1.001)
        j0 = np.floor(it).astype(np.int64)
        wt = it - j0
        f = self.factor
        value = (1.0 - wa) * ((1.0 - wt) * f[i0, j0] + wt * f[i0, j0 + 1]) + wa * (
            (1.0 - wt) * f[i0 + 1, j0] + wt * f[i0 + 1, j0 + 1]
        )
        out[positive] = value
        return out


def camera_basis(inclination_deg: float, azimuth_deg: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sky basis. ``n_obs`` points toward the observer.

    The line of nodes stays fixed while inclination runs through a full turn,
    so the picture rolls continuously through edge-on and through the far
    face-on view instead of flipping when the line of sight crosses the plane.
    """
    inc = np.deg2rad(inclination_deg)
    az = np.deg2rad(azimuth_deg)
    n_obs = np.array(
        [np.sin(inc) * np.sin(az), np.sin(inc) * np.cos(az), np.cos(inc)],
        dtype=np.float64,
    )
    e_x = np.array([-np.cos(az), np.sin(az), 0.0], dtype=np.float64)
    e_y = np.cross(n_obs, e_x)
    return e_x, e_y, n_obs


def sky_grid(npix_x: int, npix_y: int, extent_x: float, extent_y: float) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Pixel-center sky coordinates. Row 0 is the highest Y, so images display upright."""
    xs = (np.arange(npix_x) + 0.5) / npix_x * 2.0 * extent_x - extent_x
    ys = (np.arange(npix_y) + 0.5) / npix_y * 2.0 * extent_y - extent_y
    ys = ys[::-1]
    X, Y = np.meshgrid(xs, ys, indexing="xy")
    area = (2.0 * extent_x / npix_x) * (2.0 * extent_y / npix_y)
    return X, Y, area, area


def faceon_image(
    disk: Disk,
    npix: int = 96,
    extent: float = 14.0,
    grid: AttenuationGrid | None = None,
    azimuth_deg: float = 0.0,
    n_z: int = 2400,
    extinction: np.ndarray | None = None,
) -> np.ndarray:
    """Face-on surface brightness, shape (3, npix, npix), in the band order of ``extinction``."""
    extinction = EXTINCTION if extinction is None else np.asarray(extinction, dtype=np.float64)
    X, Y, _, _ = sky_grid(npix, npix, extent, extent)
    e_x, e_y, _ = camera_basis(0.0, azimuth_deg)
    x = X * e_x[0] + Y * e_y[0]
    y = X * e_x[1] + Y * e_y[1]
    stars = spatial_pattern(x, y, disk)
    dust = dust_pattern(x, y, disk)
    scales = band_scales(disk)
    tau_V = np.maximum(disk.tau_V0 * dust, 0.0)
    alpha = disk.h_star / disk.h_dust
    image = np.empty((3, npix, npix), dtype=np.float64)
    if grid is None:
        for b, k_ext in enumerate(extinction):
            factor = column_factors(k_ext * tau_V.ravel(), alpha, n_z=n_z).reshape(tau_V.shape)
            image[b] = disk.I0 * scales[b] * stars * factor
    else:
        for b, k_ext in enumerate(extinction):
            image[b] = disk.I0 * scales[b] * stars * grid.at(alpha, k_ext * tau_V)
    return image


def attenuation_of(disk: Disk, tau_V: float | np.ndarray, n_z: int = 2400) -> np.ndarray:
    """Band attenuation I/I_thin at one or more vertical V-band optical depths."""
    tau_V = np.asarray(tau_V, dtype=np.float64)
    alpha = disk.h_star / disk.h_dust
    out = np.empty(tau_V.shape + (3,), dtype=np.float64)
    flat = tau_V.ravel()
    for b, k_ext in enumerate(EXTINCTION):
        out.reshape(-1, 3)[:, b] = column_factors(k_ext * flat, alpha, n_z=n_z)
    return out


def render(
    disk: Disk,
    inclination_deg: float,
    azimuth_deg: float = 0.0,
    npix_x: int = 160,
    npix_y: int | None = None,
    extent_x: float = 14.0,
    extent_y: float | None = None,
    steps: int = 220,
    extinction: np.ndarray | None = None,
) -> np.ndarray:
    """Line-of-sight integral of stellar emissivity with foreground dust absorption.

    Returns surface brightness in g, r, i with shape (3, npix_y, npix_x).
    ``steps`` samples are placed only on the segment of each ray that crosses
    the disk.
    """
    if npix_y is None:
        npix_y = npix_x
    if extent_y is None:
        extent_y = extent_x
    extinction = EXTINCTION if extinction is None else np.asarray(extinction, dtype=np.float64)
    scales = band_scales(disk)
    X, Y, _, _ = sky_grid(npix_x, npix_y, extent_x, extent_y)
    e_x, e_y, n_obs = camera_basis(inclination_deg, azimuth_deg)
    base = X[..., None] * e_x + Y[..., None] * e_y
    s_near, s_far = _ray_segment(base, n_obs, disk, extent_x, extent_y)
    span = s_near - s_far
    image = np.zeros((3, npix_y, npix_x), dtype=np.float64)
    transmission = np.ones((3, npix_y, npix_x), dtype=np.float64)
    if not np.any(span > 0.0):
        return image
    for i in range(steps):
        s = s_near - (i + 0.5) / steps * span
        ds = span / steps
        pos = base + s[..., None] * n_obs
        x, y, z = pos[..., 0], pos[..., 1], pos[..., 2]
        stars = spatial_pattern(x, y, disk)
        dust = dust_pattern(x, y, disk)
        vertical = np.exp(-np.abs(z) / disk.h_star) / (2.0 * disk.h_star)
        emit = disk.I0 * stars * vertical * ds
        dtau_V = disk.tau_V0 * dust * np.exp(-np.abs(z) / disk.h_dust) / (2.0 * disk.h_dust) * ds
        dtau_V = np.clip(dtau_V, 0.0, 80.0)
        for b, k_ext in enumerate(extinction):
            dtau = np.minimum(k_ext * dtau_V, 80.0)
            atten = np.exp(-dtau)
            cell = np.empty_like(dtau)
            small = dtau < 1e-4
            cell[small] = 1.0 - 0.5 * dtau[small] + dtau[small] ** 2 / 6.0
            cell[~small] = (1.0 - atten[~small]) / dtau[~small]
            image[b] += scales[b] * emit * cell * transmission[b]
            transmission[b] *= atten
    return image


def _ray_segment(
    base: np.ndarray,
    n_obs: np.ndarray,
    disk: Disk,
    extent_x: float,
    extent_y: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Near and far distances along ``n_obs`` where the ray crosses the disk.

    The near value is the larger algebraic ``s`` because ``n_obs`` points toward
    the observer. Rays that miss the disk get a zero-length segment.
    """
    R_cut = 7.0 * disk.h_R
    if disk.pattern_map is not None:
        R_cut = max(R_cut, 0.98 * disk.pattern_extent)
    z_cut = 12.0 * max(disk.h_star, disk.h_dust, 0.05)
    bx, by, bz = base[..., 0], base[..., 1], base[..., 2]
    nx, ny, nz = (float(v) for v in n_obs)
    huge = R_cut + z_cut + max(extent_x, extent_y) + 1.0

    if abs(nz) < 1e-8:
        in_slab = np.abs(bz) <= z_cut
        s_z0 = np.where(in_slab, -huge, 1.0)
        s_z1 = np.where(in_slab, huge, -1.0)
    else:
        t0 = (-z_cut - bz) / nz
        t1 = (z_cut - bz) / nz
        s_z0 = np.minimum(t0, t1)
        s_z1 = np.maximum(t0, t1)

    a = nx * nx + ny * ny
    if a < 1e-12:
        inside = bx * bx + by * by <= R_cut * R_cut
        s_r0 = np.where(inside, -huge, 1.0)
        s_r1 = np.where(inside, huge, -1.0)
    else:
        b = 2.0 * (bx * nx + by * ny)
        c = bx * bx + by * by - R_cut * R_cut
        disc = b * b - 4.0 * a * c
        sqrt_disc = np.sqrt(np.maximum(disc, 0.0))
        s_r0 = (-b - sqrt_disc) / (2.0 * a)
        s_r1 = (-b + sqrt_disc) / (2.0 * a)
        miss = disc < 0.0
        s_r0 = np.where(miss, 1.0, s_r0)
        s_r1 = np.where(miss, -1.0, s_r1)

    s_far = np.maximum(s_z0, s_r0)
    s_near = np.minimum(s_z1, s_r1)
    s_far = np.clip(s_far, -huge, huge)
    s_near = np.clip(s_near, -huge, huge)
    empty = s_near <= s_far
    s_near = np.where(empty, 0.0, s_near)
    s_far = np.where(empty, 0.0, s_far)
    return s_near, s_far


def convolve_gaussian(image: np.ndarray, sigma: float) -> np.ndarray:
    """Separable Gaussian blur. ``image`` is (band, y, x) or (y, x)."""
    if sigma <= 0:
        return image
    radius = max(int(np.ceil(3.0 * sigma)), 1)
    offsets = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-0.5 * (offsets / sigma) ** 2)
    kernel /= kernel.sum()
    single = image.ndim == 2
    cube = image[None] if single else image
    blurred = _convolve_axis(cube, kernel, axis=2)
    blurred = _convolve_axis(blurred, kernel, axis=1)
    return blurred[0] if single else blurred


def _convolve_axis(image: np.ndarray, kernel: np.ndarray, axis: int) -> np.ndarray:
    radius = len(kernel) // 2
    pad = [(0, 0), (0, 0), (0, 0)]
    pad[axis] = (radius, radius)
    padded = np.pad(image, pad, mode="constant")
    out = np.zeros_like(image)
    for i, weight in enumerate(kernel):
        slc = [slice(None), slice(None), slice(None)]
        start = i
        stop = i + image.shape[axis]
        slc[axis] = slice(start, stop)
        out += weight * padded[tuple(slc)]
    return out


def tonemap(image: np.ndarray, scale: float = 0.7) -> np.ndarray:
    """Asinh RGB. ``scale`` is the surface brightness that sits near the top of the display.

    Input bands are g, r, i. Output is uint8 RGB with i in red, r in green, and g in blue.
    The knee is mild, so pixel noise is not amplified relative to the galaxy.
    """
    g = np.clip(image[0], 0.0, None)
    r = np.clip(image[1], 0.0, None)
    i = np.clip(image[2], 0.0, None)
    rgb = np.stack([i, r, g], axis=-1) / max(scale, 1e-6)
    rgb = np.arcsinh(2.2 * rgb) / np.arcsinh(2.2)
    return (np.clip(rgb, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)


def display_scale(clean: np.ndarray) -> float:
    """Scale that places the bright face-on pixels just below saturation."""
    intensity = np.clip(clean, 0.0, None).mean(axis=0)
    bright = float(np.quantile(intensity, 0.992))
    target = 0.90 * np.arcsinh(2.2)
    return float(2.2 * bright / np.sinh(target))
