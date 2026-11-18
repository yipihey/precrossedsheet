"""Isotropic 3D Gaussians whose depth integral matches a face-on image.

Each cloud is spherical: it has no preferred axis. A pixel is the far-to-near
composite of those clouds, luminosity and dust together. On the face-on view
every center sits in one layer, and the luminosities are solved so that layer
reproduces the image. Moving a cloud along galactic z, or looking from the
side, changes which dust lies in front of which stars. It does not stretch a
knot into the exponential wing of a thick disk.

A bright compact knot is a star cluster. Its light is one small cloud, not a
pile of overlapping ones. The wider clouds carry the disk underneath.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from .model import camera_basis, convolve_gaussian

# Kernel FFTs depend only on radius and the pixel grid, so frames of one movie share them.
_KERNEL_CACHE: dict[tuple[float, int, int, float, float], tuple[np.ndarray, int, int]] = {}


@dataclass(frozen=True)
class GaussianCloud:
    """Spherical star and dust clouds. Luminosity and τ are face-on column integrals."""

    star_mu: np.ndarray
    star_sigma: np.ndarray
    star_luminosity: np.ndarray
    dust_mu: np.ndarray
    dust_sigma: np.ndarray
    dust_tau: np.ndarray

    def with_z(self, star_z: np.ndarray, dust_z: np.ndarray) -> "GaussianCloud":
        stars = np.array(self.star_mu, copy=True)
        dust = np.array(self.dust_mu, copy=True)
        stars[:, 2] = star_z
        dust[:, 2] = dust_z
        return replace(self, star_mu=stars, dust_mu=dust)

    @property
    def z_rms(self) -> float:
        parts = []
        if len(self.star_mu):
            parts.append(self.star_mu[:, 2])
        if len(self.dust_mu):
            parts.append(self.dust_mu[:, 2])
        if not parts:
            return 0.0
        return float(np.sqrt(np.mean(np.concatenate(parts) ** 2)))


# Unresolved clusters share one of these radii so the renderer keeps a handful of kernels.
CLUSTER_SIGMAS = (0.05, 0.09, 0.13)


def fit_cloud(
    image: np.ndarray,
    extent: float,
    tau_v: np.ndarray,
    extinction: np.ndarray,
    star_sigmas: tuple[float, ...] = (0.14, 0.32, 0.75, 1.8),
    dust_sigmas: tuple[float, ...] = (0.90, 2.0, 4.0),
    cluster_excess: float = 0.42,
) -> GaussianCloud:
    """Non-negative spherical clouds whose mixed face-on layer matches ``image``.

    ``image`` is observed surface brightness in three bands, already the
    emergent light. ``tau_v`` is the face-on V-band optical-depth map. Star
    luminosities are the intrinsic columns that, mixed with that dust, return
    the image. Dust is its own set of spherical clouds, not a slab.

    Each bright compact knot is given one cloud from ``CLUSTER_SIGMAS``. The
    grid that follows is kept off that knot, so the cluster's light stays in
    that one small cloud and the wider clouds fit the disk around it.
    """
    extinction = np.asarray(extinction, dtype=np.float64)
    observed = np.clip(image, 0.0, None)
    tau = np.clip(tau_v, 0.0, None)
    dust_mu, dust_sigma, dust_tau = _fit_map(
        tau,
        extent,
        dust_sigmas,
        min_frac=(0.03, 0.015, 0.008),
        spacing_factor=1.35,
        iterations=6,
    )
    intrinsic = _intrinsic_columns(observed, tau, extinction)
    cluster_mu, cluster_sigma, cluster_lum = _place_clusters(observed, intrinsic, extent, cluster_excess)
    residual = intrinsic
    if len(cluster_sigma):
        residual = np.array(intrinsic, copy=True)
        ny, nx = intrinsic.shape[1:]
        for band in range(3):
            placed = _splat(
                -cluster_mu[:, 0],
                -cluster_mu[:, 1],
                cluster_sigma,
                cluster_lum[:, band],
                nx,
                ny,
                extent,
                extent,
                _KERNEL_CACHE,
            )
            residual[band] = np.clip(intrinsic[band] - placed, 0.0, None)
    star_mu, star_sigma, star_r = _fit_map(
        residual[1],
        extent,
        star_sigmas,
        min_frac=(0.10, 0.05, 0.025, 0.012),
        spacing_factor=1.25,
        iterations=8,
        avoid=None if len(cluster_mu) == 0 else cluster_mu[:, :2],
        avoid_radius=0.22,
        avoid_below=0.50,
    )
    if len(star_sigma) == 0 and len(cluster_sigma) == 0:
        raise RuntimeError("no star clouds were placed on the image")
    if len(star_sigma):
        color_g = _sample(residual[0], extent, star_mu[:, 0], star_mu[:, 1])
        color_z = _sample(residual[2], extent, star_mu[:, 0], star_mu[:, 1])
        base_r = _sample(residual[1], extent, star_mu[:, 0], star_mu[:, 1])
        color_g = np.clip(color_g / np.maximum(base_r, 1e-4), 0.2, 3.5)
        color_z = np.clip(color_z / np.maximum(base_r, 1e-4), 0.25, 4.0)
        grid_lum = star_r[:, None] * np.stack([color_g, np.ones_like(star_r), color_z], axis=1)
    else:
        grid_lum = np.zeros((0, 3))
    cloud = GaussianCloud(
        star_mu=np.concatenate([cluster_mu, star_mu], axis=0),
        star_sigma=np.concatenate([cluster_sigma, star_sigma]),
        star_luminosity=np.concatenate([cluster_lum, grid_lum], axis=0),
        dust_mu=dust_mu,
        dust_sigma=dust_sigma,
        dust_tau=dust_tau,
    )
    return _match_emergent(cloud, observed, extent, extinction, rounds=3)


def render_cloud(
    cloud: GaussianCloud,
    inclination_deg: float,
    azimuth_deg: float,
    npix_x: int,
    extent_x: float,
    npix_y: int | None = None,
    extent_y: float | None = None,
    extinction: np.ndarray | None = None,
    n_slices: int = 28,
) -> np.ndarray:
    """Far-to-near composite. Each cloud is one depth sample, spherical on the sky."""
    if npix_y is None:
        npix_y = npix_x
    if extent_y is None:
        extent_y = extent_x
    extinction = np.ones(3) if extinction is None else np.asarray(extinction, dtype=np.float64)
    sx, sy, ss = _project(cloud.star_mu, inclination_deg, azimuth_deg)
    dx, dy, ds = _project(cloud.dust_mu, inclination_deg, azimuth_deg)
    depth = np.concatenate([ss, ds]) if len(ds) else ss
    if len(depth) == 0:
        return np.zeros((3, npix_y, npix_x))
    if float(np.ptp(depth)) < 1e-4:
        edges = np.array([float(depth.min()) - 1.0, float(depth.max()) + 1.0])
    else:
        edges = np.quantile(depth, np.linspace(0.0, 1.0, n_slices + 1))
        edges = np.array(edges, copy=True)
        edges[0] -= 1e-4
        for i in range(1, len(edges)):
            if edges[i] <= edges[i - 1]:
                edges[i] = edges[i - 1] + 1e-6
        edges[-1] += 1e-4
    image = np.zeros((3, npix_y, npix_x), dtype=np.float64)
    kernels = _KERNEL_CACHE
    for lo, hi in zip(edges[:-1], edges[1:]):
        star = (ss >= lo) & (ss < hi) if len(ss) else np.zeros(0, dtype=bool)
        dust = (ds >= lo) & (ds < hi) if len(ds) else np.zeros(0, dtype=bool)
        if not np.any(star) and not np.any(dust):
            continue
        luminosity = np.zeros((3, npix_y, npix_x), dtype=np.float64)
        if np.any(star):
            for band in range(3):
                luminosity[band] = _splat(
                    sx[star],
                    sy[star],
                    cloud.star_sigma[star],
                    cloud.star_luminosity[star, band],
                    npix_x,
                    npix_y,
                    extent_x,
                    extent_y,
                    kernels,
                )
        tau_v = (
            _splat(
                dx[dust],
                dy[dust],
                cloud.dust_sigma[dust],
                cloud.dust_tau[dust],
                npix_x,
                npix_y,
                extent_x,
                extent_y,
                kernels,
            )
            if np.any(dust)
            else np.zeros((npix_y, npix_x), dtype=np.float64)
        )
        # Nearer material multiplies light already accumulated from behind.
        emitted = np.empty_like(luminosity)
        absorbed = np.empty_like(luminosity)
        for band, k_ext in enumerate(extinction):
            dtau = np.clip(k_ext * tau_v, 0.0, 80.0)
            cell = np.empty_like(dtau)
            small = dtau < 1e-4
            cell[small] = 1.0 - 0.5 * dtau[small] + dtau[small] ** 2 / 6.0
            cell[~small] = (1.0 - np.exp(-dtau[~small])) / dtau[~small]
            emitted[band] = luminosity[band] * cell
            absorbed[band] = np.exp(-dtau)
        image = image * absorbed + emitted
    return image


def depth_route(cloud: GaussianCloud, n_draws: int = 4, amplitude: float = 1.2, seed: int = 5) -> list[tuple[np.ndarray, np.ndarray]]:
    """Closed path of z offsets. Zero at both ends, so face-on frames agree.

    The offset scales with each cloud's own radius, so a knot stays near the
    plane and a large cloud can move by about its own size.
    """
    rng = np.random.default_rng(seed)
    zero_s = np.zeros(len(cloud.star_sigma))
    zero_d = np.zeros(len(cloud.dust_sigma))
    route = [(zero_s, zero_d)]
    for _ in range(n_draws):
        route.append(
            (
                amplitude * cloud.star_sigma * rng.normal(size=len(cloud.star_sigma)),
                amplitude * cloud.dust_sigma * rng.normal(size=len(cloud.dust_sigma)),
            )
        )
    route.append((zero_s, zero_d))
    return route


def blend_depth(route: list[tuple[np.ndarray, np.ndarray]], t: float) -> tuple[np.ndarray, np.ndarray]:
    """Smoothstep through depth draws. t is in [0, 1]."""
    if len(route) == 1:
        return route[0]
    x = float(np.clip(t, 0.0, 1.0)) * (len(route) - 1)
    i = min(int(np.floor(x)), len(route) - 2)
    u = x - i
    u = u * u * (3.0 - 2.0 * u)
    star = (1.0 - u) * route[i][0] + u * route[i + 1][0]
    dust = (1.0 - u) * route[i][1] + u * route[i + 1][1]
    return star, dust


def _match_emergent(
    cloud: GaussianCloud,
    observed: np.ndarray,
    extent: float,
    extinction: np.ndarray,
    rounds: int,
) -> GaussianCloud:
    luminosity = np.array(cloud.star_luminosity, copy=True)
    ny, nx = observed.shape[1:]
    for _ in range(rounds):
        rendered = render_cloud(
            replace(cloud, star_luminosity=luminosity),
            0.0,
            0.0,
            nx,
            extent,
            extinction=extinction,
            n_slices=1,
        )
        for band in range(3):
            ratio = _sample_ratio(observed[band], rendered[band], extent, cloud.star_mu[:, 0], cloud.star_mu[:, 1])
            luminosity[:, band] *= ratio
    return replace(cloud, star_luminosity=np.clip(luminosity, 0.0, None))


def _intrinsic_columns(observed: np.ndarray, tau_v: np.ndarray, extinction: np.ndarray) -> np.ndarray:
    """Invert a mixed slab so face-on radiative transfer returns ``observed``."""
    intrinsic = np.empty_like(observed)
    for band, k_ext in enumerate(extinction):
        kt = k_ext * tau_v
        factor = np.ones_like(kt)
        thick = kt > 1e-4
        factor[thick] = kt[thick] / (1.0 - np.exp(-kt[thick]))
        intrinsic[band] = observed[band] * factor
    return intrinsic


def _place_clusters(
    observed: np.ndarray,
    intrinsic: np.ndarray,
    extent: float,
    min_excess: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """One small cloud per bright compact knot, carrying that knot's excess light.

    The knot is detected on the observed r image. Its luminosity is the excess
    above the local disk in the intrinsic image, so dust applied later does not
    count the same light twice. A knot that is still bright a few pixels away
    is part of an arm or the bulge and is left to the wider clouds.
    """
    empty = (np.zeros((0, 3)), np.zeros(0), np.zeros((0, 3)))
    r_band = observed[1]
    ny, nx = r_band.shape
    if min_excess <= 0.0 or ny < 12 or nx < 12:
        return empty
    pix = 2.0 * extent / nx
    background = convolve_gaussian(r_band, max(0.35 / pix, 2.5))
    excess = r_band - background
    radius = 2
    neighbor = np.full_like(excess, -np.inf)
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            if dx * dx + dy * dy > radius * radius or (dx == 0 and dy == 0):
                continue
            neighbor = np.maximum(neighbor, np.roll(np.roll(excess, dy, axis=0), dx, axis=1))
    peak = excess > neighbor
    peak[:5, :] = False
    peak[-5:, :] = False
    peak[:, :5] = False
    peak[:, -5:] = False
    ys, xs = np.nonzero(peak & (excess > min_excess) & (r_band > 0.55))
    if len(ys) == 0:
        return empty
    mus = []
    sigmas = []
    columns = []
    # Same blur as the detection background, applied to each band.
    intrinsic_excess = intrinsic - convolve_gaussian(intrinsic, max(0.35 / pix, 2.5))
    for y, x in zip(ys.tolist(), xs.tolist()):
        window = excess[y - 4 : y + 5, x - 4 : x + 5]
        yy, xx = np.mgrid[-4:5, -4:5]
        rr = np.hypot(yy, xx)
        wing = window[(rr >= 3.0) & (rr <= 4.2)]
        if window[4, 4] <= 0.0 or float(wing.mean()) > 0.18 * float(window[4, 4]):
            continue
        aperture = rr <= 1.2
        weights = np.clip(window, 0.0, None)
        weights[~aperture] = 0.0
        if float(weights.sum()) <= 0.0:
            continue
        moment = float((weights * rr**2).sum() / weights.sum())
        sigma = float(np.clip(np.sqrt(moment) * pix, CLUSTER_SIGMAS[0], CLUSTER_SIGMAS[-1]))
        sigma = CLUSTER_SIGMAS[int(np.argmin(np.abs(np.asarray(CLUSTER_SIGMAS) - sigma)))]
        sky_x = -extent + (x + 0.5) / nx * 2.0 * extent
        sky_y = extent - (y + 0.5) / ny * 2.0 * extent
        mus.append((-sky_x, -sky_y, 0.0))
        sigmas.append(sigma)
        band_columns = []
        for band in range(3):
            patch = intrinsic_excess[band, y - 4 : y + 5, x - 4 : x + 5]
            band_columns.append(float(np.clip(patch, 0.0, None)[aperture].sum() * pix * pix))
        columns.append(band_columns)
    if not mus:
        return empty
    return (
        np.asarray(mus, dtype=np.float64),
        np.asarray(sigmas, dtype=np.float64),
        np.asarray(columns, dtype=np.float64),
    )


def _fit_map(
    target: np.ndarray,
    extent: float,
    sigmas: tuple[float, ...],
    min_frac: tuple[float, ...],
    spacing_factor: float,
    iterations: int,
    avoid: np.ndarray | None = None,
    avoid_radius: float = 0.0,
    avoid_below: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    empty = (np.zeros((0, 3)), np.zeros(0), np.zeros(0))
    peak = float(np.quantile(target, 0.99)) if target.size else 0.0
    if not np.isfinite(peak) or peak <= 0.0:
        return empty
    mus = []
    sig = []
    for sigma, frac in zip(sigmas, min_frac):
        spacing = spacing_factor * sigma
        coords = np.arange(-extent + 0.5 * spacing, extent, spacing)
        gx, gy = np.meshgrid(coords, coords, indexing="xy")
        gx = gx.ravel()
        gy = gy.ravel()
        values = _sample(target, extent, gx, gy)
        keep = (values > frac * peak) & (np.hypot(gx, gy) < extent * 0.9)
        if avoid is not None and len(avoid) and sigma <= avoid_below:
            distance = np.hypot(gx[:, None] - avoid[:, 0], gy[:, None] - avoid[:, 1])
            keep &= distance.min(axis=1) > avoid_radius
        if not np.any(keep):
            continue
        mu = np.stack([gx[keep], gy[keep], np.zeros(np.count_nonzero(keep))], axis=1)
        mus.append(mu)
        sig.append(np.full(len(mu), sigma))
    if not mus:
        return empty
    mu = np.concatenate(mus, axis=0)
    sigma = np.concatenate(sig)
    column = np.maximum(_sample(target, extent, mu[:, 0], mu[:, 1]), 0.0)
    # Area owned by each cloud, so the first guess already has the right integral.
    owned = (spacing_factor * sigma) ** 2
    column = column * owned
    ny, nx = target.shape
    sky_x = -mu[:, 0]
    sky_y = -mu[:, 1]
    for _ in range(iterations):
        # Face-on sky is the galactic plane rotated 180°, matching camera_basis at azimuth 0.
        pred = _splat(sky_x, sky_y, sigma, column, nx, ny, extent, extent, _KERNEL_CACHE)
        column = np.clip(column * _footprint_ratio(target, pred, sigma, extent, mu[:, 0], mu[:, 1]), 0.0, None)
    return mu, sigma, column


def _footprint_ratio(
    target: np.ndarray,
    pred: np.ndarray,
    sigma: np.ndarray,
    extent: float,
    gx: np.ndarray,
    gy: np.ndarray,
) -> np.ndarray:
    """Richardson–Lucy factor: each cloud sees the target through its own footprint."""
    ratio = np.ones(len(sigma))
    floor = 0.02 * float(np.quantile(target, 0.99))
    ny, nx = target.shape
    for value in np.unique(np.round(sigma, 5)):
        group = np.abs(sigma - value) < 1e-4
        if not np.any(group):
            continue
        gathered_target = _convolve_map(target, float(value), nx, ny, extent, extent)
        gathered_pred = _convolve_map(pred, float(value), nx, ny, extent, extent)
        safe = np.maximum(gathered_pred, floor)
        factor_map = gathered_target / safe
        factor_map = np.where(gathered_target < floor, np.minimum(factor_map, 1.0), factor_map)
        ratio[group] = np.clip(_sample(factor_map, extent, gx[group], gy[group]), 0.05, 8.0)
    return ratio


def _convolve_map(
    image: np.ndarray,
    sigma: float,
    npix_x: int,
    npix_y: int,
    extent_x: float,
    extent_y: float,
) -> np.ndarray:
    """Convolve a surface-brightness image with one cloud footprint."""
    key = (float(np.round(sigma, 5)), npix_x, npix_y, float(extent_x), float(extent_y))
    cached = _KERNEL_CACHE.get(key)
    if cached is None:
        cached = _padded_kernel(sigma, npix_x, npix_y, extent_x, extent_y)
        _KERNEL_CACHE[key] = cached
    kernel, pad_y, pad_x = cached
    padded = np.zeros((pad_y, pad_x), dtype=np.float64)
    padded[:npix_y, :npix_x] = image
    convolved = np.fft.irfft2(np.fft.rfft2(padded) * kernel, s=(pad_y, pad_x))
    return np.maximum(convolved[:npix_y, :npix_x], 0.0)


def _sample_ratio(
    target: np.ndarray,
    pred: np.ndarray,
    extent: float,
    gx: np.ndarray,
    gy: np.ndarray,
) -> np.ndarray:
    floor = 0.02 * float(np.quantile(target, 0.99))
    safe = np.maximum(pred, floor)
    ratio_map = target / safe
    ratio_map = np.where(target < floor, np.minimum(ratio_map, 1.0), ratio_map)
    return np.clip(_sample(ratio_map, extent, gx, gy), 0.05, 8.0)


def _project(mu: np.ndarray, inclination_deg: float, azimuth_deg: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if len(mu) == 0:
        empty = np.zeros(0)
        return empty, empty, empty
    e_x, e_y, n_obs = camera_basis(inclination_deg, azimuth_deg)
    return mu @ e_x, mu @ e_y, mu @ n_obs


def _splat(
    sky_x: np.ndarray,
    sky_y: np.ndarray,
    sigma: np.ndarray,
    column: np.ndarray,
    npix_x: int,
    npix_y: int,
    extent_x: float,
    extent_y: float,
    kernels: dict[tuple[float, int, int, float, float], tuple[np.ndarray, int, int]],
) -> np.ndarray:
    """Surface brightness, the sum of spherical clouds integrated through their depth."""
    image = np.zeros((npix_y, npix_x), dtype=np.float64)
    if len(column) == 0:
        return image
    # Group the few discrete radii and convolve each impulse image once.
    for value in np.unique(np.round(sigma, 5)):
        group = np.abs(sigma - value) < 1e-4
        if not np.any(group):
            continue
        impulses = _paint(sky_x[group], sky_y[group], column[group], npix_x, npix_y, extent_x, extent_y)
        key = (float(value), npix_x, npix_y, float(extent_x), float(extent_y))
        cached = kernels.get(key)
        if cached is None:
            cached = _padded_kernel(float(value), npix_x, npix_y, extent_x, extent_y)
            kernels[key] = cached
        kernel, pad_y, pad_x = cached
        padded = np.zeros((pad_y, pad_x), dtype=np.float64)
        padded[:npix_y, :npix_x] = impulses
        convolved = np.fft.irfft2(np.fft.rfft2(padded) * kernel, s=(pad_y, pad_x))
        image += convolved[:npix_y, :npix_x]
    np.maximum(image, 0.0, out=image)
    return image


def _paint(
    sky_x: np.ndarray,
    sky_y: np.ndarray,
    column: np.ndarray,
    npix_x: int,
    npix_y: int,
    extent_x: float,
    extent_y: float,
) -> np.ndarray:
    image = np.zeros((npix_y, npix_x), dtype=np.float64)
    fx = (sky_x + extent_x) / (2.0 * extent_x) * npix_x - 0.5
    fy = (extent_y - sky_y) / (2.0 * extent_y) * npix_y - 0.5
    x0 = np.floor(fx).astype(np.int64)
    y0 = np.floor(fy).astype(np.int64)
    wx = fx - x0
    wy = fy - y0
    for oy, wy_use in ((0, 1.0 - wy), (1, wy)):
        for ox, wx_use in ((0, 1.0 - wx), (1, wx)):
            weight = column * wy_use * wx_use
            xx = x0 + ox
            yy = y0 + oy
            keep = (xx >= 0) & (xx < npix_x) & (yy >= 0) & (yy < npix_y) & (weight > 0)
            np.add.at(image, (yy[keep], xx[keep]), weight[keep])
    return image


def _padded_kernel(
    sigma: float,
    npix_x: int,
    npix_y: int,
    extent_x: float,
    extent_y: float,
) -> tuple[np.ndarray, int, int]:
    """Fourier kernel padded so a cloud's wings do not wrap around the frame.

    The kernel is centered on the Fourier origin and normalized so its discrete
    integral is one. The returned pad is the buffer shape, not the image shape.
    """
    pix_x = 2.0 * extent_x / npix_x
    pix_y = 2.0 * extent_y / npix_y
    pad_x = int(np.ceil(4.0 * sigma / pix_x))
    pad_y = int(np.ceil(4.0 * sigma / pix_y))
    width = npix_x + 2 * pad_x
    height = npix_y + 2 * pad_y
    yy, xx = np.indices((height, width))
    x = np.where(xx > width / 2, (xx - width) * pix_x, xx * pix_x)
    y = np.where(yy > height / 2, (yy - height) * pix_y, yy * pix_y)
    radius2 = x * x + y * y
    kernel = np.exp(-radius2 / (2.0 * sigma * sigma)) / (2.0 * np.pi * sigma * sigma)
    kernel[radius2 > (4.0 * sigma) ** 2] = 0.0
    integral = float(kernel.sum() * pix_x * pix_y)
    if integral > 0.0:
        kernel /= integral
    return np.fft.rfft2(kernel), height, width


def exponential_tau(h_r: float, tau_v0: float, shape: tuple[int, int], extent: float) -> np.ndarray:
    """Face-on V-band optical depth of a smooth exponential disk, on the image grid."""
    ny, nx = shape
    xs = (np.arange(nx) + 0.5) / nx * 2.0 * extent - extent
    ys = (np.arange(ny) + 0.5) / ny * 2.0 * extent - extent
    ys = ys[::-1]
    x, y = np.meshgrid(xs, ys, indexing="xy")
    return tau_v0 * np.exp(-np.hypot(x, y) / h_r)


def _sample(image: np.ndarray, extent: float, gx: np.ndarray, gy: np.ndarray) -> np.ndarray:
    """Bilinear sample. Galactic (x, y) matches the face-on camera at azimuth 0."""
    ny, nx = image.shape
    sky_x = -gx
    sky_y = -gy
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


def prepare_observed(image: np.ndarray, star_mask: np.ndarray) -> np.ndarray:
    """Fill foreground stars so they are not turned into galactic clouds.

    The replacement is a blur of the surrounding disk with the star pixels left
    out. Blurring the star itself would leave a bright lump, and a saturated
    star would become one large red cloud.
    """
    filled = np.array(image, copy=True)
    peak = np.max(filled, axis=0)
    bright = star_mask & (peak > 1.2)
    bright = bright | _dilate_mask(peak > 8.0, 5)
    if not np.any(bright):
        return np.clip(filled, 0.0, None)
    weight = (~bright).astype(np.float64)
    norm = convolve_gaussian(weight, 5.0)
    for band in range(filled.shape[0]):
        blurred = convolve_gaussian(np.where(bright, 0.0, filled[band]), 5.0)
        filled[band, bright] = blurred[bright] / np.maximum(norm[bright], 1e-3)
    return np.clip(filled, 0.0, None)


def _dilate_mask(mask: np.ndarray, radius: int) -> np.ndarray:
    out = np.array(mask, copy=True)
    ys, xs = np.nonzero(mask)
    ny, nx = mask.shape
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            if dy * dy + dx * dx > radius * radius:
                continue
            yy = ys + dy
            xx = xs + dx
            keep = (yy >= 0) & (yy < ny) & (xx >= 0) & (xx < nx)
            out[yy[keep], xx[keep]] = True
    return out
