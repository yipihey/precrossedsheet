"""NGC 628 as observed, prepared for the face-on disk fit.

The cutout is DECaLS DR10 g, r, z from the Legacy Survey viewer, centered on
the galaxy. The distance is 9.84 Mpc (Anand et al. 2021; also McQuinn et al.
2017). The published inclination is 8.9° (Lang et al. 2020); this fit treats
the image as face-on, which is the limit in which the two scale heights stay
degenerate. A Milky Way screen of E(B−V) = 0.062 (Schlafly & Finkbeiner 2011)
is removed with the same CCM curve the disk uses.

The sampler sees an axisymmetric exponential. After the fit, the stellar light
is painted from the face-on r image so the orbit shows this galaxy's arms.
That paint is a prior, not a measurement of arm heights: the dust stays the
smooth exponential atmosphere, and the bulge is left in the image rather than
given its own component.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .infer import Observation
from .model import AttenuationGrid, Disk, ccm_extinction, convolve_gaussian, faceon_image

# DECam g, r, z wavelengths at which the CCM curve is evaluated. These are near
# the filter effective wavelengths; the integral over each filter is not computed.
DECAM_BANDS = (
    ("g", 480.0),
    ("r", 642.0),
    ("z", 917.0),
)
DECAM_EXTINCTION = np.array([ccm_extinction(w) for _, w in DECAM_BANDS], dtype=np.float64)

DISTANCE_MPC = 9.84
GALACTIC_EBV = 0.062
SEEING_FWHM_ARCSEC = 1.3
BULGE_RADIUS_KPC = 1.6
DISK_OUTER_KPC = 12.5
SKY_RADIUS_KPC = 14.0
BIN_FACTOR = 4

DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "ngc628_decals_dr10_grz.fits"

KPC_PER_ARCSEC_AT_1_MPC = 1.0e3 / 206265.0


@dataclass(frozen=True)
class PreparedGalaxy:
    """Sky-subtracted, foreground-corrected frames and the likelihood mask."""

    observation: Observation
    display: np.ndarray
    pattern_image: np.ndarray
    star_mask: np.ndarray
    prior_mu: np.ndarray
    prior_sig: np.ndarray
    distance_mpc: float
    kpc_per_arcsec: float


def read_fits_image(path: Path) -> tuple[dict[str, float], np.ndarray]:
    """Read a primary FITS image. Returns numeric header cards and a C-order array.

    NAXIS1 is the last numpy axis. Only integer and float BITPIX values are read,
    with BSCALE and BZERO applied.
    """
    raw = Path(path).read_bytes()
    cards: dict[str, str] = {}
    offset = 0
    while offset + 2880 <= len(raw):
        block = raw[offset : offset + 2880].decode("ascii", errors="replace")
        offset += 2880
        finished = False
        for start in range(0, 2880, 80):
            card = block[start : start + 80]
            if card.startswith("END"):
                finished = True
                break
            key = card[:8].strip()
            if not key or key == "COMMENT" or key == "HISTORY" or "=" not in card[8:10]:
                continue
            value = card.split("=", 1)[1].split("/", 1)[0].strip()
            if value.startswith("'"):
                value = value[1:].split("'", 1)[0].strip()
            cards[key] = value
        if finished:
            break
    bitpix = int(cards["BITPIX"])
    naxis = int(cards["NAXIS"])
    axes = [int(cards[f"NAXIS{i}"]) for i in range(1, naxis + 1)]
    count = int(np.prod(axes)) if axes else 0
    dtype = {8: np.uint8, 16: ">i2", 32: ">i4", 64: ">i8", -32: ">f4", -64: ">f8"}[bitpix]
    data = np.frombuffer(raw, dtype=dtype, count=count, offset=offset).astype(np.float64, copy=False)
    scale = float(cards.get("BSCALE", 1.0))
    zero = float(cards.get("BZERO", 0.0))
    if scale != 1.0 or zero != 0.0:
        data = data * scale + zero
    if naxis == 0:
        image = data
    else:
        image = data.reshape(tuple(reversed(axes)))
    header = {key: float(value) for key, value in cards.items() if _is_number(value)}
    return header, np.array(image, dtype=np.float64, copy=True)


def foreground_factor(extinction: np.ndarray, ebv: float = GALACTIC_EBV, r_v: float = 3.1) -> np.ndarray:
    """Multiply observed f_ν by this to remove a foreground CCM screen."""
    a_lambda = r_v * ebv * np.asarray(extinction, dtype=np.float64)
    return 10.0 ** (0.4 * a_lambda)


def arm_pattern(
    disk: Disk,
    image_r: np.ndarray,
    extent: float,
    grid: AttenuationGrid,
    star_mask: np.ndarray,
) -> np.ndarray:
    """Stellar pattern whose smooth-dust face-on image matches the r frame.

    Foreground stars, the pixels well above the disk, are filled in. Arm knots
    are kept. The dust used to predict this frame is the smooth exponential,
    so the pattern does not have to fight extra extinction inside the arms.
    """
    model = faceon_image(
        disk,
        npix=image_r.shape[0],
        extent=extent,
        grid=grid,
        extinction=DECAM_EXTINCTION,
    )[1]
    foreground = star_mask & (image_r > 1.5)
    background = convolve_gaussian(image_r, 4.0)
    filled = np.where(foreground, background, image_r)
    filled = convolve_gaussian(np.clip(filled, 0.0, None), 0.45)
    ny, nx = image_r.shape
    xs = (np.arange(nx) + 0.5) / nx * 2.0 * extent - extent
    ys = ((np.arange(ny) + 0.5) / ny * 2.0 * extent - extent)[::-1]
    radius = np.hypot(xs[None, :], ys[:, None])
    radial = np.exp(-radius / disk.h_R)
    atten = model / np.maximum(disk.I0 * radial, 1e-4)
    stellar = filled / np.maximum(disk.I0 * atten, 1e-3)
    floor = 0.025 * float(np.quantile(model, 0.99))
    weight = model / (model + floor)
    stellar = weight * stellar + (1.0 - weight) * radial
    stellar = np.where(radius > DISK_OUTER_KPC, radial, stellar)
    cap = 6.0 * np.maximum(radial, 1e-3)
    return np.clip(stellar, 0.0, cap)


def load_ngc628(path: Path = DATA_PATH) -> PreparedGalaxy:
    """Build the likelihood image, the display frame, and the color prior."""
    header, cube = read_fits_image(path)
    if cube.ndim != 3 or cube.shape[0] != 3:
        raise ValueError(f"{path} is not a 3-band image")
    bands, ny, nx = cube.shape
    if bands != 3:
        raise ValueError(f"{path} does not have three bands")
    pix_deg = abs(float(header["CD1_1"]))
    pix_arcsec = pix_deg * 3600.0
    kpc_per_arcsec = DISTANCE_MPC * KPC_PER_ARCSEC_AT_1_MPC
    kpc_per_pix = pix_arcsec * kpc_per_arcsec
    radius = _radius_kpc(ny, nx, kpc_per_pix)

    sky = np.array([np.median(cube[b, radius > SKY_RADIUS_KPC]) for b in range(3)])
    cube = cube - sky[:, None, None]
    cube *= foreground_factor(DECAM_EXTINCTION)[:, None, None]

    star_mask = _star_mask(cube[1], radius)
    annulus = (radius > 3.0) & (radius < 7.0) & ~star_mask
    level = float(np.median(cube[1, annulus]))
    if level <= 0:
        raise RuntimeError("the r-band annulus median is not positive after sky subtraction")
    cube = cube / level * 0.30

    display = cube.copy()
    pattern_image = cube[1].copy()
    binned = _block_mean(cube, BIN_FACTOR)
    binned_stars = _block_mean(star_mask.astype(np.float64)[None], BIN_FACTOR)[0]
    binned_radius = _radius_kpc(binned.shape[1], binned.shape[2], kpc_per_pix * BIN_FACTOR)
    mask = (binned_stars < 0.08) & (binned_radius > BULGE_RADIUS_KPC) & (binned_radius < DISK_OUTER_KPC)

    outer = binned_radius > SKY_RADIUS_KPC
    if not np.any(outer):
        outer = binned_radius > DISK_OUTER_KPC
    outer_r = binned[1, outer]
    sky_rms = 1.4826 * float(np.median(np.abs(outer_r - np.median(outer_r))))
    disk_annulus = mask & (binned_radius > 3.0) & (binned_radius < 7.0)
    disk_level = float(np.median(binned[1, disk_annulus]))
    # The floor is model error. Photon noise is far below the exponential mismatch.
    sigma = max(4.0 * sky_rms, 0.10 * disk_level)

    colors = _annulus_colors(binned, binned_radius, binned_stars)
    prior_mu = np.log(np.array([3.2, 1.6, 0.9, 0.40, 0.15, colors[0], colors[1]], dtype=np.float64))
    prior_sig = np.array([0.28, 0.55, 0.70, 0.55, 0.60, 0.28, 0.28], dtype=np.float64)

    extent = 0.5 * binned.shape[-1] * kpc_per_pix * BIN_FACTOR
    psf_kpc = (SEEING_FWHM_ARCSEC / 2.355) * kpc_per_arcsec
    psf_sigma = psf_kpc / (2.0 * extent / binned.shape[-1])
    observation = Observation(
        data=binned,
        sigma=sigma,
        npix=binned.shape[-1],
        extent=float(extent),
        psf_sigma=float(psf_sigma),
        clean=binned,
        extinction=DECAM_EXTINCTION.copy(),
        mask=mask,
        bands=("g", "r", "z"),
    )
    return PreparedGalaxy(
        observation=observation,
        display=display,
        pattern_image=pattern_image,
        star_mask=star_mask,
        prior_mu=prior_mu,
        prior_sig=prior_sig,
        distance_mpc=DISTANCE_MPC,
        kpc_per_arcsec=kpc_per_arcsec,
    )


def _annulus_colors(image: np.ndarray, radius: np.ndarray, star_fraction: np.ndarray) -> tuple[float, float]:
    use = (radius > 3.0) & (radius < 8.0) & (star_fraction < 0.08) & (image[1] > 0)
    g_over_r = float(np.median(image[0, use] / image[1, use]))
    z_over_r = float(np.median(image[2, use] / image[1, use]))
    return float(np.clip(g_over_r, 0.3, 2.5)), float(np.clip(z_over_r, 0.4, 3.5))


def _radius_kpc(ny: int, nx: int, kpc_per_pix: float) -> np.ndarray:
    yy, xx = np.indices((ny, nx))
    return np.hypot(xx - (nx - 1) / 2.0, yy - (ny - 1) / 2.0) * kpc_per_pix


def _star_mask(image: np.ndarray, radius: np.ndarray) -> np.ndarray:
    """Compact sources that sit well above the local disk."""
    background = convolve_gaussian(image, 7.0)
    noise_region = radius > SKY_RADIUS_KPC
    noise = 1.4826 * float(np.median(np.abs((image - background)[noise_region])))
    noise = max(noise, 1e-4)
    peaks = (image > background + max(6.0 * noise, 0.04)) & (image > np.maximum(background, 0.0) * 2.8 + 4.0 * noise)
    return _dilate(peaks, 3)


def _dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    out = mask.copy()
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


def _block_mean(image: np.ndarray, factor: int) -> np.ndarray:
    if image.ndim == 2:
        return _block_mean(image[None], factor)[0]
    bands, ny, nx = image.shape
    ny2, nx2 = ny // factor, nx // factor
    trimmed = image[:, : ny2 * factor, : nx2 * factor]
    return trimmed.reshape(bands, ny2, factor, nx2, factor).mean(axis=(2, 4))


def _is_number(value: str) -> bool:
    try:
        float(value)
    except ValueError:
        return False
    return True
