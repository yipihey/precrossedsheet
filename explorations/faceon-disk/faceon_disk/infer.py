"""Posterior of a face-on exponential disk.

The likelihood is the face-on image. It depends on the radial scale, the
unattenuated brightness, the dust column, and the ratio of the stellar and
dust scale heights. It does not depend on the absolute scale heights. Those
are set by the log-normal prior and are what an inclined view can see.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .model import AttenuationGrid, Disk, convolve_gaussian, faceon_image

# Medians are intentionally offset from the truth so the image has to move
# the dust column and the scale-height ratio. Widths on the two heights are
# the prior the inclined views are sampling.
PRIOR_MU = np.log(np.array([3.0, 1.0, 1.0, 0.32, 0.20], dtype=np.float64))
PRIOR_SIG = np.array([0.16, 0.28, 0.62, 0.50, 0.58], dtype=np.float64)
PARAM_NAMES = ("h_R", "I0", "tau_V0", "h_star", "h_dust")

TRUTH = Disk(
    h_R=3.2,
    I0=1.0,
    tau_V0=2.4,
    h_star=0.45,
    h_dust=0.10,
    spiral_amplitude=0.38,
    pitch_deg=18.0,
)


@dataclass
class Observation:
    data: np.ndarray
    sigma: float
    npix: int
    extent: float
    psf_sigma: float
    clean: np.ndarray


def make_observation(
    grid: AttenuationGrid,
    npix: int = 128,
    extent: float = 14.0,
    sigma: float = 0.05,
    psf_kpc: float = 0.42,
    seed: int = 1,
    truth: Disk = TRUTH,
) -> Observation:
    psf_sigma = psf_kpc / (2.0 * extent / npix)
    clean = convolve_gaussian(
        faceon_image(truth, npix=npix, extent=extent, grid=grid, n_z=1800),
        psf_sigma,
    )
    rng = np.random.default_rng(seed)
    data = clean + rng.normal(0.0, sigma, size=clean.shape)
    return Observation(data=data, sigma=sigma, npix=npix, extent=extent, psf_sigma=psf_sigma, clean=clean)


def predict(disk: Disk, grid: AttenuationGrid, obs: Observation) -> np.ndarray:
    image = faceon_image(disk, npix=obs.npix, extent=obs.extent, grid=grid)
    return convolve_gaussian(image, obs.psf_sigma)


def log_probability(theta: np.ndarray, grid: AttenuationGrid, obs: Observation, template: Disk = TRUTH) -> float:
    theta = np.asarray(theta, dtype=np.float64)
    if not np.all(np.isfinite(theta)):
        return -np.inf
    h_R, I0, tau, h_star, h_dust = np.exp(theta)
    alpha = h_star / h_dust
    if not (
        0.5 < h_R < 10.0
        and 0.08 < I0 < 6.0
        and 0.03 < tau < 10.0
        and 0.05 < h_star < 2.2
        and 0.025 < h_dust < 1.2
        and grid.ln_alpha[0] < np.log(alpha) < grid.ln_alpha[-1]
    ):
        return -np.inf
    prior = -0.5 * np.sum(((theta - PRIOR_MU) / PRIOR_SIG) ** 2)
    model = predict(template.with_log_params(theta), grid, obs)
    residual = (obs.data - model) / obs.sigma
    return float(prior - 0.5 * np.dot(residual.ravel(), residual.ravel()))


def run_sampler(
    grid: AttenuationGrid,
    obs: Observation,
    nwalkers: int = 24,
    nsteps: int = 500,
    seed: int = 7,
    template: Disk = TRUTH,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Ensemble stretch move. Returns chain (step, walker, param), log-prob, acceptance."""
    if nwalkers < 2 * len(PRIOR_MU) or nwalkers % 2:
        raise ValueError("nwalkers must be even and at least twice the parameter count")
    rng = np.random.default_rng(seed)
    ndim = len(PRIOR_MU)
    pos = PRIOR_MU + PRIOR_SIG * rng.normal(size=(nwalkers, ndim))
    lp = np.array([log_probability(p, grid, obs, template) for p in pos])
    chain = np.empty((nsteps, nwalkers, ndim), dtype=np.float64)
    lp_chain = np.empty((nsteps, nwalkers), dtype=np.float64)
    accepts = 0
    proposals = 0
    half = nwalkers // 2
    groups = (np.arange(0, half), np.arange(half, nwalkers))
    for step in range(nsteps):
        for group, complement in (groups, groups[::-1]):
            partners = complement[rng.integers(0, len(complement), size=len(group))]
            z = ((2.0 - 1.0) * rng.random(len(group)) + 1.0) ** 2 / 2.0
            proposal = pos[partners] + z[:, None] * (pos[group] - pos[partners])
            lp_prop = np.array([log_probability(p, grid, obs, template) for p in proposal])
            log_acc = (ndim - 1) * np.log(z) + lp_prop - lp[group]
            accept = np.log(rng.random(len(group))) < log_acc
            pos[group[accept]] = proposal[accept]
            lp[group[accept]] = lp_prop[accept]
            accepts += int(np.sum(accept))
            proposals += len(group)
        chain[step] = pos
        lp_chain[step] = lp
        if (step + 1) % 50 == 0:
            med = np.median(np.exp(pos), axis=0)
            print(
                f"  step {step + 1:4d}/{nsteps}  mean lnP {np.mean(lp):.1f}  "
                f"accept {accepts / proposals:.2f}  "
                f"median τ {med[2]:.2f}  h★/h dust {med[3] / med[4]:.2f}  h★ {med[3]:.2f}",
                flush=True,
            )
    return chain, lp_chain, accepts / max(proposals, 1)


def posterior_draws(chain: np.ndarray, lp_chain: np.ndarray, burn: int, template: Disk = TRUTH) -> tuple[np.ndarray, np.ndarray]:
    """Flatten the post-burn chain and keep the upper half by log probability."""
    flat = chain[burn:].reshape(-1, chain.shape[-1])
    flat_lp = lp_chain[burn:].reshape(-1)
    finite = np.isfinite(flat_lp)
    flat, flat_lp = flat[finite], flat_lp[finite]
    keep = flat_lp >= np.median(flat_lp)
    return flat[keep], flat_lp[keep]


def _quantile_disk(samples: np.ndarray, quantile: float, template: Disk) -> Disk:
    order = np.argsort(samples[:, 3])
    index = int(round(quantile * (len(order) - 1)))
    return template.with_log_params(samples[order[index]])


def contrast_route(samples: np.ndarray, template: Disk = TRUTH) -> list[Disk]:
    """A closed path that starts at the median disk and swings through thick and thin draws.

    The middle of the path is the blend between a thick draw and a thin draw, which
    is where the orbit reaches its highest inclination.
    """
    median = _quantile_disk(samples, 0.50, template)
    route = [
        median,
        _quantile_disk(samples, 0.82, template),
        _quantile_disk(samples, 0.96, template),
        _quantile_disk(samples, 0.18, template),
        _quantile_disk(samples, 0.04, template),
        median,
    ]
    return route


def blend_disks(path: list[Disk], t: float) -> Disk:
    """Smoothstep through the path in log parameters. t is in [0, 1]."""
    if len(path) == 1:
        return path[0]
    x = np.clip(t, 0.0, 1.0) * (len(path) - 1)
    i = min(int(np.floor(x)), len(path) - 2)
    u = x - i
    u = u * u * (3.0 - 2.0 * u)
    theta = (1.0 - u) * path[i].log_params() + u * path[i + 1].log_params()
    return path[i].with_log_params(theta)
