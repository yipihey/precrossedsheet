"""Still summary of the face-on posterior and the inclined realizations."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .infer import PRIOR_MU, PRIOR_SIG, Observation, predict
from .model import EXTINCTION, AttenuationGrid, Disk, render, tonemap


def save_summary(
    path: Path,
    obs: Observation,
    truth: Disk,
    samples: np.ndarray,
    grid: AttenuationGrid,
    scale: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(3)
    pick = rng.choice(len(samples), size=min(8, len(samples)), replace=False)
    disks = [truth.with_log_params(samples[i]) for i in pick]
    median = truth.with_log_params(np.median(samples, axis=0))
    model = predict(median, grid, obs)

    fig = plt.figure(figsize=(12.4, 7.6), layout="constrained")
    grid_spec = fig.add_gridspec(2, 3)

    ax = fig.add_subplot(grid_spec[0, 0])
    ax.imshow(tonemap(obs.data, scale=scale), origin="upper")
    ax.set_title("Face-on observation")
    ax.set_axis_off()

    ax = fig.add_subplot(grid_spec[0, 1])
    ax.imshow(tonemap(model, scale=scale), origin="upper")
    ax.set_title("Posterior median, face-on")
    ax.set_axis_off()

    ax = fig.add_subplot(grid_spec[0, 2])
    residual = (obs.data[1] - model[1]) / obs.sigma
    limit = 4
    panel = ax.imshow(residual, origin="upper", cmap="coolwarm", vmin=-limit, vmax=limit)
    ax.set_title("r-band residual / σ")
    ax.set_axis_off()
    fig.colorbar(panel, ax=ax, fraction=0.046, pad=0.04)

    ax = fig.add_subplot(grid_spec[1, 0])
    prior = np.exp(PRIOR_MU[3:5] + PRIOR_SIG[3:5] * rng.normal(size=(2500, 2)))
    ax.scatter(prior[:, 0], prior[:, 1], s=6, c="#b7c0cc", linewidths=0, label="prior", rasterized=True)
    ax.scatter(np.exp(samples[:, 3]), np.exp(samples[:, 4]), s=10, c="#c9842a", linewidths=0, label="posterior", rasterized=True)
    ax.scatter([truth.h_star], [truth.h_dust], s=60, c="black", marker="+", label="truth")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("stellar scale height (kpc)")
    ax.set_ylabel("dust scale height (kpc)")
    ax.set_title("Height prior, cut by the image")
    ax.legend(frameon=False, fontsize=8)

    ax = fig.add_subplot(grid_spec[1, 1])
    extent = 12.0
    npix = 140
    center = npix // 2
    for disk in disks:
        edge = render(disk, 78.0, azimuth_deg=8.0, npix_x=npix, extent_x=extent, steps=180)
        column = edge[1, :, center]
        y = np.linspace(extent, -extent, npix)
        ax.plot(y, column, color="#c9842a", alpha=0.45, lw=1.2)
    truth_edge = render(truth, 78.0, azimuth_deg=8.0, npix_x=npix, extent_x=extent, steps=180)
    y = np.linspace(extent, -extent, npix)
    ax.plot(y, truth_edge[1, :, center], color="black", lw=1.6, label="truth")
    ax.set_xlim(-3.5, 3.5)
    ax.set_xlabel("height on the sky at 78° (kpc)")
    ax.set_ylabel("r-band surface brightness")
    ax.set_title("Same draws, inclined")
    ax.legend(frameon=False, fontsize=8)

    ax = fig.add_subplot(grid_spec[1, 2])
    radius = _pixel_radius(obs.npix, obs.extent)
    bins = np.linspace(0.0, 10.0, 18)
    centers = 0.5 * (bins[:-1] + bins[1:])
    observed_profile, counts = _annulus(obs.data[1], radius, bins)
    model_profile, _ = _annulus(model[1], radius, bins)
    noise = obs.sigma / np.sqrt(np.maximum(counts, 1))
    ax.errorbar(centers, observed_profile, yerr=noise, fmt="o", ms=3.5, color="#1f4e79", lw=1, label="observation")
    for disk in disks:
        profile, _ = _annulus(predict(disk, grid, obs)[1], radius, bins)
        ax.plot(centers, profile, color="#c9842a", alpha=0.4, lw=1.1)
    ax.plot(centers, model_profile, color="#c9842a", lw=1.8, label="posterior draws")
    ax.set_xlabel("radius (kpc)")
    ax.set_ylabel("r-band surface brightness")
    ax.set_title("Face-on, the draws agree")
    ax.legend(frameon=False, fontsize=8)

    ratio = np.exp(samples[:, 3] - samples[:, 4])
    fig.suptitle(
        "Face-on exponential disk"
        f"   ·   posterior h★/h dust = {np.median(ratio):.2f}"
        f"   ·   truth {truth.h_star / truth.h_dust:.2f}"
        f"   ·   A_g/A_V = {EXTINCTION[0]:.2f}",
        fontsize=12,
    )
    fig.savefig(path, dpi=140)
    plt.close(fig)


def _pixel_radius(npix: int, extent: float) -> np.ndarray:
    xs = (np.arange(npix) + 0.5) / npix * 2.0 * extent - extent
    ys = xs[::-1]
    X, Y = np.meshgrid(xs, ys, indexing="xy")
    # Face-on camera at azimuth 0 flips both axes; the radius is unchanged.
    return np.hypot(X, Y)


def _annulus(image: np.ndarray, radius: np.ndarray, bins: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    index = np.digitize(radius, bins) - 1
    means = np.zeros(len(bins) - 1)
    counts = np.zeros(len(bins) - 1)
    for i in range(len(means)):
        mask = index == i
        counts[i] = np.count_nonzero(mask)
        if counts[i]:
            means[i] = np.mean(image[mask])
    return means, counts
