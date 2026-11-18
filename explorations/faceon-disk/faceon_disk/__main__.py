"""Fit NGC 628 and write the two-orbit movie.

Run from this directory:

    python3 -m faceon_disk
    python3 -m faceon_disk --no-movie
    python3 -m faceon_disk --synthetic
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from .infer import (
    PRIOR_MU,
    PRIOR_SIG,
    TRUTH,
    contrast_route,
    log_probability,
    make_observation,
    posterior_draws,
    predict,
    run_sampler,
)
from .model import AttenuationGrid, Disk, display_scale
from .movie import write_movie
from .observe import DISTANCE_MPC, arm_pattern, load_ngc628
from .report import save_summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Face-on disk posterior and orbit movie")
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parents[1] / "output")
    parser.add_argument("--no-movie", action="store_true")
    parser.add_argument("--synthetic", action="store_true", help="Fit the synthetic closure test instead of NGC 628")
    parser.add_argument("--walkers", type=int, default=24)
    parser.add_argument("--steps", type=int, default=360)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    print("Building the face-on attenuation table", flush=True)
    grid = AttenuationGrid(n_alpha=64, n_tau=96, n_z=1000)
    galaxy = None
    if args.synthetic:
        obs = make_observation(grid)
        prior_mu, prior_sig = PRIOR_MU, PRIOR_SIG
        template = TRUTH
        truth: Disk | None = TRUTH
        heading, subhead = "Synthetic disk", "Exponential star and dust atmospheres"
        observed_detail = "Synthetic g, r, i image"
        display = None
        title = "Synthetic face-on disk"
    else:
        galaxy = load_ngc628()
        obs = galaxy.observation
        prior_mu, prior_sig = galaxy.prior_mu, galaxy.prior_sig
        template = Disk(spiral_amplitude=0.0)
        truth = None
        heading, subhead = "NGC 628 · M74", f"DECaLS DR10 g, r, z · {DISTANCE_MPC:.2f} Mpc · fit as face-on"
        observed_detail = "Foreground screen removed"
        display = galaxy.display
        title = "NGC 628, axisymmetric disk fit"
        print(
            f"NGC 628  bands {obs.bands}  pixels {obs.npix}²  "
            f"extent {obs.extent:.2f} kpc  mask {int(obs.mask.sum())} px  σ={obs.sigma:.4f}",
            flush=True,
        )
    if display is None:
        scale = display_scale(obs.clean)
    else:
        tonemap_source = display.copy()
        tonemap_source[:, galaxy.star_mask] = 0.0
        scale = display_scale(tonemap_source)
    print(
        f"Observation {obs.npix}²  σ={obs.sigma:.4f}  PSF={obs.psf_sigma:.2f} px  scale={scale:.3f}",
        flush=True,
    )
    print(f"Prior center lnP {log_probability(prior_mu, grid, obs, template, prior_mu, prior_sig):.1f}", flush=True)

    started = time.perf_counter()
    chain, lp_chain, acceptance = run_sampler(
        grid,
        obs,
        nwalkers=args.walkers,
        nsteps=args.steps,
        template=template,
        prior_mu=prior_mu,
        prior_sig=prior_sig,
    )
    print(f"Sampling finished in {time.perf_counter() - started:.0f}s, acceptance {acceptance:.2f}", flush=True)
    burn = args.steps // 2
    samples, sample_lp = posterior_draws(chain, lp_chain, burn=burn)
    if galaxy is not None:
        median = template.with_log_params(np.median(samples, axis=0))
        pattern = arm_pattern(median, galaxy.pattern_image, obs.extent, grid, galaxy.star_mask)
        template = median.with_pattern(pattern, obs.extent)
        print(
            f"Arm pattern {pattern.shape[0]}²  median {np.median(pattern):.3f}  "
            f"span [{np.quantile(pattern, 0.05):.3f}, {np.quantile(pattern, 0.95):.3f}]",
            flush=True,
        )
    route = contrast_route(samples, template)
    _report_fit(samples, route, grid, obs, truth)

    np.savez(
        args.output / "posterior.npz",
        chain=chain,
        log_probability=lp_chain,
        samples=samples,
        sample_log_probability=sample_lp,
        prior_mu=prior_mu,
        prior_sig=prior_sig,
        truth=np.array([]) if truth is None else truth.log_params(),
        acceptance=acceptance,
    )
    summary = args.output / "summary.png"
    print(f"Writing {summary}", flush=True)
    # The summary shows the axisymmetric fit. The movie disks carry the arms.
    summary_template = Disk(spiral_amplitude=0.0) if galaxy is not None else TRUTH
    save_summary(summary, obs, summary_template, samples, grid, scale, truth=truth, title=title)

    if args.no_movie:
        return
    rng = np.random.default_rng(11)
    prior_heights = np.exp(prior_mu[3:5] + prior_sig[3:5] * rng.normal(size=(1800, 2)))
    posterior_heights = np.exp(samples[:, 3:5])
    movie = args.output / "faceon_rotation.mp4"
    write_movie(
        movie,
        route[0],
        route,
        prior_heights,
        posterior_heights,
        scale,
        observed_linear=display,
        extent=obs.extent if galaxy is not None else 12.0,
        extinction=obs.band_extinction(),
        heading=heading,
        subhead=subhead,
        observed_detail=observed_detail,
    )
    print(f"Wrote {movie}", flush=True)


def _report_fit(samples: np.ndarray, route: list, grid: AttenuationGrid, obs, truth: Disk | None) -> None:
    med = np.median(np.exp(samples), axis=0)
    names = ("h_R", "I0", "tau_V0", "h_star", "h_dust", "color_g", "color_red")
    if truth is None:
        print("parameter     median")
        for name, got in zip(names, med):
            print(f"  {name:8s}  {got:8.3f}")
    else:
        wanted = np.exp(truth.log_params())
        print("parameter     median    truth")
        for name, got, want in zip(names, med, wanted):
            print(f"  {name:8s}  {got:8.3f}  {want:8.3f}")
    ratio = np.exp(samples[:, 3] - samples[:, 4])
    truth_bit = "" if truth is None else f"  truth {truth.h_star / truth.h_dust:.2f}"
    print(
        f"h★/h dust median {np.median(ratio):.2f}  "
        f"16–84% [{np.quantile(ratio, 0.16):.2f}, {np.quantile(ratio, 0.84):.2f}]"
        f"{truth_bit}"
    )
    print(
        f"ln h★ std {np.std(samples[:, 3]):.3f}   "
        f"ln(h★/h dust) std {np.std(np.log(ratio)):.3f}"
    )
    for disk in route:
        model = predict(disk, grid, obs)
        residual = model - obs.data
        if obs.mask is not None:
            residual = residual[:, obs.mask]
        rms = float(np.sqrt(np.mean(residual**2)))
        print(
            f"  route h★={disk.h_star:.3f} h dust={disk.h_dust:.3f} τ={disk.tau_V0:.2f}  "
            f"face-on rms {rms:.4f} ({rms / obs.sigma:.2f} σ)"
        )


if __name__ == "__main__":
    main()
