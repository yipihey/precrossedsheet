"""Fit a synthetic face-on disk and write the two-orbit movie.

Run from this directory:

    python3 -m faceon_disk
    python3 -m faceon_disk --no-movie
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
from .model import AttenuationGrid, display_scale
from .movie import write_movie
from .report import save_summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Face-on disk posterior and orbit movie")
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parents[1] / "output")
    parser.add_argument("--no-movie", action="store_true")
    parser.add_argument("--walkers", type=int, default=24)
    parser.add_argument("--steps", type=int, default=450)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    print("Building the face-on attenuation table", flush=True)
    grid = AttenuationGrid(n_alpha=72, n_tau=112, n_z=1200)
    obs = make_observation(grid)
    scale = display_scale(obs.clean)
    print(
        f"Observation {obs.npix}²  σ={obs.sigma}  PSF={obs.psf_sigma:.2f} px  scale={scale:.3f}",
        flush=True,
    )
    print(
        f"Truth lnP {log_probability(TRUTH.log_params(), grid, obs):.1f}   "
        f"prior center lnP {log_probability(PRIOR_MU, grid, obs):.1f}",
        flush=True,
    )

    started = time.perf_counter()
    chain, lp_chain, acceptance = run_sampler(grid, obs, nwalkers=args.walkers, nsteps=args.steps)
    print(f"Sampling finished in {time.perf_counter() - started:.0f}s, acceptance {acceptance:.2f}", flush=True)
    burn = args.steps // 2
    samples, sample_lp = posterior_draws(chain, lp_chain, burn=burn)
    route = contrast_route(samples)
    _report_fit(samples, route, grid, obs)

    np.savez(
        args.output / "posterior.npz",
        chain=chain,
        log_probability=lp_chain,
        samples=samples,
        sample_log_probability=sample_lp,
        prior_mu=PRIOR_MU,
        prior_sig=PRIOR_SIG,
        truth=TRUTH.log_params(),
        acceptance=acceptance,
    )
    summary = args.output / "summary.png"
    print(f"Writing {summary}", flush=True)
    save_summary(summary, obs, TRUTH, samples, grid, scale)

    if args.no_movie:
        return
    rng = np.random.default_rng(11)
    prior_heights = np.exp(PRIOR_MU[3:5] + PRIOR_SIG[3:5] * rng.normal(size=(1800, 2)))
    posterior_heights = np.exp(samples[:, 3:5])
    movie = args.output / "faceon_rotation.mp4"
    write_movie(
        movie,
        route[0],
        route,
        prior_heights,
        posterior_heights,
        scale,
    )
    print(f"Wrote {movie}", flush=True)


def _report_fit(samples: np.ndarray, route: list, grid: AttenuationGrid, obs) -> None:
    med = np.median(np.exp(samples), axis=0)
    truth = np.exp(TRUTH.log_params())
    names = ("h_R", "I0", "tau_V0", "h_star", "h_dust")
    print("parameter     median    truth")
    for name, got, want in zip(names, med, truth):
        print(f"  {name:8s}  {got:8.3f}  {want:8.3f}")
    ratio = np.exp(samples[:, 3] - samples[:, 4])
    print(
        f"h★/h dust median {np.median(ratio):.2f}  "
        f"16–84% [{np.quantile(ratio, 0.16):.2f}, {np.quantile(ratio, 0.84):.2f}]  "
        f"truth {TRUTH.h_star / TRUTH.h_dust:.2f}"
    )
    print(
        f"ln h★ std {np.std(samples[:, 3]):.3f}   "
        f"ln(h★/h dust) std {np.std(np.log(ratio)):.3f}   "
        f"prior ln h★ std {PRIOR_SIG[3]:.3f}"
    )
    data_scale = np.std(obs.data)
    for disk in route:
        model = predict(disk, grid, obs)
        rms = np.sqrt(np.mean((model - obs.clean) ** 2))
        print(
            f"  route h★={disk.h_star:.3f} h dust={disk.h_dust:.3f} τ={disk.tau_V0:.2f}  "
            f"face-on rms {rms:.4f} ({rms / data_scale:.3f} of the image rms)"
        )


if __name__ == "__main__":
    main()
