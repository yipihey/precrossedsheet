#!/usr/bin/env python3
"""Extract the committed DiscoDJ PDF archive without rerunning a simulation.

Usage: python3 scripts/extract_published.py /path/to/DiscoDJ.jl
Needs NumPy only to read original study NPZ metadata; all plotted data come
from the exact-sheet follow-up's JSON at the same native snapshots.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np

COMMIT = "0db7baf609c7b0294d03e602e93492f128bd7f7d"
REPO = "https://github.com/yipihey/DiscoDJ.jl"


def extract(root):
    commit = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    if commit != COMMIT:
        raise ValueError(f"Expected archive commit {COMMIT}; got {commit}")
    study = root / "studies/lpt_pdf"
    paths = ["studies/lpt_pdf/sheet_pdf/results/stats.json",
             "studies/lpt_pdf/sheet_pdf/results/summary.json",
             "studies/lpt_pdf/results/summary.json"]
    stats = json.loads((root / paths[0]).read_text())
    comparisons = json.loads((root / paths[1]).read_text())
    models = [{"id": f"{n}lpt", "label": "Zel'dovich (1LPT)" if n == 1 else f"{n}LPT", "order": n}
              for n in range(1, 5)] + [{"id": "nbody", "label": "PM N-body", "order": None}]
    runs = []
    for model in models:
        for paired in (False, True):
            suffix = "_paired" if paired else ""
            meta_file = (f"results/nb_N256_m2_s100_ai0.04_log{suffix}.npz" if model["id"] == "nbody"
                         else f"results/{model['id']}_N256{suffix}.npz")
            # Study-1 crossing history belongs to the original NumPy PM or
            # LPT run. The sheet follow-up uses native BullFrog snapshots;
            # keep that distinction explicit, never relabel its history.
            with np.load(study / meta_file, allow_pickle=False) as old:
                meta = json.loads(str(old["meta"].item()))
            rel = "studies/lpt_pdf/" + meta_file
            paths.append(rel)
            crossing = {"ever_mass_fraction": float(meta["cross_ever"]),
                        "final_mass_fraction": float(meta["cross_final"]),
                        "source_path": rel,
                        "belongs_to": "original NumPy study run; native follow-up agrees within reported numerical uncertainty"}
            for estimator in ("sheet", "cic"):
                tag = f"{model['id']}_N256{suffix}|{estimator}"
                source = stats["runs"][tag]
                radii = []
                for radius in (7, 14, 28, 42):
                    row = source[f"R{radius}"]
                    distributions = {}
                    for weighting, letter in (("volume", "V"), ("mass", "M")):
                        pdf = row[f"pdf{letter}"]
                        integral = sum(p * (hi - lo) for p, lo, hi in zip(pdf, stats["LOGBINS"], stats["LOGBINS"][1:]))
                        assert abs(integral - 1) < 1e-10, (tag, radius, weighting, integral)
                        distributions[weighting] = {"pdf": pdf, "quantiles_rho": row[f"q{letter}"],
                                                    "captured_probability": None}
                    radii.append({"R": radius, **distributions, "variance": row["var"],
                                  "S3": row["S3"], "S4": row["S4"]})
                runs.append({"model": model["id"], "estimator": estimator, "paired": paired,
                             "N": 256, "source_key": tag, "crossing": crossing, "radii": radii})
    # Check the displayed ratios against the separately archived summary.
    for run in runs:
        if run["model"] == "nbody" or run["paired"]:
            continue
        nb = next(x for x in runs if x["model"] == "nbody" and x["estimator"] == run["estimator"] and not x["paired"])
        for row, ref in zip(run["radii"], nb["radii"]):
            expected = comparisons["vs_nbody"][f"{run['model']}|R{row['R']}|{run['estimator']}"]
            for name, letter in (("volume", "V"), ("mass", "M")):
                actual = [a / b - 1 for a, b in zip(row[name]["quantiles_rho"], ref[name]["quantiles_rho"])]
                assert max(abs(a - b) for a, b in zip(actual, expected[f"q{letter}"])) < 1e-12
    paths = sorted(set(paths))
    return {
        "schema_version": 1,
        "provenance": {
            "upstream_repo": REPO,
            "upstream_commit": commit,
            "report_url": "https://yipihey.github.io/DiscoDJ.jl/studies/lpt_pdf/report.html#sheet",
            "source_paths": paths,
            "source_sha256": {p: hashlib.sha256((root / p).read_bytes()).hexdigest() for p in paths},
            "extraction": "scripts/extract_published.py; no simulation rerun, histogram rebinning, or inferred data",
            "conditions": {"box_mpc_h": 300, "initial_filter": "real-space spherical top hat",
                           "initial_filter_radius_mpc_h": 14, "redshift": 0, "analysis_mesh": 256,
                           "cosmology": "Planck18 (EE+BAO+SN); Eisenstein-Hu 1998; sigma8=0.8105",
                           "amplitudes": "fixed; paired is phase shift pi of same realization",
                           "lpt_growth": "exact LCDM through order 3; EdS D1^4 at fourth order",
                           "native_nbody": "BullFrog; 64 steps; 2N PM; fd4 force gradient; 2LPT initial conditions at z=24",
                           "density_units": "rho = 1 + delta, mean density = 1"},
            "field_meanings": {
                "pdf": "density per unit log10(rho), normalized within displayed rho range 0.05..20",
                "quantiles_rho": "full-field density at quantile_probabilities; not log density",
                "mass": "each cell weighted by its unsmoothed deposited mass, with the smoothed rho sampled in that cell",
                "volume": "equal weights for Eulerian analysis cells",
                "captured_probability": "null: archived stats do not record out-of-range mass or volume",
                "variance": "variance of smoothed delta",
                "S3": "connected third moment divided by variance squared",
                "S4": "connected fourth moment divided by variance cubed",
                "crossing": "fractions are 0..1, not percentages; original study signed-tetrahedron history"},
            "restrictions": [
                "This is a mostly single-stream filtered realization, not a strictly uncrossed mask: N-body ever-crossed mass is about 0.52% fixed and 0.69% paired.",
                "These unconditional PDFs include that small crossed fraction; copula mask results are separate and not included here.",
                "Sub-percent accuracy is statistic- and scale-dependent; 4LPT dense tails and variance at 7 Mpc/h exceed 1% error.",
                "One 300 Mpc/h box, one 14 Mpc/h initial filter and a paired phase check do not establish accuracy for arbitrary cosmologies or initial spectra.",
                "Archive replay is distinct from live Rust experiments; no data here certify the Rust port.",
                "Sheet histograms integrate the piecewise-linear P0 representation by exact overlaps; this does not remove sheet-resolution, N-body, cell-window or aliasing errors."
            ]},
        "bin_edges_log10rho": stats["LOGBINS"],
        "quantile_probabilities": stats["QS"],
        "models": models,
        "estimators": [
            {"id": "sheet", "label": "Exact P0 sheet", "description": "Six Kuhn tetrahedra per Lagrangian cube; R3D exact overlap deposit; cell-average window deconvolved before top-hat smoothing"},
            {"id": "cic", "label": "CIC particles", "description": "Same native snapshots; particle CIC deposit and CIC window deconvolution before top-hat smoothing"}],
        "smoothing_radii_mpc_h": [7, 14, 28, 42],
        "runs": runs}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_repo", type=Path)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parents[1] / "app/data/published-results.json")
    args = parser.parse_args()
    result = extract(args.source_repo.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, separators=(",", ":"), allow_nan=False) + "\n")
    print(f"Wrote {len(result['runs'])} archived runs to {args.output}")
