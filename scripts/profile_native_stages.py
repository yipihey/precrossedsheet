#!/usr/bin/env python3
"""Profile major Rust lab stages from an isolated scratch copy.

The repository's core is never edited and Cargo uses a private target directory.
The scratch copy, exact requests, raw JSON outputs, and instrumented source are
retained under /private/tmp for audit/replay.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import time


ROOT = Path(__file__).resolve().parents[1]


def source_hash(core: Path) -> str:
    files = list((core / "src").rglob("*.rs")) + [
        core / "Cargo.toml", core / "Cargo.lock", core / "examples/lab_dump.rs"
    ]
    digest = hashlib.sha256()
    for path in sorted(files, key=lambda p: str(p.relative_to(core))):
        digest.update(str(path.relative_to(core)).encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"expected one insertion point for {label}, found {count}")
    return text.replace(old, new, 1)


def instrument(path: Path) -> None:
    text = path.read_text()
    text = replace_once(
        text,
        "    r.validate()?;\n    let c = initial_conditions(r);",
        "    r.validate()?;\n    let profile_total = std::time::Instant::now();\n"
        "    let mut profile_stage = std::time::Instant::now();\n    let c = initial_conditions(r);\n"
        "    eprintln!(\"PROFILE initial_conditions_and_lpt_ms={:.3}\", profile_stage.elapsed().as_secs_f64()*1000.);\n"
        "    profile_stage = std::time::Instant::now();",
        "IC/LPT stage",
    )
    text = replace_once(
        text,
        "    let pm = evolve_cosmology(",
        "    eprintln!(\"PROFILE crossing_epoch_guard_and_pm_initial_state_ms={:.3}\", profile_stage.elapsed().as_secs_f64()*1000.);\n"
        "    profile_stage = std::time::Instant::now();\n    let pm = evolve_cosmology(",
        "reference/guard stage",
    )
    text = replace_once(
        text,
        "    let rho = density(g, &x, r.mesh);\n    let pmrho = density(g, pmx, r.mesh);",
        "    eprintln!(\"PROFILE pm_evolution_and_tangent_ms={:.3}\", profile_stage.elapsed().as_secs_f64()*1000.);\n"
        "    profile_stage = std::time::Instant::now();\n"
        "    let rho = density(g, &x, r.mesh);\n    let pmrho = density(g, pmx, r.mesh);\n"
        "    eprintln!(\"PROFILE exact_lpt_and_pm_deposits_ms={:.3}\", profile_stage.elapsed().as_secs_f64()*1000.);\n"
        "    profile_stage = std::time::Instant::now();",
        "PM/deposit stages",
    )
    text = replace_once(
        text,
        "    let psp = power(&eg, &pmrho);\n    let (dpos, _) = tangent::lpt_amplitude_jvp(c.lpt_ref(), d, r.order);",
        "    let psp = power(&eg, &pmrho);\n"
        "    eprintln!(\"PROFILE smoothing_pdfs_summaries_cic_spectra_ms={:.3}\", profile_stage.elapsed().as_secs_f64()*1000.);\n"
        "    profile_stage = std::time::Instant::now();\n"
        "    let (dpos, _) = tangent::lpt_amplitude_jvp(c.lpt_ref(), d, r.order);",
        "measurement stage",
    )
    text = replace_once(
        text,
        "    let q: Vec<f64> = (0..g.size)",
        "    eprintln!(\"PROFILE direct_modes_lpt_pm_tangents_and_fd_ms={:.3}\", profile_stage.elapsed().as_secs_f64()*1000.);\n"
        "    profile_stage = std::time::Instant::now();\n    let q: Vec<f64> = (0..g.size)",
        "direct modes/tangents stage",
    )
    text = replace_once(
        text,
        '    Ok(\n        json!({"dim":r.dim,',
        '    let result = json!({"dim":r.dim,',
        "result binding",
    )
    text = replace_once(
        text,
        '        "positionUnits":"x/L","velocityUnits":"d(x/L)/da","wavenumberUnits":"h/Mpc","powerUnits":format!("(Mpc/h)^{}",r.dim),"observationFilter":"Gaussian","powerField":"unsmoothed cell averages","pdfMassWeights":"unsmoothed cell mass","pmObservation":"exact P0 sheet attached to PM vertices"}}),\n    )',
        '        "positionUnits":"x/L","velocityUnits":"d(x/L)/da","wavenumberUnits":"h/Mpc","powerUnits":format!("(Mpc/h)^{}",r.dim),"observationFilter":"Gaussian","powerField":"unsmoothed cell averages","pdfMassWeights":"unsmoothed cell mass","pmObservation":"exact P0 sheet attached to PM vertices"}});\n'
        '    eprintln!("PROFILE result_assembly_ms={:.3}", profile_stage.elapsed().as_secs_f64()*1000.);\n'
        '    eprintln!("PROFILE run_total_pre_serialization_ms={:.3}", profile_total.elapsed().as_secs_f64()*1000.);\n'
        '    Ok(result)',
        "result assembly stage",
    )
    text = replace_once(
        text,
        "        Ok(v) => v.to_string(),",
        "        Ok(v) => {\n            let serialization = std::time::Instant::now();\n"
        "            let output = v.to_string();\n"
        "            eprintln!(\"PROFILE json_serialize_ms={:.3}\", serialization.elapsed().as_secs_f64()*1000.);\n"
        "            output\n        },",
        "JSON serialization stage",
    )
    path.write_text(text)


def request_for(n: int, gaussian: bool) -> dict:
    return {
        "dim": 3, "n": n, "fraction": 0.5, "order": 4, "seed": 7,
        "ic": "gaussian" if gaussian else "waves", "smoothing": 0.03,
        "steps": 32, "mesh": n, "forceMesh": n,
        "omegaM": 0.315, "omegaB": 0.049, "h": 0.674, "ns": 0.965,
        "sigma8": 0.811, "boxSize": 300.0, "initialSmoothing": 14.0,
        "initialCutoff": 6 if gaussian else 3, "epoch": "scaleFactor",
        "scaleFactor": 0.25,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grids", type=int, nargs="+", choices=[32, 64], default=[32, 64], help="wave grids to profile (default: 32 64)")
    parser.add_argument("--gaussian64", action="store_true", help="also run a 3D64 Gaussian case")
    parser.add_argument("--scratch", type=Path, help="new scratch directory; default creates under /private/tmp")
    args = parser.parse_args()

    scratch = args.scratch or Path(tempfile.mkdtemp(prefix="precrossedsheet-profile-", dir="/private/tmp"))
    if scratch.exists() and any(scratch.iterdir()):
        parser.error(f"scratch directory must be new or empty: {scratch}")
    core = scratch / "core"
    target = scratch / "cargo-target"
    shutil.copytree(ROOT / "core", core, ignore=shutil.ignore_patterns("target", ".DS_Store"))
    reference_hash = source_hash(core)
    instrument(core / "src/laboratory.rs")
    instrumented_hash = source_hash(core)
    print(f"scratch={scratch}")
    print(f"uninstrumented_source_sha256={reference_hash}")
    print(f"instrumented_source_sha256={instrumented_hash}")
    print("hash_method=sorted src/**/*.rs, Cargo.toml, Cargo.lock, examples/lab_dump.rs; path + NUL + bytes + NUL")

    env = os.environ.copy()
    env["CARGO_TARGET_DIR"] = str(target)
    subprocess.run(
        ["cargo", "build", "--release", "--offline", "--manifest-path", str(core / "Cargo.toml"), "--example", "lab_dump"],
        cwd=core, env=env, check=True,
    )
    binary = target / "release/examples/lab_dump"
    cases = [(n, False) for n in args.grids] + ([(64, True)] if args.gaussian64 else [])
    for n, gaussian in cases:
        name = f"3d{n}-{'gaussian' if gaussian else 'waves'}"
        request = request_for(n, gaussian)
        run_dir = scratch / "runs"
        run_dir.mkdir(exist_ok=True)
        input_path = run_dir / f"{name}.request.json"
        output_path = run_dir / f"{name}.response.json"
        input_path.write_text(json.dumps(request, separators=(",", ":")))
        start = time.perf_counter()
        proc = subprocess.run(
            [str(binary)], input=input_path.read_text(), text=True,
            capture_output=True, check=True,
        )
        elapsed = time.perf_counter() - start
        response = json.loads(proc.stdout)
        if response.get("error") or response.get("request") != request:
            raise RuntimeError(f"{name}: failed numerical request or mismatched input: {response.get('error')}")
        output_path.write_text(proc.stdout)
        print(f"CASE {name} end_to_end_including_serialization_s={elapsed:.3f}")
        print(proc.stderr, end="")


if __name__ == "__main__":
    main()
