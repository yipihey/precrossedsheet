#!/usr/bin/env python3
"""Repeat native prepared-run stages without instrumenting or copying the core.

The historical one-run instrumented baseline is preserved in stage-profile.json
and Git revision 44e42c15. This profiler uses the public prepared-run API,
separates CPU analytic modes from value-only finite differences, and saves
complete inputs, samples, machine/toolchain and source fingerprints.
"""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import subprocess

ROOT = Path(__file__).resolve().parents[1]

def source_hash():
    files = list((ROOT / "core/src").rglob("*.rs")) + [ROOT / p for p in
        ("core/Cargo.toml", "core/Cargo.lock", "core/examples/lab_profile.rs")]
    digest = hashlib.sha256()
    for p in sorted(files):
        digest.update(str(p.relative_to(ROOT)).encode() + b"\0" + p.read_bytes() + b"\0")
    return digest.hexdigest()

def command(*args):
    return subprocess.run(args, cwd=ROOT, check=True, capture_output=True, text=True).stdout.strip()

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dim", type=int, choices=(1,2,3), default=3)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("repeats must be positive")
    fingerprint = source_hash()
    raw = command("cargo", "run", "--release", "--offline", "--manifest-path", "core/Cargo.toml",
        "--example", "lab_profile", "--", str(args.repeats), str(args.dim))
    if source_hash() != fingerprint:
        raise RuntimeError("Core source changed during profiling; repeat on a fixed checkout")
    report = json.loads(raw)
    report["sourceSha256"] = fingerprint
    report["hashMethod"] = "sorted core/src/**/*.rs, Cargo.toml, Cargo.lock, lab_profile.rs; path + NUL + bytes + NUL"
    report["gitRevision"] = command("git", "rev-parse", "HEAD")
    report["dirty"] = bool(command("git", "status", "--porcelain"))
    report["machine"] = {"system": platform.system(), "release": platform.release(), "architecture": platform.machine()}
    report["toolchain"] = command("rustc", "--version")
    report["dimension"] = args.dim
    text = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.write_text(text)
    print(text, end="")

if __name__ == "__main__":
    main()
