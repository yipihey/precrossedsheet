# Native laboratory benchmarks

Recorded 2026-10-10T21:48:31.947257+00:00 on an Apple M5 Max MacBook Pro (Mac17,6), 18 CPU cores, 128 GB RAM, arm64, macOS 26.7 (25G229; Darwin 25.6.0). The compiler and Cargo were both 1.98.1; the runner engine was 0.2.0. Full tool version strings and fingerprints are in [benchmarks.json](benchmarks.json).

Source SHA256: `f66987d60d201995d42cb3a682d715d6cf77e01b4780acc6ec231d9b4e76dd51`. Native executable SHA256: `1240acd8ef14037b004a48cbd5ca46d6a4b949408cb1aa411261dde9f03fd6de`. Both matched at the end of collection; all six output sample hashes matched the preceding cohort.

These are **native Rust release measurements**, not WebAssembly or browser measurements. Runs were sequential with one timed process per case, without warm-up or repeated statistical sampling. Machine load was not controlled. Do not use these numbers to predict browser runtime or browser memory use.

The measured wall interval runs from process launch to completion. It includes the public laboratory pipeline: initial fields and fourth-order LPT, PM dynamics with its amplitude tangent, sheet deposits, spectra, PDFs, Fourier observations and finite-difference checks, plus JSON serialization and output-file writes. It excludes the build and the Python collector's JSON parsing. Peak resident memory is the native child process's `os.wait4` / `rusage.ru_maxrss`, interpreted in Darwin bytes; it excludes the collector. `/usr/bin/time -l` could not provide resource statistics because this sandbox denied `sysctl kern.clockrate`.

The allocation estimate is the runner's `n^dim × 2400 + max(mesh, forceMesh)^dim × 250` model. It is **not measured memory** or a rigorous OS RSS bound; small processes can exceed it because of runtime overhead. MiB here means 1,048,576 bytes.

## Results

| IC | Grid / force / observation mesh | Wall time (s) | Measured peak RSS (MiB) | Estimated allocation (MiB) | Max mass error | Minimum simplex Jacobian | Guard |
|---|---|---:|---:|---:|---:|---:|---|
| waves | 1024 | 0.445 | 3.58 | 2.59 | 5.24e-10 | 0.780203 | positive |
| waves | 128² | 1.931 | 22.92 | 41.41 | 1.13e-10 | 0.587466 | positive |
| waves | 256² | 7.062 | 87.16 | 165.62 | 1.42e-10 | 0.587060 | positive |
| waves | 32³ | 14.907 | 68.19 | 82.81 | 5.82e-11 | 0.466901 | positive |
| waves | 64³ | 119.986 | 516.08 | 662.50 | 2.61e-11 | 0.459745 | positive |
| gaussian (cutoff 6) | 64³ | 125.262 | 526.03 | 662.50 | 6.71e-11 | 0.487142 | positive |

`Max mass error` is the larger of `|mass − 1|` and `|pmMass − 1|` from the returned exact-deposit diagnostics. The minimum simplex Jacobian combines sampled LPT history and the polynomial minimum throughout every discrete PM drift. It is a resolved diagnostic, not a proof that finite-order LPT or finite-resolution PM reproduces an exact continuum solution. Each returned density length is exactly `mesh^dim`; individual masses, density lengths and all guard fields are saved in the JSON record.

## Exact requests and replay

Each case uses `epoch="scaleFactor"`, `scaleFactor=0.25`, `steps=32`, `order=4`, `seed=7`, and `n=mesh=forceMesh` as shown above. The full inputs, including inactive default fields, are stored under each `results[].input` in [benchmarks.json](benchmarks.json). They match the complete request echoed by the runner (`returned_request_matches_input=true`). Common remaining fields are:

```json
{
  "fraction": 0.5,
  "order": 4,
  "seed": 7,
  "smoothing": 0.03,
  "steps": 32,
  "omegaM": 0.315,
  "omegaB": 0.049,
  "h": 0.674,
  "ns": 0.965,
  "sigma8": 0.811,
  "boxSize": 300.0,
  "initialSmoothing": 14.0,
  "epoch": "scaleFactor",
  "scaleFactor": 0.25
}
```

The waves cases select `ic="waves"`, `initialCutoff=3`; the Gaussian case selects `ic="gaussian"`, `initialCutoff=6`. Wave amplitudes are deterministic and independent of sigma8 and the Gaussian initial filter. The Gaussian case uses the stored cosmology, sigma8 normalization, 14 Mpc/h initial Gaussian smoothing and the explicit radial integer-mode cutoff 6.

Build and replay the first stored case from the repository root:

```sh
cargo build --release --offline --manifest-path core/Cargo.toml --example lab_dump
python3 -c 'import json; b=json.load(open("docs/benchmarks.json")); print(json.dumps(b["results"][0]["input"]))' | core/target/release/examples/lab_dump > /tmp/precrossedsheet-replay.json
```

Change the result index 0 to replay another case. Offline building assumes Cargo dependencies are already cached. The release profile uses optimization level 3, LTO, one codegen unit and aborting panics.

Each record stores a compact `output_sample` and its SHA256. The sample contains the epoch, crossing reference, diagnostics, guard, first eight density values, continuous direct mode powers and gradient errors. Its hash uses sorted-key, compact UTF-8 JSON as documented in `measurement.sample_hash`. These are fingerprints of these native samples, not cross-platform tolerance comparisons. The native executable SHA256 and a source SHA256 are recorded with their hashing convention.
