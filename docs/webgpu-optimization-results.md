# WebGPU spatial fields, geometric observation and retained work

Engine 0.6.0; recorded 2026-10-11; physical state schema 3. CPU is the default and automatic policy uses CPU. These are numerical implementation changes for the established pre-shell-crossing sheet. Crossed outputs remain visible with the existing caution.

## Implemented paths

1. Compensated GPU recovery evaluates continuous P0 mode values and their actual signed initial-amplitude JVP. It first replaces ordinary f32 LPT modes that fail the unchanged `1e-5` derivative gate; any remaining failures use retained CPU f64 integration. The centered endpoint check remains independent.
2. CPU P0 simplex JVPs reuse fixed knot buffers and retained divided-difference storage across the geometry visitor. Repeated-knot values, signed tangents and stable workspace capacity are tested.
3. A native session retains initial fields, prepared geometry/validity, completed PM state, and unsmoothed density/spectrum observations, under a combined 256 MiB budget. Keys include physical inputs and each executed precision source. Smoothing-only updates reuse all compatible levels; an observation-mesh change reuses the original PM state and crossing diagnostics. New PM evolution invalidates old observations. Invalid external density cannot poison a cache hit.
4. The compensated FD shader specializes recurrence and sorting by knot count, selects series order with an inflated conservative factorial-tail bound (maximum order 40), and reuses projected vertex midpoint/gap data at the three Gauss nodes. Separate completed submissions and 32-cell reductions are retained.
5. GPU exact deposition clips triangles/tetrahedra against Cartesian cells in local coordinates. Tetrahedron volume fans use an interior centroid. Positive f32 records and one Kahan sum owner per cell replace upstream fixed-point accumulation. Capacity, graph, conditioning and nonfinite diagnostics require CPU recovery. Laboratory fields must have finite nonnegative entries and unit mean within `1e-6`; no mass renormalization is used.
6. GPU nLPT builds every raw spatial source, displacement and gradient term through fourth order on radix lattices in 2D/3D. Rust supplies actual initial Fourier modes and retains cosmological growth, EdS merging, time evaluation and amplitude homogeneity. Successful construction does not build a hidden CPU nLPT reference. The raw CPU oracle is invoked explicitly by tests.

Initial-condition generation, growth, PDFs, gridded spectra and physical-unit finalization remain CPU operations. One-dimensional and nonradix nLPT requests retain CPU construction. Nonradix force meshes retain CPU PM. Device refusal preserves requested grids and physical parameters. Adapter-supported buffer limits up to 512 MiB are requested, with a separate complete per-stage allocation budget; 64³ raw nLPT exceeds the portable 128 MiB storage-binding default.

## Accuracy and recovery checks

The independent optimization suite passed **25,418 rows**, including all-axis plane waves, the quadratic two-wave solution, barycentric quadrature for primal/JVP channels, repeated/signed modes, periodic translation, folds and thin-element refusal. Identity and negative-translation cases exercise multiple nonzero simplex-batch offsets in both dimensions. A budget one byte below the full deposit allocation estimate rejects before GPU allocation.

The full optimization benchmark passed **46 checks**. It includes EdS and ΛCDM raw nLPT comparisons, waves and filtered Gaussian input, CPU simplex-mode and clipping references, complete worker fields/spectra/PDFs, both smoothing and observation-mesh cache changes, nonradix fallback, crossed output, and forced validation/resource/nonfinite/readback recovery. Each timed pair separately checks mass, every derivative mode, executed GPU stages, density, spectra and PDF budgets.

| Comparison | Component acceptance target | Largest observed absolute discrepancy in raw fixtures |
|---|---|---:|
| Raw nLPT fields versus Rust | `2e-6 + 2e-4*abs(reference)` | 2.98e-7 |
| Compensated analytic mode/JVP versus Rust f64 | `1e-11 + 1e-8*abs(reference)` | 3.84e-15 |
| Exact density versus Rust | `2e-5 + 1e-4*abs(reference)` | 7.15e-7 (2D), 4.77e-7 (3D) |

The separate strict FD regression passed **47 checks** at the original weak-scaled target `abs(FDgpu-FDcpu)/max(abs(FDcpu),1e-9) <= 1e-5`. Forty frozen-input wave/Gaussian executions passed this target and the `1e-12` complex-component repeat budget. Their hashes were also stable on this device; bitwise portability is not asserted. The resident PM regression passed **131 checks**, and the original observer passed **34 kernel and 47 full-request/worker checks**. Its PM cancellation fixture now requests a cold physical run so a warmed cache cannot skip the evolution step being tested.

Compensated f32 is not native f64. Device checks and explicit CPU recovery remain necessary. Exact geometric integration of an affine element is distinct from floating-point clipping accuracy, represented-sheet resolution and physical validity of its trajectory.

## Complete-request timings

Measurements used Apple M5 Max, 128 GiB RAM, macOS arm64 and Chrome 155.0.8059.40 with the Metal 3 adapter. Normal headless policy was used, without unsafe-WebGPU or disabled-sandbox flags. Times cover physical preparation, GPU/CPU stages, exact observations, finalization and structured-clone delivery, excluding rendering. They are completed-work wall times, not GPU timestamp queries.

Device checks and pipelines were warmed before timed pairs; cold refers to the physical-state cache being disabled. Three alternating cold-physical pairs were recorded for each lattice, with matching waves, ΛCDM parameters, `a=0.25`, 32 steps, and matched lattice/force/observation grids. Both arms use the current optimized GPU PM and centered-FD kernels. The baseline disables GPU spatial nLPT, geometric deposition and analytic recovery; the optimized arm enables them. Thus this paired comparison measures those three additions, rather than separately isolating FD specialization or CPU workspace reuse.

| Lattice | Baseline median (range), ms | Optimized median (range), ms | Median ratio | Less elapsed time |
|---|---:|---:|---:|---:|
| 32³ | 3,207.7 (3,186.3–3,232.2) | 1,056.4 (1,040.2–1,085.4) | 3.04× | 67.1% |
| 64³ | 21,934.6 (21,888.5–22,010.8) | 7,342.0 (7,330.9–7,402.6) | 2.99× | 66.5% |

Compatible smoothing-only runs are reported separately: median **269.7 ms** at 32³ and **4,885.1 ms** at 64³. At 32³ all four retained levels hit, using 37,988,984 bytes. At 64³, initial fields and prepared geometry hit, but the 41,943,040-byte PM entry would exceed the combined 256 MiB limit. PM and its observations are therefore recomputed; retained bytes are 252,955,768. This is a measured cache-budget tradeoff, not an all-level hit at 64³.

These are device/request results, not universal speedups. Initial unsuccessful axis, normalization, volume-fan and batch-offset checks were corrected before these measurements and are excluded from the successful timing record. The mass guard caught incomplete multi-batch deposition before accepting scientific output.

## Reproduce and inspect

```sh
make test
make test-wasm
node scripts/test_gpu.mjs --require-gpu --optimization-self --output self.json
node scripts/test_gpu.mjs --require-gpu --optimization-benchmark --output complete.json
node scripts/test_gpu.mjs --require-gpu --fd-repeat --output repeat.json
node scripts/test_gpu.mjs --require-gpu --resident --output resident.json
node scripts/test_gpu.mjs --require-gpu --full --output observer.json
make check
quarto render
```

The [JSON summary](webgpu-optimization-results.json) records requests, raw timing samples, numerical maxima/budgets, per-stage provenance, source and Wasm fingerprints, device limits and toolchain versions. The [compressed complete archive](webgpu-optimization-results-full.json.gz) retains every comparison row and frozen-input output hash; its compressed and uncompressed SHA256 checksums are in the summary. Use Python's standard `gzip` module to read it. The [reuse manifest](../app/js/gpu/reuse-manifest.json) pins the MIT Hopf–Cole sources and distinguishes their adaptations from this laboratory's numerical conventions.
