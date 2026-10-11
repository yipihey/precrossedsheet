# Resident WebGPU cosmological PM: validation and measurements

Engine 0.4.0, recorded 2026-10-11. This extends the
[engine 0.3.0 observation backend](webgpu-results.md) with resident PM evolution.
It changes numerical execution, while retaining the physical request, force
operator, cosmological clocks and conservative observation conventions.
Smaller-scale extensions remain deferred. CPU is the default; `auto` stays CPU.

## Numerical contract

Rust constructs the common fourth-order growing PM initial condition and
computes canonical KDK clock integrals for the specified flat matter–Λ or EdS
background. The GPU evolves lattice-relative displacement, canonical momentum
and both amplitude directions. Rust forms displacement in f64 before casting
to f32. CIC coordinates combine the integer lattice-to-force-grid quotient and
remainder with displacement, avoiding premature addition to an absolute f32
position. Absolute f32 positions are materialized for orientation diagnostics
and final exact observations.

Every force evaluation includes CIC density and its direction, forward FFTs,
`i (3 Ω_m / 2) k_d / |k|²`, inverse FFTs, and particle interpolation with both
the field direction and the interpolation-weight direction. The full Nyquist
wavevector enters the denominator; the corresponding odd derivative is zero.
Kicks and drifts use Rust's f64 clocks rounded to f32. Cosmology and clocks are
held fixed under differentiation. This is a JVP of the discrete CIC/PM solver.

The FFT is adapted from the MIT Hopf–Cole Stockham stages at revision
`3a3a13b9ba817c5b4f8f9f26314715845e13d4ad`, with dimension/layout and ownership
changes recorded in [reuse-manifest.json](../app/js/gpu/reuse-manifest.json).
CIC accumulation uses integer-atomic linked lists and one floating sum owner
per mesh node. List order can vary with GPU scheduling, so reproducibility
uses component budgets rather than bitwise equality.

The PM comparison target is `|GPU − reference| ≤ absolute + relative |reference|`
for each packed component in internal unit-box coordinates:

| Quantity | Absolute | Relative |
|---|---:|---:|
| Position x | 2e-5 | 2e-5 |
| Canonical momentum p | 5e-6 | 5e-5 |
| Amplitude direction dx | 2e-5 | 5e-5 |
| Momentum direction dp | 5e-6 | 1e-4 |

The harness compares the original f64 Rust state and a separate f64 Rust
reference initialized with exactly the GPU-packed displacement, momentum and
directions. The latter reconstructs `q + f32(ψ)` in f64, isolating input
quantization from GPU evolution and clock rounding. It does not round that
sum to an absolute f32 position before reference evolution. A previous
absolute-position packing experiment passed primal tests but failed sparse
CIC tangent comparisons; the displacement representation addresses that
precision loss without widening these budgets.

FFT roundtrips and normalized single-mode bins have a 2e-5 absolute/relative
budget. Isolated CIC forces and force JVPs are compared with an independent
f64 direct DFT at 1e-6 absolute plus 1e-4 relative. The independent CIC JVP
also agrees with centered differences away from cell boundaries. These
absolute floors do not promise relative accuracy for vanishing components.

Each PM drift checks determinant endpoints and all interior stationary
points. GPU and Rust f64 extrema on the same materialized f32 endpoints must
agree within 2e-4 absolute plus 2e-4 relative in the harness. Production
requests recheck candidates at most 1e-3 or invalid arithmetic in Rust f64;
this is a heuristic recheck threshold, not an interval error bound. The first
failed drift bracket is recorded and crossed results continue with a caution.
The result manifest records the original f64 initial minimum, the minimum
after displacement materialization, and the selected initial guard value.

## Recorded checks and measurements

The release-binary run passed all 146 resident checks and benchmark
comparisons: six complete numerical fixtures, independent FFT/force tests,
worker recovery/cancellation checks, and matched timing comparisons. The
fixtures span a non-radix Lagrangian lattice in 2D, 3D EdS and ΛCDM, sparse
and matched force meshes, waves and filtered Gaussian fields, and a crossed
diagnostic continuation. Every recorded comparison is enforced, including
the crossed fixture. The unchanged observer suite separately passed 34
independent kernel comparisons and 47 complete-request/worker checks.

Across the six full PM fixtures, maximum errors against the displacement-
quantized f64 reference were `1.45e-7` in x, `1.18e-7` in p, `1.24e-7` in dx
and `3.14e-7` in dp. Against the original f64 initial-state reference, maxima
were `1.46e-7`, `1.22e-7`, `1.24e-7` and `3.17e-7`, respectively. The largest
GPU/Rust drift-minimum discrepancy was `2.21e-7`; the largest exact PM density
component discrepancy was `3.82e-6`. These are observed errors for these
fixtures, rather than universal bounds on arbitrary requests.

Measurements used an Apple M5 Max, 128 GiB RAM, macOS arm64 and Chrome
155.0.8059.40 with its Metal 3 adapter. Normal headless Chrome policy was
used, without unsafe WebGPU or disabled-sandbox flags. Values below are CPU
wall times around completed work, rather than GPU timestamp queries.
Both grids are matched, with 32 KDK steps; inputs and timing scope are
specified below. The comparison includes f64 CPU versus f32 GPU arithmetic,
with all timed GPU states checked against their matched CPU reference.

| Matched lattice/force grid | CPU f64 PM median (range), ms | GPU f32 PM median (range), ms | Median ratio |
|---|---:|---:|---:|
| 32³ | 662.9 (661.4–663.7) | 70.0 (64.7–80.2) | 9.47× |
| 64³ | 5,245.0 (5,244.6–5,271.9) | 152.4 (131.5–176.6) | 34.42× |

The complete matched 32³ laboratory run had a median of **6,204.1 ms**
(6,151.3–6,209.9) with CPU PM and the GPU observer, versus **5,607.8 ms**
(5,591.3–5,623.5) with resident GPU PM and the same observer. This is a
1.106× ratio, or 9.61% less elapsed time. The resident run still spent a
median 1,466.0 ms on CPU finite differences and 3,091.9 ms on selective
f64 LPT mode recovery. These remaining stages explain why the PM-stage
ratio does not carry through to the entire experiment.

The complete [JSON record](webgpu-pm-results.json) contains raw timing
samples, requests, component budgets and errors, worker checks, device
limits, shader hashes, source/binary fingerprints and toolchain versions.
Only this device has been measured. An earlier benchmark browser became
unresponsive and produced no completed report; that run was discarded.
The runner now bounds browser commands, and benchmark warmup state arrays
are excluded from report serialization.

## Reproduce

```sh
make test
make test-wasm
node scripts/test_gpu.mjs --require-gpu --resident --output resident-checks.json
node scripts/test_gpu.mjs --require-gpu --resident-worker --output resident-worker.json
node scripts/test_gpu.mjs --require-gpu --resident-benchmark --output resident-benchmark.json
node scripts/test_gpu.mjs --require-gpu --full --output observer-checks.json
make check
quarto render
```

The resident suite is also available at
[the check page](https://yipihey.github.io/precrossedsheet/app/gpu-test.html?autorun=1&resident=1).
Use `resident-benchmark=1` for timing samples. The runner serves the repository
on loopback, uses an isolated Chrome profile, and records device limits,
Chrome flags, full requests, shader hashes and reduced error statistics.
A missing adapter is a skip; `--require-gpu` makes it a failing requirement.
GPU hardware validation is separate from hosted CI's Rust/Wasm/worker checks.

## Timing scope

The matched PM timing comparison uses a fresh deferred Rust context for every
sample. CPU PM timing starts immediately before the uncached f64 reference
call and includes result packing; GPU timing includes parameter setup,
uploads, completed submissions, scalar orientation readback and final state
readback. Initial conditions/nLPT preparation is recorded separately. One
GPU warm-up is excluded, followed by three alternating CPU/GPU pairs at each
matched 32³ and 64³ lattice/force mesh, with 32 steps.

The complete-run comparison alternates three observer-only and resident-PM
requests through one warmed persistent worker at matched 32³ grids. Both use
the same GPU continuous observer and CPU weak-mode recovery/finite difference.
It includes repeated physical preparation, exact deposition, PDFs, transfers
and finalization; rendering is excluded. Stage fields can overlap: PM
`kernelMs` includes drift-polynomial GPU work, while `diagnosticMs` measures
the initial diagnostic. Overall preparation includes resident PM and exact
observations, and finalization includes selective recovery. Do not sum all
manifest timing fields to reconstruct elapsed time.

## Scope and reuse

GPU execution is opt-in in 2D and 3D. The force mesh must be radix-2; the
Lagrangian and observation meshes have their existing limits and are not
silently resized. Initial fields, growth, nLPT construction, exact clipping,
PDFs, gridded observation spectra, finite differences and final conversion
remain Rust/Wasm. PM recovery and continuous-mode recovery are independent.
All attempted backend and precision decisions remain in the manifest.

The reusable JS entry points are `GpuFFT` and `runPm`; the deferred Rust/Wasm
boundary is `PendingLabRun`. Per-run buffers are destroyed on completion or
failure. Cancellation is observed between submitted work, and a queued
replacement cannot publish an obsolete result. A conservative allocation
check covers GPU fields and staging separately from the core's 1 GiB envelope;
retaining the f64 initial state increases memory use and is included in the
core's conservative estimate. Allocation fields are estimates, not
measurements of peak resident memory.

This release establishes no new universal 4LPT accuracy result, validated
post-crossing dynamics, cosmological-parameter gradient or reverse adjoint.
