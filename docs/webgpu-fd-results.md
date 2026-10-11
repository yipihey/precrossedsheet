# Compensated WebGPU amplitude finite differences

Engine 0.5.0, recorded 2026-10-11. The opt-in GPU observer now evaluates the
centered initial-amplitude check at the existing `h=1e-4`. CPU remains the
default, and automatic selection stays CPU. The scientific scope remains the
established pre-shell-crossing sheet; crossed diagnostic outputs still carry
a caution.

## Numerical contract

Rust retains the actual LPT geometries at `A+h` and `A-h`, with epoch,
cosmology and input modes fixed. Each displacement from the exact integer
lattice is packed as two f32 components, `hi=f32(x-q)` and
`lo=f32((x-q)-hi)`. The GPU uses the same periodic lifts, Kuhn connectivity
and fixed positive simplex masses as the CPU continuous P0 observer.

The power finite difference is

`Re[(F+ − F−) conj(F+ + F−)] / (2h)`.

Independent rounding of the two amplitudes can become significant after
subtraction and division by h. The GPU therefore accumulates their difference
through the integral of the exact continuous directional derivative along
the line joining the retained endpoints. Three-point Gauss–Legendre quadrature
uses nodes `0, ±sqrt(3/5)` and weights `8/9, 5/9, 5/9`. The direction is the
actual endpoint coordinate gap; no analytic LPT JVP is supplied. That JVP
remains the independent quantity checked by the lab.

For unit total mass, the conservative half-angle bound
`M = π maxModeL1 maxComponentEndpointGap` must be at most 0.005. The scalar
[Gauss remainder](https://dlmf.nist.gov/3.5.E19) gives a complex amplitude-delta
truncation bound `sqrt(2) M^7 / 15750`, and hence a power-FD truncation bound
`sqrt(2) M^7 / (15750 h)`. At the limit this is below 7.1e-17 in internal
units. This bound concerns quadrature truncation; it does not bound all GPU
roundoff. Requests outside this guard recover the CPU FD stage.

The shader uses compensated f32 high/low arithmetic with runtime rounding
barriers and Dekker split-product residuals. Reduced sine/cosine polynomials
have degrees 17/18. Repeated knots use normalized divided-difference series
through order 40 for angular spans below 3 radians, with up to five knots;
wider intervals use Newton recursion. The appendix derives the normalized
register recurrence and the endpoint-path identity. Endpoint and delta
calculations execute in separate completed submissions before 32-cell
fixed-owner storage reductions. Each stage waits for queue completion before
the next consumes its storage.
At 64³ with six observed modes, four endpoint input arrays occupy 16 MiB and
three interleaved complex per-cell channels another 72 MiB. Partials,
parameters and readback staging are included in the resource checks.

The independent GPU/CPU comparison target remains
`abs(FDgpu − FDcpu) / max(abs(FDcpu), 1e-9) <= 1e-5` in internal unit-box
units. The separate analytic derivative gate is unchanged:
`abs(dP/dA − FD) / max(abs(dP/dA), 1e-9) <= 1e-5`.
The finalizer applies physical volume once; the physical comparison floor is
therefore `1e-9 L^d`. A passing externally supplied FD is consumed directly,
without invoking the CPU FD observer.

Each acquired device must pass independent arithmetic and piecewise-linear
segment fixtures. Capability, phase-gap, allocation, validation, readback and
execution failures recover retained CPU endpoint integration. A failed
analytic gate after selective f64 LPT mode recovery also rechecks CPU FD
before requiring full direct-mode recovery. PM and passing direct channels
remain available. Execution metadata separately records FD precision, step,
validation, shader hashes, timings and recovery reason.

## Validation and measurements

The numerical suite includes 2D and 3D, EdS and ΛCDM, waves and filtered
Gaussian input, weak/zero/negative modes, repeated transverse knots,
translations, non-radix Lagrangian grids and a folded diagnostic continuation.
Forced failures cover validation, resource bounds, readback, nonfinite output
and device loss. Cancellation is triggered after an actual submitted endpoint
batch; a queued CPU replacement completes and shutdown is acknowledged.

The release benchmark passed all 50 rows. Across five complete numerical
requests, the largest weak-scaled GPU/CPU FD discrepancy was 1.54e-7; every
finalized analytic derivative passed its separate 1e-5 gate. Forty frozen-input
runs (twenty waves and twenty Gaussian) passed all comparisons and the 1e-12
complex-component repeat budget. On this run their output hashes were also
identical within each case; the enforced contract remains the component budget.
The existing observer separately passed 34 independent kernel and 47
complete-request/worker checks. The resident FFT/PM regression passed all
131 checks on the same release binary.

Measurements used an Apple M5 Max with 128 GiB RAM, macOS arm64 and Chrome
155.0.8059.40 with the Metal 3 adapter. Normal headless policy was used, without
unsafe WebGPU or disabled-sandbox flags. These are wall times around completed
work, rather than GPU timestamp queries.

| Lattice | CPU f64 FD median (range), ms | Compensated GPU FD median (range), ms | Median ratio |
|---|---:|---:|---:|
| 32³ | 956.5 (953.2–978.5) | 278.1 (278.0–279.4) | 3.44× |
| 64³ | 6,474.8 (6,377.5–6,476.9) | 2,112.3 (2,108.0–2,120.5) | 3.07× |

The complete matched 32³ request had a median of **4,214.8 ms**
(4,212.0–4,244.1) with CPU FD, versus **3,524.9 ms** (3,515.3–3,557.8)
with GPU FD. This is 16.37% less elapsed time, or a 1.196× median ratio.
Both policies retained GPU PM/direct observations and the same selective
CPU LPT recovery. Stage acceleration therefore has a smaller effect on the
whole request. These are measured request/device results, not a scaling
promise for other hardware or input fields.

The CPU reference series was also corrected: a zero odd coefficient in a
symmetric knot set cannot justify dropping subsequent even coefficients.
The reference now uses a conservative factorial-tail bound and normalized
coefficients. Native independent integral-series regressions check symmetric
repeated knots and reject an unmet tail bound at the work cap.

The complete [JSON record](webgpu-fd-results.json) contains requests, all
comparison rows, repeated frozen inputs/output hashes, raw timing samples,
source and Wasm fingerprints, toolchain versions and device limits. Earlier
subtractive and combined-kernel experiments failed weak-mode or repetition
checks; they are excluded from the release timing record. The final split
passes and stable endpoint delta are covered by that record. Only the listed
device has been measured; a portable f64 or bitwise accuracy guarantee is not
asserted.

## Timing scope and reproduction

The isolated FD benchmark starts with fresh prepared Rust requests at 32³ and
64³. CPU timing includes the uncached retained f64 endpoint observer. GPU
timing includes validation of input bounds, uploads, completed endpoint/delta
passes, reductions, cancellation yields and readback. Endpoint geometry
preparation/packing is outside the isolated GPU timer. The preceding
arithmetic and complete-request fixtures warm the shader pipelines; there is
no separately discarded 64³ sample. Three CPU-first CPU/GPU pairs follow at
each grid.
Every timed GPU result is checked against the corresponding CPU result.

The complete-run comparison alternates three pairs through one warmed worker
at matched 32³ lattice/force/observation grids, 32 KDK steps and `a=0.25`.
Both policies use resident GPU PM, the GPU direct observer and the same
selective CPU LPT recovery; only the FD policy changes. Timings include
physical preparation, exact deposition, PDFs, transfers and finalization.
Rendering is excluded. Nested manifest timing fields overlap and should not
be summed to reconstruct elapsed time. CPU nLPT, exact observations and
weak-mode recovery remain substantial work.

```sh
make test
make test-wasm
node scripts/test_gpu.mjs --require-gpu --fd --output fd-checks.json
node scripts/test_gpu.mjs --require-gpu --fd-repeat --output fd-repeat.json
node scripts/test_gpu.mjs --require-gpu --fd-benchmark --output fd-benchmark.json
node scripts/test_gpu.mjs --require-gpu --full --output observer-checks.json
node scripts/test_gpu.mjs --require-gpu --resident --output resident-checks.json
make check
quarto render
```

The [GPU check page](https://yipihey.github.io/precrossedsheet/app/gpu-test.html?fd=1)
provides the same FD suite. Add `fd-benchmark=1` for timings or
`autorun=1&fd-repeat=1` for frozen-input repetitions. The CLI serves the local
repository with an isolated Chrome profile and normal WebGPU policy. A
missing adapter is a skip; `--require-gpu` makes it fail. Hosted CI runs the
Rust/Wasm/worker and publication checks; hardware GPU checks are recorded
separately.

The reusable entry point is `runFiniteDifference(device, inputs, options)` in
[finite-difference.js](../app/js/gpu/finite-difference.js). It accepts four
vec4 high/low endpoint displacement arrays, int4 mode records, dimension,
lattice resolution and the canonical amplitude step, and returns amplitudes,
their delta, power finite differences, diagnostics and provenance. Owned
buffers are destroyed after completed work, cancellation or failure. Rust's
`finish_with_finite_difference` validates and normalizes externally supplied
values while the original cached CPU interface remains available.
