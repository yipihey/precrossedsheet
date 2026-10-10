# WebGPU backend plan and Hopf–Cole reuse audit

Implementation record: 2026-10-10. Milestones M0–M4 are implemented in the
current first-release hybrid path. GPU execution remains opt-in through the
`gpu` backend setting; `cpu` is the default and `auto` currently selects CPU.
M5 is future work. See the [GPU results and validation record](webgpu-results.md)
for measured outcomes; this plan makes no new benchmark claims.

## First-release behavior

The first release accelerates only the continuous fixed-mass P0 Fourier modes
and their analytic position JVPs in 2D and 3D. Rust/Wasm prepares the complete
physical run with the existing Rust precision conventions: initial conditions, cosmological growth, LPT and PM
trajectories and tangents, crossing diagnostics, exact clipped densities,
PDFs, and gridded spectra. It retains the f64 state for the reference route
and packs displacement and direction arrays for GPU input. WGSL evaluates the
LPT and PM mode/JVP channels; Rust finalizes powers, derivatives, physical
units, and the result manifest in f64.

The CPU finite-difference check stays f64 and uses the existing experiment
step. Its prepared values are cached lazily. After GPU finalization, the
worker checks the per-mode LPT tangent against that finite difference. If a
mode exceeds the existing relative-error gate of `1e-5`, only that LPT mode is
recomputed from retained f64 positions. If the check still fails, or required
validation or channels are missing, the worker reruns the direct channels on
the complete CPU reference. PM channels remain GPU-derived in the hybrid
case; they do not have this LPT finite-difference gate.

Every result records requested and executed backend, precision, stage
provenance, per-mode provenance, GPU attempt/fallback reason, device metadata,
self-check result, numerical diagnostics, and stage timings. A recovered run
is labeled `hybrid`, and an unsuccessful GPU attempt remains available in the
provenance record when the final result is CPU.

| Request policy | Current behavior |
|---|---|
| `cpu` | Default. Use the Rust/Wasm reference, including f64 continuous modes and tangents. |
| `auto` | Use the Rust/Wasm reference. Automatic GPU selection remains disabled pending a device/workload policy supported by [webgpu-results.md](webgpu-results.md). |
| `gpu` | Attempt the validated 2D/3D GPU direct-mode path. 1D, unavailable devices, failed self-checks, invalid output, device loss, or numerical-gate failure use the CPU route, with the reason recorded. |

The default GPU schedule is `cell-tile`: one workgroup handles one mode and a
64-cell tile, distributes cells across 64 lanes, and reduces the tile in a
fixed tree. `chunk-mode` remains available for self-check and comparison. The
old Hopf–Cole kernel's flattened mode/chunk mapping does not give its claimed
same-chunk lane reuse for the six-mode workload, so its schedule was not copied
as the default.

## Implemented milestones

| Milestone | Implemented behavior |
|---|---|
| M0: CPU baseline and fixtures | The prepared facade shares physical preparation across CPU and GPU observations. It exposes full f64 channels, arithmetic on the exact packed f32 GPU inputs, cached f64 finite differences, and selected-mode f64 recovery. The GPU harness separates input quantization from GPU arithmetic. |
| M1: observation boundary | `PreparedLabRun` provides typed mode/input access, the complete CPU direct reference, selective LPT mode recovery, and common f64 finalization. The existing `lab_run` path remains the CPU compatibility entry point. |
| M2: direct P0 kernel | A worker-safe WGSL kernel evaluates continuous P0 simplex modes in 2D/3D, supports integer modes and non-power-of-two grid sizes, batches work to bound dispatches, and reports numerical diagnostics rather than filtering contributions. |
| M3: analytic JVP and CPU FD | WGSL carries signed position directions through repeated-knot divided differences. The worker retains f64 CPU finite differences and selective per-mode LPT recovery, then falls back to full CPU if the gate is not met. |
| M4: worker integration and provenance | A persistent module worker owns a lazily acquired WebGPU device, validates it with independent fixtures, supports bounded batch cancellation and shutdown, and emits backend, per-mode, validation, timing, and fallback metadata. URL state carries `cpu`, `gpu`, or `auto`; version-2 links migrate to CPU. |

M0–M4 being implemented does not mean every adapter has passed or that GPU
execution is the default. The per-device self-check is labeled provisional in
run metadata; automatic policy stays on CPU until the GPU workload and browser
evidence meet the release criteria. Current limitations include f32 GPU input
and arithmetic, only 2D/3D direct modes, and CPU execution for trajectories,
clipping, PDFs, gridded spectra, finite differences, and finalization.

## Kernel contract and validation

For each Lagrangian simplex with fixed mass `M`, vertices `x_j`, direction
`u_j`, and mode `k`, define `s_j = -i k·x_j` and `ds_j = -i k·u_j`. The
continuous barycentric contribution and its JVP are

\[
F_T = M d!\,\exp[s_0,\ldots,s_d],\qquad
\partial_A F_T = M d!\sum_{j=0}^{d} ds_j\,
\exp[s_0,\ldots,s_d,s_j].
\]

No Eulerian-volume derivative or deposited-grid window enters. The GPU uses
the same interval/triangle/tetrahedron connectivity and periodic lifts as
Rust. It reduces the base exponential phase for `sin`/`cos`, but preserves
unwrapped phase differences in divided-difference denominators. The WGSL
retains the upstream local Taylor/Newton strategy, while the target kernel
adds explicit value/JVP channels, geometry packing, arbitrary integer modes,
diagnostic flags, and bounded cell-tile dispatch. No contribution is silently
discarded when a value is large or invalid.

The independent harness compares the GPU path with both the full-f64 Rust
reference and CPU arithmetic on the exact f32-packed inputs. It includes
analytic/simplex quadrature fixtures, periodic boundaries, collapsed and
reversed geometry, mode and translation checks, repeatability, both schedules,
real prepared requests, worker failure recovery, and the per-mode derivative
gate. A missing adapter is a skip in ordinary browser automation; it is not a
numerical pass. See [webgpu-results.md](webgpu-results.md) for recorded
validation and timings.

Finite differences remain on f64 CPU by default. GPU finite-difference parity
at `h=1e-4` is a separate experimental gate; this release does not use GPU
finite differences to validate the displayed tangent. The current error gate
is evaluated per mode. Weak LPT derivative modes can be selectively recovered
in f64; a remaining failure switches all direct channels to CPU. The GPU does
not add an exact-clipped-density shape derivative, reverse adjoint, or
cosmological-parameter derivative.

## Hopf–Cole provenance and reuse

The source audit used Hopf–Cole cosmology revision
[`3a3a13b9ba817c5b4f8f9f26314715845e13d4ad`](https://github.com/yipihey/hopf-cole-cosmology/tree/3a3a13b9ba817c5b4f8f9f26314715845e13d4ad).
The target's `app/js/gpu/reuse-manifest.json` records per-file hashes and
adaptation notes. Both projects are MIT licensed; substantial copied portions
retain the upstream copyright notice in `app/js/gpu/UPSTREAM-MIT-NOTICE.txt`.
The upstream `direct.js` supplied the local Taylor/Newton divided-difference
method and numerical design reference. Its P1 evaluator, random mode sampler,
and silent magnitude filters were not carried into the target kernel.

The upstream `util.js` and `reduce.js` supplied small patterns for pipeline
creation, buffer binding, dispatch and fixed tree reduction. The target's
worker compute context is independent of the upstream DOM/rendering module;
target buffers have per-run ownership and cleanup. The upstream
`viz/gpu.js` is recorded only as a probe/error-handling reference; no source
was copied from it. Hopf–Cole FFT/LPT, fixed-point CIC and exact clipping
remain references for M5, not part of the first-release backend. Its CIC
quantization/overflow policy and its FFT's 3D radix-2 restriction are not
equivalent to this project's CPU implementations.

The `quadrant` template contributes report and interface conventions only; it
has no additional compute backend. Keep source revision, path, hash, license
notice, and target adaptation in the reuse manifest whenever the copied
surface changes.

## Historical profiling evidence

The native profile in [stage-profile.json](stage-profile.json) was recorded on
2026-10-10 against the pre-backend target source at revision
[`44e42c15a30a258461f365be2e367ed265d64c79`](https://github.com/yipihey/precrossedsheet/tree/44e42c15a30a258461f365be2e367ed265d64c79).
It used one scratch-instrumented release run each for 3D wave requests at
32³ and 64³, without warm-up or repeated-run statistics. The combined direct
mode, tangent, and finite-difference stage was 93.0% and 92.8% of the summed
pre-serialization stage times. The source hashes, full requests, diagnostics,
and timings are retained in the profile file. These are historical CPU
profiling results; they are not GPU speedup results and do not quantify the
current hybrid backend. Current GPU measurements belong in
[webgpu-results.md](webgpu-results.md).

## M5: later resident compute work

M5 is not part of this first release. Profile each candidate against the
current hybrid backend before implementation. Potential areas are a resident
FFT/LPT pipeline, a matched cosmological PM force and tangent solver, and
geometric clipping. Each needs independent numerical tests and a complete-run
comparison with the f64 Rust oracle. PM must carry the canonical momentum and
Rust-computed cosmological kick/drift clocks, with matching deposit, Poisson
solve, gather, and JVP. Clipping must preserve positivity and mass accounting
on difficult degeneracies and cell-boundary cases. Reusing Hopf–Cole kernels
does not establish equivalence for these operations.
