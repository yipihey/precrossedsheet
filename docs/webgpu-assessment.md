# WebGPU acceleration assessment

The first hybrid WebGPU backend is implemented for continuous direct P0 sheet
modes and analytic position JVPs in 2D and 3D. It keeps physical preparation,
finite-difference checks, and finalization on the f64 Rust/Wasm path. GPU use is
explicit (`backend=gpu`); CPU remains the default, and `auto` currently stays
on CPU. The implementation plan and source reuse record are in
[webgpu-backend-plan.md](webgpu-backend-plan.md); measured results and
validation are recorded in [webgpu-results.md](webgpu-results.md).

## Current capability and release policy

Rust prepares trajectories, tangents, crossing diagnostics, exact deposits,
PDFs and gridded spectra. It sends packed displacement/direction arrays and
integer modes to a worker-owned WebGPU device. WGSL evaluates continuous P0
Fourier values and analytic JVPs for LPT and PM. Rust computes the power and
power derivative in f64, with the cached f64 CPU finite difference retained as
the default validation check.

GPU output carries per-mode provenance. If an LPT mode fails the existing
`1e-5` relative finite-difference gate, the worker recomputes that mode with
the retained f64 CPU reference. A failed recovery or invalid/missing GPU
output falls back to the complete CPU direct-mode result. The PM tangent has no
separate finite-difference route in this release. The worker runs device
fixtures before use and records validation status, device details, diagnostics,
timings and fallback reasons with each result. This per-device check is
provisional evidence; it does not enable `auto` selection.

The default kernel schedule is a 64-cell tile per mode, with a fixed local and
global reduction order. A chunk-per-mode schedule remains available for
comparison. The backend supports non-power-of-two 2D/3D grids. 1D and every
other scientific stage remain on CPU. GPU inputs and shader arithmetic are
f32; finalization, finite differences and CPU recovery are f64. Exact-clipped
density shape derivatives, reverse-mode adjoints, and cosmological-parameter
derivatives remain outside the backend.

Do not infer performance from the earlier profile below. Current CPU/hybrid
comparisons, adapter information, workloads and limitations are reported in
[webgpu-results.md](webgpu-results.md). No speedup claim belongs here without
those end-to-end results and a named browser/device/workload.

## Historical native profile

This profile is retained as the reason the direct-mode path was investigated.
It was recorded on 2026-10-10 against the pre-backend source at target revision
[`44e42c15a30a258461f365be2e367ed265d64c79`](https://github.com/yipihey/precrossedsheet/tree/44e42c15a30a258461f365be2e367ed265d64c79).
It ran on an Apple M5 Max MacBook Pro, arm64, macOS 26.7, with Rust 1.98.1.
The native release profile used a scratch copy, one run per request, no warm-up
and no repeated-run estimate. The profile identifies its inputs and source
hashes in [stage-profile.json](stage-profile.json).

| Stage (ms) | 32³ waves | 64³ waves |
|---|---:|---:|
| Initial conditions and LPT | 84.897 | 586.275 |
| Crossing/epoch guard and PM initial state | 230.463 | 1,939.711 |
| PM evolution and tangent | 536.354 | 4,533.123 |
| Exact LPT and PM deposits | 285.607 | 2,322.964 |
| Smoothing, PDFs, summaries, CIC, spectra | 4.430 | 39.647 |
| Direct modes, tangents, and finite differences | 15,424.461 | 122,610.906 |
| Result assembly | 19.362 | 153.669 |
| Sum through result assembly | 16,585.574 | 132,186.295 |
| JSON serialization | 16.630 | 128.172 |
| External process wall time | 16.98 s | 132.60 s |

The combined direct-mode group was 93.0% and 92.8% of summed stage time for
those two runs. It combined continuous LPT and PM simplex transforms, analytic
tangents and finite-difference checks; it did not isolate those costs. The
profile is machine-specific, exploratory CPU evidence. It is not a GPU
benchmark or representative throughput result.

## Kernel constraints and later work

The first kernel preserves the continuous affine-simplex integral and its
coincident-phase limits. It does not substitute center samples or clipping.
The inherited direct kernel's threshold guards silently skipped some
contributions; the target kernel replaces those with explicit numerical
diagnostics and fallback. The target also packs displacement separately from
integer lattice coordinates and uses cell tiles for the six-mode workload,
where the upstream flattened workgroup mapping did not provide its documented
same-chunk lane reuse.

The GPU computes in f32, so CPU equality is not expected bit for bit. Validation
separates f32 input quantization from WGSL arithmetic and reduction error. CPU
finite differences remain the default check because subtraction at
`h=1e-4` can amplify independently reduced f32 errors. GPU finite-difference
parity is a separate experimental gate. Automatic GPU policy should remain on
CPU until repeated warm and end-to-end measurements, validation, and fallback
behavior justify a tested device/workload envelope.

FFT, resident LPT, cosmological PM evolution and tangent propagation, and
clipping are later M5 candidates. The Hopf–Cole project supplies reference
implementations and infrastructure patterns, but its f32 radix-2 FFT,
fixed-point CIC, LPT conventions and clipping paths are not validated
substitutes for this model. See [webgpu-backend-plan.md](webgpu-backend-plan.md)
for the scientific and implementation boundaries.
