# Hybrid WebGPU observer: validation and measurements

This is the engine 0.3.0 observation-backend measurement record. Engine 0.4.0
also offers resident cosmological PM evolution; see
[its separate validation and timing record](webgpu-pm-results.md). Historical
numbers below retain their original preparation and timing scope.

Recorded 2026-10-10, engine 0.3.0 and browser state schema 3. The first backend
implements milestones M0–M4 of [the backend plan](webgpu-backend-plan.md).
It evaluates the continuous fixed-mass P0 sheet transform and analytic
position JVP in 2D and 3D. Initial conditions, growth, nLPT, cosmological PM,
orientation checks, exact deposition, PDFs and gridded spectra retain the
existing Rust/Wasm implementation and precision conventions. The smaller-scale
extension remains outside this work.

## Reproduce the checks

From the released checkout, with Rust, wasm-pack, Node and Chrome installed:

```sh
make test
make test-wasm
node scripts/test_gpu.mjs --require-gpu --full --output gpu-checks.json
node scripts/test_gpu.mjs --require-gpu --benchmark --output gpu-benchmark.json
node scripts/test_gpu.mjs --require-gpu --schedule-benchmark --output gpu-schedules.json
python3 scripts/profile_native_stages.py --dim 3 --repeats 3 --output native-profile.json
```

The browser runner uses an isolated Chrome profile and serves the repository
over loopback. It needs no npm packages. Set `CHROME_PATH` when Chrome is
elsewhere. It records browser, device limits, adapter information when
available, shader hashes, reduction layout, complete numerical requests,
errors and timing samples. It reports an unavailable adapter as a skip;
`--require-gpu` makes that a failure. Hosted CI checks native Rust, the built
Wasm boundary, URL migration and worker behavior; it cannot certify a GPU
adapter without a GPU-capable runner.

The interactive [GPU check page](https://yipihey.github.io/precrossedsheet/app/gpu-test.html)
exposes the independent fixtures. Append `?autorun=1&full=1` for complete
requests or `?autorun=1&benchmark=1` for repeated browser measurements.
`?autorun=1&schedule-benchmark=1` compares the two kernel layouts separately.

## Numerical contract

The CPU preparation retains f64 positions and directions and forms
displacements before casting to f32. The GPU reconstructs lattice coordinates
from integers. For every complex value and JVP component the fixture target is

\[
|c_{\rm GPU}-c_{\rm ref}|\le 10^{-6}+10^{-4}|c_{\rm ref}|.
\]

The independent reference uses converged f64 barycentric quadrature on the
exact packed f32 inputs. Complete requests also compare the full-f64 Rust
integral with the Rust integral on those quantized inputs. These comparisons
separate input quantization from shader and reduction error. The absolute
floor covers cancellation-dominated components; it does not establish uniform
relative precision for an arbitrarily weak Fourier mode.

The laboratory additionally retains its stricter amplitude-power derivative
check against centered CPU finite differences at `h=1e-4`. Its maximum relative
discrepancy must be at most `1e-5`. Specifically, the gate is
`|dp - dp_fd| / max(|dp|, 1e-9)`, where `dp` is the analytic power tangent from
the candidate channels and `dp_fd` is the CPU centered finite difference.
Both use the internal unit-volume box before multiplication by physical
volume `L^d`; the denominator floor is in those raw derivative units and the
ratio is dimensionless.
CPU finite differences use the value-only f64 integral and are cached in the
prepared run. Independently reduced GPU powers at `A±h` are not substituted.
If an LPT mode fails this gate, both its value and analytic direction are
recomputed from the retained CPU state. The result is labeled `hybrid`, with
the LPT and PM backend recorded for every integer mode. PM and passing LPT
channels remain GPU-derived and are checked against the f64 CPU difference.
For recovered modes, replacing the value and direction together avoids
forming a power tangent from a GPU value combined with a CPU direction.

The PM JVP is compared with the CPU analytic reference in the harness; this
release does not add a per-request PM finite-difference computation. A missing
adapter, failed device fixture, allocation refusal, shader/readback failure,
device loss or invalid channels triggers complete CPU direct-mode recovery.
No failure changes the requested resolution or physical experiment. The
shell-crossing caution is independent of backend selection and retains output.

## Fixture coverage

Both reduction layouts are tested on 17 independent configurations: smooth
2D/3D sheets, non-power-of-two and partially filled tiles, zero and negative
integer modes, rigid translations, large common translations, phase-branch
thresholds, partial and full collapse, orientation reversal, and wide phase
spreads. These give 34 primal/JVP comparisons. A separate translation identity
checks the signed derivative. The fixed reduction tree uses no floating-point
atomics or silently discarded contributions.

Nine complete requests span 2D lattices 16, 24, 32, 48 and 96; 3D lattices 16,
24 and 32; orders 1–4; EdS and flat ΛCDM; mixed waves and filtered Gaussian
fields; physical scale factors and the common crossing reference; and a
crossed diagnostic continuation. Requests use matched lattice/force/observation
grids and 32 PM steps. Full input dictionaries accompany the measurement
record. The shared preparation makes trajectories, deposited densities, PDFs,
orientation diagnostics and gridded statistics identical between the tested
CPU and GPU observations.

Actual worker tests force seven failure cases, check 1D CPU routing, the
CPU-only automatic policy, a GPU/hybrid request and acknowledged shutdown.
Separate worker lifecycle tests check queued requests, stale cancellation,
reuse and cleanup. These checks cover backend plumbing and numerical
equivalence for the stated fixtures; they do not establish new cosmological
4LPT accuracy claims.

All 34 independent comparisons and 53 complete-request/worker checks passed.
The largest absolute kernel fixture errors were `1.64e-7` for a primal
component and `2.31e-7` for a JVP component. In the nine complete requests,
the largest GPU-versus-quantized-reference component error was `1.09e-7`;
every combined GPU-versus-full-reference component also met the target.
Only two requests passed the derivative gate with entirely GPU LPT channels.
Seven used selective LPT recovery; all then passed, with maximum discrepancy
`7.15e-6`. The unrecovered GPU maximum was `7.32e-4`, which is why a primal/JVP
fixture pass alone does not certify the more stringent power derivative.

## Device-specific measurements

The measurement machine is an Apple M5 Max with 128 GiB RAM, macOS arm64,
Chrome 155.0.8059.40 and its Metal 3 WebGPU adapter. Chrome ran headless with
its normal GPU policy; no unsafe WebGPU or disabled-sandbox flags were used.
Timing uses CPU wall time around completed GPU submissions and mapped
readback, rather than GPU timestamp queries. Only this device has been measured
for this release, so `cpu` remains the default and `auto` remains on CPU.

The kernel-layout comparison uses prepared 3D wave sheets, six modes, one
warm-up and three measured repeats. Kernel wall times exclude physical
preparation, uploads, CPU reference work, CPU finite differences and rendering.

| Lattice | Chunk/mode median (range), ms | Cell-tile median (range), ms |
|---|---:|---:|
| 32³ | 18.3 (18.3–18.3) | 7.4 (7.3–8.1) |
| 64³ | 146.5 (143.2–148.2) | 54.3 (53.2–56.0) |

The 64-cell tile is the default. Its estimated owned GPU buffer allocation is
2,195,656 bytes at 32³ and 17,563,848 bytes at 64³, versus 2,883,784 and
23,068,872 bytes for the chunk layout. These estimates exclude the retained
Wasm state and browser overhead. A faster direct kernel does not imply the
same factor for the full experiment.

The measured full-request benchmark alternated CPU and GPU requests three
times through one persistent worker after Wasm/device validation. It used 3D
4LPT waves at `a=.25`, Ωm=.315, the parameters in the stored request, matched
32³ grids and 32 PM steps. Each sample repeated physical preparation and
included CPU finite differences, selective recovery, GPU transfers and result
structured cloning. Rendering and first-use device validation were excluded.

| Complete 32³ request | Median (range), s |
|---|---:|
| CPU reference | 11.769 (11.758–11.778) |
| Hybrid observer | 6.226 (6.197–6.237) |

This is a measured **1.89×** end-to-end reduction for that request and device.
The recovered LPT modes were integer harmonics 3–6 along the first axis.
The remaining cost is predominantly CPU recovery, finite differences and
physical preparation. Worker finalization timings include recovery; do not
add that nested timer a second time.

The same run also timed stages on a shared preparation, with three repeats
per lattice. The sums below are estimates of CPU/hybrid totals rather than
separately measured complete worker requests. They exclude worker messaging
and rendering, and use warmed GPU pipelines. Hybrid sums include the selective
CPU recovery and extra finalization that recovery entails.

| Lattice | CPU direct median, s | GPU direct incl. transfers median, ms | CPU FD median, s | CPU total estimate median (range), s | Hybrid estimate median (range), s |
|---|---:|---:|---:|---:|---:|
| 32³ | 8.509 | 24.1 | 1.478 | 11.533 (11.527–11.614) | 6.090 (6.085–6.107) |
| 64³ | 61.144 | 71.7 | 10.299 | 83.729 (83.439–84.561) | 44.560 (44.522–44.705) |

Kernel wall times differ between the isolated layout run and the longer
CPU/hybrid run; timings depend on the submission context and machine state.
The direct-kernel speedup therefore cannot be extrapolated to complete-run
throughput. The 64³ estimate suggests a reduction of about 1.88× for this
hybrid request, but a full 64³ worker timing was not measured here.

[The machine-readable record](webgpu-results.json) retains full inputs,
samples, per-mode recovery and shader/source fingerprints. Repeated device
validation error arrays are stored once and referenced; top-level duplicate
rows are omitted. Its precision legend distinguishes the inherited f32 LPT
fields/densities from f64 state and arithmetic. The source fingerprints refer
to the released working tree; the parent Git revision and metadata-only
post-measurement refinements are stated explicitly.

## Subsequent work

The current remaining cost includes CPU physical preparation, f64 finite
differences and selective weak-mode recovery. M5 should follow the measured
end-to-end breakdown. Hopf–Cole's FFT/LPT kernels and resource-management
patterns are reusable foundations; its heat-flow dynamics cannot replace
cosmological PM evolution. A resident FFT/LPT/PM implementation needs its own
force, trajectory, tangent, clock and convergence fixtures. Exact clipping,
reverse-mode adjoints, deposited-density shape derivatives and cosmological
parameter gradients remain separate tasks.
