# Decisions and conventions

## 2026-10-10 — scope and lineage

Only the established filtered pre-crossing results are developed. The user
explicitly deferred smaller scales. The first release suppressed outputs after a resolved
inversion; the subsequent user-requested laboratory revision supersedes that policy
(see the revision entry below).
Published near-pre-crossing archives retain their nonzero crossing metadata.

The quadrant template is vendored from revision
`ec91a49a7659c2529fea2754fcace5a7908c57a9`; its provenance shortcodes, plot
library, state convention and CI form the canonical interface.
The Rust core originates from MIT-licensed Hopf–Cole cosmology revision
`3a3a13b9ba817c5b4f8f9f26314715845e13d4ad`. Copyright is retained and the
historical crate name hcc_core preserves inherited test/API compatibility.

## Numerical conventions

Live dynamics: flat matter–Λ background, H0=1 and unit internal box. Growth
uses the early-time D~a convention; D=a in EdS. PM uses p=a^2 dx/dt
with cosmological kick/drift integrals and matched CIC force deposit/gather. Exact P0 sheet deposition is used for both LPT and PM
observation fields; the CIC spectrum is explicitly a different estimator.
The output epoch is a physical scale factor or a fraction of the common
minimum reference growth threshold from orders 1–4. Vertex Jacobians and signed simplex volume ratios are separate
checks; they are not a continuum proof.

Gaussian observation smoothing is separate from the low-band input cutoff.
Hard PDF bins are per log10(rho), and smoothed mass PDFs weight by raw cell
mass. Their finite-range captured probabilities are not renormalized away.

Derivative demo: initial amplitude A at fixed output epoch. Continuous P0
Fourier modes supply the differentiable observable. Exact R3D clipping is
not advertised as differentiable. Native IC JVPs and PM phase-space tangents
are at fixed cosmology; no reverse adjoints or growth-parameter derivatives.

## Portable browser tier

Wasm reference arithmetic with Canvas2D field rendering and quadrant SVG
plots. Computation in a cancelable worker. No unvalidated WebGPU solver or
silent JavaScript replacement. 3D resource caps are explicit and serialized.

## Established-result audit

The original 4LPT study uses exact LCDM time dependence through third order
and EdS D1^4 at fourth order. Native exact-sheet follow-up arrays are replayed
separately from the original NumPy PM/CIC report. Original crossing metadata
is labeled as belonging to that original run. Archived PDF ordinates are
conditional on 0.05<=rho<=20 and their captured probability is unavailable;
local live histograms are not renormalized over their plotted range.

The inherited term enumerator was restricted to no transverse terms in 1D:
there is no curl in plane symmetry, so higher orders vanish and the exact
Zeldovich solution is recovered. The native/wasm all-dimensions facade tests
pin this behavior. PM checks polynomial determinant extrema during each drift,
not only endpoints; an interior-fold regression covers that distinction.

## Baseline PM convergence and defaults (historical)

A release sanity check found substantial finite-load errors when the force
grid was refined at fixed particle count. These were insensitive to doubling
the timestep count. An independent instantaneous-force probe and joint
particle/force refinement identified lattice/CIC response as the main cause.
The solver's equations and integrator were retained; no precision claim is
made for the small browser PM grid. The default 2D experiment now uses matched
32-per-axis particle and force grids. Switching that default to 3D uses the
matched 24-per-axis tier. Guided links specify matched grids explicitly.

The mixed-wave regression refines both grids at fixed a=0.7, early epoch 0.03,
and 128 steps, with counts 16/32/64. Displacement residuals decrease from
0.193 to 0.090 to 0.037 and fundamental P_PM/P_4LPT from 0.828 to 0.948 to 0.986.
These test numerical convergence relative to a fixed LPT approximation, not
exact equality to the fluid solution or uniform accuracy across harmonics.
The UI labels the worst first-four-bin FFT power ratio and relative position
RMS explicitly, expresses them as percentages, and explains their denominators.


## 2026-10-10 — scientific laboratory revision (engine 0.2.0, schema 2)

The user explicitly requested continued output after crossing, equal spatial axis
ratios, larger grids, realistic cosmology, more technical detail, and clean
reproduction. Crossed outputs now carry a caution: LPT is a formal extrapolation,
not newly validated post-crossing physics. PM reports its first crossed drift
bracket and continues. P0 observation deposits positive stream masses. Exactly
degenerate 2D/3D elements remain a limitation of the inherited clipping path;
mass diagnostics expose any loss. Collapsed 1D intervals deposit point mass.

The live default is flat LCDM with Omega_m=.315,Omega_b=.049,h=.674,ns=.965,
sigma8=.811. EH98 equations 26, 28–31 supply a smooth baryon-suppressed transfer,
not BAO or massive-neutrino precision. sigma8 normalizes the unfiltered z=0
continuum spectrum. The initial Gaussian filter and fixed radial integer-mode
cutoff are applied afterward. A mode-keyed RNG preserves coefficients under
lattice refinement; realizations are not rescaled to a chosen RMS. Lower-dimensional
fields select a translationally invariant subspace with P_dim=P3/L^(3-d).

Internally positions use a unit box; physical L is explicit in Mpc/h. Fourier
outputs use k in h/Mpc and power in (Mpc/h)^dim. The phase panel now shows dx/da,
not dx/dD. LPT velocities and PM canonical momenta use the same LCDM growth clock.
PM uses accurate cosmological KDK integrals and fixed-cosmology analytic JVPs.
The old EdS joint-refinement regression remains an EdS benchmark.

Grid ceilings are 1024 in 1D, 256² in 2D, 64³ lattice and 128³ meshes in 3D, subject
to a conservative 1 GiB allocation envelope. Invalid requests fail explicitly;
no dimension-dependent silent clamping occurs. Gaussian 4LPT requires n>8cutoff.
Wireframe decimation affects rendering only. Density axes use equal spatial scales
and correct row-major indexing; 3D slices fix the third coordinate.

Version 2 URLs record all defaults and preserve number precision. JSON manifests
record state, executed inputs, build provenance and outputs; the native facade
accepts the same manifest. Reproducing the recorded implementation requires its
Git revision; URLs alone cannot promise binary identity across future releases.
Native benchmark timing is reported separately from browser performance.

JSON input parsing enables serde_json float_roundtrip so a serialized f64 returns
the same binary input, complementing full-precision URL serialization. Native
manifest replay prefers the stored executed result request over UI state.

## WebGPU investigation (historical, before engine 0.3.0)

A native stage profile isolates continuous simplex modes, analytic tangents and
finite-difference verification as the dominant group for the measured 3D wave
requests. The reproducible profiler instruments a scratch copy and records
complete requests and source hashes. This identifies a target for batched
simplex/mode evaluation with reproducible reduction; it is not a GPU speedup
measurement. The exact continuous integral, stable coincident-phase limits and
JVP are required for any future implementation. Portable WGSL precision differs
from the CPU f64 path, so numerical equivalence needs explicit tolerances and
analytic fixtures. Exact clipping and the Rust CPU oracle remain available.
That pre-backend release contained no GPU scientific compute backend.

## 2026-10-10 — hybrid GPU observer (engine 0.3.0, schema 3)

The implemented M0–M4 backend accelerates continuous fixed-mass P0 modes and
their signed analytic JVPs in 2D/3D. Rust/Wasm retains initial conditions,
growth, nLPT/PM trajectories, orientation, exact deposits, PDFs and gridded
spectra. `PreparedLabRun` shares physical preparation, supplies packed f32
displacements formed from f64 positions, retains the f64 oracle and finalizes
physical units consistently. The original synchronous facade remains available.

WGSL adapts the MIT Hopf–Cole divided-difference and reduction foundations
with retained notice and pinned source hashes. It preserves periodic lifts,
coincident-phase limits and signed tangent terms, reports numerical failures,
and never skips contributions through an amplitude threshold. A 64-cell tile
and fixed reduction tree replace the upstream flattened schedule for the
six-mode workload. Both schedules remain independently checked.

The amplitude finite difference stays CPU f64 at the existing step and is
cached once per prepared run. Weak LPT modes that fail its 1e-5 gate recover
their value and tangent together on CPU; passing LPT modes and PM channels can
remain on GPU. Exported per-mode provenance identifies this hybrid result.
Capability, resource, validation and execution failures recover all direct
channels from the retained CPU context, without changing physical inputs.

CPU is the default and `auto` stays CPU pending a supported device/workload
policy. Explicit GPU requests validate the device against independent analytic
fixtures. The persistent worker owns its device, serializes generations,
cancels between bounded submissions, frees per-run buffers and acknowledges
shutdown. Synchronous CPU stages use worker recreation for cancellation.
Cached-result selection retires obsolete work so a stale result cannot replace it.

Schema 3 records backend policy separately from physical requests. Version-2
links migrate with unchanged numerical controls and CPU execution. Manifests
retain stage and mode precision, device limits, shader hashes, validation,
timings and fallback reasons. Cross-device GPU reproduction uses explicit
tolerances, rather than a bitwise guarantee. Scope and measurements are in
[webgpu-results.md](webgpu-results.md); M5 resident physics remains subsequent
work, with smaller scales still deferred.

## 2026-10-11 — resident cosmological PM (engine 0.4.0)

Resident GPU PM uses the existing canonical momentum, flat matter–Λ or EdS
force operator and Rust cosmological kick/drift schedule. The deferred boundary
retains f64 initial conditions and LPT observations, so PM execution can recover
without rebuilding the realization. CPU PM and the original synchronous API
remain available. GPU PM and GPU continuous observations recover independently;
manifests record both decisions. State schema 3 and physical request schema 2
remain compatible, while the engine version changes to 0.4.0.

The new FFT adapts the pinned MIT Hopf–Cole Stockham stages to 1D–3D
last-axis-fast storage. It is reusable independently of PM. Only radix-2 force
meshes use resident PM; supported non-power-of-two lattices do not imply a
radix-2 lattice requirement. Unsupported force meshes retain CPU PM with their
original size. GPU buffer limits and the per-run allocation budget are checked
explicitly. CPU remains the default and automatic policy remains CPU.

CIC sums use integer-atomic linked lists with one floating sum per mesh node.
This removes fixed-point quantization and competing floating writes, but list
insertion order can vary. Device and repeated-run comparisons therefore use
explicit tolerances. The PM force uses CIC; displayed density and PDFs keep
exact conservative sheet deposition in Rust.

The GPU evolves displacement ψ=x−q in f32, with subtraction before packing in
f64 and force-grid lattice addressing computed by integer quotient/remainder.
Absolute-position f32 storage passed primal tests but failed sparse tangent
budgets near CIC faces. The displacement representation preserves weak initial
motions; the derivative budgets are unchanged. The quantized-input f64 oracle
reconstructs q + f32(ψ) without rounding the sum prematurely. Absolute f32
positions are materialized for orientation diagnostics and exact observations.

Drift orientation uses the complete determinant polynomial with interior
stationary points, rather than endpoint checks alone. Regular reference
simplex determinants are analytic signed powers of boxSize/n. GPU candidates
at most 1e-3 or invalid arithmetic require Rust f64 checks of the evolved f32
endpoints. That threshold is a practical recheck policy, not a rigorous global
roundoff bound. Initially crossed and later crossed outputs continue with the
existing caution and a first-drift bracket.

Numerical and performance scope is recorded in [webgpu-pm-results.md](webgpu-pm-results.md).
PM timing is separate from full worker timing. CPU weak-mode recovery and
finite differences remain substantial work; this release does not imply that
all stages are GPU-resident or that every request is faster. GPU nLPT
construction, exact clipping, more precise continuous-mode arithmetic,
cosmological gradients and adjoints remain subsequent work. Smaller scales
and newly validated post-crossing physics remain deferred.


## 2026-10-11 — compensated GPU finite differences (engine 0.5.0)

The existing `h=1e-4` centered initial-amplitude check is a value-only continuous
sheet observer at two retained LPT geometries. It now has an optional GPU
path using high/low f32 lattice-relative displacement inputs and compensated
phase, divided-difference and reduction arithmetic. The few complex endpoint
amplitudes and their independently integrated delta return to the host for a
symmetric f64 power difference and the
existing physical normalization. No analytic tangent is substituted for an
endpoint value, and the `1e-5` derivative gate is unchanged.

The small endpoint difference is computed through a three-point Gauss integral
of the exact continuous simplex directional derivative along the line joining
the retained endpoints. Only their actual coordinate gaps enter; the LPT JVP
is kept independent. A conservative half-angle bound of 0.005 gives a
quadrature power-derivative truncation bound below 7.1e-17 at h=1e-4 for unit
total mass. This is separate from GPU roundoff and device validation. Requests
above this limit recover CPU FD. The appendix gives the identity and bound.

The Rust boundary accepts validated external FD values without rerunning the
CPU endpoint observer, while the original API and cached f64 oracle remain
available. Initial GPU comparisons exposed loss of compensation under shader
optimization and an unreliable fused-product residual. Runtime rounding
barriers and split-product compensation are part of the tested kernel, with
independent device checks required before enabling the stage.

FD, analytic direct modes and PM trajectories recover independently. A failed
FD stage uses retained CPU geometry; a failed derivative gate after f64 LPT
mode recovery rechecks CPU FD before requiring complete direct recovery.
Manifests record FD precision, hashes, checks, timings and recovery reasons.
The default and automatic policies remain CPU; shell-crossing cautions and
physical scope remain unchanged. See [webgpu-fd-results.md](webgpu-fd-results.md).

Independent endpoint fixtures also motivated constant-index sorting networks,
which avoid a dynamic insertion-sort underflow path for negative modes.
Reduced-phase sine/cosine polynomials retain orders 17/18, and phase spans
below 3 radians use 40 normalized local series terms held in registers.
Each cell writes its endpoint expansion to storage before separate fixed-owner
32-cell block and mode reductions. Evaluation and reduction stages submit
separately and wait for completion before storage is consumed. Frozen-input
wave and Gaussian repetitions also check the FD gate;
endpoint component repeat differences have a separate 1e-12 absolute budget.
Bitwise output hashes are diagnostic. These
settings target cancellation in weak finite differences; the gates are not
relaxed.

The CPU reference series previously stopped after a small individual term.
Symmetric knots can make an odd coefficient exactly zero while subsequent
even coefficients remain nonzero. The reference now truncates using a
conservative factorial-tail bound and normalized coefficient recurrence.
An unmet tail bound at the resource cap returns nonfinite arithmetic for
downstream rejection. Independent integral-series tests cover values and
centered derivatives of symmetric repeated knots.


## 2026-10-11 — complete GPU optimization paths (engine 0.6.0)

Spatial nLPT construction is a separate backend boundary: Rust supplies actual Fourier initial modes and raw term specifications; WGSL builds sources and spectral fields; Rust installs them and applies existing growth and EdS merging. The explicit CPU raw-field oracle is used by tests, never invisibly in a successful GPU construction. The flat payload is term-major, with per-cell displacement, row-major gradient and three source components. Malformed fields fail before seed consumption.

Exact GPU observation preserves geometric intersection and positive fixed simplex masses. Cell-local coordinates and an interior centroid for tetrahedron moment fans reduce cancellation. Per-cell Kahan floating sums replace upstream fixed-point accumulation; integer atomics only form contribution lists. Degeneracy, capacity, graph and nonfinite diagnostics are explicit. External laboratory fields require finite nonnegative entries and unit mean within 1e-6, without renormalization. CPU clipping can recover an observation while retaining successful PM and nLPT stages.

Compensated analytic recovery receives the actual amplitude direction, independently of FD endpoint geometries. Only failing ordinary modes need this observer; an unchanged per-mode derivative gate still controls CPU recovery. CPU JVP integrals reuse fixed knot buffers and divided-difference storage. FD specializes recurrence/sorting by knot count and reuses vertex projections; an inflated factorial-tail bound selects terms up to the prior safe cap of 40. Completed submission boundaries and reduction ownership remain unchanged.

The worker owns a bounded native session with one entry per level. Keys include physical initial modes/cosmology/filters, epoch/order, PM force mesh/steps, observation mesh, and each executed precision source. Observation smoothing does not invalidate unsmoothed fields. Prepared orientation diagnostics are retained with geometry. Combined initial/geometry/PM/raw-observation retention is limited to 256 MiB and all retained capacities enter the reported count. Failed input cannot replace a valid entry. Cache history is execution metadata rather than part of the physical request.

GPU startup checks and full-request comparisons precede performance claims. Benchmarks compare the GPU path with new stages disabled against the optimized path at the same request, and report cache-hot runs separately. Initial-condition generation, growth, PDFs, gridded FFTs and finalization remain CPU. Portable compensated arithmetic is not a native-f64 guarantee. Scientific scope stays with established pre-shell-crossing results; crossed outputs remain diagnostic continuations.

## 2026-10-12 — joint density PDFs and geometric two-point statistics

The new chapter follows the one-point PDF chapter. It defines the equal-time density correlation as a volume average at a fixed displacement in a periodic box normalized by its own mean density. The full joint volume PDF has one-point volume marginals; the correlation is one mixed moment, not a reconstruction of the full joint distribution. Endpoint mass reweighting uses the raw density, with `1+xi` required when both endpoints are mass weighted at a conditioned separation. These conditioned marginals generally differ from the one-point mass PDF. Smoothed environment marks retain raw endpoint mass weights; weighting by the smoothed densities is a separately defined measure.

For a nondegenerate pre-crossing P0 tessellation, periodic overlap volumes give exact joint volume weights and density products give mass-pair weights. Independent points in elements have normalized separation kernels given by overlap divided by both volumes. Centroid differences locate these distributions, but finite extent and shape are necessary for exact correlations. In 2D/3D, centroid and volume do not uniquely determine shape; the vertices or form factors retain that information. Same-element terms are retained as continuous density autocorrelation, without a Poisson subtraction. Euclidean centroid averages use consistent unwrapped lifts, whereas the pair kernel and radial bins use periodic displacement geometry.

The inline demonstration is a unit-circle P0 interval fixture, with four fixed equal masses and interpolated positive widths. Two arrangements preserve both one-point PDFs and alter the two-point correlation. Its separate query-string state avoids changing the cosmological laboratory's hash schema, and CSV export records inputs and weights. Independent Rust spatial quadrature and Fourier integrals check exact overlaps. This is geometric documentation and a reproducible analytic widget, not new small-scale validation or a general cosmological two-point panel. After crossing, the total projected density still has an element-pair correlation expansion, but individual stream-density marks do not give its Eulerian density PDF.

## 2026-10-13 — pedagogical introduction, BAO transfer and particle deposits (engine 0.7.0)

The user asked for an introduction that is interactive, rigorous and easy to
follow, centered on two points: how the AHK tessellation improves measurement
over CIC/TSC particle deposits, and that the pre-crossing map is one-to-one with
nLPT adequate for it, checkable in the lab including models with BAO wiggles.

Core: `cosmological::transfer_bao` implements the full Eisenstein & Hu (1998)
fit (eqs. 2–24, BAO and Silk damping); the request field `transfer` selects
`smooth` (default, unchanged) or `bao`. Manifests without the field are read as
smooth, so earlier links and manifests reproduce unchanged. `LinearSpectrum`
reports the filtered, growth-scaled input spectrum (`spectra.linear`,
`spectra.directLinear`) inside the radial cutoff only, evaluated at the bin
wavenumber. `sheet::tsc_density` deposits the LPT vertices as TSC points on the
same cell-centered lattice as CIC; both fields (`cicRho`, `tscRho`, smoothed
like the sheet maps), their spectra and their mass means are returned. No
shot-noise subtraction is applied anywhere; discreteness is displayed, not
corrected.

Lab: a transfer-function control, presets for 2D/3D BAO runs and for the
sheet-vs-CIC/TSC comparison, CIC/TSC maps with an empty-cell table in the
density view, and the linear input overlay in the power view. Schema stays at
version 3 because the new key has a default and is absent from old links.

Book: `index.qmd` is rewritten as a pedagogical introduction with an inline,
pure-JavaScript, exactly solvable 1D widget (`app/js/intro-sheet.js`, tested by
`scripts/test_intro_sheet.mjs`): a three-mode Zel'dovich map deposited as the
connected sheet, CIC and TSC on a finer mesh, with errors against the converged
map and the Poisson level. The 1D box, N, M and growth fraction live in the
page hash (`sheet=f_N_M`). Quantities quoted in the introduction come from the
native runner (requests in `scripts/render_intro_figures.py`), the archived
DiscoDJ.jl sheet study (its README at the cited commit), and the widget test.
Numbers were recomputed in this container; browser timings are not quoted.

## 2026-10-29 — draft EFT dual-smoothing counterterm laboratory

User request: a Quarto/laboratory draft studying how small-scale modes influence
large-scale power via the sound-speed counterterm, using 1–4LPT, two Gaussian
IC smoothings (∼1% inverted tetrahedra vs ∼⅓ the linear σ), fine-grid exact P1
deposits, and a fit to the usual −2 cₛ² k² P_lin LEFT/EFTofLSS operator.
Follow-up: fit and compare at the displacement/vertex level after Mirbabayi &
Zaldarriaga (arXiv:1511.01889), and treat the residual after the deterministic
counterterm as a stochastic ensemble with matched residual power.

Implementation:
- `core/src/eft.rs` fits cₛ², matches σ_IR/σ_UV, runs dual-smoothing P1 spectra,
  fits the leading displacement operator ψ_ct = −cₛ² k² ψ_IR on modes
  k≤k_max, and draws a Gaussian stochastic residual ensemble from leftover
  Δψ shell power.
- Default UV radius R_UV=5 Mpc/h on the 32³/64³ draft volume (seed 7) yields
  ∼1% inverted Kuhn tetrahedra at a=1; R_IR auto-matches continuum σ/3
  (≈17.2 Mpc/h). Density companion: P_UV ≈ P_IR − 2 cₛ² k² P_lin^IR.
  Displacement primary: ψ_UV ≈ ψ_IR + ψ_ct + ψ_stoch. No PM trajectory enters
  this path.
- Lab view `counterterms` (schema keys Ruv, Rir, kFit, mlv; Rir=-1 means auto)
  reports both density and vertex-level fits plus ensemble coverage.
  `mlv=1` scales IR δ₀ by σ_UV/σ_IR(native) before LPT so continuum linear
  variances match; density plots overlay P_lin UV/IR and (when mlv=0) the
  filter-corrected nonlinear ratio.
- Chapter `chapters/05-eft.qmd` is explicitly a draft pedagogical section with
  provenance tests; it does not claim a production LEFT calibration.
