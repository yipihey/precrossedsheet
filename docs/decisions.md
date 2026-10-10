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

## WebGPU investigation

A native stage profile isolates continuous simplex modes, analytic tangents and
finite-difference verification as the dominant group for the measured 3D wave
requests. The reproducible profiler instruments a scratch copy and records
complete requests and source hashes. This identifies a target for batched
simplex/mode evaluation with reproducible reduction; it is not a GPU speedup
measurement. The exact continuous integral, stable coincident-phase limits and
JVP are required for any future implementation. Portable WGSL precision differs
from the CPU f64 path, so numerical equivalence needs explicit tolerances and
analytic fixtures. Exact clipping and the Rust CPU oracle remain available.
The current release contains no GPU scientific compute backend.
