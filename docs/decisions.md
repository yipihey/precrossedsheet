# Decisions and conventions

## 2026-10-10 — scope and lineage

Only the established filtered pre-crossing results are developed. The user
explicitly deferred smaller scales. New browser outputs require positive
resolved Jacobians and no simplex inversion throughout each discrete PM drift.
Published near-pre-crossing archives retain their nonzero crossing metadata.

The quadrant template is vendored from revision
`ec91a49a7659c2529fea2754fcace5a7908c57a9`; its provenance shortcodes, plot
library, state convention and CI form the canonical interface.
The Rust core originates from MIT-licensed Hopf–Cole cosmology revision
`3a3a13b9ba817c5b4f8f9f26314715845e13d4ad`. Copyright is retained and the
historical crate name hcc_core preserves inherited test/API compatibility.

## Numerical conventions

Live dynamics: EdS, H0=1, L=1, a=D. PM uses p=a^2 dx/dt and matched CIC
force deposit/gather. Exact P0 sheet deposition is used for both LPT and PM
observation fields; the CIC spectrum is explicitly a different estimator.
The common time is a fraction of the minimum estimated crossing time of
orders 1–4. Vertex Jacobians and signed simplex volume ratios are separate
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

## PM convergence and defaults

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
