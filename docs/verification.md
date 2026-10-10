# Verification record — 2026-10-10

## Cosmological laboratory revision

- `make test`: all 56 Rust tests and the JavaScript URL/state checks passed.
  New coverage includes smooth-spectrum normalization, Hermitian and shared
  Fourier coefficients under refinement, ΛCDM kick/drift clocks, growing-mode
  evolution, trajectory JVPs in all dimensions, and continuation after crossing.
- Complete numerical responses and browser manifests replay with round-trip
  floating-point precision. A physical epoch remains valid when the realization
  has no finite crossing reference; unsupported schemas are rejected.
- `make wasm` and `make test-wasm` passed the 1D–3D mass/derivative contracts
  and the post-crossing finite-output/conservation check.
- `make check` passed all source links, test badges and laboratory state keys.
- Native timings and measured peak RSS for six resolution fixtures are recorded
  in [benchmarks.md](benchmarks.md) and [benchmarks.json](benchmarks.json), with
  full inputs, hardware, toolchain and source/output fingerprints. They are
  aggregate native timings; they do not measure browser speed or identify a
  specific bottleneck.
- Browser checks verified manifest export/import, a 128² Gaussian ΛCDM run at
  fixed physical epoch, crossing cautions with retained sheet/PDF results, and
  full-field PDF quantiles and cumulants. At 390-pixel width, density canvases
  preserve the 960:560 intrinsic aspect ratio and the page has no horizontal
  overflow. Data maps are square within those canvases.

- Final Quarto rendering completed all 13 pages with no unresolved references
  or citations. Browser inspection found no KaTeX errors; at 390 pixels wide,
  equations scroll within their containers without page overflow. The 3D
  projection exposes rotation and labels display-only decimation; full arrays
  remain in the experiment. The worker self-test passed all seven checks.
- An isolated native stage profiler is provided in `scripts/profile_native_stages.py`.
  It leaves the core unchanged; its compiled 32³ and 64³ checks passed. The measured
  32³/64³ groups identify continuous Fourier modes/JVPs/finite-difference checks
  as the priority for GPU investigation. The assessment specifies exact
  integrals, precision, reductions, explicit tangents and a CPU fallback. No
  GPU compute backend is claimed.

The current live engine uses explicit flat matter–Λ cosmology and an
Eisenstein–Hu smooth no-wiggle initial spectrum. The controls distinguish
Lagrangian, force and observation grids, initial and observation smoothing,
physical scale factor and common crossing reference. Results continue after
orientation loss, with positive stream masses and explicit validity cautions.
This is a discrete diagnostic continuation, not a newly validated physical
post-crossing approximation. Differentiation holds cosmology and epoch fixed
and targets trajectories and continuous sheet modes.

The following sections record verification of the earlier published baseline.

## Baseline release (historical)

## Native code and browser binary

- `make test`: 42 Rust tests passed, including inherited nLPT/geometry
  regressions and new PM, arbitrary-IC JVP, Fourier-tangent, PDF, periodic
  mass, input-validation, and pre-crossing-guard tests. The URL-state codec
  checks passed. Rust documentation examples contain no runnable doctests.
- `make wasm`: release WebAssembly package built from the same Rust core.
- `make test-wasm`: 1D, 2D, and 3D requests passed mass conservation to
  1e-6, continuous-mode amplitude derivatives to 1e-5 relative to centered
  differences, and output-shape checks. Invalid time requests were rejected.
- `make check`: Rust API documentation built without warnings; all equation
  source links, named test badges, and laboratory hash keys resolved.

The new numerical tests independently check growing plane symmetry, PM step
convergence, analytic derivatives against centered differences, simplex
translation/coincident-phase limits, and a determinant that crosses zero
inside a drift despite positive endpoint determinants. They do not constitute
a new full-volume cosmological 4LPT-versus-N-body validation campaign.

An additional mixed-wave regression checks joint particle/force-grid
refinement at fixed epoch, with 16/32/64 per axis. Relative displacement
residuals are 0.192672/0.090324/0.036728; fundamental continuous-mode power
ratios P_PM/P_4LPT are 0.828302/0.948073/0.985744. This pins the reduction
of finite-resolution PM error in the teaching setup. It does not identify
4LPT as an exact solution or certify every harmonic. The default was changed
to matched 32-per-axis grids; the 3D browser tier uses matched 24-per-axis
grids when switching from that default.

## Archived evidence and figures

The published-results extraction checked every histogram normalization and
the tabulated quantile ratios against the independently committed native
summary at 1e-12. Source revision, paths, checksums, estimator conventions,
and unavailable tail probabilities are packaged with the arrays. Upstream
simulations were inspected, not rerun. See `docs/evidence-audit.md`.

`scripts/render_figures.py` generated the 1D–3D SVG/PNG figures from native
Rust laboratory output. Each figure was visually inspected. These are
illustrations of the finite represented sheet, not independent validation.

## Report and user interface

- `quarto render`: all 12 book pages rendered, with no unresolved citation
  or cross-reference warnings.
- Browser review checked rendered mathematics, figures, callouts, source
  links, and laboratory links. Narrow and desktop layouts were inspected.
- The final matched-grid 2D default and the 3D 24-per-axis transition both
  completed with positive resolved orientations in the browser. Narrow-screen
  formulas scroll within the reading column and equation numbers sit below
  the formula; the introduction uses consistent chapter numbering.
- The laboratory was checked in its sheet, density, spectra, PDF, gradient,
  and published-replay views, including 3D rotation and URL controls.
  Cached view changes preserved the numerical result. The browser self-test
  passed. Representative displayed LPT and PM masses were
  0.9999999981373549; the maximum tested LPT power-tangent discrepancy was
  5.77e-8. No scientific output was produced for a rejected time request.
- The bounded scientific review checked clocks, Fourier/PDF normalization,
  and the scope of the differentiability claims. See `docs/review-findings.md`.

The baseline live lab was a filtered EdS teaching experiment. Archived replay is an
existing filtered-cosmology result with explicitly labeled small crossing
contamination. No unqualified universal sub-percent accuracy, smaller-scale
extension, post-crossing solver, reverse adjoint, or cosmological-parameter
gradient is asserted.

## Deployment

The initial implementation was pushed to `main` as `7cd778f21`. Its
[GitHub Actions publication](https://github.com/yipihey/precrossedsheet/actions/runs/38080773484)
passed native tests, Wasm checks, provenance checks, rendering, and Pages
deployment. The deployed book, live gradient view, archived PDF replay,
and self-test page were opened in the browser. The matched-grid default,
joint-refinement regression, and final layout/label fixes are included in
the follow-up release. CI and deployment status for each revision remain
available in [Actions](https://github.com/yipihey/precrossedsheet/actions).
