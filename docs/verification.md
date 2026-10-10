# Verification record — 2026-10-10

## Native code and browser binary

- `make test`: 41 Rust tests passed, including inherited nLPT/geometry
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
- The laboratory was checked in its sheet, density, spectra, PDF, gradient,
  and published-replay views, including 3D rotation and URL controls.
  Cached view changes preserved the numerical result. The browser self-test
  passed. Representative displayed LPT and PM masses were
  0.9999999981373549; the maximum tested LPT power-tangent discrepancy was
  5.77e-8. No scientific output was produced for a rejected time request.
- The bounded scientific review checked clocks, Fourier/PDF normalization,
  and the scope of the differentiability claims. See `docs/review-findings.md`.

The live lab is a filtered EdS teaching experiment. Archived replay is an
existing filtered-cosmology result with explicitly labeled small crossing
contamination. No unqualified universal sub-percent accuracy, smaller-scale
extension, post-crossing solver, reverse adjoint, or cosmological-parameter
gradient is asserted.

## Deployment

Repository push, GitHub Actions, and the deployed Pages site are checked
separately from the local build; their final status is recorded below after
release verification.
