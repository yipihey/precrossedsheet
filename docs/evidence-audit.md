# Evidence and implementation audit

Audit date: 2026-10-10. This records the established filtered, predominantly
single-stream experiments. It does not propose an extension to smaller scales.

## Frozen sources and provenance

| Source | Inspected commit | Canonical report |
|---|---|---|
| `yipihey/hopf-cole-cosmology` | `3a3a13b9ba817c5b4f8f9f26314715845e13d4ad` | [Hopf–Cole Cosmology](https://yipihey.github.io/hopf-cole-cosmology/index.html) |
| `yipihey/DiscoDJ.jl` | `0db7baf609c7b0294d03e602e93492f128bd7f7d` | [LPT versus N-body report](https://yipihey.github.io/DiscoDJ.jl/studies/lpt_pdf/report.html) |

The local source clones were inspected at `/private/tmp/precrossedsheet-hopf`
and `/private/tmp/precrossedsheet-discodj`. No `AGENTS.md` was found inside
either clone or the target checkout. The public report fetches returned an
internal browser error, so substantive checks used the frozen repository
sources, their committed result files, and primary literature. Live report
URLs remain the canonical reading links; the commit permalinks freeze the
evidence actually audited.

## What the 4LPT comparison establishes

The experiment is a periodic 300 Mpc/h box with a 14 Mpc/h spherical top-hat
filter applied to the initial linear density. It uses Planck18 parameters,
an Eisenstein–Hu transfer function, fixed Fourier amplitudes, and a paired
phase check. The main PDF analysis uses a common 256³ Eulerian mesh and
top-hat smoothing radii 7, 14, 28, and 42 Mpc/h at redshift zero. Exact
Lambda-CDM growth is used through third order; fourth order has EdS time
dependence. The original NumPy PM reference uses 2LPT initial conditions
at z=24, a force mesh twice the particle resolution, and an fd4 gradient.
The sheet follow-up uses native BullFrog snapshots with 64 steps.

Sub-percent accuracy is a statement about specified statistics and
smoothing scales, not every aspect of pre-crossing dynamics. The main
report's 4LPT column gives the following fractional differences from the
N-body reference:

| Smoothing radius [Mpc/h] | Variance difference | Volume 0.1% / 99.9% density quantiles | Mass 99% / 99.9% density quantiles |
|---:|---:|---:|---:|
| 7 | −2.6% | +4.3% / −2.8% | −5.4% / −13.7% |
| 14 | −0.9% | +2.9% / −2.0% | −1.9% / −2.4% |
| 28 | −0.4% | largest reported quantile shift 0.5% | included in that maximum |
| 42 | −0.3% | largest reported quantile shift 0.1% | included in that maximum |

Numbers from the native sheet follow-up differ slightly because it uses
the native snapshots and a different estimator; its archived ratios are
the ones plotted in this site's published-results replay. The [original
study README](https://github.com/yipihey/DiscoDJ.jl/blob/0db7baf609c7b0294d03e602e93492f128bd7f7d/studies/lpt_pdf/README.md)
and [sheet follow-up](https://github.com/yipihey/DiscoDJ.jl/blob/0db7baf609c7b0294d03e602e93492f128bd7f7d/studies/lpt_pdf/sheet_pdf/README.md)
give the complete metric definitions and tables.

The archive is **predominantly**, not strictly, single stream. The original
N-body ever-crossed mass is 0.52% for fixed phases and 0.69% for the paired
run; the final inverted fraction is smaller. A tetrahedron is counted
as ever crossed after its orientation relative to its initial orientation
has become nonpositive at a sampled epoch. This is a finite-element and
finite-time diagnostic. PDFs are unconditional and include this mass.
The separate copula study conditions on elements single stream in both
models and must not be presented as an unconditional clustering result.
A filtered box with a small nonzero crossed fraction is useful evidence
for bulk agreement, but is not a validation of arbitrary initial spectra,
cosmologies, all tail statistics, or wholly uncrossed elements by themselves.

Resolution convergence is itself metric-dependent. The quoted N-body
uncertainty of roughly 0.5% concerns smoothed selected quantiles; the
crossed mass and unsmoothed dense tail remain more resolution-sensitive.
The 42 Mpc/h PDFs have limited independent volume in one box. A phase pair
tests realization sensitivity without being an ensemble uncertainty estimate.

## Exact deposition and the meaning of the PDFs

Six Kuhn tetrahedra fill each initial Lagrangian cube. With mean density
one each carries mass `Delta q^3/6`. The P0 representation has stream
density `m/abs(V)` within a displaced tetrahedron; Eulerian density is the
sum of all covering streams. Exact cell deposition adds
`m * overlap_volume / abs(V)` to each cell. Periodic geometry uses unwrapped
positions before folding deposited indices into the box. [Abel, Hahn and
Kaehler](https://arxiv.org/abs/1111.3944) establish the sheet interpretation;
[Powell and Abel](https://arxiv.org/abs/1412.4941) give analytic remeshing
integrals over intersections of polyhedra.

The actual helper is
[`sheet_density_exact`](https://github.com/yipihey/DiscoDJ.jl/blob/0db7baf609c7b0294d03e602e93492f128bd7f7d/studies/lpt_pdf/sheet_pdf/sheet_deposit.jl).
It deposits exact overlap volumes for nondegenerate P0 tetrahedra. If
`abs(V) < 1e-12 * Delta q^3`, it places all mass in the centroid cell;
this deliberate degenerate-element fallback should be stated when
calling the implementation exact. The follow-up reports mass conservation
to 1e-11 and tests shifted lattices, positivity, conservation, and the
point-sampling limit. Those upstream checks were inspected, not rerun here.

Volume PDFs give Eulerian cells equal weight. Gridded mass PDFs use each
cell's **unsmoothed deposited mass** as the weight while measuring the
smoothed density in that cell. Consequently one should not replace these
weights by the smoothed density or infer the plotted mass distribution by
multiplying the volume histogram by its bin centre. The gridless unsmoothed
equal-tetrahedron distribution is a PDF of **stream density**; it is not
the total Eulerian density in a multistream region. The smoothed gridless
variant samples the smoothed sheet field at tetrahedron centroids.

The sheet/CIC comparison agrees closely on the studied smoothing scales.
That estimator agreement validates the reported model ratios there; it
does not imply that all raw-cell PDF or power errors are below one percent.

## Power estimators: the error terms are distinct

The Hopf–Cole lab already implements exact clipped P0/P1 deposits,
simplex-integrated direct Fourier coefficients, and a spectrally refined
Lagrangian quadrature evaluated by a NUFFT. The exact deposit is exact for
the chosen finite-element density, while the direct spectrum integrates
that same density without an Eulerian cell grid. The NUFFT route targets
the smooth interpolated displacement map with a convergence parameter.
These are useful mutually checkable estimators; the direct/NUFFT comparison
needs convergence in the Lagrangian representation as well as FFT settings.

Exact finite-volume deposition removes Monte Carlo or centroid sampling
of the simplex mass. It leaves finite sheet-resolution error, a cell-average
window, Fourier aliasing, and any discreteness already present in N-body
trajectories. Cell averaging has amplitude window
`product_a sinc(k_a Delta x/2)` and the square of that window in power.
CIC uses its own assignment window. Deconvolution corrects a window but
does not generally undo aliased images. Near Nyquist it amplifies errors.
The familiar Poisson point-particle floor `V/Np` is not automatically the
appropriate subtraction for lattice vertices or a deterministic sheet.
See [Jing 2005](https://doi.org/10.1086/427087) and [Sefusatti et al.
2016](https://arxiv.org/abs/1512.07295) for FFT-window and alias analysis.
The previous lab's facet-tail scaling is a property measured for its
finite-element examples; it should not be used as a universal statement
about every smooth pre-crossing cosmological spectrum.

## Existing Julia APIs and differentiation boundaries

Paths below are relative to `native/DiscoDJNative/src/` at the frozen
DiscoDJ commit.

| Capability | API / source | Scope |
|---|---|---|
| General nLPT | `lpt/nlpt_core.jl`: `nlpt_kernels(res, boxsize)`, `compute_core(fphi, K; n_order, mode=:fast)` | EdS growth factors absorbed in each order, longitudinal and transverse terms, de-aliased products; fast and lean engines |
| Exact growth through 3rd order | `compute_core_exact(fphi, K; n_order=3)`; `evaluate_core(shapes, cosmo, a; n_order, exact_growth)` | Separate growth-free shapes through third order; not a complete exact-growth 4LPT implementation |
| One-shot displacement | `lpt_displacement(fphi, K, cosmo, a; n_order, exact_growth)` | Returns `(N,N,N,3)` displacements |
| N-body initial conditions | `nbody/run_nbody.jl`: `nbody_ics_lpt(cosmo, shapes, a; n_order, exact_growth=false)` | Returns displacement and `dPsi/dD`; exact-growth branch limited to order 3 |
| Native PM N-body | `run_nbody(cosmo, psi_ini, pi_ini; boxsize, a_ini, a_end, n_steps, res_pm, ...)` | BullFrog, FastPM or symplectic steppers; wrapped positions or unwrapped displacement; momentum; optional per-step callback |
| Force solver | `nbody/pm.jl`: `PMConfig`, `PMSolver`, `pm_acceleration!` | Assignment orders, fd gradient, optional interlacing and resampling; reusable mutable buffers |
| Periodic sheet | `field/sheet_periodic.jl`: `sheet_elements_periodic`, `sheet_query_periodic`, `sheet_mesh_periodic`, `refine_displacement` | Elements, signed volumes, point location and stream counts; mesh sampling differs from exact R3D overlaps |
| Differentiable CIC | `field/sheet_deposit.jl`: `cic_deposit(pos, w, res, boxsize)` | Hand-written ChainRules VJP for positions and weights |
| Differentiable sampled sheet | `sheet_deposit(x_grid, w_grid, res, boxsize; n_sub=1)` | Barycentric tetrahedron samples passed through CIC; composition is differentiable, but not exact overlap deposition |
| Exact study sheet | `studies/lpt_pdf/sheet_pdf/sheet_deposit.jl`: `sheet_density_exact(psi, L, ng)` | R3D P0 overlap deposition; separate from the sampled AD API |

The fast nLPT sources have hand-written ChainRules adjoints. The reported
field adjoints agree with the lean path and finite differences. However,
`evaluate_core` explicitly excludes growth-table interpolation from the
AD tape. Those derivatives apply to IC/potential fields at fixed cosmology
and epoch; they do not certify cosmological parameter derivatives.
The mutating native `run_nbody` code explicitly says the upstream custom-VJP
adjoint scan was not ported. A Rust PM/JVP implementation therefore needs
its own clearly stated differentiated parameters and numerical checks.
Do not treat the existing native forward PM solver, sampled AD sheet,
and exact study R3D estimator as one end-to-end exact differentiable API.

The upstream differentiable JAX solver is described in [List, Hahn, Flöss
and Winkler, DISCO-DJ II](https://arxiv.org/abs/2510.05206), published as
JCAP 06 (2026) 033. [Rampf, List and Hahn's BullFrog paper](https://arxiv.org/abs/2409.19049)
describes the time integrator. These papers establish upstream algorithms,
not parity of a new Rust implementation.

## Mathematical phrasing to preserve

Shell crossing occurs when the map ceases to be one-to-one locally and
its Jacobian vanishes; after crossing the sheet still has meaningful
connectivity, while a single-stream gravitational fluid equation no longer
describes the full collisionless dynamics. This physical boundary need
not coincide with the temporal Taylor series' mathematical radius of
convergence. [Rampf and Hahn 2021](https://arxiv.org/abs/2010.12584) find
convergence-limiting singularities later than first crossing. Thus the
phrase “LPT necessarily diverges at first shell crossing” is unsupported.
[Matsubara 2015](https://arxiv.org/abs/1505.01481) supplies general recursion
relations; [Zheligovsky and Frisch 2014](https://doi.org/10.1017/jfm.2014.221)
is a time-analyticity result for ideal-fluid trajectories and should not
be cited as a universal numerical 4LPT accuracy theorem.

## Replay datasets and economical checks

`app/data/published-results.json` contains twenty 256³ archived cases:
five models, two estimators, and fixed/paired phases. Each case has the
four established smoothing radii and both PDF weightings. Source paths,
SHA256 hashes, the full upstream Git SHA, conditions, restrictions, and
field meanings travel with the data. The histograms come from
`studies/lpt_pdf/sheet_pdf/results/stats.json`; model/reference ratios can
be independently checked against that directory's `summary.json`.
The original NPZ metadata supply crossing histories, explicitly labelled
as original-study measurements rather than native follow-up histories.

Histogram ordinate is probability density per `log10(rho)` over the
displayed range `rho=0.05..20`. Upstream `density=True` renormalizes within
that range. Its unrecorded out-of-range probability is deliberately
`null`; the archive must not infer a tail fraction from this ordinate.
Full-field density quantiles are preserved separately. Binned ratios
should hide bins where the reference PDF is zero. The committed original
NPZs also contain moments and PDF arrays; no full displacement `.npy`
snapshots were found in the checkout. The snapshot regeneration recipe
is in the upstream study rather than the browser replay.

Reproduce extraction, using a NumPy-capable Python environment:

```sh
python3 scripts/extract_published.py /path/to/DiscoDJ.jl
```

The script requires the exact audited upstream commit. It checks every
histogram integral to 1e-10 and all fixed-phase quantile ratios against
the separately committed summary to 1e-12, forbids non-finite JSON values,
and writes deterministic output. Extraction was run successfully during
this audit. It did not rerun any cosmological simulation. Literature
metadata for the accompanying bibliography were verified with SciX / NASA
ADS; core physical claims were checked against primary paper abstracts.
