---
name: quadrant-report
description: Build or extend an interactive scientific report in the quadrant format (Quarto book + tested Rust/WASM core + standalone browser lab) with provenance links to code and tests, deep links into the lab, verified citations and a release checklist. Use when adding a chapter, an equation, a lab panel, a test, or when starting a report from the quadrant template.
---

# quadrant report

The format is specified in the quadrant README (https://github.com/yipihey/quadrant);
this skill is the working procedure. The reference implementation is the
Hopf–Cole Cosmology book (https://github.com/yipihey/hopf-cole-cosmology).

## Layout you can rely on

- `index.qmd`, `chapters/*.qmd` narrative; `chapters/references.bib`;
  appendices `chapters/notation.qmd`, `chapters/status.qmd`.
- `core/` the tested numerical core (Rust → WASM with `wasm-pack build core
  --target web --release --out-dir ../app/pkg`); tests in `core/tests/`;
  WASM exports in `core/src/wasm.rs`.
- `app/` the standalone lab and the chapter widgets: all state in the URL hash,
  schema in `app/js/state.js` (`[stateKey, hashKey, type, range]`; lists and
  vectors join with `_`, never commas); plot kit in `app/js/viz/` (interactive
  SVG `LinePlot`, colormaps, colorbar, downloads); widgets mount on
  `<div class="qd-widget" data-widget="name">` from `app/js/widgets.js`.
- `_extensions/yipihey/quadrant/` shortcodes `{{< src >}}`, `{{< test >}}`,
  `{{< lab >}}`; configured under `quadrant:` in `_quarto.yml` (crate, api,
  tests, app).
- `_brand.yml` palette → `app/css/brand.css` (generated); `scripts/`:
  `serve.py` (no-cache dev server), `build_info.py` (revision in the footer),
  `brand_to_css.py`, `check_links.py` (provenance check, also in CI).
- `make wasm api check site test`; `python3 scripts/serve.py 8790 _site`;
  `rsync -a app/ _site/app/` refreshes the app copy without re-rendering.

## Adding physics: derive, test, write, link

1. **Derive** the result and check it numerically in the core before any text
   exists (a test under `core/tests/`). The test must pin the claim you intend
   to make ("matches to 10⁻³", "mass conserved").
2. **Expose** what the lab needs as a WASM export; rebuild with `make wasm`.
3. **Write** the section: pedagogy first; equations labelled `{#eq-…}`;
   sections that will be linked get `{#sec-…}`. Keep code names out of prose.
   Attach `{{< src module::item >}}` to the sentence that introduces an
   equation the code evaluates (crate-root items need no module prefix;
   methods are `Type::method`), and `{{< test name "label" >}}` to every
   quantitative claim. Never glue a shortcode to a `@ref` or a citation
   bracket without a word in between.
4. **Cite** only after verifying the reference in the field's index (ADS,
   Crossref, …); keep volume, pages and DOI in the `.bib` entry.
5. **Link the lab**: a "Try it" callout ending in `{{< lab "hash" "label" >}}`
   whose hash reproduces the experiment, using the schema keys.
6. **Explain in the app**: a caption with a back-link to the section, an
   `aria-label` on every canvas, a row in the lab guide if there is one.
7. **Update the notation appendix** for any new symbol or convention, and
   `docs/decisions.md` for anything not derivable from the code.

## Adding a lab panel or plot

- `new LinePlot(host, {width, height})`, `setAxes`, `setSeries`,
  `setMarkers`, `draw()`; zoom, pan, lin/log, ranges, SVG and CSV export come
  for free. Keep plots theme-aware: `currentColor`, no hard-coded backgrounds.
- Every new control gets one short hash key in `state.js`; defaults are not
  written to the hash.
- Heavy work runs off the UI path; panels are marked stale while recomputing.
- GPU code needs a WASM or Canvas2D fallback and a self-test.

## Delegation

Delegate code with a brief that states the physics, the verified numbers, the
files to read and the exact browser checks to perform; keep derivations and
chapter text with the stronger model. Agents must not edit `.qmd` files,
`_quarto.yml` or `core/` unless the brief says so, and must report what they
could not verify.

## Release checklist (all must hold before "done")

1. `make test` green; every ✓ badge names a passing test.
2. `make check` → "provenance links: OK".
3. `quarto render` with no unresolved cross-reference or citation warning.
4. Rendered pages looked at in a browser: equation widths, callouts,
   provenance marks, lab buttons; the lab console has no new errors.
5. Commit with a message that names the physics, the lab change and the text
   change; push; CI green; the deployed site opens the lab.
6. Record conventions and decisions in `docs/decisions.md`.
