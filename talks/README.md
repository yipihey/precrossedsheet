# Presentations & Audience Summaries (`talks/`)

This directory hosts interactive Quarto Reveal.js + React.js slide decks and structured presentation plans tailored to different audiences, venues, and talk lengths. Every presentation shares the live Rust/WebAssembly (`hcc_core`) + React.js interactive laboratory panels in [`talks/shared/react-lab-panels.js`](shared/react-lab-panels.js) and the literature/benchmark figure library in [`talks/assets/figures/`](assets/figures/).

## Presentation Tracks

| Track Directory | Target Audience | Format & Duration | Status |
| :--- | :--- | :--- | :--- |
| [`talks/eft-fli-40min/`](eft-fli-40min/) | **PT / EFTofLSS & Field-Level Inference Cosmologists** (`LEFTfield`, Shifted-EFT, `BORG`, `LULO`, `Manticore`) | **40-Minute Pedagogical Seminar** (22 Reveal.js slides + 4 live React/Wasm panels) | **Complete & Deployed** (`index.qmd`, `plan.md`) |
| [`talks/conference-15min/`](conference-15min/) | **General Survey & Cosmic Web Cosmologists** (DESI, Euclid, Rubin LSST, Roman, SPHEREx) | **15-Minute Contributed Conference Talk** (12 slides + 2 live React panels) | **Structured Plan & Track Slot** |
| [`talks/sbi-hmc-workshop/`](sbi-hmc-workshop/) | **Bayesian Inference, HMC & Simulation-Based Inference (SBI) Researchers** | **30-Minute Workshop Deep-Dive** (18 slides + JVP/VJP & likelihood panels) | **Structured Plan & Track Slot** |
| [`talks/computational-methods/`](computational-methods/) | **Computational Astrophysicists, Applied Mathematicians & GPU/Wasm Engineers** | **45-Minute Methods Seminar** (Kuhn triangulations, R3D clipping, Hermite–Genocchi divided differences, WebGPU) | **Structured Plan & Track Slot** |

## Shared Interactive Infrastructure

- [`talks/shared/react-lab-panels.js`](shared/react-lab-panels.js): Reusable React 18 ESM components backed by `../../app/pkg/hcc_core.js` (`estimator-density`, `simplex-spectra`, `lpt-convergence-variance-pdf`, and `bao-jvp-inference`).
- [`talks/shared/slides.css`](shared/slides.css): Shared academic typography, layout grids, and embedded React laboratory styles.
- [`scripts/render_talks.py`](../scripts/render_talks.py): Post-render hook invoked automatically by `quarto render` to build all `talks/*/index.qmd` Reveal.js decks into `_site/talks/*/index.html` and sync `talks/shared/` and `talks/assets/`.
