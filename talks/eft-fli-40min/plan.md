# 40-Minute Pedagogical Talk Plan: The Pre-Shell-Crossed Phase-Space Sheet for Field-Level Inference

**Target Audience:** Cosmologists fluent in Eulerian/Lagrangian Perturbation Theory (EPT/LPT) and Effective Field Theory of LSS (EFTofLSS).  
**Format:** 40 minutes presentation (+ 5–10 minutes Q&A), 26 Quarto Reveal.js slides with **4 built-in React.js + WebAssembly interactive laboratory panels** running live inside the slide deck (`talks/eft-fli-40min/index.qmd`).  
**Multi-Presentation Repository Organization (`talks/` & `chapters/08-talks.qmd`):**  
The repository is organized so that multiple presentations for distinct audiences and durations share the same Rust/WebAssembly (`hcc_core`) engine and reusable React.js slide components (`talks/shared/react-lab-panels.js`):
- `talks/eft-fli-40min/` — **40-Minute Pedagogical Seminar (PT / EFTofLSS & Field-Level Inference Audience)** *(Active Slide Deck + Plan)*
- `talks/conference-15min/` — **15-Minute Contributed Conference Talk (LSS / Survey Cosmology Summary)** *(Planned Slot)*
- `talks/sbi-hmc-workshop/` — **30-Minute Inference Workshop Deep-Dive (HMC, SBI, Differentiable Forward Models)** *(Planned Slot)*
- `talks/computational-methods/` — **45-Minute Computational Physics Seminar (R3D Voxelization, Hermite–Genocchi Modes & WebGPU)** *(Planned Slot)*

**Core Thesis:**  
1. Cold dark matter is a **continuous 3D manifold in 6D phase space**—not a cloud of point particles, and not a single-valued Eulerian fluid field that needs to be Taylor-expanded in $\delta(\mathbf{x})$.  
2. When initial conditions are smoothed at the pre-shell-crossing scale ($J = \det(\mathbf{I} + \nabla_{\mathbf{q}}\boldsymbol{\Psi}) > 0$), **density is pure geometry ($1+\delta = 1/J$)** and **4LPT on a tetrahedral phase-space sheet** is a convergent, sub-percent, zero-shot-noise, end-to-end differentiable forward model ideally suited for **field-level cosmological inference**.

---

## High-Level Pacing & Narrative Arc (40 Minutes)

```mermaid
flowchart LR
    A["Act I (0–8m)<br/>The Forward-Modeling Dilemma<br/>Why EPT & Particles Struggle"] --> B["Act II (8–18m)<br/>Geometry of the Sheet<br/>P0/P1 R3D, HG Modes & React Demo 1"]
    B --> C["Act III (18–29m)<br/>Why 4LPT Works Before Crossing<br/>Trajectory×Estimator Matrix, PDFs & Tidal Law"]
    C --> D["Act IV (29–37m)<br/>Field-Level Inference Engine<br/>Analytic JVPs, BAO & React Demo 4"]
    D --> E["Act V (37–40m)<br/>Synthesis & Outlook<br/>Where Pre-Crossing Ends"]
```

| Act | Time | Mins | Slides | Core Pedagogical Goal & Built-In React.js Interactive Slide Panels |
| :--- | :---: | :---: | :---: | :--- |
| **I. Motivation & The PT/Particle Dilemma** | 00:00–08:00 | 8m | 1–5 | Frame the tension between Eulerian PT ($\delta \sim 1$ breakdown), `LEFTfield` grid reduction, and `BORG`/`Manticore` CIC particle noise. Introduce the 3D phase-space sheet and $R_f \leftrightarrow \Lambda$ dictionary. |
| **II. Density is Geometry & Exact Estimators** | 08:00–18:00 | 10m | 6–11 | Kuhn tetrahedra + R3D voxelization ($\mathrm{P}_0$ & $\mathrm{P}_1$ vertex-Jacobian interpolation) + 16-ray Brillouin-zone Hermite–Genocchi continuous Fourier modes ($P_{\rm shot} \equiv 0$). **React Panel 1 (Slide 8):** Live Coolwarm $\log_{10}(\rho/\bar\rho) \in [-1,+1]$ density maps ($\rho/\bar\rho=1$ neutral white) + **React Panel 2 (Slide 10):** Mesh-free simplex Fourier vs. gridded FFT & cosmic-variance-free transfer $P(k)/P_{\rm lin,realized}(k)$. |
| **III. Dynamics: 4LPT vs $N$-Body & Exact Laws** | 18:00–29:00 | 11m | 12–17 | Matsubara cofactor recurrence, Cauchy curl invariant, and **React Panel 3 (Slide 13):** Live `{1LPT, 2LPT, 3LPT, 4LPT, PM} × {P₀, P₁, NGP, CIC, TSC}` matrix comparing scale-dependent variance $\sigma^2(R)$ and volume/mass PDFs. Plus `DiscoDJ.jl` 3D PDFs, copulas, and the $1100\,(\text{Gpc}/h)^3$ test of $F_M(>\bar\rho) = \frac{1}{2} + A_1\sigma + A_3\sigma^3$. |
| **IV. Built for Field-Level Inference** | 29:00–37:00 | 8m | 18–21 | Deterministic likelihoods, precomputed spatial basis $\mathbf{S}^\tau(\mathbf{q})$, exact confluent divided-difference gradients ([`tangent.rs`](https://github.com/yipihey/precrossedsheet/blob/main/core/src/tangent.rs)), and **React Panel 4 (Slide 20):** Live BAO wiggle/no-wiggle phase-matched ratio & analytic JVP explorer. |
| **V. Boundaries & Take-Home Message** | 37:00–40:00 | 3m | 22–24 | Honest discussion of the shell-crossing boundary ($J \to 0$), connection to EFT counterterms vs Lagrangian UV regularization, and open-source tools. |

---

## Detailed Slide-by-Slide Plan

### Act I: The Forward-Modeling Dilemma in LSS (00:00 – 08:00 | 8 mins)

> [!TIP]
> **Pedagogical Hook for PT/EFT Cosmologists:** Start on their home turf. They know EPT/EFTofLSS struggles with large bulk displacements (requiring IR resummation) and that Taylor-expanding $\delta(\mathbf{x})$ breaks when $\delta \sim 1$—even in voids ($\delta \to -1$) where no shell crossing has occurred!

#### **Slide 1 (0:00–1:30): Title & The Central Question**
* **Title:** *The Pre-Shell-Crossed Phase-Space Sheet: Geometry, High-Order LPT, and Field-Level Inference*
* **Visual:** 2D Lagrangian-to-Eulerian triangulated sheet deformation ([`sheet-2d.svg`](https://github.com/yipihey/precrossedsheet/blob/main/chapters/figures/sheet-2d.svg)) alongside 3D Kuhn tetrahedra ([`sheet-3d.svg`](https://github.com/yipihey/precrossedsheet/blob/main/chapters/figures/sheet-3d.svg)).
* **Message:** Can we build a forward model for large-scale structure that is simultaneously **non-perturbative in density**, **convergent in dynamics**, **identically free of shot noise**, and **analytically differentiable**?

#### **Slide 2 (1:30–3:30): What Field-Level Inference Demands vs. What Existing EFT Forward Models Actually Look Like**
* **Context:** Moving from 2-point summary statistics ($P(k)$, $\xi(s)$) to field-level inference $\mathcal{P}(\delta_{\rm init}, \boldsymbol{\theta} \mid \mathbf{d}_{\rm obs})$ (HMC, SBI, `LEFTfield`, shifted-operator EFT).
* **Visual Comparison (see Gallery below):** Show the actual density slices and pipeline flowcharts from **`LEFTfield`** (Schmidt 2021; Kostić et al. 2023; Nguyen et al. 2024) and **Shifted-Operator EFT** (Schmittfull et al. 2019, 2021):
  1. **Heavy Eulerian Grid Reduction ($W_\Lambda$):** In `LEFTfield` (Kostić et al. 2023, `arXiv:2212.07875`), a $L = 2000\,h^{-1}\mathrm{Mpc}$ box is sharp-$k$ filtered at $\Lambda = 0.10\text{–}0.14\,h\,\mathrm{Mpc}^{-1}$ onto a **$64^3$ to $90^3$ voxel grid** ($\Delta x = 22.2\text{–}31.25\,h^{-1}\mathrm{Mpc}$ per voxel!) and smoothed further by a 5-pixel Gaussian kernel for display. The resulting density fluctuations span only $\delta \in [-0.3, +0.3]$ (or $[-0.4, +0.4]$)—**all cosmic web filaments, caustics, and deep voids ($\delta \to -0.9$) are completely washed out into linear-looking Gaussian blobs!**
  2. **Why `LEFTfield` & Shifted-EFT Must Smooth So Aggressively:** Look at the `LEFTfield` pipeline flowchart (`x = q + s -> CIC -> W_Lambda reduce grid`): even when using LPT displacements, existing EFT codes **treat fluid elements as discrete point particles deposited with CIC** (which injects Poisson shot noise and leaves empty void cells at higher resolution, as visibly seen in Schmittfull et al. 2021's $\delta_Z$ slice) and **expand the density/bias perturbatively** rather than evaluating the exact geometric sheet density $1+\delta = 1/J$ on connected tetrahedra!

---

## Visual Gallery: How Smoothed Are `LEFTfield` & EFT Forward Models? (Ready for Slides 2, 5 & 8)

````carousel
![LEFTfield 2LPT real-space slice at cutoff Lambda = 0.10 h/Mpc (Kostic, Nguyen, Schmidt & Reinecke 2023, JCAP, arXiv:2212.07875, Fig. 6a). Box L = 2000 Mpc/h on a 64^3 grid (voxel size 31.25 Mpc/h!), smoothed with a 5-px Gaussian. Notice the colorbar: delta only spans [-0.3, +0.3]—no filaments or deep voids survive.](/Users/tabel/.gemini/antigravity/brain/87f2e9c5-ea5d-4f41-8a14-e9f536481c66/leftfield_slice_L010.png)
<!-- slide -->
![LEFTfield 2LPT real-space slice at cutoff Lambda = 0.14 h/Mpc (Kostic et al. 2023, arXiv:2212.07875, Fig. 6b). Box L = 2000 Mpc/h on a 90^3 grid (voxel size 22.2 Mpc/h). Even at their higher cutoff, delta only spans [-0.4, +0.4].](/Users/tabel/.gemini/antigravity/brain/87f2e9c5-ea5d-4f41-8a14-e9f536481c66/leftfield_slice_L014.png)
<!-- slide -->
![LEFTfield pipeline architecture (Schmidt 2021, JCAP, arXiv:2009.14176, Fig. 1). Notice the middle step: LPT displacements x = q + s are deposited as discrete particles via CIC onto N_g^CIC, and then hit with an Eulerian sharp-k filter W_Lambda to reduce the grid to N_g^red!](/Users/tabel/.gemini/antigravity/brain/87f2e9c5-ea5d-4f41-8a14-e9f536481c66/leftfield_flowchart_2021.png)
<!-- slide -->
![LEFTfield FBI vs. SBI (P+B) pipeline (Nguyen, Schmidt, Tucci, Reinecke & Kostic 2024, PRL, arXiv:2403.03220, Fig. 1). Even when combining LEFTfield with neural SBI at k_max = 0.10-0.12 h/Mpc, the forward model relies on the heavily cutoff-smoothed Eulerian grid.](/Users/tabel/.gemini/antigravity/brain/87f2e9c5-ea5d-4f41-8a14-e9f536481c66/leftfield_fbi_sbi_diagram.png)
<!-- slide -->
![Shifted-operator EFT field-level slabs (Schmittfull, Simonovic, Assassi & Zaldarriaga 2019, PRD, arXiv:1811.10640, Fig. 1). Bottom panel shows how Standard Eulerian bias smears the cosmic web into 20 Mpc/h blobs; middle panel shifts operators by Zel'dovich particles.](/Users/tabel/.gemini/antigravity/brain/87f2e9c5-ea5d-4f41-8a14-e9f536481c66/schmittfull_slabs_M10pt8.png)
<!-- slide -->
![Zel'dovich particle-painted density slice delta_Z (2nd panel) in Schmittfull et al. (2021, JCAP, arXiv:2012.03334, Fig. 1). Because displaced points are painted onto the grid with CIC without a connected tetrahedral sheet, unsmoothed voids and filaments exhibit grainy particle-lattice discreteness.](/Users/tabel/.gemini/antigravity/brain/87f2e9c5-ea5d-4f41-8a14-e9f536481c66/schmittfull_cic_zeldovich_slice.png)
<!-- slide -->
![1-point PDF breakdown in shifted-operator EFT (Schmittfull et al. 2019, arXiv:1811.10640, Fig. 3). While smoothed at R = 10 Mpc/h (left) the shifted cubic EFT PDF matches the simulation, as R drops to 5, 2, and 1 Mpc/h (right), the polynomial EFT density model fails in both the low-density void cutoff and the high-density tail.](/Users/tabel/.gemini/antigravity/brain/87f2e9c5-ea5d-4f41-8a14-e9f536481c66/schmittfull_pdf_shifted.png)
<!-- slide -->
![Contrast with our Pre-Crossed Phase-Space Sheet (precrossedsheet, intro-estimators-p1.png): 64^2 Gaussian LCDM vertices at z=0 deposited on a 256^2 mesh via Exact Sheet (P0), Exact Sheet (P1 linear vertex-density interpolation), CIC, and TSC. While CIC/TSC dissolve into empty void cells (75% and 44% empty!) and a white shot-noise floor, both P0 and P1 sheets resolve crisp filaments and smooth voids with 0 empty cells and zero shot noise—and P1 further suppresses the high-k facet-discretization floor by another 5x-8x!](/Users/tabel/.gemini/antigravity/brain/87f2e9c5-ea5d-4f41-8a14-e9f536481c66/precrossedsheet_intro_estimators_p1.png)
````

### Quantitative Talking Points When Showing These `LEFTfield` / EFT Images

| Framework & Reference | Cutoff / Smoothing Scale | Grid & Voxel Size | Density Dynamic Range & Visual Appearance | Why They Had to Smooth / Where Our Pre-Crossed Sheet Wins |
| :--- | :--- | :--- | :--- | :--- |
| **`LEFTfield` HMC/FLI**<br/>(Kostić, Nguyen, Schmidt, Reinecke 2023, [`arXiv:2212.07875`](https://arxiv.org/abs/2212.07875), Figs. 1, 6, 16) | Sharp-$k$ $\Lambda = 0.10\,h\,\mathrm{Mpc}^{-1}$ (effective scale $\pi/\Lambda \approx 31\,h^{-1}\mathrm{Mpc}$) or $\Lambda = 0.14\,h\,\mathrm{Mpc}^{-1}$ ($22\,h^{-1}\mathrm{Mpc}$) + 5-px Gaussian | $L = 2000\,h^{-1}\mathrm{Mpc}$ on **$64^3$ or $90^3$ grid** ($\Delta x = 31.3$ or $22.2\,h^{-1}\mathrm{Mpc}$) | **$\delta \in [-0.3, +0.3]$** (at $\Lambda=0.10$) or **$[-0.4, +0.4]$** (at $\Lambda=0.14$). Looks like a quasi-linear Gaussian random field; **no cosmic web**. | Uses 2LPT + **CIC particle assignment** + perturbative bias expansion, then throws away all Eulerian modes $k > \Lambda$. On our **4LPT tetrahedral sheet**, even with $R_f = 14\,h^{-1}\mathrm{Mpc}$ ($\Lambda_{\rm eff} \sim 0.15\,h\,\mathrm{Mpc}^{-1}$), evaluating $1+\delta = 1/J$ via R3D on a fine Eulerian mesh preserves **sharp filaments ($1+\delta \sim 10$) and deep voids ($1+\delta \sim 0.15$)** with zero CIC shot noise! |
| **`LEFTfield` FLI vs. SBI**<br/>(Nguyen, Schmidt, Tucci et al. 2024, PRL, [`arXiv:2403.03220`](https://arxiv.org/abs/2403.03220); Tucci & Schmidt 2024, [`arXiv:2310.03741`](https://arxiv.org/abs/2310.03741)) | Sharp-$k$ or Cubic-Grid cutoff $\Lambda = 0.10\text{–}0.12\,h\,\mathrm{Mpc}^{-1}$ ($R_{\rm eff} \approx 26\text{–}31\,h^{-1}\mathrm{Mpc}$) | $L = 2000\,h^{-1}\mathrm{Mpc}$, $N_g^\Lambda \sim 48^3\text{–}64^3$ | Heavy coarse-graining required so the Gaussian EFT likelihood $\mathcal{N}(\delta_{\rm det,\Lambda}, \sigma_\epsilon^2)$ with white/quadratic noise $P_\epsilon(k) = P_{\epsilon,0}(1 + \sigma_{\epsilon,2}k^2)$ holds. | Even with this heavy smoothing, FLI beats $P+B$ SBI by $3.5\times\text{–}5.2\times$ on $\sigma_8$! Imagine how much *more* non-Gaussian cosmic-web information (void PDFs, filament copulas) becomes accessible when you replace CIC + grid reduction with the **exact zero-shot-noise tetrahedral sheet**! |
| **Shifted-Operator EFT**<br/>(Schmittfull, Simonović, Assassi, Zaldarriaga 2019, [`arXiv:1811.10640`](https://arxiv.org/abs/1811.10640), Figs. 1, 3; Schmittfull et al. 2021, [`arXiv:2012.03334`](https://arxiv.org/abs/2012.03334), Fig. 1) | Standard Eulerian bias smoothed at $\sim 20\,h^{-1}\mathrm{Mpc}$; shifted 1LPT/Zel'dovich painted with CIC | $L = 1500\,h^{-1}\mathrm{Mpc}$ | Standard Eulerian bias smears structures into $20\,h^{-1}\mathrm{Mpc}$ blobs; unsmoothed CIC-painted $\delta_Z$ shows **grainy particle-lattice noise in voids**; 1-pt PDF fails for $R \le 5\,h^{-1}\mathrm{Mpc}$. | Shifting operators by Zel'dovich particles and painting with CIC still suffers from (1) **1LPT tidal deficit** (missing the 2LPT–4LPT shear focusing and Cauchy curl terms) and (2) **CIC point-particle starvation in voids**. The 4LPT R3D sheet cures both simultaneously. |

---

## Visual Gallery 2: How Do `BORG`, `Manticore`, and Ludvig Doeser's Recent Work Compare?

````carousel
![BORG-PM Supergalactic plane evolution from z=35 to z=0 (Jasche & Lavaux 2019, A&A, arXiv:1806.11117, Fig. 6). Box L = 677.7 Mpc/h on a 256^3 grid (voxel size 2.65 Mpc/h, 10-step PM + CIC). Unlike LEFTfield, BORG resolves the cosmic web with density contrasts delta in [-1, +5] (clipped at 5), overlaid with 2M++ galaxies in red.](/Users/tabel/.gemini/antigravity/brain/87f2e9c5-ea5d-4f41-8a14-e9f536481c66/borg_pm_chrono_sg.png)
<!-- slide -->
![Ludvig Doeser et al. (2024, MNRAS, arXiv:2312.09271, Fig. 1): Comparing N-body ground truth (left), BORG-1LPT + CIC (2nd panel), BORG-EM (1LPT + 3D V-Net neural emulator + CIC, 3rd panel), and N-body resimulation (right) in a 250 Mpc/h box on a 128^3 grid (voxel size 1.95 Mpc/h). Notice how fuzzy 1LPT+CIC is—which motivated Doeser et al. to train a 3D CNN emulator—whereas 4LPT + exact R3D sheet achieves sharp filament focusing analytically!](/Users/tabel/.gemini/antigravity/brain/87f2e9c5-ea5d-4f41-8a14-e9f536481c66/doeser_borg_em_fig1.png)
<!-- slide -->
![Ludvig Doeser et al. (2024, arXiv:2312.09271, Fig. 3): BORG-EM posterior mean and standard deviation on a 128^3 grid (Delta x = 1.95 Mpc/h). Top row shows final z=0 density contrast spanning delta in [-1, +12]! Bottom row shows the inferred initial conditions smoothed at 4 Mpc/h.](/Users/tabel/.gemini/antigravity/brain/87f2e9c5-ea5d-4f41-8a14-e9f536481c66/doeser_borg_em_samples.png)
<!-- slide -->
![Ludvig Doeser et al. (2024, arXiv:2312.09271, Fig. 2): BORG-EM pipeline schematic. Notice Step 2 (1LPT displacements) -> Step 3 (3D V-Net CNN with 48-voxel periodic padding) -> Step 4 (CIC particle mass assignment onto the grid).](/Users/tabel/.gemini/antigravity/brain/87f2e9c5-ea5d-4f41-8a14-e9f536481c66/doeser_borg_em_schematic.png)
<!-- slide -->
![Ludvig Doeser, Metin Ata & Jens Jasche (2025, arXiv:2502.13243, LULO, Fig. 3): Gradient-free learned neural optimizer reconstructing initial conditions (smoothed at sigma = 10 Mpc/h, left) and z=0 dark matter overdensity delta_DM in [-1, +7] (2nd column) from non-differentiable AHF halo catalogs on a 128^3 grid (Delta x = 1.95 Mpc/h).](/Users/tabel/.gemini/antigravity/brain/87f2e9c5-ea5d-4f41-8a14-e9f536481c66/doeser_lulo_recon_viz.png)
<!-- slide -->
![Manticore-Local (McAlpine, Jasche, Ata, Lavaux et al. 2025, MNRAS, arXiv:2505.10682, Fig. B1): Posterior mean 1+<delta> (left), variance (middle), and SNR (right) of the BORG+COLA final density field on a 256^3 grid in a 1000 Mpc box (voxel size 3.9 Mpc = 2.66 Mpc/h). Even in the posterior mean, 1+<delta> spans [0.1, 10] (delta in [-0.9, +9])!](/Users/tabel/.gemini/antigravity/brain/87f2e9c5-ea5d-4f41-8a14-e9f536481c66/manticore_local_borg_density.png)
<!-- slide -->
![Manticore-Local Coma Cluster posterior ensemble (McAlpine et al. 2025, arXiv:2505.10682, Fig. 8): 50 SWIFT N-body posterior resimulations (1024^3 particles) seeded from the 256^3 BORG+COLA inferred initial conditions, resolving the cosmic-web filaments feeding the Coma Cluster.](/Users/tabel/.gemini/antigravity/brain/87f2e9c5-ea5d-4f41-8a14-e9f536481c66/manticore_local_coma.png)
<!-- slide -->
![Manticore-Deep (McAlpine, Jasche, Lavaux, Doeser & Loureiro 2026, arXiv:2606.10020, Fig. 2): Extending BORG+COLA field-level inference to a 4096 Mpc/h volume (effective 1024^3 grid, Delta x = 4 Mpc/h) across 27 overlapping 256^3 tiles (~30 million CPU hours), followed by N-body resimulations zooming from the BOSS Great Wall (500 Mpc/h) down to Coma, Hercules, and Shapley (50 Mpc/h).](/Users/tabel/.gemini/antigravity/brain/87f2e9c5-ea5d-4f41-8a14-e9f536481c66/manticore_deep_post_mean.png)
<!-- slide -->
![Ludvig Doeser et al. (2026, arXiv:2606.10023, Fig. 1): Benchmarking neural generative SBI models (Normalizing Flows GLOW and Stochastic Interpolants SI) against ground-truth BORG HMC in 32^3 dimensions (L = 1000 Mpc/h, Delta x = 31.25 Mpc/h, 1LPT forward model), showing that generative SBI struggles to calibrate pixel-wise posterior variance compared to HMC.](/Users/tabel/.gemini/antigravity/brain/87f2e9c5-ea5d-4f41-8a14-e9f536481c66/doeser_2026_hmc_vs_sbi_slice.png)
````

### How `BORG`, `Manticore`, and Doeser et al. Compare to `LEFTfield` — and Where Our Pre-Crossed Sheet Fits In

| Framework & Papers | Inference Grid & Voxel Size $\Delta x$ | Gravity & Density Forward Model | Density Contrast $\delta$ Reached | Key Bottlenecks That Our Pre-Crossed Sheet Solves |
| :--- | :--- | :--- | :--- | :--- |
| **`BORG-PM`**<br/>(Jasche & Lavaux 2019, [`arXiv:1806.11117`](https://arxiv.org/abs/1806.11117)) | $L = 677.7\,h^{-1}\mathrm{Mpc}$, $256^3$ grid ($\Delta x = 2.65\,h^{-1}\mathrm{Mpc}$) | 10-step PM + **CIC particle deposit** + power-law bias | **$\delta \in [-1, +5]$** on grid (up to $\delta \sim 20$ in clusters) | Evaluates 10 PM steps + adjoint per HMC leapfrog step; **CIC point-particle assignment** starves voids ($\rho = 0$ cells) and injects shot noise. |
| **`BORG-EM`**<br/>(Doeser, Jamieson, Stopyra, Lavaux, Leclercq, Jasche 2024, [`arXiv:2312.09271`](https://arxiv.org/abs/2312.09271)) | $L = 250\,h^{-1}\mathrm{Mpc}$, $128^3$ grid ($\Delta x = 1.95\,h^{-1}\mathrm{Mpc}$, $k_{\rm Nyq} = 1.61\,h\,\mathrm{Mpc}^{-1}$) | `1LPT` $\to$ **3D V-Net CNN emulator** (padded to $224^3$) $\to$ **CIC particle deposit** | **$\delta \in [-1, +12]$** on the $1.95\,h^{-1}\mathrm{Mpc}$ grid | Needed a 3D CNN (`map2map`) because **1LPT + CIC** had blurry filaments! Still uses **CIC particle assignment**, requires 48-voxel padding, and needs a **multi-scale pooled likelihood** so small-scale emulator misspecification doesn't bias large-scale $P(k)$. |
| **`LULO`**<br/>(Doeser, Ata, Jasche 2025, [`arXiv:2502.13243`](https://arxiv.org/abs/2502.13243)) | $L = 250\,h^{-1}\mathrm{Mpc}$, $128^3$ grid ($\Delta x = 1.95\,h^{-1}\mathrm{Mpc}$) | `Gadget-IV` TreePM + non-differentiable `AHF` halos + **V-Net learned optimizer** | **$\delta_{\rm DM} \in [-1, +7]$** (MAP optimization only, no posterior) | Shows a profound physical result: even when fitting non-linear $z=0$ halos at $k \sim 2\,h\,\mathrm{Mpc}^{-1}$, **the initial conditions $\mathbf{x}$ are only correlated ($>80\%$) at $k \le 0.23\text{–}0.33\,h\,\mathrm{Mpc}^{-1}$ ($R_f \sim 6\text{–}10\,h^{-1}\mathrm{Mpc}$)**—precisely the pre-shell-crossed regime! |
| **`Manticore-Local` (I) & `Manticore-Deep` (II)**<br/>(McAlpine, Jasche, Lavaux, Doeser et al. 2025, [`arXiv:2505.10682`](https://arxiv.org/abs/2505.10682); 2026, [`arXiv:2606.10020`](https://arxiv.org/abs/2606.10020)) | **I:** $681\,h^{-1}\mathrm{Mpc}$ on $256^3$ ($\Delta x = 2.66\,h^{-1}\mathrm{Mpc}$)<br/>**II:** $4096\,h^{-1}\mathrm{Mpc}$ across $64\times 256^3$ tiles ($\Delta x = 4\,h^{-1}\mathrm{Mpc}$) | **In HMC:** 20-step `COLA` + **CIC** + 5-param STDP bias.<br/>**Post-HMC:** $1024^3$ `SWIFT` $N$-body resimulations | **$\delta \in [-0.9, +9]$** in HMC posterior mean; full halo contrast in `SWIFT` resimulations | 1. **30 million CPU hours** for `Manticore-Deep` because `COLA` runs inside every HMC step.<br/>2. **CIC void starvation** hurts their $\ln(\rho)$ sigmoid bias in voids.<br/>3. Had to add **ad-hoc priors on the initial $P(k)$ shells and final-field mean/variance/skewness** to stop HMC from distorting the white-noise field to absorb small-scale model errors! |

#### **Slide 3 (3:30–5:30): Cold Dark Matter is a 3D Submanifold in 6D Phase Space**
* **Theory ([`01-sheet.qmd`](https://github.com/yipihey/precrossedsheet/blob/main/chapters/01-sheet.qmd#L5-L45)):**
  * Primordial CDM velocity dispersion is negligible ($\sigma_v \lesssim 10^{-10}c$).
  * In 6D phase space $(\mathbf{x}, \mathbf{u})$, CDM occupies a **smooth, zero-thickness 3-manifold** parameterized by Lagrangian coordinates $\mathbf{q} \in \mathbb{T}^3$ (Abel, Hahn & Kähler 2012; Hahn, Abel & Kähler 2013):
    $$f(\mathbf{x},\mathbf{u},a) = \bar\rho_{\rm com}\int d^3q\,\delta_{\rm D}^{(3)}[\mathbf{x}-\mathbf{x}(\mathbf{q},a)]\,\delta_{\rm D}^{(3)}[\mathbf{u}-\mathbf{u}(\mathbf{q},a)]$$
  * Liouville's theorem + Hamiltonian symplecticity ($\omega = \sum dx_i \wedge dp_i$): gravity stretches and folds this sheet without ever tearing it.
* **Visual:** [`sheet-1d.svg`](https://github.com/yipihey/precrossedsheet/blob/main/chapters/figures/sheet-1d.svg) showing the 1D plane wave: $x(q)$, phase-space curve $(x,u)$, and Eulerian density $r(x) = 1+\delta(x)$.

#### **Slide 4 (5:30–7:00): Density is Pure Geometry Before Shell Crossing**
* **Key Equation ([`01-sheet.qmd`](https://github.com/yipihey/precrossedsheet/blob/main/chapters/01-sheet.qmd#L47-L65)):**
  * Write $\mathbf{x}(\mathbf{q}, a) = \mathbf{q} + \boldsymbol{\Psi}(\mathbf{q}, a)$ with deformation tensor $F_{ij} = \delta_{ij} + \Psi_{i,j}$ and Jacobian $J(\mathbf{q}, a) = \det F(\mathbf{q}, a)$.
  * Prior to shell crossing ($J > 0$ everywhere), $\mathbf{q} \mapsto \mathbf{x}$ is a **global diffeomorphism**. Mass conservation $\bar\rho\,d^3q = \rho\,d^3x$ gives the exact non-perturbative pushforward density:
    $$1 + \delta[\mathbf{x}(\mathbf{q}, a), a] = \frac{1}{J(\mathbf{q}, a)} = \frac{1}{\det\left(\delta_{ij} + \frac{\partial\Psi_i}{\partial q_j}\right)}$$
* **Contrast with EPT:** Notice what is *not* expanded! Even when $\|\nabla_{\mathbf{q}}\boldsymbol{\Psi}\| \sim 0.5$ stays firmly perturbative, $1/J$ naturally spans deep voids ($1/J \to 0.1$) and sharp filaments ($1/J \to 10$) while automatically guaranteeing $\rho > 0$ and $\langle 1+\delta \rangle_V = L^{-3}\int (1/J) J\,d^3q \equiv 1$!

#### **Slide 5 (7:00–8:00): Translating Our Smoothing Scale $R_f$ to the EFT Cutoff $\Lambda$**
* **Direct Mathematical Dictionary ($R_f \leftrightarrow \Lambda$):**
  Our Gaussian filter $W_G(k; R_f) = \exp(-k^2 R_f^2 / 2)$ (so $P_f(k) = P_{\rm lin}(k)\,e^{-k^2 R_f^2}$) translates directly into the EFT sharp-$k$ cutoff $W_\Lambda(k) = \Theta(\Lambda - k)$ in three ways that all give the **same rule of thumb $\Lambda \simeq 1/R_f$**:
  1. **$1/e$ Power Cutoff:** $P_f(k)/P_{\rm lin}(k) = e^{-1}$ at:
     $$\Lambda_{1/e} = \frac{1}{R_f}$$
  2. **Matched Fourier-Mode Volume (Degrees of Freedom):** Equating the sharp-$k$ sphere volume $\frac{4\pi}{3}\Lambda^3$ to the Gaussian Fourier volume $\int d^3k\,e^{-k^2 R_f^2} = (\sqrt{\pi}/R_f)^3$:
     $$\Lambda_{\rm dof} = \left(\frac{3\sqrt{\pi}}{4}\right)^{1/3}\frac{1}{R_f} \approx \frac{1.10}{R_f}$$
  3. **Matched Linear Variance $\sigma_{\rm lin}(z=0)$ in Planck 2018 $\Lambda$CDM:** Because $P_{\rm lin}(k)$ has slope $n_{\rm eff} \approx -1.7$ around $k \sim 0.05\text{–}0.25\,h\,\mathrm{Mpc}^{-1}$, the sharp-$k$ cutoff with the exact same $\sigma_{\rm lin}$ satisfies $\Lambda_\sigma \approx 1/R_f$ to within a few percent! (And for a real-space top-hat $R_{\rm TH}$, matching $k^2$ gives $R_f = R_{\rm TH}/\sqrt{5} \approx 0.447\,R_{\rm TH}$.)

| Our Filter Scale (`precrossedsheet` / `DiscoDJ.jl`) | Linear RMS $\sigma_{\rm lin}(z=0)$ | $1/e$ Wavenumber $1/R_f$ [$h\,\mathrm{Mpc}^{-1}$] | Equal-DOF Cutoff $\Lambda_{\rm dof} = 1.10/R_f$ | Equal-$\sigma_{\rm lin}$ Sharp-$k$ $\Lambda_\sigma$ [$h\,\mathrm{Mpc}^{-1}$] | Equivalent `LEFTfield` / EFT Regime & Crossed Mass at $z=0$ |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Gaussian $R_f = 22\,h^{-1}\mathrm{Mpc}$** (`mass_cdf`) | $0.169$ | $0.045$ | $0.050$ | **$0.046$** | Matches `LEFTfield` $\Lambda = 0.05\,h\,\mathrm{Mpc}^{-1}$ ($0.00\%$ crossed) |
| **Gaussian $R_f = 18\,h^{-1}\mathrm{Mpc}$** (`mass_cdf` primary) | $0.211$ | $0.056$ | $0.061$ | **$0.056$** | Ultra-clean single-stream regime ($0.00\%$ crossed across $400\,\text{Gpc}^3$) |
| **Gaussian $R_f = 14\,h^{-1}\mathrm{Mpc}$** (`precrossedsheet` lab) | $0.274$ | $0.071$ | $0.079$ | **$0.071$** | Matches `LEFTfield` $\Lambda = 0.08\,h\,\mathrm{Mpc}^{-1}$ ($0.00\%$ crossed) |
| **Gaussian $R_f = 10\,h^{-1}\mathrm{Mpc}$** (`mass_cdf` smallscale) | $0.376$ | $0.100$ | $0.110$ | **$0.096$** | **Matches `LEFTfield` fiducial $\Lambda = 0.10\,h\,\mathrm{Mpc}^{-1}$** ($\sim 0.006\%$ crossed) |
| **Gaussian $R_f = 8\,h^{-1}\mathrm{Mpc}$** (`mass_cdf` smallscale) | $0.461$ | $0.125$ | $0.138$ | **$0.119$** | **Matches `LEFTfield` SBI $\Lambda = 0.12\,h\,\mathrm{Mpc}^{-1}$** ($0.10\%$ crossed) |
| **Top-Hat $R_{\rm TH} = 14\,h^{-1}\mathrm{Mpc}$** (`lpt_pdf` / `copula`) | $0.535$ | — | — | **$0.142$** | **Matches `LEFTfield` high cutoff $\Lambda = 0.14\,h\,\mathrm{Mpc}^{-1}$** ($0.38\%\text{–}0.52\%$ crossed) |
| **Gaussian $R_f = 6\,h^{-1}\mathrm{Mpc}$** (`mass_cdf` smallscale) | $0.580$ | $0.167$ | $0.183$ | **$0.155$** | Matches `LEFTfield` $\Lambda = 0.15\text{–}0.16\,h\,\mathrm{Mpc}^{-1}$ ($0.90\%$ crossed) |
| **Gaussian $R_f = 4\,h^{-1}\mathrm{Mpc}$** (`mass_cdf` smallscale) | $0.780$ | $0.250$ | $0.275$ | **$0.227$** | Exceeds `LEFTfield` $\Lambda = 0.20\,h\,\mathrm{Mpc}^{-1}$ ($5.2\%$ crossed; void tail still $1.6\%$ accurate!) |

* **Why Does $\Lambda = 0.10\,h\,\mathrm{Mpc}^{-1}$ Have $R_f = 10\,h^{-1}\mathrm{Mpc}$ Yet a $31.25\,h^{-1}\mathrm{Mpc}$ Voxel Size in `LEFTfield`? (The Factor of $\pi$ and Nonlinear Harmonics):**
  * **1. The Factor of $\pi$ (Radius vs. Half-Wavelength):** In $W_G(k) = e^{-k^2 R_f^2/2}$, $R_f = 1/\Lambda = 10\,h^{-1}\mathrm{Mpc}$ is the **1-$\sigma$ standard deviation** of the real-space Gaussian $\exp(-r^2/2R_f^2)$ (whose full width is $\sim 4 R_f$, or equivalent top-hat diameter $2 R_{\rm TH} = 2\sqrt{5} R_f \approx 45\,h^{-1}\mathrm{Mpc}$). A wave at the cutoff $k = \Lambda = 0.10\,h\,\mathrm{Mpc}^{-1}$ has wavelength $\lambda_{\min} = 2\pi/\Lambda = 62.8\,h^{-1}\mathrm{Mpc}$, so its **Nyquist grid spacing (half-wavelength)** is:
    $$\Delta x_{\rm Nyq} = \frac{\pi}{\Lambda} = \pi R_f \approx 3.14 \times 10\,h^{-1}\mathrm{Mpc} = 31.4\,h^{-1}\mathrm{Mpc}$$
    That is why `LEFTfield` sets $N_g^\Lambda = \Lambda L / \pi = 64$ in a $2000\,h^{-1}\mathrm{Mpc}$ box ($\Delta x = 31.25\,h^{-1}\mathrm{Mpc}$).
  * **2. Why $\Lambda$ Still Matters—And What `LEFTfield`'s Eulerian Grid Reduction Throws Away:**
    * For a *linear* field cut off at $k \le \Lambda$, there is zero power above $\Lambda$, so a Nyquist grid of $\Delta x = \pi/\Lambda = 31.4\,h^{-1}\mathrm{Mpc}$ is lossless.
    * **But gravity is nonlinear!** Even a *single* initial plane wave $\delta_0(q) = A\cos(\Lambda q)$ with wavelength $62.8\,h^{-1}\mathrm{Mpc}$ collapses into a sharp caustic/filament whose exact Eulerian density $1+\delta(x) = \frac{1}{1 - AD\cos(\Lambda q)}$ has **high-frequency Fourier harmonics at $k = 2\Lambda, 3\Lambda, 4\Lambda, \dots \gg \Lambda$**!
    * In `LEFTfield` (`flowchart.pdf`), they apply $W_\Lambda$ **a second time in Eulerian space after displacement** (`reduce grid` back to $N_g^\Lambda = 64^3$, $\Delta x = 31.25\,h^{-1}\mathrm{Mpc}$), **throwing away all Eulerian harmonics $k > \Lambda$** that form sharp filaments and deep voids!
    * On the **Pre-Crossed Phase-Space Sheet**, we smooth **only the Lagrangian initial conditions** at $R_f$ (so $J > 0$), and **never truncate the Eulerian density field**—retaining the full non-perturbative $1/J$ harmonic cascade on fine Eulerian grids ($\Delta x \sim 1\text{–}4\,h^{-1}\mathrm{Mpc}$)!

---

### Act II: Measuring the Sheet — Tessellation vs. Particle Deposits (08:00 – 18:00 | 10 mins)

> [!IMPORTANT]
> **Key Message of Act II:** Even if you had exact particle trajectories, depositing $N$ particles as points via CIC/TSC destroys void statistics and injects a $L^3/N$ shot-noise floor. Treating those *exact same $N$ vertices* as a connected tetrahedral sheet eliminates shot noise ($P_{\rm shot} \equiv 0$) and empty cells.

#### **Slide 6 (8:00–10:00): The Crisis of Point-Particle Discretization**
* **Why CIC/TSC/SPH Fail ([`03-meshing.qmd`](https://github.com/yipihey/precrossedsheet/blob/main/chapters/03-meshing.qmd#L7-L48)):**
  1. **Void Starvation:** In expanding voids ($J \gg 1$), inter-particle spacing exceeds grid cell size $\Delta x$. Cells become completely empty ($\rho_C = 0$), causing $\ln(1+\delta) \to -\infty$ and a spurious delta-spike at $r=0$ in the volume PDF $p_V(r)$.
  2. **Caustic Smearing:** Convolving with $W_{\rm CIC}(\mathbf{k}) = \prod \operatorname{sinc}^2(k_i\Delta x/2)$ smears sharp filamentary spines over $2^d$–$3^d$ cells.
  3. **Fourier Shot-Noise Floor:** Point self-correlations $\langle \delta_p(\mathbf{x})\delta_p(\mathbf{x}')\rangle \supset \bar{n}^{-1}\delta_{\rm D}(\mathbf{x}-\mathbf{x}')$ inject white noise $P_{\rm shot} = L^d/N_p$.

#### **Slide 7 (10:00–12:00): The Simplicial Sheet & Exact Polyhedral Voxelization (R3D)**
* **Discretizing the Manifold ([`03-meshing.qmd`](https://github.com/yipihey/precrossedsheet/blob/main/chapters/03-meshing.qmd#L51-L125)):**
  * Divide each Lagrangian cube into **6 consistently oriented Kuhn tetrahedra** $T$, each carrying fixed mass $m_T = \bar\rho V_{T,0} = \bar\rho \Delta q^3 / 6$.
  * Affine interpolation inside each simplex gives constant deformation gradient $A = \nabla_{\mathbf{q}}\mathbf{x}$, volume $V_T = |\det A| V_{T,0}$, and piecewise-constant ($P_0$) density $\rho_T = m_T / |V_T| = \bar\rho / |J_T|$ (with optional $P_1$ barycentric vertex reconstruction).
* **Exact Voxelization (Powell & Abel 2015, `R3D`):**
  * Clip each deformed tetrahedron $T$ against Eulerian grid cells $C$ via Sutherland–Hodgman planes and integrate via the divergence theorem ([`r3d.rs`](https://github.com/yipihey/precrossedsheet/blob/main/core/src/r3d.rs)):
    $$\rho_C = \frac{1}{V_C}\sum_T \frac{m_T}{|V_T|}\operatorname{Vol}(T \cap C), \qquad \sum_C \rho_C V_C = \sum_T m_T \text{ (to machine precision } 10^{-15}\text{)}$$

#### **Slide 8 (12:00–14:30): Built-In React Interactive Panel 1 — The Same $N$ Vertices Across P₀, P₁, NGP, CIC & TSC (Coolwarm Log Scale)**
* **Interactive React.js + Wasm Slide Panel (`<LabSlidePanel preset="estimators" view="density" />`):**
  * Runs live inside the Reveal.js slide using the compiled Rust `hcc_core` WebAssembly worker and `LabSession` cache, alongside the static benchmark figure [`intro-estimators-p1.svg`](https://github.com/yipihey/precrossedsheet/blob/main/chapters/figures/intro-estimators-p1.svg).
  * **Logarithmic `[0.1, 10]` Coolwarm Color Scale:** Maps $\log_{10}(\rho/\bar\rho) \in [-1, +1]$ so the cosmic mean $\rho/\bar\rho = 1$ ($\log_{10}(1) = 0$) sits at the exact neutral white/grey midpoint ($t = 0.5$, marked with a `1` tick on the colorbar), immediately distinguishing expanding cosmic voids ($\rho/\bar\rho < 1$, blue) from contracting filaments and caustics ($\rho/\bar\rho > 1$, red).
  * **Live Comparison on a $64^2$ Gaussian $\Lambda$CDM Sheet at $z=0$ Deposited onto a $256^2$ Mesh ($4\times$ Finer than Vertex Spacing):**
    * **NGP, CIC & TSC:** Leave **75% (CIC)** and **44% (TSC)** of the 65,536 cells *completely empty* ($\rho=0$)! The map dissolves into a crystal lattice of dots, and $P(k)$ hits the dashed white-noise floor $L^2/N$.
    * **Exact Sheet P₀ (`order = 0`, piecewise-constant per simplex):** Leaves **0 empty cells** ($\langle\rho/\bar\rho\rangle = 1.00000000$). Voids are smooth and continuous ($\rho > 0$ everywhere), and $P(k)$ drops **8 orders of magnitude below the particle shot-noise floor**.
    * **Exact Sheet P₁ (`order = 1`, piecewise-linear barycentric interpolation of $w_i = 1/|J(\mathbf{q}_i)|$):** Inside each simplex $T$, $\rho_T(\mathbf{x}) = \frac{m_T}{|V_T|\bar{w}_T}\sum_i w_i \lambda_i(\mathbf{x})$ integrates analytically over each clipped polyhedron $T \cap C$ via first-moment divergence-theorem integrals $m_1, m_2$ ([`sheet.rs`](https://github.com/yipihey/precrossedsheet/blob/main/core/src/sheet.rs)). It eliminates the faint $C^0$ facet-edge density jumps of P₀ and **suppresses the high-$k$ simplex discretization floor beyond particle Nyquist by another factor of $5\times\text{–}8\times$**!
* **3D Benchmark Numbers (`DiscoDJ.jl` [`sheet_pdf/README.md`](https://github.com/yipihey/DiscoDJ.jl/blob/main/README.md)):**
  * In a $300\,h^{-1}\mathrm{Mpc}$ box on unsmoothed $1.17\,h^{-1}\mathrm{Mpc}$ cells, dropping resolution from $256^3$ to $64^3$ shifts the 1–99% density quantiles by **546% (volume) and 1489% (mass) for CIC**, versus **only 1.9% (volume) for the exact sheet**!

#### **Slide 9 (14:30–16:30): Gridless Continuous Fourier Modes (Hermite–Genocchi) & Built-In React Interactive Panel 2**
* **Question:** Do we even need an Eulerian grid to compute Fourier modes $\delta_{\mathbf{k}}$ and $P(k)$? **No!**
* **Derivation ([`03-meshing.qmd`](https://github.com/yipihey/precrossedsheet/blob/main/chapters/03-meshing.qmd#L154-L219), [`direct.rs`](https://github.com/yipihey/precrossedsheet/blob/main/core/src/direct.rs)):**
  * Parameterize simplex $T$ with vertices $\mathbf{x}_0, \dots, \mathbf{x}_d$ by barycentric coordinates $\mathbf{x} = \sum_{i=0}^d \lambda_i \mathbf{x}_i$ and vertex phases $s_i = -i\mathbf{k}\cdot\mathbf{x}_i$.
  * By the **Hermite–Genocchi divided-difference identity**, the exact continuous Fourier integral over a simplex is closed-form:
    $$I_T(\mathbf{k}) = \int_T \rho_T e^{-i\mathbf{k}\cdot\mathbf{x}}\,d^dx = m_T\,d!\,\exp[s_0, s_1, \dots, s_d]$$
* **Theorem:** Because mass is distributed over $d$-dimensional volume elements rather than point Dirac deltas $\delta_{\rm D}(\mathbf{x}-\mathbf{x}_p)$, **the continuous sheet has identically zero particle shot noise ($P_{\rm shot}^{\rm sheet} \equiv 0$)**.
* **Interactive React.js + Wasm Slide Panel 2 (`<LabSlidePanel preset="estimators" view="spectra" companion="direct_simplex" />`):**
  * Shows the **16-ray Brillouin-zone shell-averaged mesh-free simplex Fourier spectrum** vs. the isotropic FFT spectrum, plus one-click switching to **Nonlinear Mode Transfer $P(k) / [D(a)^2 |\delta_0(\mathbf{k})|^2]$** (which divides by the exact realized linear modes of the seed to cancel cosmic variance on large scales and expose nonlinear mode coupling into high $k$).

#### **Slide 10 (16:30–18:00): Two-Point Statistics as Sheet Geometry**
* **Geometric Meaning of $\xi(\mathbf{s})$ and $P(\mathbf{k})$ ([`05-two-point.qmd`](https://github.com/yipihey/precrossedsheet/blob/main/chapters/05-two-point.qmd#L66-L168)):**
  * Define the periodic polyhedral overlap kernel between deformed simplices $T$ and $U$: $K_{TU}(\mathbf{s}) = \operatorname{Vol}[T \cap (U - \mathbf{s})]$.
  * The exact two-point correlation function is the sum of polyhedral volume intersections:
    $$1 + \xi(\mathbf{s}) = \frac{1}{V_B}\sum_{T, U} r_T r_U K_{TU}(\mathbf{s})$$
  * In Fourier space, each simplex carries its exact polyhedral **form factor** $W_T(\mathbf{k})$ rather than a point-centroid approximation, retaining the full internal shape covariance tensor $\Sigma_T$ of stretching void elements and compressing filament elements.

---

### Act III: Dynamics — High-Order LPT, PDFs, Copulas & Exact Tidal Laws (18:00 – 29:00 | 11 mins)

> [!NOTE]
> **Pedagogical Bridge:** Now that the audience sees *how* the sheet measures observables without discreteness noise, they will ask: *"How well does finite-order LPT actually move the vertices of this sheet compared to full $N$-body gravity?"* Here you bring in the mathematical elegance of the cofactor LPT recursion and the heavyweight numerical benchmarks from `DiscoDJ.jl`.

#### **Slide 11 (18:00–20:00): Building the Sheet with 4LPT — Cofactor Equations & Cauchy Invariant**
* **Why LPT is Purely Polynomial ([`02-lpt.qmd`](https://github.com/yipihey/precrossedsheet/blob/main/chapters/02-lpt.qmd#L44-L158)):**
  * Transforming $\nabla_{\mathbf{x}} \cdot \ddot{\mathbf{x}} + \dots = -a^{-2}\nabla_{\mathbf{x}}^2\phi$ to Lagrangian coordinates via the cofactor matrix $C_{ij} = \partial J/\partial F_{ij} = J(F^{-1})_{ji}$ **cancels the $1/J$ factor from the chain rule against the $J$ in the Poisson source** (Matsubara 2015):
    $$C_{ij}\left(\frac{d^2}{dt^2} + 2H\frac{d}{dt}\right)\Psi_{i,j} = \frac{3}{2}\Omega_m(a)H(a)^2 (J - 1)$$
  * Expanding $J - 1 = \operatorname{tr}M + \mu_2(M,M) + \mu_3(M,M,M)$ yields a closed bilinear/trilinear recursion for longitudinal modes $\nabla_{\mathbf{q}}\cdot\boldsymbol{\Psi}^{(n)}$.
* **The Transverse Piece (Cauchy Vorticity Invariant):**
  * Kelvin's circulation theorem says Eulerian velocity is irrotational ($\nabla_{\mathbf{x}} \times \mathbf{u} = \mathbf{0}$) before shell crossing.
  * In Lagrangian coordinates, $\epsilon_{ijk} F_{lj}\dot{F}_{lk} = 0$ **requires a non-zero Lagrangian displacement curl starting at 3LPT**:
    $$\nabla_{\mathbf{q}} \times \boldsymbol{\Psi}^{(3)} = -\frac{1}{3}\epsilon_{ijk}\Psi_{l,j}^{(1)}\Psi_{l,k}^{(2)} \neq \mathbf{0}$$
* **Coupled $\Lambda$CDM Growth ([`growth.rs`](https://github.com/yipihey/precrossedsheet/blob/main/core/src/growth.rs)):** At $n=3, 4$, different spatial vertices $(\mu_2, \mu_3, \text{curl})$ evolve with distinct coupled $\Lambda$CDM growth ODEs $g_\tau(a)$ (1 linear + 1 quadratic + 3 cubic + 8 quartic basis fields in 3D).

#### **Slide 12 (20:00–21:30): Isolating Trajectory Error from Estimator Error — The PM Sheet & Interactive Matrix**
* **Methodological Breakthrough ([`04-validation.qmd`](https://github.com/yipihey/precrossedsheet/blob/main/chapters/04-validation.qmd#L68-L103)):**
  * When people compare LPT to $N$-body, they usually compare LPT to CIC particles—mixing *dynamical trajectory error* with *CIC assignment/shot-noise error*.
  * In both [`precrossedsheet`](https://github.com/yipihey/precrossedsheet/blob/main/chapters/04-validation.qmd) and [`DiscoDJ.jl`](https://github.com/yipihey/DiscoDJ.jl/blob/main/README.md), we evolve the vertices with a symplectic PM solver (or BullFrog / P1-FEM solver) and **keep the initial Kuhn tetrahedral connectivity on the $N$-body trajectories**, depositing both with R3D!
  * Furthermore, we track the exact cubic drift polynomial $V_T(t) = c_0 + c_1 t + c_2 t^2 + c_3 t^3$ inside every timestep to detect any transient interior shell crossing.
* **Built-In React Interactive Panel 3 (Slide 12b: Live Trajectory $\times$ Estimator Matrix, Scale-Dependent Variance $\sigma^2(R)$ & PDFs):**
  * Embedded React component powered by `hcc_core` exposing the persistent **`{1LPT, 2LPT, 3LPT, 4LPT, PM}` $\times$ `{P₀, P₁, NGP, CIC, TSC}` matrix**:
    1. **Scale-Dependent Variance $\sigma^2(R)$:** Compare Top-Hat and Gaussian window variance $\sigma^2(R) = \frac{1}{L^d}\sum_{\mathbf{k}\neq\mathbf{0}} |W(kR)|^2|\delta_{\mathbf{k}}|^2$ across `1LPT` $\to$ `4LPT` vs. `PM` under identical `P₀` sheet deposition (isolating pure dynamical convergence) vs. `CIC`/`NGP` (showing shot-noise inflation at small $R$).
    2. **One-Point Volume & Mass PDFs ($dP/d\rho$ and $dP/d\log_{10}\rho$):** Toggle directly between linear and log-density PDFs to watch `1LPT` $\to$ `2LPT` $\to$ `3LPT` $\to$ `4LPT` converge onto the `PM (P₀)` sheet in both the void tail ($\rho < \bar\rho$) and filament/wall tail ($\rho > \bar\rho$), while `NGP`/`CIC` pile up at $\rho = 0$.

#### **Slide 13 (21:30–23:30): Study 1 (`DiscoDJ.jl`) — One-Point Volume & Mass PDFs ($1\text{LPT} \to 4\text{LPT}$ vs $N$-Body)**
* **Two Fundamental Measures ([`05-pdfs.qmd`](https://github.com/yipihey/precrossedsheet/blob/main/chapters/05-pdfs.qmd)):**
  * Volume PDF $p_V(r)$ (weights simplices by deformed volume $V_T \propto J_T$; probes **voids**) vs. Mass PDF $p_M(r) = r\,p_V(r)$ (weights simplices equally by mass $m_T$; probes **filaments and peaks**).
* **The Benchmark (`studies/lpt_pdf`, $L=300\,h^{-1}\mathrm{Mpc}$, $R_f = 14\,h^{-1}\mathrm{Mpc}$, $z=0$):**
  * **Why 2LPT is not enough:** 2LPT is too narrow in both tails—its voids aren't empty enough and its peaks aren't dense enough.
  * **How 3LPT and 4LPT close the gap:**

| Smoothing $R_s$ [$h^{-1}\mathrm{Mpc}$] | Observable / Metric | 1LPT (Zel'dovich) | 2LPT | 3LPT | **4LPT** |
| :---: | :--- | :---: | :---: | :---: | :---: |
| **$7$ ($\frac{1}{2}R_f$)** | Variance $\sigma^2/\sigma^2_{\rm NB} - 1$ | $-18.3\%$ | $-9.4\%$ | $-4.0\%$ | **$-2.6\%$** |
| **$7$ ($\frac{1}{2}R_f$)** | Vol $0.1\% / 99.9\%$ quantiles | $-19.7\% / -26.4\%$ | $+10.8\% / -10.8\%$ | $-6.3\% / -5.1\%$ | **$+4.3\% / -2.8\%$** |
| **$14$ ($R_f$)** | Variance $\sigma^2/\sigma^2_{\rm NB} - 1$ | $-9.1\%$ | $-5.0\%$ | $-1.4\%$ | **$-0.9\%$** |
| **$14$ ($R_f$)** | Mass $99\% / 99.9\%$ quantiles | $-19.2\% / -23.0\%$ | $-7.5\% / -9.1\%$ | $-3.5\% / -4.2\%$ | **$-1.9\% / -2.4\%$** |
| **$28$ ($2R_f$)** | Max $|\text{quantile shift}|$ | $9.3\%$ | $2.8\%$ | $1.1\%$ | **$0.5\%$** |
| **$42$ ($3R_f$)** | Max $|\text{quantile shift}|$ | $5.1\%$ | $1.3\%$ | $0.3\%$ | **$0.1\%$** |

* **Pedagogical Takeaway:** Notice the alternating sign in the void tail ($-19.7\% \to +10.8\% \to -6.3\% \to +4.3\%$) and monotonic convergence in the dense tail. At $R_s \ge R_f$, **4LPT is a $\lesssim 1\%$ model of the full non-Gaussian 1-point distribution!**

#### **Slide 14 (23:30–25:00): Study 2 (`DiscoDJ.jl`) — Separating the Copula from the 1-Point PDF**
* **Question (`studies/lpt_pdf/copula`):** Two fields can have the same 1-point PDF $p_V(r)$ but different spatial clustering $\xi(r)$ (and vice versa). Where does the remaining difference between LPT and $N$-body live—in the **1-point marginal PDF** or in the **spatial copula** (rank-ordered spatial structure)?
* **Method:** Rank-Gaussianize the sheet densities $y = \Phi^{-1}[(\text{rank}-\frac{1}{2})/N]$ on single-stream tetrahedra so every run has an identical unit-Gaussian 1-point PDF by construction, then measure $\xi_y(r)$ and $r_y(k)$.
* **Result:**
  * Even with the 1-point PDF removed, 2LPT has a small but converged two-point copula error ($\Delta\xi_y \approx 5.7\times 10^{-3}$ at $r = 20$–$30\,h^{-1}\mathrm{Mpc}$ at $z=0$).
  * **4LPT removes 75–95% of this copula error!**
  * Meanwhile, $N$-body's extra clustering around $\sim 10\,h^{-1}\mathrm{Mpc}$ comes almost entirely from its sharper 1-point PDF tail, *not* from a rearrangement of the spatial web!

#### **Slide 15 (25:00–27:30): Study 3 (`DiscoDJ.jl/studies/mass_cdf`) — An Exact Parameter-Free Law on the Pre-Crossed Sheet**
* **A New Theoretical Result Tested Across $1100\,(h^{-1}\mathrm{Gpc})^3$ ([`SUMMARY.md`](https://github.com/yipihey/DiscoDJ.jl/blob/main/studies/mass_cdf/results/SUMMARY.md)):**
  * In linear theory, the mass fraction above the mean density is $F_M(>\bar\rho) = P(\delta_L > 0) = \frac{1}{2}$.
  * As the sheet evolves nonlinearly before shell crossing, how much mass resides above the mean density $\bar\rho$ (i.e. has compressed, $J < 1$)?
  * Because of **tidal shear focusing**, even elements with $\delta_L = 0$ get squeezed by the traceless tidal shear $T_{ij} = (\partial_i\partial_j/\nabla^2 - \frac{1}{3}\delta_{ij})\delta_L$:
    $$-\ln J = \delta_L + \frac{13}{42}\delta_L^2 + \frac{2}{7}T_{ij}T^{ij} + \mathcal{O}(\delta_L^3) \quad \Longrightarrow \quad \left.-\ln J\right|_{\delta_L = 0} \simeq \frac{2}{7}T_{ij}T^{ij} > 0$$
  * This yields a **parameter-free odd-power expansion in $\sigma$**:
    $$F_M(>\bar\rho) = P(J < 1) = \frac{1}{2} + A_1 \sigma + A_3 \sigma^3 + \mathcal{O}(\sigma^5)$$
    with exact leading coefficient:
    $$A_1^{\rm EdS} = \frac{4}{21\sqrt{2\pi}} = 0.075989\dots, \qquad A_1^{\Lambda\text{CDM}}(a) = \frac{1 + D_2(a)/D_1(a)^2}{3\sqrt{2\pi}} = 0.075509 \quad (z=0)$$
* **High-Statistics Confirmation (`studies/mass_cdf/results/figures/`):**
  * Show [`fig1_Yodd_over_sigma_vs_sigma2.pdf`](https://github.com/yipihey/DiscoDJ.jl/blob/main/studies/mass_cdf/results/figures/fig1_Yodd_over_sigma_vs_sigma2.pdf) and [`fig4_shear_focusing.pdf`](https://github.com/yipihey/DiscoDJ.jl/blob/main/studies/mass_cdf/results/figures/fig4_shear_focusing.pdf):
    * **Measured $A_1$ (400 Gaussian $\text{Gpc}^3$, 4LPT):** $0.075984 \pm 0.000054$ ($-0.1\sigma$ pull; $0.07\%$ precision!).
    * **Measured exact $\Lambda$CDM growth $A_1$:** $0.075496 \pm 0.000030$ (resolves the $0.6\%$ dark-energy suppression of $D_2/D_1^2$ at $>15\sigma$!).
    * **Even powers ($A_2, A_4$):** Vanish identically within $10^{-5}$.
    * **Cubic coefficient $A_3$:** Measured at $(2.1 \pm 1.0)\times 10^{-3}$ across $880\,(h^{-1}\mathrm{Gpc})^3$ vs analytic prediction $1.6\times 10^{-3}$ (and requires **4LPT** to converge; 4LPT $\to$ 5LPT $\to$ 6LPT shifts $A_3$ by $< 4\times 10^{-5}$!).
    * **One-way street ($F_{-+} \equiv 0$):** Show [`fig3_transition_fractions.pdf`](https://github.com/yipihey/DiscoDJ.jl/blob/main/studies/mass_cdf/results/figures/fig3_transition_fractions.pdf) — *zero* initially overdense elements expand below $\bar\rho$; the entire excess $Y \approx 1.6\%$ comes from initially underdense elements ($\delta_L < 0$) focused above $\bar\rho$ by tidal shear!

#### **Slide 16 (27:30–29:00): FEM & PM $N$-Body vs 4LPT down to Smaller Scales**
* **FEM $N$-Body Check (`studies/mass_cdf` §9):**
  * Evolving the Kuhn tetrahedra with a P1 Finite-Element Poisson solver (`disco-dj-fem`) and matching the linear discrete operator yields agreement with 4LPT on $F_M(>\bar\rho)$ to **$(3\text{–}4)\times 10^{-5}$ ($0.2\%$ of the nonlinear migration)**!
* **What Happens as We Push $R_f$ Down Past First Crossing? (`studies/mass_cdf` §16, $R_f = 4\text{–}14\,h^{-1}\mathrm{Mpc}$):**
  * Even at $R_f = 6\,h^{-1}\mathrm{Mpc}$ ($\sigma = 0.58$, $0.9\%$ mass crossed) and $R_f = 4\,h^{-1}\mathrm{Mpc}$ ($\sigma = 0.78$, $5.2\%$ mass crossed), the three-term series $\frac{1}{2} + A_1\sigma + A_3\sigma^3$ matches PM $N$-body within **$0.5\%$**, and the 4LPT void mass fraction below $0.5\bar\rho$ remains accurate to **$0.4\%$–$1.6\%$**!

---

### Act IV: Why the Pre-Crossed Sheet is Ideal for Field-Level Inference (29:00 – 37:00 | 8 mins)

> [!IMPORTANT]
> **Key Message of Act IV:** Connect the physics directly to the inference pipeline. Why should a cosmologist building an HMC sampler or SBI pipeline use the pre-crossed sheet instead of EPT or PM?

#### **Slide 17 (29:00–31:00): Four Structural Advantages for Field-Level Inference**
1. **Zero Shot Noise $\Rightarrow$ Smooth Likelihood Surfaces:** No discrete particle counting noise, no empty void cells where $\ln(1+\delta)$ blows up, no stochastic floor in $P(k)$.
2. **Space–Time Factorization of nLPT:** The 13 spatial basis tensors $\mathbf{S}^\tau(\mathbf{q})$ of 4LPT depend *only* on the initial modes. Varying redshift $a$ or growth histories $g_\tau(a)$ along a lightcone is an instant linear combination—no timestepping required!
3. **Positivity & Conservation by Construction:** $1+\delta = 1/J > 0$ everywhere before crossing and $\sum_C \rho_C V_C = M_{\rm tot}$ to machine precision.
4. **End-to-End Differentiability:** Both in JAX/Julia ([`DiscoDJ.jl`](https://github.com/yipihey/DiscoDJ.jl/blob/main/lib/DiscoDJLib/src/DiscoDJLib.jl)) and in Rust/WebGPU ([`precrossedsheet`](https://github.com/yipihey/precrossedsheet/blob/main/chapters/06-inference.qmd)).

#### **Slide 18 (31:00–33:00): Exact Analytic Gradients of Continuous Sheet Modes**
* **Differentiating Without Grids or Finite Differences ([`06-inference.qmd`](https://github.com/yipihey/precrossedsheet/blob/main/chapters/06-inference.qmd#L19-L85), [`tangent.rs`](https://github.com/yipihey/precrossedsheet/blob/main/core/src/tangent.rs)):**
  * Forward-mode Jacobian-Vector Products (JVPs) propagate any initial mode tangent $\dot{\delta}_0(\mathbf{k})$ through the bilinear $\mu_2$ and trilinear $\mu_3$ LPT recurrence via exact Leibniz product rules:
    $$\dot{\mu}_2(A, B) = \mu_2(\dot{A}, B) + \mu_2(A, \dot{B})$$
  * How do we differentiate the continuous Hermite–Genocchi Fourier mode $I_T(\mathbf{k}) = m_T d!\,\exp[s_0,\dots,s_d]$ with respect to vertex positions $\dot{\mathbf{x}}_i$ ($\dot{s}_i = -i\mathbf{k}\cdot\dot{\mathbf{x}}_i$)?
  * By the **confluent divided-difference identity**:
    $$\frac{\partial}{\partial s_i}\exp[s_0, \dots, s_d] = \exp[s_0, \dots, s_d, s_i] \quad \Longrightarrow \quad \dot{I}_T(\mathbf{k}) = m_T\,d!\sum_{i=0}^d \dot{s}_i\,\exp[s_0, \dots, s_d, s_i]$$
  * Gives **machine-precision analytic derivatives** $\dot{\delta}_{\mathbf{k}}$ and $\dot{P}(\mathbf{k}) = 2L^d\operatorname{Re}[\delta_{\mathbf{k}}^*\dot{\delta}_{\mathbf{k}}]$ without any grid interpolation or finite-difference step-size tuning!

#### **Slide 19 (33:00–35:00): Baryon Acoustic Oscillations Survive the Nonlinear Sheet Map**
* **Visual:** Show [`intro-bao.svg`](https://github.com/yipihey/precrossedsheet/blob/main/chapters/figures/intro-bao.svg) and table [`_intro-bao-table.qmd`](https://github.com/yipihey/precrossedsheet/blob/main/chapters/_intro-bao-table.qmd) ([`02-cosmology.qmd`](https://github.com/yipihey/precrossedsheet/blob/main/chapters/02-cosmology.qmd#L76-L95)):
  * Evolve identical random phases with the full oscillating Eisenstein & Hu BAO transfer function vs. the smooth no-wiggle transfer function to $z=0$.
  * The subtle $\sim 6\%$ acoustic modulation in $P(k)$ across the first four BAO peaks is carried through the 4LPT nonlinear displacement map, the simplicial tessellation, and the R3D estimator with a **mean modulation error of $0.47\%$ in 2D and $0.12\%$ in 3D**—without being smeared out by assignment windows or buried under a shot-noise floor.

#### **Slide 20 (35:00–37:00): Built-In React Panel 4 (Live BAO & Analytic JVP Explorer) + Ecosystem**
* **Built-In React Interactive Panel 4 (Slide 20: BAO Wiggle vs. No-Wiggle & Exact Confluent Divided-Difference JVPs):**
  * Embedded React component inside the slide deck allowing live switching between:
    1. **Phase-Matched BAO vs. No-Wiggle Comparison:** Computes two phase-matched `hcc_core` realizations at the same seed (full EH98 BAO vs. smooth no-wiggle transfer) and plots $P_{\text{BAO}}(k)/P_{\text{nw}}(k)$ across `P₀`, `P₁`, and `CIC`, demonstrating how the sheet preserves acoustic peaks while `CIC` shot noise corrupts high-$k$ oscillations.
    2. **Analytic Confluent Divided-Difference JVP vs. Central Finite Differences:** Computes exact forward-mode tangent derivatives $\partial P_0(k)/\partial A$ and $\partial \rho/\partial A$ via `tangent.rs` alongside centered finite differences, showing machine-precision agreement.
* **Software Ecosystem & Multi-Presentation Hub ([`08-talks.qmd`](https://github.com/yipihey/precrossedsheet/blob/main/chapters/08-talks.qmd)):**
  * **[`DiscoDJ.jl`](https://github.com/yipihey/DiscoDJ.jl/blob/main/README.md) & `DiscoDJNative`:** High-performance Julia + JAX differentiable 1–6LPT, PM/BullFrog/FEM $N$-body, exact R3D sheet deposition, and lightcone projection (`lpt_lightcone`), integrated with multi-code injectors and constrained-realization pipelines (**QuaiaICs** / **ECHOES**).
  * **[`precrossedsheet`](https://github.com/yipihey/precrossedsheet/blob/main/README.md) (`hcc_core`):** Self-contained Rust/WebAssembly + WebGPU laboratory and multi-audience presentation suite served on GitHub Pages (`https://yipihey.github.io/precrossedsheet/`):
    * `talks/eft-fli-40min/` — **40-Minute Pedagogical Seminar for PT / EFTofLSS & Field-Level Inference Cosmologists** (this Reveal.js + React deck).
    * `talks/conference-15min/` — **15-Minute Contributed Conference Talk** (Cosmic Web & Survey Cosmology executive summary).
    * `talks/sbi-hmc-workshop/` — **30-Minute Inference Workshop Deep-Dive** (HMC, SBI, and differentiable forward models).
    * `talks/computational-methods/` — **45-Minute Computational Methods Seminar** (R3D voxelization, Hermite–Genocchi modes & WebGPU/Wasm).

---

### Act V: Scope, EFT Connection & Conclusions (37:00 – 40:00 | 3 mins)

#### **Slide 21 (37:00–38:30): Conceptual Comparison for EFT Cosmologists — Where Does the Sheet Sit?**
* Present a crisp synthesis table for the audience:

| Aspect | Eulerian PT / EFTofLSS | Standard PM / $N$-Body | **Pre-Crossed Phase-Space Sheet (4LPT + R3D/HG)** |
| :--- | :--- | :--- | :--- |
| **Expansion Variable** | Density $\delta(\mathbf{x})$ (breaks at $|\delta|\sim 1$) | Discrete point trajectories $\mathbf{x}_p(t)$ | Lagrangian displacement $\boldsymbol{\Psi}(\mathbf{q})$; **exact** $1+\delta = 1/J$ |
| **UV Regularization** | Eulerian smoothing + EFT counterterms $c_s^2 k^2\delta$ | Force softening + grid cutoff | Lagrangian smoothing $R_f$ keeping $J(\mathbf{q}, a) > 0$ |
| **Voids ($\delta \to -1$)** | Poor (Taylor series in $\delta$ fails) | Catastrophic empty cells ($\rho=0$), shot noise | **Exact & smooth** ($\rho_T = m_T/V_T > 0$ everywhere) |
| **Shot Noise $P_{\rm shot}$** | Stochastic counterterm $\sim k^0, k^2$ | $L^3 / N_p$ white noise floor | **Identically zero ($P_{\rm shot} \equiv 0$)** |
| **1-pt PDFs & Copulas** | Cumulant expansion diverges in tails | Noisy in voids; expensive | **Sub-percent match to $N$-body** (4LPT removes 75–95% of 2LPT copula error) |
| **Differentiability** | Analytic for $n$-pt functions | Needs adjoint PM + smoothed CIC | **Exact analytic JVPs** via confluent divided differences |

#### **Slide 22 (38:30–40:00): Summary & Take-Home Messages**
1. **The Tessellation Measures What Particles Cannot:** Connecting vertices into mass-carrying Kuhn tetrahedra eliminates empty void cells and drives particle shot noise to identically zero ($P_{\rm shot} \equiv 0$), reducing $64^3$ density quantile errors from $>500\%$ (CIC) to $<2\%$ (Sheet).
2. **Before Shell Crossing, 4LPT is Remarkably Complete:** On scales filtered at $R_f \sim 14$–$18\,h^{-1}\mathrm{Mpc}$ ($>99.5\%$ single-stream at $z=0$), 4LPT reproduces $N$-body 1-point PDFs to $\lesssim 1\%$, removes 75–95% of 2LPT's 2-point copula error, and satisfies the exact tidal shear focusing law $F_M(>\bar\rho) = \frac{1}{2} + A_1\sigma + A_3\sigma^3$ to $10^{-4}$.
3. **Ready for Field-Level Inference:** With precomputed spatial LPT tensors, exact R3D voxelization, and closed-form Hermite–Genocchi Fourier derivatives, the pre-crossed sheet provides a fast, noise-free, differentiable forward model for the next generation of galaxy surveys.
* **Links on Final Slide:**
  * Interactive Book & Wasm/WebGPU Lab: `https://yipihey.github.io/precrossedsheet/` ([`precrossedsheet`](https://github.com/yipihey/precrossedsheet/blob/main/README.md))
  * Code & Benchmarks: `https://github.com/yipihey/DiscoDJ.jl` ([`DiscoDJ.jl`](https://github.com/yipihey/DiscoDJ.jl/blob/main/README.md))

---

## Anticipated Questions from PT/EFT Cosmologists (and Your Ready Answers)

1. **"In EFTofLSS, UV modes that have shell-crossed back-react on large-scale modes via the effective speed of sound $c_s^2 k^2 \delta$. When you smooth the initial conditions at $R_f$, what about the back-reaction of the filtered-out small scales?"**
   * *Answer:* Great question—that is the exact distinction between filtering *initial* modes vs. filtering *final* fields. In our baseline validation, we test the dynamics of the $R_f$-filtered Hamiltonian system itself (where both LPT and $N$-body evolve the same filtered ICs, agreeing to $<1\%$). For full unfiltered universes, the pre-crossed sheet serves as the **exact deterministic infrared/mesoscale backbone** (resolving all nonlinear displacements, voids, and tidal shear non-perturbatively), on top of which small-scale multi-stream/baryonic physics below $R_f$ can be parameterized as a local Lagrangian bias/response on the tetrahedra ($J, T_{ij}T^{ij}, \operatorname{tr}T^3$)—with the huge advantage that the large-scale IR displacement and void geometry are already exact rather than Taylor-expanded!
2. **"Does the LPT series diverge at first shell crossing?"**
   * *Answer:* Physically, single-stream fluid equations cease to be valid once trajectories cross ($J \le 0$) because multi-stream gravitational forces differ from single-stream continuation. Mathematically, however, Rampf & Hahn (2021) showed that the time-Taylor radius of convergence of LPT is governed by complex-time singularities that generally lie *later* than first shell crossing. And empirically in our `mass_cdf` small-scale study (`R_f = 4–6 Mpc/h`, where $1\%–5\%$ of mass has crossed), the remaining $95\%+$ uncrossed sheet elements still match PM $N$-body void fractions at $\rho = 0.5\bar\rho$ to $\sim 0.4\%–1.6\%$!
3. **"Why go to 4LPT instead of stopping at 2LPT or 3LPT?"**
   * *Answer:* Two reasons:
     1. **Qualitative physics:** Transverse (curl) Lagrangian displacements required by Kelvin's circulation theorem (Cauchy invariant) only enter at **3LPT**, and their coupling into longitudinal density modes enters at **4LPT**. Furthermore, in the exact expansion $F_M(>\bar\rho) = \frac{1}{2} + A_1\sigma + A_3\sigma^3$, all degree-$k$ parts of $-\ln J$ up to $k=4$ enter $A_3$ at $\mathcal{O}(\sigma^3)$—so 2LPT and 3LPT get even the *sign/magnitude* of $A_3$ wrong ($-0.0073$ and $+0.0103$), whereas **4LPT converges** ($+0.0021$, unchanged at 5LPT and 6LPT to $4\times 10^{-5}$!).
     2. **Quantitative tails & copulas:** In the $R_f = 14\,h^{-1}\mathrm{Mpc}$ benchmark, 2LPT underpredicts the mass 99.9% quantile by $29\%$ at $R_s = 7\,h^{-1}\mathrm{Mpc}$ and has a $5.7\times 10^{-3}$ copula error; 4LPT cuts the quantile error by $>2\times$ and removes $75\text{–}95\%$ of the copula error.
4. **"How expensive is evaluating the continuous Hermite–Genocchi modes or R3D clipping compared to CIC?"**
   * *Answer:* For the 4LPT dynamics itself, spatial basis fields $\mathbf{S}^\tau(\mathbf{q})$ are computed once via FFTs (in `DiscoDJNative`, fused kernels compute 4LPT at $64^3$ in 1.6s on CPU and milliseconds on GPU). For observation, cell-local R3D clipping and Hermite–Genocchi modes are embarrassingly parallel across tetrahedra and implemented in Rust + WebGPU shaders ([`precrossedsheet`](https://github.com/yipihey/precrossedsheet/blob/main/README.md)). And for Lagrangian field-level statistics or lightcone vertex evaluations ($\rho = \bar\rho / J(\mathbf{q})$ directly at sheet elements, as in `QuaiaICs` and `mass_cdf`), **no Eulerian grid deposition is needed at all**—you evaluate $\det(\mathbf{I} + \nabla_{\mathbf{q}}\boldsymbol{\Psi})$ pointwise on the grid in $\mathcal{O}(N^3)$!
