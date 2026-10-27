# 45-Minute Computational Methods Seminar (`talks/computational-methods/`)

- **Target Audience:** Computational physicists, applied mathematicians, scientific computing researchers, and Rust / Julia / WebGPU developers.
- **Duration:** 40 + 5 minutes (24 slides).
- **Core Narrative:**
  1. **Symplectic & Perturbative Sheet Discretizations (10 min):** Cofactor LPT recursion, Cauchy curl invariants, coupled $\Lambda$CDM growth ODEs, and exact cubic-in-time drift polynomials $V_T(t) = c_0 + c_1 t + c_2 t^2 + c_3 t^3$ for sub-step caustic detection.
  2. **Exact R3D Convex Polytope Clipping & $P_0 / P_1$ Voxelization (12 min):** Kuhn Freudenthal $d!$ simplex decomposition, Sutherland–Hodgman / Powell–Abel R3D plane clipping, and barycentric linear $P_1$ centroid integration.
  3. **Hermite–Genocchi Divided Differences & Confluent Taylor Stabilization (10 min):** Removing coincident-phase singularities $|s_i - s_j| \to 0$ to machine precision (`1e-15`) and evaluating 16-ray Brillouin-zone shell-averaged mesh-free spectra + live React Panel (`simplex-spectra`).
  4. **Browser-Native Rust / WebAssembly / WebGPU Architecture (8 min):** Sharing a single Rust kernel (`hcc_core`) + WGSL compute pipeline across native tests, interactive Quarto book widgets, and embedded React Reveal.js presentation panels.
