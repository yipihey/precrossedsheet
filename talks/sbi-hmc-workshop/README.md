# 30-Minute Inference Workshop Deep-Dive (`talks/sbi-hmc-workshop/`)

- **Target Audience:** Bayesian field-level inference practitioners, Hamiltonian Monte Carlo (HMC) developers (`BORG`, `LEFTfield`, `Manticore`), and Simulation-Based Inference (SBI) / normalizing-flow researchers.
- **Duration:** 25 + 5 minutes (18 slides).
- **Core Narrative:**
  1. **Why Shot Noise & Empty Cells Hurt HMC Leapfrog Integrators (6 min):** How CIC zero-density void cells ($\delta = -1$) and stochastic high-$k$ particle noise degrade HMC step sizes and SBI training efficiency.
  2. **Space–Time Factorization of 4LPT on Lightcones (7 min):** Precomputing the 13 spatial basis tensors $\mathbf{S}^\tau(\mathbf{q})$ and evaluating redshift-space distortions and lightcones (`lpt_lightcone`, `QuaiaICs`, `ECHOES`) via instant linear combinations.
  3. **Exact Confluent Divided-Difference JVPs & VJPs (8 min):** Differentiating continuous Hermite–Genocchi simplex Fourier modes $I_T(\mathbf{k}) = m_T d!\,\exp[s_0,\dots,s_d]$ via $\partial_{s_i}\exp[s_0,\dots,s_d] = \exp[s_0,\dots,s_d,s_i]$ + live React Panel (`bao-jvp-inference`).
  4. **Copula Fidelity & BAO Phase Preservation (4 min):** Why 4LPT + Sheet gives an unbiased, zero-shot-noise forward model down to $R_f \sim 10\text{–}14\,h^{-1}\mathrm{Mpc}$.
