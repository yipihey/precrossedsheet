# LPT closure timing baseline

Neural Sheet closures `C₃` / `C₄` in `core/src/lpt_closure.rs` are exact EdS
source distillations (μ₂ / μ₃ / curl of lower-order deformation tensors + spectral
Helmholtz). They match native merged nLPT to ~10⁻⁷ relative RMS.

**Do not** quote speedups against `Lpt::new` alone as a production win.
`Lpt::new` enumerates unmerged parent terms separately (many FFTs). That path is
analogous to DiscoDJNative `compute_core(...; mode=:lean)`.

The production Hahn–Rampf / DISCO-DJ engine is:

- Repository: [yipihey/DiscoDJ.jl](https://github.com/yipihey/DiscoDJ.jl)
- API: `DiscoDJNative.compute_core(fphi, K; n_order, mode=:fast)`
- Behaviour: shared extended-grid derivatives + fused source kernels + 3/2-rule
  de-aliasing (List, Hahn, Winkler & Floess; Rampf–Hahn recursion)

On a representative host, `:fast` is already ~4–7× faster than `:lean` at
3LPT/4LPT (see Project research log). Any Neural Sheet performance claim must
beat `:fast` under matched resolution, dtype, threads, and dealiasing.
