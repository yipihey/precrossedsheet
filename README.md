# Pre-crossed sheet

A scientific Quarto report and Rust/WebAssembly laboratory for the cold phase-space
sheet **before shell crossing**. Built from [yipihey/quadrant](https://github.com/yipihey/quadrant),
with the MIT-licensed nLPT and exact-deposition core from
[Hopf–Cole cosmology](https://github.com/yipihey/hopf-cole-cosmology).

The book covers 1D–3D sheet geometry, nLPT through fourth order, conservative
meshing and continuous Fourier modes, qualified validation against N-body,
volume/mass PDFs, and directional derivatives for forward modeling. The live
lab uses flat ΛCDM or EdS growth, an EH98 smooth physical spectrum and matched PM dynamics; the published replay preserves the original
DiscoDJ filtered-cosmology results and their limitations. Smaller scales and
validated post-crossing LPT dynamics are future work. Crossed outputs remain available with a caution.

## Build

Requires Rust, wasm-pack, the wasm32-unknown-unknown target, Python 3, Node.js,
and Quarto 1.6 or newer.

```sh
make test
make wasm api check site
python3 scripts/serve.py 8790 _site
```

Open `http://localhost:8790`, or `app/index.html` through that HTTP server.
The browser needs the generated Wasm package; it has no numerical JS fallback.
Plots export SVG/CSV and field panels export PNG. Versioned URLs record every control. Full JSON manifests preserve inputs, numerical results and build provenance. Supported browser lattice ceilings are 1024 in 1D, 256² in 2D and 64³ in 3D, with explicit mesh/memory limits rather than silent reductions.

## Verification and reuse

See [the verification record](docs/verification.md), [evidence audit](docs/evidence-audit.md),
and [decisions](docs/decisions.md). The Rust API is generated with `make api`.
The [WebGPU backend plan](docs/webgpu-backend-plan.md) documents the implemented first
hybrid observation backend and the reusable Hopf–Cole kernels.
Use a Cargo path dependency on `core/` to reuse the numerical library without
the web interface. Retained crate name: `hcc_core`.

The exact clipped-deposit path and the differentiable continuous Fourier path
are distinct. Arbitrary IC JVPs and PM tangents are at fixed cosmology; no
reverse adjoint or cosmological growth gradient is claimed.

MIT license; upstream copyright retained. `CITATION.cff` describes this project.

The optional WebGPU backend evaluates continuous 2D/3D sheet modes and JVPs,
with selective f64 CPU recovery for weak LPT modes. It also evolves cosmological
PM positions, canonical momenta and amplitude tangents on the GPU, using Rust
kick/drift clocks and a reusable radix-2 FFT. CPU remains the default. Initial
conditions, nLPT construction, exact meshing and PDFs use Rust/Wasm.
Centered amplitude finite differences use compensated GPU endpoint integrals
after device validation, with f64 CPU recovery and the unchanged derivative gate. Non-power-of-two force meshes retain CPU PM with an explicit reason.
Run `node scripts/test_gpu.mjs --require-gpu --full` and
`node scripts/test_gpu.mjs --require-gpu --resident` after `make wasm`; see the
[observer record](docs/webgpu-results.md), [resident PM record](docs/webgpu-pm-results.md),
and [finite-difference record](docs/webgpu-fd-results.md).
Run `node scripts/test_gpu.mjs --require-gpu --fd` for endpoint parity and worker checks.
