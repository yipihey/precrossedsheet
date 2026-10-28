import { makeDefaults } from './state.js';

export const PRESETS = [
  { id: 'safe-sheet', label: 'Gentle 2D sheet', note: 'A low-cost 2D translationally invariant ΛCDM sheet before shell crossing.', state: { dim: 2, n: 32, mesh: 32, forceMesh: 32, fraction: 0.35, epoch: 'fraction', order: 4, smoothing: 0.03, view: 'sheet' } },
  { id: 'approach', label: 'Crossing sequence (EdS)', note: 'Use an Einstein–de Sitter matter-only reference to follow the 1D sheet through first crossing; post-crossing outputs remain visible with a warning.', state: { dim: 1, n: 64, forceMesh: 64, omegaM: 1, omegaB: 0, fraction: 1.2, epoch: 'fraction', order: 4, smoothing: 0.01, view: 'sheet' } },
  { id: 'coarse-mesh', label: 'Vertex vs observation grids', note: 'Vary Lagrangian vertex resolution and Eulerian observation resolution independently; maps use equal spatial scales.', state: { dim: 2, n: 32, forceMesh: 24, mesh: 64, fraction: 0.48, order: 4, view: 'density' } },
  { id: 'smoothing', label: 'Observation smoothing', note: 'Change the Gaussian observation filter and inspect volume- and mass-weighted PDFs.', state: { dim: 2, n: 32, forceMesh: 32, fraction: 0.5, order: 2, smoothing: 0.1, view: 'pdf' } },
  { id: 'lcdm-gaussian-3d', label: 'Gaussian ΛCDM volume', note: 'A 3D Gaussian realization with EH no-wiggle power; the 64³ sheet is a periodic finite-volume realization.', state: { dim: 3, n: 64, mesh: 64, forceMesh: 64, ic: 'gaussian', initialCutoff: 6, initialSmoothing: 14, epoch: 'scaleFactor', scaleFactor: .5, order: 4, view: 'sheet' } },
  { id: 'bao-2d', label: 'BAO wiggles (2D, 1000 Mpc/h)', note: 'A 256² translationally invariant ΛCDM sheet in a 1000 Mpc/h box with the EH98 acoustic-oscillation transfer, resolved to k ≈ 0.19 h/Mpc at z = 0. Compare the sheet, particle, and linear spectra in the power view.', state: { dim: 2, n: 256, mesh: 256, forceMesh: 256, ic: 'gaussian', transfer: 'bao', initialCutoff: 31, boxSize: 1000, initialSmoothing: 14, smoothing: 0, epoch: 'scaleFactor', scaleFactor: 1, order: 4, steps: 64, view: 'spectra' } },
  { id: 'bao-3d', label: 'BAO wiggles (3D, 600 Mpc/h)', note: 'A 64³ ΛCDM volume in a 600 Mpc/h box with the acoustic-oscillation transfer, resolving the first acoustic trough (k ≤ 0.073 h/Mpc) at z = 0.', state: { dim: 3, n: 64, mesh: 64, forceMesh: 64, ic: 'gaussian', transfer: 'bao', initialCutoff: 7, boxSize: 600, initialSmoothing: 14, smoothing: 0, epoch: 'scaleFactor', scaleFactor: 1, order: 4, steps: 64, view: 'spectra' } },
  { id: 'estimators', label: 'Sheet vs CIC/TSC (same vertices)', note: 'Deposit the same 64² LPT vertices on a 256² observation mesh four times finer than the particle spacing: the exact sheet stays smooth and mass-conserving while CIC and TSC show empty cells and a shot-noise floor.', state: { dim: 2, n: 64, mesh: 256, forceMesh: 64, ic: 'gaussian', initialCutoff: 7, smoothing: 0, epoch: 'scaleFactor', scaleFactor: 1, order: 4, steps: 64, view: 'density' } },
  { id: 'rsd', label: 'Redshift-space distortions (Kaiser & FoG)', note: 'Compare the real-space sheet, the exact redshifted simplicial sheet, and simplex velocity-gradient Gaussian packets across Kaiser squashing, redshift-space shell crossing, and Fingers-of-God dispersion.', state: { dim: 2, n: 64, mesh: 128, forceMesh: 64, ic: 'gaussian', transfer: 'bao', initialCutoff: 12, boxSize: 500, initialSmoothing: 8, smoothing: 0, epoch: 'scaleFactor', scaleFactor: 1, order: 4, steps: 64, view: 'rsd' } },
  { id: 'published', label: 'Published result', note: 'Replay the packaged published-run data and compare it with this browser model.', state: { view: 'published' } },
];

export function presetState(id) {
  const preset = PRESETS.find((p) => p.id === id);
  return { ...makeDefaults(), ...(preset ? preset.state : {}) };
}
