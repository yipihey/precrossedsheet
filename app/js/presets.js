import { makeDefaults } from './state.js';

export const PRESETS = [
  { id: 'safe-sheet', label: 'Gentle 2D sheet', note: 'A low-cost 2D translationally invariant ΛCDM sheet with matched particle and force grids.', state: { dim: 2, n: 32, mesh: 32, forceMesh: 32, fraction: 0.35, epoch: 'fraction', order: 4, smoothing: 0.03, view: 'sheet' } },
  { id: 'approach', label: 'Crossing sequence (EdS)', note: 'Use an Einstein–de Sitter matter-only reference to follow the 1D sheet through first crossing; post-crossing outputs remain visible with a warning.', state: { dim: 1, n: 64, forceMesh: 64, omegaM: 1, omegaB: 0, fraction: 1.2, epoch: 'fraction', order: 4, smoothing: 0.01, view: 'sheet' } },
  { id: 'coarse-mesh', label: 'Force/observation grids', note: 'Vary PM force and observation resolution independently; maps use equal spatial scales.', state: { dim: 2, n: 32, forceMesh: 24, mesh: 64, fraction: 0.48, order: 4, view: 'density' } },
  { id: 'smoothing', label: 'Observation smoothing', note: 'Change the Gaussian observation filter and inspect volume- and mass-weighted PDFs.', state: { dim: 2, n: 32, forceMesh: 32, fraction: 0.5, order: 2, smoothing: 0.1, view: 'pdf' } },
  { id: 'lcdm-gaussian-3d', label: 'Gaussian ΛCDM volume', note: 'A 3D Gaussian realization with EH no-wiggle power; the 64³ sheet is a periodic finite-volume realization.', state: { dim: 3, n: 64, mesh: 64, forceMesh: 64, ic: 'gaussian', initialCutoff: 6, initialSmoothing: 14, epoch: 'scaleFactor', scaleFactor: .5, order: 4, view: 'sheet' } },
  { id: 'published', label: 'Published result', note: 'Replay the packaged published-run data and compare it with this browser model.', state: { view: 'published' } },
];

export function presetState(id) {
  const preset = PRESETS.find((p) => p.id === id);
  return { ...makeDefaults(), ...(preset ? preset.state : {}) };
}
