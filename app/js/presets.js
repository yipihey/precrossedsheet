import { makeDefaults } from './state.js';

export const PRESETS = [
  { id: 'safe-sheet', label: 'Gentle sheet', note: 'Matched particle and force grids with a broad pre-crossing margin.', state: { dim: 2, n: 32, forceMesh: 32, fraction: 0.35, order: 4, smoothing: 0.03, view: 'sheet' } },
  { id: 'approach', label: 'Approach crossing', note: 'Increase the growth fraction and watch the minimum Jacobian approach zero.', state: { dim: 1, n: 32, forceMesh: 32, fraction: 0.72, order: 4, smoothing: 0.01, view: 'sheet' } },
  { id: 'coarse-mesh', label: 'Mesh comparison', note: 'Compare exact sheet deposition with CIC and particle-mesh density.', state: { dim: 2, n: 16, forceMesh: 16, fraction: 0.48, order: 4, mesh: 32, view: 'density' } },
  { id: 'smoothing', label: 'Smooth the field', note: 'Explore how observational smoothing changes the measured density PDF.', state: { dim: 2, n: 32, forceMesh: 32, fraction: 0.5, order: 2, smoothing: 0.1, view: 'pdf' } },
  { id: 'published', label: 'Published result', note: 'Replay the packaged published-run data and compare it with this browser model.', state: { view: 'published' } },
];

export function presetState(id) {
  const preset = PRESETS.find((p) => p.id === id);
  return { ...makeDefaults(), ...(preset ? preset.state : {}) };
}
