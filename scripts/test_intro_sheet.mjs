import assert from 'node:assert/strict';
import {
  MODES, crossingGrowth, delta0, depositCic, depositSheet, depositTsc, errorStats, jacobian,
  parseIntroState, powerSpectrum, psi, referenceDensity, serializeIntroState, sheetVertices,
} from '../app/js/intro-sheet.js';

const close = (a, b, tol, msg) => assert.ok(Math.abs(a - b) <= tol, `${msg || ''} ${a} != ${b} ± ${tol}`);
const mean = (v) => Array.from(v).reduce((s, x) => s + x, 0) / v.length;

// Ψ is the Zel'dovich displacement of δ0: dΨ/dq = −δ0, checked by a centered difference.
for (const q of [0.03, 0.21, 0.5, 0.77]) {
  const h = 1e-5;
  close((psi(q + h) - psi(q - h)) / (2 * h), -delta0(q), 1e-6, 'dΨ/dq');
}
assert.equal(MODES.length, 3);
const dsc = crossingGrowth();
assert.ok(dsc > 0 && dsc < 2);
let minJ = Infinity; for (let i = 0; i < 8192; i++) minJ = Math.min(minJ, jacobian(i / 8192, dsc));
close(minJ, 0, 2e-5, 'J vanishes at the crossing growth');
console.log(`Crossing growth D_sc = ${dsc.toFixed(5)}.`);

// Mass conservation and the uniform limit for every estimator.
for (const cells of [64, 256]) for (const count of [16, 64, 256]) {
  const { x } = sheetVertices(count, 0.6 * dsc);
  for (const [name, rho] of [['sheet', depositSheet(x, cells)], ['cic', depositCic(x, cells)], ['tsc', depositTsc(x, cells)]]) {
    close(mean(rho), 1, 1e-12, `${name} N=${count} M=${cells} mass`);
    assert.ok(Array.from(rho).every((v) => v >= 0), `${name} nonnegative`);
  }
  const lattice = sheetVertices(count, 0).x, uniform = (rho) => Array.from(rho).every((v) => Math.abs(v - 1) < 1e-12);
  assert.ok(uniform(depositSheet(lattice, cells)), 'the undeformed sheet is uniform on any mesh');
  if (cells === count) assert.ok(uniform(depositCic(lattice, cells)) && uniform(depositTsc(lattice, cells)), 'point deposits are uniform only on a matching mesh');
  else if (cells > count) assert.ok(!uniform(depositCic(lattice, cells)), 'a finer mesh resolves the particles');
}

// The exact interval deposit converges to the continuum map: before crossing the map is
// monotone, so the exact mass in a cell is the label measure q(x_{j+1}) − q(x_j) found by bisection.
{
  const growth = 0.7 * dsc, cells = 128, reference = referenceDensity(growth, cells);
  const position = (q) => q + growth * psi(q);
  const inverse = (x) => { let lo = -1, hi = 2; for (let i = 0; i < 80; i++) { const mid = 0.5 * (lo + hi); if (position(mid) < x) lo = mid; else hi = mid; } return 0.5 * (lo + hi); };
  for (let j = 0; j < cells; j++) {
    const exact = (inverse((j + 1) / cells) - inverse(j / cells)) * cells;
    close(reference[j] / exact, 1, 1e-4, `cell ${j}`);
  }
  // The connected sheet with few vertices is far closer to the converged map than point deposits.
  const { x } = sheetVertices(32, growth);
  const sheet = errorStats(depositSheet(x, cells), reference), cic = errorStats(depositCic(x, cells), reference), tsc = errorStats(depositTsc(x, cells), reference);
  assert.ok(sheet.rms < 0.25 * cic.rms && sheet.rms < 0.25 * tsc.rms, `sheet ${sheet.rms} cic ${cic.rms} tsc ${tsc.rms}`);
  assert.equal(sheet.empty, 0);
  assert.ok(cic.empty > 0 && tsc.empty > 0, 'point deposits leave empty cells when M > N');
  console.log(`N=32 on M=128 cells: RMS sheet ${sheet.rms.toFixed(4)}, CIC ${cic.rms.toFixed(4)} (${cic.empty} empty), TSC ${tsc.rms.toFixed(4)} (${tsc.empty} empty).`);
  // Point deposits carry a white floor near the Poisson level; the sheet does not.
  const ps = powerSpectrum(depositSheet(x, cells)), pc = powerSpectrum(depositCic(x, cells)), pr = powerSpectrum(reference);
  const tail = (p) => p.k.map((k, i) => [k, p.p[i]]).filter(([k]) => k > 24 && k <= 64).map(([, v]) => v);
  const tailMean = (p) => mean(tail(p));
  assert.ok(tailMean(pc) > 0.25 / 32 && tailMean(pc) < 4 / 32, `CIC tail ${tailMean(pc)} near L/N = ${1 / 32}`);
  assert.ok(tailMean(ps) < 0.1 * tailMean(pc), `sheet tail ${tailMean(ps)} well below the shot floor`);
  assert.ok(Math.abs(ps.p[0] / pr.p[0] - 1) < 0.05, 'fundamental mode agrees');
}

// Folds: beyond crossing the exact deposit still conserves mass and stays positive.
{
  const { x } = sheetVertices(64, 1.2 * dsc), rho = depositSheet(x, 256);
  close(mean(rho), 1, 1e-12, 'crossed mass'); assert.ok(Array.from(rho).every((v) => v >= 0));
}
// Permalink round trip and validation.
const s = { fraction: 0.85, particles: 128, cells: 512 };
assert.deepEqual(parseIntroState(`#${serializeIntroState(s)}`), s);
assert.deepEqual(parseIntroState('#sheet=9_7_3'), { fraction: 1.2, particles: 64, cells: 256 });
assert.deepEqual(parseIntroState(''), { fraction: 0.7, particles: 64, cells: 256 });
console.log('Introduction sheet widget checks passed.');
