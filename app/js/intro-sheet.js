// "From particles to a sheet": an exact one-dimensional laboratory for the
// introduction. A plane-symmetric field evolves with the Zel'dovich map (exact
// in 1D before crossing); its N vertices are deposited on an M-cell mesh three
// ways: as the connected sheet (exact interval overlaps), and as CIC / TSC
// point particles. Everything here is pure JavaScript; the full Wasm lab covers
// 2D/3D and cosmological growth. Lengths are fractions of the periodic box.
import { LinePlot } from './viz/plot.js';
import { downloadText } from './viz/download.js';

// Initial density contrast δ0(q) = Σ amp cos(2π m q + 2π phase): three
// commensurate modes, so the sheet has a hierarchy of crests and voids.
export const MODES = Object.freeze([
  Object.freeze({ m: 1, amp: 0.5, phase: 0.17 }),
  Object.freeze({ m: 2, amp: 0.3, phase: 0.43 }),
  Object.freeze({ m: 3, amp: 0.15, phase: 0.7 }),
]);
export const PARTICLE_CHOICES = Object.freeze([16, 32, 64, 128, 256]);
export const MESH_CHOICES = Object.freeze([64, 128, 256, 512]);
const TWO_PI = 2 * Math.PI;
const REFERENCE_VERTICES = 16384;

export function delta0(q) { let s = 0; for (const { m, amp, phase } of MODES) s += amp * Math.cos(TWO_PI * (m * q + phase)); return s; }
/** Zel'dovich displacement per unit growth, Ψ = −∂φ/∂q with ∂²φ/∂q² = δ0. */
export function psi(q) { let s = 0; for (const { m, amp, phase } of MODES) s -= amp * Math.sin(TWO_PI * (m * q + phase)) / (TWO_PI * m); return s; }
export function jacobian(q, growth) { return 1 - growth * delta0(q); }
/** First crossing: the growth at which 1 − D·max δ0 reaches zero. */
export function crossingGrowth(samples = 4096) {
  let peak = -Infinity;
  for (let i = 0; i < samples; i++) peak = Math.max(peak, delta0(i / samples));
  return 1 / peak;
}
/** N equal-mass labels q = i/N, their positions and growth velocities dx/dD. */
export function sheetVertices(count, growth) {
  const q = new Float64Array(count), x = new Float64Array(count), v = new Float64Array(count);
  for (let i = 0; i < count; i++) { q[i] = i / count; v[i] = psi(q[i]); x[i] = q[i] + growth * v[i]; }
  return { q, x, v };
}
/** Exact periodic interval deposit of the connected sheet (folds add positive mass). */
export function depositSheet(x, cells) {
  const count = x.length, rho = new Float64Array(cells), dx = 1 / cells, mass = 1 / count;
  for (let i = 0; i < count; i++) {
    const a = x[i], b = x[(i + 1) % count] + (i + 1 === count ? 1 : 0);
    const lo = Math.min(a, b), hi = Math.max(a, b);
    if (hi === lo) { rho[mod(Math.floor(lo / dx), cells)] += mass / dx; continue; }
    const density = mass / (hi - lo), first = Math.floor(lo / dx), last = Math.ceil(hi / dx);
    for (let j = first; j < last; j++) {
      const overlap = Math.max(0, Math.min(hi, (j + 1) * dx) - Math.max(lo, j * dx));
      rho[mod(j, cells)] += density * overlap / dx;
    }
  }
  return rho;
}
/** Cloud-in-cell deposit of equal-mass points on cell-centered nodes. */
export function depositCic(x, cells) {
  const rho = new Float64Array(cells), w = cells / x.length;
  for (const value of x) {
    const s = wrap(value) * cells - 0.5, f = Math.floor(s), t = s - f;
    rho[mod(f, cells)] += (1 - t) * w; rho[mod(f + 1, cells)] += t * w;
  }
  return rho;
}
/** Triangular-shaped-cloud deposit of equal-mass points on cell-centered nodes. */
export function depositTsc(x, cells) {
  const rho = new Float64Array(cells), w = cells / x.length;
  for (const value of x) {
    const s = wrap(value) * cells - 0.5, i = Math.floor(s + 0.5), d = s - i;
    rho[mod(i - 1, cells)] += 0.5 * (0.5 - d) ** 2 * w;
    rho[mod(i, cells)] += (0.75 - d * d) * w;
    rho[mod(i + 1, cells)] += 0.5 * (0.5 + d) ** 2 * w;
  }
  return rho;
}
/** Converged cell averages of the continuum map: a very fine sheet deposit. */
export function referenceDensity(growth, cells) { return depositSheet(sheetVertices(REFERENCE_VERTICES, growth).x, cells); }
/** P(k) of the contrast on the mesh, k = 2π m / L in units of the fundamental, P = |δ_k|² L with L = 1. */
export function powerSpectrum(rho) {
  const cells = rho.length, half = Math.floor(cells / 2), k = [], p = [];
  let mean = 0; for (const v of rho) mean += v; mean /= cells;
  for (let m = 1; m <= half; m++) {
    let re = 0, im = 0;
    for (let j = 0; j < cells; j++) { const d = rho[j] / mean - 1, a = TWO_PI * m * j / cells; re += d * Math.cos(a); im -= d * Math.sin(a); }
    k.push(m); p.push((re * re + im * im) / (cells * cells));
  }
  return { k, p };
}
export function errorStats(estimate, reference) {
  let sq = 0, empty = 0, min = Infinity, max = -Infinity, mass = 0;
  for (let i = 0; i < estimate.length; i++) {
    const v = estimate[i]; sq += (v - reference[i]) ** 2; mass += v;
    if (v <= 1e-9) empty++; min = Math.min(min, v); max = Math.max(max, v);
  }
  return { rms: Math.sqrt(sq / estimate.length), empty, min, max, mass: mass / estimate.length };
}
export function defaultIntroState() { return { fraction: 0.7, particles: 64, cells: 256 }; }
export function parseIntroState(hash) {
  const state = defaultIntroState();
  const entry = String(hash || '').replace(/^#/, '').split('&').find((p) => p.startsWith('sheet='));
  if (!entry) return state;
  const [fraction, particles, cells] = entry.slice(6).split('_').map(Number);
  if (Number.isFinite(fraction)) state.fraction = Math.min(1.2, Math.max(0.05, fraction));
  if (PARTICLE_CHOICES.includes(particles)) state.particles = particles;
  if (MESH_CHOICES.includes(cells)) state.cells = cells;
  return state;
}
export function serializeIntroState(state) { return `sheet=${state.fraction}_${state.particles}_${state.cells}`; }

function mod(i, n) { return ((i % n) + n) % n; }
function wrap(v) { return v - Math.floor(v); }
function stairs(rho) {
  const cells = rho.length, x = [], y = [];
  for (let j = 0; j < cells; j++) { x.push(j / cells, (j + 1) / cells); y.push(rho[j], rho[j]); }
  return { x, y };
}
const fmt = (v, digits = 3) => Number.isFinite(v) ? Number(v).toPrecision(digits).replace(/\.?0+$/, '') : '—';

/** Mount the interactive introduction on a `.qd-widget[data-widget="intro-sheet"]` host. */
export function mountIntroSheetWidget(host) {
  const state = parseIntroState(location.hash), crossing = crossingGrowth();
  host.classList.add('intro-sheet-widget');
  host.innerHTML = '';
  const h = (tag, cls, parent, text) => { const e = document.createElement(tag); if (cls) e.className = cls; if (text != null) e.textContent = text; parent.append(e); return e; };
  h('p', 'is-kicker', host, 'Interactive · exact one-dimensional sheet');
  const controls = h('div', 'is-controls', host);
  const sliderWrap = h('label', 'is-control', controls); h('span', '', sliderWrap, 'Growth, as a fraction of the first-crossing time');
  const slider = h('input', '', sliderWrap); slider.type = 'range'; slider.min = '0.05'; slider.max = '1.2'; slider.step = '0.01'; slider.value = String(state.fraction); slider.setAttribute('aria-label', 'Growth as a fraction of the first-crossing time');
  const sliderOut = h('output', 'is-readout', sliderWrap);
  const particleWrap = h('label', 'is-control', controls); h('span', '', particleWrap, 'Vertices (particles) N');
  const particleSelect = h('select', '', particleWrap); particleSelect.setAttribute('aria-label', 'Number of sheet vertices');
  for (const n of PARTICLE_CHOICES) { const o = h('option', '', particleSelect, String(n)); o.value = String(n); }
  particleSelect.value = String(state.particles);
  const meshWrap = h('label', 'is-control', controls); h('span', '', meshWrap, 'Observation cells M');
  const meshSelect = h('select', '', meshWrap); meshSelect.setAttribute('aria-label', 'Number of observation cells');
  for (const m of MESH_CHOICES) { const o = h('option', '', meshSelect, String(m)); o.value = String(m); }
  meshSelect.value = String(state.cells);
  const status = h('p', 'is-status', host);
  const grid = h('div', 'is-grid', host);
  const panel = (title) => { const p = h('section', 'is-panel', grid); h('h4', '', p, title); return p; };
  const phasePanel = panel('The sheet in phase space: vertices joined in label order');
  const densityPanel = panel('Density on M cells from the same N vertices');
  const powerPanel = panel('Power spectrum of each estimate');
  const plots = {
    phase: new LinePlot(h('div', 'is-plot', phasePanel), { width: 560, height: 330 }),
    density: new LinePlot(h('div', 'is-plot', densityPanel), { width: 560, height: 330 }),
    power: new LinePlot(h('div', 'is-plot', powerPanel), { width: 560, height: 330 }),
  };
  const table = h('table', 'is-table', host);
  const caption = h('caption', '', table, 'Cell-average errors against the converged map (densities in units of the mean)');
  const head = h('thead', '', table), hr = h('tr', '', head);
  for (const label of ['Estimator', 'RMS error', 'Empty cells', 'Minimum', 'Maximum', 'Mean (mass)']) h('th', '', hr, label);
  const body = h('tbody', '', table);
  const actions = h('div', 'is-actions', host);
  const linkButton = h('button', 'qd-btn', actions, 'Copy permalink'); linkButton.type = 'button';
  const csvButton = h('button', 'qd-btn', actions, 'Download CSV'); csvButton.type = 'button';
  const live = h('span', 'is-live', actions); live.setAttribute('aria-live', 'polite');
  let latest = null;

  function compute() {
    const growth = state.fraction * crossing, cells = state.cells;
    const sheet = sheetVertices(state.particles, growth);
    const reference = referenceDensity(growth, cells);
    const estimates = {
      sheet: depositSheet(sheet.x, cells),
      cic: depositCic(sheet.x, cells),
      tsc: depositTsc(sheet.x, cells),
    };
    let minJ = Infinity; for (let i = 0; i < 2048; i++) minJ = Math.min(minJ, jacobian(i / 2048, growth));
    return { growth, cells, sheet, reference, estimates, minJ, crossed: minJ <= 0 };
  }
  function render() {
    latest = compute();
    const { growth, cells, sheet, reference, estimates, minJ, crossed } = latest;
    sliderOut.value = `${fmt(state.fraction, 3)} · D = ${fmt(growth, 3)}`;
    status.textContent = crossed
      ? `Beyond first crossing (minimum Jacobian ${fmt(minJ, 3)}): the sheet has folded, and the exact deposit now adds the positive mass of every stream. The one-to-one map that makes density equal to 1/J no longer holds; this is where the pre-crossing description stops.`
      : `Single stream: minimum Jacobian ${fmt(minJ, 3)} > 0, so every position has exactly one label and the density is 1/J. The N = ${state.particles} vertices are connected into N mass-carrying intervals and deposited on M = ${cells} cells; CIC and TSC deposit the same vertices as points.`;
    // phase space: continuum curve plus the vertices in label order (closing segment back to q=1)
    const fine = 1024, cx = [], cv = [];
    for (let i = 0; i <= fine; i++) { const q = i / fine; cx.push(q + growth * psi(q)); cv.push(psi(q)); }
    const vx = Array.from(sheet.x), vv = Array.from(sheet.v); vx.push(1 + growth * psi(0)); vv.push(psi(0));
    plots.phase.setAxes({ xlabel: 'position x (box units, unwrapped)', ylabel: 'velocity dx/dD', xlog: false, ylog: false });
    plots.phase.setSeries([
      { x: cx, y: cv, label: 'continuum sheet', color: '#999999', width: 1 },
      { x: vx, y: vv, label: `N = ${state.particles} vertices, joined`, color: '#0072B2', width: 1.6, points: true, radius: 2.2 },
    ]);
    plots.phase.draw();
    const ref = stairs(reference), sh = stairs(estimates.sheet), ci = stairs(estimates.cic), ts = stairs(estimates.tsc);
    plots.density.setAxes({ xlabel: 'position x / L', ylabel: 'ρ / mean (cell average)', xlog: false, ylog: false });
    plots.density.setSeries([
      { x: ref.x, y: ref.y, label: 'converged map (reference)', color: '#999999', width: 3 },
      { x: sh.x, y: sh.y, label: 'connected sheet, exact overlaps', color: '#0072B2', width: 1.6 },
      { x: ci.x, y: ci.y, label: 'CIC particles', color: '#D55E00', width: 1.2 },
      { x: ts.x, y: ts.y, label: 'TSC particles', color: '#009E73', width: 1.2 },
    ]);
    plots.density.draw();
    const spectra = { reference: powerSpectrum(reference), sheet: powerSpectrum(estimates.sheet), cic: powerSpectrum(estimates.cic), tsc: powerSpectrum(estimates.tsc) };
    const shot = spectra.reference.k.map(() => 1 / state.particles);
    const peak = Math.max(...spectra.reference.p, ...spectra.cic.p);
    plots.power.setAxes({ xlabel: 'k / k_fundamental', ylabel: 'P(k) / L', xlog: true, ylog: true, ylim: [peak * 1e-7, peak * 3] });
    plots.power.setSeries([
      { x: spectra.reference.k, y: spectra.reference.p, label: 'converged map', color: '#999999', width: 3 },
      { x: spectra.sheet.k, y: spectra.sheet.p, label: 'connected sheet', color: '#0072B2', width: 1.6 },
      { x: spectra.cic.k, y: spectra.cic.p, label: 'CIC particles', color: '#D55E00', width: 1.2 },
      { x: spectra.tsc.k, y: spectra.tsc.p, label: 'TSC particles', color: '#009E73', width: 1.2 },
      { x: spectra.reference.k, y: shot, label: 'Poisson shot noise L/N', color: '#CC79A7', width: 1, dash: '5 3' },
    ]);
    plots.power.setMarkers([{ x: state.particles / 2, label: 'N/2' }]);
    plots.power.draw();
    body.replaceChildren();
    for (const [label, values] of [['Connected sheet (exact)', estimates.sheet], ['CIC particles', estimates.cic], ['TSC particles', estimates.tsc]]) {
      const s = errorStats(values, reference), tr = h('tr', '', body);
      h('th', '', tr, label); h('td', '', tr, fmt(s.rms, 3)); h('td', '', tr, `${s.empty} of ${cells}`); h('td', '', tr, fmt(s.min, 3)); h('td', '', tr, fmt(s.max, 3)); h('td', '', tr, fmt(s.mass, 6));
    }
    caption.textContent = `Cell-average errors against the converged map at D = ${fmt(growth, 3)} (densities in units of the mean; the converged map has minimum ${fmt(Math.min(...reference), 3)} and maximum ${fmt(Math.max(...reference), 3)})`;
    history.replaceState(null, '', `#${serializeIntroState(state)}`);
  }
  slider.addEventListener('input', () => { state.fraction = Number(slider.value); render(); });
  particleSelect.addEventListener('change', () => { state.particles = Number(particleSelect.value); render(); });
  meshSelect.addEventListener('change', () => { state.cells = Number(meshSelect.value); render(); });
  linkButton.addEventListener('click', async () => {
    const url = `${location.origin}${location.pathname}#${serializeIntroState(state)}`;
    try { await navigator.clipboard.writeText(url); live.textContent = 'Permalink copied.'; } catch { live.textContent = url; }
  });
  csvButton.addEventListener('click', () => {
    if (!latest) return;
    const rows = ['cell,x_center,reference,sheet,cic,tsc'];
    for (let j = 0; j < latest.cells; j++) rows.push([j, (j + 0.5) / latest.cells, latest.reference[j], latest.estimates.sheet[j], latest.estimates.cic[j], latest.estimates.tsc[j]].join(','));
    downloadText(rows.join('\n'), `intro-sheet-D${fmt(latest.growth, 4)}-N${state.particles}-M${latest.cells}.csv`, 'text/csv');
    live.textContent = 'CSV downloaded.';
  });
  render();
}
