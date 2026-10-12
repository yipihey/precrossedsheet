const ELEMENT_MASSES = Object.freeze([0.25, 0.25, 0.25, 0.25]);
const FINAL_WIDTHS = Object.freeze([0.1, 0.2, 0.3, 0.4]);
const ORDERS = Object.freeze({ A: Object.freeze([0, 1, 2, 3]), B: Object.freeze([0, 2, 1, 3]) });
const SVG_NS = 'http://www.w3.org/2000/svg';
const COLORS = ['#4477aa', '#ee6677', '#228833', '#aa3377'];
let widgetSerial = 0;

function clamp(value, lo, hi) { return Math.min(hi, Math.max(lo, value)); }
function validArrangement(value) { return value === 'B' ? 'B' : 'A'; }

export function elementWidths(contrast) {
  const t = clamp(Number(contrast), 0, 1);
  if (!Number.isFinite(t)) throw new TypeError('contrast must be finite');
  return FINAL_WIDTHS.map(w => (1 - t) * 0.25 + t * w);
}

export function intervalLayout(contrast, arrangement = 'A') {
  const widths = elementWidths(contrast), order = ORDERS[validArrangement(arrangement)];
  const starts = Array(4), centers = Array(4);
  let cursor = 0;
  for (const i of order) {
    starts[i] = cursor;
    centers[i] = cursor + widths[i] / 2;
    cursor += widths[i];
  }
  return { widths, order: [...order], starts, centers };
}

export function periodicOverlap(startT, widthT, startU, widthU, separation) {
  let overlap = 0;
  const translated = startU - separation;
  for (let image = -2; image <= 2; image++) {
    const lo = Math.max(startT, translated + image);
    const hi = Math.min(startT + widthT, translated + image + widthU);
    if (hi > lo) overlap += hi - lo;
  }
  return overlap;
}

export function pairStatistics(contrast, arrangementT, arrangementU, separation) {
  const t = intervalLayout(contrast, arrangementT), u = intervalLayout(contrast, arrangementU);
  const s = clamp(Number(separation), 0, 1);
  if (!Number.isFinite(s)) throw new TypeError('separation must be finite');
  const density = t.widths.map((w, i) => ELEMENT_MASSES[i] / w);
  const overlaps = Array.from({ length: 4 }, (_, i) => Array.from({ length: 4 }, (_, j) =>
    periodicOverlap(t.starts[i], t.widths[i], u.starts[j], u.widths[j], s)));
  const pairWeights = overlaps.map((row, i) => row.map((overlap, j) => density[i] * density[j] * overlap));
  const onePlusXi = pairWeights.flat().reduce((sum, value) => sum + value, 0);
  return { overlaps, pairWeights, normalizedPairMass: pairWeights.map(row => row.map(value => value / onePlusXi)), xi: onePlusXi - 1 };
}

export function exactCorrelation(contrast, arrangementT, arrangementU, separation) {
  return pairStatistics(contrast, arrangementT, arrangementU, separation).xi;
}

export function correlationCurves(contrast, samples = 201) {
  if (!Number.isInteger(samples) || samples < 2) throw new RangeError('samples must be an integer >= 2');
  return Array.from({ length: samples }, (_, k) => {
    const separation = 0.5 * k / (samples - 1);
    return { separation, xiA: exactCorrelation(contrast, 'A', 'A', separation), xiB: exactCorrelation(contrast, 'B', 'B', separation) };
  });
}

export function separationAveragedJointMass(contrast, arrangementT, arrangementU, samples = 4096) {
  if (!Number.isInteger(samples) || samples < 1) throw new RangeError('samples must be a positive integer');
  const result = Array.from({ length: 4 }, () => Array(4).fill(0));
  for (let k = 0; k < samples; k++) {
    const s = (k + 0.5) / samples;
    const pairs = pairStatistics(contrast, arrangementT, arrangementU, s).pairWeights;
    for (let i = 0; i < 4; i++) for (let j = 0; j < 4; j++) result[i][j] += pairs[i][j] / samples;
  }
  return result;
}

export function defaultTwoPointState() { return { contrast: 1, arrangement: 'A', separation: 0.25 }; }

export function parseTwoPointState(search = '') {
  const params = new URLSearchParams(search.startsWith('?') ? search.slice(1) : search);
  const parse = (key, fallback, lo, hi) => {
    const value = params.has(key) ? Number(params.get(key)) : fallback;
    return Number.isFinite(value) ? clamp(value, lo, hi) : fallback;
  };
  return {
    contrast: parse('tp', 1, 0, 1),
    separation: parse('tps', 0.25, 0, 0.5),
    arrangement: validArrangement(params.get('tpa') || 'A'),
  };
}

export function serializeTwoPointState(state, existingSearch = '') {
  const params = new URLSearchParams(existingSearch.startsWith('?') ? existingSearch.slice(1) : existingSearch);
  params.set('tp', String(clamp(Number(state.contrast), 0, 1)));
  params.set('tps', String(clamp(Number(state.separation), 0, 0.5)));
  params.set('tpa', validArrangement(state.arrangement));
  return params.toString();
}

function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text != null) node.textContent = text;
  return node;
}
function svgEl(tag, attrs = {}, text = null) {
  const node = document.createElementNS(SVG_NS, tag);
  for (const [key, value] of Object.entries(attrs)) node.setAttribute(key, String(value));
  if (text != null) node.textContent = text;
  return node;
}
function fmt(value, digits = 4) { return Number(value).toFixed(digits); }

function addSvgTitle(svg, id, title, description) {
  const titleId = `${id}-title`, descId = `${id}-desc`;
  svg.setAttribute('role', 'img');
  svg.setAttribute('aria-labelledby', `${titleId} ${descId}`);
  svg.append(svgEl('title', { id: titleId }, title), svgEl('desc', { id: descId }, description));
}

function makeStrip(id, state) {
  const svg = svgEl('svg', { viewBox: '0 0 720 164', class: 'tp-strip', preserveAspectRatio: 'xMidYMid meet' });
  addSvgTitle(svg, id, 'Periodic interval arrangements', `Four equal-mass intervals are shown in arrangements A and B at width contrast ${fmt(state.contrast, 2)}. Dots mark interval centroids.`);
  const x0 = 94, plotW = 590, rowY = { A: 45, B: 99 }, h = 24;
  svg.append(svgEl('text', { x: 8, y: rowY.A + 5, class: 'tp-svg-label' }, 'A · 0123'));
  svg.append(svgEl('text', { x: 8, y: rowY.B + 5, class: 'tp-svg-label' }, 'B · 0213'));
  for (const arrangement of ['A', 'B']) {
    const layout = intervalLayout(state.contrast, arrangement), y = rowY[arrangement];
    svg.append(svgEl('line', { x1: x0, y1: y + h + 7, x2: x0 + plotW, y2: y + h + 7, class: 'tp-axis' }));
    for (let i = 0; i < 4; i++) {
      const x = x0 + layout.starts[i] * plotW, width = layout.widths[i] * plotW, cx = x0 + layout.centers[i] * plotW;
      svg.append(svgEl('rect', { x, y: y - h / 2, width, height: h, fill: COLORS[i], class: 'tp-interval' }));
      svg.append(svgEl('text', { x: x + width / 2, y: y + 4, class: 'tp-interval-label', 'text-anchor': 'middle' }, String(i)));
      svg.append(svgEl('line', { x1: cx, y1: y - h / 2 - 6, x2: cx, y2: y - h / 2, class: 'tp-centroid' }));
      svg.append(svgEl('circle', { cx, cy: y - h / 2 - 7, r: 3.5, class: 'tp-centroid-dot' }));
    }
  }
  for (const f of [0, 0.25, 0.5, 0.75, 1]) {
    const x = x0 + plotW * f;
    svg.append(svgEl('line', { x1: x, y1: 128, x2: x, y2: 132, class: 'tp-axis' }));
    svg.append(svgEl('text', { x, y: 143, class: 'tp-tick', 'text-anchor': 'middle' }, String(f)));
  }
  svg.append(svgEl('text', { x: x0 + plotW / 2, y: 160, class: 'tp-axis-title', 'text-anchor': 'middle' }, 'periodic coordinate x'));
  return svg;
}

function makeCorrelationPlot(id, state, curves) {
  const svg = svgEl('svg', { viewBox: '0 0 720 300', class: 'tp-plot', preserveAspectRatio: 'xMidYMid meet' });
  addSvgTitle(svg, id, 'Exact autocorrelation by separation', 'Exact periodic interval overlaps give the autocorrelation for arrangement A and the autocorrelation for arrangement B. The vertical marker follows the selected arrangement.');
  const left = 64, right = 694, top = 24, bottom = 244, width = right - left, height = bottom - top;
  const values = curves.flatMap(p => [p.xiA, p.xiB]).concat([0]);
  const lo = Math.min(...values), hi = Math.max(...values), pad = Math.max((hi - lo) * 0.08, 0.02), ymin = lo - pad, ymax = hi + pad;
  const x = s => left + 2 * s * width, y = v => bottom - (v - ymin) / (ymax - ymin) * height;
  svg.append(svgEl('line', { x1: left, y1: bottom, x2: right, y2: bottom, class: 'tp-axis' }));
  svg.append(svgEl('line', { x1: left, y1: top, x2: left, y2: bottom, class: 'tp-axis' }));
  svg.append(svgEl('line', { x1: left, y1: y(0), x2: right, y2: y(0), class: 'tp-zero' }));
  const pathFor = key => curves.map((point, i) => `${i ? 'L' : 'M'}${x(point.separation).toFixed(2)},${y(point[key]).toFixed(2)}`).join(' ');
  svg.append(svgEl('path', { d: pathFor('xiA'), class: `tp-curve tp-curve-a${state.arrangement === 'A' ? ' tp-selected' : ''}` }));
  svg.append(svgEl('path', { d: pathFor('xiB'), class: `tp-curve tp-curve-b${state.arrangement === 'B' ? ' tp-selected' : ''}` }));
  const markerX = x(state.separation), marker = pairStatistics(state.contrast, state.arrangement, state.arrangement, state.separation).xi;
  svg.append(svgEl('line', { x1: markerX, y1: top, x2: markerX, y2: bottom, class: 'tp-marker' }));
  svg.append(svgEl('circle', { cx: markerX, cy: y(marker), r: 5, class: 'tp-marker-point' }));
  for (let i = 0; i <= 4; i++) {
    const f = i / 4, xv = left + width * f, separation = f / 2;
    svg.append(svgEl('line', { x1: xv, y1: bottom, x2: xv, y2: bottom + 4, class: 'tp-axis' }));
    svg.append(svgEl('text', { x: xv, y: bottom + 20, class: 'tp-tick', 'text-anchor': 'middle' }, fmt(separation, 2)));
  }
  for (let i = 0; i <= 4; i++) {
    const value = ymin + (ymax - ymin) * i / 4, yv = y(value);
    svg.append(svgEl('text', { x: left - 8, y: yv + 4, class: 'tp-tick', 'text-anchor': 'end' }, fmt(value, 2)));
  }
  svg.append(svgEl('text', { x: left + width / 2, y: 289, class: 'tp-axis-title', 'text-anchor': 'middle' }, 'separation s'));
  svg.append(svgEl('text', { x: 16, y: top + height / 2, class: 'tp-axis-title', 'text-anchor': 'middle', transform: `rotate(-90 16 ${top + height / 2})` }, 'ξ(s)'));
  svg.append(svgEl('line', { x1: 425, y1: 12, x2: 445, y2: 12, class: 'tp-legend-a' }), svgEl('text', { x: 451, y: 16, class: 'tp-svg-label' }, 'Arrangement A'));
  svg.append(svgEl('line', { x1: 555, y1: 12, x2: 575, y2: 12, class: 'tp-legend-b' }), svgEl('text', { x: 581, y: 16, class: 'tp-svg-label' }, 'Arrangement B'));
  return { svg, xi: marker };
}

function renderElementTable(table, contrast) {
  const widths = elementWidths(contrast), density = widths.map((w, i) => ELEMENT_MASSES[i] / w);
  const head = el('thead'), hr = el('tr');
  for (const label of ['Element', 'Width / volume fraction wᵢ', 'Density ratio rᵢ', 'Mass fraction']) hr.append(el('th', '', label));
  head.append(hr);
  const body = el('tbody');
  for (let i = 0; i < 4; i++) {
    const row = el('tr');
    for (const value of [String(i), fmt(widths[i], 4), fmt(density[i], 4), fmt(ELEMENT_MASSES[i], 4)]) row.append(el('td', '', value));
    body.append(row);
  }
  table.replaceChildren(head, body);
}

function renderPairTable(table, contrast, arrangement, separation) {
  const { overlaps, pairWeights, normalizedPairMass } = pairStatistics(contrast, arrangement, arrangement, separation);
  const head = el('thead'), hr = el('tr');
  for (const label of [`${arrangement} element i`, `${arrangement} element j`, 'Volume overlap Kᵢⱼ(s) / V', 'rᵢ rⱼ Kᵢⱼ', 'Normalized pair mass']) hr.append(el('th', '', label));
  head.append(hr);
  const body = el('tbody');
  for (let i = 0; i < 4; i++) for (let j = 0; j < 4; j++) if (overlaps[i][j] > 1e-10) {
    const row = el('tr');
    for (const value of [String(i), String(j), fmt(overlaps[i][j], 5), fmt(pairWeights[i][j], 5), fmt(normalizedPairMass[i][j], 5)]) row.append(el('td', '', value));
    body.append(row);
  }
  table.replaceChildren(head, body);
}

function csvCell(value) {
  const text = value == null ? '' : String(value);
  return /[",\n]/.test(text) ? `"${text.replaceAll('"', '""')}"` : text;
}

export function twoPointCsv(state) {
  const header = ['row_type','t','arrangement','s','element_i','element_j','width_i','width_j','density_i','density_j','volume_pair_weight','unnormalized_mass_pair_weight','normalized_mass_pair_weight','xi_A','xi_B'];
  const rows = [header];
  const curves = correlationCurves(state.contrast, 501);
  const xiA = exactCorrelation(state.contrast, 'A', 'A', state.separation);
  const xiB = exactCorrelation(state.contrast, 'B', 'B', state.separation);
  rows.push(['metadata', state.contrast, state.arrangement, state.separation, '', '', '', '', '', '', '', '', '', xiA, xiB]);
  const widths = elementWidths(state.contrast), density = widths.map((w, i) => ELEMENT_MASSES[i] / w);
  for (const arrangement of ['A', 'B']) for (let i = 0; i < 4; i++) {
    rows.push(['element', state.contrast, arrangement, state.separation, i, '', widths[i], '', density[i], '', '', '', '', xiA, xiB]);
  }
  for (const arrangement of ['A', 'B']) {
    const stats = pairStatistics(state.contrast, arrangement, arrangement, state.separation);
    for (let i = 0; i < 4; i++) for (let j = 0; j < 4; j++) if (stats.overlaps[i][j] > 1e-10) {
      rows.push(['pair', state.contrast, arrangement, state.separation, i, j, widths[i], widths[j], density[i], density[j], stats.overlaps[i][j], stats.pairWeights[i][j], stats.normalizedPairMass[i][j], xiA, xiB]);
    }
  }
  for (const point of curves) rows.push(['curve', state.contrast, '', point.separation, '', '', '', '', '', '', '', '', '', point.xiA, point.xiB]);
  return rows.map(row => row.map(csvCell).join(',')).join('\n');
}

export function mountTwoPointWidget(host) {
  const id = `two-point-${++widgetSerial}`;
  const initial = parseTwoPointState(globalThis.location?.search || '');
  let state = initial;
  const root = el('section', 'two-point-widget');
  const heading = el('h3', 'tp-heading', 'Two-point geometry');
  const controls = el('div', 'tp-controls');
  const control = (labelText, input) => { const label = el('label', 'tp-control'); label.append(el('span', '', labelText), input); return label; };

  const tInput = el('input'); tInput.type = 'range'; tInput.min = '0'; tInput.max = '1'; tInput.step = '0.01'; tInput.value = String(state.contrast); tInput.setAttribute('aria-label', 'Width contrast t');
  const tReadout = el('output', 'tp-readout');
  const tWrap = el('label', 'tp-control'); tWrap.append(el('span', '', 'Width contrast t'), tInput, tReadout);
  const aSelect = el('select'); aSelect.setAttribute('aria-label', 'Selected autocorrelation arrangement');
  for (const [value, label] of [['A','Arrangement A · 0123'],['B','Arrangement B · 0213']]) { const option = el('option', '', label); option.value = value; aSelect.append(option); }
  aSelect.value = state.arrangement;
  const sInput = el('input'); sInput.type = 'range'; sInput.min = '0'; sInput.max = '0.5'; sInput.step = '0.005'; sInput.value = String(state.separation); sInput.setAttribute('aria-label', 'Separation s');
  const sReadout = el('output', 'tp-readout');
  const sWrap = el('label', 'tp-control'); sWrap.append(el('span', '', 'Separation s'), sInput, sReadout);
  controls.append(tWrap, control('Selected autocorrelation', aSelect), sWrap);

  const intro = el('p', 'tp-intro', 'Four elements each carry mass 0.25. Their widths change with t, while their order changes the exact periodic pair overlap. The centroid marks locate each interval; widths and overlaps determine each layout’s autocorrelation.');
  const stripTitle = el('h4', 'tp-subheading', 'Intervals and centroids');
  const stripSlot = el('div', 'tp-visual-slot');
  const statLine = el('p', 'tp-statline');
  const tableGrid = el('div', 'tp-table-grid');
  const oneTitle = el('h4', 'tp-subheading', 'One-point elements');
  const oneTable = el('table', 'tp-table');
  const oneSection = el('div', 'tp-table-section'); oneSection.append(oneTitle, oneTable);
  const pairTitle = el('h4', 'tp-subheading', 'Nonzero autocorrelation pair weights at selected s');
  const pairTable = el('table', 'tp-table');
  const pairSection = el('div', 'tp-table-section'); pairSection.append(pairTitle, pairTable);
  tableGrid.append(oneSection, pairSection);
  const plotTitle = el('h4', 'tp-subheading', 'Exact autocorrelation');
  const plotSlot = el('div', 'tp-visual-slot');
  const note = el('p', 'tp-note', 'ξ(s) is computed from exact interval intersections on the unit circle. The table’s volume weight is Kᵢⱼ/V; here V=1. The selected marker and pair table follow the selected layout’s autocorrelation; no centroid point-pair approximation is used. The two arrangements have the same one-point density PDF but different autocorrelations.');
  const actions = el('div', 'tp-actions');
  const copy = el('button', 'tp-button', 'Copy permalink'); copy.type = 'button';
  const csv = el('button', 'tp-button', 'Download geometry and correlation CSV'); csv.type = 'button';
  const live = el('span', 'tp-live'); live.setAttribute('aria-live', 'polite');
  actions.append(copy, csv, live);
  root.append(heading, controls, intro, stripTitle, stripSlot, statLine, tableGrid, plotTitle, plotSlot, note, actions);
  host.replaceChildren(root);

  function writeState() {
    const search = serializeTwoPointState(state, globalThis.location?.search || '');
    if (globalThis.history?.replaceState && globalThis.location) history.replaceState(null, '', `${location.pathname}?${search}${location.hash}`);
  }
  function render() {
    tReadout.value = fmt(state.contrast, 2); tReadout.textContent = fmt(state.contrast, 2);
    sReadout.value = fmt(state.separation, 3); sReadout.textContent = fmt(state.separation, 3);
    stripSlot.replaceChildren(makeStrip(`${id}-strip`, state));
    const curves = correlationCurves(state.contrast, 201), plot = makeCorrelationPlot(`${id}-plot`, state, curves);
    plotSlot.replaceChildren(plot.svg);
    const onePlus = plot.xi + 1;
    statLine.textContent = `Autocorrelation, arrangement ${state.arrangement} · ξ(${fmt(state.separation, 3)}) = ${fmt(plot.xi, 6)} · 1+ξ = ${fmt(onePlus, 6)}`;
    renderElementTable(oneTable, state.contrast);
    renderPairTable(pairTable, state.contrast, state.arrangement, state.separation);
    writeState();
  }
  tInput.addEventListener('input', () => { state = { ...state, contrast: Number(tInput.value) }; render(); });
  sInput.addEventListener('input', () => { state = { ...state, separation: Number(sInput.value) }; render(); });
  aSelect.addEventListener('change', () => { state = { ...state, arrangement: validArrangement(aSelect.value) }; render(); });
  copy.addEventListener('click', async () => {
    const url = new URL(globalThis.location?.href || 'https://example.invalid/');
    url.search = serializeTwoPointState(state, url.search);
    try { await navigator.clipboard.writeText(url.toString()); live.textContent = 'Permalink copied.'; }
    catch { live.textContent = url.toString(); }
  });
  csv.addEventListener('click', () => {
    const blob = new Blob([twoPointCsv(state)], { type: 'text/csv;charset=utf-8' });
    const url = URL.createObjectURL(blob), link = el('a');
    link.href = url; link.download = 'two-point-correlation.csv'; link.click(); URL.revokeObjectURL(url);
    live.textContent = 'Correlation CSV downloaded.';
  });
  render();
  return { getState: () => ({ ...state }), destroy: () => root.remove() };
}
