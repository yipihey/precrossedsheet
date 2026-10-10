// The experiment state is deliberately small, readable, and fully encoded in the URL.
export const SCHEMA = {
  dim: ['1', '2', '3'], n: [16, 24, 32], fraction: [0.1, 0.85], order: [1, 4], seed: [0, 9999],
  ic: ['waves', 'gaussian'], smoothing: [0, 0.15], steps: [32, 64, 128], mesh: [32, 48, 64], forceMesh: [16, 24, 32, 48, 64],
  view: ['sheet', 'density', 'spectra', 'pdf', 'gradient', 'published'], weight: ['volume', 'mass'], rotate: [-180, 180], tilt: [-80, 80],
  pubmodel: ['1lpt', '2lpt', '3lpt', '4lpt', 'nbody'], pubest: ['sheet', 'cic'], pubr: [7, 42], paired: [true, false],
};
const DEFAULTS = Object.freeze({ dim: 2, n: 32, fraction: 0.5, order: 4, seed: 7, ic: 'waves', smoothing: 0.03, steps: 64, mesh: 32, forceMesh: 32, view: 'sheet', weight: 'volume', rotate: 35, tilt: 25, pubmodel: '4lpt', pubest: 'sheet', pubr: 14, paired: true });
export const HASH_SCHEMA = Object.freeze([
  ['dim', 'dim', 'num', [1, 3]], ['n', 'n', 'num', [16, 32]], ['fraction', 'f', 'num', [.1, .85]],
  ['order', 'order', 'num', [1, 4]], ['seed', 'seed', 'num', [0, 9999]], ['ic', 'ic', 'enum', ['waves', 'gaussian']],
  ['smoothing', 'smoothing', 'num', [0, .15]], ['steps', 'steps', 'num', [32, 128]], ['mesh', 'mesh', 'num', [32, 64]], ['forceMesh', 'force', 'num', [16, 64]],
  ['view', 'view', 'enum', ['sheet', 'density', 'spectra', 'pdf', 'gradient', 'published']],
  ['weight', 'weight', 'enum', ['volume', 'mass']], ['rotate', 'rotate', 'num', [-180, 180]], ['tilt', 'tilt', 'num', [-80, 80]],
  ['pubmodel', 'pubmodel', 'enum', ['1lpt', '2lpt', '3lpt', '4lpt', 'nbody']], ['pubest', 'pubest', 'enum', ['sheet', 'cic']], ['pubr', 'pubr', 'num', [7, 42]], ['paired', 'paired', 'bool'],
]);
const SHORT = Object.freeze(Object.fromEntries(HASH_SCHEMA.map(([stateKey, hashKey]) => [stateKey, hashKey])));

export function makeDefaults() { return { ...DEFAULTS }; }

function normalize(key, value) {
  const rule = SCHEMA[key];
  if (!rule) return undefined;
  if (key === 'paired') return value === true || value === '1' || value === 'true';
  if (['ic', 'view', 'weight', 'pubmodel', 'pubest'].includes(key)) return rule.includes(String(value)) ? String(value) : DEFAULTS[key];
  const n = Number(value);
  if (!Number.isFinite(n)) return DEFAULTS[key];
  if (key === 'dim') return [1, 2, 3].includes(Math.round(n)) ? Math.round(n) : DEFAULTS.dim;
  if (key === 'n') return [16, 24, 32].includes(Math.round(n)) ? Math.round(n) : DEFAULTS.n;
  if (key === 'steps') return [32, 64, 128].includes(Math.round(n)) ? Math.round(n) : DEFAULTS.steps;
  if (key === 'mesh') return [32, 48, 64].includes(Math.round(n)) ? Math.round(n) : DEFAULTS.mesh;
  if (key === 'forceMesh') return [16, 24, 32, 48, 64].includes(Math.round(n)) ? Math.round(n) : DEFAULTS.forceMesh;
  if (key === 'pubr') return [7, 14, 28, 42].includes(Math.round(n)) ? Math.round(n) : DEFAULTS.pubr;
  if (key === 'order' || key === 'seed') return Math.round(Math.min(rule[1], Math.max(rule[0], n)));
  return Math.min(rule[1], Math.max(rule[0], n));
}

export function normalizeState(state = {}) {
  const out = makeDefaults();
  for (const key of Object.keys(DEFAULTS)) if (key in state) out[key] = normalize(key, state[key]);
  if (out.dim === 3 && out.n > 24) {
    out.n = 24;
    if(out.forceMesh===32)out.forceMesh=24;
  }
  if (out.dim === 3 && out.mesh > 32) out.mesh = 32;
  if (out.dim === 3 && out.forceMesh > 32) out.forceMesh = 32;
  return out;
}

export function encodeHash(state) {
  const s = normalizeState(state);
  return Object.keys(DEFAULTS).filter((k) => s[k] !== DEFAULTS[k]).map((k) => `${SHORT[k]}=${encodeURIComponent(typeof s[k] === 'boolean' ? (s[k]?'1':'0') : typeof s[k] === 'number' ? String(+s[k].toPrecision(6)) : s[k])}`).join('&');
}

export function decodeHash(hash) {
  const s = makeDefaults();
  for (const pair of String(hash || '').replace(/^#/, '').split('&')) {
    if (!pair) continue;
    const i = pair.indexOf('='); if (i < 0) continue;
    const short = pair.slice(0, i), raw = pair.slice(i + 1);
    const key = Object.keys(SHORT).find((k) => SHORT[k] === short);
    if (!key) continue;
    try { const value=decodeURIComponent(raw);s[key] = key==='paired' ? value==='1' : normalize(key, value); } catch (_) { /* malformed percent encoding is ignored */ }
  }
  return normalizeState(s);
}

export function copyState(state) { return { ...normalizeState(state) }; }
