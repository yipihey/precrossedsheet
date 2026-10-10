// Versioned, complete experiment state. URLs record defaults as well as edits so
// a future default change cannot silently alter an old experiment link.
export const STATE_VERSION = 2;
export const GRIDS = Object.freeze([16, 24, 32, 48, 64, 96, 128, 256, 512, 1024]);
export const STEPS = Object.freeze([32, 64, 128, 256, 512]);
export const LIMITS = Object.freeze({
  1: { n: 1024, mesh: 1024, forceMesh: 1024 },
  2: { n: 256, mesh: 256, forceMesh: 256 },
  3: { n: 64, mesh: 128, forceMesh: 128 },
});
export const SCHEMA = Object.freeze({
  dim: [1, 3], n: GRIDS, fraction: [.05, 1.5], epoch: ['fraction', 'scaleFactor'], scaleFactor: [.01, 2],
  order: [1, 4], seed: [0, 9999], ic: ['waves', 'gaussian'], initialCutoff: [1, 32],
  omegaM: [.05, 1], omegaB: [0, .2], h: [.4, 1], ns: [.8, 1.2], sigma8: [.1, 1.5],
  boxSize: [50, 2000], initialSmoothing: [0, 200], smoothing: [0, .15], steps: STEPS,
  mesh: GRIDS, forceMesh: GRIDS, view: ['sheet', 'density', 'spectra', 'pdf', 'gradient', 'published'],
  weight: ['volume', 'mass'], rotate: [-180, 180], tilt: [-80, 80], schemaVersion: [STATE_VERSION, STATE_VERSION],
  pubmodel: ['1lpt', '2lpt', '3lpt', '4lpt', 'nbody'], pubest: ['sheet', 'cic'], pubr: [7, 14, 28, 42], paired: [true, false],
});
const DEFAULTS = Object.freeze({
  dim: 2, n: 32, fraction: .5, epoch: 'fraction', scaleFactor: .5, order: 4, seed: 7,
  ic: 'waves', initialCutoff: 3, omegaM: .315, omegaB: .049, h: .674, ns: .965, sigma8: .811,
  boxSize: 300, initialSmoothing: 14, smoothing: .03, steps: 64, mesh: 32, forceMesh: 32,
  view: 'sheet', weight: 'volume', rotate: 35, tilt: 25,
  pubmodel: '4lpt', pubest: 'sheet', pubr: 14, paired: true, schemaVersion: STATE_VERSION,
});

// The long state key and compact fragment key are both part of the checked
// chapter-link schema. Keep this explicit so book links remain auditable.
export const HASH_SCHEMA = Object.freeze([
  ['dim', 'dim', 'num', [1, 3]], ['n', 'n', 'num', [16, 1024]], ['fraction', 'f', 'num', [.05, 1.5]],
  ['epoch', 'epoch', 'enum', ['fraction', 'scaleFactor']], ['scaleFactor', 'a', 'num', [.01, 2]],
  ['order', 'order', 'num', [1, 4]], ['seed', 'seed', 'num', [0, 9999]], ['ic', 'ic', 'enum', ['waves', 'gaussian']],
  ['initialCutoff', 'cutoff', 'num', [1, 32]], ['omegaM', 'Om', 'num', [.05, 1]], ['omegaB', 'Ob', 'num', [0, .2]],
  ['h', 'h', 'num', [.4, 1]], ['ns', 'ns', 'num', [.8, 1.2]], ['sigma8', 's8', 'num', [.1, 1.5]],
  ['boxSize', 'L', 'num', [50, 2000]], ['initialSmoothing', 'Rini', 'num', [0, 200]], ['smoothing', 'smoothing', 'num', [0, .15]],
  ['steps', 'steps', 'num', [32, 512]], ['mesh', 'mesh', 'num', [16, 1024]], ['forceMesh', 'force', 'num', [16, 1024]],
  ['schemaVersion', 'v', 'num', [STATE_VERSION, STATE_VERSION]],
  ['view', 'view', 'enum', ['sheet', 'density', 'spectra', 'pdf', 'gradient', 'published']],
  ['weight', 'weight', 'enum', ['volume', 'mass']], ['rotate', 'rotate', 'num', [-180, 180]], ['tilt', 'tilt', 'num', [-80, 80]],
  ['pubmodel', 'pubmodel', 'enum', ['1lpt', '2lpt', '3lpt', '4lpt', 'nbody']], ['pubest', 'pubest', 'enum', ['sheet', 'cic']],
  ['pubr', 'pubr', 'num', [7, 42]], ['paired', 'paired', 'bool'],
]);
const HASH_KEY = Object.freeze(Object.fromEntries(HASH_SCHEMA.map(([stateKey, hashKey]) => [stateKey, hashKey])));
const ENUMS = new Set(['epoch', 'ic', 'view', 'weight', 'pubmodel', 'pubest']);
const GRID_CHOICES = new Set(['n', 'steps', 'mesh', 'forceMesh']);
const INTEGER_RANGE = new Set(['dim', 'order', 'seed', 'initialCutoff', 'schemaVersion']);
const ROTATION_RANGE = new Set(['rotate','tilt']);

export function makeDefaults() { return { ...DEFAULTS }; }

function normalize(key, value) {
  const rule = SCHEMA[key];
  if (!rule) return undefined;
  if (key === 'paired') return value === true || value === '1' || value === 'true';
  if (ENUMS.has(key)) return rule.includes(String(value)) ? String(value) : DEFAULTS[key];
  const n = Number(value);
  if (!Number.isFinite(n)) return DEFAULTS[key];
  if (GRID_CHOICES.has(key)) return rule.includes(Math.round(n)) ? Math.round(n) : DEFAULTS[key];
  if (key==='pubr') return rule.includes(Math.round(n)) ? Math.round(n) : DEFAULTS[key];
  if (INTEGER_RANGE.has(key)) return Math.round(Math.min(rule[1], Math.max(rule[0], n)));
  if (ROTATION_RANGE.has(key)) return Math.max(rule[0],Math.min(rule[1],n));
  if (Array.isArray(rule)) return Math.min(rule[1], Math.max(rule[0], n));
  return DEFAULTS[key];
}

export function normalizeState(state = {}) {
  const out = makeDefaults();
  for (const key of Object.keys(DEFAULTS)) if (key in state) out[key] = normalize(key, state[key]);
  return out;
}

function serialValue(value) { return typeof value === 'boolean' ? (value ? '1' : '0') : String(value); }

export function encodeHash(state) {
  const s = normalizeState(state);
  return HASH_SCHEMA.map(([key]) => `${HASH_KEY[key]}=${encodeURIComponent(serialValue(s[key]))}`).join('&');
}

export function decodeHash(hash) {
  const pairs = String(hash || '').replace(/^#/, '').split('&');
  const rawVersion = pairs.find((p) => p.startsWith('v='))?.slice(2);
  if (rawVersion != null && rawVersion !== String(STATE_VERSION)) return makeDefaults();
  const out = makeDefaults();
  for (const pair of pairs) {
    if (!pair) continue;
    const i = pair.indexOf('='); if (i < 0) continue;
    const short = pair.slice(0, i), raw = pair.slice(i + 1);
    const key = Object.keys(HASH_KEY).find((k) => HASH_KEY[k] === short);
    if (!key || key === 'schemaVersion') continue;
    try { out[key] = normalize(key, decodeURIComponent(raw)); } catch (_) { /* malformed component leaves default */ }
  }
  out.schemaVersion = STATE_VERSION;
  return normalizeState(out);
}

export function copyState(state) { return { ...normalizeState(state) }; }

export function coreRequest(state) {
  const { view, weight, rotate, tilt, pubmodel, pubest, pubr, paired, schemaVersion, ...request } = normalizeState(state);
  return request;
}

export function gridLimit(dim, key) { return LIMITS[Number(dim)]?.[key] ?? 16; }
