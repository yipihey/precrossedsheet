import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';

// Load the browser ES module as a data URL so this check needs no package or build setup.
const source = await readFile(new URL('../app/js/state.js', import.meta.url), 'utf8');
const state = await import(`data:text/javascript;base64,${Buffer.from(source).toString('base64')}`);
const defaults = state.makeDefaults();
assert.equal(state.decodeHash('').dim, 2);
assert.equal(state.encodeHash(defaults), '');

const experiment = { ...defaults, dim: 3, n: 24, fraction: .623456789, order: 3, seed: 81, ic: 'gaussian', smoothing: .04, steps: 128, mesh: 32, forceMesh: 24, view: 'gradient', weight: 'mass', rotate: -51, tilt: 17, pubmodel: 'nbody', pubest: 'cic', pubr: 42, paired: false };
const roundTrip = state.decodeHash(state.encodeHash(experiment));
assert.deepEqual(roundTrip, { ...state.normalizeState(experiment), fraction: .623457 });

const untrusted = state.decodeHash('#dim=4&n=99&f=NaN&order=99&ic=bogus&view=unknown&smoothing=-5&extra=1');
assert.deepEqual(untrusted, { ...defaults, n: 16, order: 4, smoothing: 0 });
const capped3d = state.normalizeState({ ...defaults, dim: 3, n: 32, mesh: 64, forceMesh: 64 });
assert.equal(capped3d.n, 24);
assert.equal(capped3d.mesh, 32);
assert.equal(capped3d.forceMesh, 32);
assert.deepEqual(state.decodeHash('#f=0.73&view=published'), { ...defaults, fraction: .73, view: 'published' });
console.log('Lab state URL codec checks passed.');
