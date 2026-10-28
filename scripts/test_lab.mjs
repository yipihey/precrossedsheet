import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';

const source=await readFile(new URL('../app/js/state.js',import.meta.url),'utf8');
const state=await import(`data:text/javascript;base64,${Buffer.from(source).toString('base64')}`);
const defaults=state.makeDefaults();
assert.equal(defaults.schemaVersion,3);
assert.equal(defaults.backend,'cpu');
assert.equal(state.decodeHash('').dim,2);
assert.match(state.encodeHash(defaults),/(^|&)v=3(&|$)/);
for(const [key,hashKey] of state.HASH_SCHEMA)assert.match(state.encodeHash(defaults),new RegExp(`(^|&)${hashKey}=`),`complete link omits ${key}`);

const experiment={...defaults,dim:3,n:64,fraction:.6234567891234567,epoch:'scaleFactor',scaleFactor:1.2345678901234567,order:3,seed:2650,ic:'gaussian',transfer:'bao',initialCutoff:6,omegaM:.31500000000001,omegaB:.049,h:.674,ns:.965,sigma8:.811,boxSize:300.123456789,initialSmoothing:14.25,smoothing:.047123456789,steps:512,mesh:128,forceMesh:64,view:'gradient',weight:'mass',backend:'gpu',rotate:-51.25,tilt:17.125,pubmodel:'nbody',pubest:'cic',pubr:42,paired:false};
const encoded=state.encodeHash(experiment),decoded=state.decodeHash(encoded);
assert.equal(decoded.schemaVersion,3);
assert.deepEqual(decoded,state.normalizeState(experiment));
assert.equal(decoded.seed,2650,'integer-valued random seeds must survive normalization');
assert.equal(decoded.omegaM,experiment.omegaM,'decimal values must retain round-trip precision');
assert.equal(decoded.scaleFactor,experiment.scaleFactor);

const shortLegacy=state.decodeHash('#dim=3&f=0.62&seed=2650&cutoff=6&epoch=scaleFactor&a=0.50000000000001');
assert.equal(shortLegacy.dim,3);
assert.equal(shortLegacy.fraction,.62);
assert.equal(shortLegacy.seed,2650);
assert.equal(shortLegacy.initialCutoff,6);
assert.equal(shortLegacy.scaleFactor,.50000000000001);

const v2=state.decodeHash('#v=2&dim=3&n=48&f=0.73&seed=11&ic=gaussian&cutoff=5');
assert.equal(v2.transfer,'smooth','links without a transfer fit keep the smooth EH98 spectrum');
assert.equal(state.decodeHash('#v=3&ic=gaussian&transfer=bao').transfer,'bao');
assert.equal(state.decodeHash('#v=3&ic=gaussian&transfer=wiggly').transfer,'smooth');
assert.equal(state.coreRequest(state.decodeHash('#v=3&transfer=bao')).transfer,'bao','the transfer fit is a physical core input');
assert.equal(v2.schemaVersion,3,'version-2 links migrate into the current schema');
assert.equal(v2.dim,3);
assert.equal(v2.n,48);
assert.equal(v2.fraction,.73);
assert.equal(v2.backend,'cpu','version-2 links migrate to the CPU reference backend');
assert.match(state.encodeHash(v2),/(^|&)v=3(&|$)/,'migrated links serialize as version 3');
for(const backend of ['cpu','gpu','auto'])assert.equal(state.normalizeState({...defaults,backend}).backend,backend);

const invalid=state.decodeHash('#v=99&dim=3&n=256');
assert.deepEqual(invalid,defaults,'unknown URL schemas must not be interpreted as current state');
const malformed=state.decodeHash('#dim=4&n=99&f=NaN&order=99&seed=2650&ic=bogus&view=unknown&backend=bogus&smoothing=-5&extra=1');
assert.equal(malformed.n,defaults.n);
assert.equal(malformed.fraction,defaults.fraction);
assert.equal(malformed.seed,2650);
assert.equal(malformed.order,4);
assert.equal(malformed.ic,defaults.ic);
assert.equal(malformed.backend,'cpu');
assert.equal(malformed.smoothing,0);

const oversized=state.normalizeState({...defaults,dim:3,n:256,mesh:256,forceMesh:256});
assert.equal(oversized.n,256,'resource-invalid requests stay intact for explicit core rejection');
assert.equal(state.gridLimit(3,'n'),64);
assert.equal(state.gridLimit(3,'mesh'),128);
assert.equal(state.gridLimit(3,'forceMesh'),128);
assert.equal(state.gridLimit(2,'n'),1024);
assert.equal(state.gridLimit(1,'n'),1024);
assert.equal(state.gridLimit(3,'n',true),128);
assert.equal(state.gridLimit(3,'mesh',true),256);
assert.equal(state.gridLimit(3,'forceMesh',true),256);
assert.equal(state.gridLimit(2,'n',true),2048);
assert.equal(state.gridLimit(1,'n',true),2048);

const replayModSource = await readFile(new URL('../app/js/replay.js', import.meta.url), 'utf8');
const replayMod = await import(`data:text/javascript;base64,${Buffer.from(replayModSource).toString('base64')}`);
const mock3dResult = {
  dim: 3, n: 64, mesh: 64, forceMesh: 64, a: 0.5, D: 0.4,
  q: new Float64Array(64**3 * 3),
  x: new Float64Array(64**3 * 3),
  v: new Float64Array(64**3 * 3),
  rho: new Float32Array(64**3).fill(1.0),
  pdf: {
    logrho: [-1, 0, 1], dy: 1.0,
    volume: [0.1, 0.8, 0.1], mass: [0.1, 0.8, 0.1],
    variance: 0.25, capturedVolume: 1.0, capturedMass: 1.0,
    summary: { probabilities: [0.5], volumeQuantiles: [1.0], massQuantiles: [1.0], variance: 0.25, S3: 3.1, S4: 18.2 },
  },
  spectra: { k: [0.1, 0.2], sheet: [100, 50], pm: [100, 50] },
};
const compactManifest = replayMod.buildCompactManifest({ state: { ...defaults, dim: 3, n: 64, mesh: 64, forceMesh: 64 }, request: {}, result: mock3dResult, execution: null, build: null, runMilliseconds: 1200 });
assert.equal(Array.isArray(compactManifest.result.rho), true, 'TypedArray rho must be converted to a plain 2D slice array for JSON serialization');
assert.equal(compactManifest.result.rho.length, 64 * 64, '3D 64^3 density field must serialize as a 64x64 central 2D slice');
assert.equal(compactManifest.result.densitySliceStats.legacy_sheet_p0.totalCells, 64**3, 'Full 3D cell count must be retained in densitySliceStats');
assert.equal(Boolean(compactManifest.result.publishedReplay?.runs?.length), true, 'Saved manifest must include a publishedReplay block for the Published Replay panel');

const request=state.coreRequest(experiment);
assert.equal(request.epoch,'scaleFactor');
assert.equal(request.scaleFactor,experiment.scaleFactor);
assert.equal(request.omegaM,experiment.omegaM);
assert.deepEqual(state.executionPolicy(experiment),{requested:'gpu'});
for(const key of ['view','weight','backend','rotate','tilt','pubmodel','pubest','pubr','paired','schemaVersion'])assert.equal(key in request,false,`${key} is UI metadata and should not enter the numerical request`);

const manifest={manifestFormat:'precrossedsheet-experiment',schemaVersion:3,state:{...defaults,backend:'gpu'},request:state.coreRequest({...defaults,backend:'gpu'}),execution:{requestedBackend:'gpu',executedBackend:'cpu',fallbackReason:'device unavailable',precision:'f64 CPU reference'}};
const replay=JSON.parse(JSON.stringify(manifest));
assert.equal(replay.state.backend,'gpu','replay state retains the requested backend');
assert.equal('backend' in replay.request,false,'replay physical request remains backend-independent');
assert.equal(replay.execution.requestedBackend,'gpu');
assert.equal(replay.execution.executedBackend,'cpu');
assert.equal(replay.execution.fallbackReason,'device unavailable');

console.log('Lab state migration and replay provenance checks passed.');

import { readdir } from 'node:fs/promises';
import { execFileSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import path from 'node:path';

const appJsDir = fileURLToPath(new URL('../app/js', import.meta.url));
const entries = await readdir(appJsDir, { withFileTypes: true, recursive: true });
const jsFiles = entries.filter(e => e.isFile() && e.name.endsWith('.js')).map(e => path.join(e.parentPath || e.path, e.name));
for (const f of jsFiles) {
  execFileSync(process.execPath, ['--check', f]);
}
console.log(`Verified syntax of ${jsFiles.length} JavaScript files in app/js.`);
