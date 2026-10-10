import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';

const source=await readFile(new URL('../app/js/state.js',import.meta.url),'utf8');
const state=await import(`data:text/javascript;base64,${Buffer.from(source).toString('base64')}`);
const defaults=state.makeDefaults();
assert.equal(defaults.schemaVersion,2);
assert.equal(state.decodeHash('').dim,2);
assert.match(state.encodeHash(defaults),/(^|&)v=2(&|$)/);
for(const [key,hashKey] of state.HASH_SCHEMA)assert.match(state.encodeHash(defaults),new RegExp(`(^|&)${hashKey}=`),`complete link omits ${key}`);

const experiment={...defaults,dim:3,n:64,fraction:.6234567891234567,epoch:'scaleFactor',scaleFactor:1.2345678901234567,order:3,seed:2650,ic:'gaussian',initialCutoff:6,omegaM:.31500000000001,omegaB:.049,h:.674,ns:.965,sigma8:.811,boxSize:300.123456789,initialSmoothing:14.25,smoothing:.047123456789,steps:512,mesh:128,forceMesh:64,view:'gradient',weight:'mass',rotate:-51.25,tilt:17.125,pubmodel:'nbody',pubest:'cic',pubr:42,paired:false};
const encoded=state.encodeHash(experiment),decoded=state.decodeHash(encoded);
assert.equal(decoded.schemaVersion,2);
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

const invalid=state.decodeHash('#v=99&dim=3&n=256');
assert.deepEqual(invalid,defaults,'unknown URL schemas must not be interpreted as current state');
const malformed=state.decodeHash('#dim=4&n=99&f=NaN&order=99&seed=2650&ic=bogus&view=unknown&smoothing=-5&extra=1');
assert.equal(malformed.n,defaults.n);
assert.equal(malformed.fraction,defaults.fraction);
assert.equal(malformed.seed,2650);
assert.equal(malformed.order,4);
assert.equal(malformed.ic,defaults.ic);
assert.equal(malformed.smoothing,0);

const oversized=state.normalizeState({...defaults,dim:3,n:256,mesh:256,forceMesh:256});
assert.equal(oversized.n,256,'resource-invalid requests stay intact for explicit core rejection');
assert.equal(state.gridLimit(3,'n'),64);
assert.equal(state.gridLimit(3,'mesh'),128);
assert.equal(state.gridLimit(3,'forceMesh'),128);
assert.equal(state.gridLimit(2,'n'),256);
assert.equal(state.gridLimit(1,'n'),1024);

const request=state.coreRequest(experiment);
assert.equal(request.epoch,'scaleFactor');
assert.equal(request.scaleFactor,experiment.scaleFactor);
assert.equal(request.omegaM,experiment.omegaM);
for(const key of ['view','weight','rotate','tilt','pubmodel','pubest','pubr','paired','schemaVersion'])assert.equal(key in request,false,`${key} is UI metadata and should not enter the numerical request`);
console.log('Lab state URL and replay-manifest checks passed.');
