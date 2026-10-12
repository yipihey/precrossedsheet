import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import init, {lab_run, PendingLabRun, pm_drift_minimum, LabSession, GpuLptSeed, exact_sheet_density} from '../app/pkg/hcc_core.js';
await init({module_or_path:await readFile(new URL('../app/pkg/hcc_core_bg.wasm',import.meta.url))});
for(const dim of [1,2,3]) {
  const n=dim===3?24:32;
  const d=JSON.parse(lab_run(JSON.stringify({dim,n,forceMesh:n,fraction:.35})));
  assert.equal(d.guard?.ok,true,JSON.stringify(d.guard||d));
  for(const key of ['mass','pmMass'])assert.ok(Math.abs(d.diagnostics[key]-1)<1e-6,`${dim}D ${key}`);
  assert.ok(d.gradient.relativeError.every(e=>Number.isFinite(e)&&e<1e-5),`${dim}D mode derivatives`);
  assert.equal(d.rho.length,d.mesh**dim);
  console.log(`Wasm ${dim}D: mass and fixed-time derivatives pass.`);
}
const bad=JSON.parse(lab_run('{"fraction":-1}'));
assert.ok(bad.error&&!bad.rho,'Wasm rejects invalid parameters before allocation.');

const crossed=JSON.parse(lab_run(JSON.stringify({dim:1,n:64,forceMesh:64,fraction:1.2,omegaM:1})));
assert.equal(crossed.guard?.ok,false);
assert.ok(crossed.rho.every(Number.isFinite));
assert.ok(Math.abs(crossed.diagnostics.mass-1)<1e-6);
console.log('Wasm crossing: finite fields remain visible and mass is conserved.');

// The external exact-density boundary consumes vec4<f32> lattice positions.
// Identity maps in both supported dimensions have the independent analytic
// result rho=1 in every Eulerian cell.
for(const dim of [2,3]) {
  const n=4,mesh=8,positions=new Float32Array(n**dim*4);
  for(let i=0;i<n**dim;i++) {
    let q=i;
    for(let axis=dim-1;axis>=0;axis--){positions[4*i+axis]=(q%n)/n;q=Math.floor(q/n);}
  }
  const rho=exact_sheet_density(dim,n,mesh,positions);
  assert.ok(rho instanceof Float32Array);assert.equal(rho.length,mesh**dim);
  assert.ok(rho.every(value=>Number.isFinite(value)&&Math.abs(value-1)<1e-6),`${dim}D identity deposit`);
  const mean=rho.reduce((sum,value)=>sum+value,0)/rho.length;
  assert.ok(Math.abs(mean-1)<1e-7,`${dim}D identity mass mean`);
}
console.log('Wasm exact deposit: 2D/3D identity sheets have unit cell density.');

// A smoothing-only update should reuse all retained physical/cache tiers while
// recomputing smoothed observations identically to a fresh CPU session.
const cacheRequest={dim:2,n:16,mesh:16,forceMesh:16,steps:32,epoch:'scaleFactor',scaleFactor:.25,smoothing:0};
const cachedSession=new LabSession(),coldPending=cachedSession.prepare_pending(JSON.stringify(cacheRequest));
let coldPrepared,hotPrepared,freshSession,freshPending,freshPrepared;
try {
  coldPrepared=cachedSession.finish_cpu(coldPending);
  const coldResult=JSON.parse(coldPrepared.finish(coldPrepared.direct_cpu(),'{}'));
  const smoothRequest={...cacheRequest,smoothing:.05};
  hotPrepared=cachedSession.try_cached_observed_source(JSON.stringify(smoothRequest),'f64','cpu-f64','cpu-exact');
  assert.ok(hotPrepared,'Smoothing-only request should reuse the cached physical run.');
  const hotResult=JSON.parse(hotPrepared.finish(hotPrepared.direct_cpu(),'{}'));
  for(const level of ['initial','pending','pm','observations'])assert.equal(hotResult.cache?.[level]?.status,'hit',`${level} cache hit`);
  assert.equal(hotResult.cache.observations.smoothing,'recomputed');
  assert.ok(hotResult.cache.bytes>0&&hotResult.cache.bytes<=hotResult.cache.limitBytes);

  freshSession=new LabSession();freshPending=freshSession.prepare_pending(JSON.stringify(smoothRequest));
  freshPrepared=freshSession.finish_cpu(freshPending);
  const freshResult=JSON.parse(freshPrepared.finish(freshPrepared.direct_cpu(),'{}'));
  for(const key of ['rho','rhoPm','spectra','pdf','diagnostics','guard','gradient'])
    assert.deepEqual(hotResult[key],freshResult[key],`cached smoothing result matches fresh ${key}`);
  assert.notDeepEqual(coldResult.rho,hotResult.rho,'Changing smoothing changes the output field.');
  console.log('Wasm cache: smoothing-only update hits all four retained levels and matches a fresh run.');
} finally {
  coldPrepared?.free();hotPrepared?.free();freshPrepared?.free();coldPending.free();freshPending?.free();cachedSession.free();freshSession?.free();
}

// Exercise the nLPT construction boundary using its independent CPU raw-term
// oracle as the packed-field input, without requiring a GPU device.
const seedRequest={dim:2,n:16,mesh:16,forceMesh:16,steps:32,epoch:'scaleFactor',scaleFactor:.25};
const seed=new GpuLptSeed(JSON.stringify(seedRequest)),seedSession=new LabSession();
let seedPending,seedPrepared;
try {
  const seedMeta=JSON.parse(seed.metadata());
  assert.equal(seedMeta.fieldStride,9);
  assert.equal(seedMeta.termScalars,seedRequest.n**2*9);
  const delta=seed.delta_hat();
  assert.ok(delta instanceof Float32Array&&delta.length>0&&delta.every(Number.isFinite));
  const fields=seed.reference_fields();
  assert.ok(fields instanceof Float32Array);
  assert.equal(fields.length,seedMeta.totalScalars);
  assert.ok(fields.every(Number.isFinite));
  assert.throws(()=>seed.finish_gpu(seedSession,fields.subarray(1)),/length|packed|field/i);
  assert.deepEqual(seed.delta_hat(),delta,'Malformed fields must leave the seed available for retry.');
  seedPending=seed.finish_gpu(seedSession,fields);
  assert.equal(JSON.parse(seedPending.metadata()).initialSource,'gpu-f32');
  seedPrepared=seedSession.finish_cpu(seedPending);
  const seedResult=JSON.parse(seedPrepared.finish(seedPrepared.direct_cpu(),'{}'));
  assert.equal(seedResult.conventions.initialSource,'gpu-f32');
  assert.ok(seedResult.guard?.ok===true);
  assert.ok(seedResult.rho.every(Number.isFinite));
  console.log('Wasm nLPT seed: CPU oracle fields pass the GPU-input lifecycle and CPU finish.');
} finally {seedPrepared?.free();seedPending?.free();seed.free();seedSession.free();}

// Exercise the actual typed-array/ownership boundary used before resident PM.
const pending=new PendingLabRun(JSON.stringify({dim:2,n:16,mesh:16,forceMesh:16,steps:32,epoch:'scaleFactor',scaleFactor:.25}));
let prepared;
try {
  const meta=JSON.parse(pending.metadata());
  const x=pending.pm_x(),psi=pending.pm_displacement(),p=pending.pm_p(),dx=pending.pm_dx(),dp=pending.pm_dp(),clocks=pending.pm_clocks();
  for(const field of [x,psi,p,dx,dp]){assert.ok(field instanceof Float32Array);assert.equal(field.length,16*16*4);}
  assert.ok(Number.isFinite(meta.pmInitialMinimumJacobianDisplacementF32));
  assert.ok(clocks instanceof Float64Array);assert.equal(clocks.length,32*5);
  assert.ok(pm_drift_minimum(2,16,x,x)>0);
  assert.throws(()=>pending.finish_gpu_pm(x.subarray(1),p,dx,dp,new Float64Array(32).fill(1)),/length/);
  assert.deepEqual(pending.pm_x(),x,'Rejected GPU output must retain the initial state for fallback.');
  const reference=pending.pm_reference();
  assert.ok(reference instanceof Float64Array);assert.equal(reference.length,16*16*16);
  assert.deepEqual(pending.pm_reference(),reference,'The cached f64 PM oracle must be stable.');
  assert.equal(pending.pm_reference_quantized().length,reference.length);
  const displacementReference=pending.pm_reference_displacement_quantized();
  assert.ok(displacementReference instanceof Float64Array);
  assert.equal(displacementReference.length,reference.length);
  assert.ok(displacementReference.every(Number.isFinite));
  prepared=pending.finish_cpu();
  const fdPacked=[prepared.finite_difference_plus_hi(),prepared.finite_difference_plus_lo(),prepared.finite_difference_minus_hi(),prepared.finite_difference_minus_lo()];
  for(const field of fdPacked){assert.ok(field instanceof Float32Array);assert.equal(field.length,16*16*4);assert.ok(field.every(Number.isFinite));}
  const channels=prepared.direct_cpu(),fd=prepared.finite_difference_cpu();
  assert.equal(fd.length,4);assert.ok(fd.every(Number.isFinite));
  const result=JSON.parse(prepared.finish(channels,'{}'));
  const suppliedFd=Array.from(fd,value=>value+0.125);
  const externalResult=JSON.parse(prepared.finish_with_finite_difference(channels,suppliedFd,'{"backend":"external-fd"}'));
  assert.deepEqual(externalResult.gradient.finiteDifference,suppliedFd.map(value=>value*result.boxSize**meta.dim));
  assert.notDeepEqual(externalResult.gradient.finiteDifference,result.gradient.finiteDifference);
  assert.equal(externalResult.execution.backend,'external-fd');
  assert.throws(()=>prepared.finish_with_finite_difference(channels,fd.subarray(1),'{}'),/finite-difference|length/i);
  const badFd=fd.slice();badFd[0]=Number.NaN;
  assert.throws(()=>prepared.finish_with_finite_difference(channels,badFd,'{}'),/finite|nonfinite/i);
  assert.equal(result.conventions.pmArithmetic,'f64');
  assert.equal(result.request.forceMesh,meta.forceMesh);
  assert.ok(result.gradient.relativeError.every(e=>Number.isFinite(e)&&e<=1e-5));
  x[0]+=1;
  assert.notEqual(pending.pm_x()[0],x[0],'Wasm getter arrays must be owned copies.');
  console.log('Wasm pending PM: typed clocks/state, cached oracle, and retained CPU fallback pass.');
} finally {prepared?.free();pending.free();}
