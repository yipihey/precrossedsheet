import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import init, {lab_run, PendingLabRun, pm_drift_minimum} from '../app/pkg/hcc_core.js';
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
