import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import init, {lab_run} from '../app/pkg/hcc_core.js';
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
const bad=JSON.parse(lab_run('{"fraction":1.2}'));
assert.ok(bad.error&&!bad.rho,'Wasm must reject a post-crossing request before publishing scientific fields.');
