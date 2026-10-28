import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { Worker } from 'node:worker_threads';

const stateSource=await readFile(new URL('../app/js/state.js',import.meta.url),'utf8');
const state=await import(`data:text/javascript;base64,${Buffer.from(stateSource).toString('base64')}`);
const workerUrl=new URL('../app/js/worker.js',import.meta.url).href;
const coreUrl=new URL('../app/pkg/hcc_core.js?v=20261027h',import.meta.url).href;
const wasmPath=new URL('../app/pkg/hcc_core_bg.wasm',import.meta.url).pathname;

// Bootstrap preinitializes the real wasm-bindgen module with bytes, then loads
// the production worker behind a Worker-like self/onmessage shim.
const bootstrap=`
const {parentPort,workerData}=require('node:worker_threads');
const fs=require('node:fs/promises');
(async()=>{
  if(workerData.mockGpu)Object.defineProperty(globalThis,'navigator',{configurable:true,value:{gpu:{requestAdapter:()=>new Promise(resolve=>setTimeout(()=>resolve(null),100))}}});
  const core=await import(workerData.coreUrl);
  await core.default({module_or_path:new Uint8Array(await fs.readFile(workerData.wasmPath))});
  globalThis.self={postMessage:(message)=>parentPort.postMessage(message),onmessage:null};
  await import(workerData.workerUrl);
  parentPort.on('message',(data)=>globalThis.self.onmessage?.({data}));
  parentPort.postMessage({type:'booted'});
})().catch(error=>parentPort.postMessage({type:'bootstrap-error',error:String(error?.stack||error)}));
`;

function request(dim=1,n=16,backend='cpu'){
  return state.coreRequest(state.normalizeState({dim,n,backend,ic:'waves',order:1,epoch:'scaleFactor',scaleFactor:.2,fraction:.3,
    initialSmoothing:0,steps:32,mesh:16,forceMesh:16,smoothing:0}));
}
function startWorker(mockGpu=false){
  const worker=new Worker(bootstrap,{eval:true,workerData:{workerUrl,coreUrl,wasmPath,mockGpu}});
  const messages=[],waiters=[];
  worker.on('message',(message)=>{
    const idx=waiters.findIndex((w)=>w.predicate(message));
    if(idx>=0){const [waiter]=waiters.splice(idx,1);clearTimeout(waiter.timer);waiter.resolve(message);}else messages.push(message);
  });
  worker.on('error',(error)=>{for(const waiter of waiters.splice(0)){clearTimeout(waiter.timer);waiter.reject(error);}});
  const next=(predicate,timeoutMs=60000)=>{
    const index=messages.findIndex(predicate);if(index>=0)return Promise.resolve(messages.splice(index,1)[0]);
    return new Promise((resolve,reject)=>{const waiter={predicate,resolve,reject,timer:setTimeout(()=>{const i=waiters.indexOf(waiter);if(i>=0)waiters.splice(i,1);reject(new Error(`Timed out waiting for worker message; buffered: ${messages.map(m=>m.type+':'+m.requestId).join(', ')}`));},timeoutMs)};waiters.push(waiter);});
  };
  return {worker,messages,next,send:(message)=>worker.postMessage(message)};
}
async function boot(client){
  const message=await client.next((m)=>m.type==='booted'||m.type==='bootstrap-error',30000);
  if(message.type==='bootstrap-error')throw new Error(message.error);
}
const runMessage=(id,req,backend='cpu')=>({type:'run',requestId:id,key:`run-${id}`,state:req,policy:{requested:backend}});
async function resultFor(client,id){return client.next((m)=>m.requestId===id&&(m.type==='result'||m.type==='cancelled'),90000);}
async function shutdown(client){
  client.send({type:'shutdown'});
  await client.next((m)=>m.type==='shutdown-ack',10000);
  await client.worker.terminate();
}

const client=startWorker();
await boot(client);
client.send(runMessage(1,request()));
const first=await resultFor(client,1);
assert.equal(first.type,'result',JSON.stringify(first));
assert.equal(first.result.execution.executedBackend,'cpu');
assert.equal(first.result.execution.requestedBackend,'cpu');
assert.equal(first.result.execution.stageBackends?.trajectories?.statePrecision,'f64');
assert.equal(first.result.execution.stageBackends?.trajectories?.gpuPackedInputPrecision,'f32');
assert.equal(first.result.execution.stageBackends?.exactDeposit?.backend,'cpu');
assert.equal(first.result.execution.stageBackends?.exactDeposit?.inputPrecision,'f32 positions');
assert.equal(first.result.execution.stageBackends?.exactDeposit?.outputPrecision,'f32 density field');
assert.equal(first.result.execution.stageBackends?.directPrimal?.lpt?.provenance,'cpu-f64');
assert.equal(first.result.execution.stageBackends?.finiteDifference?.precision,'f64');

// A newer request received during synchronous preparation cancels that run;
// the one most recently queued request is the one that completes.
client.send(runMessage(2,request(3,16)));
await client.next((m)=>m.type==='phase'&&m.requestId===2&&m.phase==='preparing',30000);
client.send(runMessage(3,request(1,16)));
client.send(runMessage(4,request(1,24)));
const superseded=await resultFor(client,2);
assert.equal(superseded.type,'cancelled');
const latest=await resultFor(client,4);
assert.equal(latest.type,'result',JSON.stringify(latest));
assert.equal(latest.key,'run-4');
assert.equal(client.messages.some((m)=>m.requestId===3&&m.type==='result'),false,'an obsolete queued request must not produce a result');

// A stale cancellation ID must not cancel a newer generation.
client.send(runMessage(5,request()));
await client.next((m)=>m.type==='phase'&&m.requestId===5&&m.phase==='preparing',30000);
client.send({type:'cancel',requestId:4});
const afterStaleCancel=await resultFor(client,5);
assert.equal(afterStaleCancel.type,'result',JSON.stringify(afterStaleCancel));
const multiEstReq={...request(2,16),order:4,estimators:['2lpt:p0','3lpt:p0','4lpt:p0','pm:p0','4lpt:p1','pm:p1','4lpt:ngp','4lpt:cic','4lpt:tsc','pm:cic']};
client.send(runMessage(51,multiEstReq));
const multiEstRes=await resultFor(client,51);
assert.equal(multiEstRes.type,'result',JSON.stringify(multiEstRes));
for(const key of multiEstReq.estimators){
  const entry=multiEstRes.result.customSeries?.[key];
  assert.ok(entry,`customSeries missing ${key}`);
  assert.equal(entry.rho.length,256,`rho length mismatch for ${key}`);
  assert.ok(entry.power.length>0,`power empty for ${key}`);
  assert.ok(entry.variance.tophat.length>0,`tophat variance empty for ${key}`);
  assert.equal(entry.pdf.volume.length,80,`pdf volume length mismatch for ${key}`);
  if(key.endsWith(':p0'))assert.ok(Array.isArray(entry.directShell)&&entry.directShell.length>0,`directShell missing for ${key}`);
}
await shutdown(client);

// Test the async GPU cancellation boundary without requiring a GPU: the
// worker sees a capability API whose adapter request resolves to null later.
const gpuClient=startWorker(true);
await boot(gpuClient);
gpuClient.send(runMessage(6,request(2,16,'gpu'),'gpu'));
await gpuClient.next((m)=>m.type==='phase'&&m.requestId===6&&m.phase==='direct'&&m.cancellable===true,30000);
gpuClient.send({type:'cancel',requestId:6});
const gpuCancelled=await resultFor(gpuClient,6);
assert.equal(gpuCancelled.type,'cancelled');
gpuClient.send(runMessage(7,request()));
const afterGpuCancel=await resultFor(gpuClient,7);
assert.equal(afterGpuCancel.type,'result',JSON.stringify(afterGpuCancel));
await shutdown(gpuClient);

console.log('Persistent numerical worker CPU, latest-queue, stale-cancel, async GPU-cancel, and shutdown checks passed.');
