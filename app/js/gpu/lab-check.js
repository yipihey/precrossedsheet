// Complete-request comparisons keep Rust physics preparation identical and
// isolate input quantization, WGSL arithmetic and the amplitude derivative gate.
import init, {PreparedLabRun} from '../../pkg/hcc_core.js';
import {runDirect} from './direct.js';

const now=()=>performance.now();
const max=(a)=>Math.max(0,...a);
const median=(a)=>[...a].sort((x,y)=>x-y)[Math.floor(a.length/2)];
const summary=(a)=>({median:median(a),minimum:Math.min(...a),maximum:Math.max(...a),samples:a});
function comparison(actual,reference) {
  let absolute=0,scaled=0,failed=0;
  for(let i=0;i<actual.length;i++) {
    const error=Math.abs(actual[i]-reference[i]);
    absolute=Math.max(absolute,error);
    scaled=Math.max(scaled,error/(1e-6+1e-4*Math.abs(reference[i])));
    if(!Number.isFinite(actual[i])||error>1e-6+1e-4*Math.abs(reference[i]))failed++;
  }
  return {maximumAbsolute:absolute,maximumBudgetFraction:scaled,failedComponents:failed};
}
function inputs(p) {
  const m=JSON.parse(p.metadata());
  return {dim:m.dim,n:m.n,modes:p.mode_records(),lptPsi:p.lpt_displacement(),lptDirection:p.lpt_direction(),pmPsi:p.pm_displacement(),pmDirection:p.pm_direction()};
}
function decode(p,c) {return JSON.parse(p.finish(c,JSON.stringify({test:'GPU complete-request comparison'})));}
function recover(p,channels) {
  const result=decode(p,channels),failed=[];
  result.gradient.relativeError.forEach((e,i)=>{if(!Number.isFinite(e)||e>1e-5)failed.push(i);});
  const recovered=new Float64Array(channels),started=now();
  if(failed.length){const cpu=p.direct_lpt_cpu_modes(Uint32Array.from(failed));failed.forEach((m,i)=>recovered.set(cpu.subarray(4*i,4*i+4),8*m));}
  return {channels:recovered,result:decode(p,recovered),failed,cpuRecoveryMs:now()-started};
}

const fixtures=[
  {dim:2,n:16,order:1,omegaM:.315,scaleFactor:.25},
  {dim:2,n:24,order:2,omegaM:.315,scaleFactor:.5},
  {dim:2,n:32,order:3,omegaM:1,epoch:'fraction',fraction:.35},
  {dim:2,n:48,order:4,omegaM:.315,scaleFactor:.25},
  {dim:2,n:96,order:4,omegaM:.315,scaleFactor:.5,ic:'gaussian',initialCutoff:3,seed:2650},
  {dim:3,n:16,order:1,omegaM:1,scaleFactor:.25},
  {dim:3,n:24,order:2,omegaM:.315,scaleFactor:.5},
  {dim:3,n:32,order:4,omegaM:.315,scaleFactor:.5,ic:'gaussian',initialCutoff:3,seed:2650},
  {dim:3,n:16,order:4,omegaM:1,epoch:'fraction',fraction:1.2},
];
function request(settings) {
  return {epoch:'scaleFactor',fraction:.35,steps:32,mesh:settings.n,forceMesh:settings.n,omegaB:.049,h:.674,ns:.965,sigma8:.811,boxSize:300,initialSmoothing:14,initialCutoff:3,seed:7,ic:'waves',smoothing:.03,...settings};
}

export async function runLabChecks({device,onProgress=()=>{},benchmark=false}={}) {
  await init();
  const rows=[],cases=[],timings=[],numericalErrors=[];
  for(let i=0;i<fixtures.length;i++) {
    const r=request(fixtures[i]);onProgress({stage:'Complete request',caseIndex:i+1,cases:fixtures.length});
    let p;
    try {
      const started=now();p=new PreparedLabRun(JSON.stringify(r));const prepareMs=now()-started;
      const input=inputs(p),gpu=await runDirect(device,input);
      const cpuStart=now(),cpu=p.direct_cpu(),cpuDirectMs=now()-cpuStart;
      const quantized=p.direct_quantized_cpu();
      const fdStart=now();p.finite_difference_cpu();const cpuFdMs=now()-fdStart;
      const g=decode(p,gpu.channels),c=decode(p,cpu);
      const quantization=comparison(quantized,cpu),arithmetic=comparison(gpu.channels,quantized),combined=comparison(gpu.channels,cpu);
      const derivativeError=max(g.gradient.relativeError),cpuDerivativeError=max(c.gradient.relativeError);
      const recovered=recover(p,gpu.channels),recoveredDerivativeError=max(recovered.result.gradient.relativeError);
      const label=`${r.dim}D ${r.n}, order ${r.order}, ${r.ic}, ${r.omegaM===1?'EdS':'LCDM'}, ${r.epoch==='fraction'?`f=${r.fraction}`:`a=${r.scaleFactor}`}`;
      rows.push({label:`${label}: GPU complex channels vs quantized f64`,ok:arithmetic.failedComponents===0,detail:arithmetic});
      rows.push({label:`${label}: CPU reference finite difference`,ok:cpuDerivativeError<1e-5,detail:{maximumRelativeError:cpuDerivativeError}});
      rows.push({label:`${label}: selective recovery preserves the derivative gate`,ok:recoveredDerivativeError<=1e-5,detail:{maximumRelativeError:recoveredDerivativeError,cpuLptModeIndices:recovered.failed}});
      const preserved=['q','x','v','pmX','pmV','rho','pmRho','pdf','guard','diagnostics'];
      rows.push({label:`${label}: shared physical observations`,ok:preserved.every(k=>JSON.stringify(g[k])===JSON.stringify(c[k])),detail:{fields:preserved}});
      // gpuEligible describes the uncorrected GPU channels. The worker recovers
      // only failing LPT modes and records the resulting mixed provenance.
      cases.push({request:r,label,quantization,arithmetic,combined,derivativeError,cpuDerivativeError,gpuEligible:derivativeError<=1e-5,cpuRecoveredLptModeIndices:recovered.failed,recoveredDerivativeError,crossed:g.guard.crossed,gpu:gpu.diagnostics,timings:{prepareMs,cpuDirectMs,cpuFdMs,cpuRecoveryMs:recovered.cpuRecoveryMs,gpu:gpu.timings}});
      timings.push({label,prepareMs,cpuDirectMs,cpuFdMs,gpu:gpu.timings});
      numericalErrors.push({label,quantization,arithmetic,combined,derivativeError});
    } catch(error) {rows.push({label:`Complete request ${i+1}`,ok:false,detail:String(error.stack||error)});}
    finally {p?.free();}
    await new Promise(r=>setTimeout(r,0));
  }
  if(benchmark) for(const n of [32,64]) {
    const r=request({dim:3,n,order:4,omegaM:.315,scaleFactor:.25});
    const samples=[];
    for(let repeat=0;repeat<3;repeat++) {
      onProgress({stage:`Benchmark ${n}³ repeat ${repeat+1}`,caseIndex:repeat+1,cases:3});
      let p;
      try {
        let t=now();p=new PreparedLabRun(JSON.stringify(r));const prepareMs=now()-t;
        t=now();const input=inputs(p);const packMs=now()-t;
        const gpu=await runDirect(device,input);
        t=now();const cpu=p.direct_cpu(),cpuDirectMs=now()-t;
        t=now();p.finite_difference_cpu();const cpuFdMs=now()-t;
        t=now();const g=decode(p,gpu.channels);const finishMs=now()-t;
        t=now();JSON.stringify(g);const serializeMs=now()-t;
        const derivativeError=max(g.gradient.relativeError),gpuEligible=derivativeError<=1e-5,recovered=recover(p,gpu.channels);
        samples.push({prepareMs,packMs,cpuDirectMs,cpuFdMs,finishMs,serializeMs,gpu:gpu.timings,derivativeError,gpuEligible,
          cpuRecoveredLptModeIndices:recovered.failed,cpuRecoveryMs:recovered.cpuRecoveryMs,recoveredDerivativeError:max(recovered.result.gradient.relativeError),
          cpuTotalMs:prepareMs+cpuDirectMs+cpuFdMs+finishMs+serializeMs,
          hybridTotalMs:prepareMs+packMs+gpu.timings.totalMs+cpuFdMs+finishMs+serializeMs+recovered.cpuRecoveryMs,
          allocation:gpu.diagnostics.estimatedBytes,comparison:comparison(gpu.channels,cpu)});
      } finally {p?.free();}
      await new Promise(r=>setTimeout(r,0));
    }
    timings.push({benchmark:true,request:r,repeats:3,samples,
      cpuDirectMs:summary(samples.map(s=>s.cpuDirectMs)),gpuDirectMs:summary(samples.map(s=>s.gpu.totalMs)),cpuFdMs:summary(samples.map(s=>s.cpuFdMs)),
      cpuTotalMs:summary(samples.map(s=>s.cpuTotalMs)),hybridTotalMs:summary(samples.map(s=>s.hybridTotalMs)),
      description:'Stage-summed estimates on shared preparation; hybrid includes selective f64 LPT recovery for failing modes. GPU device and pipelines already warm. Not a separate cold-start worker measurement.'});
  }
  const worker=await workerChecks(onProgress,benchmark);rows.push(...worker.rows);timings.push(...worker.timings);
  return {status:rows.every(r=>r.ok)?'pass':'fail',rows,cases,timings,numericalErrors,
    precision:'f64 full reference; f64 kernel on packed f32 input reference; f32 GPU; f64 CPU FD',
    fullRequestGate:'The worker requires maximum amplitude derivative relative error <=1e-5; failing LPT modes recover together with their tangents on CPU f64, with per-mode provenance.'};
}

async function workerChecks(onProgress,benchmark) {
  const rows=[],timings=[],worker=new Worker(new URL('../worker.js',import.meta.url),{type:'module'});
  const r=request({dim:2,n:16,order:4,omegaM:.315,scaleFactor:.25});
  let serial=0;
  const run=(policy,state=r)=>new Promise((resolve,reject)=>{
    const requestId=++serial,timeout=setTimeout(()=>reject(new Error('Worker test timed out')),120000);
    worker.onmessage=({data})=>{if(data.type==='result'&&data.requestId===requestId){clearTimeout(timeout);data.error?reject(new Error(data.error)):resolve(data.result);}};
    worker.onerror=e=>{clearTimeout(timeout);reject(new Error(e.message));};
    worker.postMessage({type:'run',requestId,state,policy});
  });
  try {
    const reference=await run({requested:'cpu'});
    for(const failure of ['missing-api','limits','shader','device-destroy','readback','nonfinite','no-result']) {
      onProgress({stage:`Worker fallback: ${failure}`});
      const result=await run({requested:'gpu',forceGpuFailure:failure});
      const e=result.execution;
      const same=['gradient','spectra','pdf','guard','diagnostics'].every(k=>JSON.stringify(result[k])===JSON.stringify(reference[k]));
      rows.push({label:`Worker ${failure} recovers on retained CPU context`,ok:e?.executedBackend==='cpu'&&Boolean(e.fallbackReason)&&same,detail:{execution:e,samePhysicalResults:same}});
    }
    const oneD=await run({requested:'gpu'},request({dim:1,n:16,order:4,omegaM:.315,scaleFactor:.25}));
    rows.push({label:'1D GPU request reports CPU route',ok:oneD.execution?.executedBackend==='cpu'&&Boolean(oneD.execution.fallbackReason),detail:oneD.execution});
    const auto=await run({requested:'auto'});
    rows.push({label:'Auto selection remains CPU during validation',ok:auto.execution?.executedBackend==='cpu'&&Boolean(auto.execution.fallbackReason),detail:auto.execution});
    const gpu=await run({requested:'gpu'});
    rows.push({label:'Worker actual GPU request meets gate or reports fallback',ok:gpu.gradient.relativeError.every(e=>Number.isFinite(e)&&e<=1e-5)&&['cpu','gpu','hybrid'].includes(gpu.execution?.executedBackend),detail:gpu.execution});
    if(benchmark) {
      const benchRequest=request({dim:3,n:32,order:4,omegaM:.315,scaleFactor:.25});
      const cpuSamples=[],gpuSamples=[];
      // Device/Wasm validation above is excluded. Each recorded request still
      // repeats all physical preparation and includes result cloning/readback.
      for(let repeat=0;repeat<3;repeat++) for(const requested of ['cpu','gpu']) {
        onProgress({stage:`End-to-end 32³ ${requested}, repeat ${repeat+1}`});
        const started=now(),result=await run({requested},benchRequest),milliseconds=now()-started;
        const sample={milliseconds,execution:result.execution,maximumDerivativeError:max(result.gradient.relativeError)};
        (requested==='cpu'?cpuSamples:gpuSamples).push(sample);
        rows.push({label:`End-to-end 32³ ${requested} repeat ${repeat+1} retains derivative gate`,ok:sample.maximumDerivativeError<=1e-5,detail:{milliseconds,backend:result.execution.executedBackend}});
      }
      timings.push({benchmark:true,endToEndWorker:true,request:benchRequest,repeats:3,cpu:summary(cpuSamples.map(s=>s.milliseconds)),hybrid:summary(gpuSamples.map(s=>s.milliseconds)),cpuSamples,gpuSamples,
        description:'Alternating full requests through one persistent worker after Wasm/device validation; includes repeated physical preparation, CPU FD, selective recovery, GPU transfers and result structured cloning. Rendering excluded.'});
    }
    await new Promise((resolve,reject)=>{const timeout=setTimeout(()=>reject(new Error('GPU worker shutdown acknowledgement timeout')),10000);worker.onmessage=({data})=>{if(data.type==='shutdown-ack'){clearTimeout(timeout);resolve();}};worker.postMessage({type:'shutdown'});});
    rows.push({label:'Persistent GPU worker acknowledges cleanup',ok:true});
  } catch(error) {rows.push({label:'Worker lifecycle and fallback checks',ok:false,detail:String(error.stack||error)});}
  finally {worker.terminate();}
  return {rows,timings};
}
