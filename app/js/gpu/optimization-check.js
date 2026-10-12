// Device-side checks for the independently callable GPU optimization stages.
// Full Wasm-backed comparisons and end-to-end benchmarks are added below this
// small startup suite so kernel compilation can be exercised in isolation.
import { runCompensatedDirectSelfTests } from './compensated-check.js';
import { runLptSelfTests } from './lpt-check.js';
import { runExactDepositChecks } from './deposit-check.js';
import { runExactDeposit } from './deposit.js';
import { runLpt } from './lpt.js';
import { runCompensatedDirect } from './compensated-direct.js';
import { runFiniteDifference } from './finite-difference.js';
import { runFiniteDifferenceSelfTest } from './fd-check.js';

const now=()=>performance.now();

/** Run independent observer, raw-LPT, and exact-deposit fixtures on this device. */
export async function runOptimizationSelfChecks({device,signal,onProgress=()=>{},maxGpuBytes}={}) {
  if(!device)throw new TypeError('runOptimizationSelfChecks requires a WebGPU device');
  const checks=[
    ['compensated observer',()=>runCompensatedDirectSelfTests(device,{signal,onProgress,maxGpuBytes})],
    ['finite-difference endpoint path',()=>runFiniteDifferenceSelfTest({device,signal,onProgress})],
    ['raw nLPT',()=>runLptSelfTests(device,{signal,onProgress,maxGpuBytes})],
    ['exact deposit',()=>runExactDepositChecks(device,{signal,onProgress,maxGpuBytes})],
  ];
  const suites=[],rows=[];
  for(const [name,run] of checks){
    if(signal?.aborted)throw new DOMException('Optimization self-test canceled','AbortError');
    const started=now();
    try{
      const result=await run();
      const suiteRows=(result.rows??[]).map(row=>({suite:name,...row}));
      const suiteStatus=result.status??(suiteRows.every(row=>row.pass!==false&&row.ok!==false)?'pass':'fail');
      if(suiteStatus==='fail'&&!suiteRows.some(row=>row.pass===false||row.ok===false))suiteRows.push({suite:name,label:`${name} self-test status`,pass:false,error:'The suite returned fail without an explicit failed row.'});
      suites.push({name,status:suiteStatus,durationMs:now()-started,result:compactReport(result)});
      rows.push(...suiteRows.map(row=>compactReport(row)));
    }catch(error){
      const failure={suite:name,label:`${name} self-test completed`,pass:false,error:String(error?.stack??error)};
      rows.push(failure);suites.push({name,status:'fail',durationMs:now()-started,error:failure.error});
    }
    await onProgress({stage:`Optimization self-test: ${name}`});
    await new Promise(resolve=>setTimeout(resolve,0));
  }
  const failures=rows.filter(row=>row.pass===false||row.ok===false),failedSuite=suites.some(suite=>suite.status==='fail'),skippedSuite=suites.some(suite=>suite.status==='skip');
  return {status:failures.length||failedSuite?'fail':skippedSuite?'skip':'pass',rows,failures,suites,device:{deviceBacked:true},timings:suites.map(({name,durationMs})=>({name,durationMs}))};
}

const BASE_REQUEST={epoch:'scaleFactor',fraction:.35,steps:32,mesh:16,forceMesh:16,omegaB:.049,h:.674,ns:.965,sigma8:.811,boxSize:300,initialSmoothing:14,initialCutoff:3,seed:7,ic:'waves',smoothing:.03,order:4,omegaM:.315,scaleFactor:.25};
const request=(settings={})=>({...BASE_REQUEST,...settings});
const copy=(C,v)=>v instanceof C?new C(v):C.from(v??[]);
const parse=(v,label)=>typeof v==='string'?JSON.parse(v):v&&typeof v==='object'?v:(()=>{throw new Error(`Wasm ${label} was not valid JSON`);})();
const f32=(x)=>Math.fround(x);
function compare(a,b,{absolute=1e-11,relative=1e-8}={}){
  let maxAbs=0,maxFraction=0,failed=0,nonfinite=0;
  if(a.length!==b.length)return {pass:false,lengthA:a.length,lengthB:b.length,failedComponents:Math.max(a.length,b.length),nonfinite:0,maximumAbsolute:Infinity,maximumBudgetFraction:Infinity};
  for(let i=0;i<a.length;i++){const x=a[i],y=b[i];if(!Number.isFinite(x)||!Number.isFinite(y)){nonfinite++;failed++;continue;}const e=Math.abs(x-y),budget=absolute+relative*Math.abs(y);maxAbs=Math.max(maxAbs,e);maxFraction=Math.max(maxFraction,e/budget);if(e>budget)failed++;}
  return {pass:failed===0,failedComponents:failed,nonfinite,maximumAbsolute:maxAbs,maximumBudgetFraction:maxFraction,absolute,relative};
}
function compareNumericTree(actual,reference,{absolute,relative}={}){
  let maxAbs=0,maxFraction=0,failed=0,count=0,nonfinite=0;
  const visit=(a,b,path)=>{
    if(typeof a==='number'||typeof b==='number'){
      if(typeof a!=='number'||typeof b!=='number'||!Number.isFinite(a)||!Number.isFinite(b)){nonfinite++;failed++;return;}
      count++;const e=Math.abs(a-b),budget=absolute+relative*Math.abs(b);maxAbs=Math.max(maxAbs,e);maxFraction=Math.max(maxFraction,e/Math.max(budget,Number.MIN_VALUE));if(e>budget)failed++;return;
    }
    if(a==null||b==null){if(a!==b)failed++;return;}
    const aa=Array.isArray(a)||ArrayBuffer.isView(a),bb=Array.isArray(b)||ArrayBuffer.isView(b);
    if(aa||bb){if(!aa||!bb||a.length!==b.length){failed++;return;}for(let i=0;i<a.length;i++)visit(a[i],b[i],`${path}[${i}]`);return;}
    if(typeof a==='object'||typeof b==='object'){if(typeof a!=='object'||typeof b!=='object') {failed++;return;}const ka=Object.keys(a).filter(k=>typeof a[k]==='number'||Array.isArray(a[k])||ArrayBuffer.isView(a[k])||typeof a[k]==='object').sort(),kb=Object.keys(b).filter(k=>typeof b[k]==='number'||Array.isArray(b[k])||ArrayBuffer.isView(b[k])||typeof b[k]==='object').sort();if(ka.join('|')!==kb.join('|')){failed++;return;}for(const k of ka)visit(a[k],b[k],`${path}.${k}`);}
  };
  visit(actual,reference,'$');return {pass:failed===0,failedComponents:failed,nonfinite,numericComponents:count,maximumAbsolute:maxAbs,maximumBudgetFraction:maxFraction,absolute,relative};
}
function assertFiniteTyped(values,label){if(!ArrayBuffer.isView(values)||!values.length)throw new TypeError(`${label} is empty or not a typed array`);for(let i=0;i<values.length;i++)if(!Number.isFinite(values[i]))throw new Error(`${label}[${i}] is nonfinite`);}
function assertedFinite(values){return (Array.isArray(values)||ArrayBuffer.isView(values))&&values.length>0&&Array.from(values).every(Number.isFinite);}
function maximumFinite(values){if(!assertedFinite(values))return {pass:false,maximum:Infinity,count:values?.length??0};let maximum=0;for(const value of values)maximum=Math.max(maximum,value);return {pass:true,maximum,count:values.length};}
function meanValue(values){if(!assertedFinite(values))return NaN;let sum=0;for(const value of values)sum+=value;return sum/values.length;}
function benchmarkQuality(result,{optimized=false}={}){
  const derivative=maximumFinite(result.gradient?.relativeError),rhoMean=meanValue(result.rho),pmMean=meanValue(result.pmRho),rhoMassError=Math.abs(rhoMean-1),pmMassError=Math.abs(pmMean-1),massPass=Number.isFinite(rhoMassError)&&Number.isFinite(pmMassError)&&rhoMassError<=1e-6&&pmMassError<=1e-6;
  const stages=result.execution?.stageBackends,analyticBackend=stages?.analyticRecovery?.backend,backendPass=!optimized||(stages?.trajectories?.lptBackend==='gpu'&&stages?.exactDeposit?.backend==='gpu'&&stages?.exactDeposit?.provenance==='gpu-f32-geometry-overlap'&&['gpu','unused'].includes(analyticBackend));
  const derivativePass=!optimized||derivative.pass&&derivative.maximum<=1e-5;
  return {pass:massPass&&derivativePass&&backendPass,mass:{pass:massPass,rhoMean,pmMean,rhoMassError,pmMassError,tolerance:1e-6},derivativeGate:{...derivative,pass:derivativePass,tolerance:1e-5},backends:{pass:backendPass,lpt:stages?.trajectories?.lptBackend,deposit:stages?.exactDeposit?.backend,depositProvenance:stages?.exactDeposit?.provenance,analyticRecovery:analyticBackend},executedBackend:result.execution?.executedBackend};
}
function compareBenchmarkPair(reference,optimized){
  const density=compare(optimized.rho,reference.rho,{absolute:2e-5,relative:1e-4}),pmDensity=compare(optimized.pmRho,reference.pmRho,{absolute:5e-4,relative:2e-3});
  const spectrumKeys=['sheet','cic','pm','direct','directPM'],spectra=compareNumericTree(Object.fromEntries(spectrumKeys.map(k=>[k,optimized.spectra?.[k]])),Object.fromEntries(spectrumKeys.map(k=>[k,reference.spectra?.[k]])),{absolute:5e-4,relative:2e-3});
  const pdfKeys=['volume','mass','pmVolume','pmMass','mean','variance','pmVariance','summary','pmSummary'],pdf=compareNumericTree(Object.fromEntries(pdfKeys.map(k=>[k,optimized.pdf?.[k]])),Object.fromEntries(pdfKeys.map(k=>[k,reference.pdf?.[k]])),{absolute:2e-3,relative:2e-2});
  return {pass:density.pass&&pmDensity.pass&&spectra.pass&&pdf.pass,density,pmDensity,spectra,pdf,budgets:{rho:{absolute:2e-5,relative:1e-4},pmRho:{absolute:5e-4,relative:2e-3},spectra:{absolute:5e-4,relative:2e-3},pdf:{absolute:2e-3,relative:2e-2}}};
}
function rowsFromComparison(label,detail,rows){rows.push({label,pass:detail.pass,...detail});}
function sameShapeAndFinite(a,b){
  if(typeof a==='number'||typeof b==='number')return typeof a==='number'&&typeof b==='number'&&Number.isFinite(a)&&Number.isFinite(b);
  if(a===null||b===null)return a===b;
  if(Array.isArray(a)||ArrayBuffer.isView(a))return (Array.isArray(b)||ArrayBuffer.isView(b))&&a.length===b.length&&Array.from(a).every((v,i)=>sameShapeAndFinite(v,b[i]));
  if(typeof a==='object'||typeof b==='object')return typeof a==='object'&&typeof b==='object'&&Object.keys(a).sort().join('|')===Object.keys(b).sort().join('|')&&Object.keys(a).every(k=>sameShapeAndFinite(a[k],b[k]));
  return typeof a===typeof b;
}
// Keep validation measurements and hashes while avoiding the same full sample
// arrays being serialized in every worker stage record.
function compactReport(value){
  if(ArrayBuffer.isView(value))return {type:value.constructor.name,length:value.length};
  if(Array.isArray(value)){
    if(value.length>32&&value.every(item=>typeof item==='number'||typeof item==='string'))return {length:value.length,sample:value.slice(0,4)};
    return value.map(item=>compactReport(item));
  }
  if(value&&typeof value==='object')return Object.fromEntries(Object.entries(value).map(([key,item])=>[key,compactReport(item)]));
  return value;
}

/** Wasm-backed independent GPU terms, observer and deposit comparisons. */
export async function runOptimizationChecks({device,benchmark=false,onProgress=()=>{},signal,maxGpuBytes}={}){
  if(!device)throw new TypeError('runOptimizationChecks requires a WebGPU device');
  const wasm=await import('../../pkg/hcc_core.js');await wasm.default();
  const rows=[],cases=[],timings=[];
  await onProgress({stage:'Optimization Wasm initialized'});
  const GpuLptSeed=wasm.GpuLptSeed,Prepared=wasm.PreparedLabRun;
  if(typeof GpuLptSeed!=='function')rows.push({label:'Wasm GPU LPT seed API is present',pass:false,error:'GpuLptSeed export is unavailable; rebuild the Wasm package.'});
  else for(const settings of [{dim:2,n:16,ic:'waves'},{dim:3,n:16,ic:'waves'},{dim:2,n:16,ic:'waves',omegaM:1},{dim:3,n:16,ic:'waves',omegaM:1},{dim:3,n:32,ic:'gaussian',seed:2650,initialCutoff:3}] ){
    if(signal?.aborted)throw new DOMException('Optimization checks canceled','AbortError');
    const r=request({...settings,mesh:settings.n,forceMesh:settings.n}),started=now();let seed;
    try{
      seed=typeof GpuLptSeed.new==='function'?GpuLptSeed.new(JSON.stringify(r)):new GpuLptSeed(JSON.stringify(r));const meta=parse(seed.metadata(),'GPU LPT seed metadata'),deltaHat=copy(Float32Array,seed.delta_hat()),specs=meta.specs;
      const got=await runLpt(device,{dim:meta.dim,n:meta.n,deltaHat,specs},{signal,maxGpuBytes:maxGpuBytes??512*1024*1024,onProgress});
      const reference=copy(Float32Array,seed.reference_fields());const detail=compare(got.fields,reference,{absolute:2e-6,relative:2e-4});rowsFromComparison(`${r.dim}D n=${r.n} raw GPU nLPT fields match Rust terms`,detail,rows);
      cases.push({kind:'lpt',request:r,fieldStride:got.stride,termCount:got.terms.length,comparison:detail,gpu:got.diagnostics,timings:got.timings});timings.push({kind:'lpt',request:r,...got.timings});
    }catch(error){rows.push({label:`${r.dim}D n=${r.n} raw GPU nLPT fields match Rust terms`,pass:false,error:String(error?.stack??error)});}
    finally{seed?.free?.();}
    await new Promise(resolve=>setTimeout(resolve,0));
  }
  if(typeof Prepared!=='function')rows.push({label:'Wasm prepared observer API is present',pass:false,error:'PreparedLabRun export is unavailable; rebuild the Wasm package.'});
  else for(const settings of [{dim:2,n:16,ic:'waves'},{dim:3,n:16,ic:'waves'},{dim:3,n:32,ic:'gaussian',seed:2650,initialCutoff:3}] ){
    if(signal?.aborted)throw new DOMException('Optimization checks canceled','AbortError');
    const r=request({...settings,mesh:settings.n,forceMesh:settings.n}),started=now();let prepared;
    try{
      prepared=new Prepared(JSON.stringify(r));const meta=parse(prepared.metadata(),'prepared observer metadata'),modes=copy(Int32Array,prepared.mode_records()),input={dim:meta.dim,n:meta.n,modes,psiHi:copy(Float32Array,prepared.lpt_displacement_hi()),psiLo:copy(Float32Array,prepared.lpt_displacement_lo()),directionHi:copy(Float32Array,prepared.lpt_direction_hi()),directionLo:copy(Float32Array,prepared.lpt_direction_lo())};
      const got=await runCompensatedDirect(device,input,{signal,maxGpuBytes:maxGpuBytes??512*1024*1024,onProgress}),reference=copy(Float64Array,prepared.direct_cpu()),expected=new Float64Array(got.channels.length);
      for(let m=0;m<meta.modeCount;m++)for(let c=0;c<4;c++)expected[4*m+c]=reference[8*m+c];
      const detail=compare(got.channels,expected,{absolute:1e-11,relative:1e-8});rowsFromComparison(`${r.dim}D n=${r.n} compensated observer LPT primal and tangent match Rust f64`,detail,rows);
      cases.push({kind:'compensated-direct',request:r,comparison:detail,gpu:got.diagnostics,timings:got.timings});timings.push({kind:'compensated-direct',request:r,...got.timings});
      const fdInput={dim:meta.dim,n:meta.n,modes,h:meta.h,plusHi:copy(Float32Array,prepared.finite_difference_plus_hi()),plusLo:copy(Float32Array,prepared.finite_difference_plus_lo()),minusHi:copy(Float32Array,prepared.finite_difference_minus_hi()),minusLo:copy(Float32Array,prepared.finite_difference_minus_lo())};
      const fd=await runFiniteDifference(device,fdInput,{signal,maxGpuBytes:maxGpuBytes??512*1024*1024,onProgress}),fdReference=copy(Float64Array,prepared.finite_difference_cpu()),fdDetail=compare(fd.values,fdReference,{absolute:1e-9,relative:1e-5});rowsFromComparison(`${r.dim}D n=${r.n} power finite difference matches retained f64 oracle`,fdDetail,rows);
      cases.push({kind:'finite-difference',request:r,comparison:fdDetail,h:meta.h,diagnostics:fd.diagnostics,timings:fd.timings});timings.push({kind:'finite-difference',request:r,...fd.timings});
    }catch(error){rows.push({label:`${r.dim}D n=${r.n} compensated observer and power-FD comparisons`,pass:false,error:String(error?.stack??error)});}
    finally{prepared?.free?.();}
    await new Promise(resolve=>setTimeout(resolve,0));
  }
  const depositCpu=wasm.exact_sheet_density??wasm.exactSheetDensity;
  if(typeof depositCpu!=='function')rows.push({label:'Wasm exact sheet density oracle is present',pass:false,error:'exact_sheet_density export is unavailable; rebuild the Wasm package.'});
  else for(const dim of [2,3]){
    const n=4,meshN=8,N=n**dim,positions=new Float32Array(N*4);
    for(let i=0;i<N;i++){let t=i,co=new Array(dim);for(let a=dim-1;a>=0;a--){co[a]=(t%n)/n;t=Math.floor(t/n);}for(let a=0;a<dim;a++)positions[4*i+a]=f32(co[a]+0.11*Math.sin(2*Math.PI*co[(a+1)%dim]));}
    const fixture={dim,n,meshN,boxSize:1,positions};
    try{const started=now(),got=await runExactDeposit(device,fixture,{maxGpuBytes:maxGpuBytes??512*1024*1024,signal}),reference=copy(Float32Array,depositCpu(dim,n,meshN,positions)),detail=compare(got.density,reference,{absolute:2e-5,relative:1e-4});rowsFromComparison(`${dim}D exact GPU deposition matches Rust f32 density field`,detail,rows);let densitySum=0;for(const x of got.density)densitySum+=x;const densityMean=densitySum/got.density.length,massError=Math.abs(densityMean-1),massPass=massError<=1e-6;rows.push({label:`${dim}D deposited sheet conserves unit mean density`,pass:massPass,densityMean,totalDensity:densitySum,massError,tolerance:1e-6});cases.push({kind:'deposit',dim,n,meshN,comparison:detail,massError,densityMean,diagnostics:got.diagnostics,timings:got.timings});timings.push({kind:'deposit',dim,n,meshN,...got.timings,wallMs:now()-started});}
    catch(error){rows.push({label:`${dim}D exact GPU deposition matches Rust f32 density field`,pass:false,error:String(error?.stack??error)});}
  }
  let worker=null;
  try{const out=await optimizationWorkerChecks(onProgress,benchmark,signal);worker=out;rows.push(...out.rows);timings.push(...out.timings);}catch(error){rows.push({label:'Optimization worker lifecycle and cache checks',pass:false,error:String(error?.stack??error)});}
  return {status:rows.every(r=>r.pass!==false&&r.ok!==false)?'pass':'fail',rows,failures:rows.filter(r=>r.pass===false||r.ok===false),cases,timings,worker,precision:{lpt:{absolute:2e-6,relative:2e-4},compensated:{absolute:1e-11,relative:1e-8},finiteDifference:{absolute:1e-9,relative:1e-5},deposit:{absolute:2e-5,relative:1e-4,massAbsolute:1e-6}},benchmark};
}

async function optimizationWorkerChecks(onProgress,benchmark,signal){
  const rows=[],timings=[],worker=new Worker(new URL('../worker.js',import.meta.url),{type:'module'}),base=request({dim:2,n:16,mesh:16,forceMesh:16,order:2});let serial=0;
  const run=(state,policy)=>new Promise((resolve,reject)=>{if(signal?.aborted)return reject(new DOMException('Optimization worker checks canceled','AbortError'));const requestId=++serial,timer=setTimeout(()=>reject(new Error('Optimization worker check timed out')),180000);worker.onmessage=({data})=>{if(data.type==='result'&&data.requestId===requestId){clearTimeout(timer);data.error?reject(new Error(data.error)):resolve(data.result);}};worker.onerror=e=>{clearTimeout(timer);reject(new Error(e.message));};worker.postMessage({type:'run',requestId,state,policy});});
  try{
    const cpu=await run(base,{requested:'cpu'}),gpu=await run(base,{requested:'gpu'}),keys=['rho','pmRho','spectra','pdf','guard'];const wellFormed=keys.every(k=>sameShapeAndFinite(cpu[k],gpu[k]));rows.push({label:'Full optimized worker request preserves finite CPU observation shapes',pass:wellFormed,comparisonKeys:keys,exactCpuEquality:keys.every(k=>JSON.stringify(cpu[k])===JSON.stringify(gpu[k])),requestedBackend:gpu.execution?.requestedBackend,executedBackend:gpu.execution?.executedBackend,stages:compactReport(gpu.execution?.stageBackends)});
    const density=compare(gpu.rho,cpu.rho,{absolute:2e-5,relative:1e-4}),pmDensity=compare(gpu.pmRho,cpu.pmRho,{absolute:5e-4,relative:2e-3});rows.push({label:'Worker exact GPU density remains within raw P0 f32 budget',pass:density.pass,...density});rows.push({label:'Worker GPU PM density remains within f32 trajectory budget',pass:pmDensity.pass,...pmDensity});
    const spectra=compareNumericTree({sheet:gpu.spectra?.sheet,cic:gpu.spectra?.cic,pm:gpu.spectra?.pm,direct:gpu.spectra?.direct,directPM:gpu.spectra?.directPM},{sheet:cpu.spectra?.sheet,cic:cpu.spectra?.cic,pm:cpu.spectra?.pm,direct:cpu.spectra?.direct,directPM:cpu.spectra?.directPM},{absolute:5e-4,relative:2e-3});rows.push({label:'Worker spectra stay within f32 density and trajectory budgets',...spectra});
    const pdfKeys=['volume','mass','pmVolume','pmMass','mean','variance','pmVariance','summary','pmSummary'],pdf=compareNumericTree(Object.fromEntries(pdfKeys.map(k=>[k,gpu.pdf?.[k]])),Object.fromEntries(pdfKeys.map(k=>[k,cpu.pdf?.[k]])),{absolute:2e-3,relative:2e-2});rows.push({label:'Worker PDF histograms and moments stay within f32 density budgets',...pdf});
    const exactDeposit=gpu.execution?.stageBackends?.exactDeposit,depositRecord=gpu.execution?.exactDeposit,depositSource=exactDeposit?.provenance,depositPass=depositSource==='gpu-f32-geometry-overlap'&&exactDeposit?.backend==='gpu'&&depositRecord?.backend==='gpu';rows.push({label:'Well-conditioned worker request executes exact P0 deposition on GPU',pass:depositPass,deposit:compactReport({stage:exactDeposit,attempt:depositRecord})});
    const smoothingA=request({dim:2,n:16,mesh:16,forceMesh:16,order:2,smoothing:.03}),smoothingB={...smoothingA,smoothing:.05};const cold=await run(smoothingA,{requested:'gpu'}),hot=await run(smoothingB,{requested:'gpu'}),fresh=await run(smoothingB,{requested:'gpu',disableCache:true});const cacheOf=result=>result.cache??result.execution?.cache??null,coldCache=cacheOf(cold),hotCache=cacheOf(hot),freshCache=cacheOf(fresh),cacheDeposit=hot.execution?.stageBackends?.exactDeposit?.provenance,depositStable=cacheDeposit===fresh.execution?.stageBackends?.exactDeposit?.provenance,cacheFields=['rho','pmRho','spectra','pdf'],cacheDetail=compareNumericTree(Object.fromEntries(cacheFields.map(k=>[k,hot[k]])),Object.fromEntries(cacheFields.map(k=>[k,fresh[k]])),{absolute:5e-4,relative:2e-3}),gpuDeposit=cacheDeposit==='gpu-f32-geometry-overlap',cacheStages=['initial','pending','pm','observations'],cacheHits=Object.fromEntries(cacheStages.map(k=>[k,hotCache?.[k]?.status==='hit'])),allCacheHits=cacheStages.every(k=>cacheHits[k]),matches=cacheDetail.pass&&depositStable;rows.push({label:'Observation-only smoothing update reuses all GPU physical and observation stages and matches fresh numerical fields',pass:gpuDeposit&&allCacheHits&&matches,coldCache:compactReport(coldCache),hotCache:compactReport(hotCache),freshCache:compactReport(freshCache),samePhysicalResults:matches,numericalComparison:cacheDetail,depositProvenanceStable:depositStable,cacheHits,cacheExpectation:'all four compatible GPU cache stages hit'});
    const meshWarmState=request({dim:2,n:16,mesh:16,forceMesh:16,order:2,smoothing:.03}),meshChangedState={...meshWarmState,mesh:24};await run(meshWarmState,{requested:'gpu'});const meshChanged=await run(meshChangedState,{requested:'gpu'}),meshFresh=await run(meshChangedState,{requested:'gpu',disableCache:true}),meshCache=cacheOf(meshChanged),meshCacheHits=Object.fromEntries(['initial','pending','pm'].map(k=>[k,meshCache?.[k]?.status==='hit'])),meshObservationMiss=meshCache?.observations?.status==='miss',meshReuse=Object.values(meshCacheHits).every(Boolean)&&meshObservationMiss,meshComparison=compareBenchmarkPair(meshFresh,meshChanged),meshDeposit=meshChanged.execution?.stageBackends?.exactDeposit,meshDepositGpu=meshDeposit?.backend==='gpu'&&meshDeposit?.provenance==='gpu-f32-geometry-overlap';rows.push({label:'Observation mesh change retains compatible physical stages and recomputes fresh GPU observations',pass:meshReuse&&meshComparison.pass&&meshDepositGpu,cache:compactReport(meshCache),cacheHits:meshCacheHits,observationsStatus:meshCache?.observations?.status,comparison:meshComparison,deposit:compactReport(meshDeposit),freshExecution:compactReport(meshFresh.execution)});
    const expectedStage={lpt:['trajectories','lptBackend'],deposit:['exactDeposit','backend'],compensated:['analyticRecovery','backend']};
    for(const stage of ['lpt','deposit','compensated']){const policy={requested:'gpu',[`disable${stage[0].toUpperCase()}${stage.slice(1)}`]:true},result=await run(base,policy),record=result.execution?.stageBackends?.[expectedStage[stage][0]],stageBackend=record?.[expectedStage[stage][1]];rows.push({label:`Explicit ${stage} CPU reference selection is recorded per stage`,pass:stageBackend==='cpu',execution:compactReport(result.execution),stageBackend});}
    for(const stage of ['lpt','deposit','compensated'])for(const failure of ['validation','limits','nonfinite','readback']){const policy={requested:'gpu',[`force${stage[0].toUpperCase()}${stage.slice(1)}Failure`]:failure};const result=await run(base,policy),record=result.execution?.stageBackends?.[expectedStage[stage][0]],stageBackend=record?.[expectedStage[stage][1]],correctBackend=stageBackend==='cpu';rows.push({label:`Forced ${stage} ${failure} recovers through its CPU path`,pass:correctBackend,execution:compactReport(result.execution),stageBackend});}
    const nonradixState=request({dim:2,n:24,mesh:16,forceMesh:16,order:2}),nonradixCpu=await run(nonradixState,{requested:'cpu'}),nonradixGpu=await run(nonradixState,{requested:'gpu'}),nonradixStages=nonradixGpu.execution?.stageBackends,nonradixLpt=nonradixStages?.trajectories?.lptBackend,nonradixFields=compareNumericTree({rho:nonradixGpu.rho,pmRho:nonradixGpu.pmRho,spectra:nonradixGpu.spectra,pdf:nonradixGpu.pdf},{rho:nonradixCpu.rho,pmRho:nonradixCpu.pmRho,spectra:nonradixCpu.spectra,pdf:nonradixCpu.pdf},{absolute:5e-4,relative:2e-3}),nonradixPass=nonradixLpt==='cpu'&&nonradixFields.pass&&sameShapeAndFinite(nonradixCpu.guard,nonradixGpu.guard);rows.push({label:'Full non-radix worker request preserves CPU nLPT fallback and finite observations',pass:nonradixPass,lptBackend:nonradixLpt,lptFallbackReason:nonradixGpu.execution?.lptConstruction?.fallbackReason,pmBackend:nonradixStages?.trajectories?.pmBackend,deposit:compactReport(nonradixStages?.exactDeposit),comparison:nonradixFields});
    const crossedState=request({dim:2,n:16,mesh:16,forceMesh:16,order:4,epoch:'fraction',fraction:1.1,omegaM:1,ic:'waves'}),crossed=await run(crossedState,{requested:'gpu'}),crossedRho=assertedFinite(crossed.rho)&&assertedFinite(crossed.pmRho),crossedPdf=assertedFinite(crossed.pdf?.volume)&&assertedFinite(crossed.pdf?.pmVolume),crossingReported=Boolean(crossed.guard?.crossed),caution=typeof crossed.guard?.reason==='string'&&/crossing/i.test(crossed.guard.reason);rows.push({label:'Crossed worker result remains visible with finite positive-stream observations and a caution',pass:crossingReported&&crossedRho&&crossedPdf&&caution,crossed:crossed.guard?.crossed,minimumJacobian:crossed.guard?.minimumJacobian,reason:crossed.guard?.reason,finiteDensities:crossedRho,finitePdf:crossedPdf,execution:compactReport(crossed.execution)});
    if(benchmark){for(const n of [32,64]){
      if(signal?.aborted)throw new DOMException('Optimization benchmark canceled','AbortError');
      const state=request({dim:3,n,mesh:n,forceMesh:n,order:4,omegaM:.315,scaleFactor:.25,ic:'waves'}),pairs=[];
      for(let rep=0;rep<3;rep++){
        const results={};pairs[rep]={};
        for(const [name,policy] of [['baselineGpu',{requested:'gpu',disableLpt:true,disableDeposit:true,disableCompensated:true,disableCache:true}],['optimizedGpu',{requested:'gpu',disableCache:true}]]){
          await onProgress({stage:`Optimization benchmark ${n}³ ${name}`,caseIndex:rep+1,cases:3});
          const started=now(),result=await run(state,policy),wallMs=now()-started;results[name]=result;
          pairs[rep][name]={wallMs,quality:benchmarkQuality(result,{optimized:name==='optimizedGpu'}),execution:compactReport(result.execution),guard:result.guard?.minimumJacobian};
        }
        const baseline=results.baselineGpu,optimized=results.optimizedGpu,comparison=compareBenchmarkPair(baseline,optimized),qualityPass=pairs[rep].baselineGpu.quality.pass&&pairs[rep].optimizedGpu.quality.pass,pairPass=qualityPass&&comparison.pass;
        pairs[rep].comparison=comparison;pairs[rep].pass=pairPass;
        rows.push({label:`${n}³ paired GPU benchmark ${rep+1} preserves mass, derivative gate, backend, and field budgets`,pass:pairPass,baselineQuality:pairs[rep].baselineGpu.quality,optimizedQuality:pairs[rep].optimizedGpu.quality,comparison});
        await new Promise(resolve=>setTimeout(resolve,0));
      }
      timings.push({benchmark:true,kind:'physical-cold-paired-gpu',request:state,repeats:3,paired:pairs,description:'Alternating baseline GPU (nLPT GPU, exact GPU deposition, and analytic GPU recovery disabled; PM and the centered finite-difference shader remain active) with all optimized stages enabled. Each repeat records mass, per-mode derivative-gate maximum, selected stage backends, and baseline-to-optimized density, power-spectrum, and PDF comparisons.'});
      const warmState={...state,smoothing:.03},hotState={...state,smoothing:.05};await run(warmState,{requested:'gpu'});const hotSamples=[];
      for(let rep=0;rep<3;rep++){await onProgress({stage:`Optimization benchmark ${n}³ observation-cache hot`,caseIndex:rep+1,cases:3});const started=now(),result=await run(hotState,{requested:'gpu'});hotSamples.push({wallMs:now()-started,cache:compactReport(result.cache??result.execution?.cache??null),execution:compactReport(result.execution)});}
      timings.push({benchmark:true,kind:'observation-cache-hot',request:hotState,repeats:hotSamples.length,samples:hotSamples,description:'Full request after a compatible physical-state warmup with only smoothing changed; reported separately from cold physical preparation timings.'});
    }
    }
    return {rows,timings};
  }finally{await new Promise(resolve=>{const timer=setTimeout(resolve,3000);worker.onmessage=({data})=>{if(data.type==='shutdown-ack'){clearTimeout(timer);resolve();}};worker.postMessage({type:'shutdown'});});worker.terminate();}
}
