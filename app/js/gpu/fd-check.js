// GPU centered finite-difference checks against the retained Rust f64 oracle.
// The independent startup fixture integrates a piecewise-linear sinusoidal
// sheet analytically, including repeated phase knots along its transverse axis.
import init, { PreparedLabRun } from '../../pkg/hcc_core.js';
import { runFiniteDifference, runPrecisionProbe } from './finite-difference.js';

const FD_GATE=1e-5;
const FD_AIM=1e-6;
const WEAK_FLOOR=1e-9;
const FD_REPEAT_ENDPOINT_BUDGET=1e-12;
const request=(overrides={})=>({epoch:'scaleFactor',fraction:.35,steps:32,mesh:32,forceMesh:32,omegaB:.049,h:.674,ns:.965,sigma8:.811,boxSize:300,initialSmoothing:14,initialCutoff:3,seed:7,ic:'waves',smoothing:.03,dim:2,n:32,order:4,omegaM:.315,scaleFactor:.25,...overrides});
const now=()=>performance.now();
const aborted=(signal)=>signal?.aborted;
function assert(condition,message){if(!condition)throw new Error(message);}
function packModes(records){
  if(records instanceof Int32Array)return records;
  const out=new Int32Array(records.length*4);
  for(let i=0;i<records.length;i++){
    const mode=records[i];out.set([mode[0]??mode.x??0,mode[1]??mode.y??0,mode[2]??mode.z??0,0],4*i);
  }
  return out;
}
function inputFromPrepared(p){
  const meta=JSON.parse(p.metadata());
  return {dim:meta.dim,n:meta.n,modes:packModes(p.mode_records()),
    plusHi:p.finite_difference_plus_hi(),plusLo:p.finite_difference_plus_lo(),
    minusHi:p.finite_difference_minus_hi(),minusLo:p.finite_difference_minus_lo(),h:meta.h};
}
function compare(actual,reference,{weakFloor=WEAK_FLOOR}={}){
  assert(actual.length===reference.length,`finite-difference output length ${actual.length} != reference length ${reference.length}`);
  let maxAbs=0,maxRel=0,maxGateFraction=0,failed=0,weak=0,withinAim=true;
  const perMode=[];
  for(let i=0;i<actual.length;i++){
    const a=actual[i],r=reference[i],abs=Math.abs(a-r),rel=abs/Math.max(Math.abs(r),weakFloor),gate=rel/FD_GATE;
    if(Math.abs(r)<weakFloor)weak++;
    if(!Number.isFinite(a)||!Number.isFinite(r)||rel>FD_GATE)failed++;
    maxAbs=Math.max(maxAbs,abs);maxRel=Math.max(maxRel,rel);maxGateFraction=Math.max(maxGateFraction,gate);withinAim&&=Number.isFinite(rel)&&rel<=FD_AIM;
    perMode.push({index:i,reference:r,actual:a,absoluteError:abs,relativeError:rel,weak:Math.abs(r)<weakFloor,pass:Number.isFinite(rel)&&rel<=FD_GATE});
  }
  return {pass:failed===0,maximumAbsolute:maxAbs,maximumWeakScaledRelative:maxRel,maximumGateFraction:maxGateFraction,failedModes:failed,weakModes:weak,withinAim,perMode};
}
function readFinalizer(p,channels,fd,execution){return JSON.parse(p.finish_with_finite_difference(channels,fd,JSON.stringify(execution)));}
function finalizerValues(result){return result?.gradient?.finiteDifference;}

/**
 * Integrate exp(-i 2πm x(q)) exactly for a piecewise-linear periodic sheet
 * x(q)=q+A sin(2πq) sampled at n knots. The deformation is constant on the
 * transverse direction, which intentionally gives the simplex divided
 * difference evaluator repeated knots without changing the one-dimensional
 * integral.
 */
function exactPiecewiseMode(dim,n,mode,hi,lo){
  const stride=n**(dim-1);
  let re=0,im=0;
  for(let i=0;i<n;i++){
    const j=(i+1)%n,q0=i/n,q1=(i+1)/n,off0=4*i*stride,off1=4*j*stride;
    const x0=q0+hi[off0]+lo[off0],x1=q1+hi[off1]+lo[off1],delta=x1-x0,k=2*Math.PI*mode;
    const half=.5*k*delta,sinc=Math.abs(half)<1e-12?1:Math.sin(half)/half,phase=k*(x0+.5*delta);
    re+=Math.cos(phase)*sinc/n;im-=Math.sin(phase)*sinc/n;
  }
  return [re,im];
}
function analyticFixture({dim=2,n=16,h=1e-4,amplitude=.08,translation=0,modes=[0,1,-1,6]}={}){
  const size=n**dim,stride=n**(dim-1),plusHi=new Float32Array(4*size),plusLo=new Float32Array(4*size),minusHi=new Float32Array(4*size),minusLo=new Float32Array(4*size);
  const split=(a,hi,lo)=>{for(let i=0;i<size;i++){const q=Math.floor(i/stride),value=translation+a*Math.sin(2*Math.PI*q/n),high=Math.fround(value);hi[4*i]=high;lo[4*i]=Math.fround(value-high);}};
  split(amplitude+h,plusHi,plusLo);split(amplitude-h,minusHi,minusLo);
  const packedModes=new Int32Array(modes.length*4);modes.forEach((m,i)=>packedModes.set([m,0,0,0],4*i));
  const refPlus=modes.flatMap(m=>exactPiecewiseMode(dim,n,m,plusHi,plusLo)),refMinus=modes.flatMap(m=>exactPiecewiseMode(dim,n,m,minusHi,minusLo));
  const reference=modes.map((m,i)=>(refPlus[2*i]**2+refPlus[2*i+1]**2-refMinus[2*i]**2-refMinus[2*i+1]**2)/(2*h));
  return {input:{dim,n,modes:packedModes,plusHi,plusLo,minusHi,minusLo,h},reference,refPlus,refMinus,amplitude,translation,h,integerModes:modes};
}

/** Fast, Wasm-independent GPU startup check suitable for worker device gating. */
export async function runFiniteDifferenceSelfTest({device,signal,onProgress=()=>{}}={}){
  const rows=[];
  if(!device)return {status:'skip',rows,budget:`weak-scaled relative error <= ${FD_GATE}`,reason:'No WebGPU device was provided'};
  try{
    if(aborted(signal))throw new DOMException('Finite-difference GPU check canceled','AbortError');
    onProgress({stage:'Finite-difference arithmetic precision probe'});
    const precisionProbe=await runPrecisionProbe(device,{signal});
    rows.push(...precisionProbe.rows.map(row=>({...row,shaderHash:precisionProbe.shaderHash,arithmetic:precisionProbe.arithmetic})));
    const analyticFixtures=[
      {label:'2D sinusoid',fixture:analyticFixture({dim:2,n:16,amplitude:.08,modes:[0,1,-1,6]})},
      {label:'3D non-radix sinusoid',fixture:analyticFixture({dim:3,n:12,amplitude:.08,modes:[0,1,-1,5]})},
      {label:'translated sinusoid',fixture:analyticFixture({dim:2,n:16,amplitude:.08,translation:.173,modes:[0,1,-1]})},
      {label:'folded sinusoid',fixture:analyticFixture({dim:2,n:16,amplitude:.2,modes:[0,1,-1,6]})},
    ];
    const outputs=[];
    for(const [index,item] of analyticFixtures.entries()){
      if(aborted(signal))throw new DOMException('Finite-difference GPU check canceled','AbortError');
      onProgress({stage:`Finite-difference independent analytic fixture: ${item.label}`,caseIndex:index+1,cases:analyticFixtures.length});
      const got=await runFiniteDifference(device,item.fixture.input,{signal,onProgress});
      const error=compare(got.values,item.fixture.reference);outputs.push({label:item.label,got:{values:got.values,plus:got.plus,minus:got.minus,delta:got.delta,diagnostics:got.diagnostics},provenance:got.provenance,fixture:item.fixture});
      rows.push({label:`${item.label} FD agrees with independent piecewise-linear segment integral`,...error,dim:item.fixture.input.dim,n:item.fixture.input.n,amplitude:item.fixture.amplitude,translation:item.fixture.translation,h:item.fixture.h,modes:item.fixture.integerModes,repeatedTransverseKnots:true,reference:'closed-form integral over each linear sheet segment',tolerance:{weakScaledRelative:FD_GATE},timings:got.timings,diagnostics:got.diagnostics,provenance:got.provenance??got.diagnostics?.provenance,endpointPlus:Array.from(got.plus),endpointMinus:Array.from(got.minus),referencePlus:item.fixture.refPlus,referenceMinus:item.fixture.refMinus});
    }
    const first=outputs[0],zeroIndex=first.fixture.integerModes.indexOf(0),plusIndex=first.fixture.integerModes.indexOf(1),minusIndex=first.fixture.integerModes.indexOf(-1);
    rows.push({label:'Analytic finite-difference fixture includes zero and negative modes',pass:zeroIndex>=0&&plusIndex>=0&&minusIndex>=0&&Math.abs(first.got.values[zeroIndex])<=1e-9&&Math.abs(first.got.values[plusIndex]-first.got.values[minusIndex])<=FD_GATE*Math.max(Math.abs(first.got.values[plusIndex]),WEAK_FLOOR),zeroModeValue:first.got.values[zeroIndex],positiveMode:first.got.values[plusIndex],negativeMode:first.got.values[minusIndex]});
    const expectedDelta=first.fixture.refPlus.map((value,i)=>value-first.fixture.refMinus[i]);
    const componentError=(actual,reference)=>{if(actual.length!==reference.length)return Infinity;let maximum=0;for(let i=0;i<actual.length;i++)maximum=Math.max(maximum,Math.abs(actual[i]-reference[i]));return maximum;};
    const stableDeltaErrors={plus:componentError(first.got.plus,first.fixture.refPlus),minus:componentError(first.got.minus,first.fixture.refMinus),delta:componentError(first.got.delta,expectedDelta)};
    const stableDeltaMaximum=Math.max(...Object.values(stableDeltaErrors));
    rows.push({label:'GPU plus/minus/delta amplitudes agree with independent exact piecewise-segment integrals',pass:Number.isFinite(stableDeltaMaximum)&&stableDeltaMaximum<=1e-9,maximumAbsoluteComponentError:stableDeltaMaximum,componentErrors:stableDeltaErrors,tolerance:{absolute:1e-9},deltaReference:'independently integrated F(A+h) minus F(A-h) from the exact piecewise-linear segments'});
    const deltaProvenance=first.provenance??first.got.diagnostics?.provenance,deltaQuadrature=first.got.diagnostics?.endpointDeltaQuadrature;
    rows.push({label:'Finite-difference metadata describes the endpoint-delta quadrature path',pass:deltaQuadrature==='3-point Gauss-Legendre, exact continuous directional derivative along retained endpoints'&&String(deltaProvenance?.method??'').includes('stable endpoint amplitude delta from three-point Gauss integration'),endpointDeltaQuadrature:deltaQuadrature,method:deltaProvenance?.method??null});
    const failures=rows.filter(row=>row.pass===false||row.failedModes>0);
    return {status:failures.length?'fail':'pass',rows,failures,budget:{weakScaledRelative:FD_GATE,aim:FD_AIM,weakModeDenominatorFloor:WEAK_FLOOR},precision:'f32 expansion input and GPU arithmetic compared with independent f64 segment integral',precisionProbe,provenance:outputs[0]?.provenance};
  }catch(error){return {status:error?.name==='AbortError'?'skip':'fail',reason:String(error?.stack||error),rows,failures:rows.filter(row=>row.pass===false),budget:{weakScaledRelative:FD_GATE,aim:FD_AIM,weakModeDenominatorFloor:WEAK_FLOOR}};}
}

async function failureChecks(device,signal){
  const rows=[],fixture=analyticFixture({dim:2,n:16});
  const abortedController=new AbortController();abortedController.abort();
  try{await runFiniteDifference(device,fixture.input,{signal:abortedController.signal});rows.push({label:'Pre-aborted GPU finite-difference request stops before execution',pass:false});}
  catch(error){rows.push({label:'Pre-aborted GPU finite-difference request stops before execution',pass:error?.name==='AbortError',reason:String(error)});}
  const activeController=new AbortController();let activeProgress=null;
  const forwardAbort=()=>activeController.abort();signal?.addEventListener('abort',forwardAbort,{once:true});
  try{
    await runFiniteDifference(device,fixture.input,{signal:activeController.signal,batchChunks:1,onProgress:(p)=>{if(p.stage==='endpoint-integrals'&&p.batch===1){activeProgress={stage:p.stage,batch:p.batch,batches:p.batches};activeController.abort();}}});
    rows.push({label:'Cancellation between GPU finite-difference endpoint batches stops remaining work',pass:false,activeProgress});
  }catch(error){rows.push({label:'Cancellation between GPU finite-difference endpoint batches stops remaining work',pass:error?.name==='AbortError'&&activeProgress?.batches>1,activeProgress,reason:String(error)});}
  finally{signal?.removeEventListener('abort',forwardAbort);}
  try{await runFiniteDifference({limits:{maxStorageBufferBindingSize:8,maxBufferSize:8}},fixture.input,{signal});rows.push({label:'Finite-difference storage limit is rejected before allocation',pass:false});}
  catch(error){rows.push({label:'Finite-difference storage limit is rejected before allocation',pass:/limit|exceed|buffer/i.test(String(error?.message||error)),reason:String(error)});}
  const invalidCases=[
    {label:'Noncanonical finite-difference step is rejected',input:{...fixture.input,h:2e-4},pattern:/canonical h=1e-4/},
    {label:'Non-Float32 packed endpoint array is rejected',input:{...fixture.input,plusHi:Float64Array.from(fixture.input.plusHi)},pattern:/plusHi must be packed vec4 Float32Array/},
    {label:'Nonfinite endpoint displacement is rejected',input:(()=>{const input={...fixture.input,plusHi:fixture.input.plusHi.slice()};input.plusHi[0]=NaN;return input;})(),pattern:/nonfinite/},
    {label:'2D finite-difference mode with a nonzero z component is rejected',input:(()=>{const input={...fixture.input,modes:fixture.input.modes.slice()};input.modes[2]=1;return input;})(),pattern:/nonzero z component in dim=2/},
    {label:'Endpoint phase-gap bound above 0.005 is rejected',input:(()=>{const input={...fixture.input,plusHi:fixture.input.plusHi.slice()};input.plusHi[0]+=1e-3;return input;})(),pattern:/endpoint half-angle bound .* exceeds the centered-difference quadrature limit 0\.005/},
  ];
  for(const item of invalidCases){
    try{await runFiniteDifference(device,item.input,{signal});rows.push({label:item.label,pass:false});}
    catch(error){rows.push({label:item.label,pass:item.pattern.test(String(error?.message||error)),reason:String(error)});}
  }
  return rows;
}

async function oneRequest(device,r,{signal,onProgress}={}){
  if(aborted(signal))throw new DOMException('Finite-difference checks canceled','AbortError');
  let prepared;
  try{
    prepared=new PreparedLabRun(JSON.stringify(r));
    const input=inputFromPrepared(prepared),cpuStart=now(),cpu=prepared.finite_difference_cpu(),cpuMs=now()-cpuStart;
    const gpuStart=now(),gpu=await runFiniteDifference(device,input,{signal,onProgress}),gpuWallMs=now()-gpuStart;
    const accuracy=compare(gpu.values,cpu),direct=prepared.direct_cpu();
    const execution={requestedBackend:'gpu',executedBackend:'gpu',stageBackends:{finiteDifference:{backend:'gpu',precision:'f32; f64-expanded input'}},channelProvenance:{finiteDifference:'gpu-f32'}};
    const gpuFinal=readFinalizer(prepared,direct,gpu.values,execution),cpuFinal=readFinalizer(prepared,direct,cpu,{...execution,executedBackend:'cpu',stageBackends:{finiteDifference:{backend:'cpu',precision:'f64'}},channelProvenance:{finiteDifference:'cpu-f64'}});
    const finalizerWeakFloor=WEAK_FLOOR*r.boxSize**r.dim;
    const finalized=compare(finalizerValues(gpuFinal),finalizerValues(cpuFinal),{weakFloor:finalizerWeakFloor});
    const relativeError=gpuFinal?.gradient?.relativeError??[];
    const gatePass=relativeError.length===cpu.length&&relativeError.every(Number.isFinite)&&relativeError.every(value=>value<=FD_GATE);
    const crossingRequired=r.dim===3&&r.n===16&&r.scaleFactor===1,pmCrossed=Boolean(gpuFinal?.guard?.pmCrossed);
    return {label:`${r.dim}D n=${r.n} ${r.ic} ${r.omegaM===1?'EdS':'LCDM'} a=${r.scaleFactor}`,request:{...r},pass:accuracy.pass&&finalized.pass&&gatePass&&(!crossingRequired||pmCrossed),
      accuracy:{...accuracy,tolerance:{weakScaledRelative:FD_GATE,aim:FD_AIM,weakModeDenominatorFloor:WEAK_FLOOR}},finalizer:{pass:finalized.pass,...finalized,weakFloorInPhysicalUnits:finalizerWeakFloor,gatePass,maximumAmplitudeDerivativeRelativeError:Math.max(0,...relativeError),fields:['gradient.finiteDifference','gradient.relativeError']},
      crossing:{required:crossingRequired,pmCrossed},timings:{cpuFiniteDifferenceMs:cpuMs,gpuFiniteDifferenceWallMs:gpuWallMs,gpu:gpu.timings},diagnostics:gpu.diagnostics,provenance:gpu.provenance??gpu.diagnostics?.provenance};
  }finally{prepared?.free?.();}
}

async function hashTypedArray(values){
  if(!globalThis.crypto?.subtle)return null;
  const bytes=new Uint8Array(values.buffer,values.byteOffset,values.byteLength);
  const digest=await crypto.subtle.digest('SHA-256',bytes);
  return [...new Uint8Array(digest)].map(value=>value.toString(16).padStart(2,'0')).join('');
}

async function runFiniteDifferenceRepeatCase(device,r,label,{signal,onProgress,repeats}){
  let prepared;
  const rows=[],runs=[];
  try{
    prepared=new PreparedLabRun(JSON.stringify(r));
    const base=inputFromPrepared(prepared);
    const input={...base,modes:base.modes.slice(),plusHi:base.plusHi.slice(),plusLo:base.plusLo.slice(),minusHi:base.minusHi.slice(),minusLo:base.minusLo.slice()};
    const cpu=Float64Array.from(prepared.finite_difference_cpu());
    const inputHashes={modes:await hashTypedArray(input.modes),plusHi:await hashTypedArray(input.plusHi),plusLo:await hashTypedArray(input.plusLo),minusHi:await hashTypedArray(input.minusHi),minusLo:await hashTypedArray(input.minusLo),cpuFiniteDifference:await hashTypedArray(cpu)};
    const hashesAvailable=Object.values(inputHashes).every(value=>typeof value==='string');
    const outputHashes=[];
    for(let repeat=1;repeat<=repeats;repeat++){
      if(aborted(signal))throw new DOMException('Finite-difference repeated-run check canceled','AbortError');
      onProgress({stage:`Finite-difference stable-input repeat: ${label}`,caseIndex:repeat,cases:repeats});
      const gpu=await runFiniteDifference(device,input,{signal,onProgress});
      const [valuesHash,plusHash,minusHash,deltaHash]=await Promise.all([hashTypedArray(gpu.values),hashTypedArray(gpu.plus),hashTypedArray(gpu.minus),hashTypedArray(gpu.delta)]);
      const inputHashesAfter={modes:await hashTypedArray(input.modes),plusHi:await hashTypedArray(input.plusHi),plusLo:await hashTypedArray(input.plusLo),minusHi:await hashTypedArray(input.minusHi),minusLo:await hashTypedArray(input.minusLo)};
      const inputsUnchanged=Object.entries(inputHashesAfter).every(([name,value])=>value===inputHashes[name]);
      const finite=[...gpu.values,...gpu.plus,...gpu.minus,...gpu.delta].every(Number.isFinite);
      const accuracy=compare(gpu.values,cpu);
      const first=runs[0];
      const maximumDifference=(actual,reference)=>{
        if(!reference)return 0;
        if(actual.length!==reference.length)return Infinity;
        let maximum=0;for(let i=0;i<actual.length;i++)maximum=Math.max(maximum,Math.abs(actual[i]-reference[i]));return maximum;
      };
      const componentRepeatError={plusMaximumAbsolute:maximumDifference(gpu.plus,first?.endpointPlus),minusMaximumAbsolute:maximumDifference(gpu.minus,first?.endpointMinus),deltaMaximumAbsolute:maximumDifference(gpu.delta,first?.endpointDelta)};
      componentRepeatError.maximumAbsolute=Math.max(componentRepeatError.plusMaximumAbsolute,componentRepeatError.minusMaximumAbsolute,componentRepeatError.deltaMaximumAbsolute);
      componentRepeatError.pass=componentRepeatError.maximumAbsolute<=FD_REPEAT_ENDPOINT_BUDGET;
      outputHashes.push({values:valuesHash,plus:plusHash,minus:minusHash,delta:deltaHash});
      runs.push({caseLabel:label,repeat,finite,inputsUnchanged,accuracy:{pass:accuracy.pass,maximumAbsolute:accuracy.maximumAbsolute,maximumWeakScaledRelative:accuracy.maximumWeakScaledRelative,maximumGateFraction:accuracy.maximumGateFraction,failedModes:accuracy.failedModes,withinAim:accuracy.withinAim,tolerance:{weakScaledRelative:FD_GATE,aim:FD_AIM,weakModeDenominatorFloor:WEAK_FLOOR}},componentRepeatError,inputHashesAfter,outputHashes:{values:valuesHash,plus:plusHash,minus:minusHash,delta:deltaHash},sameAsFirst:repeat===1?true:valuesHash===outputHashes[0].values&&plusHash===outputHashes[0].plus&&minusHash===outputHashes[0].minus&&deltaHash===outputHashes[0].delta,values:Array.from(gpu.values),endpointPlus:Array.from(gpu.plus),endpointMinus:Array.from(gpu.minus),endpointDelta:Array.from(gpu.delta),diagnostics:gpu.diagnostics,provenance:gpu.provenance??gpu.diagnostics?.provenance});
    }
    const outputsStable=runs.length===repeats&&runs.every(run=>run.sameAsFirst);
    rows.push({caseLabel:label,label:'Repeated GPU FD calls retain byte-identical captured inputs',pass:hashesAvailable&&runs.length===repeats&&runs.every(run=>run.inputsUnchanged),inputHashes,referenceCpuFiniteDifference:Array.from(cpu),repeats:runs.length});
    rows.push({caseLabel:label,label:'Every repeated GPU FD result agrees with CPU FD within the existing 1e-5 gate',pass:runs.length===repeats&&runs.every(run=>run.accuracy.pass),tolerance:{weakScaledRelative:FD_GATE,aim:FD_AIM,weakModeDenominatorFloor:WEAK_FLOOR},accuracy:runs.map(({repeat,accuracy})=>({repeat,...accuracy}))});
    rows.push({caseLabel:label,label:'Repeated GPU FD plus/minus/delta components agree within the endpoint repeat budget',pass:runs.length===repeats&&runs.every(run=>run.finite&&run.componentRepeatError.pass),maximumComponentAbsoluteDifferenceVsFirst:Math.max(0,...runs.map(run=>run.componentRepeatError.maximumAbsolute)),tolerance:{absolute:FD_REPEAT_ENDPOINT_BUDGET},outputsStable,outputHashes});
    return {label,request:r,repeats:runs.length,status:rows.every(row=>row.pass)?'pass':'fail',inputHashes,cpuFiniteDifference:Array.from(cpu),endpointRepeatBudget:{absolute:FD_REPEAT_ENDPOINT_BUDGET},outputsStable,runs,rows};
  }finally{prepared?.free?.();}
}

/** Run fixed-input determinism diagnostics on wave and Gaussian LCDM requests. */
export async function runFiniteDifferenceRepeatCheck({device,signal,onProgress=()=>{},repeats=20}={}){
  if(!device)return {status:'skip',reason:'No WebGPU device was provided',rows:[],runs:[],cases:[]};
  await init();
  const requests=[
    {label:'3D n=32 waves LCDM a=0.25',request:request({dim:3,n:32,mesh:32,forceMesh:32,order:4,ic:'waves',omegaM:.315,scaleFactor:.25})},
    {label:'3D n=32 Gaussian LCDM a=0.25',request:request({dim:3,n:32,mesh:32,forceMesh:32,order:4,ic:'gaussian',initialCutoff:3,omegaM:.315,scaleFactor:.25})},
  ];
  const cases=[];
  try{
    for(const item of requests){
      if(aborted(signal))throw new DOMException('Finite-difference repeated-run check canceled','AbortError');
      cases.push(await runFiniteDifferenceRepeatCase(device,item.request,item.label,{signal,onProgress,repeats}));
    }
    const rows=cases.flatMap(item=>item.rows),runs=cases.flatMap(item=>item.runs);
    return {status:cases.every(item=>item.status==='pass')?'pass':'fail',repeatsPerCase:repeats,totalRepeats:runs.length,cases,rows,runs,inputHashes:Object.fromEntries(cases.map(item=>[item.label,item.inputHashes])),endpointRepeatBudget:{absolute:FD_REPEAT_ENDPOINT_BUDGET},precision:'Two single prepared Rust requests with fixed captured arrays; each has one CPU FD oracle and twenty GPU calls. Raw values, endpoint amplitudes, delta, accuracy, component repeat errors, and hashes are retained.'};
  }catch(error){
    const rows=cases.flatMap(item=>item.rows);
    return {status:error?.name==='AbortError'?'skip':'fail',reason:String(error?.stack||error),repeatsPerCase:repeats,totalRepeats:cases.reduce((sum,item)=>sum+item.runs.length,0),cases,rows,runs:cases.flatMap(item=>item.runs),endpointRepeatBudget:{absolute:FD_REPEAT_ENDPOINT_BUDGET}};
  }
}

const FIXTURES=[
  request({dim:2,n:32,mesh:32,forceMesh:32,order:3,ic:'waves',omegaM:.315,scaleFactor:.25}),
  request({dim:3,n:32,mesh:32,forceMesh:32,order:4,ic:'waves',omegaM:1,scaleFactor:.25}),
  request({dim:2,n:24,mesh:24,forceMesh:24,order:2,ic:'waves',omegaM:.315,scaleFactor:.5}),
  request({dim:3,n:32,mesh:32,forceMesh:32,order:4,ic:'gaussian',initialCutoff:3,omegaM:.315,scaleFactor:.25}),
  request({dim:3,n:16,mesh:16,forceMesh:32,order:4,ic:'waves',omegaM:.315,scaleFactor:1}),
];

async function runBenchmarks(device,{signal,onProgress}={}){
  const rows=[],timings=[];
  for(const n of [32,64]){
    const r=request({dim:3,n,mesh:n,forceMesh:n,order:4,ic:'waves',omegaM:.315,scaleFactor:.25});
    const samples=[];
    for(let repeat=1;repeat<=3;repeat++){
      if(aborted(signal))throw new DOMException('Finite-difference benchmark canceled','AbortError');
      let p;
      try{
        p=new PreparedLabRun(JSON.stringify(r));const input=inputFromPrepared(p);
        onProgress?.({stage:`Finite-difference ${n}³ benchmark`,caseIndex:repeat,cases:3});
        const cpuStart=now(),cpu=p.finite_difference_cpu(),cpuMs=now()-cpuStart;
        const gpuStart=now(),gpu=await runFiniteDifference(device,input,{signal,onProgress}),gpuWallMs=now()-gpuStart;
        const accuracy=compare(gpu.values,cpu);
        samples.push({repeat,cpuMs,gpuWallMs,gpuTimings:gpu.timings,pass:accuracy.pass,maximumWeakScaledRelative:accuracy.maximumWeakScaledRelative,failedModes:accuracy.failedModes});
      }finally{p?.free?.();}
    }
    rows.push({label:`Matched CPU f64 and GPU finite-difference timing for 3D n=${n}`,request:{...r},pass:samples.length===3&&samples.every(s=>s.pass),samples,description:'Three fresh prepared requests per grid; CPU oracle and GPU kernel are both included, no numerical arrays retained.'});
    timings.push({request:{...r},repeats:samples.length,cpuMs:samples.map(s=>s.cpuMs),gpuWallMs:samples.map(s=>s.gpuWallMs),gpuKernelMs:samples.map(s=>s.gpuTimings?.kernelMs??null),samples});
  }
  return {rows,timings};
}

/** Complete-request FD oracle suite and optional matched CPU/GPU timings. */
export async function runFiniteDifferenceChecks({device,signal,onProgress=()=>{},benchmark=false}={}){
  if(!device)return {status:'skip',rows:[],timings:[],reason:'No WebGPU device was provided'};
  await init();const rows=[],cases=[],timings=[];
  try{
    const selfTest=await runFiniteDifferenceSelfTest({device,signal,onProgress});
    rows.push(...(selfTest.rows.length?selfTest.rows:[{label:'Independent GPU finite-difference startup fixtures',pass:selfTest.status==='pass',reason:selfTest.reason??null}]));
    rows.push(...await failureChecks(device,signal));
    for(let i=0;i<FIXTURES.length;i++){
      if(aborted(signal))throw new DOMException('Finite-difference checks canceled','AbortError');
      const r=FIXTURES[i];onProgress({stage:'Finite-difference complete request',caseIndex:i+1,cases:FIXTURES.length});
      try{const result=await oneRequest(device,r,{signal,onProgress});cases.push(result);rows.push({label:`${result.label}: GPU FD matches retained CPU FD at h=1e-4`,pass:result.pass,request:result.request,accuracy:result.accuracy,finalizer:result.finalizer});timings.push({request:result.request,...result.timings});}
      catch(error){const row={label:`Finite-difference request ${i+1}`,pass:false,request:{...r},error:String(error?.stack||error)};rows.push(row);cases.push(row);}
    }
    if(benchmark){const measured=await runBenchmarks(device,{signal,onProgress});rows.push(...measured.rows);timings.push(...measured.timings);}
    const worker=await runFiniteDifferenceWorkerChecks({signal,onProgress,benchmark});
    rows.push(...worker.rows);timings.push(...worker.timings);
    if(worker.status==='fail'&&!worker.rows.some(row=>row.pass===false))rows.push({label:'FD worker lifecycle checks return successfully',pass:false,reason:worker.reason??'Worker lifecycle check failed without a detailed row.'});
    const failures=rows.filter(row=>row.pass===false||row.failedModes>0);
    return {status:failures.length?'fail':'pass',rows,cases,timings,failures,selfTest,worker,precision:'GPU finite difference uses f32 arithmetic over Rust f64 hi/lo position expansions; matched with the retained full-request Rust f64 oracle',gate:{weakScaledRelative:FD_GATE,aim:FD_AIM,weakModeDenominatorFloor:WEAK_FLOOR,h:1e-4},fullRequestGate:'Maximum weak-scaled relative error <= 1e-5; the zero/weak-mode denominator floor is 1e-9.'};
  }catch(error){return {status:error?.name==='AbortError'?'skip':'fail',reason:String(error?.stack||error),rows,cases,timings,failures:rows.filter(row=>row.pass===false),gate:{weakScaledRelative:FD_GATE,aim:FD_AIM,weakModeDenominatorFloor:WEAK_FLOOR,h:1e-4}};}
}

let workerSerial=0;
function startFdWorker(){
  if(typeof Worker!=='function')throw new Error('Module workers are unavailable in this browser');
  return new Worker(new URL('../worker.js',import.meta.url),{type:'module',name:'finite-difference-harness-worker'});
}
function fdWorkerRun(worker,state,policy,{signal,onMessage=()=>{},timeoutMs=600000}={}){
  if(aborted(signal))return Promise.reject(new DOMException('Finite-difference worker check canceled','AbortError'));
  const requestId=++workerSerial,started=now();
  return new Promise((resolve,reject)=>{
    let timer;
    const cleanup=()=>{clearTimeout(timer);signal?.removeEventListener('abort',abort);worker.removeEventListener('message',message);worker.removeEventListener('error',error);};
    const abort=()=>worker.postMessage({type:'cancel',requestId});
    const error=(event)=>{cleanup();reject(new Error(event.message||'Finite-difference worker failed'));};
    const message=({data})=>{
      if(data?.requestId!==requestId)return;
      if(data.type==='phase'||data.type==='progress'){onMessage(data);return;}
      if(data.type==='cancelled'){cleanup();resolve({type:'cancelled',data,wallMilliseconds:now()-started});return;}
      if(data.type==='result'){cleanup();resolve({type:'result',data,wallMilliseconds:now()-started});}
    };
    worker.addEventListener('message',message);worker.addEventListener('error',error);
    timer=setTimeout(()=>{cleanup();reject(new Error(`Finite-difference worker request ${requestId} timed out`));},timeoutMs);
    if(signal){if(signal.aborted)abort();else signal.addEventListener('abort',abort,{once:true});}
    worker.postMessage({type:'run',requestId,key:`fd-check-${requestId}`,state,policy});
  });
}
function fdWorkerOutput(reply){return reply?.data?.result??null;}
function fdWorkerExecution(reply){return reply?.data?.execution??reply?.data?.result?.execution??null;}
function sameJsonFields(a,b,fields){return fields.every(key=>JSON.stringify(a?.[key])===JSON.stringify(b?.[key]));}
function compareFinalizedFd(actual,reference,dim,boxSize){
  const a=actual?.gradient?.finiteDifference??[],b=reference?.gradient?.finiteDifference??[];
  if(a.length!==b.length)return {pass:false,failedModes:Math.max(a.length,b.length),maximumWeakScaledRelative:Infinity};
  const floor=WEAK_FLOOR*boxSize**dim;let maximum=0,failed=0;
  for(let i=0;i<a.length;i++){const rel=Math.abs(a[i]-b[i])/Math.max(Math.abs(b[i]),floor);maximum=Math.max(maximum,rel);if(!Number.isFinite(rel)||rel>FD_GATE)failed++;}
  return {pass:failed===0,failedModes:failed,maximumWeakScaledRelative:maximum,weakFloorInPhysicalUnits:floor};
}
async function stopFdWorker(worker,timeoutMs=5000){
  return new Promise(resolve=>{let timer;const done=ack=>{clearTimeout(timer);worker.removeEventListener('message',message);worker.terminate();resolve(ack);};const message=({data})=>{if(data?.type==='shutdown-ack')done(true);};worker.addEventListener('message',message);timer=setTimeout(()=>done(false),timeoutMs);worker.postMessage({type:'shutdown'});});
}

async function cancellationWorkerCheck({signal,onProgress}={}){
  const rows=[],worker=startFdWorker();let shutdownAck=false;
  try{
    const small=request({dim:2,n:16,mesh:16,forceMesh:16,steps:32,order:2,scaleFactor:.25});
    const warm=await fdWorkerRun(worker,small,{requested:'gpu'},{signal,onMessage:onProgress});
    const warmExec=fdWorkerExecution(warm),warmFd=warmExec?.finiteDifference;
    const warmPass=warm.type==='result'&&['gpu','hybrid'].includes(warmExec?.executedBackend)&&warmFd?.backend==='gpu';
    rows.push({label:'FD cancellation worker warms a complete GPU direct/PM/FD path',pass:warmPass,workerStatus:warm.type,directBackend:warmExec?.executedBackend,fdBackend:warmFd?.backend,request:{...small}});
    if(!warmPass)rows.push({label:'FD cancellation is sent only after an endpoint-integral GPU batch',pass:false,reason:'GPU finite-difference warmup did not complete'});
    else {
    const batchRequest=request({dim:3,n:32,mesh:32,forceMesh:32,steps:32,order:4,scaleFactor:.25});
    const activeId=++workerSerial,queuedId=++workerSerial,replacement=request({dim:2,n:16,mesh:16,forceMesh:16,steps:32,order:2,scaleFactor:.25});
    let activeProgress=null;
    const outcome=await new Promise((resolve,reject)=>{
      let timer,canceled=false,replaced=null,cancelSent=false;
      const cleanup=()=>{clearTimeout(timer);worker.removeEventListener('message',message);worker.removeEventListener('error',error);};
      const error=event=>{cleanup();reject(new Error(event.message||'FD cancellation worker failed'));};
      const message=({data})=>{
        if(data?.requestId===activeId&&data.type==='progress'&&data.phase==='finite-difference'&&String(data.stage??'').toLowerCase().includes('endpoint-integrals')&&!cancelSent){
          const batch=Number(data.batch??data.step??1);
          if(batch>=1){activeProgress={stage:data.stage,batch,batches:data.batches??null};cancelSent=true;
            worker.postMessage({type:'cancel',requestId:activeId});
            worker.postMessage({type:'run',requestId:queuedId,key:'fd-cancel-queued-cpu',state:replacement,policy:{requested:'cpu'}});
          }
        }
        if(data?.requestId===activeId&&data.type==='cancelled')canceled=true;
        if(data?.requestId===queuedId&&data.type==='result')replaced=data;
        if(canceled&&replaced){cleanup();resolve({canceled,replaced,activeProgress});}
      };
      worker.addEventListener('message',message);worker.addEventListener('error',error);timer=setTimeout(()=>{cleanup();reject(new Error('FD cancellation and queued replacement timed out'));},600000);
      worker.postMessage({type:'run',requestId:activeId,key:'fd-cancel-active',state:batchRequest,policy:{requested:'gpu',fdBatchChunks:8}});
    });
    const replacementExec=outcome.replaced?.execution??outcome.replaced?.result?.execution;
    rows.push({label:'FD cancellation waits for an endpoint-integral batch, then accepts a queued CPU replacement',pass:Boolean(outcome.activeProgress)&&outcome.activeProgress.batches>1&&outcome.canceled&&outcome.replaced?.type==='result'&&!outcome.replaced?.error&&replacementExec?.executedBackend==='cpu',activeProgress:outcome.activeProgress,canceled:outcome.canceled,replacementBackend:replacementExec?.executedBackend,request:{...batchRequest},queuedRequest:{...replacement}});
    }
  }catch(error){rows.push({label:'FD cancellation and queue lifecycle completes',pass:false,error:String(error?.stack||error)});}
  finally{shutdownAck=await stopFdWorker(worker);}
  rows.push({label:'FD cancellation worker acknowledges shutdown',pass:shutdownAck,shutdownAck});
  onProgress?.({stage:'Finite-difference cancellation worker complete'});
  return {rows,timings:[]};
}

async function finiteDifferenceWorkerChecks({signal,onProgress=()=>{},benchmark=false}={}){
  const rows=[],timings=[],worker=startFdWorker();
  try{
    const small=request({dim:2,n:16,mesh:16,forceMesh:16,steps:32,order:2,scaleFactor:.25});
    onProgress({stage:'FD worker CPU finite-difference baseline'});
    const baseline=await fdWorkerRun(worker,small,{requested:'gpu',disablePm:true,disableFd:true},{signal,onMessage:onProgress});
    const baseResult=fdWorkerOutput(baseline),baseExec=fdWorkerExecution(baseline),baseFd=baseExec?.finiteDifference;
    rows.push({label:'Worker GPU observation path with explicit CPU FD baseline',pass:baseline.type==='result'&&['gpu','hybrid'].includes(baseExec?.executedBackend)&&baseFd?.backend==='cpu'&&/disabled/i.test(baseFd?.fallbackReason??''),directBackend:baseExec?.executedBackend,fdBackend:baseFd?.backend,request:{...small}});
    for(const failure of ['validation','limits','readback','nonfinite','device-loss']){
      if(aborted(signal))throw new DOMException('Finite-difference worker checks canceled','AbortError');
      onProgress({stage:`FD worker recovery: ${failure}`});
      const reply=await fdWorkerRun(worker,small,{requested:'gpu',disablePm:true,forceFdFailure:failure},{signal,onMessage:onProgress});
      const output=fdWorkerOutput(reply),execution=fdWorkerExecution(reply),fd=execution?.finiteDifference;
      const same=sameJsonFields(output,baseResult,['gradient','spectra','pdf','guard','diagnostics']);
      const fallbackProvenance=execution?.channelProvenance?.finiteDifference==='cpu-f64'&&execution?.stageBackends?.finiteDifference?.backend==='cpu';
      const expectedReason={validation:/self-check did not pass/i,limits:/maxGpuBytes/i,readback:/readback failure/i,nonfinite:/malformed or non-finite/i,'device-loss':/device loss/i}[failure];
      const failureReached=fd?.gpuAttempt?.validation?.status==='pass'&&expectedReason.test(fd?.fallbackReason??'');
      rows.push({label:`Worker ${failure} FD failure falls back to retained CPU finite difference`,pass:reply.type==='result'&&fd?.backend==='cpu'&&failureReached&&same&&fallbackProvenance,fdBackend:fd?.backend,fallbackReason:fd?.fallbackReason,failureReached,gpuAttemptValidation:fd?.gpuAttempt?.validation?.status??null,fallbackProvenance,channelProvenance:execution?.channelProvenance?.finiteDifference??null,stageBackend:execution?.stageBackends?.finiteDifference?.backend??null,physicalResultsMatchCpuFdBaseline:same,request:{...small}});
    }
    const recovered=await fdWorkerRun(worker,small,{requested:'gpu'},{signal,onMessage:onProgress});
    const recoveredExec=fdWorkerExecution(recovered),recoveredFd=recoveredExec?.finiteDifference;
    rows.push({label:'Worker reacquires GPU finite-difference after all injected failures',pass:recovered.type==='result'&&['gpu','hybrid'].includes(recoveredExec?.executedBackend)&&recoveredFd?.backend==='gpu'&&recoveredExec?.channelProvenance?.finiteDifference==='gpu-compensated-f32',backend:recoveredExec?.executedBackend,fdBackend:recoveredFd?.backend,request:{...small}});

    if(benchmark){
      const state=request({dim:3,n:32,mesh:32,forceMesh:32,steps:32,order:4,scaleFactor:.25});
      const samples=[];
      for(let repeat=1;repeat<=3;repeat++){
        if(aborted(signal))throw new DOMException('Finite-difference end-to-end benchmark canceled','AbortError');
        const order=repeat%2?['cpu-fd','gpu-fd']:['gpu-fd','cpu-fd'],pair={repeat,order,request:{...state},samples:{},outputs:{}};
        for(const kind of order){
          const policy=kind==='cpu-fd'?{requested:'gpu',disableFd:true}:{requested:'gpu'};
          onProgress({stage:`Worker 32³ end-to-end ${kind}`,caseIndex:repeat,cases:3});
          const reply=await fdWorkerRun(worker,state,policy,{signal,onMessage:onProgress}),execution=fdWorkerExecution(reply),fd=execution?.finiteDifference;
          const sample={kind,repeat,status:reply.type,error:reply.data?.error??null,wallMilliseconds:reply.wallMilliseconds,requestedBackend:execution?.requestedBackend??null,executedBackend:execution?.executedBackend??null,pmBackend:execution?.stageBackends?.trajectories?.pmBackend??null,fdBackend:fd?.backend??null,fdReason:fd?.fallbackReason??null,fdTiming:execution?.timings?.stages?.finiteDifferenceMilliseconds??null,provenance:execution?.channelProvenance?.finiteDifference??null};
          samples.push(sample);pair.samples[kind]=sample;pair.outputs[kind]=fdWorkerOutput(reply);
        }
        const comparison=compareFinalizedFd(pair.outputs['gpu-fd'],pair.outputs['cpu-fd'],state.dim,state.boxSize);
        pair.samples['gpu-fd'].comparisonToCpuFd=comparison;pair.samples['cpu-fd'].comparisonToGpuFd=comparison;
        delete pair.outputs;
        pair.comparison=comparison;
      }
      const baseline=samples.filter(s=>s.kind==='cpu-fd'),gpu=samples.filter(s=>s.kind==='gpu-fd');
      const pairRows=[];for(let i=0;i<samples.length;i+=2){const first=samples[i],second=samples[i+1],cpu=first.kind==='cpu-fd'?first:second,gpuSample=first.kind==='gpu-fd'?first:second;pairRows.push({repeat:cpu.repeat,pass:Boolean(cpu.comparisonToGpuFd?.pass&&gpuSample.comparisonToCpuFd?.pass),comparison:cpu.comparisonToGpuFd});}
      const pass=baseline.length===3&&gpu.length===3&&baseline.every(s=>s.status==='result'&&s.fdBackend==='cpu'&&['gpu','hybrid'].includes(s.executedBackend)&&s.pmBackend==='gpu')&&gpu.every(s=>s.status==='result'&&s.fdBackend==='gpu'&&['gpu','hybrid'].includes(s.executedBackend)&&s.pmBackend==='gpu')&&pairRows.length===3&&pairRows.every(row=>row.pass);
      const row={label:'GPU observer/PM with CPU FD vs GPU FD 32³ worker timings',pass,request:{...state},repeats:3,cpuFdWallMs:baseline.map(s=>s.wallMilliseconds),gpuFdWallMs:gpu.map(s=>s.wallMilliseconds),pairedAccuracy:pairRows,samples,description:'Alternating six complete requests through one persistent worker after validation; both modes use GPU direct/PM, and only the FD backend differs.'};
      rows.push(row);timings.push(row);
    }
  }catch(error){rows.push({label:'FD worker fallback, reacquisition, and timing checks complete',pass:false,error:String(error?.stack||error)});}
  finally{const shutdownAck=await stopFdWorker(worker);rows.push({label:'FD recovery worker acknowledges shutdown',pass:shutdownAck,shutdownAck});}
  const cancellation=await cancellationWorkerCheck({signal,onProgress});rows.push(...cancellation.rows);timings.push(...cancellation.timings);
  return {rows,timings};
}

/** Public report helper for FD-specific worker fallback and cancellation checks. */
export async function runFiniteDifferenceWorkerChecks({signal,onProgress=()=>{},benchmark=false}={}){
  try{
    const result=await finiteDifferenceWorkerChecks({signal,onProgress,benchmark});
    return {...result,status:result.rows.some(row=>row.pass===false)?'fail':'pass',failures:result.rows.filter(row=>row.pass===false)};
  }catch(error){return {status:error?.name==='AbortError'?'skip':'fail',reason:String(error?.stack||error),rows:[],timings:[],failures:[]};}
}
