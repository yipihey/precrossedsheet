// Numerical and lifecycle checks for the resident FFT / particle-mesh path.
// The CPU references are Rust/Wasm exports. PM output comparisons distinguish
// f32 input/state rounding from f32 GPU arithmetic; exact sheet deposits and
// PDF semantics are checked through the existing Rust finalizer.
import init, { PendingLabRun, pm_drift_minimum } from '../../pkg/hcc_core.js';
import { GpuFFT } from './fft.js';
import { runPm } from './pm.js';
import { createGpuContext } from './context.js';
import { readBuffer } from './util.js';

const FFT_ABS = 2e-5;
const PM_BUDGET = Object.freeze({
  x:  { absolute: 2e-5, relative: 2e-5 },
  p:  { absolute: 5e-6, relative: 5e-5 },
  dx: { absolute: 2e-5, relative: 5e-5 },
  dp: { absolute: 5e-6, relative: 1e-4 },
});

const request = (overrides={}) => ({
  epoch:'scaleFactor', fraction:.35, steps:32, mesh:16, forceMesh:16,
  omegaB:.049, h:.674, ns:.965, sigma8:.811, boxSize:300,
  initialSmoothing:14, initialCutoff:3, seed:7, ic:'waves', smoothing:.03,
  dim:2, n:16, order:4, omegaM:.315, scaleFactor:.25, ...overrides,
});
const aborted = (signal) => signal?.aborted;
const yieldTask = () => new Promise((resolve)=>setTimeout(resolve,0));

function assert(condition,message){if(!condition)throw new Error(message);}
function finiteArray(a,label){assert(a&&Number.isFinite(a.length),`${label} is missing`);for(let i=0;i<a.length;i++)if(!Number.isFinite(a[i]))throw new Error(`${label}[${i}] is nonfinite`);}
function compare(actual,reference,{absolute,relative}){
  assert(actual.length===reference.length,`length mismatch ${actual.length} vs ${reference.length}`);
  let maxAbs=0,maxBudgetFraction=0,rms2=0,failed=0;
  for(let i=0;i<actual.length;i++){
    const error=Math.abs(actual[i]-reference[i]),budget=absolute+relative*Math.abs(reference[i]);
    maxAbs=Math.max(maxAbs,error);maxBudgetFraction=Math.max(maxBudgetFraction,error/budget);rms2+=error*error;
    if(!Number.isFinite(actual[i])||error>budget)failed++;
  }
  return {maximumAbsolute:maxAbs,rms:Math.sqrt(rms2/Math.max(1,actual.length)),maximumBudgetFraction: maxBudgetFraction,failedComponents:failed,pass:failed===0};
}

function fftBuffer(device,label,data){
  const b=device.createBuffer({label,size:Math.max(16,data.byteLength),usage:0x80|0x04|0x08});
  device.queue.writeBuffer(b,0,data);return b;
}
function complexFixture(n,dim){
  const size=n**dim,data=new Float32Array(size*2),coords=Array(dim).fill(0);
  for(let i=0;i<size;i++){
    let x=i;for(let a=dim-1;a>=0;a--){coords[a]=x%n;x=Math.floor(x/n);}
    const phase=2*Math.PI*coords.reduce((s,q,a)=>s+q*(a+1)/n,0);
    data[2*i]=.4*Math.sin(phase)+.1*Math.cos(phase*.5);data[2*i+1]=.3*Math.cos(phase)-.07*Math.sin(phase*.5);
  }
  return data;
}
async function fftChecks(device){
  const rows=[];
  for(const [dim,n] of [[1,16],[2,16],[3,16]]){
    const fft=await new GpuFFT(device,n,dim).init(),input=complexFixture(n,dim),a=fftBuffer(device,`FFT input ${dim}D`,input),b=fftBuffer(device,`FFT spectrum ${dim}D`,new Float32Array(input.length)),c=fftBuffer(device,`FFT output ${dim}D`,new Float32Array(input.length));
    try{
      fft.forward(a,b);fft.inverse(b,c);await device.queue.onSubmittedWorkDone();
      const got=await readBuffer(device,c,input.byteLength,Float32Array);
      const result=compare(got,input,{absolute:FFT_ABS,relative:FFT_ABS});
      rows.push({label:`${dim}D n=${n} complex FFT roundtrip`,...result,tolerance:`${FFT_ABS} absolute + ${FFT_ABS} relative`});
    }finally{fft.destroy();a.destroy();b.destroy();c.destroy();}
    await yieldTask();
  }
  // A pure positive-frequency complex wave must land in its exact row-major bin.
  const dim=3,n=16,size=n**dim,mode=[2,1,3],data=new Float32Array(size*2);let coords=[0,0,0];
  for(let i=0;i<size;i++){let x=i;for(let a=2;a>=0;a--){coords[a]=x%n;x=Math.floor(x/n);}const phase=2*Math.PI*coords.reduce((s,q,a)=>s+q*mode[a]/n,0);data[2*i]=Math.cos(phase);data[2*i+1]=Math.sin(phase);}
  const fft=await new GpuFFT(device,n,dim).init(),src=fftBuffer(device,'FFT single-mode input',data),dst=fftBuffer(device,'FFT single-mode output',new Float32Array(data.length));
  try{
    fft.forward(src,dst);await device.queue.onSubmittedWorkDone();const got=await readBuffer(device,dst,data.byteLength,Float32Array);
    const target=((mode[0]*n+mode[1])*n+mode[2])*2;let maxLeak=0;for(let i=0;i<got.length;i++)if(i!==target&&i!==target+1)maxLeak=Math.max(maxLeak,Math.abs(got[i]));
    const amplitudeError=Math.hypot(got[target]-size,got[target+1])/size;
    rows.push({label:'3D n=16 single Fourier mode lands in requested Poisson bin',pass:amplitudeError<=2e-5&&maxLeak/size<=2e-5,amplitudeError,maxLeakFraction:maxLeak/size,tolerance:'2e-5 normalized amplitude and leakage'});
  }finally{fft.destroy();src.destroy();dst.destroy();}
  return rows;
}

function clocksFromFlat(flat){
  assert(flat.length%5===0,'Rust PM clocks must contain five values per step');
  const count=flat.length/5,kick0=new Float64Array(count),drift=new Float64Array(count),kick1=new Float64Array(count);
  for(let i=0;i<count;i++){kick0[i]=flat[5*i];drift[i]=flat[5*i+1];kick1[i]=flat[5*i+2];}
  return {kick0,drift,kick1};
}
function clockError(gpu,reference){
  let max=0;for(const key of ['kick0','drift','kick1'])for(let i=0;i<gpu[key].length;i++)max=Math.max(max,Math.abs(Math.fround(gpu[key][i])-reference[key][i]));return max;
}
function splitReference(packed,dim){
  const nv=packed.length/16,out={};for(const [name,block] of [['x',0],['p',1],['dx',2],['dp',3]]){
    const a=new Float64Array(nv*4);a.set(packed.subarray(block*nv*4,(block+1)*nv*4));out[name]=a;
  }
  return out;
}
function pmInput(r,pending){
  const metadata=JSON.parse(pending.metadata());
  const clocks=clocksFromFlat(pending.pm_clocks());
  const initialMinimum=Math.min(metadata.pmInitialMinimumJacobianF64,metadata.pmInitialMinimumJacobianDisplacementF32);
  assert(Number.isFinite(initialMinimum),'Rust initial simplex minimum metadata is missing');
  return {dim:r.dim,n:r.n,meshN:r.forceMesh,boxSize:1,omegaM:r.omegaM,x:pending.pm_displacement(),p:pending.pm_p(),dx:pending.pm_dx(),dp:pending.pm_dp(),clocks,initialMinimum,positionRepresentation:'lattice-displacement',forceKernel:'rust-spectral-cic-v1'};
}
function absolutePositionsFromDisplacement(displacement,dim,n){
  const absolute=new Float32Array(displacement.length);
  for(let i=0;i<n**dim;i++)for(let axis=0;axis<dim;axis++){
    const stride=n**(dim-axis-1),q=Math.floor(i/stride)%n,cell=Math.fround(Math.fround(q)/Math.fround(n));
    absolute[i*4+axis]=Math.fround(cell+displacement[i*4+axis]);
  }
  return absolute;
}
async function pmFixture(device,r,{signal,onProgress,checkAllDrifts=true,compareFullRun=false}={}){
  if(aborted(signal))throw new DOMException('Resident GPU checks canceled','AbortError');
  const pending=new PendingLabRun(JSON.stringify(r));
  try{
  const input=pmInput(r,pending),modeled=clocksFromFlat(pending.pm_clocks()),driftRows=[];
  const got=await runPm(device,input,{signal,checkAllDrifts,allowCrossing:true,onProgress,
    checkDrift:async(start,end,step)=>pm_drift_minimum(r.dim,r.n,start,end),
    onDrift:(row)=>{driftRows.push(row);onProgress?.({stage:'PM drift diagnostics',step:row.step,steps:input.clocks.drift.length});}});
  const expected=splitReference(pending.pm_reference(),r.dim);
  const quantized=typeof pending.pm_reference_displacement_quantized==='function'?splitReference(pending.pm_reference_displacement_quantized(),r.dim):null;
  const output={label:`${r.dim}D n=${r.n} force=${r.forceMesh} ${r.omegaM===1?'EdS':'LCDM'} ${r.ic} a=${r.scaleFactor}`,request:{...r},rows:[],timings:got.timings,diagnostics:got.diagnostics};
  const fullErrors={};
  for(const name of ['x','p','dx','dp']){
    const budget=PM_BUDGET[name],refQuantized=quantized?.[name]??expected[name],error=compare(got[name],refQuantized,budget),full=compare(got[name],expected[name],budget);
    fullErrors[name]=full;
    output.rows.push({label:`${output.label} PM ${name} vs ${quantized?'f32-input Rust':'f64 Rust'} reference`,...error,fullF64MaximumAbsolute:full.maximumAbsolute,fullF64BudgetFraction:full.maximumBudgetFraction,tolerance:budget});
  }
  output.rows.push({label:`${output.label} PM state also satisfies full f64 Rust error budgets`,pass:Object.values(fullErrors).every((e)=>e.pass),components:fullErrors,tolerance:PM_BUDGET});
  const clocksMaxError=clockError(input.clocks,modeled);
  output.rows.push({label:`${output.label} Rust KDK clock packing`,pass:clocksMaxError<=5e-7,maximumF32ClockRounding:clocksMaxError,tolerance:'5e-7 absolute rounding bound'});
  const driftMins=got.driftMinima??[];
  const initialAbsolute=absolutePositionsFromDisplacement(input.x,r.dim,r.n),initialCpuMinimum=pm_drift_minimum(r.dim,r.n,initialAbsolute,initialAbsolute),initialFullMinimum=input.initialMinimum,initialGpuError=Math.abs(got.initialGpuMinimum-initialCpuMinimum),initialFullError=Math.abs(got.initialMinimum-initialFullMinimum),initialBudget=2e-4+2e-4*Math.abs(initialCpuMinimum);
  output.rows.push({label:`${output.label} initial GPU simplex minimum matches f64 checks`,pass:Number.isFinite(got.initialGpuMinimum)&&initialGpuError<=initialBudget&&initialFullError<=1e-12,gpuMinimum:got.initialGpuMinimum,quantizedCpuMinimum:initialCpuMinimum,conservativeRustMinimum:initialFullMinimum,usedMinimum:got.initialMinimum,absoluteError:initialGpuError,fullMetadataError:initialFullError,tolerance:initialBudget});
  const driftErrors=driftRows.map((row)=>({step:row.step,gpuMinimum:row.gpuMinimum,cpuMinimum:row.cpuMinimum,error:Math.abs(row.gpuMinimum-row.cpuMinimum),budget:2e-4+2e-4*Math.abs(row.cpuMinimum),gpuNonfinite:row.gpuNonfinite}));
  output.rows.push({label:`${output.label} GPU drift minima match the Rust f64 polynomial checker`,pass:driftMins.length===r.steps&&driftMins.every(Number.isFinite)&&driftRows.length===r.steps&&driftErrors.every((e)=>!e.gpuNonfinite&&Number.isFinite(e.cpuMinimum)&&e.error<=e.budget),steps:driftMins.length,cpuCheckedSteps:driftRows.filter(x=>x.checkedByCpu).length,maximumGpuCpuError:Math.max(0,...driftErrors.map(e=>e.error)),tolerance:'2e-4 absolute + 2e-4 relative',minimum:Math.min(got.initialMinimum??Infinity,...driftMins),driftErrors});
  if(compareFullRun){
    const cpu=pending.finish_cpu();
    const verifiedMinima=Float64Array.from(driftRows.map((row)=>row.minimum));
    let gpu;
    try{gpu=pending.finish_gpu_pm(got.x,got.p,got.dx,got.dp,verifiedMinima.length===r.steps?verifiedMinima:Float64Array.from(driftMins));
    const pmEvolution={backend:'gpu',precision:'f32 state, tangent, CIC and FFT; Rust clocks rounded to f32',diagnostics:got.diagnostics,timings:got.timings};
    const stages={trajectories:{backend:'mixed',lptBackend:'cpu',pmBackend:'gpu',lptStatePrecision:'f64',pmStatePrecision:'f32'},exactDeposit:{backend:'cpu',precision:'f64 geometry and mass accumulation',outputPrecision:'f32 density field'},pdfs:{backend:'cpu',precision:'f64 histogram and moment accumulation'},griddedSpectra:{backend:'cpu',precision:'f64'}};
    const execution={requestedBackend:'gpu',executedBackend:'hybrid',precision:'f32 GPU PM with CPU LPT/direct/finalization',pmEvolution,channelProvenance:{lpt:'cpu-f64',pm:'cpu-f64',finiteDifference:'cpu-f64'},stageBackends:stages};
    const cpuStages={...stages,trajectories:{backend:'cpu',lptBackend:'cpu',pmBackend:'cpu',lptStatePrecision:'f64',pmStatePrecision:'f64'}};
    const cpuResult=JSON.parse(cpu.finish(cpu.direct_cpu(),JSON.stringify({...execution,executedBackend:'cpu',pmEvolution:{backend:'cpu',precision:'f64'},stageBackends:cpuStages})));
    const gpuResult=JSON.parse(gpu.finish(gpu.direct_cpu(),JSON.stringify(execution)));
    const provenance=gpuResult.execution?.stageBackends;
    output.rows.push({label:`${output.label} full-run provenance preserves GPU PM and Rust deposit/PDF backends`,pass:gpuResult.execution?.pmEvolution?.backend==='gpu'&&provenance?.trajectories?.pmBackend==='gpu'&&provenance?.exactDeposit?.backend==='cpu'&&provenance?.pdfs?.backend==='cpu',pmEvolution:gpuResult.execution?.pmEvolution,stageBackends:provenance});
    for(const key of ['rho','pmRho']){
      const err=compare(gpuResult[key],cpuResult[key],{absolute:2e-4,relative:2e-3});
      output.rows.push({label:`${output.label} Rust observation ${key} output remains within f32 trajectory budget`,...err,tolerance:{absolute:2e-4,relative:2e-3}});
    }
    for(const key of ['pdf.volume','pdf.mass','pdf.pmVolume','pdf.pmMass']){
      const a=key.split('.').reduce((x,k)=>x?.[k],gpuResult),b=key.split('.').reduce((x,k)=>x?.[k],cpuResult),err=compare(a,b,{absolute:2e-4,relative:3e-3});
      output.rows.push({label:`${output.label} Rust ${key} uses conserved exact-deposit semantics`,...err,tolerance:{absolute:2e-4,relative:3e-3}});
    }
    const dy=3.6/80,integratedMass=gpuResult.pdf.mass.reduce((s,v)=>s+v*dy,0),integratedPmMass=gpuResult.pdf.pmMass.reduce((s,v)=>s+v*dy,0);
    output.rows.push({label:`${output.label} mass and volume PDFs preserve the Rust captured-probability accounting`,pass:Math.abs(integratedMass-gpuResult.pdf.capturedMass)<=1e-12&&Math.abs(integratedPmMass-gpuResult.pdf.pmCapturedMass)<=1e-12&&Math.abs(gpuResult.diagnostics.pmMass-1)<=1e-6,integratedMass,capturedMass:gpuResult.pdf.capturedMass,integratedPmMass,capturedPmMass:gpuResult.pdf.pmCapturedMass,pmSheetMass:gpuResult.diagnostics.pmMass});
    output.rows.push({label:`${output.label} CPU and GPU finalizers retain the same LPT PDF and spectrum semantics`,pass:JSON.stringify(gpuResult.rho)===JSON.stringify(cpuResult.rho)&&JSON.stringify(gpuResult.pdf.volume)===JSON.stringify(cpuResult.pdf.volume)&&JSON.stringify(gpuResult.spectra.sheet)===JSON.stringify(cpuResult.spectra.sheet),fields:['rho','pdf.volume','spectra.sheet']});
    const cpuCrossed=Boolean(cpuResult.guard?.pmCrossed),gpuCrossed=Boolean(gpuResult.guard?.pmCrossed),cpuStep=cpuResult.guard?.firstPmCrossing?.step??null,gpuStep=gpuResult.guard?.firstPmCrossing?.step??null;
    output.rows.push({label:`${output.label} full-run shell-crossing continuation matches Rust status`,pass:cpuCrossed===gpuCrossed&&(!cpuCrossed||Math.abs(cpuStep-gpuStep)<=1),cpuCrossed,gpuCrossed,cpuFirstStep:cpuStep,gpuFirstStep:gpuStep});
    }finally{cpu.free?.();gpu?.free?.();}
  }
  return output;
  }finally{pending.free?.();}
}

function fixturePacked(n,dim,{translation=[0,0,0],tangent=[0,0,0],velocity=[0,0,0],momentumTangent=[0,0,0],singleMode=false}={}){
  const nv=n**dim,x=new Float32Array(4*nv),p=new Float32Array(4*nv),dx=new Float32Array(4*nv),dp=new Float32Array(4*nv),dxcell=1/n;
  for(let i=0;i<nv;i++){
    let k=i,c=new Array(dim);for(let a=dim-1;a>=0;a--){c[a]=k%n;k=Math.floor(k/n);}
    const phase=2*Math.PI*c[0]/n;
    for(let a=0;a<dim;a++){
      const off=4*i+a,base=c[a]/n+translation[a];
      x[off]=singleMode&&a===0?base+.17*dxcell+.07*dxcell*Math.sin(phase):base+(singleMode ? .29*dxcell : 0);
      dx[off]=singleMode&&a===0?.03*dxcell*Math.cos(phase):tangent[a];
      p[off]=velocity[a];dp[off]=momentumTangent[a];
    }
  }
  return {x,p,dx,dp};
}

// Independent f64 matched CIC / spectral force and directional derivative.
// The FFT is a direct O(N^2) DFT, small enough for the n=16 Poisson fixture.
function spectralForceReference({n,dim,boxSize,omegaM,x,dx}){
  const size=n**dim,np=x.length/4,rho=new Float64Array(size),drho=new Float64Array(size),strides=Array(dim);
  for(let a=0;a<dim;a++)strides[a]=n**(dim-1-a);
  const nodes=(pos,dir)=>{
    const base=[],frac=[];for(let a=0;a<dim;a++){const u=((pos[a]/boxSize)%1+1)%1*n;base[a]=Math.floor(u)%n;frac[a]=u-Math.floor(u);}
    const out=[];for(let bits=0;bits<(1<<dim);bits++){let flat=0,w=1,dw=0,axisWeights=[];for(let a=0;a<dim;a++){const upper=(bits>>a)&1,wa=upper?frac[a]:1-frac[a],sign=upper?1:-1;axisWeights.push(wa);flat+=((base[a]+upper)%n)*strides[a];w*=wa;}
      if(dir)for(let a=0;a<dim;a++){let term=((bits>>a)&1?1:-1)*dir[a]/boxSize*n;for(let b=0;b<dim;b++)if(b!==a)term*=axisWeights[b];dw+=term;}out.push([flat,w,dw]);}return out;
  };
  for(let i=0;i<np;i++){const pos=[],dir=[];for(let a=0;a<dim;a++){pos.push(x[4*i+a]);dir.push(dx?.[4*i+a]??0);}for(const [node,w,dw] of nodes(pos,dir)){rho[node]+=size/np*w;drho[node]+=size/np*dw;}}
  const coordinates=(flat)=>strides.map((s)=>Math.floor(flat/s)%n),forward=(field)=>{
    const re=new Float64Array(size),im=new Float64Array(size);
    for(let q=0;q<size;q++){const qc=coordinates(q);for(let j=0;j<size;j++){const jc=coordinates(j),phase=-2*Math.PI*qc.reduce((v,c,a)=>v+c*jc[a]/n,0),z=field[j];re[q]+=z*Math.cos(phase);im[q]+=z*Math.sin(phase);}}return [re,im];
  };
  const inverseForce=(hat,a)=>{const out=new Float64Array(size);for(let j=0;j<size;j++){const jc=coordinates(j);let sum=0;for(let q=0;q<size;q++){const qc=coordinates(q),mode=qc.map(v=>v>=n/2?v-n:v);let k2=0;for(const m of mode){const k=2*Math.PI*m/boxSize;k2+=k*k;}const qa=mode[a],kd=n%2===0&&qc[a]===n/2?0:2*Math.PI*qa/boxSize;if(!k2||!kd)continue;const coeff=1.5*omegaM*kd/k2,phase=2*Math.PI*qc.reduce((v,c,b)=>v+c*jc[b]/n,0),hr=hat[0][q],hi=hat[1][q];sum+=coeff*(-hr*Math.sin(phase)-hi*Math.cos(phase));}out[j]=sum/size;}return out;};
  const rh=forward(rho),drh=forward(drho),force=[],dforce=[];for(let a=0;a<dim;a++){force.push(inverseForce(rh,a));dforce.push(inverseForce(drh,a));}
  const f=new Float64Array(x.length),df=new Float64Array(x.length);
  for(let i=0;i<np;i++){const pos=[],dir=[];for(let a=0;a<dim;a++){pos.push(x[4*i+a]);dir.push(dx?.[4*i+a]??0);}for(const [node,w,dw] of nodes(pos,dir))for(let a=0;a<dim;a++){f[4*i+a]+=w*force[a][node];df[4*i+a]+=w*dforce[a][node]+dw*force[a][node];}}
  return {force:f,jvp:df,rho,drho};
}

async function isolatedChecks(device,{signal,onProgress}={}){
  const rows=[];const n=16,dim=2,meshN=16,fixture=fixturePacked(n,dim,{singleMode:true});
  const checkDrift=(start,end)=>pm_drift_minimum(dim,n,start,end);
  const reference=spectralForceReference({n:meshN,dim,boxSize:1,omegaM:.315,x:fixture.x,dx:fixture.dx});
  const kick=await runPm(device,{dim,n,meshN,boxSize:1,omegaM:.315,...fixture,initialMinimum:checkDrift(fixture.x,fixture.x),clocks:{kick0:new Float64Array([1]),drift:new Float64Array([0]),kick1:new Float64Array([0])}},{signal,allowCrossing:true,checkDrift});
  const forceBudget={absolute:1e-6,relative:1e-4},jvpBudget={absolute:1e-6,relative:1e-4};
  const forceErr=compare(kick.p,reference.force,forceBudget),jvpErr=compare(kick.dp,reference.jvp,jvpBudget);
  rows.push({label:'2D n=16 single-mode displacement Poisson force and CIC JVP',...forceErr,forceTolerance:forceBudget,referenceInputPrecision:'f32 packed fixture evaluated by independent f64 direct DFT/CIC',jvp:{...jvpErr,tolerance:jvpBudget},pass:forceErr.pass&&jvpErr.pass});
  const eps=1e-4,xplus=Float64Array.from(fixture.x),xminus=Float64Array.from(fixture.x);for(let i=0;i<xplus.length;i++){xplus[i]+=eps*fixture.dx[i];xminus[i]-=eps*fixture.dx[i];}
  const fp=spectralForceReference({n:meshN,dim,boxSize:1,omegaM:.315,x:xplus}),fm=spectralForceReference({n:meshN,dim,boxSize:1,omegaM:.315,x:xminus});
  const tangentFD=new Float64Array(fixture.dx.length);for(let i=0;i<tangentFD.length;i++)tangentFD[i]=(fp.force[i]-fm.force[i])/(2*eps);
  const fdErr=compare(reference.jvp,tangentFD,{absolute:2e-7,relative:2e-4});rows.push({label:'Single-mode CPU CIC force JVP agrees with centered finite difference away from cell boundaries',...fdErr,tolerance:{absolute:2e-7,relative:2e-4}});
  onProgress?.({stage:'Resident PM isolated checks'});
  const uniform=fixturePacked(n,dim,{translation:[.23,.31],velocity:[.07,-.04],tangent:[.02,-.03],momentumTangent:[.005,.006]});
  const identity=await runPm(device,{dim,n,meshN,boxSize:1,omegaM:.315,...uniform,initialMinimum:checkDrift(uniform.x,uniform.x),clocks:{kick0:new Float64Array([.2,.3]),drift:new Float64Array([.4,.6]),kick1:new Float64Array([.1,.2])}},{signal,allowCrossing:true,checkDrift});
  const expectedX=uniform.x.slice(),expectedP=uniform.p.slice(),expectedDx=uniform.dx.slice(),expectedDp=uniform.dp.slice();const dt=.4+.6;
  for(let i=0;i<expectedX.length;i++){expectedX[i]+=dt*uniform.p[i];expectedDx[i]+=dt*uniform.dp[i];}
  for(const [name,got,ref,tol] of [['p',identity.p,expectedP,2e-5],['dp',identity.dp,expectedDp,2e-5],['x',identity.x,expectedX,3e-5],['dx',identity.dx,expectedDx,3e-5]]){const e=compare(got,ref,{absolute:tol,relative:tol});rows.push({label:`Uniform periodic density preserves ${name} through KDK kicks/drifts`,...e,tolerance:tol});}
  const wrapped=fixturePacked(n,dim,{singleMode:true}),shifted=wrapped.x.slice();for(let i=0;i<shifted.length;i+=4){if(i%3===0)shifted[i]+=1; if(dim>1&&i%5===0)shifted[i+1]-=2;}
  const wrapBase=await runPm(device,{dim,n,meshN,boxSize:1,omegaM:.315,...wrapped,initialMinimum:checkDrift(wrapped.x,wrapped.x),clocks:{kick0:new Float64Array([1]),drift:new Float64Array([0]),kick1:new Float64Array([0])}},{signal,allowCrossing:true,checkDrift});
  const wrapShift=await runPm(device,{dim,n,meshN,boxSize:1,omegaM:.315,...wrapped,x:shifted,initialMinimum:checkDrift(shifted,shifted),clocks:{kick0:new Float64Array([1]),drift:new Float64Array([0]),kick1:new Float64Array([0])}},{signal,allowCrossing:true,checkDrift});
  const wrapBudget={absolute:1e-6,relative:1e-4},wrapErr=compare(wrapShift.p,wrapBase.p,wrapBudget);rows.push({label:'Periodic particle wrapping leaves CIC force invariant for unwrapped coordinates',...wrapErr,tolerance:wrapBudget,referenceInputPrecision:'both unwrapped variants are independently f32-quantized'});
  const crossing=fixturePacked(16,2);for(let i=0;i<crossing.x.length;i+=4)if(i===4)crossing.x[i]=-.8;
  let checked=0;const crossed=await runPm(device,{dim:2,n:16,meshN:16,boxSize:1,omegaM:.315,...crossing,initialMinimum:pm_drift_minimum(2,16,crossing.x,crossing.x),clocks:{kick0:new Float64Array([0,0]),drift:new Float64Array([.001,.001]),kick1:new Float64Array([0,0])}},{signal,allowCrossing:true,checkAllDrifts:true,checkDrift:(a,b)=>pm_drift_minimum(2,16,a,b),onDrift:()=>checked++});
  const crossedMinimum=Math.min(crossed.initialMinimum??Infinity,...(crossed.driftMinima??[]));finiteArray(crossed.x,'crossed x');
  rows.push({label:'Initially crossed periodic sheet continues through PM steps and records crossing',pass:crossedMinimum<=0&&crossed.driftMinima?.length===2&&checked===2,minimumJacobian:crossedMinimum,completedSteps:crossed.driftMinima?.length,checkedDrifts:checked});
  return rows;
}

async function failureAndResourceChecks(device){
  const rows=[];
  try{await createGpuContext({navigator:{}});rows.push({label:'Missing WebGPU API is reported as unavailable',pass:false});}
  catch(e){rows.push({label:'Missing WebGPU API is reported as unavailable',pass:/WebGPU is unavailable/.test(String(e)),reason:String(e)});}
  try{new GpuFFT({limits:{maxStorageBufferBindingSize:8,maxBufferSize:8}},16,3);rows.push({label:'FFT resource limit is rejected before allocation',pass:false});}
  catch(e){rows.push({label:'FFT resource limit is rejected before allocation',pass:/exceeds device storage-buffer limits/.test(String(e))||/exceeds/.test(String(e)),reason:String(e)});}
  const resourceFixture=fixturePacked(16,2);
  try{await runPm({limits:{maxStorageBufferBindingSize:8,maxBufferSize:8}}, {dim:2,n:16,meshN:16,boxSize:1,omegaM:.315,...resourceFixture,clocks:{kick0:new Float32Array([1]),drift:new Float32Array([0]),kick1:new Float32Array([0])}});rows.push({label:'PM resource limit is rejected before GPU allocation',pass:false});}
  catch(e){rows.push({label:'PM resource limit is rejected before GPU allocation',pass:/exceeds device limits/.test(String(e)),reason:String(e)});}
  const ac=new AbortController();ac.abort();const zero=fixturePacked(16,2);try{await runPm(device,{dim:2,n:16,meshN:16,boxSize:1,omegaM:.315,...zero,clocks:{kick0:new Float64Array([0]),drift:new Float64Array([0]),kick1:new Float64Array([0])}},{signal:ac.signal});rows.push({label:'Aborted PM request stops before evolution',pass:false});}
  catch(e){rows.push({label:'Aborted PM request stops before evolution',pass:e?.name==='AbortError',reason:String(e)});}
  return rows;
}

async function timePmReference(r){
  const prepStart=performance.now();let pending;
  try{
    pending=new PendingLabRun(JSON.stringify(r));const inputReady=pmInput(r,pending);const preparationMs=performance.now()-prepStart;
    const evolutionStart=performance.now(),reference=pending.pm_reference(),pmMs=performance.now()-evolutionStart;
    return {backend:'cpu-f64-reference',request:{...r},preparationMs,pmMs,totalMs:preparationMs+pmMs,packedComponents:reference.length,clockSteps:inputReady.clocks.drift.length,pass:true,_state:splitReference(reference,r.dim)};
  }catch(error){return {backend:'cpu-f64-reference',request:{...r},pass:false,error:String(error?.message||error)};}
  finally{pending?.free?.();}
}

async function timePmGpu(device,r,{signal}={}){
  const prepStart=performance.now();let pending;
  try{
    pending=new PendingLabRun(JSON.stringify(r));const input=pmInput(r,pending),preparationMs=performance.now()-prepStart;
    const runStart=performance.now(),result=await runPm(device,input,{signal,allowCrossing:true,checkDrift:(a,b)=>pm_drift_minimum(r.dim,r.n,a,b)}),pmMs=performance.now()-runStart;
    return {backend:'gpu-f32-resident',request:{...r},preparationMs,pmMs,totalMs:preparationMs+pmMs,clockSteps:input.clocks.drift.length,timings:result.timings,diagnostics:result.diagnostics,pass:true,_state:{x:result.x,p:result.p,dx:result.dx,dp:result.dp}};
  }catch(error){return {backend:'gpu-f32-resident',request:{...r},pass:false,error:String(error?.message||error)};}
  finally{pending?.free?.();}
}

function startWorker(){
  if(typeof Worker!=='function')throw new Error('Module workers are unavailable in this browser');
  return new Worker(new URL('../worker.js',import.meta.url),{type:'module',name:'resident-harness-worker'});
}
let nextHarnessRequestId=1;
function postWorkerRun(worker,state,policy,{signal,onPhase=()=>{},timeoutMs=600000}={}){
  if(aborted(signal))return Promise.reject(new DOMException('Resident worker check canceled','AbortError'));
  const requestId=nextHarnessRequestId++;
  const started=performance.now();
  return new Promise((resolve,reject)=>{
    let timer;
    const cleanup=()=>{clearTimeout(timer);signal?.removeEventListener('abort',abort);worker.removeEventListener('message',message);worker.removeEventListener('error',error);};
    const abort=()=>worker.postMessage({type:'cancel',requestId});
    const error=(event)=>{cleanup();reject(new Error(event.message||'Resident worker failed'));};
    const message=({data})=>{
      if(data?.requestId!==requestId)return;
      if(data.type==='phase'||data.type==='progress'){onPhase(data);return;}
      if(data.type==='cancelled'){cleanup();resolve({type:'cancelled',requestId,data,wallMilliseconds:performance.now()-started});return;}
      if(data.type==='result'){cleanup();resolve({type:'result',requestId,data,wallMilliseconds:performance.now()-started});}
    };
    worker.addEventListener('message',message);worker.addEventListener('error',error);
    timer=setTimeout(()=>{cleanup();reject(new Error(`Worker request ${requestId} timed out after ${timeoutMs} ms`));},timeoutMs);
    if(signal){if(signal.aborted)abort();else signal.addEventListener('abort',abort,{once:true});}
    worker.postMessage({type:'run',requestId,key:`resident-check-${requestId}`,state,policy});
  });
}

async function shutdownWorker(worker,timeoutMs=5000){
  if(!worker)return false;const ack=await new Promise((resolve)=>{let timer;const done=(ok)=>{clearTimeout(timer);worker.removeEventListener('message',message);resolve(ok);},message=({data})=>{if(data?.type==='shutdown-ack')done(true);};worker.addEventListener('message',message);timer=setTimeout(()=>done(false),timeoutMs);worker.postMessage({type:'shutdown'});});worker.terminate();return ack;
}

function workerOutput(data){return data?.result??null;}
function workerExecution(data){return data?.execution??data?.result?.execution??null;}
function workerTiming(data){return {durationMilliseconds:data?.durationMilliseconds??null,stages:data?.execution?.timings?.stages??null,pm:data?.execution?.pmEvolution?.timings??null};}
function compareWorkerPmOutput(actual,reference){
  const rows=[];
  for(const key of ['pmRho','pdf.pmMass','pdf.pmVolume']){
    const a=key.split('.').reduce((x,k)=>x?.[k],actual),b=key.split('.').reduce((x,k)=>x?.[k],reference);
    rows.push({field:key,...compare(a,b,{absolute:2e-6,relative:2e-6})});
  }
  return {pass:rows.every(r=>r.pass),rows};
}

async function workerFallbackChecks({signal,onProgress=()=>{}}={}){
  const rows=[],worker=startWorker();
  try{
    const small=request({dim:3,n:16,forceMesh:32,mesh:16,steps:32,scaleFactor:.25});
    const cpuReply=await postWorkerRun(worker,small,{requested:'cpu'},{signal});
    const cpuResult=workerOutput(cpuReply.data);
    const forced=await postWorkerRun(worker,small,{requested:'gpu',forcePmFailure:'limits'},{signal,onPhase:onProgress});
    const forcedResult=workerOutput(forced.data),forcedExec=workerExecution(forced.data),forcedPm=forcedExec?.pmEvolution??{};
    const physicalCompare=compareWorkerPmOutput(forcedResult,cpuResult);
    rows.push({label:'Worker recovers a PM allocation-limit failure with exact CPU PM',pass:cpuReply.type==='result'&&forced.type==='result'&&forcedPm.backend==='cpu'&&/maxGpuBytes/i.test(forcedPm.fallbackReason??'')&&forcedExec?.stageBackends?.trajectories?.pmBackend==='cpu'&&Boolean(forcedPm.gpuAttempt)&&physicalCompare.pass,backend:forcedPm.backend,fallbackReason:forcedPm.fallbackReason,gpuAttempt:forcedPm.gpuAttempt?{timings:forcedPm.gpuAttempt.timings,diagnostics:forcedPm.gpuAttempt.diagnostics}:null,request:{...small},physicalOutput:physicalCompare});

    const reacquired=await postWorkerRun(worker,small,{requested:'gpu'},{signal,onPhase:onProgress});
    const reacquiredExec=workerExecution(reacquired.data),reacquiredPm=reacquiredExec?.pmEvolution??{};
    rows.push({label:'Worker reacquires a usable GPU device after PM failure recovery',pass:reacquired.type==='result'&&reacquiredPm.backend==='gpu'&&reacquiredExec?.stageBackends?.trajectories?.pmBackend==='gpu',pmBackend:reacquiredPm.backend,fallbackReason:reacquiredPm.fallbackReason??null,request:{...small}});

    const unsupported=request({dim:3,n:16,forceMesh:24,mesh:16,steps:32,scaleFactor:.25});
    const unsupportedReply=await postWorkerRun(worker,unsupported,{requested:'gpu'},{signal}),unsupportedExec=workerExecution(unsupportedReply.data),unsupportedPm=unsupportedExec?.pmEvolution??{};
    rows.push({label:'Worker retains requested non-radix force mesh on CPU PM',pass:unsupportedReply.type==='result'&&unsupportedPm.backend==='cpu'&&/power-of-two/i.test(unsupportedPm.fallbackReason??'')&&unsupportedExec?.stageBackends?.trajectories?.pmBackend==='cpu',request:{...unsupported},pmBackend:unsupportedPm.backend,fallbackReason:unsupportedPm.fallbackReason});

    const cancelWorker=startWorker();let shutdownAck=false;
    try{
      const warm=await postWorkerRun(cancelWorker,small,{requested:'gpu'},{signal,onPhase:onProgress});
      const warmExec=workerExecution(warm.data),warmPm=warmExec?.pmEvolution??{};
      const warmGpu=warm.type==='result'&&warmPm.backend==='gpu'&&warmExec?.stageBackends?.trajectories?.pmBackend==='gpu';
      if(!warmGpu){
        rows.push({label:'Worker cancellation fixture warms a real GPU PM path before testing cancellation',pass:false,workerStatus:warm.type,pmBackend:warmPm.backend??null,error:warm.data?.error??null,request:{...small}});
        rows.push({label:'Worker cancellation is triggered after an actual GPU PM evolution step',pass:false,reason:'GPU PM warmup did not complete',request:{...small}});
      }else{
        rows.push({label:'Worker cancellation fixture warms a real GPU PM path before testing cancellation',pass:true,pmBackend:warmPm.backend,request:{...small}});
        let stepSeen=false,cancelSent=false,progressStep=null;
        const queuedId=nextHarnessRequestId++,activeId=nextHarnessRequestId++;
        const replacement=request({dim:2,n:16,forceMesh:16,mesh:16,steps:32,scaleFactor:.25});
        const canceledAndReplaced=new Promise((resolve,reject)=>{
          let timer;let canceled=false,replaced=null;
          const cleanup=()=>{clearTimeout(timer);cancelWorker.removeEventListener('message',message);cancelWorker.removeEventListener('error',error);};
          const error=(event)=>{cleanup();reject(new Error(event.message||'Cancellation worker failed'));};
          const message=({data})=>{
            if(data?.type==='progress'&&data.requestId===activeId&&data.phase==='pm-evolution'&&Number(data.step)>=1&&!cancelSent){
              stepSeen=true;progressStep=data.step;cancelSent=true;
              cancelWorker.postMessage({type:'cancel',requestId:activeId});
              cancelWorker.postMessage({type:'run',requestId:queuedId,key:'resident-queued-replacement',state:replacement,policy:{requested:'cpu'}});
            }
            if(data?.type==='cancelled'&&data.requestId===activeId)canceled=true;
            if(data?.type==='result'&&data.requestId===queuedId)replaced=data;
            if(canceled&&replaced){cleanup();resolve({stepSeen,progressStep,canceled,replaced});}
          };
          cancelWorker.addEventListener('message',message);cancelWorker.addEventListener('error',error);timer=setTimeout(()=>{cleanup();reject(new Error('Worker cancel/replacement check timed out'));},600000);
          cancelWorker.postMessage({type:'run',requestId:activeId,key:'resident-cancel-active',state:small,policy:{requested:'gpu'}});
        });
        const replacementResult=await canceledAndReplaced,replacementExec=workerExecution(replacementResult.replaced);
        rows.push({label:'Worker cancellation is triggered after an actual GPU PM evolution step',pass:replacementResult.stepSeen&&replacementResult.canceled&&replacementResult.replaced?.type==='result'&&!replacementResult.replaced?.error&&replacementExec?.pmEvolution?.backend==='cpu',stepSeen:replacementResult.stepSeen,progressStep:replacementResult.progressStep,canceled:replacementResult.canceled,replacementBackend:replacementExec?.pmEvolution?.backend,queuedRequest:{...replacement}});
      }
    }finally{shutdownAck=await shutdownWorker(cancelWorker);}
    rows.push({label:'Worker cancellation fixture shuts down cleanly',pass:shutdownAck,shutdownAck});
  }catch(error){rows.push({label:'Worker PM fallback and cancellation checks complete',pass:false,error:String(error?.stack||error)});}
  finally{await shutdownWorker(worker);}
  return rows;
}

/** Run just the worker fallback, recovery, cancellation, and shutdown checks. */
export async function runResidentWorkerChecks({signal,onProgress=()=>{}}={}){
  try{
    const rows=await workerFallbackChecks({signal,onProgress});
    const failures=rows.filter((row)=>row.pass===false||row.failedComponents>0);
    return {status:signal?.aborted?'skip':failures.length?'fail':'pass',rows,failures,timings:[],precision:'worker lifecycle checks; GPU PM recovery compared with Rust CPU PM'};
  }catch(error){
    return {status:error?.name==='AbortError'?'skip':'fail',reason:String(error?.stack||error),rows:[],failures:[],timings:[],precision:'worker lifecycle checks; GPU PM recovery compared with Rust CPU PM'};
  }
}

function summarizeWorkerSample(reply,state,policy,sampleKind,repeat){
  const exec=workerExecution(reply.data),pm=exec?.pmEvolution??{};
  return {sampleKind,repeat,request:{...state},policy:{...policy},workerStatus:reply.type,error:reply.data?.error??null,
    wallMilliseconds:reply.wallMilliseconds,
    requestedBackend:exec?.requestedBackend??null,executedBackend:exec?.executedBackend??null,pmBackend:pm.backend??null,
    fallbackReason:pm.fallbackReason??null,timing:workerTiming(reply.data),
    provenance:{trajectories:exec?.stageBackends?.trajectories,exactDeposit:exec?.stageBackends?.exactDeposit,pdfs:exec?.stageBackends?.pdfs}};
}

async function runResidentBenchmarks(device,{signal,onProgress=()=>{}}={}){
  const timings=[],rows=[];
  for(const n of [32,64]){
    const r=request({dim:3,n,forceMesh:n,mesh:n,steps:32,omegaM:.315,scaleFactor:.25});
    onProgress({stage:`PM benchmark warmup ${n}³`});
    const warm=await timePmGpu(device,r,{signal});delete warm._state;timings.push({...warm,sampleKind:'gpu-warmup',repeat:0});
    rows.push({label:`Matched ${n}³ PM benchmark GPU warmup`,pass:warm.pass,error:warm.error??null});
    for(let repeat=1;repeat<=3;repeat++){
      if(aborted(signal))throw new DOMException('Resident benchmark canceled','AbortError');
      const order=repeat%2===1?['cpu','gpu']:['gpu','cpu'];
      const paired={};
      for(const backend of order){
        onProgress({stage:`${backend==='cpu'?'CPU':'GPU'} ${n}³ PM benchmark`,caseIndex:repeat,cases:3});
        const sample=backend==='cpu'?await timePmReference(r):await timePmGpu(device,r,{signal});
        paired[backend]=sample._state;delete sample._state;
        timings.push({...sample,sampleKind:'matched-pm',repeat});
      }
      const errors=paired.gpu&&paired.cpu?Object.fromEntries(['x','p','dx','dp'].map((name)=>[name,compare(paired.gpu[name],paired.cpu[name],PM_BUDGET[name])])):null;
      const match={label:`Matched ${n}³ PM GPU state agrees with uncached CPU reference, repeat ${repeat}`,pass:Boolean(errors)&&Object.values(errors).every(e=>e.pass),request:{...r},components:errors,tolerance:PM_BUDGET};rows.push(match);
      for(const sample of timings.filter(s=>s.sampleKind==='matched-pm'&&s.request.n===n&&s.repeat===repeat))sample.comparisonToCpuReference={pass:match.pass,components:errors};
    }
    const matched=timings.filter(s=>s.sampleKind==='matched-pm'&&s.request.n===n);
    const stateMatches=rows.filter(row=>row.request?.n===n&&/agrees with uncached CPU reference/.test(row.label));
    rows.push({label:`Matched ${n}³ CPU uncached and warm GPU PM timings`,pass:matched.length===6&&matched.every(s=>s.pass)&&stateMatches.length===3&&stateMatches.every(row=>row.pass),meshN:n,steps:32,repeats:3,cpuPmMs:matched.filter(s=>s.backend==='cpu-f64-reference').map(s=>s.pmMs),gpuPmMs:matched.filter(s=>s.backend==='gpu-f32-resident').map(s=>s.pmMs),gpuKernelMs:matched.filter(s=>s.backend==='gpu-f32-resident').map(s=>s.timings?.kernelMs),request:{...r},timingSource:'new PendingLabRun for every sample; CPU pm_reference is uncached; GPU runs after one excluded warmup; setup/evolution separated'});
  }

  const worker=startWorker();
  try{
    const state=request({dim:3,n:32,forceMesh:32,mesh:32,steps:32,omegaM:.315,scaleFactor:.25});
    // Warm both paths before recording the alternating end-to-end samples.
    onProgress({stage:'Worker observer-only end-to-end warmup'});
    const observerWarm=await postWorkerRun(worker,state,{requested:'gpu',disablePm:true},{signal});
    timings.push(summarizeWorkerSample(observerWarm,state,{requested:'gpu',disablePm:true},'worker-warmup',0));
    onProgress({stage:'Worker resident-PM end-to-end warmup'});
    const residentWarm=await postWorkerRun(worker,state,{requested:'gpu'},{signal});
    timings.push(summarizeWorkerSample(residentWarm,state,{requested:'gpu'},'worker-warmup',0));
    rows.push({label:'Worker warmups execute observer-only CPU PM and resident GPU PM',pass:workerExecution(observerWarm.data)?.pmEvolution?.backend==='cpu'&&workerExecution(residentWarm.data)?.pmEvolution?.backend==='gpu',observerPmBackend:workerExecution(observerWarm.data)?.pmEvolution?.backend,residentPmBackend:workerExecution(residentWarm.data)?.pmEvolution?.backend,request:{...state}});
    const samples=[];
    for(let repeat=1;repeat<=3;repeat++){
      if(aborted(signal))throw new DOMException('Resident benchmark canceled','AbortError');
      const order=repeat%2===1?['observer','resident']:['resident','observer'];
      const pair={repeat,request:{...state},order,results:{}};
      for(const mode of order){
        const policy=mode==='observer'?{requested:'gpu',disablePm:true}:{requested:'gpu'};
        onProgress({stage:`Worker ${mode==='observer'?'observer-only':'resident'} 32³ end-to-end`,caseIndex:repeat,cases:3});
        const reply=await postWorkerRun(worker,state,policy,{signal});
        const sample=summarizeWorkerSample(reply,state,policy,'worker-e2e',repeat);timings.push(sample);samples.push(sample);pair.results[mode]=sample;
      }
      pair.pass=pair.results.observer?.workerStatus==='result'&&pair.results.observer?.pmBackend==='cpu'&&pair.results.resident?.workerStatus==='result'&&pair.results.resident?.pmBackend==='gpu';
      rows.push({label:`Worker end-to-end pair ${repeat} retains requested PM backends`,pass:pair.pass,request:{...state},order,observer:pair.results.observer,resident:pair.results.resident});
    }
    rows.push({label:'Alternating observer-only and resident PM worker timings',pass:samples.length===6&&samples.every(s=>s.workerStatus==='result'&&!s.error),repeats:3,request:{...state},samples,source:'same persistent module worker; each request includes preparation, direct modes, PM evolution, finite-difference validation and finalization'});
  }finally{await shutdownWorker(worker);}
  return {rows,timings};
}

/** Fast per-device gate. Run once before selecting resident PM for a worker. */
export async function runPmSelfTests({device,signal,onProgress=()=>{}}={}){
  if(!device)return {status:'skip',rows:[],reason:'No WebGPU device was provided'};
  await init();const rows=[];
  try{
    rows.push(...await fftChecks(device));
    rows.push(...await isolatedChecks(device,{signal,onProgress}));
    const cases=[request({dim:2,n:16,forceMesh:16,mesh:16,omegaM:1}),request({dim:3,n:16,forceMesh:16,mesh:16,omegaM:.315})];
    for(let i=0;i<cases.length;i++){
      if(aborted(signal))throw new DOMException('Resident PM self-test canceled','AbortError');
      onProgress({stage:'Resident PM reference check',caseIndex:i+1,cases:cases.length});
      const result=await pmFixture(device,cases[i],{signal,onProgress,compareFullRun:false});rows.push(...result.rows);
    }
    rows.push(...await failureAndResourceChecks(device));
    const failures=rows.filter(r=>r.pass===false||r.failedComponents>0);
    return {status:failures.length?'fail':'pass',rows,failures,precision:'f32 GPU FFT/PM; Rust f64 and f32-input f64 references',budget:PM_BUDGET};
  }catch(error){return {status:error?.name==='AbortError'?'skip':'fail',reason:String(error.stack||error),rows,precision:'f32 GPU FFT/PM; Rust f64 and f32-input f64 references',budget:PM_BUDGET};}
}

/** Complete resident benchmark/validation page suite. */
export async function runResidentChecks({device,signal,onProgress=()=>{},benchmark=false}={}){
  if(!device)return {status:'skip',rows:[],reason:'No WebGPU device was provided'};
  await init();const rows=[],fixtures=[];let timings=[];
  try{
    rows.push(...await fftChecks(device));
    rows.push(...await isolatedChecks(device,{signal,onProgress}));
    rows.push(...await failureAndResourceChecks(device));
    const cases=[
      request({dim:2,n:24,forceMesh:32,mesh:24,omegaM:.315,scaleFactor:.25}),
      request({dim:3,n:16,forceMesh:32,mesh:16,omegaM:1,scaleFactor:.25}),
      request({dim:3,n:16,forceMesh:32,mesh:16,omegaM:.315,scaleFactor:.25}),
      request({dim:3,n:32,forceMesh:32,mesh:32,omegaM:.315,scaleFactor:.25}),
      request({dim:3,n:32,forceMesh:32,mesh:16,omegaM:.315,scaleFactor:.25,ic:'gaussian',initialCutoff:3,initialSmoothing:14}),
    ];
    for(let i=0;i<cases.length;i++){
      if(aborted(signal))throw new DOMException('Resident GPU checks canceled','AbortError');
      onProgress({stage:'Resident PM full fixture',caseIndex:i+1,cases:cases.length});
      const result=await pmFixture(device,cases[i],{signal,onProgress,checkAllDrifts:true,compareFullRun:true});fixtures.push(result);rows.push(...result.rows);await yieldTask();
    }
    // A long-range crossed run exercises the continuation diagnostic on a
    // physical prepared fixture. Its trajectory remains usable for deposits.
    const crossedRequest=request({dim:3,n:16,forceMesh:32,mesh:16,omegaM:.315,scaleFactor:1});
    const crossed=await pmFixture(device,crossedRequest,{signal,onProgress,checkAllDrifts:true,compareFullRun:true});
    const crossingRow=crossed.rows.find(r=>/full-run shell-crossing continuation/.test(r.label));
    rows.push(...crossed.rows);fixtures.push(crossed);
    rows.push({label:'Long-range LCDM continuation fixture actually crosses in both references',pass:Boolean(crossingRow?.cpuCrossed&&crossingRow?.gpuCrossed),cpuCrossed:crossingRow?.cpuCrossed??false,gpuCrossed:crossingRow?.gpuCrossed??false,request:{...crossedRequest}});
    rows.push(...await workerFallbackChecks({signal,onProgress}));
    if(benchmark){
      const benchmarkResult=await runResidentBenchmarks(device,{signal,onProgress});timings=benchmarkResult.timings;rows.push(...benchmarkResult.rows);
    }
    const failures=rows.filter(r=>r.pass===false||r.failedComponents>0);
    return {status:failures.length?'fail':'pass',rows,fixtures,timings,failures,precision:'GPU state/FFT f32; f64 Rust native and f32-input reference; exact deposit and PDFs stay Rust',budget:PM_BUDGET};
  }catch(error){return {status:error?.name==='AbortError'?'skip':'fail',reason:String(error.stack||error),rows,fixtures,timings,precision:'GPU state/FFT f32; Rust f64 references',budget:PM_BUDGET};}
}
