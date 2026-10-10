// Independent numerical checks for the direct fixed-mass P0 kernel.
import { runDirect } from './direct.js';
import init, {PreparedLabRun} from '../../pkg/hcc_core.js';

const TWO_PI=2*Math.PI;
const TETS=[[0,1,3,7],[0,1,5,7],[0,2,3,7],[0,2,6,7],[0,4,5,7],[0,4,6,7]];
const TRI=[[0,1,3],[0,3,2]];

function gauss01(order){
  const x=[],w=[];
  for(let i=0;i<order;i++){
    let z=Math.cos(Math.PI*(i+.75)/(order+.5)),pp=0;
    for(let it=0;it<50;it++){
      let p1=1,p2=0;for(let j=0;j<order;j++){const p3=p2;p2=p1;p1=((2*j+1)*z*p2-j*p3)/(j+1);}
      pp=order*(z*p1-p2)/(z*z-1);const dz=p1/pp;z-=dz;if(Math.abs(dz)<2e-15)break;
    }
    x.push((1-z)/2);w.push(1/((1-z*z)*pp*pp));
  }
  return {x,w};
}
function simplexQuadrature(dim,order){
  const {x,w}=gauss01(order),out=[];
  if(dim===2){for(let a=0;a<order;a++)for(let b=0;b<order;b++){const u=x[a],v=x[b],x1=u,x2=v*(1-u);out.push({lambda:[1-x1-x2,x1,x2],weight:w[a]*w[b]*(1-u)});}}
  else for(let a=0;a<order;a++)for(let b=0;b<order;b++)for(let c=0;c<order;c++){const u=x[a],v=x[b],t=x[c],x1=u,x2=v*(1-u),x3=t*(1-u)*(1-v);out.push({lambda:[1-x1-x2-x3,x1,x2,x3],weight:w[a]*w[b]*w[c]*(1-u)**2*(1-v)});}
  return out;
}
function gridVertices(dim,n,psi,dir){
  const get=(coord)=>{const wrapped=coord.map((x)=>x>=n?x-n:x),shift=coord.map((x)=>x>=n?1:0);const flat=dim===2?wrapped[0]*n+wrapped[1]:(wrapped[0]*n+wrapped[1])*n+wrapped[2];return {flat,shift};};
  const cells=[];
  for(let cell=0;cell<n**dim;cell++){
    let base;if(dim===2)base=[Math.floor(cell/n),cell%n,0];else base=[Math.floor(cell/(n*n)),Math.floor(cell/n)%n,cell%n];
    const corners=[];for(let b=0;b<(1<<dim);b++){const c=[base[0]+(b&1),base[1]+((b>>1)&1),dim===3?base[2]+((b>>2)&1):0];corners.push(get(c));}
    cells.push(corners);
  }
  return {cells,psi,dir};
}
function reference(dim,n,modes,psi,dir,order=14){
  const {cells}=gridVertices(dim,n,psi,dir),simplices=dim===2?TRI:TETS,quad=simplexQuadrature(dim,order),out=new Float64Array(modes.length/4*4),scale=1/(n**dim);
  for(let mi=0;mi<modes.length/4;mi++){
    const m=[modes[4*mi],modes[4*mi+1],modes[4*mi+2]],kr=TWO_PI*m[0],ki=TWO_PI*m[1],kj=TWO_PI*m[2];let fr=0,fi=0,jr=0,ji=0;
    for(const corners of cells)for(const simplex of simplices){
      const vertices=simplex.map((b)=>{const {flat,shift}=corners[b],p=[],v=[];for(let a=0;a<dim;a++){const q=[Math.floor(flat/(n**(dim-a-1)))%n];const qa=q[0]/n+shift[a]+psi[4*flat+a];p.push(qa);v.push(dir[4*flat+a]);}return {p,v};});
      for(const q of quad){let phase=0,velocity=0;for(let j=0;j<vertices.length;j++){const v=vertices[j];phase+=q.lambda[j]*(kr*v.p[0]+ki*v.p[1]+kj*(dim===3?v.p[2]:0));velocity+=q.lambda[j]*(kr*v.v[0]+ki*v.v[1]+kj*(dim===3?v.v[2]:0));}const c=Math.cos(phase),s=Math.sin(phase),weight=scale*q.weight;fr+=weight*c;fi-=weight*s;jr-=weight*velocity*s;ji-=weight*velocity*c;}
    }
    out.set([fr,fi,jr,ji],4*mi);
  }
  return out;
}
function makeFixture(dim,n,kind){
  const nv=n**dim,psi=new Float32Array(4*nv),dir=new Float32Array(4*nv);
  for(let flat=0;flat<nv;flat++){
    const coord=dim===2?[Math.floor(flat/n),flat%n,0]:[Math.floor(flat/(n*n)),Math.floor(flat/n)%n,flat%n];
    for(let a=0;a<dim;a++){
      const q=coord[a]/n;
      let p=kind==='fully-collapsed'?-q:kind==='orientation-reversed'?-2*q:kind==='collapsed-axis'&&a===0?-q:kind==='large-translation'?[4096.125,-8192.25,2048.5][a]:kind==='wide-spread'?2.7*Math.sin(2.17*flat+0.7*a):kind==='branch-threshold'?0.04*Math.sin(1.31*flat+0.4*a):0.025*Math.sin(1.31*flat+0.4*a);
      let v=(kind==='translation'||kind==='large-translation')?[0.13,-0.21,0.07][a]:0.09*Math.cos(0.63*flat+0.8*a);
      psi[4*flat+a]=p;dir[4*flat+a]=v;
    }
  }
  return {psi,dir};
}
function packModes(records){const out=new Int32Array(records.length*4);records.forEach((m,i)=>out.set([m[0],m[1],m[2],0],4*i));return out;}
function err(a,b){return Math.abs(a-b);}

export const GPU_ERROR_BUDGET=Object.freeze({
  status:'fixture acceptance target; device pass does not certify every laboratory request',
  primal:'componentwise absolute <= 1e-6 + relative <= 1e-4 against f64 quadrature on f32-quantized inputs',
  jvp:'componentwise absolute <= 1e-6 + relative <= 1e-4 against f64 quadrature on f32-quantized inputs',
  amplitudeDerivative:'full request must retain the existing relative error <= 1e-5 gate; this harness does not certify it',
});

/** Run small complete-grid GPU fixtures against independent Duffy quadrature. */
export async function runGpuSelfTests({device,onProgress,signal,run=runDirect,schedules=['chunk-mode','cell-tile']}={}){
  const rows=[],timings=[],numericalErrors=[];
  const limitNames=['maxStorageBuffersPerShaderStage','maxStorageBufferBindingSize','maxBufferSize','maxComputeInvocationsPerWorkgroup','maxComputeWorkgroupSizeX','maxComputeWorkgroupsPerDimension'];
  const deviceInfo={limits:Object.fromEntries(limitNames.map((name)=>[name,device?.limits?.[name]]).filter(([,v])=>typeof v==='number')),features:[...(device?.features??[])]};
  if(!device)return {status:'skip',rows,numericalErrors,timings,deviceInfo,budget:GPU_ERROR_BUDGET,reason:'No WebGPU device was provided'};
  const cases=[
    {dim:2,n:4,kind:'smooth'},
    {dim:2,n:3,kind:'smooth'},
    {dim:2,n:5,kind:'translation'},
    {dim:2,n:3,kind:'large-translation'},
    {dim:2,n:3,kind:'branch-threshold'},
    {dim:2,n:3,kind:'collapsed-axis'},
    {dim:2,n:4,kind:'fully-collapsed'},
    {dim:2,n:4,kind:'orientation-reversed'},
    {dim:3,n:4,kind:'smooth'},
    {dim:3,n:3,kind:'smooth'},
    {dim:3,n:3,kind:'translation'},
    {dim:3,n:3,kind:'large-translation'},
    {dim:3,n:3,kind:'branch-threshold'},
    {dim:3,n:3,kind:'collapsed-axis'},
    {dim:3,n:4,kind:'fully-collapsed'},
    {dim:3,n:4,kind:'orientation-reversed'},
    {dim:3,n:3,kind:'wide-spread'},
  ];
  const modeRecords=[[0,0,0],[1,0,0],[-1,0,0],[0,1,0],[1,-1,0],[0,0,1]];
  for(let ci=0;ci<cases.length;ci++){
    if(signal?.aborted)throw new DOMException('GPU self-test canceled','AbortError');
    const c=cases[ci], fixtureModes=modeRecords.map((m)=>c.dim===2?[m[0],m[1],0]:m),modes=packModes(fixtureModes), g=makeFixture(c.dim,c.n,c.kind);
    const input={dim:c.dim,n:c.n,modes,lptPsi:g.psi,lptDirection:g.dir,pmPsi:g.psi,pmDirection:g.dir};
    const ref=reference(c.dim,c.n,modes,g.psi,g.dir,16);
    for(const schedule of schedules){
    if(signal?.aborted)throw new DOMException('GPU self-test canceled','AbortError');
    const start=performance.now();const got=await run(device,input,{signal,schedule,onProgress:(p)=>onProgress?.({...p,caseIndex:ci+1,cases:cases.length,schedule})});
    const elapsed=performance.now()-start;timings.push({name:`${c.kind}-d${c.dim}-n${c.n}-${schedule}`,...got.timings,totalMs:elapsed});
    let maxPrimalAbs=0,maxJvpAbs=0,maxPrimalScaled=0,maxJvpScaled=0,maxTranslationIdentityAbs=0,pass=true;
    for(let mi=0;mi<modeRecords.length;mi++){
      // LPT and PM are separately checked; the two use equal fixture inputs here.
      for(const base of [0,4])for(const group of [0,2])for(let part=0;part<2;part++){
        const expected=ref[4*mi+group+part],actual=got.channels[8*mi+base+group+part],e=err(actual,expected),scaled=e/Math.max(1e-6,Math.abs(expected));
        if(e>1e-6+1e-4*Math.abs(expected))pass=false;
        if(group===0){maxPrimalAbs=Math.max(maxPrimalAbs,e);maxPrimalScaled=Math.max(maxPrimalScaled,scaled);}else{maxJvpAbs=Math.max(maxJvpAbs,e);maxJvpScaled=Math.max(maxJvpScaled,scaled);}
      }
    }
    let translationIdentityWorst=null;
    if(c.kind==='translation'||c.kind==='large-translation')for(let mi=0;mi<fixtureModes.length;mi++){
      const mode=fixtureModes[mi],speed=TWO_PI*(mode[0]*0.13+mode[1]*-0.21+(c.dim===3?mode[2]*0.07:0));
      for(const base of [0,4]){
        const fr=got.channels[8*mi+base],fi=got.channels[8*mi+base+1],dr=got.channels[8*mi+base+2],di=got.channels[8*mi+base+3];
        for(const [actual,expected] of [[dr,speed*fi],[di,-speed*fr]]){const e=err(actual,expected);if(e>maxTranslationIdentityAbs){maxTranslationIdentityAbs=e;translationIdentityWorst={mode,base,actual,expected,F:[fr,fi],speed};}if(e>1e-6+1e-4*Math.abs(expected))pass=false;}
      }
    }
    const row={name:`${c.kind}-d${c.dim}-n${c.n}-${schedule}`,schedule,pass,maxPrimalAbs,maxJvpAbs,maxPrimalScaled,maxJvpScaled,maxTranslationIdentityAbs,translationIdentityWorst,componentTolerance:'1e-6 + 1e-4 * abs(reference)',diagnostics:got.diagnostics};rows.push(row);
    numericalErrors.push({fixture:row.name,maxPrimalAbs,maxJvpAbs,maxPrimalScaled,maxJvpScaled});
    if(!pass)return {status:'fail',rows,numericalErrors,timings,deviceInfo,budget:GPU_ERROR_BUDGET};
    }
  }
  return {status:'pass',rows,numericalErrors,timings,deviceInfo,budget:GPU_ERROR_BUDGET};
}

/** Kernel-only timing on current PreparedLabRun wave inputs; no CPU solver/FD calls. */
export async function runScheduleBenchmarks({device,onProgress=()=>{},repeats=3,run=runDirect}={}){
  if(!device)return {status:'skip',rows:[],reason:'No WebGPU device was provided'};
  await init();
  const rows=[];
  for(const n of [32,64]){
    const settings={epoch:'scaleFactor',fraction:.35,steps:32,mesh:n,forceMesh:n,omegaB:.049,h:.674,ns:.965,sigma8:.811,boxSize:300,initialSmoothing:14,initialCutoff:3,seed:7,ic:'waves',smoothing:.03,dim:3,n,order:4,omegaM:.315,scaleFactor:.25};
    const p=new PreparedLabRun(JSON.stringify(settings));
    try{
      const metadata=JSON.parse(p.metadata());
      const input={dim:metadata.dim,n:metadata.n,modes:p.mode_records(),lptPsi:p.lpt_displacement(),lptDirection:p.lpt_direction(),pmPsi:p.pm_displacement(),pmDirection:p.pm_direction()};
      const samples=[];
      // Warm each shader/pipeline before recording; alternate order to reduce
      // order bias while retaining independent queue submissions per sample.
      for(const schedule of ['chunk-mode','cell-tile'])await run(device,input,{schedule});
      for(let i=0;i<repeats;i++)for(const schedule of (i%2?['cell-tile','chunk-mode']:['chunk-mode','cell-tile'])){
        onProgress({stage:`Schedule benchmark ${n}³ ${schedule}`,caseIndex:i+1,cases:repeats});
        const result=await run(device,input,{schedule});
        samples.push({schedule,repeat:i+1,kernelMs:result.timings.kernelMs,totalMs:result.timings.totalMs,partials:result.diagnostics.partialChunks,estimatedBytes:result.diagnostics.estimatedBytes,shaderSha256:result.diagnostics.provenance.evaluationShaderSha256});
      }
      for(const schedule of ['chunk-mode','cell-tile']){
        const values=samples.filter(s=>s.schedule===schedule).map(s=>s.kernelMs).sort((a,b)=>a-b);
        rows.push({label:`3D n=${n} waves`,request:settings,schedule,repeats,medianKernelMs:values[Math.floor(values.length/2)],minimumKernelMs:values[0],maximumKernelMs:values.at(-1),samples:samples.filter(s=>s.schedule===schedule),source:'current PreparedLabRun fields; kernelMs excludes preparation, uploads, and CPU reference/FD'});
      }
    }finally{p.free();}
    await new Promise(resolve=>setTimeout(resolve,0));
  }
  return {status:'pass',rows};
}
