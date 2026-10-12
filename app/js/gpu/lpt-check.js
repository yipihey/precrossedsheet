// Independent closed-form plane-wave checks for GPU nLPT construction.
import { runLpt } from './lpt.js';

function addCosMode(deltaHat,n,dim,axis,amplitude=1){
  const np=n**dim, k=[0,0,0];k[axis]=1;
  const rowMajor=(coords)=>dim===2?coords[0]*n+coords[1]:(coords[0]*n+coords[1])*n+coords[2];
  const plus=k.map(v=>v),minus=k.map(v=>v);for(let a=0;a<dim;a++)minus[a]=(n-minus[a])%n;
  deltaHat[2*rowMajor(plus)]+=amplitude*np/2;deltaHat[2*rowMajor(minus)]+=amplitude*np/2;
}
function assertFields(actual,expected,label,rows){
  for(let i=0;i<expected.length;i++){const a=actual[i],e=expected[i],absoluteError=Math.abs(a-e),tolerance=2e-6+2e-4*Math.abs(e);rows.push({label:`${label} scalar ${i}`,actual:a,reference:e,absoluteError,tolerance,pass:Number.isFinite(a)&&absoluteError<=tolerance});}
}
function coords(cell,n,dim){return dim===2?[Math.floor(cell/n),cell%n,0]:[Math.floor(cell/(n*n)),Math.floor(cell/n)%n,cell%n];}
function linearReference(n,dim,axis=0){
  const np=n**dim,stride=dim+dim*dim+3,out=new Float32Array(np*stride),k=2*Math.PI;
  for(let i=0;i<np;i++){const c=coords(i,n,dim),phase=2*Math.PI*c[axis]/n,cos=Math.cos(phase),sin=Math.sin(phase),base=i*stride;out[base+axis]=-sin/k;out[base+dim+axis*dim+axis]=-cos;out[base+dim+dim*dim]=-cos;}
  return out;
}
function linearTwoWaveReference2D(n){
  const np=n*n,dim=2,stride=dim+dim*dim+3,out=new Float32Array(np*stride),k=2*Math.PI;
  for(let i=0;i<np;i++){const [x,y]=coords(i,n,dim),px=2*Math.PI*x/n,py=2*Math.PI*y/n,base=i*stride;out[base]=-Math.sin(px)/k;out[base+1]=-Math.sin(py)/k;out[base+2]=-Math.cos(px);out[base+5]=-Math.cos(py);out[base+6]=-Math.cos(px)-Math.cos(py);}
  return out;
}
function secondOrder2DReference(n){
  const dim=2,np=n*n,stride=dim+dim*dim+3,out=new Float32Array(np*stride),k=2*Math.PI;
  for(let i=0;i<np;i++){const [x,y]=coords(i,n,dim),px=2*Math.PI*x/n,py=2*Math.PI*y/n,cx=Math.cos(px),cy=Math.cos(py),sx=Math.sin(px),sy=Math.sin(py),base=i*stride;
    out[base]=sx*cy/(2*k);out[base+1]=cx*sy/(2*k);
    out[base+2]=0.5*cx*cy;out[base+3]=-0.5*sx*sy;out[base+4]=-0.5*sx*sy;out[base+5]=0.5*cx*cy;
    out[base+6]=cx*cy;
  }
  return out;
}

/** Device self-test for derivative-safe Fourier solves, raw source layout, and CPU fallbacks. */
export async function runLptSelfTests(device,options={}){
  const rows=[];if(options.signal?.aborted||options.shouldCancel?.())throw new DOMException('LPT self-test canceled','AbortError');
  const cases=[];
  for(const dim of [2,3])for(let axis=0;axis<dim;axis++){const n=8,np=n**dim,deltaHat=new Float32Array(2*np);addCosMode(deltaHat,n,dim,axis);const specs=[{order:1,kind:0}];cases.push({name:`${dim}D linear plane wave along axis ${axis}`,input:{dim,n,deltaHat,specs},expected:linearReference(n,dim,axis)});}
  {const dim=2,n=8,np=n*n,deltaHat=new Float32Array(2*np);addCosMode(deltaHat,n,dim,0);addCosMode(deltaHat,n,dim,1);const specs=[{order:1,kind:0},{order:2,kind:1,a:0,b:0}];const stride=dim+dim*dim+3,expected=new Float32Array(specs.length*np*stride);expected.set(linearTwoWaveReference2D(n),0);expected.set(secondOrder2DReference(n),np*stride);cases.push({name:'2D quadratic two-wave 2LPT',input:{dim,n,deltaHat,specs},expected});}
  for(let index=0;index<cases.length;index++){
    if(options.signal?.aborted||options.shouldCancel?.())throw new DOMException('LPT self-test canceled','AbortError');const test=cases[index],actual=await runLpt(device,test.input,{...options,maxGpuBytes:options.maxGpuBytes??128*1024*1024});assertFields(actual.fields,test.expected,test.name,rows);await options.onProgress?.({stage:'lpt-selftest',case:index+1,cases:cases.length});
  }
  let nonRadixRejected=false;try{await runLpt(device,{dim:2,n:3,deltaHat:new Float32Array(18),specs:[{order:1,kind:0}]},options);}catch(error){nonRadixRejected=/radix-2/.test(error.message);}rows.push({label:'non-radix mesh selects CPU fallback',actual:nonRadixRejected,reference:true,pass:nonRadixRejected});
  let oneDimRejected=false;try{await runLpt(device,{dim:1,n:8,deltaHat:new Float32Array(16),specs:[{order:1,kind:0}]},options);}catch(error){oneDimRejected=/1D/.test(error.message);}rows.push({label:'1D mesh selects CPU fallback',actual:oneDimRejected,reference:true,pass:oneDimRejected});
  return {status:rows.every(row=>row.pass)?'pass':'fail',rows,budget:{absolute:2e-6,relative:2e-4},fixtures:cases.map(test=>test.name)};
}
