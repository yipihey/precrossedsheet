// Direct P0 Fourier transform and fixed-mass analytic JVP for the precrossedsheet lab.
// Adapted in part from Hopf–Cole Cosmology app/js/gpu/direct.js, revision
// 3a3a13b9ba817c5b4f8f9f26314715845e13d4ad. Copyright (c) 2026 Tom Abel;
// MIT license; see reuse-manifest.json. The stable local-Taylor/Newton
// divided-difference method is retained. The P1 evaluator, random mode sampler,
// and silent magnitude filters are not used here.

import { STORAGE, UNIFORM, makePipeline, bindGroup, dispatch1d, readBuffer } from './util.js';

export const DIRECT_KERNEL_PROVENANCE = Object.freeze({
  revision: 'precrossedsheet-gpu-p0-2026-10-10',
  upstream: 'hopf-cole-cosmology@3a3a13b9ba817c5b4f8f9f26314715845e13d4ad:app/js/gpu/direct.js',
  method: 'stable f32 local Taylor/Newton exponential divided differences; fixed-mass P0 JVP',
  channels: '[F.re,F.im,dF.re,dF.im,PM.re,PM.im,dPM.re,dPM.im]',
  reduction: 'fixed 64-lane tree over lane-strided chunk sums',
  schedule: 'selected per run; see diagnostics.provenance.schedule',
});
export const DEFAULT_DIRECT_SCHEDULE='cell-tile';

async function sha256(text) {
  const subtle=globalThis.crypto?.subtle;
  if(!subtle)throw new Error('Web Crypto SHA-256 is unavailable in this worker context');
  const digest=await subtle.digest('SHA-256',new TextEncoder().encode(text));
  return [...new Uint8Array(digest)].map((b)=>b.toString(16).padStart(2,'0')).join('');
}

const EVAL_WGSL = /* wgsl */`
struct Params { dim:u32, n:u32, ncell:u32, nm:u32, cpc:u32, ctotal:u32, chunk0:u32, chunk1:u32, invn:f32, dxd:f32, pad0:u32, pad1:u32 };
@group(0) @binding(0) var<uniform> u:Params;
@group(0) @binding(1) var<storage,read> lptPsi:array<vec4<f32>>;
@group(0) @binding(2) var<storage,read> lptDir:array<vec4<f32>>;
@group(0) @binding(3) var<storage,read> pmPsi:array<vec4<f32>>;
@group(0) @binding(4) var<storage,read> pmDir:array<vec4<f32>>;
@group(0) @binding(5) var<storage,read> modes:array<vec4<i32>>;
@group(0) @binding(6) var<storage,read_write> outLpt:array<vec4<f32>>;
@group(0) @binding(7) var<storage,read_write> outPm:array<vec4<f32>>;
@group(0) @binding(8) var<storage,read_write> diag:array<atomic<u32>>;
const TWO_PI:f32=6.283185307179586;
const THR:f32=0.5;
const BAD_MAG:f32=1.0e30;
var<private> SIMP:array<u32,24>;

fn cph(t:f32)->vec2<f32>{let a=-TWO_PI*fract(t);return vec2<f32>(cos(a),sin(a));}
fn cm(a:vec2<f32>,b:vec2<f32>)->vec2<f32>{return vec2<f32>(a.x*b.x-a.y*b.y,a.x*b.y+a.y*b.x);}
fn ddTaylor(x:array<f32,5>,lo:u32,cnt:u32)->vec2<f32>{
 var tb=0.0; for(var a=0u;a<cnt;a++){tb+=x[lo+a];} tb/=f32(cnt);
 var radius=0.0; for(var a=0u;a<cnt;a++){radius+=abs(x[lo+a]-tb);} radius*=TWO_PI;
 let l=cnt-1u; var f0=1.0; for(var q=2u;q<=l;q++){f0/=f32(q);}
 var terms=1u; var bound=f0;
 loop { bound=bound*radius/f32(terms+l); if(bound<1.0e-7*f0||terms>=24u){break;} terms++; }
 var h:array<f32,25>; h[0]=1.0; for(var m=1u;m<=terms;m++){h[m]=0.0;}
 for(var a=0u;a<cnt;a++){let z=-TWO_PI*(x[lo+a]-tb);for(var m=1u;m<=terms;m++){h[m]+=z*h[m-1u];}}
 var sr=0.0;var si=0.0;var coeff=f0;
 for(var m=0u;m<=terms;m++){if(m>0u){coeff/=f32(m+l);}let t=h[m]*coeff;switch(m&3u){case 0u:{sr+=t;}case 1u:{si+=t;}case 2u:{sr-=t;}default:{si-=t;}}}
 return cm(vec2<f32>(sr,si),cph(tb));
}
fn ddEval(source:array<f32,5>,cnt:u32)->vec2<f32>{
 var x=source;
 for(var i=1u;i<cnt;i++){let v=x[i];var j=i;loop{if(j==0u||x[j-1u]<=v){break;}x[j]=x[j-1u];j--;}x[j]=v;}
 if(TWO_PI*(x[cnt-1u]-x[0])<THR){return ddTaylor(x,0u,cnt);}
 var table:array<vec2<f32>,25>;for(var i=0u;i<cnt;i++){table[i*5u+i]=cph(x[i]);}
 for(var level=1u;level<cnt;level++){for(var i=0u;i+level<cnt;i++){
  let span=TWO_PI*(x[i+level]-x[i]);var value:vec2<f32>;
  if(span<THR){if(level==1u){let half=span*0.5;var sinc=1.0-half*half/6.0;if(half>1.0e-3){sinc=sin(half)/half;}value=cm(cph((x[i]+x[i+1u])*0.5),vec2<f32>(sinc,0.0));}else{value=ddTaylor(x,i,level+1u);}}
  else{let d=table[(i+1u)*5u+i+level]-table[i*5u+i+level-1u];value=vec2<f32>(-d.y,d.x)/span;}
  table[i*5u+i+level]=value;
 }}
 return table[cnt-1u];
}
fn checked(v:vec2<f32>)->vec2<f32>{
 if(v.x!=v.x||v.y!=v.y||max(abs(v.x),abs(v.y))>BAD_MAG){atomicOr(&diag[0],1u);atomicAdd(&diag[1],1u);}
 return v;
}
fn checked4(v:vec4<f32>)->vec4<f32>{
 if(v.x!=v.x||v.y!=v.y||v.z!=v.z||v.w!=v.w||max(max(abs(v.x),abs(v.y)),max(abs(v.z),abs(v.w)))>BAD_MAG){atomicOr(&diag[0],1u);atomicAdd(&diag[1],1u);}
 return v;
}
`;

function geometryFn(name, psiName, dirName, dim) {
  const ns = dim === 2 ? 2 : 6;
  const tets = dim === 2
    ? [0,1,3,0,3,2]
    : [0,1,3,7,0,1,5,7,0,2,3,7,0,2,6,7,0,4,5,7,0,4,6,7];
  return /* wgsl */`
fn ${name}(cell:u32, md:vec4<i32>)->vec4<f32>{
 let n=u.n;var i0=0u;var j0=0u;var k0=0u;
 if(u.dim==2u){i0=cell/n;j0=cell%n;}else{i0=cell/(n*n);j0=(cell/n)%n;k0=cell%n;}
 var psi:array<vec3<f32>,8>;var direction:array<vec3<f32>,8>;
 for(var b=0u;b<(1u<<u.dim);b++){
  var ix=i0+(b&1u);if(ix>=n){ix-=n;}var jy=j0+((b>>1u)&1u);if(jy>=n){jy-=n;}
  var kz=0u;if(u.dim==3u){kz=k0+((b>>2u)&1u);if(kz>=n){kz-=n;}}
  let idx=select(ix*n+jy,(ix*n+jy)*n+kz,u.dim==3u);let v=${psiName}[idx];let dv=${dirName}[idx];
  psi[b]=vec3<f32>(v.xyz);direction[b]=vec3<f32>(dv.xyz);
 }
 var total=vec4<f32>(0.0);var ss:array<f32,5>;var aug:array<f32,5>;
 for(var si=0u;si<${ns}u;si++){
  let nt=u.dim+1u;for(var a=0u;a<nt;a++){SIMP[a]=${tets.length ? '0u' : '0u'};}
  // The simplex table is emitted inline below to keep vertex order identical to Rust.
  for(var a=0u;a<nt;a++){SIMP[a]=${dim === 2 ? 'array<u32,6>('+tets.map(x=>x+'u').join(',')+')[si*3u+a]' : 'array<u32,24>('+tets.map(x=>x+'u').join(',')+')[si*4u+a]'};}
  let b0=SIMP[0];let base=vec3<i32>(i32(i0+(b0&1u)),i32(j0+((b0>>1u)&1u)),i32(k0+((b0>>2u)&1u)));
  let raw=md.x*base.x+md.y*base.y+md.z*base.z;let rem=((raw%i32(n))+i32(n))%i32(n);
  // Reduce only the base exponential phase before adding the small exact-lattice
  // fraction; this avoids losing its low bits when m·psi contains a large shift.
  let t0=fract(f32(rem)*u.invn+fract(dot(vec3<f32>(md.xyz),psi[b0])));
  for(var a=0u;a<nt;a++){
   let ba=SIMP[a];let delta=vec3<i32>(i32(ba&1u)-i32(b0&1u),i32((ba>>1u)&1u)-i32((b0>>1u)&1u),i32((ba>>2u)&1u)-i32((b0>>2u)&1u));
   let mdelta=md.x*delta.x+md.y*delta.y+md.z*delta.z;
   ss[a]=f32(mdelta)*u.invn+dot(vec3<f32>(md.xyz),psi[ba]-psi[b0]);
  }
  var f=ddEval(ss,nt);f=cm(f,cph(t0));total=total+vec4<f32>(checked(f),0.0,0.0);
  for(var a=0u;a<nt;a++){
   for(var q=0u;q<nt;q++){aug[q]=ss[q];}aug[nt]=ss[a];
   let repeated=ddEval(aug,nt+1u);let du=dot(vec3<f32>(md.xyz),direction[SIMP[a]]);
   // ds/dA = -i 2 pi (m . u), with the vertex direction kept attached to its original phase.
   let term=cm(vec2<f32>(0.0,-TWO_PI*du),cm(repeated,cph(t0)));
   total=total+vec4<f32>(0.0,0.0,checked(term));
  }
 }
 return checked4(total*u.dxd);
}
`;
}

function makeEvalShader(dim, schedule) {
  const geometry = EVAL_WGSL + geometryFn('evalLpt', 'lptPsi', 'lptDir', dim) + geometryFn('evalPm', 'pmPsi', 'pmDir', dim);
  if (schedule === 'cell-tile') return geometry + /* wgsl */`
var<workgroup> lptTile:array<vec4<f32>,64>;
var<workgroup> pmTile:array<vec4<f32>,64>;
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wg:vec3<u32>,@builtin(local_invocation_index) lane:u32,@builtin(num_workgroups) nwg:vec3<u32>){
 let group=wg.y*nwg.x+wg.x;let count=(u.chunk1-u.chunk0)*u.nm;
 let validGroup=group<count;let mi=group%u.nm;let chunk=u.chunk0+group/u.nm;
 let cell=chunk*u.cpc+lane;let validCell=validGroup&&cell<u.ncell;let md=modes[mi];
 var a=vec4<f32>(0.0);var b=vec4<f32>(0.0);
 if(validCell){a=evalLpt(cell,md);b=evalPm(cell,md);}
 lptTile[lane]=a;pmTile[lane]=b;workgroupBarrier();
 for(var stride=32u;stride>0u;stride>>=1u){if(lane<stride){lptTile[lane]+=lptTile[lane+stride];pmTile[lane]+=pmTile[lane+stride];}workgroupBarrier();}
 if(lane==0u&&validGroup){outLpt[mi*u.ctotal+chunk]=checked4(lptTile[0]);outPm[mi*u.ctotal+chunk]=checked4(pmTile[0]);}
}
`;
  return geometry + /* wgsl */`
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid:vec3<u32>,@builtin(num_workgroups) nwg:vec3<u32>){
 let id=gid.y*nwg.x*64u+gid.x;let count=(u.chunk1-u.chunk0)*u.nm;if(id>=count){return;}
 let mi=id%u.nm;let chunk=u.chunk0+id/u.nm;let start=chunk*u.cpc;let end=min(start+u.cpc,u.ncell);let md=modes[mi];
 var a=vec4<f32>(0.0);var b=vec4<f32>(0.0);
 for(var cell=start;cell<end;cell++){a+=evalLpt(cell,md);b+=evalPm(cell,md);}
  a=vec4<f32>(checked(a.xy),checked(a.zw));b=vec4<f32>(checked(b.xy),checked(b.zw));
 outLpt[mi*u.ctotal+chunk]=a;outPm[mi*u.ctotal+chunk]=b;
}
`;
}

const REDUCE_WGSL = /* wgsl */`
struct RP{nm:u32,ctotal:u32};@group(0) @binding(0) var<uniform> r:RP;
@group(0) @binding(1) var<storage,read> inp:array<vec4<f32>>;@group(0) @binding(2) var<storage,read_write> out:array<vec4<f32>>;
var<workgroup> sh:array<vec4<f32>,64>;
@compute @workgroup_size(64) fn main(@builtin(workgroup_id) wg:vec3<u32>,@builtin(local_invocation_index) lane:u32){
 let mode=wg.x;if(mode>=r.nm){return;}var s=vec4<f32>(0.0);var i=lane;
 loop{if(i>=r.ctotal){break;}s+=inp[mode*r.ctotal+i];i+=64u;}sh[lane]=s;workgroupBarrier();
 for(var stride=32u;stride>0u;stride>>=1u){if(lane<stride){sh[lane]+=sh[lane+stride];}workgroupBarrier();}
 if(lane==0u){out[mode]=sh[0];}
}`;

const PIPELINES = new WeakMap();
async function pipelines(device, dim, schedule) {
  let cache = PIPELINES.get(device);
  if (!cache) { cache = new Map(); PIPELINES.set(device, cache); }
  const key = `${dim}:${schedule}`;
  if (!cache.has(key)) {
    const pending = Promise.all([
      makePipeline(device, makeEvalShader(dim, schedule), 'main', [
        { binding:0, visibility:4, buffer:{type:'uniform'} },
        ...[1,2,3,4,5].map((binding) => ({ binding, visibility:4, buffer:{type:'read-only-storage'} })),
        ...[6,7,8].map((binding) => ({ binding, visibility:4, buffer:{type:'storage'} })),
      ], `direct P0 ${schedule} d${dim}`),
      makePipeline(device, REDUCE_WGSL, 'main', [
        { binding:0, visibility:4, buffer:{type:'uniform'} },
        { binding:1, visibility:4, buffer:{type:'read-only-storage'} },
        { binding:2, visibility:4, buffer:{type:'storage'} },
      ], 'direct fixed reduction'),
    ]);
    cache.set(key, pending);
  }
  return cache.get(key);
}

function validateInput(device, input, options) {
  const { dim, n, modes, lptPsi, lptDirection, pmPsi, pmDirection } = input ?? {};
  if (![2,3].includes(dim) || !Number.isInteger(n) || n < 2) throw new TypeError('runDirect requires dim=2 or 3 and integer n >= 2');
  if (!(modes instanceof Int32Array) || modes.length % 4) throw new TypeError('modes must be Int32Array int4 records');
  const nv = n ** dim;
  for (const [name, a] of [['lptPsi',lptPsi],['lptDirection',lptDirection],['pmPsi',pmPsi],['pmDirection',pmDirection]]) {
    if (!(a instanceof Float32Array) || a.length !== 4 * nv) throw new TypeError(`${name} must be Float32Array with ${4*nv} packed vec4 values`);
    for (let i=0;i<a.length;i++) if (!Number.isFinite(a[i])) throw new RangeError(`${name}[${i}] is nonfinite`);
  }
  const nm = modes.length / 4;
  for (let i=0;i<nm;i++) {
    const phaseBound=n*(Math.abs(modes[4*i])+Math.abs(modes[4*i+1])+Math.abs(modes[4*i+2]));
    if (!Number.isSafeInteger(phaseBound)||phaseBound>0x7fffffff) throw new RangeError(`mode ${i} can overflow signed 32-bit periodic phase reconstruction`);
  }
  const limits = device.limits ?? {};
  if((limits.maxStorageBuffersPerShaderStage??8)<8)throw new RangeError('direct P0 needs 8 storage buffers per shader stage');
  if((limits.maxComputeWorkgroupSizeX??64)<64||(limits.maxComputeInvocationsPerWorkgroup??64)<64)throw new RangeError('direct P0 needs a 64-invocation workgroup');
  const inputBytes=4*nv*16;const perInputBytes=nv*16;const peakLimit=options.maxGpuBytes??(512*1024*1024);
  if(inputBytes>peakLimit)throw new RangeError(`GPU inputs need ${inputBytes} bytes, over maxGpuBytes=${peakLimit}`);
  if(perInputBytes>(limits.maxBufferSize??Infinity))throw new RangeError('GPU input array exceeds device.limits.maxBufferSize');
  if(perInputBytes>(limits.maxStorageBufferBindingSize??Infinity))throw new RangeError('one input array exceeds maxStorageBufferBindingSize');
  if(nv>0xffffffff)throw new RangeError('grid is too large for 32-bit shader indexing');
  return {nv,nm,ncell:nv,inputBytes,peakLimit,limits};
}

function makeBuffer(device, label, size, usage=STORAGE) {
  return device.createBuffer({label,size:Math.max(16,(size+3)&~3),usage});
}
function canceled(options) { return options.signal?.aborted || options.shouldCancel?.(); }

/**
 * Evaluate fixed-mass continuous P0 modes and analytic position JVPs for LPT and PM.
 * `input` is {dim,n,modes:Int32Array[4*nm],lptPsi,lptDirection,pmPsi,pmDirection},
 * where each geometry array is packed vec4<f32> per Lagrangian vertex.
 * Resolves {channels:Float64Array[8*nm], diagnostics, timings}.
 */
export async function runDirect(device, input, options = {}) {
  const tStart=performance.now();const checkedInput=validateInput(device,input,options);
  const {dim,n,modes,lptPsi,lptDirection,pmPsi,pmDirection}=input;
  const {nv,nm,ncell,inputBytes,peakLimit,limits}=checkedInput;
  if(!nm)return {channels:new Float64Array(0),diagnostics:{invalidCount:0,flags:0,contributions:0},timings:{totalMs:performance.now()-tStart,compileMs:0,uploadMs:0,kernelMs:0,readbackMs:0}};
  const schedule=options.schedule??DEFAULT_DIRECT_SCHEDULE;
  if(!['chunk-mode','cell-tile'].includes(schedule))throw new RangeError(`unknown direct GPU schedule: ${schedule}`);
  const cellsPerChunk=schedule==='cell-tile'?64:Math.max(1,Math.min(1024,Math.floor(options.cellsPerChunk??8)));
  const ctotal=Math.ceil(ncell/cellsPerChunk);const outputBytes=nm*ctotal*16*2+nm*16*2+8;
  const estimatedBytes=inputBytes+outputBytes;
  if(estimatedBytes>peakLimit)throw new RangeError(`estimated direct GPU allocation ${estimatedBytes} bytes exceeds maxGpuBytes=${peakLimit}`);
  if(nm*ctotal*16>(limits.maxStorageBufferBindingSize??Infinity))throw new RangeError('direct partial buffer exceeds maxStorageBufferBindingSize');
  const maxGroups=limits.maxComputeWorkgroupsPerDimension??65535;
  const maxChunksByDispatch=schedule==='cell-tile'?Math.floor(maxGroups*maxGroups/nm):Math.floor(maxGroups*maxGroups*64/nm);
  const defaultBatchChunks=schedule==='cell-tile'?Math.max(1,Math.min(4096,Math.floor(maxGroups/nm))):4096;
  const batchChunks=Math.max(1,Math.min(ctotal,options.batchChunks??defaultBatchChunks,maxChunksByDispatch));
  const batches=Math.ceil(ctotal/batchChunks);
  let compileMs=0,uploadMs=0,kernelMs=0,readbackMs=0,shaderHashMs=0;
  const owned=[];let submitted=false;
  const alloc=(...args)=>{const b=makeBuffer(device,...args);owned.push(b);return b;};
  const pushScope=device.pushErrorScope?.bind(device);const popScope=device.popErrorScope?.bind(device);
  try {
    if(canceled(options))throw new DOMException('GPU direct request canceled','AbortError');
    pushScope?.('validation');pushScope?.('out-of-memory');
    const tCompile=performance.now();const [main,reduce]=await pipelines(device,dim,schedule);compileMs=performance.now()-tCompile;
    const tHash=performance.now();const shaderHashes={evaluationShaderSha256:await sha256(makeEvalShader(dim,schedule)),reductionShaderSha256:await sha256(REDUCE_WGSL)};shaderHashMs=performance.now()-tHash;
    const psiL=alloc('direct LPT displacement',lptPsi.byteLength);const dirL=alloc('direct LPT direction',lptDirection.byteLength);
    const psiP=alloc('direct PM displacement',pmPsi.byteLength);const dirP=alloc('direct PM direction',pmDirection.byteLength);
    const modeBuf=alloc('direct int4 modes',modes.byteLength);const partL=alloc('direct LPT partials',nm*ctotal*16);const partP=alloc('direct PM partials',nm*ctotal*16);
    const outL=alloc('direct LPT result',nm*16);const outP=alloc('direct PM result',nm*16);const diag=alloc('direct diagnostics',8);
    const uniform=alloc('direct parameters',48,UNIFORM);
    const tUpload=performance.now();for(const [b,a] of [[psiL,lptPsi],[dirL,lptDirection],[psiP,pmPsi],[dirP,pmDirection],[modeBuf,modes]])device.queue.writeBuffer(b,0,a);
    device.queue.writeBuffer(diag,0,new Uint32Array([0,0]));uploadMs=performance.now()-tUpload;
    const cellsPerChunkU=cellsPerChunk;
    const groupsTotal=schedule==='cell-tile'?nm*batchChunks:Math.ceil(nm*batchChunks/64);const groups=groupsTotal;
    if(groups>maxGroups*maxGroups)throw new RangeError('dispatch exceeds device workgroup dimension limits');
    const tKernel=performance.now();
    for(let batch=0;batch<batches;batch++){
      if(canceled(options))throw new DOMException('GPU direct request canceled','AbortError');
      const chunk0=batch*batchChunks,chunk1=Math.min(ctotal,chunk0+batchChunks);
      const p=new ArrayBuffer(48);const u32=new Uint32Array(p);const f32=new Float32Array(p);
      u32[0]=dim;u32[1]=n;u32[2]=ncell;u32[3]=nm;u32[4]=cellsPerChunkU;u32[5]=ctotal;u32[6]=chunk0;u32[7]=chunk1;f32[8]=1/n;f32[9]=1/(n**dim);
      device.queue.writeBuffer(uniform,0,p);
      const bg=bindGroup(device,main.bgl,[uniform,psiL,dirL,psiP,dirP,modeBuf,partL,partP,diag],'direct P0 batch');
      const enc=device.createCommandEncoder({label:`direct P0 batch ${batch+1}/${batches}`});const pass=enc.beginComputePass({label:'direct P0 evaluate'});
      pass.setPipeline(main.pipeline);pass.setBindGroup(0,bg);if(groups<=maxGroups)pass.dispatchWorkgroups(groups);else pass.dispatchWorkgroups(maxGroups,Math.ceil(groups/maxGroups));pass.end();device.queue.submit([enc.finish()]);submitted=true;
      options.onProgress?.({stage:'evaluate',batch:batch+1,batches});
      await device.queue.onSubmittedWorkDone();
    }
    kernelMs=performance.now()-tKernel;
    if(canceled(options))throw new DOMException('GPU direct request canceled','AbortError');
    const redUniform=alloc('direct reduction parameters',16,UNIFORM);const redParams=new Uint32Array([nm,ctotal,0,0]);device.queue.writeBuffer(redUniform,0,redParams);
    const bgL=bindGroup(device,reduce.bgl,[redUniform,partL,outL],'direct LPT reduction');const bgP=bindGroup(device,reduce.bgl,[redUniform,partP,outP],'direct PM reduction');
    const enc=device.createCommandEncoder({label:'direct fixed reduction'});for(const bg of [bgL,bgP]){const pass=enc.beginComputePass({label:'direct fixed reduction'});pass.setPipeline(reduce.pipeline);pass.setBindGroup(0,bg);pass.dispatchWorkgroups(nm);pass.end();}device.queue.submit([enc.finish()]);submitted=true;
    const tRead=performance.now();const [lpt,pm,diagValues]=await Promise.all([readBuffer(device,outL,nm*16),readBuffer(device,outP,nm*16),readBuffer(device,diag,8,Uint32Array)]);readbackMs=performance.now()-tRead;
    const flags=diagValues[0],invalidCount=diagValues[1];
    if(flags||invalidCount)throw new Error(`GPU numerical diagnostics failed (flags=0x${flags.toString(16)}, invalidCount=${invalidCount}); no element contributions were intentionally filtered`);
    for(const values of [lpt,pm])for(let i=0;i<values.length;i++)if(!Number.isFinite(values[i])||Math.abs(values[i])>1e30)throw new Error(`GPU reduction produced an invalid magnitude at output ${i}; no element contributions were intentionally filtered`);
    const channels=new Float64Array(8*nm);for(let i=0;i<nm;i++){channels.set([lpt[4*i],lpt[4*i+1],lpt[4*i+2],lpt[4*i+3],pm[4*i],pm[4*i+1],pm[4*i+2],pm[4*i+3]],8*i);}
    const scopeErrors=[];if(popScope){const outErr=await popScope();const valErr=await popScope();for(const e of [valErr,outErr])if(e)scopeErrors.push(e.message??String(e));}
    if(scopeErrors.length)throw new Error(`WebGPU error scope: ${scopeErrors.join('; ')}`);
    return {channels,diagnostics:{invalidCount,flags,contributions:ncell*(dim===2?2:6)*nm,estimatedBytes,inputBytes,partialChunks:ctotal,batches,provenance:{...DIRECT_KERNEL_PROVENANCE,schedule,...shaderHashes}},timings:{totalMs:performance.now()-tStart,compileMs,shaderHashMs,uploadMs,kernelMs,readbackMs}};
  } catch(error) {
    if(submitted){try{await device.queue.onSubmittedWorkDone();}catch{/* device lost */}}
    if(popScope){try{await popScope();await popScope();}catch{/* scope already consumed */}}
    throw error;
  } finally {for(const b of owned){try{b.destroy();}catch{/* device lost */}}}
}
