// Reusable worker-safe radix-2 complex FFT for 1D, 2D, or 3D row-major grids.
// The Stockham stages follow Hopf–Cole Cosmology's MIT fft3d.js implementation,
// generalized to the dimensions used by precrossedsheet. Inverse transforms are
// normalized by 1/n^dim in the last stage.
// Copyright (c) 2026 Tom Abel. See UPSTREAM-MIT-NOTICE.txt and reuse-manifest.json.
import { makePipeline, dispatch1d } from './util.js';

export const FFT_PROVENANCE = Object.freeze({
  upstream: 'hopf-cole-cosmology@3a3a13b9ba817c5b4f8f9f26314715845e13d4ad:app/js/gpu/fft3d.js',
  license: 'MIT; copyright notice in app/js/gpu/UPSTREAM-MIT-NOTICE.txt',
  method: 'Stockham radix-2 autosort; f32 complex; inverse normalized by n^dim',
});

const FFT_WGSL = /* wgsl */`
struct Params { n:u32, size:u32, dim:u32, axis:u32, inner:u32, ns:u32, twstep:u32, sign:f32, scale:f32, pad0:u32, pad1:u32, pad2:u32 };
@group(0) @binding(0) var<uniform> p:Params;
@group(0) @binding(1) var<storage,read> src:array<vec2<f32>>;
@group(0) @binding(2) var<storage,read_write> dst:array<vec2<f32>>;
@group(0) @binding(3) var<storage,read> tw:array<vec2<f32>>;
@compute @workgroup_size(256)
fn stage(@builtin(global_invocation_id) gid:vec3<u32>,@builtin(num_workgroups) nwg:vec3<u32>){
 let t=gid.y*(nwg.x*256u)+gid.x;
 let half=p.n>>1u; let total=p.size>>1u; if(t>=total){return;}
 let line=t/half; let j=t%half; let outer=line/p.inner; let innerIndex=line%p.inner;
 let base=outer*p.n*p.inner+innerIndex;
 let k=j&(p.ns-1u); let w=tw[k*p.twstep]; let c=vec2<f32>(w.x,p.sign*w.y);
 let v0=src[base+j*p.inner]; let q=src[base+(j+half)*p.inner];
 let v1=vec2<f32>(q.x*c.x-q.y*c.y,q.x*c.y+q.y*c.x);
 let jo=2u*j-k;
 dst[base+jo*p.inner]=(v0+v1)*p.scale;
 dst[base+(jo+p.ns)*p.inner]=(v0-v1)*p.scale;
}`;

const PACK_WGSL = /* wgsl */`
struct P{count:u32,pad0:u32,pad1:u32,pad2:u32};
@group(0) @binding(0) var<uniform> p:P;
@group(0) @binding(1) var<storage,read> src:array<f32>;
@group(0) @binding(2) var<storage,read_write> dst:array<vec2<f32>>;
@compute @workgroup_size(256) fn main(@builtin(global_invocation_id) gid:vec3<u32>,@builtin(num_workgroups) nwg:vec3<u32>){let i=gid.y*(nwg.x*256u)+gid.x;if(i<p.count){dst[i]=vec2<f32>(src[i],0.0);}}`;
const UNPACK_WGSL = /* wgsl */`
struct P{count:u32,pad0:u32,pad1:u32,pad2:u32};
@group(0) @binding(0) var<uniform> p:P;
@group(0) @binding(1) var<storage,read> src:array<vec2<f32>>;
@group(0) @binding(2) var<storage,read_write> dst:array<f32>;
@compute @workgroup_size(256) fn main(@builtin(global_invocation_id) gid:vec3<u32>,@builtin(num_workgroups) nwg:vec3<u32>){let i=gid.y*(nwg.x*256u)+gid.x;if(i<p.count){dst[i]=src[i].x;}}`;

const PIPELINES = new WeakMap();
async function pipelines(device) {
  let pending=PIPELINES.get(device);
  if(!pending){pending=Promise.all([
    makePipeline(device,FFT_WGSL,'stage',[
      {binding:0,visibility:4,buffer:{type:'uniform',hasDynamicOffset:true}},
      {binding:1,visibility:4,buffer:{type:'read-only-storage'}},
      {binding:2,visibility:4,buffer:{type:'storage'}},
      {binding:3,visibility:4,buffer:{type:'read-only-storage'}},
    ],'radix-2 FFT stage'),
    makePipeline(device,PACK_WGSL,'main',[
      {binding:0,visibility:4,buffer:{type:'uniform'}},{binding:1,visibility:4,buffer:{type:'read-only-storage'}},{binding:2,visibility:4,buffer:{type:'storage'}},
    ],'FFT real pack'),
    makePipeline(device,UNPACK_WGSL,'main',[
      {binding:0,visibility:4,buffer:{type:'uniform'}},{binding:1,visibility:4,buffer:{type:'read-only-storage'}},{binding:2,visibility:4,buffer:{type:'storage'}},
    ],'FFT real unpack'),
  ]);PIPELINES.set(device,pending);}
  return pending;
}

function groups(pass,count){dispatch1d(pass,count,256);}
function uniform(device,size,label){return device.createBuffer({label,size,usage:0x40|0x08});}
function buffer(device,label,size,usage=0x80|0x04|0x08){return device.createBuffer({label,size:Math.max(16,(size+3)&~3),usage});}

export class GpuFFT {
  constructor(device,n,dim=3){
    if(![1,2,3].includes(dim)||!Number.isInteger(n)||n<2||(n&(n-1))!==0)throw new RangeError('GpuFFT supports dimensions 1..3 and radix-2 n >= 2');
    this.device=device;this.n=n;this.dim=dim;this.size=n**dim;this.levels=Math.log2(n);this.bytes=this.size*8;
    const lim=device.limits??{};if(this.bytes>(lim.maxStorageBufferBindingSize??Infinity)||this.bytes>(lim.maxBufferSize??Infinity))throw new RangeError(`FFT complex field (${this.bytes} bytes) exceeds device storage-buffer limits`);
    this.initialized=false;
  }
  async init(){
    if(this.initialized)return this;
    const d=this.device;this.pipes=await pipelines(d);
    const tw=new Float32Array(this.n);for(let i=0;i<this.n/2;i++){tw[2*i]=Math.cos(2*Math.PI*i/this.n);tw[2*i+1]=Math.sin(2*Math.PI*i/this.n);}
    this.tw=buffer(d,'FFT twiddles',tw.byteLength,0x80|0x08);d.queue.writeBuffer(this.tw,0,tw);
    const slot=256,ns=2*this.dim*this.levels,raw=new ArrayBuffer(ns*slot),u=new Uint32Array(raw),f=new Float32Array(raw);
    for(let dir=0;dir<2;dir++)for(let axis=0;axis<this.dim;axis++)for(let s=0;s<this.levels;s++){
      const off=((dir*this.dim+axis)*this.levels+s)*slot/4,inner=this.n**(this.dim-axis-1),stageNs=2**s;
      u[off]=this.n;u[off+1]=this.size;u[off+2]=this.dim;u[off+3]=axis;u[off+4]=inner;u[off+5]=stageNs;u[off+6]=this.n/(2*stageNs);
      f[off+7]=dir===0?-1:1;f[off+8]=dir===1&&axis===this.dim-1&&s===this.levels-1?1/this.size:1;
    }
    this.params=buffer(d,'FFT stage parameters',raw.byteLength,0x40|0x08);d.queue.writeBuffer(this.params,0,raw);
    this.simpleParams=buffer(d,'FFT pack parameters',16,0x40|0x08);d.queue.writeBuffer(this.simpleParams,0,new Uint32Array([this.size,0,0,0]));
    this.scratch=buffer(d,'FFT ping-pong field',this.bytes);this.bgCache=new WeakMap();this.initialized=true;return this;
  }
  _bind(src,dst){let m=this.bgCache.get(src);if(!m){m=new WeakMap();this.bgCache.set(src,m);}let bg=m.get(dst);if(!bg){bg=this.device.createBindGroup({layout:this.pipes[0].bgl,entries:[{binding:0,resource:{buffer:this.params,size:48}},{binding:1,resource:{buffer:src}},{binding:2,resource:{buffer:dst}},{binding:3,resource:{buffer:this.tw}}]});m.set(dst,bg);}return bg;}
  transform(src,dst,direction='forward',encoder=null){
    if(!this.initialized)throw new Error('Call await fft.init() before transform()');
    if(src===dst)throw new RangeError('FFT source and destination buffers must differ');
    if(!['forward','inverse'].includes(direction))throw new RangeError('FFT direction must be forward or inverse');
    const own=!encoder,enc=encoder??this.device.createCommandEncoder({label:`${direction} ${this.dim}D FFT`}),dir=direction==='inverse'?1:0,total=this.size/2;
    let cur=src;const pass=enc.beginComputePass({label:`${direction} ${this.dim}D radix-2 FFT`});pass.setPipeline(this.pipes[0].pipeline);
    let stage=0;for(let axis=0;axis<this.dim;axis++)for(let s=0;s<this.levels;s++,stage++){
      const out=((this.dim*this.levels-1-stage)%2===0)?dst:this.scratch;
      const offset=((dir*this.dim+axis)*this.levels+s)*256;
      pass.setBindGroup(0,this._bind(cur,out),[offset]);groups(pass,total);cur=out;
    }
    pass.end();if(own)this.device.queue.submit([enc.finish()]);
  }
  forward(src,dst,encoder=null){this.transform(src,dst,'forward',encoder);}
  inverse(src,dst,encoder=null){this.transform(src,dst,'inverse',encoder);}
  _simple(pipe,src,dst,encoder){
    const own=!encoder,enc=encoder??this.device.createCommandEncoder({label:`FFT ${pipe}`});
    const p=this.pipes[pipe],bg=this.device.createBindGroup({layout:p.bgl,entries:[{binding:0,resource:{buffer:this.simpleParams}},{binding:1,resource:{buffer:src}},{binding:2,resource:{buffer:dst}}]});const pass=enc.beginComputePass({label:`FFT ${pipe}`});pass.setPipeline(p.pipeline);pass.setBindGroup(0,bg);groups(pass,this.size);pass.end();if(own)this.device.queue.submit([enc.finish()]);
  }
  packReal(src,dst,encoder=null){this._simple(1,src,dst,encoder);}
  unpackReal(src,dst,encoder=null){this._simple(2,src,dst,encoder);}
  destroy(){for(const b of [this.tw,this.params,this.simpleParams,this.scratch])b?.destroy();this.tw=this.params=this.simpleParams=this.scratch=null;this.bgCache=new WeakMap();this.initialized=false;}
}

export const FFT_SHADER_SOURCE=`${FFT_WGSL}\n${PACK_WGSL}\n${UNPACK_WGSL}`;
