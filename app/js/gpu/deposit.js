// Conservative GPU P0 sheet deposition by geometric simplex/cell clipping.
// Upstream repository pinned at hopf-cole-cosmology@3a3a13b9ba817c5b4f8f9f26314715845e13d4ad;
// the source clipper files were last changed in commit 5ae0550eaea89c85781de8cebb78462cb50aca6c.
// (MIT; see app/js/gpu/UPSTREAM-MIT-NOTICE.txt). The production path here uses
// positive f32 overlap masses and cell-owned compensated sums, never fixed-point bins.
import { WGSL_CLIP3D } from './deposit-clip3d.js';
import { readBuffer } from './util.js';

export const EXACT_DEPOSIT_PROVENANCE = Object.freeze({
  method: 'P0 positive-mass geometric clipping; one cell owner with compensated f32 sum',
  arithmetic: 'WGSL f32 geometry in cell-local coordinates; f32 mass accumulation; f32 density output',
  upstream: 'hopf-cole-cosmology@3a3a13b9ba817c5b4f8f9f26314715845e13d4ad (exact2d.js + exact3d.js clipper from 5ae0550eaea89c85781de8cebb78462cb50aca6c)',
  upstreamSha256: {
    exact2d: 'a66431508af962f0aa428f12ff13cc0312e9999312eefb28122ffd7cd0781a40',
    exact3d: '22fdfab5a7ad21fc108c7145bad9dfb03c109c42457b79da399fbefcd320cbb4',
  },
  accumulation: 'atomic linked contribution lists, then one Kahan-compensated f32 owner per output cell',
});

const USAGE_STORAGE_COPY = 0x80 | 0x04 | 0x08;
const USAGE_UNIFORM = 0x40 | 0x08;
const PIPELINES = new WeakMap();

const WGSL_CLIP2D = /* wgsl */`
const CAP2:u32=10u;
struct Poly2{p:array<vec2<f32>,CAP2>,n:u32};
var<private> T:Poly2; var<private> X:Poly2; var<private> A:Poly2; var<private> B:Poly2;
var<private> badClip:u32;
fn clip2(src:ptr<private,Poly2>,dst:ptr<private,Poly2>,ax:u32,coord:f32,sgn:f32)->u32{
 let n=(*src).n;var m=0u;
 for(var i=0u;i<n;i++){
  let j=select(i+1u,0u,i+1u==n);let a=(*src).p[i];let b=(*src).p[j];
  let sa=sgn*(a[ax]-coord);let sb=sgn*(b[ax]-coord);
  if(sa<=0.0){if(m>=CAP2){badClip=1u;return 0u;}(*dst).p[m]=a;m++;}
  if((sa<=0.0)!=(sb<=0.0)){if(m>=CAP2){badClip=1u;return 0u;}let t=sa/(sa-sb);(*dst).p[m]=a+(b-a)*t;m++;}
 }
 (*dst).n=m;return select(0u,m,m>=3u);
}
fn slab2(src:ptr<private,Poly2>,dst:ptr<private,Poly2>,ax:u32,lo:f32,hi:f32)->u32{
 if(clip2(src,&A,ax,lo,-1.0)==0u){return 0u;}return clip2(&A,dst,ax,hi,1.0);
}
fn area2(p:ptr<private,Poly2>)->f32{
 var s=0.0;for(var i=0u;i<(*p).n;i++){let j=select(i+1u,0u,i+1u==(*p).n);let a=(*p).p[i];let b=(*p).p[j];s+=a.x*b.y-b.x*a.y;}return 0.5*s;
}
`;

const WGSL_DEPOSIT_2D = /* wgsl */`
${WGSL_CLIP2D}
struct Desc{range:vec4<u32>};
struct Rec{mass:f32,next:u32};
struct P{u:vec4<u32>,f:vec4<f32>};
@group(0) @binding(0) var<uniform> p:P;
@group(0) @binding(1) var<storage,read> pos:array<vec4<f32>>;
@group(0) @binding(2) var<storage,read> desc:array<Desc>;
@group(0) @binding(3) var<storage,read_write> heads:array<atomic<u32>>;
@group(0) @binding(4) var<storage,read_write> rec:array<Rec>;
@group(0) @binding(5) var<storage,read_write> diag:array<atomic<u32>>;
var<private> baseCell:vec2<i32>;var<private> densityScale:f32;var<private> recOffset:u32;var<private> recCap:u32;var<private> recUsed:u32;
fn linear_id(gid:vec3<u32>)->u32{return gid.y*(65535u*64u)+gid.x;}
fn wrap(c:i32,n:i32)->u32{return u32(((c%n)+n)%n);}
fn append(cell:vec2<i32>,poly:ptr<private,Poly2>){
 let area=area2(poly);if(area==0.0){return;}if(area<0.0){atomicOr(&diag[0],32u);return;}
 let mass=densityScale*area;
 if(!(mass>0.0)||mass>3.402823e38){atomicOr(&diag[0],1u);return;}
 if(recUsed>=recCap){atomicOr(&diag[0],2u);return;}
 let c=baseCell+cell;let ix=wrap(c.x,i32(p.u.z));let iy=wrap(c.y,i32(p.u.z));let out=ix*p.u.z+iy;
 let id=recOffset+recUsed;recUsed++;
 let previous=atomicExchange(&heads[out],id+1u);rec[id].mass=mass;rec[id].next=previous;
}
fn walk(blo:vec2<i32>,bhi:vec2<i32>){
 for(var i=blo.x;i<bhi.x;i++){
  if(slab2(&T,&X,0u,f32(i),f32(i+1))==0u){continue;}
  var lo=X.p[0].y;var hi=lo;for(var v=1u;v<X.n;v++){lo=min(lo,X.p[v].y);hi=max(hi,X.p[v].y);}
  let j0=max(blo.y,i32(floor(lo)));let j1=min(bhi.y,i32(ceil(hi)));
  for(var j=j0;j<j1;j++){if(slab2(&X,&B,1u,f32(j),f32(j+1))!=0u){append(vec2<i32>(i,j),&B);}}
 }
}
fn vertexIndex(ix:u32,iy:u32,bx:u32,by:u32,n:u32)->u32{return ((ix+bx)%n)*n+((iy+by)%n);}
@compute @workgroup_size(64) fn clip_simplex(@builtin(global_invocation_id) gid:vec3<u32>){
 let localId=linear_id(gid);if(localId>=p.u.x){return;}let id=localId+p.u.w;badClip=0u;recUsed=0u;
 let n=p.u.y;let triId=id%2u;let cell=id/2u;let ix=cell/n;let iy=cell%n;let d=desc[id];recOffset=d.range.x;recCap=d.range.y;baseCell=vec2<i32>(i32(d.range.z&1023u),i32((d.range.z>>10u)&1023u));
 let tri0=vec3<u32>(0u,1u,3u);let tri1=vec3<u32>(0u,3u,2u);
 var bits:array<vec2<u32>,4>;bits[0]=vec2<u32>(0u,0u);bits[1]=vec2<u32>(1u,0u);bits[2]=vec2<u32>(0u,1u);bits[3]=vec2<u32>(1u,1u);
 let baseIndex=vertexIndex(ix,iy,0u,0u,n);var cp:array<vec2<f32>,4>;
 for(var b=0u;b<4u;b++){
  let q=bits[b];let vi=vertexIndex(ix,iy,q.x,q.y,n);var delta=pos[vi].xy-pos[baseIndex].xy;
  if(ix+q.x==n){delta.x+=p.f.y;}if(iy+q.y==n){delta.y+=p.f.y;}
  let scaledBase=pos[baseIndex].xy*p.f.x;let origin=scaledBase-floor(scaledBase);cp[b]=origin+delta*p.f.x;
 }
 var a=cp[0];var b=cp[select(tri0.y,tri1.y,triId==1u)];var c=cp[select(tri0.z,tri1.z,triId==1u)];
 let e1=b-a;let e2=c-a;var det=e1.x*e2.y-e1.y*e2.x;
 let scale=abs(e1.x*e2.y)+abs(e1.y*e2.x);
 if(det==0.0){atomicAdd(&diag[1],1u);if(scale>0.0){atomicOr(&diag[2],1u);}return;}
 if(abs(det)<=1.1920929e-7*16.0*scale){atomicOr(&diag[2],1u);return;}
 if(det<0.0){let t=b;b=c;c=t;det=-det;}
 let lo=min(a,min(b,c));let hi=max(a,max(b,c));let blo=vec2<i32>(floor(lo));let bhi=max(vec2<i32>(ceil(hi)),blo+vec2<i32>(1));
 densityScale=(p.f.z*p.f.z)/det;
 T.p[0]=a;T.p[1]=b;T.p[2]=c;T.n=3u;walk(blo,bhi);
 if(badClip!=0u){atomicOr(&diag[0],4u);}
}
`;

const WGSL_GATHER = /* wgsl */`
struct Rec{mass:f32,next:u32};struct P{u:vec4<u32>,f:vec4<f32>};
@group(0) @binding(0) var<uniform> p:P;
@group(0) @binding(1) var<storage,read_write> heads:array<atomic<u32>>;
@group(0) @binding(2) var<storage,read> rec:array<Rec>;
@group(0) @binding(3) var<storage,read_write> rho:array<f32>;
@group(0) @binding(4) var<storage,read_write> diag:array<atomic<u32>>;
fn linear_id(gid:vec3<u32>)->u32{return gid.y*(65535u*128u)+gid.x;}
@compute @workgroup_size(128) fn gather(@builtin(global_invocation_id) gid:vec3<u32>){
 let i=linear_id(gid);if(i>=p.u.x){return;}var id=atomicLoad(&heads[i]);var sum=0.0;var c=0.0;var visits=0u;
 loop{if(id==0u){break;}if(visits>=p.u.y){atomicOr(&diag[0],8u);return;}
  let r=rec[id-1u];let y=r.mass-c;let t=sum+y;c=(t-sum)-y;sum=t;id=r.next;visits++;
 }
 let scale=p.f.x;if(!(sum>=0.0)||sum>3.402823e38){atomicOr(&diag[0],16u);return;}
 rho[i]=sum*scale;
}
`;

// Production 3-D tetrahedral path is assembled from the reviewed graph clipper.
const WGSL_DEPOSIT_3D = /* wgsl */`
${WGSL_CLIP3D}
struct Desc{range:vec4<u32>};struct Rec{mass:f32,next:u32};struct P{u:vec4<u32>,f:vec4<f32>};
@group(0) @binding(0) var<uniform> p:P;@group(0) @binding(1) var<storage,read> pos:array<vec4<f32>>;@group(0) @binding(2) var<storage,read> desc:array<Desc>;
@group(0) @binding(3) var<storage,read_write> heads:array<atomic<u32>>;@group(0) @binding(4) var<storage,read_write> rec:array<Rec>;@group(0) @binding(5) var<storage,read_write> diag:array<atomic<u32>>;
var<private> cBase:vec3<i32>;var<private> cScale:f32;var<private> cOff:u32;var<private> cCap:u32;var<private> cUsed:u32;
fn linear_id(gid:vec3<u32>)->u32{return gid.y*(65535u*32u)+gid.x;}
fn wrap3(c:i32,n:i32)->u32{return u32(((c%n)+n)%n);}
fn emit(cell:vec3<i32>,poly:ptr<private,Poly>){
 if(g_ovf!=0u){return;}let vol=moments1(poly,vec3<f32>(cell)).x;
 if(g_ovf!=0u){return;}if(vol==0.0){return;}if(vol<0.0){atomicOr(&diag[0],32u);return;}let mass=cScale*vol;
 if(!(mass>0.0)||mass>3.402823e38){atomicOr(&diag[0],1u);return;}if(cUsed>=cCap){atomicOr(&diag[0],2u);return;}
 let c=cBase+cell;let i=wrap3(c.x,i32(p.u.z));let j=wrap3(c.y,i32(p.u.z));let k=wrap3(c.z,i32(p.u.z));let out=(i*p.u.z+j)*p.u.z+k;
 let rid=cOff+cUsed;cUsed++;let previous=atomicExchange(&heads[out],rid+1u);rec[rid].mass=mass;rec[rid].next=previous;
}
fn vi(ix:u32,iy:u32,iz:u32,bx:u32,by:u32,bz:u32,n:u32)->u32{return (((ix+bx)%n)*n+((iy+by)%n))*n+((iz+bz)%n);}
const T0:array<u32,24>=array<u32,24>(0u,1u,3u,7u,0u,1u,5u,7u,0u,2u,3u,7u,0u,2u,6u,7u,0u,4u,5u,7u,0u,4u,6u,7u);
@compute @workgroup_size(32) fn clip_simplex(@builtin(global_invocation_id) gid:vec3<u32>){
 let localId=linear_id(gid);if(localId>=p.u.x){return;}let id=localId+p.u.w;g_ovf=0u;cUsed=0u;let n=p.u.y;let tet=id%6u;let cell=id/6u;let ix=cell/(n*n);let iy=(cell/n)%n;let iz=cell%n;
 let d=desc[id];cOff=d.range.x;cCap=d.range.y;cBase=vec3<i32>(i32(d.range.z&1023u),i32((d.range.z>>10u)&1023u),i32((d.range.z>>20u)&1023u));
 var cube:array<vec3<f32>,8>;let base=vi(ix,iy,iz,0u,0u,0u,n);
 for(var b=0u;b<8u;b++){let bx=b&1u;let by=(b>>1u)&1u;let bz=(b>>2u)&1u;let v=vi(ix,iy,iz,bx,by,bz,n);var delta=pos[v].xyz-pos[base].xyz;
  if(ix+bx==n){delta.x+=p.f.y;}if(iy+by==n){delta.y+=p.f.y;}if(iz+bz==n){delta.z+=p.f.y;}let scaledBase=pos[base].xyz*p.f.x;let origin=scaledBase-floor(scaledBase);cube[b]=origin+delta*p.f.x;
 }
 let o=tet*4u;var a=cube[T0[o]];var b=cube[T0[o+1u]];var c=cube[T0[o+2u]];var d0=cube[T0[o+3u]];
 let e1=b-a;let e2=c-a;let e3=d0-a;let det=dot(e1,cross(e2,e3));let den=abs(e1.x*(e2.y*e3.z-e2.z*e3.y))+abs(e1.y*(e2.x*e3.z-e2.z*e3.x))+abs(e1.z*(e2.x*e3.y-e2.y*e3.x));
 if(det==0.0){atomicAdd(&diag[1],1u);if(den>0.0){atomicOr(&diag[2],1u);}return;}if(abs(det)<=1.1920929e-7*32.0*den){atomicOr(&diag[2],1u);return;}
 if(det<0.0){let q=b;b=c;c=q;}
 init_tet(a,b,c,d0,vec4<f32>(1.0));var lo=a;var hi=a;for(var v=1u;v<4u;v++){lo=min(lo,P_T.pos[v]);hi=max(hi,P_T.pos[v]);}
 let blo=vec3<i32>(floor(lo));let bhi=max(vec3<i32>(ceil(hi)),blo+vec3<i32>(1));cScale=(p.f.z*p.f.z*p.f.z)/abs(det);walk_tet(blo,bhi);
 if(g_ovf!=0u){atomicOr(&diag[0],4u);}
}
`;

function groups(pass, count, workgroupSize) {
  const g = Math.max(1, Math.ceil(count / workgroupSize));
  if (g <= 65535) pass.dispatchWorkgroups(g);
  else pass.dispatchWorkgroups(65535, Math.ceil(g / 65535));
}
function buffer(device, label, bytes, usage = USAGE_STORAGE_COPY) {
  return device.createBuffer({ label, size: Math.max(16, (bytes + 3) & ~3), usage });
}
function uniform(device, label, values) {
  const b = buffer(device, label, 32, USAGE_UNIFORM);
  device.queue.writeBuffer(b, 0, values);
  return b;
}
async function hash(text) {
  if (!globalThis.crypto?.subtle) return null;
  const bytes = await crypto.subtle.digest('SHA-256', new TextEncoder().encode(text));
  return [...new Uint8Array(bytes)].map(x => x.toString(16).padStart(2, '0')).join('');
}
async function getPipelines(device, dim) {
  let byDim = PIPELINES.get(device);
  if (!byDim) { byDim = new Map(); PIPELINES.set(device, byDim); }
  if (!byDim.has(dim)) {
    const code = dim === 2 ? WGSL_DEPOSIT_2D : WGSL_DEPOSIT_3D;
    const module = device.createShaderModule({ label: `exact P0 deposit ${dim}D`, code });
    const info = await module.getCompilationInfo();
    const errors = info.messages.filter(m => m.type === 'error');
    if (errors.length) throw new Error(`exact ${dim}D deposition shader failed: ${errors.map(m => m.message).join('; ')}`);
    const clipLayout = device.createBindGroupLayout({ entries: [
      {binding:0,visibility:4,buffer:{type:'uniform'}},{binding:1,visibility:4,buffer:{type:'read-only-storage'}},
      {binding:2,visibility:4,buffer:{type:'read-only-storage'}},{binding:3,visibility:4,buffer:{type:'storage'}},
      {binding:4,visibility:4,buffer:{type:'storage'}},{binding:5,visibility:4,buffer:{type:'storage'}},
    ]});
    const clip = device.createComputePipeline({layout:device.createPipelineLayout({bindGroupLayouts:[clipLayout]}),compute:{module,entryPoint:'clip_simplex'}});
    const gatherModule = device.createShaderModule({label:'exact P0 cell gather',code:WGSL_GATHER});
    const gatherInfo = await gatherModule.getCompilationInfo();
    const gatherErrors = gatherInfo.messages.filter(m => m.type === 'error');
    if (gatherErrors.length) throw new Error(`exact P0 gather shader failed: ${gatherErrors.map(m => m.message).join('; ')}`);
    const gatherLayout = device.createBindGroupLayout({entries:[
      {binding:0,visibility:4,buffer:{type:'uniform'}},{binding:1,visibility:4,buffer:{type:'storage'}},
      {binding:2,visibility:4,buffer:{type:'read-only-storage'}},{binding:3,visibility:4,buffer:{type:'storage'}},
      {binding:4,visibility:4,buffer:{type:'storage'}},
    ]});
    const gather = device.createComputePipeline({layout:device.createPipelineLayout({bindGroupLayouts:[gatherLayout]}),compute:{module:gatherModule,entryPoint:'gather'}});
    byDim.set(dim,{clip,clipLayout,gather,gatherLayout,code:`${code}\n${WGSL_GATHER}`});
  }
  return byDim.get(dim);
}

function modulo(i, n) { return ((i % n) + n) % n; }
function vertex2(i,j,bx,by,n) { return ((i+bx)%n)*n+(j+by)%n; }
function vertex3(i,j,k,bx,by,bz,n) { return (((i+bx)%n)*n+(j+by)%n)*n+(k+bz)%n; }

function simplexBounds(input, id, dim, triList) {
  const {n,meshN,boxSize,positions} = input;
  const cell=Math.floor(id/(dim===2?2:6)), simplex=id%(dim===2?2:6);
  const ix=dim===2?Math.floor(cell/n):Math.floor(cell/(n*n));
  const iy=dim===2?cell%n:Math.floor(cell/n)%n;
  const iz=dim===3?cell%n:0;
  const corners=dim===2?[[0,0,0],[1,0,0],[0,1,0],[1,1,0]]:[[0,0,0],[1,0,0],[0,1,0],[1,1,0],[0,0,1],[1,0,1],[0,1,1],[1,1,1]];
  const tets=[[0,1,3,7],[0,1,5,7],[0,2,3,7],[0,2,6,7],[0,4,5,7],[0,4,6,7]];
  const tris=[[0,1,3],[0,3,2]];
  const selected=dim===2?tris[simplex]:tets[simplex];
  const baseCorner=selected[0], baseBits=corners[baseCorner];
  const baseIdx=dim===2?vertex2(ix,iy,baseBits[0],baseBits[1],n):vertex3(ix,iy,iz,...baseBits,n);
  const basePos=Array.from({length:dim},(_,a)=>positions[4*baseIdx+a]);
  const baseCell=Array.from({length:dim},(_,a)=>Math.floor(basePos[a]/(boxSize/meshN)));
  if(baseCell.some(x=>!Number.isSafeInteger(x)))throw new RangeError('exact deposit base cell index exceeds JavaScript exact-integer range');
  const scale=Math.fround(meshN/boxSize);
  const origin=Array.from({length:dim},(_,a)=>{const scaled=Math.fround(basePos[a]*scale);return Math.fround(scaled-Math.floor(scaled));});
  const points=[];
  for(const c of selected){
    const bits=corners[c];const vi=dim===2?vertex2(ix,iy,bits[0],bits[1],n):vertex3(ix,iy,iz,...bits,n);
    const p=[],bound=[];
    for(let a=0;a<dim;a++){
      const lift=(a===0&&ix+bits[0]===n)||(a===1&&iy+bits[1]===n)||(a===2&&iz+bits[2]===n)?boxSize:0;
      const delta=Math.fround(Math.fround(positions[4*vi+a]-basePos[a])+Math.fround(lift));
      const product=Math.fround(delta*scale);
      const separate=Math.fround(origin[a]+product);
      const fused=Math.fround(origin[a]+delta*scale);
      p.push(separate);
      bound.push([separate,fused]);
    }
    if(p.some(x=>!Number.isFinite(x)||Math.abs(x)>1073741824)||bound.some(pair=>pair.some(x=>!Number.isFinite(x)||Math.abs(x)>1073741824)))throw new RangeError('simplex cell-local coordinate exceeds safe i32 clipping range');
    points.push(bound);
  }
  const lo=Array(dim).fill(Infinity),hi=Array(dim).fill(-Infinity);
  for(const p of points)for(let a=0;a<dim;a++){lo[a]=Math.min(lo[a],p[a][0],p[a][1]);hi[a]=Math.max(hi[a],p[a][0],p[a][1]);}
  const blo=lo.map(Math.floor),bhi=hi.map((x,a)=>Math.max(Math.ceil(x),blo[a]+1));
  const cap=blo.reduce((v,x,a)=>v*Math.max(1,bhi[a]-x),1);
  const baseMod=Array.from({length:dim},(_,a)=>{
    // Match Rust's base = floor(p0 / (boxSize / meshN)); the local cell
    // geometry is computed independently from the first vertex.
    const q=Math.floor(basePos[a]/(boxSize/meshN));
    return modulo(q,meshN);
  });
  return {cap,baseMod};
}

function preflight(input, dim, maxBytes, maxBatch) {
  const ns=input.n**dim*(dim===2?2:6), cellCount=input.meshN**dim;
  if(!Number.isSafeInteger(ns)||!Number.isSafeInteger(cellCount))throw new RangeError('simplex or output count exceeds safe integer range');
  if(input.meshN>1024)throw new RangeError('exact deposit compact cell addressing supports meshN <= 1024');
  const batches=Math.ceil(ns/maxBatch);
  const descriptorBytes=ns*16;
  const limits=input.device?.limits??{};const fixedBytes=input.positions.byteLength+descriptorBytes+cellCount*12+32;
  if(!Number.isSafeInteger(fixedBytes)||fixedBytes>maxBytes)throw new RangeError(`exact deposit minimum buffers need ${fixedBytes} bytes, exceeding maxGpuBytes=${maxBytes}`);
  const maxBuffer=limits.maxBufferSize??Infinity,maxStorage=limits.maxStorageBufferBindingSize??Infinity;
  for(const [name,size] of Object.entries({positions:input.positions.byteLength,descriptors:descriptorBytes,heads:cellCount*4,density:cellCount*4})){
    if(size>maxBuffer||size>maxStorage)throw new RangeError(`exact deposit ${name} buffer (${size} bytes) exceeds the device storage-buffer limit`);
  }
  const descriptors=new Uint32Array(ns*4);let totalRecords=0;
  for(let id=0;id<ns;id++){
    const b=simplexBounds(input,id,dim);
    if(totalRecords+b.cap>0xffffffff)throw new RangeError('exact deposit record indices exceed u32');
    const o=id*4;descriptors[o]=totalRecords;descriptors[o+1]=b.cap;
    descriptors[o+2]=b.baseMod[0]|((b.baseMod[1]??0)<<10)|((b.baseMod[2]??0)<<20);
    totalRecords+=b.cap;
  }
  const bytes={positions:input.positions.byteLength,descriptors:descriptors.byteLength,heads:cellCount*4,records:totalRecords*8,density:cellCount*4,diagnostics:16,readback:cellCount*4+16,batchUniforms:batches*32,gatherUniform:32};
  const estimatedBytes=Object.values(bytes).reduce((a,b)=>a+b,0);
  for(const [name,size] of Object.entries({positions:bytes.positions,descriptors:bytes.descriptors,heads:bytes.heads,records:bytes.records,density:bytes.density})){
    if(size>maxBuffer||size>maxStorage)throw new RangeError(`exact deposit ${name} buffer (${size} bytes) exceeds the device storage-buffer limit`);
  }
  if(estimatedBytes>maxBytes)throw new RangeError(`exact deposit needs ${estimatedBytes} bytes, exceeding maxGpuBytes=${maxBytes}`);
  return {descriptors,totalRecords,cellCount,simplexCount:ns,batches,bytes,estimatedBytes};
}

function checkInput(device,input) {
  if(!input||![2,3].includes(input.dim))throw new RangeError('dim must be 2 or 3');
  const {dim,n,meshN,positions}=input;
  if(!Number.isInteger(n)||n<2||!Number.isInteger(meshN)||meshN<2)throw new RangeError('n and meshN must be integers >= 2');
  if(!(positions instanceof Float32Array)||positions.length!==n**dim*4)throw new RangeError('positions must be Float32Array with four f32 values per lattice vertex');
  if(!positions.every(Number.isFinite))throw new RangeError('positions must contain only finite values');
  const boxSize=input.boxSize??1;
  if(!(Number.isFinite(boxSize)&&boxSize>0))throw new RangeError('boxSize must be finite and positive');
  if(!(Math.fround(boxSize)>0&&Number.isFinite(Math.fround(boxSize))&&Number.isFinite(Math.fround(meshN/boxSize))))throw new RangeError('boxSize and mesh scale must be representable as finite f32 values');
  if(!device?.createBuffer||!device?.createShaderModule)throw new TypeError('a WebGPU device is required');
  return {...input,boxSize,device};
}

function aborted(options){return options.signal?.aborted===true;}
function abortError(){return new DOMException('Exact deposit canceled','AbortError');}

/**
 * Exact P0 sheet density on a periodic mesh. The input is the retained f32
 * trajectory, packed vec4 per vertex; only its first `dim` components are used.
 * Any clipping, capacity, or nonfinite diagnostic rejects the result so the
 * caller can use the Rust f64 deposit as the recovery path.
 */
export async function runExactDeposit(device,input,options={}){
  const started=performance.now();input=checkInput(device,input);
  const maxGpuBytes=options.maxGpuBytes??512*1024*1024;
  if(!Number.isSafeInteger(maxGpuBytes)||maxGpuBytes<=0)throw new RangeError('maxGpuBytes must be a positive safe integer');
  const maxBatch=options.batchSimplices??16384;
  if(!Number.isSafeInteger(maxBatch)||maxBatch<=0)throw new RangeError('batchSimplices must be a positive safe integer');
  const preflightStart=performance.now();const prep=preflight(input,input.dim,maxGpuBytes,maxBatch);const preflightMs=performance.now()-preflightStart;const {descriptors,totalRecords,cellCount,simplexCount,batches,bytes,estimatedBytes}=prep;
  const compileStart=performance.now();const pipes=await getPipelines(device,input.dim);const compileMs=performance.now()-compileStart;
  if(aborted(options))throw abortError();
  const owned=[];const alloc=(label,size,usage)=>{const b=buffer(device,label,size,usage);owned.push(b);return b;};
  let uploadMs=0,kernelMs=0,readbackMs=0,submitted=false;
  try{
    const pbuf=alloc('exact deposit packed positions',bytes.positions,USAGE_STORAGE_COPY);
    const dbuf=alloc('exact deposit simplex descriptors',bytes.descriptors,USAGE_STORAGE_COPY);
    const heads=alloc('exact deposit cell heads',bytes.heads,USAGE_STORAGE_COPY);
    const rec=alloc('exact deposit positive mass records',bytes.records,USAGE_STORAGE_COPY);
    const rho=alloc('exact deposit density output',bytes.density,USAGE_STORAGE_COPY);
    const diag=alloc('exact deposit diagnostics',16,USAGE_STORAGE_COPY);
    const tUpload=performance.now();device.queue.writeBuffer(pbuf,0,input.positions);device.queue.writeBuffer(dbuf,0,descriptors);device.queue.writeBuffer(diag,0,new Uint32Array(4));
    const zeroHeads=device.createCommandEncoder({label:'clear exact deposit heads'});zeroHeads.clearBuffer(heads);device.queue.submit([zeroHeads.finish()]);submitted=true;uploadMs=performance.now()-tUpload;
    const scale=Math.fround(input.meshN/input.boxSize),ratio=Math.fround(input.meshN/input.n);
    const tKernel=performance.now();
    for(let batch=0;batch<batches;batch++){
      if(aborted(options))throw abortError();
      const base=batch*maxBatch,count=Math.min(maxBatch,simplexCount-base);
      const u=new Uint32Array([count,input.n,input.meshN,base]);
      const f=new Float32Array([scale,input.boxSize,ratio,0]);const raw=new ArrayBuffer(32);new Uint32Array(raw).set(u);new Float32Array(raw).set(f,4);
      const ubuf=alloc(`exact deposit parameters ${batch}`,32,USAGE_UNIFORM);device.queue.writeBuffer(ubuf,0,raw);
      const bg=device.createBindGroup({layout:pipes.clipLayout,entries:[{binding:0,resource:{buffer:ubuf}},{binding:1,resource:{buffer:pbuf}},{binding:2,resource:{buffer:dbuf}},{binding:3,resource:{buffer:heads}},{binding:4,resource:{buffer:rec}},{binding:5,resource:{buffer:diag}}]});
      const enc=device.createCommandEncoder({label:`exact deposit clipping batch ${batch+1}/${batches}`});const pass=enc.beginComputePass({label:'clip simplices and link positive mass'});pass.setPipeline(pipes.clip);pass.setBindGroup(0,bg);groups(pass,count,input.dim===2?64:32);pass.end();device.queue.submit([enc.finish()]);submitted=true;
      await device.queue.onSubmittedWorkDone();if(options.onProgress)await options.onProgress({phase:'clip',batch:batch+1,batches,completed:simplexCount===0?1:(base+count)/simplexCount});
      ubuf.destroy();owned.splice(owned.indexOf(ubuf),1);
    }
    if(aborted(options))throw abortError();
    // Each clipped contribution is already normalized by the source simplex
    // volume in cell-local mesh coordinates, so the cell gather only sums it.
    const gatherU=new Uint32Array([cellCount,totalRecords,input.meshN,input.dim]);const gatherF=new Float32Array([1,0,0,0]);const raw=new ArrayBuffer(32);new Uint32Array(raw).set(gatherU);new Float32Array(raw).set(gatherF,4);
    const gu=alloc('exact deposit gather parameters',32,USAGE_UNIFORM);device.queue.writeBuffer(gu,0,raw);
    const bg=device.createBindGroup({layout:pipes.gatherLayout,entries:[{binding:0,resource:{buffer:gu}},{binding:1,resource:{buffer:heads}},{binding:2,resource:{buffer:rec}},{binding:3,resource:{buffer:rho}},{binding:4,resource:{buffer:diag}}]});
    const enc=device.createCommandEncoder({label:'exact deposit cell gather'});const pass=enc.beginComputePass({label:'one compensated sum owner per output cell'});pass.setPipeline(pipes.gather);pass.setBindGroup(0,bg);groups(pass,cellCount,128);pass.end();device.queue.submit([enc.finish()]);submitted=true;await device.queue.onSubmittedWorkDone();kernelMs=performance.now()-tKernel;
    if(aborted(options))throw abortError();
    const tRead=performance.now(),density=await readBuffer(device,rho,bytes.density),diagWords=await readBuffer(device,diag,16,Uint32Array);readbackMs=performance.now()-tRead;
    if(diagWords[0]||diagWords[2]){
      const diagnostic={dim:input.dim,n:input.n,meshN:input.meshN,simplexCount,totalRecords,estimatedBytes,bufferBytes:bytes,degenerateSimplices:diagWords[1],illConditionedSimplices:diagWords[2],flags:diagWords[0]};
      const error=new Error(`exact deposit rejected by GPU diagnostics flags=0x${diagWords[0].toString(16)}, illConditioned=${diagWords[2]}`);
      error.name='ExactDepositError';error.code='EXACT_DEPOSIT_GPU_DIAGNOSTIC';error.diagnostics=diagnostic;throw error;
    }
    if(!density.every(Number.isFinite))throw new Error('exact deposit produced nonfinite density');
    const shaderHash=await hash(pipes.code);return {density,diagnostics:{dim:input.dim,n:input.n,meshN:input.meshN,simplexCount,totalRecords,estimatedBytes,bufferBytes:bytes,degenerateSimplices:diagWords[1],illConditionedSimplices:diagWords[2],flags:diagWords[0],batches,accumulation:'positive f32 mass records, one Kahan-compensated f32 owner per cell',shaderHash,provenance:{...EXACT_DEPOSIT_PROVENANCE,shaderHash}},timings:{preflightMs,compileMs,uploadMs,kernelMs,readbackMs,totalMs:performance.now()-started},provenance:{...EXACT_DEPOSIT_PROVENANCE,shaderHash}};
  }finally{
    if(submitted){try{await device.queue.onSubmittedWorkDone();}catch{}}
    for(const b of owned){try{b.destroy();}catch{}}
  }
}

export const EXACT_DEPOSIT_SHADER_SOURCE = Object.freeze({
  twoD: `${WGSL_DEPOSIT_2D}\n${WGSL_GATHER}`,
  threeD: `${WGSL_DEPOSIT_3D}\n${WGSL_GATHER}`,
});
