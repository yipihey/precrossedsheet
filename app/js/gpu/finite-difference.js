// GPU centered finite-difference observer for continuous P0 simplex modes.
// Inputs are f32 high/low expansions of the retained f64 endpoint displacements
// from the exact Lagrangian lattice. No tangent/JVP expression is substituted.

import { makePipeline, readBuffer } from './util.js';

export const FINITE_DIFFERENCE_PROVENANCE = Object.freeze({
  method: 'centered finite difference of continuous P0 simplex Fourier-mode power using retained high/low endpoint displacement expansions; stable endpoint amplitude delta from three-point Gauss integration of the exact continuous directional derivative along the two endpoints',
  arithmetic: 'double-single f32 expansions with explicit runtime rounding barriers and Dekker product residuals; sin through degree 17, cos through degree 18',
  dividedDifferences: 'fixed compare-swap sorting network; local Taylor series for angular spans <= 3 radians (40 normalized terms; conservative factorial-tail bound below 1e-17); Newton recursion otherwise',
  reduction: 'per-cell plus/minus/delta endpoint values written to one interleaved storage buffer; one invocation sequentially double-single reduces each 32-cell chunk, then one invocation per mode sequentially reduces chunk partials',
  step: 1e-4,
  source: 'core/src/observables.rs::sheet_mode and core/src/direct.rs::exp_divided_difference_with_scratch',
});

const TWO_PI = 2 * Math.PI;
const f32 = (x) => new Float32Array([x])[0];
function pairLiteral(x) {
  const hi=f32(x),lo=f32(x-hi);
  return `DS(${hi.toPrecision(9)},${lo.toPrecision(9)})`;
}
function horner(name, degree, factorialOffset, negativeFirst) {
  const c=[];
  for(let k=0;k<=degree;k++){
    let fact=1;
    for(let i=2;i<=factorialOffset+2*k;i++)fact*=i;
    let value=1/fact;
    if((k&1)!==0)value=-value;
    c.push(value);
  }
  let s=`var ${name}=${pairLiteral(c[degree])};`;
  for(let k=degree-1;k>=0;k--)s+=`${name}=dsAdd(dsMul(${name},z),${pairLiteral(c[k])});`;
  return s;
}

function shaderFor(dim) {
  const vertices=dim+1;
  const maxNodes=vertices+1;
  const simplices=dim===2?[0,1,3,0,3,2]:[0,1,3,7,0,1,5,7,0,2,3,7,0,2,6,7,0,4,5,7,0,4,6,7];
  const qCode=simplices.map(x=>`${x}u`).join(',');
  const pi2=pairLiteral(TWO_PI);
  const thetaRoot=pairLiteral(Math.sqrt(3/5));
  const weightMid=pairLiteral(8/9);
  const weightSide=pairLiteral(5/9);
  const sortPairs=[];for(let pass=0;pass<maxNodes-1;pass++)for(let a=0;a<maxNodes-pass-1;a++)sortPairs.push([a,a+1]);
  const sortCode=sortPairs.map(([a,b])=>`if(${b}u<count){if(dsCmp(x[${a}u],x[${b}u])>0){let tmp=x[${a}u];x[${a}u]=x[${b}u];x[${b}u]=tmp;}}`).join('');
  const sinSource=horner('sp',8,1);
  const cosSource=horner('cp',9,0);
  return /* wgsl */`
struct DS { h:f32, l:f32 };
struct DC { r:DS, i:DS };
struct Params { dim:u32,n:u32,np:u32,nm:u32,cpc:u32,ctotal:u32,chunk0:u32,chunk1:u32,invn:f32,invmass:f32,pad0:u32,pad1:u32 };
@group(0) @binding(0) var<uniform> u:Params;
@group(0) @binding(1) var<storage,read> plusHi:array<vec4<f32>>;
@group(0) @binding(2) var<storage,read> plusLo:array<vec4<f32>>;
@group(0) @binding(3) var<storage,read> minusHi:array<vec4<f32>>;
@group(0) @binding(4) var<storage,read> minusLo:array<vec4<f32>>;
@group(0) @binding(5) var<storage,read> modes:array<vec4<i32>>;
@group(0) @binding(6) var<storage,read_write> cellValues:array<vec4<f32>>;
@group(0) @binding(7) var<storage,read_write> diag:array<atomic<u32>>;
const PI2:DS=${pi2};
const SIMP:array<u32,${simplices.length}>=array<u32,${simplices.length}>(${qCode});
fn rn(v:f32)->f32{return bitcast<f32>(bitcast<u32>(v)^u.pad0);}
fn ds(a:f32)->DS{return DS(a,0.0);}
fn dsNorm(a:f32,b:f32)->DS{let s=rn(a+b);let v=rn(s-a);let sv=rn(s-v);let aa=rn(a-sv);let bb=rn(b-v);let e=rn(aa+bb);let hi=rn(s+e);let hs=rn(hi-s);return DS(hi,rn(e-hs));}
fn dsAdd(a:DS,b:DS)->DS{let s=dsNorm(a.h,b.h);let e=rn(s.l+a.l);return dsNorm(s.h,rn(e+b.l));}
fn dsNeg(a:DS)->DS{return DS(-a.h,-a.l);}
fn dsSub(a:DS,b:DS)->DS{return dsAdd(a,dsNeg(b));}
fn twoProd(a:f32,b:f32)->DS{let p=rn(a*b);let ca=rn(4097.0*a);let aa=rn(ca-rn(ca-a));let ab=rn(a-aa);let cb=rn(4097.0*b);let ba=rn(cb-rn(cb-b));let bb=rn(b-ba);let e0=rn(rn(aa*ba)-p);let e1=rn(aa*bb);let e2=rn(ab*ba);let e3=rn(ab*bb);let e=rn(rn(rn(e0+e1)+e2)+e3);return DS(p,e);}
fn dsMul(a:DS,b:DS)->DS{let hh=twoProd(a.h,b.h);let e1=rn(a.h*b.l);let e2=rn(a.l*b.h);let e3=rn(a.l*b.l);let e=rn(rn(rn(hh.l+e1)+e2)+e3);return dsNorm(hh.h,e);}
fn dsDiv(a:DS,b:DS)->DS{let q0=rn(a.h/b.h);let r0=dsSub(a,dsMul(b,ds(q0)));let q1=rn(rn(r0.h+r0.l)/b.h);let q=dsAdd(ds(q0),ds(q1));let r1=dsSub(a,dsMul(b,q));let q2=rn(rn(r1.h+r1.l)/b.h);return dsAdd(q,ds(q2));}
fn dsDivInt(a:DS,b:u32)->DS{return dsDiv(a,ds(f32(b)));}
fn dsCmp(a:DS,b:DS)->i32{if(a.h<b.h){return -1;}if(a.h>b.h){return 1;}if(a.l<b.l){return -1;}if(a.l>b.l){return 1;}return 0;}
fn dsValue(a:DS)->f32{return rn(a.h+a.l);}
fn dsFinite(a:DS)->bool{return a.h==a.h&&a.l==a.l&&abs(a.h)<1.0e30&&abs(a.l)<1.0e30;}
fn dcAdd(a:DC,b:DC)->DC{return DC(dsAdd(a.r,b.r),dsAdd(a.i,b.i));}
fn dcSub(a:DC,b:DC)->DC{return DC(dsSub(a.r,b.r),dsSub(a.i,b.i));}
fn dcMul(a:DC,b:DC)->DC{return DC(dsSub(dsMul(a.r,b.r),dsMul(a.i,b.i)),dsAdd(dsMul(a.r,b.i),dsMul(a.i,b.r)));}
fn dcScale(a:DC,s:DS)->DC{return DC(dsMul(a.r,s),dsMul(a.i,s));}
fn dcZero()->DC{return DC(ds(0.0),ds(0.0));}
fn cph(phase:DS)->DC{
 let q=i32(round(dsValue(dsMul(phase,ds(4.0)))));
 let rem=dsSub(phase,ds(0.25*f32(q)));
 let x=dsMul(rem,PI2);let z=dsMul(x,x);
 ${sinSource}${cosSource}
 let sn=dsMul(x,sp);let co=cp;let qm=((q%4)+4)%4;var c=co;var s=sn;
 if(qm==1){c=dsNeg(sn);s=co;}else if(qm==2){c=dsNeg(co);s=dsNeg(sn);}else if(qm==3){c=sn;s=dsNeg(co);}
 return DC(c,dsNeg(s));
}
fn ddivTaylor(x:array<DS,${maxNodes}>,start:u32,count:u32)->DC{
 var mean=ds(0.0);for(var a=0u;a<count;a++){mean=dsAdd(mean,x[start+a]);}mean=dsDivInt(mean,count);
 let l=count-1u;
 // Elementary symmetric coefficients and normalized Taylor terms stay in
 // registers; no mutable dynamically indexed coefficient workspace.
 var e1=ds(0.0);var e2=ds(0.0);var e3=ds(0.0);var e4=ds(0.0);var e5=ds(0.0);
 for(var a=0u;a<count;a++){
  let z=dsNeg(dsMul(PI2,dsSub(x[start+a],mean)));
  e5=dsAdd(e5,dsMul(z,e4));e4=dsAdd(e4,dsMul(z,e3));e3=dsAdd(e3,dsMul(z,e2));e2=dsAdd(e2,dsMul(z,e1));e1=dsAdd(e1,z);
 }
 var g0=ds(1.0);for(var q=2u;q<=l;q++){g0=dsDivInt(g0,q);}var g1=g0;var g2=ds(0.0);var g3=ds(0.0);var g4=ds(0.0);var g5=ds(0.0);
 var sum=DC(g0,ds(0.0));
 for(var m=1u;m<=40u;m++){
  let d=m+l;var term=dsDivInt(dsMul(e1,g1),d);
  if(m>=2u){term=dsSub(term,dsDivInt(dsDivInt(dsMul(e2,g2),d),d-1u));}
  if(m>=3u){term=dsAdd(term,dsDivInt(dsDivInt(dsDivInt(dsMul(e3,g3),d),d-1u),d-2u));}
  if(m>=4u){term=dsSub(term,dsDivInt(dsDivInt(dsDivInt(dsDivInt(dsMul(e4,g4),d),d-1u),d-2u),d-3u));}
  if(m>=5u){term=dsAdd(term,dsDivInt(dsDivInt(dsDivInt(dsDivInt(dsDivInt(dsMul(e5,g5),d),d-1u),d-2u),d-3u),d-4u));}
  switch(m&3u){case 0u:{sum.r=dsAdd(sum.r,term);}case 1u:{sum.i=dsAdd(sum.i,term);}case 2u:{sum.r=dsSub(sum.r,term);}default:{sum.i=dsSub(sum.i,term);}}
  g5=g4;g4=g3;g3=g2;g2=g1;g1=term;
 }
 return dcMul(sum,cph(mean));
}
fn ddEval(source:array<DS,${maxNodes}>,count:u32)->DC{
 var x=source;${sortCode}
 let angular=dsValue(dsMul(dsSub(x[count-1u],x[0]),PI2));if(angular<3.0){return ddivTaylor(x,0u,count);}
 var table:array<DC,${maxNodes*maxNodes}>;for(var i=0u;i<count;i++){table[i*${maxNodes}u+i]=cph(x[i]);}
 for(var level=1u;level<count;level++){for(var i=0u;i+level<count;i++){let span=dsMul(dsSub(x[i+level],x[i]),PI2);let angle=dsValue(span);var value:DC;if(angle<3.0){value=ddivTaylor(x,i,level+1u);}else{let d=dcSub(table[(i+1u)*${maxNodes}u+i+level],table[i*${maxNodes}u+i+level-1u]);value=DC(dsDiv(dsNeg(d.i),span),dsDiv(d.r,span));}table[i*${maxNodes}u+i+level]=value;}}
 return table[count-1u];
}
fn coords(cell:u32)->vec3<u32>{if(u.dim==2u){return vec3<u32>(cell/u.n,cell%u.n,0u);}return vec3<u32>(cell/(u.n*u.n),(cell/u.n)%u.n,cell%u.n);}
fn vertexId(base:vec3<u32>,bits:u32)->u32{let x=(base.x+(bits&1u))%u.n;let y=(base.y+((bits>>1u)&1u))%u.n;if(u.dim==2u){return x*u.n+y;}let z=(base.z+((bits>>2u)&1u))%u.n;return (x*u.n+y)*u.n+z;}
fn qbit(bits:u32,a:u32)->i32{return i32((bits>>a)&1u);}
fn displacementDot(id:u32,md:vec3<i32>,plus:bool)->DS{let hi=select(minusHi[id],plusHi[id],plus);let lo=select(minusLo[id],plusLo[id],plus);var v=ds(0.0);for(var a=0u;a<u.dim;a++){v=dsAdd(v,dsMul(ds(f32(md[a])),DS(hi[a],lo[a])));}return v;}
fn midpointDisplacementDot(id:u32,md:vec3<i32>,theta:DS)->DS{let p=displacementDot(id,md,true);let m=displacementDot(id,md,false);let midpoint=dsDivInt(dsAdd(p,m),2u);let halfgap=dsDivInt(dsSub(p,m),2u);return dsAdd(midpoint,dsMul(theta,halfgap));}
fn halfGapDot(id:u32,md:vec3<i32>)->DS{return dsDivInt(dsSub(displacementDot(id,md,true),displacementDot(id,md,false)),2u);}
fn endpointCell(cell:u32,md:vec4<i32>,plus:bool)->DC{
 let base=coords(cell);let ns=${vertices}u;var total=dcZero();let scale=dsDivInt(ds(1.0),u.np);
 let raw=md.x*i32(base.x)+md.y*i32(base.y)+md.z*i32(base.z);let rem=((raw%i32(u.n))+i32(u.n))%i32(u.n);let qphase=dsDivInt(ds(f32(rem)),u.n);
 for(var si=0u;si<${simplices.length/vertices}u;si++){
  let b0=SIMP[si*${vertices}u];let id0=vertexId(base,b0);let phase0=dsAdd(qphase,displacementDot(id0,md.xyz,plus));var nodes:array<DS,${maxNodes}>;
  for(var a=0u;a<ns;a++){let bits=SIMP[si*${vertices}u+a];let id=vertexId(base,bits);let dm=i32(qbit(bits,0u)-qbit(b0,0u))*md.x+i32(qbit(bits,1u)-qbit(b0,1u))*md.y+i32(qbit(bits,2u)-qbit(b0,2u))*md.z;let qrel=dsDivInt(ds(f32(dm)),u.n);nodes[a]=dsAdd(qrel,dsSub(displacementDot(id,md.xyz,plus),displacementDot(id0,md.xyz,plus)));}
  let term=dcMul(ddEval(nodes,ns),cph(phase0));total=dcAdd(total,term);
 }
 total=dcScale(total,scale);if(!dsFinite(total.r)||!dsFinite(total.i)){atomicOr(&diag[0],1u);atomicAdd(&diag[1],1u);}return total;
}
fn directionalCellAt(cell:u32,md:vec4<i32>,theta:DS)->DC{
 let base=coords(cell);let ns=${vertices}u;var total=dcZero();let scale=dsDivInt(ds(1.0),u.np);
 let raw=md.x*i32(base.x)+md.y*i32(base.y)+md.z*i32(base.z);let rem=((raw%i32(u.n))+i32(u.n))%i32(u.n);let qphase=dsDivInt(ds(f32(rem)),u.n);
 for(var si=0u;si<${simplices.length/vertices}u;si++){
  let b0=SIMP[si*${vertices}u];let id0=vertexId(base,b0);let phase0=dsAdd(qphase,midpointDisplacementDot(id0,md.xyz,theta));var nodes:array<DS,${maxNodes}>;var gaps:array<DS,${vertices}>;
  for(var a=0u;a<ns;a++){let bits=SIMP[si*${vertices}u+a];let id=vertexId(base,bits);let dm=i32(qbit(bits,0u)-qbit(b0,0u))*md.x+i32(qbit(bits,1u)-qbit(b0,1u))*md.y+i32(qbit(bits,2u)-qbit(b0,2u))*md.z;let qrel=dsDivInt(ds(f32(dm)),u.n);nodes[a]=dsAdd(qrel,dsSub(midpointDisplacementDot(id,md.xyz,theta),midpointDisplacementDot(id0,md.xyz,theta)));gaps[a]=halfGapDot(id,md.xyz);}
  var derivative=dcZero();for(var a=0u;a<ns;a++){var repeated=nodes;repeated[ns]=nodes[a];derivative=dcAdd(derivative,dcScale(ddEval(repeated,ns+1u),gaps[a]));}
  let phase=dcMul(derivative,cph(phase0));let dtheta=DC(dsMul(phase.i,PI2),dsNeg(dsMul(phase.r,PI2)));total=dcAdd(total,dtheta);
 }
 return dcScale(total,scale);
}
fn endpointDeltaCell(cell:u32,md:vec4<i32>)->DC{let root=${thetaRoot};let mid=directionalCellAt(cell,md,ds(0.0));let hi=directionalCellAt(cell,md,root);let lo=directionalCellAt(cell,md,dsNeg(root));return dcAdd(dcScale(mid,${weightMid}),dcScale(dcAdd(hi,lo),${weightSide}));}
@compute @workgroup_size(32) fn evaluate_endpoints(@builtin(workgroup_id) wg:vec3<u32>,@builtin(local_invocation_index) lane:u32,@builtin(num_workgroups) nwg:vec3<u32>){let group=wg.y*nwg.x+wg.x;let groups=(u.chunk1-u.chunk0)*u.nm;if(group>=groups){return;}let mi=group%u.nm;let chunk=u.chunk0+group/u.nm;let cell=chunk*u.cpc+lane;if(cell<u.np){let fp=endpointCell(cell,modes[mi],true);let fm=endpointCell(cell,modes[mi],false);let at=(mi*u.np+cell)*3u;cellValues[at]=vec4<f32>(fp.r.h,fp.r.l,fp.i.h,fp.i.l);cellValues[at+1u]=vec4<f32>(fm.r.h,fm.r.l,fm.i.h,fm.i.l);}}
@compute @workgroup_size(32) fn evaluate_delta(@builtin(workgroup_id) wg:vec3<u32>,@builtin(local_invocation_index) lane:u32,@builtin(num_workgroups) nwg:vec3<u32>){let group=wg.y*nwg.x+wg.x;let groups=(u.chunk1-u.chunk0)*u.nm;if(group>=groups){return;}let mi=group%u.nm;let chunk=u.chunk0+group/u.nm;let cell=chunk*u.cpc+lane;if(cell<u.np){let fd=endpointDeltaCell(cell,modes[mi]);if(!dsFinite(fd.r)||!dsFinite(fd.i)){atomicOr(&diag[0],2u);atomicAdd(&diag[1],1u);}let at=(mi*u.np+cell)*3u;cellValues[at+2u]=vec4<f32>(fd.r.h,fd.r.l,fd.i.h,fd.i.l);}}
`;
}

const CHUNK_REDUCE = /* wgsl */`
struct DS{h:f32,l:f32};struct Params{np:u32,nm:u32,cpc:u32,ctotal:u32,chunk0:u32,chunk1:u32,pad0:u32,pad1:u32};
@group(0) @binding(0) var<uniform> u:Params;
@group(0) @binding(1) var<storage,read> cellIn:array<vec4<f32>>;
@group(0) @binding(2) var<storage,read_write> partialOut:array<vec4<f32>>;
struct DC{r:DS,i:DS};
fn rn(v:f32)->f32{return bitcast<f32>(bitcast<u32>(v)^u.pad0);}
fn dsNorm(a:f32,b:f32)->DS{let s=rn(a+b);let v=rn(s-a);let sv=rn(s-v);let aa=rn(a-sv);let bb=rn(b-v);let e=rn(aa+bb);let hi=rn(s+e);let hs=rn(hi-s);return DS(hi,rn(e-hs));}
fn dsAdd(a:DS,b:DS)->DS{let s=dsNorm(a.h,b.h);let e=rn(s.l+a.l);return dsNorm(s.h,rn(e+b.l));}
fn dcAdd(a:DC,b:DC)->DC{return DC(dsAdd(a.r,b.r),dsAdd(a.i,b.i));}
fn loadDC(a:vec4<f32>)->DC{return DC(DS(a.x,a.y),DS(a.z,a.w));}
fn storeDC(a:DC)->vec4<f32>{return vec4<f32>(a.r.h,a.r.l,a.i.h,a.i.l);}
@compute @workgroup_size(1) fn chunk_reduce(@builtin(workgroup_id) wg:vec3<u32>,@builtin(num_workgroups) nwg:vec3<u32>){let group=wg.y*nwg.x+wg.x;let count=(u.chunk1-u.chunk0)*u.nm;if(group>=count){return;}let mi=group%u.nm;let chunk=u.chunk0+group/u.nm;let start=chunk*u.cpc;var p0=DC(DS(0.0,0.0),DS(0.0,0.0));var p1=p0;var p2=p0;for(var i=0u;i<u.cpc;i++){let cell=start+i;if(cell<u.np){let at=(mi*u.np+cell)*3u;p0=dcAdd(p0,loadDC(cellIn[at]));p1=dcAdd(p1,loadDC(cellIn[at+1u]));p2=dcAdd(p2,loadDC(cellIn[at+2u]));}}let out=(mi*u.ctotal+chunk)*3u;partialOut[out]=storeDC(p0);partialOut[out+1u]=storeDC(p1);partialOut[out+2u]=storeDC(p2);}
`;

const REDUCE = /* wgsl */`
struct DS{h:f32,l:f32};struct Params{nm:u32,ctotal:u32,pad0:u32,pad1:u32};
@group(0) @binding(0) var<uniform> u:Params;
@group(0) @binding(1) var<storage,read> partialIn:array<vec4<f32>>;
@group(0) @binding(2) var<storage,read_write> resultOut:array<vec4<f32>>;
struct DC{r:DS,i:DS};
fn rn(v:f32)->f32{return bitcast<f32>(bitcast<u32>(v)^u.pad0);}
fn dsNorm(a:f32,b:f32)->DS{let s=rn(a+b);let v=rn(s-a);let sv=rn(s-v);let aa=rn(a-sv);let bb=rn(b-v);let e=rn(aa+bb);let hi=rn(s+e);let hs=rn(hi-s);return DS(hi,rn(e-hs));}
fn dsAdd(a:DS,b:DS)->DS{let s=dsNorm(a.h,b.h);let e=rn(s.l+a.l);return dsNorm(s.h,rn(e+b.l));}
fn dcAdd(a:DC,b:DC)->DC{return DC(dsAdd(a.r,b.r),dsAdd(a.i,b.i));}
fn loadDC(a:vec4<f32>)->DC{return DC(DS(a.x,a.y),DS(a.z,a.w));}
fn storeDC(a:DC)->vec4<f32>{return vec4<f32>(a.r.h,a.r.l,a.i.h,a.i.l);}
@compute @workgroup_size(1) fn reduce(@builtin(workgroup_id) wg:vec3<u32>,@builtin(num_workgroups) nwg:vec3<u32>){let mi=wg.y*nwg.x+wg.x;if(mi>=u.nm){return;}var p0=DC(DS(0.0,0.0),DS(0.0,0.0));var p1=p0;var p2=p0;for(var i=0u;i<u.ctotal;i++){let at=(mi*u.ctotal+i)*3u;p0=dcAdd(p0,loadDC(partialIn[at]));p1=dcAdd(p1,loadDC(partialIn[at+1u]));p2=dcAdd(p2,loadDC(partialIn[at+2u]));}let out=mi*3u;resultOut[out]=storeDC(p0);resultOut[out+1u]=storeDC(p1);resultOut[out+2u]=storeDC(p2);}
`;

const PIPELINES=new WeakMap();
async function pipelines(device,dim){let p=PIPELINES.get(device);if(!p){p=new Map();PIPELINES.set(device,p);}if(!p.has(dim)){p.set(dim,Promise.all([
  makePipeline(device,shaderFor(dim),'evaluate_endpoints',[
    {binding:0,visibility:4,buffer:{type:'uniform'}},
    ...[1,2,3,4,5].map((binding)=>({binding,visibility:4,buffer:{type:'read-only-storage'}})),
    {binding:6,visibility:4,buffer:{type:'storage'}},{binding:7,visibility:4,buffer:{type:'storage'}},
  ],`centered P0 finite difference endpoints d${dim}`),
  makePipeline(device,shaderFor(dim),'evaluate_delta',[
    {binding:0,visibility:4,buffer:{type:'uniform'}},
    ...[1,2,3,4,5].map((binding)=>({binding,visibility:4,buffer:{type:'read-only-storage'}})),
    {binding:6,visibility:4,buffer:{type:'storage'}},{binding:7,visibility:4,buffer:{type:'storage'}},
  ],`centered P0 finite difference endpoint delta d${dim}`),
  makePipeline(device,CHUNK_REDUCE,'chunk_reduce',[
    {binding:0,visibility:4,buffer:{type:'uniform'}},{binding:1,visibility:4,buffer:{type:'read-only-storage'}},{binding:2,visibility:4,buffer:{type:'storage'}},
  ],'centered P0 per-chunk endpoint reduction'),
  makePipeline(device,REDUCE,'reduce',[
    {binding:0,visibility:4,buffer:{type:'uniform'}},{binding:1,visibility:4,buffer:{type:'read-only-storage'}},{binding:2,visibility:4,buffer:{type:'storage'}},
  ],'centered P0 endpoint reduction'),
]));}return p.get(dim);}

// A compact arithmetic diagnostic used by the finite-difference GPU suite. It exercises
// the same generated WGSL helpers as the endpoint kernel and returns raw
// double-single pairs so the caller can compare them with independent f64
// expectations. It is deliberately separate from normal FD evaluation.
export async function runPrecisionProbe(device,options={}){
  const probeEntry=/* wgsl */`
@compute @workgroup_size(1) fn precision_probe(){
 let a=DS(1.2345670461654663,0.00000001999999987845058);
 let b=DS(-1.2345659732818604,-0.00000000999999993922529);
 let av=dsAdd(a,b);let mv=dsMul(a,DS(1.0000001192092896,0.00000000999999993922529));
 let dv=dsDiv(DS(1.0,0.000000029999999583867185),DS(3.0,0.00000000999999993922529));
 let cv=cph(DS(0.12345679104328156,0.000000000123456789));
 var close:array<DS,4>;close[0]=DS(0.125,0.0);close[1]=DS(0.12500999867916107,0.0);close[2]=DS(0.12498000264167786,0.0);close[3]=ds(0.0);
 let tv=ddivTaylor(close,0u,3u);
 var spread:array<DS,4>;spread[0]=ds(0.01);spread[1]=ds(0.23);spread[2]=ds(0.61);spread[3]=ds(0.0);let nv=ddEval(spread,3u);
 var moderate:array<DS,4>;moderate[0]=ds(0.0);moderate[1]=ds(0.06);moderate[2]=ds(0.06);moderate[3]=ds(0.0);let mv2=ddivTaylor(moderate,0u,3u);
 var symmetric:array<DS,4>;symmetric[0]=ds(0.0);symmetric[1]=ds(0.06);symmetric[2]=ds(0.12);symmetric[3]=ds(0.0);let sv2=ddivTaylor(symmetric,0u,3u);
 var repeated:array<DS,4>;repeated[0]=ds(0.0);repeated[1]=ds(0.51);repeated[2]=ds(0.51);repeated[3]=ds(0.0);let rv2=ddEval(repeated,3u);
 var threshold:array<DS,4>;threshold[0]=ds(0.0);threshold[1]=ds(0.236);threshold[2]=ds(0.472);threshold[3]=ds(0.0);let thv=ddEval(threshold,3u);
 cellValues[0]=vec4<f32>(av.h,av.l,mv.h,mv.l);
 cellValues[1]=vec4<f32>(dv.h,dv.l,cv.r.h,cv.r.l);
 cellValues[2]=vec4<f32>(cv.i.h,cv.i.l,tv.r.h,tv.r.l);
 cellValues[3]=vec4<f32>(tv.i.h,tv.i.l,nv.r.h,nv.r.l);
 cellValues[4]=vec4<f32>(nv.i.h,nv.i.l,mv2.r.h,mv2.r.l);
 cellValues[5]=vec4<f32>(mv2.i.h,mv2.i.l,sv2.r.h,sv2.r.l);
 cellValues[6]=vec4<f32>(sv2.i.h,sv2.i.l,rv2.r.h,rv2.r.l);
 cellValues[7]=vec4<f32>(rv2.i.h,rv2.i.l,thv.r.h,thv.r.l);
}
`;
  const source=shaderFor(2).replace('@compute @workgroup_size(32) fn evaluate_endpoints',`${probeEntry}\n@compute @workgroup_size(32) fn evaluate_endpoints`);
  if(source===shaderFor(2))throw new Error('failed to inject finite-difference precision probe');
  const owned=[];let submitted=false;const push=device.pushErrorScope?.bind(device),pop=device.popErrorScope?.bind(device);let scopes=false;
  try{
    if(canceled(options))throw new DOMException('GPU precision probe canceled','AbortError');push?.('validation');push?.('out-of-memory');scopes=!!push&&!!pop;
    const pipe=await makePipeline(device,source,'precision_probe',[
      {binding:0,visibility:4,buffer:{type:'uniform'}},...[1,2,3,4,5].map(binding=>({binding,visibility:4,buffer:{type:'read-only-storage'}})),
      {binding:6,visibility:4,buffer:{type:'storage'}},{binding:7,visibility:4,buffer:{type:'storage'}},
    ],'centered FD arithmetic precision probe');
    const allocProbe=(label,size,usage)=>{const b=device.createBuffer({label,size:Math.max(16,(size+3)&~3),usage});owned.push(b);return b;};
    const uniform=allocProbe('FD probe params',48,0x40|0x08);const ro=[1,2,3,4,5].map(i=>allocProbe(`FD probe input ${i}`,16,0x80|0x04|0x08));const output=allocProbe('FD probe outputs',128,0x80|0x04|0x08);const diag=allocProbe('FD probe diagnostics',8,0x80|0x04|0x08);
    const raw=new ArrayBuffer(48),u=new Uint32Array(raw),f=new Float32Array(raw);u.set([2,2,4,1,64,1,0,1]);f[8]=0.5;f[9]=0.25;device.queue.writeBuffer(uniform,0,raw);device.queue.writeBuffer(diag,0,new Uint32Array([0,0]));
    const bg=device.createBindGroup({layout:pipe.bgl,entries:[{binding:0,resource:{buffer:uniform}},...ro.map((buffer,i)=>({binding:i+1,resource:{buffer}})),{binding:6,resource:{buffer:output}},{binding:7,resource:{buffer:diag}}]});
    const enc=device.createCommandEncoder({label:'FD arithmetic precision probe'});const pass=enc.beginComputePass({label:'FD arithmetic helper probe'});pass.setPipeline(pipe.pipeline);pass.setBindGroup(0,bg);pass.dispatchWorkgroups(1);pass.end();device.queue.submit([enc.finish()]);submitted=true;await device.queue.onSubmittedWorkDone();
    const p=await readBuffer(device,output,128);if(canceled(options))throw new DOMException('GPU precision probe canceled','AbortError');
    const pairs={add:[p[0],p[1]],multiply:[p[2],p[3]],divide:[p[4],p[5]],cos:[p[6],p[7]],sinNeg:[p[8],p[9]],closeTaylorReal:[p[10],p[11]],closeTaylorImag:[p[12],p[13]],spreadNewtonReal:[p[14],p[15]],spreadNewtonImag:[p[16],p[17]],moderateRepeatedTaylorReal:[p[18],p[19]],moderateRepeatedTaylorImag:[p[20],p[21]],symmetricTaylorReal:[p[22],p[23]],symmetricTaylorImag:[p[24],p[25]],repeatedNewtonReal:[p[26],p[27]],repeatedNewtonImag:[p[28],p[29]],thresholdTaylorReal:[p[30],p[31]]};
    const exact=(hi,lo)=>hi+lo;const aa=exact(1.2345670461654663,0.00000001999999987845058),bb=exact(-1.2345659732818604,-0.00000000999999993922529);const phase=exact(0.12345679104328156,Math.fround(0.000000000123456789));
    const expected={add:aa+bb,multiply:aa*exact(1.0000001192092896,0.00000000999999993922529),divide:(1+0.000000029999999583867185)/(3+0.00000000999999993922529),cos:Math.cos(2*Math.PI*phase),sinNeg:-Math.sin(2*Math.PI*phase)};
    const close=[0.125,0.12500999867916107,0.12498000264167786],spread=[Math.fround(0.01),Math.fround(0.23),Math.fround(0.61)],moderate=[0,Math.fround(0.06),Math.fround(0.06)],symmetric=[0,Math.fround(0.06),Math.fround(0.12)],repeated=[0,Math.fround(0.51),Math.fround(0.51)],threshold=[0,Math.fround(0.236),Math.fround(0.472)];
    const ddQuad=(xs)=>{const n=64,roots=[],weights=[];for(let i=0;i<n/2;i++){let z=Math.cos(Math.PI*(i+0.75)/(n+0.5)),pp=0,dp=0;for(let it=0;it<12;it++){let p0=1,p1=z;for(let k=2;k<=n;k++){const p=((2*k-1)*z*p1-(k-1)*p0)/k;p0=p1;p1=p;}pp=p1;dp=n*(z*p1-p0)/(z*z-1);const next=z-pp/dp;if(next===z){z=next;break;}z=next;}const weight=2/((1-z*z)*dp*dp);roots.push((1-z)/2,(1+z)/2);weights.push(weight/2,weight/2);}let r=0,i=0;for(let a=0;a<n;a++)for(let b=0;b<n;b++){const u=roots[a],v=roots[b],lambda1=u,lambda2=(1-u)*v,lambda0=(1-u)*(1-v),phase=lambda0*xs[0]+lambda1*xs[1]+lambda2*xs[2],angle=2*Math.PI*phase,w=weights[a]*weights[b]*(1-u);r+=w*Math.cos(angle);i-=w*Math.sin(angle);}return {r,i};};
    const t=ddQuad(close),s=ddQuad(spread),mr=ddQuad(moderate),sy=ddQuad(symmetric),rp=ddQuad(repeated),th=ddQuad(threshold);for(const [prefix,value] of [['closeTaylor',t],['spreadNewton',s],['moderateRepeatedTaylor',mr],['symmetricTaylor',sy],['repeatedNewton',rp],['thresholdTaylor',th]]){expected[`${prefix}Real`]=value.r;expected[`${prefix}Imag`]=value.i;}
    const errors={},rows=[];for(const [name,pair] of Object.entries(pairs)){const actual=pair[0]+pair[1],reference=expected[name];const absoluteError=Math.abs(actual-reference),pass=Number.isFinite(actual)&&Number.isFinite(reference)&&absoluteError<=1e-11;errors[name]=actual-reference;rows.push({label:`FD arithmetic probe: ${name}`,actual,reference,absoluteError,pass});}
    const shaderHash=await hash(source);const error=[];if(pop){for(const e of [await pop(),await pop()])if(e)error.push(e.message??String(e));scopes=false;}if(error.length)throw new Error(`WebGPU precision probe error scope: ${error.join('; ')}`);
    return {status:rows.every(row=>row.pass)?'pass':'fail',rows,budget:{absoluteError:1e-11},pairs,expected,errors,shaderHash,arithmetic:FINITE_DIFFERENCE_PROVENANCE.arithmetic};
  }catch(error){if(submitted){try{await device.queue.onSubmittedWorkDone();}catch{}}if(pop&&scopes){try{await pop();await pop();}catch{}}throw error;}finally{for(const b of owned){try{b.destroy();}catch{}}}
}

function canceled(options){return options.signal?.aborted||options.shouldCancel?.();}
function alloc(device,label,size,usage,owned){const b=device.createBuffer({label,size:Math.max(16,(size+3)&~3),usage});owned.push(b);return b;}
async function hash(source){const subtle=globalThis.crypto?.subtle;if(!subtle)return null;const d=await subtle.digest('SHA-256',new TextEncoder().encode(source));return [...new Uint8Array(d)].map(x=>x.toString(16).padStart(2,'0')).join('');}

function validate(device,input,options){
  const {dim,n,modes,h}=input??{};
  if(![2,3].includes(dim)||!Number.isInteger(n)||n<2)throw new TypeError('runFiniteDifference requires dim=2 or 3 and integer n >= 2');
  if(!(modes instanceof Int32Array)||!modes.length||modes.length%4)throw new TypeError('modes must be nonempty Int32Array int4 records');
  if(h!==1e-4)throw new RangeError('GPU centered finite differences require the canonical h=1e-4');
  const np=n**dim,nm=modes.length/4;if(!Number.isSafeInteger(np)||np>0xffffffff)throw new RangeError('finite-difference grid exceeds 32-bit indexing');
  const arrays=['plusHi','plusLo','minusHi','minusLo'];let maxDisplacement=0;for(const name of arrays){const a=input[name];if(!(a instanceof Float32Array)||a.length!==4*np)throw new TypeError(`${name} must be packed vec4 Float32Array with ${4*np} values`);for(let i=0;i<a.length;i++){if(!Number.isFinite(a[i]))throw new RangeError(`${name}[${i}] is nonfinite`);if((i&3)<dim)maxDisplacement=Math.max(maxDisplacement,Math.abs(a[i]));}}
  let maxModeL1=0;for(let i=0;i<nm;i++){if(dim===2&&modes[4*i+2]!==0)throw new RangeError(`mode ${i} has a nonzero z component in dim=2`);const sum=Math.abs(modes[4*i])+Math.abs(modes[4*i+1])+Math.abs(modes[4*i+2]);maxModeL1=Math.max(maxModeL1,sum);const bound=n*sum;if(!Number.isSafeInteger(bound)||bound>0x7fffffff||sum>0x1000000)throw new RangeError(`mode ${i} overflows exact signed integer or f32 double-single phase arithmetic`);}
  let maxGap=0;for(let i=0;i<np;i++)for(let a=0;a<dim;a++){const k=4*i+a;maxGap=Math.max(maxGap,Math.abs((input.plusHi[k]+input.plusLo[k])-(input.minusHi[k]+input.minusLo[k])));}const maxHalfAngle=Math.PI*maxModeL1*maxGap;if(maxHalfAngle>0.005)throw new RangeError(`endpoint half-angle bound ${maxHalfAngle} exceeds the centered-difference quadrature limit 0.005`);
  if(4*maxModeL1*(maxDisplacement+1e-7)>0x1000000)throw new RangeError('endpoint displacement and mode magnitudes exceed exact f32 quarter-turn reduction range');
  const lim=device.limits??{};if((lim.maxComputeWorkgroupSizeX??32)<32||(lim.maxComputeInvocationsPerWorkgroup??32)<32)throw new RangeError('finite-difference kernels require a 32-lane workgroup');if((lim.maxStorageBuffersPerShaderStage??8)<8)throw new RangeError('finite-difference kernel requires 8 storage buffers');
  const maxBuffer=lim.maxBufferSize??Infinity,maxStorage=lim.maxStorageBufferBindingSize??Infinity;const cpc=32,ctotal=Math.ceil(np/cpc);const maxGroups=lim.maxComputeWorkgroupsPerDimension??65535;
  if(nm>maxGroups*maxGroups)throw new RangeError('mode count exceeds finite-difference dispatch limits');const requestedBatch=options.batchChunks??options.fdBatchChunks;if(requestedBatch!=null&&(!Number.isInteger(requestedBatch)||requestedBatch<1))throw new RangeError('batchChunks must be a positive integer');const maxChunks=Math.max(1,Math.min(ctotal,requestedBatch??256,Math.max(1,Math.floor(maxGroups*maxGroups/nm))));const batchCount=Math.ceil(ctotal/maxChunks);const cellBytes=nm*np*16*3,partialBytes=nm*ctotal*16*3,outputBytes=nm*16*3;if(!Number.isSafeInteger(cellBytes)||!Number.isSafeInteger(partialBytes)||!Number.isSafeInteger(outputBytes))throw new RangeError('finite-difference endpoint storage size exceeds safe integer range');const inputBytes=arrays.length*np*16+modes.byteLength;const totalBytes=inputBytes+cellBytes+partialBytes+outputBytes+16+48+32+16+Math.max(16,outputBytes)+16;
  const sizes=[['each endpoint displacement array',np*16],['mode records',modes.byteLength],['interleaved per-cell plus/minus/delta values',cellBytes],['interleaved endpoint partials',partialBytes],['interleaved endpoint results',outputBytes],['diagnostic counters',8],['evaluation parameters',48],['chunk-reduction parameters',32]];for(const [name,size] of sizes)if(size>maxBuffer||size>maxStorage)throw new RangeError(`${name} buffer needs ${size} bytes and exceeds device storage limits`);
  const maxGpuBytes=options.maxGpuBytes??512*1024*1024;if(totalBytes>maxGpuBytes)throw new RangeError(`estimated finite-difference allocation ${totalBytes} exceeds maxGpuBytes=${maxGpuBytes}`);
  return {np,nm,cpc,ctotal,maxChunks,batchCount,cellBytes,partialBytes,outputBytes,inputBytes,totalBytes,maxGroups,maxGpuBytes,maxGap,maxHalfAngle};
}

/** Evaluate centered endpoint values; returns f64 mode powers' centered derivative. */
export async function runFiniteDifference(device,input,options={}){
  const t0=performance.now();const v=validate(device,input,options);const {np,nm,cpc,ctotal,maxChunks,batchCount,cellBytes,partialBytes,outputBytes,inputBytes,totalBytes,maxGroups,maxGap,maxHalfAngle}=v;const {dim,n,modes,h}=input;const owned=[];let submitted=false;let compileMs=0,uploadMs=0,kernelMs=0,readbackMs=0;const push=device.pushErrorScope?.bind(device),pop=device.popErrorScope?.bind(device);let scopes=false;
  try{
    if(canceled(options))throw new DOMException('GPU finite-difference request canceled','AbortError');push?.('validation');push?.('out-of-memory');scopes=!!push&&!!pop;
    const tCompile=performance.now();const [endpointPipe,deltaPipe,chunkPipe,reducePipe]=await pipelines(device,dim);compileMs=performance.now()-tCompile;
    const inputBuffers={};for(const name of ['plusHi','plusLo','minusHi','minusLo'])inputBuffers[name]=alloc(device,`FD ${name}`,input[name].byteLength,0x80|0x04|0x08,owned);
    const modeBuf=alloc(device,'FD int4 modes',modes.byteLength,0x80|0x04|0x08,owned);const cellValues=alloc(device,'FD interleaved per-cell plus/minus/delta values',cellBytes,0x80|0x04|0x08,owned);const partialValues=alloc(device,'FD interleaved plus/minus/delta partials',partialBytes,0x80|0x04|0x08,owned);const outValues=alloc(device,'FD interleaved plus/minus/delta amplitudes',outputBytes,0x80|0x04|0x08,owned);const diag=alloc(device,'FD numerical diagnostics',8,0x80|0x04|0x08,owned);const uniform=alloc(device,'FD evaluation parameters',48,0x40|0x08,owned);const chunkUniform=alloc(device,'FD chunk-reduction parameters',32,0x40|0x08,owned);const redUniform=alloc(device,'FD reduction parameters',16,0x40|0x08,owned);
    const tUpload=performance.now();for(const name of Object.keys(inputBuffers))device.queue.writeBuffer(inputBuffers[name],0,input[name]);device.queue.writeBuffer(modeBuf,0,modes);device.queue.writeBuffer(diag,0,new Uint32Array([0,0]));uploadMs=performance.now()-tUpload;
    const endpointBG=device.createBindGroup({layout:endpointPipe.bgl,entries:[{binding:0,resource:{buffer:uniform}},...[inputBuffers.plusHi,inputBuffers.plusLo,inputBuffers.minusHi,inputBuffers.minusLo,modeBuf,cellValues,diag].map((buffer,i)=>({binding:i+1,resource:{buffer}}))]});const deltaBG=device.createBindGroup({layout:deltaPipe.bgl,entries:[{binding:0,resource:{buffer:uniform}},...[inputBuffers.plusHi,inputBuffers.plusLo,inputBuffers.minusHi,inputBuffers.minusLo,modeBuf,cellValues,diag].map((buffer,i)=>({binding:i+1,resource:{buffer}}))]});const chunkBG=device.createBindGroup({layout:chunkPipe.bgl,entries:[{binding:0,resource:{buffer:chunkUniform}},{binding:1,resource:{buffer:cellValues}},{binding:2,resource:{buffer:partialValues}}]});
    const maxWG=maxGroups;const tKernel=performance.now();
    for(let batch=0;batch<batchCount;batch++){
      if(canceled(options))throw new DOMException('GPU finite-difference request canceled','AbortError');const chunk0=batch*maxChunks,chunk1=Math.min(ctotal,chunk0+maxChunks);const raw=new ArrayBuffer(48),u=new Uint32Array(raw),f=new Float32Array(raw);u[0]=dim;u[1]=n;u[2]=np;u[3]=nm;u[4]=cpc;u[5]=ctotal;u[6]=chunk0;u[7]=chunk1;f[8]=1/n;f[9]=1/np;device.queue.writeBuffer(uniform,0,raw);device.queue.writeBuffer(chunkUniform,0,new Uint32Array([np,nm,cpc,ctotal,chunk0,chunk1,0,0]));
      const count=(chunk1-chunk0)*nm,gx=Math.min(count,maxWG),gy=Math.ceil(count/gx);if(gy>maxWG)throw new RangeError('finite-difference batch exceeds dispatch dimensions');
      const endpointEncoder=device.createCommandEncoder({label:`FD endpoint integral batch ${batch+1}/${batchCount}`});const endpointPass=endpointEncoder.beginComputePass({label:'FD paired P0 endpoint integrals'});endpointPass.setPipeline(endpointPipe.pipeline);endpointPass.setBindGroup(0,endpointBG);endpointPass.dispatchWorkgroups(gx,gy);endpointPass.end();device.queue.submit([endpointEncoder.finish()]);submitted=true;await device.queue.onSubmittedWorkDone();
      if(canceled(options))throw new DOMException('GPU finite-difference request canceled','AbortError');const deltaEncoder=device.createCommandEncoder({label:`FD stable delta batch ${batch+1}/${batchCount}`});const deltaPass=deltaEncoder.beginComputePass({label:'FD stable endpoint amplitude delta'});deltaPass.setPipeline(deltaPipe.pipeline);deltaPass.setBindGroup(0,deltaBG);deltaPass.dispatchWorkgroups(gx,gy);deltaPass.end();device.queue.submit([deltaEncoder.finish()]);await device.queue.onSubmittedWorkDone();
      if(canceled(options))throw new DOMException('GPU finite-difference request canceled','AbortError');const reduceEncoder=device.createCommandEncoder({label:`FD chunk reduction batch ${batch+1}/${batchCount}`});const reducePass=reduceEncoder.beginComputePass({label:'FD deterministic per-chunk endpoint reduction'});reducePass.setPipeline(chunkPipe.pipeline);reducePass.setBindGroup(0,chunkBG);reducePass.dispatchWorkgroups(gx,gy);reducePass.end();device.queue.submit([reduceEncoder.finish()]);await device.queue.onSubmittedWorkDone();await options.onProgress?.({stage:'endpoint-integrals',batch:batch+1,batches:batchCount});await new Promise(resolve=>setTimeout(resolve,0));
    }
    kernelMs=performance.now()-tKernel;if(canceled(options))throw new DOMException('GPU finite-difference request canceled','AbortError');
    device.queue.writeBuffer(redUniform,0,new Uint32Array([nm,ctotal,0,0]));const redBG=device.createBindGroup({layout:reducePipe.bgl,entries:[{binding:0,resource:{buffer:redUniform}},{binding:1,resource:{buffer:partialValues}},{binding:2,resource:{buffer:outValues}}]});const enc=device.createCommandEncoder({label:'FD deterministic endpoint reduction'});const pass=enc.beginComputePass({label:'FD double-single endpoint reduction'});pass.setPipeline(reducePipe.pipeline);pass.setBindGroup(0,redBG);pass.dispatchWorkgroups(Math.min(nm,maxGroups),Math.ceil(nm/Math.min(nm,maxGroups)));pass.end();device.queue.submit([enc.finish()]);submitted=true;
    const tr=performance.now();const [raw,diagRaw]=await Promise.all([readBuffer(device,outValues,outputBytes),readBuffer(device,diag,8,Uint32Array)]);readbackMs=performance.now()-tr;const flags=diagRaw[0],invalidCount=diagRaw[1];if(flags||invalidCount)throw new Error(`GPU centered finite-difference diagnostics failed (flags=0x${flags.toString(16)}, invalidCount=${invalidCount})`);
    const plus=new Float64Array(2*nm),minus=new Float64Array(2*nm),delta=new Float64Array(2*nm),values=new Float64Array(nm);for(let i=0;i<nm;i++){const k=12*i;plus[2*i]=raw[k]+raw[k+1];plus[2*i+1]=raw[k+2]+raw[k+3];minus[2*i]=raw[k+4]+raw[k+5];minus[2*i+1]=raw[k+6]+raw[k+7];delta[2*i]=raw[k+8]+raw[k+9];delta[2*i+1]=raw[k+10]+raw[k+11];const sr=plus[2*i]+minus[2*i],si=plus[2*i+1]+minus[2*i+1];values[i]=(delta[2*i]*sr+delta[2*i+1]*si)/(2*h);if(!Number.isFinite(values[i])||![plus[2*i],plus[2*i+1],minus[2*i],minus[2*i+1],delta[2*i],delta[2*i+1]].every(Number.isFinite))throw new Error(`GPU centered finite-difference output ${i} is nonfinite`);}
    const error=[];if(pop){for(const e of [await pop(),await pop()])if(e)error.push(e.message??String(e));scopes=false;}if(error.length)throw new Error(`WebGPU finite-difference error scope: ${error.join('; ')}`);
    const shaderHash=await hash(shaderFor(dim)),reductionShaderHash=await hash(`${CHUNK_REDUCE}\n${REDUCE}`),provenance={...FINITE_DIFFERENCE_PROVENANCE,shaderHash,reductionShaderHash};return {values,plus,minus,delta,provenance,diagnostics:{dim,n,modeCount:nm,cells:np,endpointIntegrals:2*np*(dim===2?2:6),endpointDeltaQuadrature:'3-point Gauss-Legendre, exact continuous directional derivative along retained endpoints',maxComponentGap:maxGap,maxHalfAngle,halfAngleLimit:0.005,h,estimatedBytes:totalBytes,inputBytes,perCellEndpointBytes:cellBytes,batches:batchCount,partialChunks:ctotal,flags,invalidCount,shaderHash,reductionShaderHash,provenance},timings:{compileMs,uploadMs,kernelMs,readbackMs,totalMs:performance.now()-t0}};
  }catch(error){if(submitted){try{await device.queue.onSubmittedWorkDone();}catch{}}if(pop&&scopes){try{await pop();await pop();}catch{}}throw error;}finally{for(const b of owned){try{b.destroy();}catch{}}}
}
