// GPU-resident periodic particle-mesh KDK evolution and its analytic JVP.
// CIC fields are accumulated by per-mesh-cell linked lists. Integer atomics
// build the lists; one invocation owns each cell's floating-point sum, avoiding
// fixed-point CIC quantization and floating-point write races.
import { GpuFFT, FFT_SHADER_SOURCE } from './fft.js';
import { makePipeline, dispatch1d, readBuffer } from './util.js';

export const PM_KERNEL_PROVENANCE = Object.freeze({
  revision: 'precrossedsheet-gpu-pm-p1-2026-10-11',
  forceKernel: 'rust-spectral-cic-v1',
  accumulation: 'integer atomic linked-list construction; serial f32 gather per mesh cell; insertion order is device-scheduling dependent',
  precision: 'f32 particle state, CIC sums, and complex FFT; clocks are uploaded as f32',
  fft: 'Stockham radix-2, normalized inverse; see fft.js provenance',
});

const SHADERS = /* wgsl */`
struct ClearP { count:u32, pad0:u32, pad1:u32, pad2:u32 };
@group(0) @binding(0) var<uniform> clearP:ClearP;
@group(0) @binding(1) var<storage,read_write> heads:array<atomic<u32>>;
@compute @workgroup_size(256) fn clear_heads(@builtin(global_invocation_id) gid:vec3<u32>,@builtin(num_workgroups) nwg:vec3<u32>){let i=gid.y*(nwg.x*256u)+gid.x;if(i<clearP.count){atomicStore(&heads[i],0xffffffffu);}}

struct LinkP { np:u32, n:u32, dim:u32, corners:u32, refs:u32, sheetN:u32, invdx:f32, boxSize:f32, usePsi:u32, pad2:u32, pad3:u32, pad4:u32 };
@group(0) @binding(0) var<uniform> linkP:LinkP;
@group(0) @binding(1) var<storage,read> x:array<f32>;
@group(0) @binding(2) var<storage,read_write> head:array<atomic<u32>>;
@group(0) @binding(3) var<storage,read_write> next:array<u32>;
fn wrapi(a:i32,n:u32)->u32{return u32((a%i32(n)+i32(n))%i32(n));}
fn flatIndex(x:u32,y:u32,z:u32,n:u32,d:u32)->u32{if(d==1u){return x;}if(d==2u){return x*n+y;}return (x*n+y)*n+z;}
fn coords(i:u32,n:u32,d:u32)->vec3<u32>{if(d==1u){return vec3<u32>(i,0u,0u);}if(d==2u){return vec3<u32>(i/n,i%n,0u);}return vec3<u32>(i/(n*n),(i/n)%n,i%n);}
fn positionCell(pos:f32,latticeCoord:u32,nSheet:u32,nMesh:u32,invdx:f32,usePsi:u32)->vec2<f32>{if(usePsi!=0u){let numerator=latticeCoord*nMesh;let base=numerator/nSheet;let rem=numerator%nSheet;let shifted=f32(rem)/f32(nSheet)+pos*invdx;let delta=floor(shifted);return vec2<f32>(f32(wrapi(i32(base)+i32(delta),nMesh)),shifted-delta);}let u=pos*invdx;let cell=floor(u);return vec2<f32>(f32(wrapi(i32(cell),nMesh)),u-cell);}
@compute @workgroup_size(256) fn build_links(@builtin(global_invocation_id) gid:vec3<u32>,@builtin(num_workgroups) nwg:vec3<u32>){
 let i=gid.y*(nwg.x*256u)+gid.x;if(i>=linkP.np){return;}let d=linkP.dim;let n=linkP.n;let sheet=coords(i,linkP.sheetN,d);
 let cx=positionCell(x[i*4u],sheet.x,linkP.sheetN,n,linkP.invdx,linkP.usePsi);let cy=positionCell(x[i*4u+1u],sheet.y,linkP.sheetN,n,linkP.invdx,linkP.usePsi);let cz=positionCell(x[i*4u+2u],sheet.z,linkP.sheetN,n,linkP.invdx,linkP.usePsi);let ix=u32(cx.x);let iy=u32(cy.x);let iz=u32(cz.x);
 for(var c=0u;c<linkP.corners;c++){let bx=c&1u;let by=(c>>1u)&1u;let bz=(c>>2u)&1u;let node=flatIndex(wrapi(i32(ix)+i32(bx),n),wrapi(i32(iy)+i32(by),n),wrapi(i32(iz)+i32(bz),n),n,d);let linkId=i*8u+c;let old=atomicExchange(&head[node],linkId);next[linkId]=old;}
}

struct MaterializeP { np:u32, sheetN:u32, dim:u32, usePsi:u32, boxSize:f32, pad0:f32, pad1:u32, pad2:u32 };
@group(0) @binding(0) var<uniform> matP:MaterializeP;
@group(0) @binding(1) var<storage,read> matX:array<f32>;
@group(0) @binding(2) var<storage,read_write> matOut:array<f32>;
fn matCoords(i:u32,n:u32,d:u32)->vec3<u32>{if(d==1u){return vec3<u32>(i,0u,0u);}if(d==2u){return vec3<u32>(i/n,i%n,0u);}return vec3<u32>(i/(n*n),(i/n)%n,i%n);}
@compute @workgroup_size(256) fn materialize_x(@builtin(global_invocation_id) gid:vec3<u32>,@builtin(num_workgroups) nwg:vec3<u32>){let i=gid.y*(nwg.x*256u)+gid.x;if(i>=matP.np){return;}let q=matCoords(i,matP.sheetN,matP.dim);for(var a=0u;a<matP.dim;a++){let j=i*4u+a;let absolute=select(matX[j],f32(q[a])*matP.boxSize/f32(matP.sheetN)+matX[j],matP.usePsi!=0u);matOut[j]=absolute;}for(var a=matP.dim;a<4u;a++){matOut[i*4u+a]=matX[i*4u+a];}}

struct CellP { ncell:u32, n:u32, dim:u32, corners:u32, refs:u32, hasJvp:u32, invdx:f32, massNorm:f32, sheetN:u32, usePsi:u32, pad2:u32, pad3:u32 };
@group(0) @binding(0) var<uniform> cellP:CellP;
@group(0) @binding(1) var<storage,read_write> cellHead:array<atomic<u32>>;
@group(0) @binding(2) var<storage,read> cellNext:array<u32>;
@group(0) @binding(3) var<storage,read> cellX:array<f32>;
@group(0) @binding(4) var<storage,read> cellDx:array<f32>;
@group(0) @binding(5) var<storage,read_write> rho:array<f32>;
@group(0) @binding(6) var<storage,read_write> drho:array<f32>;
@compute @workgroup_size(256) fn gather_cells(@builtin(global_invocation_id) gid:vec3<u32>,@builtin(num_workgroups) nwg:vec3<u32>){
 let node=gid.y*(nwg.x*256u)+gid.x;if(node>=cellP.ncell){return;}let n=cellP.n;let d=cellP.dim;let cxyz=coords(node,n,d);let ix=cxyz.x;let iy=cxyz.y;let iz=cxyz.z;var sr=0.0;var sd=0.0;var linkId=atomicLoad(&cellHead[node]);
 loop{if(linkId==0xffffffffu){break;}let i=linkId/8u;let c=linkId&7u;let sheet=coords(i,cellP.sheetN,d);let cx=positionCell(cellX[i*4u],sheet.x,cellP.sheetN,n,cellP.invdx,cellP.usePsi);let cy=positionCell(cellX[i*4u+1u],sheet.y,cellP.sheetN,n,cellP.invdx,cellP.usePsi);let cz=positionCell(cellX[i*4u+2u],sheet.z,cellP.sheetN,n,cellP.invdx,cellP.usePsi);let fx=cx.y;let fy=select(0.0,cy.y,d>1u);let fz=select(0.0,cz.y,d>2u);let bx=c&1u;let by=(c>>1u)&1u;let bz=(c>>2u)&1u;let px=wrapi(i32(u32(cx.x))+i32(bx),n);let py=wrapi(i32(u32(cy.x))+i32(by),n);let pz=wrapi(i32(u32(cz.x))+i32(bz),n);if(px==ix&&py==iy&&pz==iz){let wx=select(1.0-fx,fx,bx==1u);let wy=select(1.0-fy,fy,by==1u);let wz=select(1.0-fz,fz,bz==1u);let w=wx*wy*wz;sr+=w;if(cellP.hasJvp!=0u){let vx=cellDx[i*4u]*cellP.invdx;let vy=select(0.0,cellDx[i*4u+1u]*cellP.invdx,d>1u);let vz=select(0.0,cellDx[i*4u+2u]*cellP.invdx,d>2u);let sx=select(-1.0,1.0,bx==1u);let sy=select(-1.0,1.0,by==1u);let sz=select(-1.0,1.0,bz==1u);sd+=(sx*vx*wy*wz+sy*vy*wx*wz+sz*vz*wx*wy);}}
 linkId=cellNext[linkId];}
 rho[node]=sr*cellP.massNorm;drho[node]=sd*cellP.massNorm;
}

struct SpectralP { n:u32, dim:u32, axis:u32, size:u32, boxSize:f32, omega:f32, pad0:u32, pad1:u32 };
@group(0) @binding(0) var<uniform> sp:SpectralP;
@group(0) @binding(1) var<storage,read> fieldHat:array<vec2<f32>>;
@group(0) @binding(2) var<storage,read_write> forceHat:array<vec2<f32>>;
@compute @workgroup_size(256) fn apply_force(@builtin(global_invocation_id) gid:vec3<u32>,@builtin(num_workgroups) nwg:vec3<u32>){let i=gid.y*(nwg.x*256u)+gid.x;if(i>=sp.size){return;}let n=sp.n;let xyz=coords(i,n,sp.dim);let ix=xyz.x;let iy=xyz.y;let iz=xyz.z;let q=select(select(iz,iy,sp.axis==1u),ix,sp.axis==0u);let signed=select(i32(q),i32(q)-i32(n),q>=n/2u);let k1x=select(f32(i32(ix)),f32(i32(ix)-i32(n)),ix>=n/2u);let k1y=select(f32(i32(iy)),f32(i32(iy)-i32(n)),iy>=n/2u);let k1z=select(f32(i32(iz)),f32(i32(iz)-i32(n)),iz>=n/2u);let kf=6.283185307179586/sp.boxSize;let k2=kf*kf*(k1x*k1x+k1y*k1y+k1z*k1z);let kd=select(f32(signed)*kf,0.0,(n&1u)==0u&&q==n/2u);if(k2==0.0){forceHat[i]=vec2<f32>(0.0);}else{let c=1.5*sp.omega*kd/k2;let a=fieldHat[i];forceHat[i]=vec2<f32>(-c*a.y,c*a.x);}}

struct PackAxisP { count:u32, axis:u32, pad0:u32, pad1:u32 };
@group(0) @binding(0) var<uniform> ap:PackAxisP;
@group(0) @binding(1) var<storage,read> scalar:array<vec2<f32>>;
@group(0) @binding(2) var<storage,read_write> vector:array<vec4<f32>>;
@compute @workgroup_size(256) fn pack_axis(@builtin(global_invocation_id) gid:vec3<u32>,@builtin(num_workgroups) nwg:vec3<u32>){let i=gid.y*(nwg.x*256u)+gid.x;if(i<ap.count){var v=vector[i];if(ap.axis==0u){v=vec4<f32>(0.0);}v[ap.axis]=scalar[i].x;vector[i]=v;}}

struct GatherP { np:u32, n:u32, dim:u32, corners:u32, hasJvp:u32, sheetN:u32, usePsi:u32, pad0:u32, invdx:f32, pad1:f32, pad2:u32, pad3:u32 };
@group(0) @binding(0) var<uniform> gp:GatherP;
@group(0) @binding(1) var<storage,read> gx:array<f32>;
@group(0) @binding(2) var<storage,read> gdx:array<f32>;
@group(0) @binding(3) var<storage,read> gf:array<vec4<f32>>;
@group(0) @binding(4) var<storage,read> gdf:array<vec4<f32>>;
@group(0) @binding(5) var<storage,read_write> outF:array<f32>;
@group(0) @binding(6) var<storage,read_write> outDF:array<f32>;
@compute @workgroup_size(256) fn gather_particles(@builtin(global_invocation_id) gid:vec3<u32>,@builtin(num_workgroups) nwg:vec3<u32>){let i=gid.y*(nwg.x*256u)+gid.x;if(i>=gp.np){return;}let d=gp.dim;let n=gp.n;let sheet=coords(i,gp.sheetN,d);let cx=positionCell(gx[i*4u],sheet.x,gp.sheetN,n,gp.invdx,gp.usePsi);let cy=positionCell(gx[i*4u+1u],sheet.y,gp.sheetN,n,gp.invdx,gp.usePsi);let cz=positionCell(gx[i*4u+2u],sheet.z,gp.sheetN,n,gp.invdx,gp.usePsi);let ix=u32(cx.x);let iy=u32(cy.x);let iz=u32(cz.x);let fx=cx.y;let fy=select(0.0,cy.y,d>1u);let fz=select(0.0,cz.y,d>2u);let vx=select(0.0,gdx[i*4u]*gp.invdx,gp.hasJvp!=0u);let vy=select(0.0,gdx[i*4u+1u]*gp.invdx,d>1u&&gp.hasJvp!=0u);let vz=select(0.0,gdx[i*4u+2u]*gp.invdx,d>2u&&gp.hasJvp!=0u);var f=vec3<f32>(0.0);var df=f;
 for(var c=0u;c<gp.corners;c++){let bx=c&1u;let by=(c>>1u)&1u;let bz=(c>>2u)&1u;let wx=select(1.0-fx,fx,bx==1u);let wy=select(1.0-fy,fy,by==1u);let wz=select(1.0-fz,fz,bz==1u);let w=wx*wy*wz;let sx=select(-1.0,1.0,bx==1u);let sy=select(-1.0,1.0,by==1u);let sz=select(-1.0,1.0,bz==1u);let dw=sx*vx*wy*wz+sy*vy*wx*wz+sz*vz*wx*wy;let node=flatIndex(wrapi(i32(ix)+i32(bx),n),wrapi(i32(iy)+i32(by),n),wrapi(i32(iz)+i32(bz),n),n,d);let ff=gf[node].xyz;f+=w*ff;if(gp.hasJvp!=0u){df+=w*gdf[node].xyz+dw*ff;}}
 for(var a=0u;a<d;a++){outF[i*4u+a]=f[a];outDF[i*4u+a]=df[a];}}

struct StepP { np:u32, dim:u32, hasJvp:u32, pad0:u32, clock:f32, pad1:f32, pad2:u32, pad3:u32 };
@group(0) @binding(0) var<uniform> kp:StepP;
@group(0) @binding(1) var<storage,read_write> kx:array<f32>;
@group(0) @binding(2) var<storage,read_write> kpm:array<f32>;
@group(0) @binding(3) var<storage,read_write> kdx:array<f32>;
@group(0) @binding(4) var<storage,read_write> kdp:array<f32>;
@group(0) @binding(5) var<storage,read> kf:array<f32>;
@group(0) @binding(6) var<storage,read> kdf:array<f32>;
@compute @workgroup_size(256) fn kick_drift(@builtin(global_invocation_id) gid:vec3<u32>,@builtin(num_workgroups) nwg:vec3<u32>){let i=gid.y*(nwg.x*256u)+gid.x;if(i>=kp.np){return;}let d=kp.dim;for(var a=0u;a<d;a++){let j=i*4u+a;kpm[j]+=kp.clock*kf[j];if(kp.hasJvp!=0u){kdp[j]+=kp.clock*kdf[j];}}}
@compute @workgroup_size(256) fn drift_state(@builtin(global_invocation_id) gid:vec3<u32>,@builtin(num_workgroups) nwg:vec3<u32>){let i=gid.y*(nwg.x*256u)+gid.x;if(i>=kp.np){return;}let d=kp.dim;for(var a=0u;a<d;a++){let j=i*4u+a;kx[j]+=kp.clock*kpm[j];if(kp.hasJvp!=0u){kdx[j]+=kp.clock*kdp[j];}}}

struct DiagP { n:u32, dim:u32, count:u32, pad0:u32, boxSize:f32, pad1:f32, flagIndex:u32, pad2:u32 };
@group(0) @binding(0) var<uniform> diagP:DiagP;
@group(0) @binding(1) var<storage,read> startX:array<f32>;
@group(0) @binding(2) var<storage,read> endX:array<f32>;
@group(0) @binding(3) var<storage,read_write> diagPartials:array<f32>;
@group(0) @binding(4) var<storage,read_write> diagFlags:array<atomic<u32>>;
fn finitef(v:f32)->bool{return v==v&&abs(v)<=3.0e38;}
fn invalidMinimum(reason:u32)->f32{atomicOr(&diagFlags[diagP.flagIndex],reason);return -1.0e30;}
fn detRows(a:vec3<f32>,b:vec3<f32>,c:vec3<f32>,d:u32)->f32{if(d==1u){return a.x;}if(d==2u){return a.x*b.y-a.y*b.x;}return a.x*(b.y*c.z-b.z*c.y)-a.y*(b.x*c.z-b.z*c.x)+a.z*(b.x*c.y-b.y*c.x);}
fn poly(c:vec4<f32>,t:f32)->f32{return c.x+t*(c.y+t*(c.z+t*c.w));}
fn checkedPoly(c:vec4<f32>,t:f32)->f32{let v=poly(c,t);if(!finitef(v)){return invalidMinimum(4u);}return v;}
fn simplexVertex(s:u32,j:u32,d:u32)->u32{
 if(d==1u){return j;}
 if(d==2u){if(s==0u){if(j==0u){return 0u;}if(j==1u){return 1u;}return 3u;}if(j==0u){return 0u;}if(j==1u){return 3u;}return 2u;}
 if(s==0u){if(j==0u){return 0u;}if(j==1u){return 1u;}if(j==2u){return 3u;}return 7u;}
 if(s==1u){if(j==0u){return 0u;}if(j==1u){return 1u;}if(j==2u){return 5u;}return 7u;}
 if(s==2u){if(j==0u){return 0u;}if(j==1u){return 2u;}if(j==2u){return 3u;}return 7u;}
 if(s==3u){if(j==0u){return 0u;}if(j==1u){return 2u;}if(j==2u){return 6u;}return 7u;}
 if(s==4u){if(j==0u){return 0u;}if(j==1u){return 4u;}if(j==2u){return 5u;}return 7u;}
 if(j==0u){return 0u;}if(j==1u){return 4u;}if(j==2u){return 6u;}return 7u;
}
fn vertexPosition(cell:u32,b:u32,isEnd:bool)->vec3<f32>{let n=diagP.n;let d=diagP.dim;let base=coords(cell,n,d);let rx=base.x+(b&1u);let ry=base.y+((b>>1u)&1u);let rz=base.z+((b>>2u)&1u);let id=flatIndex(rx%n,ry%n,rz%n,n,d);let lift=vec3<f32>(select(0.0,diagP.boxSize,rx>=n),select(0.0,diagP.boxSize,ry>=n),select(0.0,diagP.boxSize,rz>=n));var pos:vec3<f32>;if(isEnd){pos=vec3<f32>(endX[id*4u],endX[id*4u+1u],endX[id*4u+2u]);}else{pos=vec3<f32>(startX[id*4u],startX[id*4u+1u],startX[id*4u+2u]);}return pos+lift;}
fn simplexMinimum(cell:u32)->f32{
 let n=diagP.n;let d=diagP.dim;let ns=select(select(1u,2u,d==2u),6u,d==3u);var best=3.0e38;let dx=diagP.boxSize/f32(n);var baseDenominator=dx;if(d>1u){baseDenominator*=dx;}if(d>2u){baseDenominator*=dx;}
 for(var s=0u;s<ns;s++){
  let id0=simplexVertex(s,0u,d);let id1=simplexVertex(s,1u,d);let id2=simplexVertex(s,2u,d);let id3=simplexVertex(s,3u,d);
  let p00=vertexPosition(cell,id0,false);let p01=vertexPosition(cell,id1,false);let p02=vertexPosition(cell,id2,false);let p03=vertexPosition(cell,id3,false);
  let q00=vertexPosition(cell,id0,true);let q01=vertexPosition(cell,id1,true);let q02=vertexPosition(cell,id2,true);let q03=vertexPosition(cell,id3,true);
  let e0=p01-p00;let e1=p02-p00;let e2=p03-p00;let de0=(q01-p01)-(q00-p00);let de1=(q02-p02)-(q00-p00);let de2=(q03-p03)-(q00-p00);
  var denominator=baseDenominator;if(d==3u&&(s==1u||s==2u||s==5u)){denominator=-denominator;}if(!finitef(denominator)||denominator==0.0){return invalidMinimum(1u);}
  var c0=detRows(e0,e1,e2,d)/denominator;var c1=0.0;var c2=0.0;var c3=0.0;
  if(d==1u){c1=detRows(de0,e1,e2,d)/denominator;}
  if(d==2u){c1=(detRows(de0,e1,e2,d)+detRows(e0,de1,e2,d))/denominator;c2=detRows(de0,de1,e2,d)/denominator;}
  if(d==3u){c1=(detRows(de0,e1,e2,d)+detRows(e0,de1,e2,d)+detRows(e0,e1,de2,d))/denominator;c2=(detRows(de0,de1,e2,d)+detRows(de0,e1,de2,d)+detRows(e0,de1,de2,d))/denominator;c3=detRows(de0,de1,de2,d)/denominator;}
  if(!finitef(c0)||!finitef(c1)||!finitef(c2)||!finitef(c3)){return invalidMinimum(2u);}let coef=vec4<f32>(c0,c1,c2,c3);best=min(best,checkedPoly(coef,0.0));best=min(best,checkedPoly(coef,1.0));
  if(d==2u&&abs(c2)>1.0e-20){let t=-c1/(2.0*c2);if(t>0.0&&t<1.0){best=min(best,checkedPoly(coef,t));}}
  if(d==3u){let aa=3.0*c3;let bb=2.0*c2;let cc=c1;if(abs(aa)<=1.0e-14*max(max(abs(bb),abs(cc)),1.0e-30)){if(bb!=0.0){let t=-cc/bb;if(t>0.0&&t<1.0){best=min(best,checkedPoly(coef,t));}}}else{let disc=bb*bb-4.0*aa*cc;if(!finitef(disc)){return invalidMinimum(8u);}if(disc>=0.0){let root=sqrt(disc);let t0=(-bb-root)/(2.0*aa);let t1=(-bb+root)/(2.0*aa);if(t0>0.0&&t0<1.0){best=min(best,checkedPoly(coef,t0));}if(t1>0.0&&t1<1.0){best=min(best,checkedPoly(coef,t1));}}}}
 }
 if(!finitef(best)){return invalidMinimum(16u);}return best;
}
var<workgroup> dmins:array<f32,256>;
@compute @workgroup_size(256) fn drift_min(@builtin(global_invocation_id) gid:vec3<u32>,@builtin(local_invocation_index) lane:u32,@builtin(num_workgroups) nwg:vec3<u32>,@builtin(workgroup_id) wg:vec3<u32>){let i=gid.y*(nwg.x*256u)+gid.x;var v=3.0e38;if(i<diagP.count){v=simplexMinimum(i);if(!finitef(v)){atomicOr(&diagFlags[diagP.flagIndex],32u);v=-1.0e30;}}dmins[lane]=v;workgroupBarrier();for(var stride=128u;stride>0u;stride>>=1u){if(lane<stride){dmins[lane]=min(dmins[lane],dmins[lane+stride]);}workgroupBarrier();}if(lane==0u){diagPartials[wg.y*nwg.x+wg.x]=dmins[0];}}
struct ReduceP{count:u32,pad0:u32,pad1:u32,pad2:u32};@group(0) @binding(0) var<uniform> redP:ReduceP;@group(0) @binding(1) var<storage,read> redIn:array<f32>;@group(0) @binding(2) var<storage,read_write> redOut:array<f32>;var<workgroup> redVals:array<f32,256>;
@compute @workgroup_size(256) fn reduce_min(@builtin(local_invocation_index) lane:u32){var v=3.0e38;var i=lane;loop{if(i>=redP.count){break;}v=min(v,redIn[i]);i+=256u;}redVals[lane]=v;workgroupBarrier();for(var stride=128u;stride>0u;stride>>=1u){if(lane<stride){redVals[lane]=min(redVals[lane],redVals[lane+stride]);}workgroupBarrier();}if(lane==0u){redOut[0]=redVals[0];}}
`;

const PARAM_SLOT=256;
class ParamSlots {
  constructor(device,count){this.device=device;this.count=Math.max(256,count);this.index=0;this.buffer=device.createBuffer({label:'PM dynamic parameters',size:this.count*PARAM_SLOT,usage:0x40|0x08});}
  write(words,types){if(this.index>=this.count)throw new RangeError('PM parameter ring exhausted; reduce step count or split the evolution');const slot=this.index++,raw=new ArrayBuffer(PARAM_SLOT),u=new Uint32Array(raw),f=new Float32Array(raw);for(let i=0;i<words.length;i++)(types[i]==='f'?f:u)[i]=words[i];this.device.queue.writeBuffer(this.buffer,slot*PARAM_SLOT,raw,0,Math.max(16,words.length*4));return slot*PARAM_SLOT;}
  resource(size=48){return {buffer:this.buffer,size};}
  destroy(){this.buffer.destroy();}
}
const PIPELINES=new WeakMap();
async function getPipelines(device){let p=PIPELINES.get(device);if(!p){const u=(binding,size=48)=>({binding,visibility:4,buffer:{type:'uniform',hasDynamicOffset:true,minBindingSize:size}});const ro=(binding)=>({binding,visibility:4,buffer:{type:'read-only-storage'}});const rw=(binding)=>({binding,visibility:4,buffer:{type:'storage'}});const mk=(entry,layout,label)=>makePipeline(device,SHADERS,entry,layout,label);p=Promise.all([
  mk('clear_heads',[u(0,16),rw(1)],'PM clear CIC heads'),mk('build_links',[u(0,48),ro(1),rw(2),rw(3)],'PM build CIC lists'),mk('gather_cells',[u(0,48),rw(1),ro(2),ro(3),ro(4),rw(5),rw(6)],'PM CIC density and JVP'),mk('apply_force',[u(0,32),ro(1),rw(2)],'PM spectral force'),mk('pack_axis',[u(0,16),ro(1),rw(2)],'PM pack force axis'),mk('gather_particles',[u(0,48),ro(1),ro(2),ro(3),ro(4),rw(5),rw(6)],'PM CIC force gather and JVP'),mk('kick_drift',[u(0,32),rw(1),rw(2),rw(3),rw(4),ro(5),ro(6)],'PM canonical momentum kick'),mk('drift_state',[u(0,32),rw(1),rw(2),rw(3),rw(4),ro(5),ro(6)],'PM canonical position drift'),mk('drift_min',[u(0,32),ro(1),ro(2),rw(3),rw(4)],'PM continuous simplex minimum'),mk('reduce_min',[u(0,16),ro(1),rw(2)],'PM simplex minimum reduction'),mk('materialize_x',[u(0,32),ro(1),rw(2)],'PM absolute position materialization')]);PIPELINES.set(device,p);}return p;}

function makeBuffer(device,label,bytes,owned){const b=device.createBuffer({label,size:Math.max(16,(bytes+3)&~3),usage:0x80|0x04|0x08});owned.push(b);return b;}
function bg(device,pipeline,params,offset,buffers,size=48){return {group:device.createBindGroup({layout:pipeline.bgl,entries:[{binding:0,resource:{buffer:params,size}},...buffers.map((buffer,i)=>({binding:i+1,resource:{buffer}}))]}),offset};}
function runPass(encoder,pipeline,binding,count){const pass=encoder.beginComputePass({label:pipeline.pipeline.label});pass.setPipeline(pipeline.pipeline);pass.setBindGroup(0,binding.group,[binding.offset]);dispatch1d(pass,count,256);pass.end();}
function canceled(options){return options.signal?.aborted||options.shouldCancel?.();}
function checkArray(name,a,length,required=true){if(a==null&&!required)return;if(!(a instanceof Float32Array)||a.length!==length)throw new TypeError(`${name} must be Float32Array with ${length} values`);for(let i=0;i<a.length;i++)if(!Number.isFinite(a[i]))throw new RangeError(`${name}[${i}] is nonfinite`);}
function clocks(a){return a instanceof Float32Array||a instanceof Float64Array;}
function validate(device,input,options){
  const {dim,n,meshN,boxSize,omegaM,x,p,dx,dp,clocks:times}=input??{};
  if(![1,2,3].includes(dim))throw new TypeError('runPm requires dim=1, 2, or 3');
  if(!Number.isInteger(n)||n<2||!Number.isInteger(meshN)||meshN<2)throw new TypeError('runPm requires integer sheet n and force meshN >= 2');
  if((meshN&(meshN-1))!==0)throw new RangeError(`PM FFT currently supports radix-2 force meshes; meshN=${meshN} requires the Rust CPU fallback`);
  if(input.positionRepresentation!=null&&!['absolute','lattice-displacement'].includes(input.positionRepresentation))throw new RangeError(`unsupported PM position representation: ${input.positionRepresentation}`);
  if(options.allowCrossing===false&&input.initialMinimum==null)throw new TypeError('allowCrossing:false requires the Rust f64 initialMinimum');
  if(options.checkAllDrifts&&typeof options.checkDrift!=='function')throw new TypeError('checkAllDrifts requires an async checkDrift(start,end,step) callback');
  if(input.forceKernel!=null&&input.forceKernel!=='rust-spectral-cic-v1')throw new RangeError(`unsupported PM force kernel: ${input.forceKernel}`);
  if(!(boxSize>0&&Number.isFinite(boxSize))||!(omegaM>0&&omegaM<=1&&Number.isFinite(omegaM)))throw new RangeError('runPm requires finite boxSize > 0 and 0 < omegaM <= 1');
  if(input.initialMinimum!=null&&!Number.isFinite(input.initialMinimum))throw new RangeError('initialMinimum must be finite when supplied');
  if(!(x instanceof Float32Array)||!x.length||x.length%4)throw new TypeError('x must be a packed vec4 Float32Array');const count=x.length;
  const np=n**dim;if(!Number.isSafeInteger(np)||count!==np*4)throw new RangeError('packed particle arrays must contain exactly n^dim vec4 records');
  checkArray('x',x,count);checkArray('p',p,count);const hasJvp=dx!=null||dp!=null;if(hasJvp){checkArray('dx',dx,count);checkArray('dp',dp,count);}if(!times||!clocks(times.kick0)||!clocks(times.drift)||!clocks(times.kick1)||!times.kick0.length||times.kick0.length!==times.drift.length||times.kick0.length!==times.kick1.length)throw new TypeError('clocks must contain equal-length nonempty kick0, drift, and kick1 typed arrays');for(const k of ['kick0','drift','kick1'])for(let i=0;i<times[k].length;i++)if(!Number.isFinite(times[k][i]))throw new RangeError(`clocks.${k}[${i}] is nonfinite`);
  const ncell=meshN**dim,corners=1<<dim,refs=np*8;if(ncell>0xffffffff||np>0xffffffff||refs>0xffffffff)throw new RangeError('PM exceeds 32-bit shader indexing');
  const limits=device.limits??{};const maxStorage=limits.maxStorageBufferBindingSize??Infinity;const maxBuffer=limits.maxBufferSize??Infinity;const groups=(size)=>Math.ceil(size/256);if((limits.maxComputeInvocationsPerWorkgroup??256)<256||(limits.maxComputeWorkgroupSizeX??256)<256)throw new RangeError('PM kernels require a 256-invocation workgroup');const fftLevels=Math.log2(meshN);const sizes=[['particle positions',count*4],['materialized absolute positions',count*4],['particle momenta',count*4],['position tangent',count*4],['momentum tangent',count*4],['CIC heads',ncell*4],['CIC links',refs*4],['density field',ncell*4],['density JVP',ncell*4],...Array.from({length:8},(_,i)=>[`complex field ${i+1}`,ncell*8]),['FFT scratch',ncell*8],['force vector field',ncell*16],['force JVP vector field',ncell*16],['particle force',count*4],['particle force JVP',count*4],['drift snapshot',count*4],['drift partial minima',Math.max(16,groups(np)*4)],['drift minimum',16],['drift diagnostic flags',Math.max(16,(times.kick0.length+1)*4)],...Array.from({length:6},(_,i)=>[`GPU readback staging ${i+1}`,count*4]),['PM parameters',Math.max(256,times.kick0.length*(10+4*dim)+7+4*dim+8)*PARAM_SLOT],['FFT twiddles',Math.max(16,meshN*4)],['FFT stage params',2*dim*fftLevels*PARAM_SLOT],['FFT simple params',16]];for(const [name,b] of sizes)if(b>maxStorage||b>maxBuffer)throw new RangeError(`${name} buffer needs ${b} bytes and exceeds device limits`);
  const paramSlots=times.kick0.length*(10+4*dim)+7+4*dim+8;const estimatedBytes=sizes.reduce((s,[,b])=>s+b,0);const maxGpuBytes=options.maxGpuBytes??512*1024*1024;if(estimatedBytes>maxGpuBytes)throw new RangeError(`estimated PM allocation ${estimatedBytes} bytes exceeds maxGpuBytes=${maxGpuBytes}`);
  return {n,meshN,np,count,ncell,corners,refs,steps:times.kick0.length,hasJvp,paramSlots,estimatedBytes,maxGpuBytes};
}

/**
 * Evolve a periodic PM state with Rust-supplied canonical KDK clocks and the
 * full discrete tangent. `x,p,dx,dp` are packed vec4 Float32Array values in
 * physical box units; `n` is the Lagrangian sheet side and `meshN` is force-grid side. The clocks
 * contain Rust's half-kick, full-drift, and half-kick integrals for every step.
 */
export async function runPm(device,input,options={}){
  const started=performance.now(),v=validate(device,input,options),{n,meshN,np,count,ncell,corners,refs,steps,hasJvp,paramSlots,estimatedBytes}=v,dim=input.dim,boxSize=input.boxSize,omegaM=input.omegaM,times=input.clocks,allowCrossing=options.allowCrossing??true;
  const compileStart=performance.now();const pipes=await getPipelines(device);let compileMs=performance.now()-compileStart;if(canceled(options))throw new DOMException('GPU PM request canceled','AbortError');
  const owned=[];let fft;const alloc=(label,bytes)=>makeBuffer(device,label,bytes,owned);const params=new ParamSlots(device,Math.max(256,paramSlots));owned.push(params.buffer);let xbuf,absXBuf,pbuf,dxbuf,dpbuf,heads,next,rho,drho,rhoC,drhoC,rhoHat,drhoHat,forceHat,dforceHat,forceC,dforceC,field,dfield,particleF,particleDF,stepStart,diagPartials,diagMinimum,diagFlags;
  const tUpload=performance.now();let uploadMs=0,kernelMs=0,diagnosticMs=0,cpuCheckMs=0,readbackMs=0;const queue=device.queue;let popErrorScope=null,scopesOpen=false;
  try{
    const push=device.pushErrorScope?.bind(device);popErrorScope=device.popErrorScope?.bind(device);push?.('validation');push?.('out-of-memory');scopesOpen=!!push&&!!popErrorScope;
    xbuf=alloc('PM particle positions',count*4);absXBuf=alloc('PM absolute particle positions',count*4);pbuf=alloc('PM canonical momenta',count*4);dxbuf=alloc('PM position tangent',count*4);dpbuf=alloc('PM momentum tangent',count*4);heads=alloc('PM CIC cell heads',ncell*4);next=alloc('PM CIC particle links',refs*4);rho=alloc('PM density',ncell*4);drho=alloc('PM density JVP',ncell*4);rhoC=alloc('PM density complex',ncell*8);drhoC=alloc('PM density JVP complex',ncell*8);rhoHat=alloc('PM density spectrum',ncell*8);drhoHat=alloc('PM density JVP spectrum',ncell*8);forceHat=alloc('PM force spectrum',ncell*8);dforceHat=alloc('PM force JVP spectrum',ncell*8);forceC=alloc('PM force complex',ncell*8);dforceC=alloc('PM force JVP complex',ncell*8);field=alloc('PM force vector field',ncell*16);dfield=alloc('PM force JVP vector field',ncell*16);particleF=alloc('PM particle force',count*4);particleDF=alloc('PM particle force JVP',count*4);stepStart=alloc('PM drift starting positions',count*4);diagPartials=alloc('PM drift minimum partials', Math.ceil(np/256)*4);diagMinimum=alloc('PM drift minimum scalar',4);diagFlags=alloc('PM drift diagnostic flags',(steps+1)*4);queue.writeBuffer(diagFlags,0,new Uint32Array(steps+1));
    const zeros=hasJvp?null:new Float32Array(count);queue.writeBuffer(xbuf,0,input.x);queue.writeBuffer(pbuf,0,input.p);queue.writeBuffer(dxbuf,0,input.dx??zeros);queue.writeBuffer(dpbuf,0,input.dp??zeros);uploadMs=performance.now()-tUpload;
    const fftCompileStart=performance.now();fft=new GpuFFT(device,meshN,dim);await fft.init();compileMs+=performance.now()-fftCompileStart;
  const groupCache=new Map(),resourceIds=new WeakMap();let nextResourceId=1;const resourceId=(b)=>{let id=resourceIds.get(b);if(!id){id=nextResourceId++;resourceIds.set(b,id);}return id;};
    const usePsi=input.positionRepresentation==='lattice-displacement';
    const dispatch=(pipe,words,types,buffers,encoder,size=48,countWork=ncell)=>{const key=`${pipe}:${buffers.map(resourceId).join(',')}`;let group=groupCache.get(key);if(!group){group=bg(device,pipes[pipe],params.buffer,0,buffers,size).group;groupCache.set(key,group);}const offset=params.write(words,types);runPass(encoder,pipes[pipe],{group,offset},countWork);};
    const fftPair=(source,dest,direction,encoder)=>fft.transform(source,dest,direction,encoder);
    const materialize=(encoder)=>dispatch(10,[np,n,dim,usePsi?1:0,boxSize,0,0,0],['u','u','u','u','f','f','u','u'],[xbuf,absXBuf],encoder,32,np);
    async function measureDrift(startBuffer,endBuffer,step,flagIndex){
      const start=performance.now();const enc=device.createCommandEncoder({label:`PM simplex drift check ${step}`});dispatch(8,[n,dim,np,0,boxSize,0,flagIndex,0],['u','u','u','u','f','f','u','u'],[startBuffer,endBuffer,diagPartials,diagFlags],enc,32,np);dispatch(9,[Math.ceil(np/256),0,0,0],['u','u','u','u'],[diagPartials,diagMinimum],enc,16,256);queue.submit([enc.finish()]);await queue.onSubmittedWorkDone();const a=await readBuffer(device,diagMinimum,4);diagnosticMs+=performance.now()-start;return a[0];
    }
    const initialEnc=device.createCommandEncoder({label:'PM initial absolute positions'});materialize(initialEnc);queue.submit([initialEnc.finish()]);await queue.onSubmittedWorkDone();const initialGpuMinimum=await measureDrift(absXBuf,absXBuf,0,0);const initialFlagBits=(await readBuffer(device,diagFlags,4,Uint32Array))[0];const initialGpuNonfinite=initialFlagBits!==0||!Number.isFinite(initialGpuMinimum)||initialGpuMinimum<=-1e30;const initialMinimum=input.initialMinimum==null?initialGpuMinimum:Number(input.initialMinimum);
    if(initialGpuNonfinite&&input.initialMinimum==null)throw new Error(`GPU PM initial simplex minimum is nonfinite (candidate ${initialGpuMinimum}, flags 0x${initialFlagBits.toString(16)}); CPU initial diagnostic is required`);
    if(initialMinimum<=0&&!allowCrossing)throw new Error('PM initial sheet has crossed');
    async function force(phase){
      if(canceled(options))throw new DOMException('GPU PM request canceled','AbortError');const forceStarted=performance.now();const enc=device.createCommandEncoder({label:`PM force ${phase}`});
      dispatch(0,[ncell],['u'],[heads],enc,16,ncell);
      dispatch(1,[np,meshN,dim,corners,refs,n,meshN/boxSize,boxSize,usePsi?1:0,0,0,0],['u','u','u','u','u','u','f','f','u','u','u','u'],[xbuf,heads,next],enc,48,np);
      dispatch(2,[ncell,meshN,dim,corners,refs,hasJvp?1:0,meshN/boxSize,ncell/np,n,usePsi?1:0,0,0],['u','u','u','u','u','u','f','f','u','u','u','u'],[heads,next,xbuf,dxbuf,rho,drho],enc,48,ncell);
      fft.packReal(rho,rhoC,enc);fft.packReal(drho,drhoC,enc);fftPair(rhoC,rhoHat,'forward',enc);fftPair(drhoC,drhoHat,'forward',enc);
      for(let axis=0;axis<dim;axis++){
        dispatch(3,[meshN,dim,axis,ncell,boxSize,omegaM,0,0],['u','u','u','u','f','f','u','u'],[rhoHat,forceHat],enc,32,ncell);
        dispatch(3,[meshN,dim,axis,ncell,boxSize,omegaM,0,0],['u','u','u','u','f','f','u','u'],[drhoHat,dforceHat],enc,32,ncell);
        fftPair(forceHat,forceC,'inverse',enc);fftPair(dforceHat,dforceC,'inverse',enc);
        dispatch(4,[ncell,axis,0,0],['u','u','u','u'],[forceC,field],enc,16,ncell);dispatch(4,[ncell,axis,0,0],['u','u','u','u'],[dforceC,dfield],enc,16,ncell);
      }
      dispatch(5,[np,meshN,dim,corners,hasJvp?1:0,n,usePsi?1:0,0,meshN/boxSize,0,0,0],['u','u','u','u','u','u','u','u','f','f','u','u'],[xbuf,dxbuf,field,dfield,particleF,particleDF],enc,48,np);
      queue.submit([enc.finish()]);await queue.onSubmittedWorkDone();kernelMs+=performance.now()-forceStarted;
    }
    await force('initial');const driftMinima=new Float64Array(steps),driftGpuMinima=new Float64Array(steps);let maxGpuCpuMinError=0;
    for(let s=0;s<steps;s++){
      if(canceled(options))throw new DOMException('GPU PM request canceled','AbortError');const enc=device.createCommandEncoder({label:`PM KDK step ${s+1}`});
      dispatch(6,[np,dim,hasJvp?1:0,0,times.kick0[s],0,0,0],['u','u','u','u','f','f','u','u'],[xbuf,pbuf,dxbuf,dpbuf,particleF,particleDF],enc,32,np);
      enc.copyBufferToBuffer(absXBuf,0,stepStart,0,count*4);
      dispatch(7,[np,dim,hasJvp?1:0,0,times.drift[s],0,0,0],['u','u','u','u','f','f','u','u'],[xbuf,pbuf,dxbuf,dpbuf,particleF,particleDF],enc,32,np);
      materialize(enc);dispatch(8,[n,dim,np,0,boxSize,0,s+1,0],['u','u','u','u','f','f','u','u'],[stepStart,absXBuf,diagPartials,diagFlags],enc,32,np);dispatch(9,[Math.ceil(np/256),0,0,0],['u','u','u','u'],[diagPartials,diagMinimum],enc,16,256);
      const kdkStarted=performance.now();queue.submit([enc.finish()]);await queue.onSubmittedWorkDone();kernelMs+=performance.now()-kdkStarted;
      const rawMinimum=(await readBuffer(device,diagMinimum,4))[0];let minimum=rawMinimum,cpuMinimum=null,checkedByCpu=false;const gpuFlag=(await readBuffer(device,diagFlags,(s+2)*4,Uint32Array))[s+1]!==0;const gpuNonfinite=gpuFlag||!Number.isFinite(rawMinimum)||rawMinimum<=-1e30;
      if(gpuNonfinite||rawMinimum<=1e-3||options.checkAllDrifts){if(typeof options.checkDrift==='function'){const [start,end]=await Promise.all([readBuffer(device,stepStart,count*4),readBuffer(device,absXBuf,count*4)]);const checkStarted=performance.now();const checked=await options.checkDrift(start,end,s+1);cpuCheckMs+=performance.now()-checkStarted;const exact=typeof checked==='number'?checked:checked?.minimumJacobian;if(Number.isFinite(exact)){minimum=exact;cpuMinimum=exact;checkedByCpu=true;if(Number.isFinite(rawMinimum)&&!gpuNonfinite)maxGpuCpuMinError=Math.max(maxGpuCpuMinError,Math.abs(rawMinimum-exact));}else throw new Error(`PM drift ${s+1} requires a finite CPU simplex minimum recheck`);}else throw new Error(`PM drift ${s+1} requires CPU f64 simplex minimum recheck (candidate ${rawMinimum}, flags 0x${(await readBuffer(device,diagFlags,(s+2)*4,Uint32Array))[s+1].toString(16)})`);}
      driftGpuMinima[s]=rawMinimum;driftMinima[s]=minimum;options.onDrift?.({step:s+1,minimum,gpuMinimum:rawMinimum,cpuMinimum,checkedByCpu,gpuNonfinite});if(minimum<=0&&!allowCrossing)throw new Error(`PM sheet crossed during drift ${s+1}`);
      await force(`drift-${s+1}`);
      const end=device.createCommandEncoder({label:`PM final kick ${s+1}`});dispatch(6,[np,dim,hasJvp?1:0,0,times.kick1[s],0,0,0],['u','u','u','u','f','f','u','u'],[xbuf,pbuf,dxbuf,dpbuf,particleF,particleDF],end,32,np);const finalKickStarted=performance.now();queue.submit([end.finish()]);await queue.onSubmittedWorkDone();kernelMs+=performance.now()-finalKickStarted;
      options.onProgress?.({stage:'GPU particle-mesh evolution',step:s+1,steps});
    }
    const tRead=performance.now();const [outX,outP,outDx,outDp,invalidDriftPolynomialFlags]=await Promise.all([readBuffer(device,absXBuf,count*4),readBuffer(device,pbuf,count*4),readBuffer(device,dxbuf,count*4),readBuffer(device,dpbuf,count*4),readBuffer(device,diagFlags,(steps+1)*4,Uint32Array)]);readbackMs=performance.now()-tRead;
    const oom=await popErrorScope?.();const validation=await popErrorScope?.();scopesOpen=false;if(oom)throw new Error(`WebGPU PM allocation error: ${oom.message}`);if(validation)throw new Error(`WebGPU PM validation error: ${validation.message}`);
    for(const [name,a] of [['x',outX],['p',outP],['dx',outDx],['dp',outDp]])for(let i=0;i<a.length;i++)if(!Number.isFinite(a[i]))throw new Error(`GPU PM ${name}[${i}] is nonfinite`);
    const shaderSha256=await shaderHash(SHADERS),fftShaderSha256=await shaderHash(FFT_SHADER_SOURCE);
    return {x:outX,p:outP,dx:outDx,dp:outDp,driftMinima,driftGpuMinima,initialMinimum,initialGpuMinimum,diagnostics:{dim,sheetN:n,meshN,particleCount:np,steps,hasJvp,estimatedBytes,allowCrossing,positionRepresentation:input.positionRepresentation??'absolute',maxGpuCpuMinError,invalidDriftPolynomialFlags:[...invalidDriftPolynomialFlags],forceKernel:'rust-spectral-cic-v1',cicAccumulation:PM_KERNEL_PROVENANCE.accumulation,precision:input.positionRepresentation==='lattice-displacement'?'f32 lattice displacement psi, materialized absolute f32 output x, f32 canonical p/tangents and clocks':'f32 absolute x, canonical p/tangents and clocks',shaderSha256,fftShaderSha256},timings:{compileMs,uploadMs,kernelMs,diagnosticMs,cpuCheckMs,readbackMs,totalMs:performance.now()-started}};
  }finally{if(scopesOpen){try{await popErrorScope();await popErrorScope();}catch{}}fft?.destroy();for(const b of owned){try{b.destroy();}catch{}}}
}

async function shaderHash(source){const subtle=globalThis.crypto?.subtle;if(!subtle)return null;const digest=await subtle.digest('SHA-256',new TextEncoder().encode(source));return [...new Uint8Array(digest)].map((b)=>b.toString(16).padStart(2,'0')).join('');}
export const PM_SHADER_SOURCE=SHADERS;
