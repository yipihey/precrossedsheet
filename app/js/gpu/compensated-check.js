// Independent device checks for the compensated continuous P0 observer.
// The reference uses Duffy-mapped Gauss quadrature of the actual simplex
// integrals and tangent integrands; it does not call the GPU DD recurrence.
import { runCompensatedDirect } from './compensated-direct.js';

function gauss01(count=12){
  const x=new Float64Array(count),w=new Float64Array(count),half=count/2;
  for(let i=0;i<half;i++){
    let z=Math.cos(Math.PI*(i+0.75)/(count+0.5)),p=0,dp=0;
    for(let it=0;it<24;it++){let a=1,b=z;for(let k=2;k<=count;k++){const q=((2*k-1)*z*b-(k-1)*a)/k;a=b;b=q;}p=b;dp=count*(z*b-a)/(z*z-1);const next=z-p/dp;if(next===z){z=next;break;}z=next;}
    const weight=2/((1-z*z)*dp*dp);x[i]=(1-z)/2;x[count-1-i]=(1+z)/2;w[i]=weight/2;w[count-1-i]=weight/2;
  }
  return {x,w};
}
const f32=(x)=>Math.fround(x);
function packHiLo(values){const hi=new Float32Array(values.length),lo=new Float32Array(values.length);for(let i=0;i<values.length;i++){hi[i]=f32(values[i]);lo[i]=f32(values[i]-hi[i]);}return [hi,lo];}
function makeFixture(dim){
  const n=3,np=n**dim,psi=new Float64Array(np*4),direction=new Float64Array(np*4);
  for(let i=0;i<np;i++){
    const c=dim===2?[Math.floor(i/n),i%n,0]:[Math.floor(i/(n*n)),Math.floor(i/n)%n,i%n];
    const q=c.map(v=>v/n),phase=q.map(v=>2*Math.PI*v);
    psi[4*i]=0.31+0.28*Math.sin(phase[0])+0.11*Math.cos(phase[1]);
    psi[4*i+1]=-0.23+0.24*Math.sin(phase[1])-0.09*Math.cos(phase[0]);
    psi[4*i+2]=dim===3?0.17+0.19*Math.sin(phase[2]+phase[0]):0;
    direction[4*i]=0.013*Math.cos(phase[0]+0.2)+0.004*Math.sin(phase[1]);
    direction[4*i+1]=-0.011*Math.sin(phase[1]-0.1)+0.003*Math.cos(phase[0]);
    direction[4*i+2]=dim===3?0.009*Math.cos(phase[2]-phase[0]):0;
  }
  const [psiHi,psiLo]=packHiLo(psi),[directionHi,directionLo]=packHiLo(direction);
  const rows=[[0,0,0,0],[1,0,0,0],[-1,1,0,0],[2,-1,0,0]];if(dim===3)rows.push([0,1,-1,0]);
  const modes=new Int32Array(rows.flat());return {dim,n,np,psiHi,psiLo,directionHi,directionLo,modes,psi,direction};
}
function simplices(dim){return dim===2?[[0,1,3],[0,3,2]]:[[0,1,3,7],[0,1,5,7],[0,2,3,7],[0,2,6,7],[0,4,5,7],[0,4,6,7]];}
function vertexCoords(base,bits,dim,n){const coords=[0,1,2].map(a=>base[a]+((bits>>a)&1));return coords.map(v=>v/n);}
function vertexIndex(base,bits,dim,n){const q=base.map((v,a)=>(v+((bits>>a)&1))%n);return dim===2?q[0]*n+q[1]:(q[0]*n+q[1])*n+q[2];}
function reference(fixture){
  const {dim,n,np,modes,psiHi,psiLo,directionHi,directionLo}=fixture,{x,w}=gauss01(12),nm=modes.length/4,ans=new Float64Array(nm*4),simp=simplices(dim),factorial=dim===2?2:6;
  for(let cell=0;cell<np;cell++){
    const base=dim===2?[Math.floor(cell/n),cell%n,0]:[Math.floor(cell/(n*n)),Math.floor(cell/n)%n,cell%n];
    for(const bitsList of simp){
      const ids=bitsList.map(bits=>vertexIndex(base,bits,dim,n)),q=bitsList.map(bits=>vertexCoords(base,bits,dim,n));
      for(let ia=0;ia<x.length;ia++)for(let ib=0;ib<x.length;ib++)for(let ic=0;ic<(dim===3?x.length:1);ic++){
        let lambda,weight;
        if(dim===2){const u=x[ia],v=x[ib];lambda=[1-u,u*(1-v),u*v];weight=2*u*w[ia]*w[ib];}
        else{const u=x[ia],v=x[ib],z=x[ic];lambda=[(1-u)*(1-v)*(1-z),u,(1-u)*v,(1-u)*(1-v)*z];weight=6*(1-u)*(1-u)*(1-v)*w[ia]*w[ib]*w[ic];}
        const px=[0,0,0],dv=[0,0,0];for(let j=0;j<lambda.length;j++){const id=ids[j];for(let a=0;a<dim;a++){const off=4*id+a;px[a]+=lambda[j]*(q[j][a]+psiHi[off]+psiLo[off]);dv[a]+=lambda[j]*(directionHi[off]+directionLo[off]);}}
        for(let mi=0;mi<nm;mi++){const md=modes.subarray(4*mi,4*mi+3);let phase=0,tangent=0;for(let a=0;a<dim;a++){phase+=md[a]*px[a];tangent+=md[a]*dv[a];}const angle=2*Math.PI*phase,cr=Math.cos(angle),ci=-Math.sin(angle),scale=weight/(factorial*np);const k=4*mi;ans[k]+=scale*cr;ans[k+1]+=scale*ci;ans[k+2]+=scale*(2*Math.PI*tangent*ci);ans[k+3]+=scale*(-2*Math.PI*tangent*cr);}
      }
    }
  }
  return ans;
}

/** Run both non-radix 2D and 3D device checks against independent quadrature. */
export async function runCompensatedDirectSelfTests(device,options={}){
  const rows=[];for(const dim of [2,3]){
    if(options.signal?.aborted||options.shouldCancel?.())throw new DOMException('Compensated self-test canceled','AbortError');
    const fixture=makeFixture(dim),expected=reference(fixture),actual=await runCompensatedDirect(device,{dim:fixture.dim,n:fixture.n,modes:fixture.modes,psiHi:fixture.psiHi,psiLo:fixture.psiLo,directionHi:fixture.directionHi,directionLo:fixture.directionLo},{...options,maxGpuBytes:options.maxGpuBytes??64*1024*1024});
    for(let m=0;m<fixture.modes.length/4;m++)for(let c=0;c<4;c++){const index=4*m+c,value=actual.channels[index],referenceValue=expected[index],absoluteError=Math.abs(value-referenceValue),tolerance=2e-8+2e-7*Math.abs(referenceValue);rows.push({label:`compensated analytic ${dim}D n=3 mode ${Array.from(fixture.modes.slice(4*m,4*m+3)).join(',')} channel ${c}`,actual:value,reference:referenceValue,absoluteError,tolerance,pass:Number.isFinite(value)&&absoluteError<=tolerance});}
    await options.onProgress?.({stage:'compensated-selftest',dim});
  }
  return {status:rows.every(r=>r.pass)?'pass':'fail',rows,budget:{absolute:2e-8,relative:2e-7},method:'independent 12-point Duffy Gauss quadrature of continuous simplex field and tangent integrands'};
}
