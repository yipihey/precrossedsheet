// Orthogonal exact-deposit checks: analytic uniform-sheet cases plus optional
// comparisons against the retained Rust f64 exact voxelizer.
import { runExactDeposit } from './deposit.js';

function packedSheet(dim,n,transform){
  const out=new Float32Array(n**dim*4);
  for(let flat=0;flat<n**dim;flat++){
    let q=flat,coord=[];
    for(let a=dim-1;a>=0;a--){coord[a]=(q%n)/n;q=Math.floor(q/n);}
    const x=transform(coord);
    for(let a=0;a<dim;a++)out[flat*4+a]=Math.fround(x[a]);
  }
  return out;
}
function makeFixture(dim,n,meshN,kind){
  const identity=q=>q;
  const translation=q=>q.map(x=>x+0.375);
  const collapsed=q=>q.map(()=>0.375);
  const shear=q=>dim===2?
    [q[0]+0.22*Math.sin(2*Math.PI*q[1]),q[1]+0.19*Math.sin(2*Math.PI*q[0])]:
    [q[0]+0.17*Math.sin(2*Math.PI*q[1]),q[1]+0.16*Math.sin(2*Math.PI*q[2]),q[2]+0.18*Math.sin(2*Math.PI*q[0])];
  const sliver=q=>dim===2?[q[0]+q[1],q[0]+q[1]+1e-6*q[0]]:[q[0]+q[1]+q[2],q[0]+q[1]+q[2]+1e-6*q[1],q[0]+q[1]+q[2]+1e-6*q[2]];
  const transform={identity,translation,collapsed,shear,sliver}[kind];
  return {dim,n,meshN,boxSize:1,kind,positions:packedSheet(dim,n,transform)};
}
function stats(a,b){
  let maxAbs=0,maxRel=0,sumA=0,sumB=0,nonfinite=0;
  for(let i=0;i<a.length;i++){
    const x=a[i],y=b[i];if(!Number.isFinite(x)||!Number.isFinite(y)){nonfinite++;continue;}
    const e=Math.abs(x-y);maxAbs=Math.max(maxAbs,e);maxRel=Math.max(maxRel,e/Math.max(1e-8,Math.abs(y)));sumA+=x;sumB+=y;
  }
  return {maxAbsolute:maxAbs,maxRelative:maxRel,meanMassA:sumA/a.length,meanMassB:sumB/b.length,nonfinite};
}
function uniformError(a){
  let max=0,mean=0;for(const x of a){max=Math.max(max,Math.abs(x-1));mean+=x;}
  return {maximumAbsolute:max,mean:mean/a.length};
}
async function cpuDensity(cpuReference,fixture){
  if(!cpuReference)return null;
  const value=await cpuReference({dim:fixture.dim,n:fixture.n,meshN:fixture.meshN,boxSize:fixture.boxSize,positions:fixture.positions});
  const density=value?.density??value;
  if(!(density instanceof Float32Array)||density.length!==fixture.meshN**fixture.dim)throw new TypeError('cpuReference must return the exact Rust Float32 density field');
  return density;
}

/**
 * Exercise identity and translated sheets with independent analytic uniform
 * density, folded smooth sheets against a supplied Rust oracle, collapsed
 * simplices against the oracle where available, and slivers through explicit
 * GPU rejection/CPU recovery. The caller supplies cpuReference(fixture) from
 * the retained Rust exact deposit; no GPU output is used to make its oracle.
 */
export async function runExactDepositChecks(device,{cpuReference=null,signal,onProgress,maxGpuBytes}={}){
  const rows=[],timings=[];
  for(const dim of [2,3]){
    const n=4,meshN=8;
    for(const kind of ['identity','translation']){
      const fixture=makeFixture(dim,n,meshN,kind);const start=performance.now();
      const got=await runExactDeposit(device,fixture,{signal,onProgress,maxGpuBytes});const analytic=uniformError(got.density);let cpu=null;
      if(cpuReference){const ref=await cpuDensity(cpuReference,fixture);cpu=stats(got.density,ref);}
      const tolerance=cpuReference?{absolute:3e-5,relative:3e-4}:{uniform:3e-5};
      const pass=analytic.maximumAbsolute<=3e-5&&(!cpu||cpu.nonfinite===0&&cpu.maxAbsolute<=tolerance.absolute&&cpu.maxRelative<=tolerance.relative);
      rows.push({label:`${dim}D ${kind} exact sheet remains uniform`,pass,analytic,cpu,tolerance,diagnostics:got.diagnostics,provenance:got.provenance});
      timings.push({dim,kind,...got.timings,wallMs:performance.now()-start});
      if(dim===2&&kind==='identity'){
        const bytes=got.diagnostics.bufferBytes,accounted=Object.values(bytes).reduce((sum,value)=>sum+value,0);
        let allocated=0,compiled=0,budgetRejected=false;
        try{
          await runExactDeposit({createBuffer(){allocated++;},createShaderModule(){compiled++;}},fixture,{maxGpuBytes:got.diagnostics.estimatedBytes-1});
        }catch(error){budgetRejected=error instanceof RangeError&&String(error.message).includes(`needs ${got.diagnostics.estimatedBytes} bytes`)&&String(error.message).includes('maxGpuBytes');}
        const footprint=bytes.batchUniforms===32*got.diagnostics.batches&&bytes.gatherUniform===32&&accounted===got.diagnostics.estimatedBytes;
        rows.push({label:'Exact-deposit preflight accounts parameter buffers and rejects one byte under budget before GPU allocation',pass:footprint&&budgetRejected&&allocated===0&&compiled===0,estimatedBytes:got.diagnostics.estimatedBytes,bufferBytes:bytes,batches:got.diagnostics.batches,budgetRejected,allocated,compiled});
      }
    }
    const collapsed=makeFixture(dim,n,meshN,'collapsed');
    try{
      const got=await runExactDeposit(device,collapsed,{signal,onProgress,maxGpuBytes});const expected=1/n**dim;let max=0,mean=0;for(const x of got.density){max=Math.max(max,Math.abs(x-expected));mean+=x;}mean/=got.density.length;let cpu=null;
      if(cpuReference)cpu=stats(got.density,await cpuDensity(cpuReference,collapsed));
      const pass=max<=3e-5&&Math.abs(mean-expected)<=3e-5&&(!cpu||cpu.nonfinite===0&&cpu.maxAbsolute<=3e-5);
      rows.push({label:`${dim}D collapsed sheet retains the positive periodic seam stream`,pass,maximumAbsoluteFromExpected:max,meanDensity:mean,expectedMean:expected,cpu,degenerateSimplices:got.diagnostics.degenerateSimplices,diagnostics:got.diagnostics});timings.push({dim,kind:'collapsed',...got.timings});
    }catch(error){rows.push({label:`${dim}D collapsed sheet retains the positive periodic seam stream`,pass:false,error:String(error)});}
    const folded=makeFixture(dim,n,meshN,'shear');
    if(cpuReference){
      try{const got=await runExactDeposit(device,folded,{signal,onProgress,maxGpuBytes});const ref=await cpuDensity(cpuReference,folded);const err=stats(got.density,ref);const pass=err.nonfinite===0&&err.maxAbsolute<=5e-5&&err.maxRelative<=5e-4;
        rows.push({label:`${dim}D folded exact stream mass matches Rust f64 clipping`,pass,error:err,tolerance:{absolute:5e-5,relative:5e-4},diagnostics:got.diagnostics,provenance:got.provenance});timings.push({dim,kind:'folded',...got.timings});
      }catch(error){rows.push({label:`${dim}D folded exact stream mass matches Rust f64 clipping`,pass:false,error:String(error)});}
    }
    const sliver=makeFixture(dim,n,meshN,'sliver');let rejected=false,errorText='';
    try{await runExactDeposit(device,sliver,{signal,onProgress,maxGpuBytes});}catch(error){rejected=error?.code==='EXACT_DEPOSIT_GPU_DIAGNOSTIC'&&Boolean(error?.diagnostics?.flags||error?.diagnostics?.illConditionedSimplices);errorText=String(error);}
    let cpuPass=null;if(cpuReference){try{const ref=await cpuDensity(cpuReference,sliver);cpuPass=ref.every(Number.isFinite);}catch{cpuPass=false;}}
    rows.push({label:`${dim}D thin simplex requests CPU recovery instead of silently losing mass`,pass:rejected&&cpuPass!==false,rejected,error:errorText,cpuReferenceFinite:cpuPass});
  }
  // Force several submissions on larger meshes. These analytic uniform maps
  // catch batch-offset mistakes that n=4 fixtures cannot exercise; the
  // negative translation also keeps the first and later batches on wrapped
  // periodic cells.
  for(const {dim,n,batchSimplices} of [{dim:2,n:32,batchSimplices:512},{dim:3,n:16,batchSimplices:8192}]){
    const meshN=8;
    for(const kind of ['identity','negative-translation']){
      const fixture=kind==='identity'?makeFixture(dim,n,meshN,'identity'):{dim,n,meshN,boxSize:1,kind,positions:packedSheet(dim,n,q=>q.map(x=>x-.375))};
      try{
        const got=await runExactDeposit(device,fixture,{signal,onProgress,maxGpuBytes,batchSimplices});
        const analytic=uniformError(got.density);let cpu=null;
        if(cpuReference)cpu=stats(got.density,await cpuDensity(cpuReference,fixture));
        const pass=got.diagnostics.batches>1&&analytic.maximumAbsolute<=3e-5&&Math.abs(analytic.mean-1)<=1e-6&&
          (!cpu||cpu.nonfinite===0&&cpu.maxAbsolute<=3e-5&&cpu.maxRelative<=3e-4);
        rows.push({label:`${dim}D n=${n} ${kind} multi-batch deposit preserves analytic unit density`,pass,batches:got.diagnostics.batches,batchSimplices,analytic,cpu,tolerance:{uniformAbsolute:3e-5,meanAbsolute:1e-6,cpuAbsolute:3e-5,cpuRelative:3e-4},diagnostics:got.diagnostics});
        timings.push({dim,n,kind:'multi-batch-'+kind,...got.timings});
      }catch(error){rows.push({label:`${dim}D n=${n} ${kind} multi-batch deposit preserves analytic unit density`,pass:false,batchSimplices,error:String(error)});}
    }
  }
  const failures=rows.filter(r=>r.pass===false);
  return {status:failures.length?'fail':'pass',rows,failures,timings,precision:'GPU positive f32 geometric overlap with explicit conditioning/capacity fallback; Rust f64 exact clipping is the optional independent oracle',budget:{uniformAbsolute:3e-5,foldedAbsolute:5e-5,foldedRelative:5e-4}};
}

export const EXACT_DEPOSIT_FIXTURE_KINDS=Object.freeze(['identity','translation','collapsed','shear','sliver']);
