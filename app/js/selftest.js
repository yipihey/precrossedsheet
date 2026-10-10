import { coreRequest, makeDefaults, normalizeState } from './state.js';

const button=document.querySelector('#run-tests'),list=document.querySelector('#test-results'),status=document.querySelector('#test-status');
button.addEventListener('click',()=>{
  button.disabled=true;status.textContent='Computing…';list.replaceChildren();
  const safe=normalizeState({...makeDefaults(),dim:1,n:16,mesh:16,forceMesh:16,steps:32,epoch:'fraction',fraction:.35,order:2});
  const crossing=normalizeState({...safe,omegaM:1,omegaB:0,fraction:1.2});
  const oversized=normalizeState({...safe,dim:3,n:256,mesh:128,forceMesh:128});
  const worker=new Worker('./js/worker.js',{type:'module'});const jobs=[['safe',coreRequest(safe)],['crossing',coreRequest(crossing)],['oversize',coreRequest(oversized)]];let phase=0;
  const timeout=setTimeout(()=>{show('Worker completes the complete check set',false,'Timed out after 90 seconds');finish();status.textContent='Failed';},90000);
  const finish=()=>{clearTimeout(timeout);worker.terminate();button.disabled=false;};
  worker.onmessage=({data})=>{
    if(data.requestId!==phase+1)return;
    const name=jobs[phase][0],d=data.result||{},err=data.error||d.error;
    if(name==='safe'){
      const diag=d.diagnostics||{};
      show('ΛCDM reference request is echoed for the run',Boolean(d.request?.omegaM===.315&&d.request?.boxSize===300&&d.request?.epoch==='fraction'),`Ωm=${d.request?.omegaM??'missing'}, L=${d.request?.boxSize??'missing'} Mpc/h, epoch=${d.request?.epoch??'missing'}`);
      show('Resolved LPT/PM guard passes for the safe fixture',d.guard?.ok===true,`guard.ok=${String(d.guard?.ok)}`);
      for(const [label,value] of [['LPT sheet mass ratio',Number(diag.mass)],['PM sheet mass ratio',Number(diag.pmMass)]])show(`${label} is conserved to 1e-6`,Number.isFinite(value)&&Math.abs(value-1)<=1e-6,`ratio=${Number.isFinite(value)?value:'missing'}`);
      const raw=d.gradient?.relativeError,values=ArrayBuffer.isView(raw)?Array.from(raw):Array.isArray(raw)?raw:[],maxError=values.reduce((m,v)=>Math.max(m,Number(v)||0),0);
      show('Amplitude tangent agrees with finite differences below 1e-5',values.length>0&&values.every(Number.isFinite)&&maxError<1e-5,`${values.length} modes; max relative error=${values.length?maxError:'missing'}`);
    }else if(name==='crossing'){
      const hasFields=Boolean(d.x&&d.rho&&d.spectra&&d.pdf);
      show('Post-crossing guard is reported while diagnostic fields remain available',!err&&d.guard?.ok===false&&hasFields,err||`guard.ok=${String(d.guard?.ok)}; x=${Boolean(d.x)}, density=${Boolean(d.rho)}, spectra=${Boolean(d.spectra)}, PDF=${Boolean(d.pdf)}`);
    }else{
      show('Dimension/resolution limit returns an explicit error without silent reduction',Boolean(err)&&!d.x&&!d.rho,err||`returned n=${d.n}, dim=${d.dim}`);
    }
    phase++;if(phase<jobs.length){worker.postMessage({type:'run',requestId:phase+1,state:jobs[phase][1]});return;}
    const ok=[...list.children].every(li=>li.classList.contains('test-pass'));status.textContent=ok?'All checks passed':'Some checks failed';finish();
  };
  worker.onerror=e=>{show('Worker completes the checks',false,e.message||'Worker error');status.textContent='Failed';finish();};
  worker.postMessage({type:'run',requestId:1,state:jobs[0][1]});
});
function show(label,ok,detail){const li=document.createElement('li');li.className=ok?'test-pass':'test-fail';li.textContent=`${ok?'PASS':'FAIL'} — ${label}: ${detail}`;list.append(li);}
