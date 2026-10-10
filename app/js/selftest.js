import { makeDefaults } from './state.js';

const button = document.querySelector('#run-tests'), list = document.querySelector('#test-results'), status = document.querySelector('#test-status');
button.addEventListener('click', () => {
  button.disabled = true; status.textContent = 'Computing…'; list.replaceChildren();
  const state = { ...makeDefaults(), dim: 1, n: 16, fraction: .35, order: 2, smoothing: .02, mesh: 32, forceMesh: 32, view: 'gradient' };
  const worker = new Worker('./js/worker.js', { type: 'module' }); let phase = 'safe';
  const finish = () => { clearTimeout(timeout); worker.terminate(); button.disabled = false; };
  const timeout = setTimeout(() => { show('Worker completes both checks within 30 seconds', false, 'Timed out'); finish(); status.textContent = 'Failed'; }, 30000);
  worker.onmessage = ({ data }) => {
    if (phase === 'safe') {
      if (data.error) { show('WASM worker returns a result', false, data.error); finish(); status.textContent='Failed'; return; }
      const d=data.result,diag=d?.diagnostics||{},guard=d?.guard?.ok;
      show('Pre-crossing guard passes for the gentle fixture',guard===true,`guard.ok = ${String(guard)}`);
      const mass=Number(diag.mass),pmMass=Number(diag.pmMass);
      show('Exact-sheet deposited mass is conserved to 1e-6',Number.isFinite(mass)&&Math.abs(mass-1)<=1e-6,`mass = ${Number.isFinite(mass)?mass:'missing'}`);
      show('PM-trajectory deposited mass is conserved to 1e-6',Number.isFinite(pmMass)&&Math.abs(pmMass-1)<=1e-6,`PM mass = ${Number.isFinite(pmMass)?pmMass:'missing'}`);
      const raw=d?.gradient?.relativeError,values=ArrayBuffer.isView(raw)?Array.from(raw):Array.isArray(raw)?raw:[],maxError=values.reduce((m,v)=>Math.max(m,Number(v)||0),0);
      show('Amplitude derivative agrees with finite differences below 1e-5',values.length>0&&values.every(Number.isFinite)&&maxError<1e-5,`${values.length} samples; max relative error = ${values.length?maxError:'missing'}`);
      phase='invalid';const invalid={...state,fraction:1.01};const {pubmodel,pubest,pubr,paired,...coreState}=invalid;worker.postMessage({type:'run',requestId:2,state:coreState});return;
    }
    const d=data.result,hasError=Boolean(data.error||d?.error),hasGuardSuccess=d?.guard?.ok===true,hasScience=Boolean(d?.q||d?.x||d?.rho||d?.spectra);
    show('Out-of-range growth fraction is rejected without scientific output',hasError&&!hasGuardSuccess&&!hasScience,data.error||d?.error||'No error returned');
    const ok=[...list.children].every((li)=>li.classList.contains('test-pass'));status.textContent=ok?'All checks passed':'Some checks failed';finish();
  };
  worker.onerror=(e)=>{show('WASM worker completes the checks',false,e.message||'Worker error');finish();status.textContent='Failed';};
  const {pubmodel,pubest,pubr,paired,...coreState}=state;worker.postMessage({type:'run',requestId:1,state:coreState});
});
function show(label,ok,detail){const li=document.createElement('li');li.className=ok?'test-pass':'test-fail';li.textContent=`${ok?'PASS':'FAIL'} — ${label}: ${detail}`;list.append(li);}
