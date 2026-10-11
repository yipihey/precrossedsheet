import { coreRequest, decodeHash, encodeHash, executionPolicy, GRIDS, gridLimit, HASH_SCHEMA, normalizeState, STATE_VERSION, STEPS } from './state.js';
import { PRESETS, presetState } from './presets.js';
import { LinePlot } from './viz/plot.js';
import { copyText, downloadBlob, flash } from './viz/download.js';

function fragmentVersion(hash){const entry=String(hash||'').replace(/^#/,'').split('&').find(p=>p.startsWith('v='));return entry?entry.slice(2):null;}
let unsupportedSchema=fragmentVersion(location.hash)!=null&&!(['2',String(STATE_VERSION)].includes(fragmentVersion(location.hash)));
let S = decodeHash(location.hash), worker = null, timer = 0, requestId = 0, current = null, computedFor = '', activeKey = '', activeRequestId = null, activePhase = '', pendingRun = null, cancelTimer = 0, publishedPromise = null, publishedData = null, lastResult = null, lastResultKey = '', buildInfoPromise = null, runMilliseconds = null, execution = null;
const root = document.querySelector('#lab'), status = document.querySelector('#run-state');
const VIEWS = [
  ['sheet', 'Phase-space sheet'], ['density', 'Density'], ['spectra', 'Power spectra'],
  ['pdf', 'PDFs'], ['gradient', 'Differentiability'], ['published', 'Published replay'],
];
const chapterRefs = { sheet: '../chapters/01-sheet.html#sec-sheet', density: '../chapters/03-meshing.html#sec-meshing', spectra: '../chapters/04-validation.html#sec-validation', pdf: '../chapters/05-pdfs.html#sec-pdfs', gradient: '../chapters/06-inference.html#sec-inference', published: '../chapters/07-lab.html#sec-lab' };
const $ = (tag, cls, parent, value) => { const e = document.createElement(tag); if (cls) e.className = cls; if (value != null) e.textContent = value; parent.append(e); return e; };
const row = (parent, cls = '') => $('div', `lab-control ${cls}`, parent);

document.querySelector('#copy-link').addEventListener('click', async (ev) => { const b = ev.currentTarget; await copyText(location.href); flash(b, 'Copied'); });

const intro = $('section', 'lab-intro', root);
intro.innerHTML = '<p class="eyebrow">Cosmological Lagrangian perturbation theory</p><p>Evolve a discretized Lagrangian sheet with deterministic mode benchmarks or a Gaussian field generated from an Eisenstein–Hu no-wiggle ΛCDM spectrum. The model is periodic and uses translationally invariant lower-dimensional geometry for 1D and 2D runs; only 3D runs represent a full spatial realization. Results remain visible after orientation loss, with the single-stream guard status shown as a scientific caution.</p>';
const hasOldHash=location.hash.length>1&&!location.hash.slice(1).split('&').some(p=>p.startsWith('v='));
if(hasOldHash)$('p','legacy-link-note',intro,'This experiment link predates versioned manifests and the cosmological engine update. Its visible controls have been migrated to current defaults; it cannot reproduce an earlier engine result exactly. Copy or export the current versioned link to preserve this run from here onward.');
const unsupportedNote=unsupportedSchema?$('p','unsupported-link-note',intro,`This URL uses unsupported experiment schema v${fragmentVersion(location.hash)}. The experiment has not been run. Select a preset or edit a control to start a new version-${STATE_VERSION} run; the saved values cannot be safely interpreted.`):null;
function clearUnsupportedSchema(){unsupportedSchema=false;unsupportedNote?.remove();document.querySelectorAll('.unsupported-link-note').forEach((el)=>el.remove());}
const presets = $('div', 'lab-presets', root);
$('span', 'control-caption', presets, 'Start with');
const presetSelect = $('select', 'lab-select', presets); presetSelect.setAttribute('aria-label', 'Choose an experiment preset');
const customOption = $('option', '', presetSelect, 'Current settings'); customOption.value = '';
customOption.disabled = true;
for (const p of PRESETS) { const o = $('option', '', presetSelect, p.label); o.value = p.id; }
const presetNote = $('span', 'preset-note', presets, PRESETS[0].note);
presetSelect.value = '';
presetNote.textContent = 'Adjust the controls or choose a guided experiment.';
presetSelect.addEventListener('change', () => { clearUnsupportedSchema();S = normalizeState(presetState(presetSelect.value)); controls.sync(); selectView(S.view, false); updateURL(); run(); presetNote.textContent = PRESETS.find((p) => p.id === presetSelect.value)?.note || ''; });
const exportJsonButton=$('button','qd-btn',presets,'Export experiment JSON');exportJsonButton.type='button';exportJsonButton.disabled=true;exportJsonButton.setAttribute('aria-label','Download the complete experiment state, result, and build manifest as JSON');
const importLabel=$('label','import-json-button qd-btn',presets,'Import settings JSON');const importInput=$('input','',importLabel);importInput.type='file';importInput.accept='application/json,.json';importInput.setAttribute('aria-label','Import experiment settings from a JSON manifest');
exportJsonButton.addEventListener('click',()=>exportExperiment());
importInput.addEventListener('change',async()=>{const file=importInput.files?.[0];if(!file)return;try{const manifest=JSON.parse(await file.text());if(manifest.manifestFormat!=='precrossedsheet-experiment'||manifest.schemaVersion!==STATE_VERSION||!manifest.state)throw new Error(`Expected a precrossedsheet experiment manifest at schema version ${STATE_VERSION}.`);const imported=normalizeState(manifest.state);if(imported.schemaVersion!==STATE_VERSION)throw new Error('The experiment state version is missing or inconsistent.');for(const [key] of HASH_SCHEMA){if(!(key in manifest.state))throw new Error(`The manifest is missing reproducibility field “${key}”.`);const raw=manifest.state[key],value=imported[key];if(typeof raw==='number'&&!Object.is(raw,value)||typeof raw==='string'&&raw!==value||typeof raw==='boolean'&&raw!==value)throw new Error(`The saved value for “${key}” is outside the current experiment schema.`);}clearUnsupportedSchema();S=imported;controls.sync();presetSelect.value='';presetNote.textContent=`Imported experiment settings from ${file.name}.`;selectView(S.view,false);updateURL();run();}catch(error){setNotice(`Could not import settings: ${error?.message||error}`,'error');}finally{importInput.value='';}});
async function exportExperiment(){if(!current||(!current?.published&&computedFor!==physicsKey())){setNotice('Wait until the result matches the current experiment settings before exporting.');return;}try{const build=await getBuildInfo();const isPublished=S.view==='published',result=isPublished?current?.published:current;const manifest={manifestFormat:'precrossedsheet-experiment',schemaVersion:STATE_VERSION,kind:isPublished?'archiveReplay':'liveNumericalRun',createdAt:new Date().toISOString(),state:{...S},request:isPublished?null:(current?.request||coreRequest(S)),execution:isPublished?null:(result?.execution||execution||null),build,run:{durationMilliseconds:runMilliseconds,guard:result?.guard||null,epoch:result?.request?.epoch??S.epoch,scaleFactor:result?.a??S.scaleFactor,growthFactor:result?.D??null,shellCrossingReference:result?.shellCrossing??null},source:isPublished?result?.provenance||null:null,result};const blob=new Blob([JSON.stringify(manifest)],{type:'application/json'});downloadBlob(blob,`precrossedsheet-${Date.now()}.json`);flash(exportJsonButton,'JSON saved');}catch(error){setNotice(`Could not export the experiment: ${error?.message||error}`,'error');}}
function getBuildInfo(){if(!buildInfoPromise)buildInfoPromise=fetch('./data/build-info.json').then(r=>{if(!r.ok)throw new Error(`build metadata HTTP ${r.status}`);return r.json();}).catch(()=>({engineVersion:'unknown',gitRevision:'unavailable',buildTime:null}));return buildInfoPromise;}

const layout = $('div', 'lab-layout', root), sidebar = $('aside', 'lab-sidebar', layout), main = $('section', 'lab-main', layout);
$('h2', 'section-heading', sidebar, 'Experiment controls');
const controlWrap = $('div', 'control-list', sidebar), controlHandles = {};
function controlSection(title) { const section=$('section','control-section',controlWrap);$('h3','control-subheading',section,title);return section; }
let activeControlGroup=controlSection('Geometry and epoch');
function selectControl(key, label, opts, help) {
  const wrap = $('label', 'control-field', activeControlGroup); $('span', 'control-name', wrap, label);
  const sel = $('select', 'lab-select', wrap); sel.setAttribute('aria-label', label); for (const [v, text] of opts) { const o = $('option', '', sel, text); o.value = v; }
  if (help) $('small', 'control-help', wrap, help);
  sel.addEventListener('change', () => { S[key] = ['dim','n','order','seed','steps','mesh','forceMesh','initialCutoff'].includes(key) ? Number(sel.value) : sel.value; S = normalizeState(S); controls.sync(); updateGridOptions(); changed(); });
  controlHandles[key] = { set: (v) => { sel.value = String(v); }, el: sel };
}
function rangeControl(key, label, min, max, step, help, format = (v) => String(+Number(v).toPrecision(3))) {
  const wrap = $('label', 'control-field', activeControlGroup); const top = $('span', 'control-top', wrap); $('span', 'control-name', top, label); const out = $('output', 'control-value', top); const input = $('input', '', wrap); input.type = 'range'; input.min = min; input.max = max; input.step = step; input.setAttribute('aria-label', label);
  if (help) $('small', 'control-help', wrap, help);
  const set = (v) => { input.value = String(v); out.value = format(Number(v)); out.textContent = format(Number(v)); };
  set(S[key]); input.addEventListener('input', () => { S[key] = Number(input.value); out.value = format(S[key]); out.textContent = format(S[key]); changed(false); }); input.addEventListener('change', () => changed(true));
  controlHandles[key] = { set, el: input };
}
selectControl('dim', 'Spatial dimension', [['1','1D periodic model'],['2','2D periodic model'],['3','3D volume']], 'The 1D and 2D cases are translationally invariant reductions, not full spatial cosmological realizations.');
selectControl('n', 'Lagrangian vertices per axis', GRIDS.map(v=>[v,`${v} per axis`]), 'Supported maxima: 1D 1024, 2D 256, 3D 64. Higher options are disabled for the selected dimension; imported unsupported settings receive an explicit core error, never a silent reduction.');
selectControl('epoch', 'Output epoch is specified by', [['fraction','Fraction of crossing reference'],['scaleFactor','Scale factor a']], 'Select the independent clock. The other clock value remains in the complete experiment record but is not used for evolution.');
rangeControl('fraction', 'Crossing-reference fraction', .05, 1.5, .005, 'Values above 1 may be post-crossing. The guard reports orientation loss; it does not hide the resulting fields.');
rangeControl('scaleFactor', 'Scale factor a', .01, 2, .005, 'Output scale factor a. The ΛCDM growth table is valid over this range.');
activeControlGroup=controlSection('Cosmology and initial conditions');
function numberControl(key,label,min,max,step,help){const wrap=$('label','control-field',activeControlGroup);$('span','control-name',wrap,label);const input=$('input','lab-number',wrap);input.type='number';input.min=min;input.max=max;input.step=step;input.value=String(S[key]);input.setAttribute('aria-label',label);if(help)$('small','control-help',wrap,help);input.addEventListener('change',()=>{S[key]=Number(input.value);S=normalizeState(S);controls.sync();changed(true);});controlHandles[key]={set:v=>{input.value=String(v);},el:input};}
numberControl('omegaM','Matter density Ωm',.05,1,.001,'Present-day matter density parameter.');
numberControl('omegaB','Baryon density Ωb',0,.2,.001,'Present-day baryon density parameter; the core requires Ωb/Ωm ≤ 0.5.');
numberControl('h','Reduced Hubble parameter h',.4,1,.001,'H₀ = 100 h km s⁻¹ Mpc⁻¹.');
numberControl('ns','Primordial spectral index ns',.8,1.2,.001,'Tilt of the primordial scalar spectrum.');
numberControl('sigma8','Linear amplitude σ₈',.1,1.5,.001,'Gaussian ICs are normalized to this RMS before initial filtering and cutoff. Deterministic mode benchmarks use fixed amplitudes independent of σ₈.');
numberControl('boxSize','Periodic box length L (Mpc/h)',50,2000,1,'Physical comoving box length. Coordinates in the sheet output are fractions of L.');
numberControl('initialSmoothing','Initial-condition smoothing Rini (Mpc/h)',0,200,.1,'Gaussian initial power smoothing, separate from the observation smoothing below.');
selectControl('ic', 'Initial conditions', [['waves','Deterministic modes'],['gaussian','Gaussian EH no-wiggle field']], 'Gaussian initial conditions use an Eisenstein–Hu no-wiggle spectrum and fixed low-band cutoff.');
selectControl('initialCutoff','Gaussian radial Fourier cutoff',Array.from({length:32},(_,i)=>[i+1,`|m| ≤ ${i+1}`]),'Conservative integer radial Fourier-mode cutoff; Gaussian requests that do not fit the Lagrangian grid are rejected.');
numberControl('seed', 'Gaussian random seed', 0, 9999, 1, 'Integer mode-keyed seed, used only by Gaussian initial conditions.');
selectControl('order', 'LPT perturbative order', [[1,'1st order'],[2,'2nd order'],[3,'3rd order'],[4,'4th order']], 'Order used for the sheet prediction.');
activeControlGroup=controlSection('Evolution and observation grids');
selectControl('steps', 'PM integration steps', STEPS.map(v=>[v,`${v} steps`]), 'Number of time steps in the particle-mesh trajectory.');
selectControl('backend', 'Compute backend', [['cpu','CPU reference'],['gpu','GPU experimental'],['auto','Automatic (CPU until validated)']], 'CPU is the default. GPU attempts validated cosmological PM evolution, continuous sheet modes and compensated centered finite differences in 2D/3D. The PM force FFT requires a power-of-two force grid; other grids retain CPU PM. Weak LPT modes may use f64 CPU recovery. Exact sheet deposition stays on CPU; 1D and automatic selection use CPU.');
selectControl('mesh', 'Observation grid cells per axis', GRIDS.map(v=>[v,`${v} cells`]), 'Controls density, PDFs, and spectra; this does not change force resolution.');
selectControl('forceMesh', 'PM force grid cells per axis', GRIDS.map(v=>[v,`${v} cells`]), 'Independent force resolution. Change it separately to measure force-grid sensitivity.');
activeControlGroup=controlSection('Density observation');
rangeControl('smoothing', 'Observation smoothing R/L', 0, .15, .0025, 'Gaussian measurement filter as a fraction of the box length. This is not initial-condition smoothing.');
selectControl('weight','Primary PDF weighting',[['volume','Volume weighted'],['mass','Mass weighted']],'Both PDFs are plotted; the selected weighting is emphasized, and the URL records the choice.');
const controls = { sync() { for (const [k, h] of Object.entries(controlHandles)) h.set(S[k]); updateGridOptions(); updateClockControls(); updateMemoryEstimate(); } };
function estimatePeakBytes(state=S){const N=state.n**state.dim,M=Math.max(state.mesh,state.forceMesh)**state.dim;return N*2400+M*250;}
function updateGridOptions(){for(const key of ['n','mesh','forceMesh']){const select=controlHandles[key]?.el;if(!select)continue;const max=gridLimit(S.dim,key);for(const option of select.options){const val=Number(option.value),candidate={...S,[key]:val},tooLarge=estimatePeakBytes(candidate)>1024**3;option.disabled=val>max||tooLarge;option.textContent=`${val}${val>max?' · dimension limit':tooLarge?' · allocation limit':''}`;}}}
function updateClockControls(){for(const key of ['fraction','scaleFactor']){const field=controlHandles[key]?.el;const active=key==='fraction'?S.epoch==='fraction':S.epoch==='scaleFactor';if(field){field.disabled=!active;field.closest('.control-field')?.classList.toggle('control-disabled',!active);}}}
const memoryEstimate=$('p','resource-estimate',sidebar);
function updateMemoryEstimate(){const bytes=estimatePeakBytes(),fits=bytes<=1024**3;memoryEstimate.textContent=`Conservative core allocation estimate: ${(bytes/1048576).toFixed(1)} MiB (budget 1024 MiB; estimate, not a measured peak). ${fits?'The current grid combination is within the estimate.':'This selected combination will be rejected explicitly; grids will not be reduced.'}`;memoryEstimate.classList.toggle('resource-over-limit',!fits);}
const selftestLink=$('a','chapter-link',sidebar,'Run the GPU self-test ↗');selftestLink.href='./gpu-test.html';

const chapterLink = $('a', 'chapter-link', sidebar, 'Read the chapter behind this view ↗'); chapterLink.href = chapterRefs[S.view];
const notice = $('div', 'lab-notice', main); notice.setAttribute('role', 'status'); notice.setAttribute('aria-live', 'polite');
const tabs = $('nav', 'view-tabs', main); tabs.setAttribute('aria-label', 'Choose an experiment view');
const viewButtons = {};
for (const [key, label] of VIEWS) { const b = $('button', 'view-tab', tabs, label); b.type = 'button'; b.setAttribute('aria-pressed', 'false'); b.addEventListener('click', () => selectView(key)); viewButtons[key] = b; }
const viewHelp = $('p', 'view-help', main);
const viewHost = $('div', 'view-host', main);
const diagnostics = $('div', 'diagnostics', main);
const backendMeta = $('aside', 'backend-provenance', main);
const theory = $('aside', 'lab-theory', root);

function selectView(view, doRun = true) { const changedView = S.view !== view; S.view = view; for (const [k, b] of Object.entries(viewButtons)) b.setAttribute('aria-pressed', String(k === view)); chapterLink.href = chapterRefs[view] || chapterRefs.sheet; updateURL(); if (changedView && doRun) run(); else render(); }
function updateURL() { if(unsupportedSchema)return;history.replaceState(null, '', '#' + encodeHash(S)); }
function changed(immediate = false) { clearUnsupportedSchema();S = normalizeState(S); controls.sync(); presetSelect.value=''; presetNote.textContent='Custom experiment settings.';exportJsonButton.disabled=true; updateURL(); selectView(S.view, false); if (timer) clearTimeout(timer); if (immediate) run(); else { status.textContent = 'Waiting for input…'; timer = setTimeout(run, 180); } }
function physicsKey() { const { view, rotate, tilt, weight, paired, pubmodel, pubest, pubr, schemaVersion, ...physics } = S; return JSON.stringify(physics); }
function run() {
  if (timer) clearTimeout(timer); timer = 0;
  if(unsupportedSchema){S=decodeHash('');controls.sync();status.textContent='Unsupported saved experiment';setNotice('Unknown URL schema. No numerical run was started. Select a new preset or change a control before running.','error');exportJsonButton.disabled=true;clearHost();return;}
  if (S.view === 'published') { retireWorker(); activeKey=''; current = null; execution=null; exportJsonButton.disabled=true; loadPublished(); return; }
  const key = physicsKey();
  if (lastResultKey === key && lastResult) { cancelObsoleteForCachedResult(); current=lastResult;execution=current.execution||null;computedFor=key;exportJsonButton.disabled=false;status.textContent=current?.guard?.ok===false?'Post-crossing · interpret cautiously':'Ready';render();return; }
  if (computedFor === key && current && !current.published) {cancelObsoleteForCachedResult();exportJsonButton.disabled=false;status.textContent=current?.guard?.ok===false?'Post-crossing · interpret cautiously':'Ready';render();return;}
  const job={key,state:{...S},request:coreRequest(S),policy:executionPolicy(S)};
  if(activeRequestId!=null){pendingRun=job;cancelActiveRun();return;}
  dispatchRun(job);
}

function ensureWorker(){
  if(worker)return worker;
  worker=new Worker('./js/worker.js',{type:'module'});
  worker.onmessage=({data})=>{
    if(data.type==='phase'&&data.requestId===activeRequestId){activePhase=data.cancellable===false?`${data.phase||''}-sync`:data.phase||'';status.textContent=({preparing:'Preparing trajectories…',direct:'Evaluating direct modes…', 'direct-sync':'Evaluating CPU direct modes…','finite-difference':'Evaluating GPU finite differences…','finite-difference-sync':'Checking the CPU finite difference…','pm-evolution':'Evolving PM trajectories on GPU…','observations-sync':'Depositing the sheet exactly…','pm-recovery-sync':'Recovering PM trajectories on CPU…',finalizing:'Finalizing observables…'}[activePhase]||'Computing…');notice.textContent=data.message||'Running the ΛCDM Lagrangian calculation in a background worker.';return;}
    if(data.type==='progress'&&data.requestId===activeRequestId){if(data.phase)activePhase=data.phase;const completed=data.completed??data.step??data.caseIndex,total=data.total??data.steps??data.cases;status.textContent=`${data.stage||'Evaluating GPU modes'}${completed!=null?` (${completed}/${total})`:''}…`;return;}
    if(data.type==='cancelled'&&data.requestId===activeRequestId){clearTimeout(cancelTimer);cancelTimer=0;activeRequestId=null;activePhase='';activeKey='';if(pendingRun){const next=pendingRun;pendingRun=null;dispatchRun(next);}return;}
    if(data.type==='shutdown-ack'){hardTerminateWorker();return;}
    if(data.type!=='result'||data.requestId!==activeRequestId)return;
    clearTimeout(cancelTimer);cancelTimer=0;activeRequestId=null;activePhase='';activeKey='';runMilliseconds=data.durationMilliseconds??null;
    if(data.error){current={error:data.error,execution:data.execution||null};execution=data.execution||null;computedFor='';status.textContent='Run failed';}
    else{current=data.result;execution=current?.execution||data.execution||null;lastResult=current;lastResultKey=data.key;computedFor=data.key;status.textContent=current?.guard?.ok===false?'Post-crossing · interpret cautiously':current?.error?'Run failed':'Ready';exportJsonButton.disabled=false;}
    render();
    if(pendingRun){const next=pendingRun;pendingRun=null;if(next.key!==data.key)dispatchRun(next);}
  };
  worker.onerror=(event)=>{clearTimeout(cancelTimer);cancelTimer=0;worker?.terminate();worker=null;activeRequestId=null;activePhase='';activeKey='';computedFor='';current={error:event.message||'Could not start the numerical worker.'};status.textContent='Run failed';render();};
  return worker;
}
function dispatchRun(job){const w=ensureWorker(),id=++requestId;activeRequestId=id;activePhase='queued';activeKey=job.key;runMilliseconds=null;execution=null;exportJsonButton.disabled=true;status.textContent='Preparing…';notice.textContent='Preparing the ΛCDM trajectories and observations in a background worker.';w.postMessage({type:'run',requestId:id,key:job.key,state:job.request,policy:job.policy});}
function cancelActiveRun(){if(!worker||activeRequestId==null)return;const id=activeRequestId;if(['preparing','preparing-sync','finalizing','finalizing-sync','direct-sync','direct-recovery-sync','finite-difference-sync','observations-sync','pm-recovery-sync','queued'].includes(activePhase)){hardTerminateWorker();activeRequestId=null;activePhase='';activeKey='';if(pendingRun){const next=pendingRun;pendingRun=null;dispatchRun(next);}return;}worker.postMessage({type:'cancel',requestId:id});status.textContent='Stopping obsolete GPU work…';clearTimeout(cancelTimer);cancelTimer=setTimeout(()=>{if(activeRequestId!==id)return;hardTerminateWorker();activeRequestId=null;activePhase='';activeKey='';if(pendingRun){const next=pendingRun;pendingRun=null;dispatchRun(next);}},5000);}
function cancelObsoleteForCachedResult(){if(activeRequestId!=null||pendingRun)retireWorker();}
function retireWorker(){clearTimeout(cancelTimer);cancelTimer=0;const old=worker;worker=null;pendingRun=null;activeRequestId=null;activePhase='';activeKey='';if(!old)return;old.onmessage=({data})=>{if(data?.type==='shutdown-ack')old.terminate();};old.onerror=()=>old.terminate();try{old.postMessage({type:'shutdown'});}catch(_){old.terminate();return;}setTimeout(()=>old.terminate(),5000);}
function hardTerminateWorker(){clearTimeout(cancelTimer);cancelTimer=0;const old=worker;worker=null;if(!old)return;old.onmessage=null;old.onerror=null;old.terminate();}

function explain() {
  const messages = {
    sheet: S.dim === 1
      ? 'Each Lagrangian label q maps to position x and velocity dx/da. The curve remains visible after orientation loss, with the single-stream interpretation explicitly flagged.'
      : S.dim === 2
        ? 'Each Lagrangian label q maps to a position (x₀,x₁). The triangulated plane shows the deformed sheet in position space; the velocity coordinates are not projected into this view.'
        : 'Each Lagrangian label q maps to a position (x₀,x₁,x₂). This wireframe projects the three-dimensional position sheet from the full six-dimensional (x,v) phase-space sheet.',
    density: 'Both maps integrate the P₀ sheet exactly at the LPT and PM positions, then apply the selected Gaussian observation filter. Square map pixels retain equal spatial-axis scales. The PM force grid and observation grid are independent.',
    spectra: `Compare Fourier estimates in physical units: k is h/Mpc and P(k) is (Mpc/h)^${S.dim}. Lower-dimensional translationally invariant runs are not directly comparable to 3D power spectra.`,
    pdf: 'Volume weighting samples space uniformly. Mass weighting uses unsmoothed raw cell mass. Observation smoothing is separate from the initial-condition filter; smoothed density values with raw mass weights are not related by a simple ρ pV transform.',
    gradient: 'Compare the derivative of continuous P₀ Fourier power with respect to initial amplitude A at fixed epoch against a finite difference. This does not differentiate clipped or deposited density.',
    published: 'This panel replays packaged reference data. These curves retain the original simulation conditions and estimator conventions; the live laboratory runs the independently specified cosmology and resolution.',
  };
  viewHelp.textContent = messages[S.view];
}
function setNotice(text, kind = '') { notice.textContent = text; notice.className = `lab-notice ${kind}`; }
function num(v, digits = 3) { return v == null || v === '' || !Number.isFinite(Number(v)) ? '—' : Number(v).toPrecision(digits); }
function arr(a) { return ArrayBuffer.isView(a) ? a : Array.isArray(a) ? a : []; }
function clearHost() { viewHost.replaceChildren(); diagnostics.replaceChildren(); backendMeta.replaceChildren(); }
function panel(title, description = '') { const p = $('section', 'result-panel', viewHost); $('h3', '', p, title); if (current?.guard?.ok===false) $('p','post-crossing-note',p,'Post-crossing diagnostic. The map has lost positive orientation; the plotted density and perturbative fields are not a single-stream prediction.');if (description) $('p', 'panel-note', p, description); return p; }
function addPngButton(canvas, name) { const b = $('button', 'export-button', canvas.parentElement, 'Save PNG'); b.type = 'button'; b.setAttribute('aria-label', `Save ${name} image as PNG`); b.addEventListener('click', () => canvas.toBlob((blob) => { if (blob) { downloadBlob(blob, `${name}.png`); flash(b, 'Saved'); } })); }
function canvasFor(parent, label, wide = false) { const c = $('canvas', `lab-canvas${wide ? ' canvas-wide' : ''}`, parent); c.width = 960; c.height = wide ? 560 : 680; c.setAttribute('role', 'img'); c.setAttribute('aria-label', label); return c; }
function paintCanvas(c, draw) { const ctx = c.getContext('2d'); ctx.clearRect(0, 0, c.width, c.height); ctx.fillStyle = getComputedStyle(c).backgroundColor; ctx.fillRect(0,0,c.width,c.height); draw(ctx,c.width,c.height); }
function hasScientificOutput(d){return Boolean(d&&(d.q||d.x||d.rho||d.spectra||d.pdf||d.gradient));}
function guarded() {
  if (current?.error || current?.errorMessage) { setNotice(`The numerical worker could not complete this run: ${current.error || current.errorMessage}`, 'error'); return false; }
  if (!current) { setNotice('Waiting for the calculation to finish.'); return false; }
  if (current.guard?.ok === false) { const why = current.guard.reason || 'A simplex changed orientation.'; setNotice(`Post-crossing diagnostic: ${why} Minimum LPT vertex Jacobian: ${num(current.guard.minimumJacobian,4)}; minimum LPT/PM simplex Jacobian: ${num(current.guard.minimumSimplexJacobian,4)}. Outputs are shown with caution; they no longer represent an everywhere one-to-one single-stream map.`, 'guard'); return hasScientificOutput(current); }
  if (current.error) { setNotice(current.error, 'error'); return false; }
  setNotice('Resolved LPT orientations and continuous extrema checks within every PM drift remained positive.', 'success'); return true;
}
function render() {
  explain(); clearHost();
  if (S.view === 'published') { renderPublished(); return; }
  if (computedFor && computedFor !== physicsKey()) { setNotice('The controls changed. Waiting for the matching guarded calculation.'); return; }
  if (!guarded()) { showDiagnostics(current); return; }
  if (S.view === 'sheet') renderSheet(current);
  else if (S.view === 'density') renderDensity(current);
  else if (S.view === 'spectra') renderSpectra(current);
  else if (S.view === 'pdf') renderPdfs(current);
  else if (S.view === 'gradient') renderGradient(current);
  showDiagnostics(current);
  renderTheory();
}
function project3(x,y,z,rotation,tilt,scale,cx,cy) { const a=rotation*Math.PI/180,b=tilt*Math.PI/180; const X=x*Math.cos(a)-y*Math.sin(a), Y=x*Math.sin(a)+y*Math.cos(a); return [cx+scale*X,cy+scale*(Y*Math.sin(b)-z*Math.cos(b))]; }
function renderSheet(d) {
  const dimension=Number(d.dim||S.dim),caption=dimension===1?'A curve in position–velocity space: x horizontally and dx/da vertically.':dimension===2?'A triangulated sheet in the x₀–x₁ position plane, with equal scales on both spatial axes.': 'An orthographic position projection of the 3D Lagrangian sheet in the full six-dimensional (x,v) phase space. Spatial axes use a common scale; drag or use controls to rotate.';
  const p=panel(dimension===1?'One-dimensional phase-space curve':dimension===2?'Two-dimensional triangulated position sheet':'Three-dimensional tetrahedral position-sheet projection', `Coordinates are fractions of the ${num(S.boxSize,5)} Mpc/h periodic box. ${caption}`);
  const alt=dimension===1?'One-dimensional phase-space curve with position x and velocity dx/da':dimension===2?'Triangulated two-dimensional Lagrangian sheet in the x0–x1 position plane with equal spatial-axis scales':'Rotatable, equal-scale orthographic position projection of the three-dimensional Lagrangian sheet embedded in six-dimensional phase space';
  const c=canvasFor(p,alt,true);
  const draw=()=>paintCanvas(c,(ctx,W,H)=>{
    const n=Number(d.n||S.n),dim=Number(d.dim||S.dim),q=arr(d.q),x=arr(d.x),v=arr(d.v); const nv=Math.floor(Math.min(q.length,x.length,v.length)/(dim||1));
    if (!nv) { ctx.fillStyle=getComputedStyle(c).color; ctx.fillText('Sheet coordinates were not returned.',32,40); return; }
    const idx=(i,j=0,k=0)=>dim===1?i:dim===2?i*n+j:(i*n+j)*n+k;
    const budget=dim===1?2000:dim===2?10000:8500,stride=Math.max(1,Math.ceil((n**dim/budget)**(1/dim)));
    const sampleAxis=()=>{const a=[];for(let i=0;i<n;i+=stride)a.push(i);if(a.at(-1)!==n-1)a.push(n-1);return a;},axis=sampleAxis();
    const get=(arr,index,k)=>Number(arr[index*dim+k]??0);
    const project=(index)=>{let a,b;if(dim===1){a=get(x,index,0);b=get(v,index,0);}else if(dim===2){a=get(x,index,0);b=get(x,index,1);}else [a,b]=project3(get(x,index,0),get(x,index,1),get(x,index,2),S.rotate,S.tilt,1,0,0);return [a,b];};
    let minx=Infinity,maxx=-Infinity,miny=Infinity,maxy=-Infinity;
    if(dim===1){for(let i=0;i<nv;i++){const [a,b]=project(i);minx=Math.min(minx,a);maxx=Math.max(maxx,a);miny=Math.min(miny,b);maxy=Math.max(maxy,b);}}
    else if(dim===2){for(const i of axis)for(const j of axis){const [a,b]=project(idx(i,j));minx=Math.min(minx,a);maxx=Math.max(maxx,a);miny=Math.min(miny,b);maxy=Math.max(maxy,b);}}
    else for(const i of axis)for(const j of axis)for(const k of axis){const [a,b]=project(idx(i,j,k));minx=Math.min(minx,a);maxx=Math.max(maxx,a);miny=Math.min(miny,b);maxy=Math.max(maxy,b);}
    if(minx===maxx){minx-=1;maxx+=1;}if(miny===maxy){miny-=1;maxy+=1;}
    const pad=55,plotW=W-2*pad,plotH=H-2*pad,sc=Math.min(plotW/(maxx-minx),plotH/(maxy-miny)),centerX=W/2,centerY=H/2;
    const xyFor=(index)=>{const [a,b]=project(index);if(dim===1)return [pad+(a-minx)*plotW/(maxx-minx),H-pad-(b-miny)*plotH/(maxy-miny)];return [centerX+(a-(minx+maxx)/2)*sc,centerY-(b-(miny+maxy)/2)*sc];};
    ctx.strokeStyle=getComputedStyle(c).color;ctx.lineWidth=dim===3?.65:1;ctx.globalAlpha=dim===3?.25:.72;
    const edge=(a,b)=>{if(a<nv&&b<nv){const u=xyFor(a),v2=xyFor(b);ctx.beginPath();ctx.moveTo(...u);ctx.lineTo(...v2);ctx.stroke();}};
    if(dim===1){for(let i=0;i+1<n;i++)edge(i,i+1);}
    else if(dim===2){for(let ia=0;ia+1<axis.length;ia++)for(let ja=0;ja+1<axis.length;ja++){const i=axis[ia],j=axis[ja],ii=axis[ia+1],jj=axis[ja+1],a=idx(i,j),b=idx(ii,j),cc=idx(i,jj),e=idx(ii,jj);edge(a,b);edge(a,cc);edge(a,e);edge(b,e);edge(cc,e);}}
    else {for(let ia=0;ia+1<axis.length;ia++)for(let ja=0;ja+1<axis.length;ja++)for(let ka=0;ka+1<axis.length;ka++){const i=axis[ia],j=axis[ja],k=axis[ka],ii=axis[ia+1],jj=axis[ja+1],kk=axis[ka+1],a=idx(i,j,k),b=idx(ii,j,k),cc=idx(i,jj,k),dd=idx(i,j,kk),e=idx(ii,jj,k),f=idx(ii,j,kk),g=idx(i,jj,kk),h=idx(ii,jj,kk);for(const [u,v2] of [[a,b],[a,cc],[a,dd],[a,e],[a,f],[a,g],[a,h],[b,e],[b,f],[cc,e],[cc,g],[dd,f],[dd,g],[e,h],[f,h],[g,h]])edge(u,v2);}}
    ctx.globalAlpha=1;if(dim===1){for(let i=0;i<n;i++){const [px,py]=xyFor(i),t=i/Math.max(1,n-1);ctx.fillStyle=`hsl(${210-195*t} 76% 42%)`;ctx.beginPath();ctx.arc(px,py,2,0,Math.PI*2);ctx.fill();}}else if(dim===2){for(const i of axis)for(const j of axis){const [px,py]=xyFor(idx(i,j)),t=(i*n+j)/Math.max(1,n*n-1);ctx.fillStyle=`hsl(${210-195*t} 76% 42%)`;ctx.beginPath();ctx.arc(px,py,1.8,0,Math.PI*2);ctx.fill();}}else for(const i of axis)for(const j of axis)for(const k of axis){const [px,py]=xyFor(idx(i,j,k)),t=((i*n+j)*n+k)/Math.max(1,n**3-1);ctx.fillStyle=`hsl(${210-195*t} 76% 42%)`;ctx.beginPath();ctx.arc(px,py,1.5,0,Math.PI*2);ctx.fill();}
    ctx.fillStyle=getComputedStyle(c).color;ctx.font='15px system-ui';ctx.textAlign='center';ctx.fillText(dim===1?'position x (box fraction)':dim===2?'position x₀/L':'orthographic horizontal / L',W/2,H-13);ctx.save();ctx.translate(18,H/2+28);ctx.rotate(-Math.PI/2);ctx.fillText(dim===1?'velocity dx/da':dim===2?'position x₁/L':'orthographic vertical / L',0,0);ctx.restore();ctx.textAlign='start';
  }); draw(); addPngButton(c,'phase-space-sheet');
  if(dimension>1){const n=Number(d.n||S.n),shown=Math.min(n,Math.ceil(n/Math.max(1,Math.ceil((n**dimension/(dimension===2?10000:8500))**(1/dimension)))));$('p','panel-note',p,`Display-only lattice decimation: approximately ${shown} samples per axis are drawn for legibility. The experiment and exported result retain all ${n} vertices per axis.`);}
  if(S.dim===3){let drag=null;c.style.touchAction='none';c.addEventListener('pointerdown',ev=>{drag=[ev.clientX,ev.clientY,S.rotate,S.tilt];c.setPointerCapture(ev.pointerId);});c.addEventListener('pointermove',ev=>{if(!drag)return;S.rotate=Math.max(-180,Math.min(180,drag[2]+(ev.clientX-drag[0])*.55));S.tilt=Math.max(-80,Math.min(80,drag[3]+(ev.clientY-drag[1])*.45));syncRotationControls();updateURL();draw();});for(const event of ['pointerup','pointercancel','lostpointercapture'])c.addEventListener(event,()=>{drag=null;});
    const controlsBox=$('div','rotation-controls',viewHost);$('span','control-caption',controlsBox,'Rotate 3D projection by dragging the sheet or using the angle controls');const rotationInputs={};for(const [key,label,min,max] of [['rotate','Azimuth',-180,180],['tilt','Elevation',-80,80]]){const l=$('label','rotation-range',controlsBox);$('span','',l,label);const r=$('input','',l);r.type='range';r.min=min;r.max=max;r.value=S[key];r.setAttribute('aria-label',`${label} sheet angle`);r.addEventListener('input',()=>{S[key]=Number(r.value);updateURL();draw();});rotationInputs[key]=r;}function syncRotationControls(){for(const [key,input] of Object.entries(rotationInputs))input.value=S[key];}}
}
function renderDensity(d) {
  const p=panel('Exact P₀-sheet density on LPT and PM trajectories', `Both maps start from exact P₀ cell averages and apply a Gaussian observation filter R/L = ${num(S.smoothing,4)} (${num(S.smoothing*S.boxSize,4)} Mpc/h). They use the same spatial axis scale and common ρ/mean color range. For 3D, the view is the central x₀–x₁ slice at x₂/L≈0.5. Observation grid: ${S.mesh} cells per axis; force grid: ${S.forceMesh} cells per axis.`);
  const maps=$('div','density-pair',p), dim=Number(d.dim||S.dim), mesh=Number(d.mesh||S.mesh);
  const exact=arr(d.rho),pm=arr(d.pmRho);let sum=0,count=0,maximum=0;for(const source of [exact,pm])for(let i=0;i<source.length;i++){const value=Number(source[i])||0;sum+=value;count++;maximum=Math.max(maximum,value);}const mean=count?sum/count:1,sharedMax=maximum/(mean||1);
  drawDensityMap(maps,'LPT sheet density',exact,'Observed density from exact LPT sheet deposition',dim,mesh,'exact-deposit-density',sharedMax,mean);
  drawDensityMap(maps,'PM trajectory sheet density',pm,'Observed density from exact sheet deposition along particle-mesh trajectories',dim,mesh,'pm-trajectory-sheet-density',sharedMax,mean);
}
function drawDensityMap(parent,title,values,label,dim,mesh,name,scaleMax,mean){
  const block=$('div','density-map',parent);$('h4','',block,title);const c=canvasFor(block,label,true),rho=arr(values);
  paintCanvas(c,(ctx,W,H)=>{
    const ink=getComputedStyle(c).color;ctx.fillStyle=ink;
    if(!rho.length){ctx.fillText('Density cells were not returned.',32,42);return;}
    const n=mesh,max=scaleMax||1,left=48,right=W-40,top=24,bottom=H-98;
    if(dim===1){for(let i=0;i<rho.length;i++){const height=Math.max(0,rho[i]/((mean||1)*max))*(bottom-top);ctx.fillStyle=densityColor(rho[i]/((mean||1)*max));ctx.fillRect(left+i*(right-left)/rho.length,bottom-height,Math.max(1,(right-left)/rho.length),height);}}
    else {const side=Math.min(right-left,bottom-top),x0=(W-side)/2,y0=top+(bottom-top-side)/2,cell=side/n,kMid=Math.floor(mesh/2);for(let i=0;i<n;i++)for(let j=0;j<n;j++){const index=dim===3?(i*n+j)*n+kMid:i*n+j;const val=Number(rho[index]??0);ctx.fillStyle=densityColor(val/((mean||1)*max));ctx.fillRect(x0+i*cell,y0+(n-1-j)*cell,cell+.5,cell+.5);}
      ctx.strokeStyle=ink;ctx.lineWidth=1;ctx.strokeRect(x0,y0,side,side);
      for(const [v,x] of [['0',x0],['0.5',x0+side/2],['1',x0+side]]){ctx.fillStyle=ink;ctx.fillText(v,x-4,y0+side+17);ctx.beginPath();ctx.moveTo(x,y0+side);ctx.lineTo(x,y0+side+4);ctx.stroke();}
      for(const [v,y] of [['0',y0+side],['0.5',y0+side/2],['1',y0]])ctx.fillText(v,x0-26,y+4);
      ctx.fillStyle=ink;ctx.textAlign='center';ctx.fillText(`x₀/L  (L = ${num(S.boxSize,5)} Mpc/h)`,x0+side/2,y0+side+34);ctx.save();ctx.translate(x0-43,y0+side/2);ctx.rotate(-Math.PI/2);ctx.fillText('x₁/L',0,0);ctx.restore();ctx.textAlign='start';}
    ctx.fillStyle=ink;ctx.font='12px system-ui';
    const gx=left+85,gy=H-30,gw=Math.max(80,right-left-175,80),gh=10,grad=ctx.createLinearGradient(gx,0,gx+gw,0);for(let i=0;i<=20;i++)grad.addColorStop(i/20,densityColor(i/20));ctx.fillStyle=grad;ctx.fillRect(gx,gy,gw,gh);ctx.fillStyle=ink;ctx.fillText('ρ / mean',left,gy+9);ctx.fillText('0',gx-12,gy-5);ctx.fillText(num(max,3),gx+gw+4,gy-5);
  });addPngButton(c,name);
}
function densityColor(t){const x=Math.max(0,Math.min(1,t));return `hsl(${218-205*x} ${50+40*x}% ${97-53*x}%)`;}
function plotHost(parent){return $('div','plot-host',parent);}
function addPlot(parent,title,x,seriesDefs,axes,d) {
  const box=$('section','plot-panel',parent);$('h4','',box,title); const host=plotHost(box), plot=new LinePlot(host,{width:720,height:390,interactive:true});
  const xs=arr(x); if(!xs.length){$('p','panel-note',box,'This run did not return data for this chart.');return;}
  const series=[]; for(const def of seriesDefs){const ys=arr(d?.[def.key] ?? d?.spectra?.[def.key] ?? d?.pdf?.[def.key] ?? d?.gradient?.[def.key]);const sx=arr(def.xkey?(d?.[def.xkey]??d?.spectra?.[def.xkey]??d?.gradient?.[def.xkey]):xs);if(ys.length&&sx.length)series.push({x:sx.slice(0,Math.min(sx.length,ys.length)),y:ys.slice(0,Math.min(sx.length,ys.length)),label:def.label});}
  if(!series.length){$('p','panel-note',box,'The selected output did not include the requested curves.');return;}
  plot.setAxes({...axes,title,xlog:axes.xlog??false,ylog:axes.ylog??false});plot.setSeries(series);plot.draw();
}
function renderSpectra(d) {const p=panel('Power spectrum comparison','The first chart compares isotropically binned FFT estimates. The second compares direct first-axis modes on their own deterministic k grid. These are physical spectra: k in h/Mpc and P(k) in (Mpc/h)^d.');addPlot(p,'Isotropic FFT-bin power',d.spectra?.k,[{key:'sheet',label:'Exact sheet'},{key:'cic',label:'CIC particles'},{key:'pm',label:'PM trajectories'}],{xlabel:'k [h/Mpc]',ylabel:`P(k) [(Mpc/h)^${S.dim}]`,xlog:true,ylog:true},d);addPlot(p,'Direct first-axis modes',d.spectra?.directK,[{key:'direct',label:'Direct sheet LPT'},{key:'directPM',label:'Direct PM'}],{xlabel:'k [h/Mpc]',ylabel:`P(k) [(Mpc/h)^${S.dim}]`,xlog:true,ylog:true},d);}
function renderPdfs(d) {const p=panel('Density probability distributions','The selected weighting is charted first and marked as primary. Volume PDFs use the observed density. Mass PDFs use raw unsmoothed cell mass as weights, including when the density value has been smoothed.');$('p','panel-note',p,`Primary weighting: ${S.weight}. Observation filter R/L = ${num(S.smoothing,4)}. The selected density range may not capture all probability; captured integrals are reported below.`);const volumeDefs=[{key:'volume',label:'LPT sheet'},{key:'pmVolume',label:'PM trajectories'}],massDefs=[{key:'mass',label:'LPT sheet'},{key:'pmMass',label:'PM trajectories'}];if(S.weight==='mass'){addPlot(p,'PRIMARY · mass-weighted PDF',d.pdf?.logrho,massDefs,{xlabel:'log₁₀ ρ',ylabel:'mass-weighted density',ylim:[0,undefined]},d);addPlot(p,'Volume-weighted PDF',d.pdf?.logrho,volumeDefs,{xlabel:'log₁₀ ρ',ylabel:'volume-weighted density',ylim:[0,undefined]},d);}else{addPlot(p,'PRIMARY · volume-weighted PDF',d.pdf?.logrho,volumeDefs,{xlabel:'log₁₀ ρ',ylabel:'volume-weighted density',ylim:[0,undefined]},d);addPlot(p,'Mass-weighted PDF',d.pdf?.logrho,massDefs,{xlabel:'log₁₀ ρ',ylabel:'mass-weighted density',ylim:[0,undefined]},d);}renderPdfSummaries(p,d.pdf||{});}
function renderPdfSummaries(parent,pdf){const table=$('table','summary-table',parent),caption=$('caption','',table,'Full-field summaries and displayed-bin probability integrals');const head=$('thead','',table),hr=$('tr','',head);for(const label of ['Statistic','LPT sheet','PM trajectories'])$('th','',hr,label);const body=$('tbody','',table),lpt=pdf.summary||{},pm=pdf.pmSummary||{},qps=arr(lpt.probabilities),lq=arr(lpt.volumeQuantiles),lmq=arr(lpt.massQuantiles),pq=arr(pm.volumeQuantiles),pmq=arr(pm.massQuantiles);const rows=[['Mean density',lpt.mean,pm.mean],['Variance',lpt.variance,pm.variance],['Reduced skewness S₃',lpt.S3,pm.S3],['Reduced kurtosis S₄',lpt.S4,pm.S4],['Captured volume-PDF probability',pdf.capturedVolume,pdf.pmCapturedVolume],['Captured mass-PDF probability',pdf.capturedMass,pdf.pmCapturedMass]];for(let i=0;i<qps.length;i++){const suffix=` (${num(Number(qps[i])*100,3)}%)`;rows.push([`Volume-weighted ρ quantile${suffix}`,lq[i],pq[i]],[`Mass-weighted ρ quantile${suffix}`,lmq[i],pmq[i]]);}for(const [label,a,b] of rows){const tr=$('tr','',body);$('th','',tr,label);$('td','',tr,num(a,5));$('td','',tr,num(b,5));}}
function renderGradient(d) {const p=panel('Initial-amplitude response at fixed epoch','This compares continuous P₀ Fourier power as initial amplitude A changes. The derivative is not taken through clipped or deposited density. The finite-difference check validates the sheet-LPT tangent; the PM tangent is shown as its own response.');addPlot(p,'Amplitude derivative by wavenumber',d.gradient?.k,[{key:'power',label:'P₀ power'},{key:'tangent',label:'Sheet-LPT tangent'},{key:'finiteDifference',label:'Sheet-LPT finite difference'},{key:'pmTangent',label:'PM tangent'}],{xlabel:'k [h/Mpc]',ylabel:`P₀ and ∂P₀/∂A [(Mpc/h)^${S.dim}]`,xlog:true},d);const err=arr(d.gradient?.relativeError);if(err.length)$('p','panel-note',p,`Maximum sheet-LPT tangent relative error: ${num(err.reduce((m,v)=>Math.max(m,Number(v)||0),0),4)}.`);}
function showDiagnostics(d) {
  if(!d)return;
  showExecution(d.execution||execution);
  const g=d.guard||{},m=d.diagnostics||{},p=$('div','metric-strip',diagnostics);
  const percent=v=>Number.isFinite(v)?`${num(100*v,3)}%`:'—';
  for(const [label,val] of [
    ['Output epoch a',num(d.a,6)],
    ['Redshift z',num(d.redshift,5)],
    ['Linear growth D(a)',num(d.D,6)],
    ['Growth D(1)',num(d.growthAtOne,6)],
    ['Crossing reference D',num(d.shellCrossing,6)],
    ['Crossing scale factor',num(d.aShellCrossing,6)],
    ['Minimum LPT vertex Jacobian',num(g.minimumJacobian,4)],
    ['Minimum LPT/PM simplex Jacobian',num(g.minimumSimplexJacobian,4)],
    ['LPT final inverted mass',percent(g.lptFinalInvertedMass)],
    ['PM final inverted mass',percent(g.pmFinalInvertedMass)],
    ['LPT mass ratio',num(m.mass,8)],
    ['PM mass ratio',num(m.pmMass,8)],
    ['Position RMS (box units)',num(m.positionRms,4)],
    ['Relative position RMS',percent(m.positionRelativeRms)],
    ['Max low-bin power residual',percent(m.powerRelativeError)],
  ]) {
    const item=$('div','metric',p);
    $('span','metric-label',item,label);$('strong','metric-value',item,val);
  }
  if(d.diagnostics){const crossing=g.firstPmCrossing,first=crossing?` First PM orientation loss occurred during step ${crossing.step} (a=${num(crossing.aStart,5)}–${num(crossing.aEnd,5)}; continuous drift minimum J=${num(crossing.minimumJacobian,5)}).`:'';$('p','panel-note',diagnostics,`Epoch a=${num(d.a,6)} (z=${num(d.redshift,5)}); D(a)=${num(d.D,6)} and D(1)=${num(d.growthAtOne,6)}. The crossing reference is ${d.crossingReference||'a diagnostic estimate'}, with Dsc=${num(d.shellCrossing,6)}${d.aShellCrossing==null?' (not reached within the growth-table range)':`; a_sc=${num(d.aShellCrossing,6)}`}. LPT final inverted-mass fractions count orientation-reversed elements; PM minima use continuous determinant extrema over each drift.${first} Mass ratios compare deposited with initial mass. Relative position RMS is normalized by the LPT displacement RMS. The power residual is max |P_LPT/P_PM − 1| over the first four populated FFT bins; weak-power bins can give large ratios. This finite-resolution PM run requires a convergence check.`);}
}
function showExecution(meta){
  if(!meta)return;
  backendMeta.replaceChildren();
  $('h3','control-subheading',backendMeta,'Run backend and precision');
  const grid=$('div','execution-meta',backendMeta);
  const compact=(label,value)=>{if(value==null||value==='')return;const wide=['Precision','GPU validation','Validation budget','PM evolution','PM validation','PM timings','Finite difference','FD validation','FD timings','Timings','Stage backends'].includes(label);const item=$('div',`execution-meta-item${wide?' execution-meta-item-wide':''}`,grid);$('span','metric-label',item,label);$('strong','metric-value',item,String(value));};
  compact('Requested backend',meta.requestedBackend);
  compact('Executed backend',meta.executedBackend);
  compact('Precision',meta.precision);
  const pm=meta.pmEvolution;
  if(pm){
    compact('PM evolution',`${pm.backend} · ${pm.precision}${pm.fallbackReason?` · ${pm.fallbackReason}`:''}`);
    if(pm.validation)compact('PM validation',`${pm.validation.status} · ${pm.validation.rows?.length||0} FFT/force/trajectory checks`);
    if(pm.timings){const labels={compileMs:'setup',uploadMs:'upload',kernelMs:'force/KDK',diagnosticMs:'initial orientation',cpuCheckMs:'CPU orientation rechecks',readbackMs:'final readback',totalMs:'total'};compact('PM timings',Object.entries(pm.timings).filter(([,v])=>typeof v==='number'&&Number.isFinite(v)).map(([k,v])=>`${labels[k]||k} ${num(v,3)} ms`).join(' · '));}
  }
  const fd=meta.finiteDifference;
  if(fd){
    compact('Finite difference',`${fd.backend} · h=${fd.h} · ${fd.precision}${fd.fallbackReason?` · ${fd.fallbackReason}`:''}`);
    if(fd.validation)compact('FD validation',`${fd.validation.status} · ${fd.validation.rows?.length||0} endpoint/finite-difference checks`);
    if(fd.timings)compact('FD timings',Object.entries(fd.timings).filter(([,v])=>typeof v==='number'&&Number.isFinite(v)).map(([k,v])=>`${k.replace(/Ms$/,'')} ${num(v,3)} ms`).join(' · '));
  }
  if(meta.fallbackReason){const item=$('p','execution-fallback',backendMeta,meta.fallbackReason);item.setAttribute('role','note');}

  const device=meta.device||meta.gpuAttempt?.device;
  if(device){
    const info=device.adapterInfo||{},identity=[info.vendor,info.architecture,info.device||info.description].filter(Boolean).join(' · ');
    compact('GPU adapter',identity||'Adapter details unavailable');
    const limits=device.limits||{},limitParts=[];
    for(const key of ['maxStorageBufferBindingSize','maxBufferSize'])if(Number.isFinite(Number(limits[key])))limitParts.push(`${key==='maxBufferSize'?'buffer':'storage'} ${(Number(limits[key])/1048576).toFixed(0)} MiB`);
    for(const [key,label] of [['maxComputeWorkgroupSizeX','max workgroup width'],['maxComputeWorkgroupsPerDimension','max groups per axis']])if(Number.isFinite(Number(limits[key])))limitParts.push(`${label} ${limits[key]}`);
    if(limitParts.length)compact('Relevant GPU limits',limitParts.join(' · '));
  }

  const validation=meta.validation||meta.device?.validation||meta.gpuAttempt?.validation||meta.gpuAttempt?.device?.validation;
  if(validation){
    const errors=Array.isArray(validation.errors)?validation.errors:[],maxOf=(key)=>errors.reduce((max,row)=>typeof row?.[key]==='number'&&Number.isFinite(row[key])?Math.max(max,row[key]):max,0);
    const status=validation.status||'unknown',fixtureCount=errors.length;
    const summary=[status,fixtureCount?`${fixtureCount} fixtures`:'no fixture records',`worst |primal| ${num(maxOf('maxPrimalAbs'),3)}`,`worst |JVP| ${num(maxOf('maxJvpAbs'),3)}`].join(' · ');
    compact('GPU validation',summary);
    const budget=validation.budget||{};
    const budgetText=[budget.status,(budget.primal||budget.jvp)&&'independent f64 quadrature: abs ≤ 1e-6 + rel ≤ 1e-4 (primal/JVP)',budget.amplitudeDerivative&&'full-request amplitude gate ≤ 1e-5 remains separate'].filter(Boolean).join(' · ');
    if(budgetText)compact('Validation budget',budgetText);
  }

  const timings=meta.timings||meta.gpuAttempt?.timings||{},stages=timings.stages||{};
  const stageLabels={preparationMilliseconds:'preparation',directMilliseconds:device?'direct/setup':'CPU direct',finiteDifferenceMilliseconds:`${fd?.backend==='gpu'?'GPU':'CPU'} finite difference`,cpuRecoveryMilliseconds:'CPU recovery',finalizationMilliseconds:stages.cpuRecoveryMilliseconds>0?'finalization (includes recovery)':'finalization'};
  const stageParts=Object.entries(stages).filter(([,v])=>typeof v==='number'&&Number.isFinite(v)).map(([k,v])=>`${stageLabels[k]||k.replace(/Milliseconds$/,'')} ${num(v,3)} ms`);
  if(typeof timings.kernelMs==='number'&&Number.isFinite(timings.kernelMs))stageParts.unshift(`GPU kernel ${num(timings.kernelMs,3)} ms`);
  if(typeof timings.totalMs==='number'&&Number.isFinite(timings.totalMs))stageParts.unshift(`GPU total ${num(timings.totalMs,3)} ms`);
  if(stageParts.length)compact('Timings',stageParts.join(' · '));

  const backends=meta.stageBackends||{};
  const direct=backends.directPrimal||{};
  const directSummary=['LPT','PM'].map((name)=>{const item=direct[name.toLowerCase()];return item?`${name} ${item.provenance||item.backend||'unknown'}`:null;}).filter(Boolean);
  const precisionSummary=[];
  if(backends.trajectories)precisionSummary.push(`nLPT fields f32 → LPT state/tangent f64 · PM ${backends.trajectories.pmBackend||'cpu'} ${backends.trajectories.pmStatePrecision||'f64'}`);
  if(backends.exactDeposit)precisionSummary.push('deposit f32 positions→f64 geometry/mass→f32 density');
  if(backends.pdfs)precisionSummary.push('PDF f32 density→f64 stats');
  if(backends.griddedSpectra)precisionSummary.push('FFT f32 field→f64 transform/bins');
  if(backends.finiteDifference)precisionSummary.push(`FD ${backends.finiteDifference.backend} · ${backends.finiteDifference.precision}`);
  if(backends.finalization)precisionSummary.push('finalization f64');
  if(precisionSummary.length||directSummary.length)compact('Stage backends',[...precisionSummary,...directSummary].join(' · '));

  if(Array.isArray(meta.modeProvenance)&&meta.modeProvenance.length){
    const table=$('table','summary-table execution-mode-table',backendMeta),caption=$('caption','',table,'Direct-mode channel precision');
    const head=$('thead','',table),row=$('tr','',head);for(const label of ['Integer mode','LPT','PM'])$('th','',row,label);
    const body=$('tbody','',table);
    for(const mode of meta.modeProvenance){const tr=$('tr','',body),integerMode=Array.isArray(mode.integerMode)?mode.integerMode.join(', '):String(mode.integerMode??'unknown');$('th','',tr,integerMode);$('td','',tr,mode.lpt||'unknown');$('td','',tr,mode.pm||'unknown');}
  }

  const details=$('details','published-provenance execution-raw',backendMeta);$('summary','',details,'Full execution metadata');
  const pre=$('pre','execution-raw-json',details);pre.textContent=JSON.stringify(meta,null,2);
}
function renderTheory(){theory.replaceChildren();$('h2','section-heading',theory,'Interpretation and conventions');const text={sheet:'The resolved LPT vertex orientations and continuous determinant extrema along each PM drift are monitored throughout the evolution. Beyond the first orientation loss, the displayed density remains a sum of positive streams, while single-stream fluid interpretation and controlled pre-crossing LPT validity no longer apply.',density:'Both maps use exact cell integration of the P₀ sheet: one follows LPT trajectories and the other follows PM trajectories. CIC particle assignment is shown separately in the spectrum view.',spectra:'The high-k end is where resolution, assignment, and perturbation-order differences become clearest. Compare force-grid and observation-grid changes independently.',pdf:'Mass weighting uses raw unsmoothed cell mass. At R > 0, the density value is smoothed, so the mass and volume PDFs are not related by a simple ρ pV transform.',gradient:'The tangent is the derivative of continuous P₀ Fourier power with respect to initial amplitude A at fixed epoch. It is not a derivative of clipped, deposited density.',published:'The archive panel replays the packaged reference measurements and records their simulation restrictions. It is separate from the live ΛCDM experiment.'};$('p','',theory,text[S.view]||text.sheet);const a=$('a','chapter-link',theory,'Open the chapter ↗');a.href=chapterRefs[S.view]||chapterRefs.sheet;}
async function loadPublished(){if(publishedData){current={published:publishedData};exportJsonButton.disabled=false;status.textContent='Published data ready';renderPublished();return;}status.textContent='Loading published replay…';notice.textContent='Loading packaged reference data.';try{if(!publishedPromise)publishedPromise=fetch('./data/published-results.json').then((res)=>{if(!res.ok)throw Error(`HTTP ${res.status}`);return res.json();});const data=await publishedPromise;if(S.view!=='published')return;publishedData=data;current={published:data};exportJsonButton.disabled=false;status.textContent='Published data ready';renderPublished();}catch(e){if(S.view!=='published')return;publishedPromise=null;current={publishedError:String(e.message||e)};exportJsonButton.disabled=true;status.textContent='Published data unavailable';renderPublished();}}
function renderPublished(){
  explain();clearHost();renderTheory();const data=current?.published;
  if(!data){setNotice(`Published replay is unavailable: ${current?.publishedError||'Loading packaged reference data.'}`,current?.publishedError?'error':'');return;}
  setNotice('Published reference replay loaded. Its archive and single-stream restrictions are listed below.','success');
  const p=panel(data.title||'Published density replay','Reported density PDFs and moments from the packaged simulation archive. This package contains no published power-spectrum curves.');
  const choose=$('div','published-controls',p);
  addPublishedSelect(choose,'Model','pubmodel',(data.models||[]).map((m)=>[m.id,m.label]));
  addPublishedSelect(choose,'Estimator','pubest',(data.estimators||[]).map((e)=>[e.id,e.label]));
  const radii=(data.smoothing_radii_mpc_h||[7,14,28,42]);addPublishedSelect(choose,'Smoothing radius R','pubr',radii.map((r)=>[r,`${r} Mpc/h`]));
  addPublishedSelect(choose,'Snapshot pairing','paired',[[true,'Phase-shifted companion pair'],[false,'Unpaired run']]);
  const model=S.pubmodel,est=S.pubest,R=Number(S.pubr);
  const runRec=(m,e,paired)=>data.runs?.find((r)=>r.model===m&&r.estimator===e&&(paired===undefined||r.paired===paired)&&r.radii?.some((x)=>Number(x.R)===R));
  const selectedRun=runRec(model,est,S.paired),selectedRadius=selectedRun?.radii?.find((x)=>Number(x.R)===R);
  if(!selectedRun||!selectedRadius){$('p','panel-note',p,'No packaged run matches this model, estimator, and smoothing radius.');return;}
  const weight=S.weight,otherModel=model==='nbody'?'4lpt':'nbody',otherRun=runRec(otherModel,est,selectedRun.paired),otherRadius=otherRun?.radii?.find((x)=>Number(x.R)===R);
  const estimatorOther=est==='sheet'?'cic':'sheet',pairedRun=selectedRun.paired?runRec(model,estimatorOther,true):null,pairedRadius=pairedRun?.radii?.find((x)=>Number(x.R)===R);
  const bins=arr(data.bin_edges_log10rho),centers=bins.slice(0,-1).map((x,i)=>(Number(x)+Number(bins[i+1]))/2);
  const curves={selected:selectedRadius[weight]?.pdf||[]};const definitions=[{key:'selected',label:`${model.toUpperCase()} ${est} · ${weight}`}];
  if(otherRadius){curves.comparison=otherRadius[weight]?.pdf||[];definitions.push({key:'comparison',label:`${otherModel.toUpperCase()} ${est} · ${weight}`});}
  if(pairedRadius){curves.paired=pairedRadius[weight]?.pdf||[];definitions.push({key:'paired',label:`${model.toUpperCase()} ${estimatorOther} · ${weight}`});}
  addPlot(p,'Published density PDF',centers,definitions,{xlabel:'log₁₀ ρ',ylabel:'conditional probability density',ylim:[0,undefined]},curves);
  const metrics=$('div','metric-strip',p),qs=selectedRadius[weight]?.quantiles_rho||[];
  const qps=arr(data.quantile_probabilities);for(let i=0;i<qps.length;i++)addMetric(metrics,`ρ quantile ${num(Number(qps[i])*100,3)}%`,num(qs[i],4));
  const captured=selectedRadius[weight]?.captured_probability;addMetric(metrics,'Captured PDF probability',captured==null?'not reported':num(captured,4));
  addMetric(metrics,'Variance',num(selectedRadius.variance,4));addMetric(metrics,'S₃',num(selectedRadius.S3,4));addMetric(metrics,'S₄',num(selectedRadius.S4,4));
  const crossing=selectedRun.crossing||{};addMetric(metrics,'Ever crossed mass fraction',crossing.ever_mass_fraction==null?'not reported':`${num(Number(crossing.ever_mass_fraction)*100,3)}%`);addMetric(metrics,'Final crossed mass fraction',crossing.final_mass_fraction==null?'not reported':`${num(Number(crossing.final_mass_fraction)*100,3)}%`);
  if(crossing.belongs_to)$('p','panel-note',p,`Crossing metadata source: ${crossing.belongs_to}.`);
  const meanings=data.provenance?.field_meanings?.pdf;
  $('p','panel-note',p,typeof meanings==='string'?meanings:'The PDF is conditional on 0.05 ≤ ρ ≤ 20 and normalized within those displayed bins. Quantiles use the full field; the archive does not provide a captured-probability tail fraction.');
  const source=data.provenance||{},prov=$('details','published-provenance',p);$('summary','',prov,'Data provenance and limits');
  if(source.upstream_repo)$('p','',prov,`Upstream: ${source.upstream_repo} (${source.upstream_commit||'commit not recorded'})`);
  if(source.report_url){const a=$('a','',prov,'Open the source report ↗');a.href=source.report_url;a.target='_blank';a.rel='noopener noreferrer';}
  for(const [label,value] of [['Conditions',source.conditions],['Restrictions',source.restrictions]])if(value){const block=$('div','published-detail',prov);$('strong','',block,`${label}: `);$('span','',block,typeof value==='object'?JSON.stringify(value):String(value));}
  if(Array.isArray(source.restrictions)){const visible=$('section','published-restrictions',p);$('h4','',visible,'Scope and limits');const ul=$('ul','',visible);for(const restriction of source.restrictions)$('li','',ul,String(restriction));}
  if(Array.isArray(source.source_paths))$('p','panel-note',prov,`Source files: ${source.source_paths.join(', ')}`);
  if(data.runs?.some((r)=>r.crossing?.ever_mass_fraction>0))$('p','published-caveat',p,'These archives are predominantly single stream, but the recorded N-body runs include a small nonzero shell-crossed mass fraction.');
}
function addPublishedSelect(parent,label,key,options){const wrap=$('label','published-select',parent);$('span','',wrap,label);const sel=$('select','lab-select',wrap);sel.setAttribute('aria-label',label);for(const [value,text] of options){const o=$('option','',sel,text);o.value=String(value);}sel.value=String(S[key]);sel.addEventListener('change',()=>{S[key]=key==='pubr'?Number(sel.value):key==='paired'?sel.value==='true':sel.value;S=normalizeState(S);updateURL();renderPublished();});}
function addMetric(parent,label,value){const item=$('div','metric',parent);$('span','metric-label',item,label);$('strong','metric-value',item,value);}

function setDefaultView(){selectView(S.view, false);}
controls.sync(); setDefaultView(); updateURL(); run();
window.addEventListener('hashchange',()=>{const version=fragmentVersion(location.hash);unsupportedSchema=version!=null&&!(['2',String(STATE_VERSION)].includes(version));if(unsupportedSchema){if(!document.querySelector('.unsupported-link-note'))$('p','unsupported-link-note',intro,`This URL uses unsupported experiment schema v${version}. The experiment has not been run. Select a preset or edit a control to start a new version-${STATE_VERSION} run; the saved values cannot be safely interpreted.`);S=decodeHash('');controls.sync();run();return;}clearUnsupportedSchema();const next=decodeHash(location.hash);if(encodeHash(next)===encodeHash(S))return;S=next;controls.sync();selectView(S.view,false);run();});
window.labExperimentState = () => ({...S});
