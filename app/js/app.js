import { coreRequest, decodeHash, encodeHash, executionPolicy, GRIDS, gridLimit, HASH_SCHEMA, normalizeState, STATE_VERSION, STEPS } from './state.js';
import { PRESETS, presetState } from './presets.js';
import { LinePlot, pchipInterpolate, resampleCurve } from './viz/plot.js';
import { COLORMAPS } from './viz/colormaps.js';
import { copyText, downloadBlob, flash } from './viz/download.js';
import { buildCompactManifest, buildPublishedReplayFromResult, deleteReplayFromLibrary, listReplayLibrary, saveReplayToLibrary } from './replay.js';

function fragmentVersion(hash){const entry=String(hash||'').replace(/^#/,'').split('&').find(p=>p.startsWith('v='));return entry?entry.slice(2):null;}
function fragmentPreset(hash){const entry=String(hash||'').replace(/^#/,'').split('&').find(p=>p.startsWith('preset='));if(!entry)return null;const raw=decodeURIComponent(entry.slice(7));return raw==='estimators2d'?'estimators':raw;}
function decodeHashWithPreset(hash){
  const pid=fragmentPreset(hash);
  if(!pid||!PRESETS.some(p=>p.id===pid))return decodeHash(hash);
  const base=presetState(pid);
  const pairs=String(hash||'').replace(/^#/,'').split('&').filter(p=>p&&!p.startsWith('preset=')&&!p.startsWith('v='));
  if(!pairs.length)return normalizeState(base);
  const overrideHash='#v='+STATE_VERSION+'&'+pairs.join('&');
  const over=decodeHash(overrideHash);
  const explicitKeys=new Set(pairs.map(p=>p.split('=')[0]));
  const merged={...base};
  for(const [stateKey,shortKey] of HASH_SCHEMA){if(explicitKeys.has(shortKey))merged[stateKey]=over[stateKey];}
  return normalizeState(merged);
}
let unsupportedSchema=fragmentVersion(location.hash)!=null&&!(['2',String(STATE_VERSION)].includes(fragmentVersion(location.hash)));
let S = decodeHashWithPreset(location.hash), worker = null, timer = 0, requestId = 0, current = null, computedFor = '', activeKey = '', activeRequestId = null, activePhase = '', pendingRun = null, cancelTimer = 0, publishedPromise = null, publishedData = null, lastResult = null, lastResultKey = '', buildInfoPromise = null, runMilliseconds = null, execution = null;
function requiresExtendedLimits(state = S) {
  return state.n > gridLimit(state.dim, 'n', false) || state.mesh > gridLimit(state.dim, 'mesh', false) || state.forceMesh > gridLimit(state.dim, 'forceMesh', false) || state.steps > 512;
}
let extendedMode = requiresExtendedLimits(S), autoSaveToDisk = false, replayLibrary = [], activeReplaySourceId = 'packaged', overlayPackagedInReplay = true;
void listReplayLibrary().then((items) => { replayLibrary = items; if (S.view === 'published') renderPublished(); });
const densityViewOpts = { cmap: 'coolwarm', scale: 'log', vmin: 0.1, vmax: 10, interp: 'smooth' };
const pdfViewOpts = { ordinate: 'linear', bins: null };
const spectraViewOpts = { companion: 'ratio_sheet', filter: 'tophat', varianceOrdinate: 'sigma2', bins: null };
const rsdViewOpts = { sheetMode: 'rsd_crossing', specMode: 'ratio', showSamples: true };
const TRAJECTORIES = [
  { id: '1lpt', label: '1LPT', color: '#56B4E9' },
  { id: '2lpt', label: '2LPT', color: '#E69F00' },
  { id: '3lpt', label: '3LPT', color: '#009E73' },
  { id: '4lpt', label: '4LPT', color: '#0072B2' },
  { id: 'pm',   label: 'PM',   color: '#D55E00' },
];
const ESTIMATORS = [
  { id: 'p0',  short: 'P₀',  label: 'P₀ sheet',      dash: undefined,       width: 2.1 },
  { id: 'p1',  short: 'P₁',  label: 'P₁ sheet',      dash: '7 3',           width: 1.9 },
  { id: 'ngp', short: 'NGP', label: 'NGP particles', dash: '1.5 3',         width: 1.6 },
  { id: 'cic', short: 'CIC', label: 'CIC particles', dash: '4 2.5',         width: 1.75 },
  { id: 'tsc', short: 'TSC', label: 'TSC particles', dash: '6 2.5 1.5 2.5', width: 1.75 },
];
const SERIES_PALETTE = ['#0072B2', '#D55E00', '#009E73', '#CC79A7', '#E69F00', '#56B4E9', '#882255', '#44AA99', '#332288', '#DDCC77', '#AA4499', '#117733'];
const DEFAULT_ESTIMATOR_SELECTION = ['4lpt:p0', '4lpt:p1', 'pm:p0', '4lpt:cic', '4lpt:tsc'];
let selectedEstimators = (() => {
  try {
    const saved = JSON.parse(sessionStorage.getItem('precrossedsheet:selectedEstimators') || 'null');
    if (Array.isArray(saved) && saved.length > 0 && saved.every(k => typeof k === 'string' && k.includes(':'))) return saved;
  } catch (_) {}
  return [...DEFAULT_ESTIMATOR_SELECTION];
})();
function saveSelectedEstimators() {
  try { sessionStorage.setItem('precrossedsheet:selectedEstimators', JSON.stringify(selectedEstimators)); } catch (_) {}
}
const root = document.querySelector('#lab'), status = document.querySelector('#run-state');
const VIEWS = [
  ['sheet', 'Phase-space sheet'], ['density', 'Density'], ['spectra', 'Power spectra'],
  ['variance', 'Variance vs scale'], ['pdf', 'PDFs'], ['rsd', 'Redshift space (RSD)'], ['gradient', 'Differentiability'], ['published', 'Published replay'],
];
const chapterRefs = { sheet: '../chapters/01-sheet.html#sec-sheet', density: '../chapters/03-meshing.html#sec-meshing', spectra: '../chapters/04-validation.html#sec-validation', variance: '../chapters/05-pdfs.html#sec-pdfs', pdf: '../chapters/05-pdfs.html#sec-pdfs', rsd: '../chapters/05-rsd.html#sec-rsd', gradient: '../chapters/06-inference.html#sec-inference', published: '../chapters/07-lab.html#sec-lab' };
const $ = (tag, cls, parent, value) => { const e = document.createElement(tag); if (cls) e.className = cls; if (value != null) e.textContent = value; parent.append(e); return e; };
const row = (parent, cls = '') => $('div', `lab-control ${cls}`, parent);

document.querySelector('#copy-link').addEventListener('click', async (ev) => { const b = ev.currentTarget; await copyText(location.href); flash(b, 'Copied'); });

const intro = $('section', 'lab-intro', root);
intro.innerHTML = '<p class="eyebrow">Cosmological Lagrangian perturbation theory</p><p>Evolve a discretized Lagrangian sheet with deterministic mode benchmarks or a Gaussian ΛCDM field generated from an Eisenstein–Hu linear spectrum, with or without baryon acoustic oscillations. The model is periodic and uses translationally invariant lower-dimensional geometry for 1D and 2D runs; only 3D runs represent a full spatial realization. Results remain visible after orientation loss, with the single-stream guard status shown as a scientific caution.</p>';
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
const initialPresetId = fragmentPreset(location.hash);
const initialPreset = PRESETS.find((p) => p.id === initialPresetId);
presetSelect.value = initialPreset ? initialPreset.id : '';
presetNote.textContent = initialPreset ? initialPreset.note : 'Adjust the controls or choose a guided experiment.';
presetSelect.addEventListener('change', () => { clearUnsupportedSchema();S = normalizeState(presetState(presetSelect.value)); controls.sync(); selectView(S.view, false); updateURL(); run(); presetNote.textContent = PRESETS.find((p) => p.id === presetSelect.value)?.note || ''; });
const exportJsonButton=$('button','qd-btn',presets,'Save run to disk (.json)');exportJsonButton.type='button';exportJsonButton.disabled=true;exportJsonButton.setAttribute('aria-label','Download the complete experiment state, compact replay result, and build manifest as JSON');
const importLabel=$('label','import-json-button qd-btn',presets,'Load / replay JSON');const importInput=$('input','',importLabel);importInput.type='file';importInput.accept='application/json,.json';importInput.setAttribute('aria-label','Load and replay an experiment JSON manifest from disk');
exportJsonButton.addEventListener('click',()=>exportExperiment(true));
async function importExperimentManifest(manifest, fileName = 'disk manifest', switchToView = null) {
  if (!manifest || manifest.manifestFormat !== 'precrossedsheet-experiment') {
    throw new Error('Expected a precrossedsheet-experiment JSON manifest.');
  }
  const hasSavedResult = Boolean(manifest.result && (manifest.result.spectra || manifest.result.pdf || manifest.result.rho || manifest.result.publishedReplay));
  if (manifest.state) {
    const imported = normalizeState(manifest.state);
    if (!hasSavedResult) {
      if (manifest.schemaVersion !== STATE_VERSION || imported.schemaVersion !== STATE_VERSION) {
        throw new Error(`Expected a precrossedsheet experiment manifest at schema version ${STATE_VERSION}.`);
      }
      for (const [key] of HASH_SCHEMA) {
        if (!(key in manifest.state)) throw new Error(`The manifest is missing reproducibility field “${key}”.`);
        const raw = manifest.state[key], value = imported[key];
        if ((typeof raw === 'number' && !Object.is(raw, value)) || (typeof raw === 'string' && raw !== value) || (typeof raw === 'boolean' && raw !== value)) {
          throw new Error(`The saved value for “${key}” is outside the current experiment schema.`);
        }
      }
    }
    clearUnsupportedSchema();
    S = imported;
    if (requiresExtendedLimits(S) || manifest.request?.extended) extendedMode = true;
    if (Array.isArray(manifest.request?.estimators) && manifest.request.estimators.length) {
      selectedEstimators = manifest.request.estimators;
      saveSelectedEstimators();
      syncEstimatorMatrixUI();
    }
  }
  if (hasSavedResult) {
    const res = manifest.result;
    if (!res.publishedReplay) res.publishedReplay = buildPublishedReplayFromResult(res, S);
    const savedEntry = {
      ...manifest,
      id: manifest.id || `imported-${Date.now()}-${Math.random().toString(36).slice(2, 6)}`,
      label: manifest.label || `${res.dim || S.dim}D · N=${res.n || S.n}^${res.dim || S.dim} · mesh=${res.mesh || S.mesh}^${res.dim || S.dim} (${fileName})`,
      createdAt: manifest.createdAt || new Date().toISOString(),
    };
    await saveReplayToLibrary(savedEntry);
    replayLibrary = await listReplayLibrary();
    activeReplaySourceId = savedEntry.id;
    if (switchToView) S.view = switchToView;
    controls.sync();
    presetSelect.value = '';
    presetNote.textContent = `Loaded saved simulation from ${fileName} (instant replay active).`;
    const key = physicsKey();
    lastResult = res;
    lastResultKey = key;
    computedFor = key;
    current = S.view === 'published' ? { published: res.publishedReplay, savedRun: savedEntry } : res;
    execution = res.execution || manifest.execution || null;
    runMilliseconds = manifest.run?.durationMilliseconds ?? null;
    exportJsonButton.disabled = false;
    selectView(S.view, false);
    updateURL();
    status.textContent = 'Replaying saved run';
    setNotice(`Loaded saved simulation from ${fileName}. Instant replay is active across all Laboratory tabs and in Published replay.`, 'success');
    return;
  }
  clearUnsupportedSchema();
  controls.sync();
  presetSelect.value = '';
  presetNote.textContent = `Imported experiment settings from ${fileName}.`;
  selectView(S.view, false);
  updateURL();
  run();
}
importInput.addEventListener('change',async()=>{const file=importInput.files?.[0];if(!file)return;try{const manifest=JSON.parse(await file.text());await importExperimentManifest(manifest,file.name);}catch(error){setNotice(`Could not import settings: ${error?.message||error}`,'error');}finally{importInput.value='';}});
async function exportExperiment(downloadToDisk = true){
  const sourceResult = S.view === 'published' ? (current?.savedRun?.result || lastResult || (current?.published ? { published: current.published } : null)) : current;
  if(!sourceResult||(!sourceResult?.published&&S.view!=='published'&&computedFor!==physicsKey())){
    setNotice('Wait until the result matches the current experiment settings before saving.');
    return null;
  }
  try{
    const build=await getBuildInfo();
    const manifest=buildCompactManifest({
      state: S,
      request: sourceResult?.request || { ...coreRequest(S), estimators: selectedEstimators, extended: extendedMode || requiresExtendedLimits(S) },
      result: sourceResult,
      execution: sourceResult?.execution || execution || null,
      build,
      runMilliseconds,
    });
    if (!sourceResult.published) {
      await saveReplayToLibrary(manifest);
      replayLibrary = await listReplayLibrary();
      activeReplaySourceId = manifest.id;
    }
    if (downloadToDisk) {
      const blob=new Blob([JSON.stringify(manifest)],{type:'application/json'});
      const tag = sourceResult.published ? 'archive' : `${S.dim}d-N${S.n}-M${S.mesh}-F${S.forceMesh}`;
      downloadBlob(blob,`precrossedsheet-${tag}-${Date.now()}.json`);
      flash(exportJsonButton,'Saved to disk & library');
    }
    return manifest;
  }catch(error){
    setNotice(`Could not save the experiment: ${error?.message||error}`,'error');
    return null;
  }
}
function getBuildInfo(){if(!buildInfoPromise)buildInfoPromise=fetch('./data/build-info.json').then(r=>{if(!r.ok)throw new Error(`build metadata HTTP ${r.status}`);return r.json();}).catch(()=>({engineVersion:'unknown',gitRevision:'unavailable',buildTime:null}));return buildInfoPromise;}

const layout = $('div', 'lab-layout', root), sidebar = $('aside', 'lab-sidebar', layout), main = $('section', 'lab-main', layout);
$('h2', 'section-heading', sidebar, 'Experiment controls');
const controlWrap = $('div', 'control-list', sidebar), controlHandles = {};
function controlSection(title) { const section=$('section','control-section',controlWrap);$('h3','control-subheading',section,title);return section; }
const estimatorMatrixCheckboxes = new Map();
function syncEstimatorMatrixUI() {
  const activeSet = new Set(selectedEstimators);
  for (const [k, cb] of estimatorMatrixCheckboxes.entries()) cb.checked = activeSet.has(k);
}
function applyEstimatorSelection(nextList) {
  const valid = [];
  const seen = new Set();
  for (const t of TRAJECTORIES) {
    for (const e of ESTIMATORS) {
      const k = `${t.id}:${e.id}`;
      if (nextList.includes(k) && !seen.has(k)) {
        seen.add(k);
        valid.push(k);
      }
    }
  }
  if (!valid.length) valid.push(`${S.order || 4}lpt:p0`);
  selectedEstimators = valid;
  saveSelectedEstimators();
  syncEstimatorMatrixUI();
  exportJsonButton.disabled = true;
  if (timer) clearTimeout(timer);
  run();
}
{
  const matrixSection = controlSection('Trajectories & density estimators');
  const wrap = $('div', 'estimator-matrix-wrap', matrixSection);
  $('small', 'control-help', wrap, 'Select any combination of particle trajectories (1–4LPT, PM) and density estimators (P₀, P₁, NGP, CIC, TSC). Choices stay fixed across all views while you adjust experimental controls or switch presets.');
  const table = $('table', 'estimator-matrix-table', wrap);
  const thead = $('thead', '', table), htr = $('tr', '', thead);
  $('th', '', htr, 'Traj \\ Est');
  for (const est of ESTIMATORS) {
    const th = $('th', '', htr);
    const btn = $('button', 'matrix-hdr-btn', th, est.short);
    btn.type = 'button';
    btn.title = `Toggle ${est.label} for active trajectories`;
    btn.addEventListener('click', () => {
      const set = new Set(selectedEstimators);
      const activeTrajs = [...new Set(selectedEstimators.map(k => k.split(':')[0]))];
      const targets = (activeTrajs.length ? activeTrajs : TRAJECTORIES.map(t => t.id)).map(tid => `${tid}:${est.id}`);
      const allOn = targets.every(k => set.has(k));
      for (const k of targets) { if (allOn) set.delete(k); else set.add(k); }
      applyEstimatorSelection([...set]);
    });
  }
  const tbody = $('tbody', '', table);
  for (const traj of TRAJECTORIES) {
    const tr = $('tr', '', tbody);
    const rth = $('th', '', tr);
    const rbtn = $('button', 'matrix-hdr-btn', rth, traj.label);
    rbtn.type = 'button';
    rbtn.title = `Toggle ${traj.label} trajectory`;
    rbtn.addEventListener('click', () => {
      const set = new Set(selectedEstimators);
      const rowKeys = ESTIMATORS.map(e => `${traj.id}:${e.id}`);
      const anyOn = rowKeys.some(k => set.has(k));
      if (anyOn) {
        for (const k of rowKeys) set.delete(k);
      } else {
        const activeEsts = [...new Set(selectedEstimators.map(k => k.split(':')[1]))];
        for (const eid of (activeEsts.length ? activeEsts : ['p0'])) set.add(`${traj.id}:${eid}`);
      }
      applyEstimatorSelection([...set]);
    });
    for (const est of ESTIMATORS) {
      const key = `${traj.id}:${est.id}`;
      const td = $('td', '', tr);
      const lbl = $('label', 'matrix-cell', td);
      lbl.title = `${traj.label} · ${est.label}`;
      const cb = $('input', '', lbl);
      cb.type = 'checkbox';
      cb.setAttribute('aria-label', `${traj.label} ${est.label}`);
      cb.checked = selectedEstimators.includes(key);
      cb.addEventListener('change', () => {
        const set = new Set(selectedEstimators);
        if (cb.checked) set.add(key); else set.delete(key);
        applyEstimatorSelection([...set]);
      });
      estimatorMatrixCheckboxes.set(key, cb);
    }
  }
  const presetRow = $('div', 'estimator-matrix-presets', wrap);
  for (const [label, keys] of [
    ['Default', DEFAULT_ESTIMATOR_SELECTION],
    ['2–4LPT+PM (P₀)', ['2lpt:p0', '3lpt:p0', '4lpt:p0', 'pm:p0']],
    ['2–4LPT+PM (P₀,P₁)', ['2lpt:p0', '2lpt:p1', '3lpt:p0', '3lpt:p1', '4lpt:p0', '4lpt:p1', 'pm:p0', 'pm:p1']],
    ['1–4LPT+PM (P₀)', ['1lpt:p0', '2lpt:p0', '3lpt:p0', '4lpt:p0', 'pm:p0']],
    ['All 4LPT', ['4lpt:p0', '4lpt:p1', '4lpt:ngp', '4lpt:cic', '4lpt:tsc']],
    ['All PM', ['pm:p0', 'pm:p1', 'pm:ngp', 'pm:cic', 'pm:tsc']],
  ]) {
    const b = $('button', 'estimator-preset-btn', presetRow, label);
    b.type = 'button';
    b.addEventListener('click', () => applyEstimatorSelection(keys));
  }
}
let activeControlGroup=controlSection('Geometry and epoch');
function selectControl(key, label, opts, help) {
  const wrap = $('label', 'control-field', activeControlGroup); $('span', 'control-name', wrap, label);
  const sel = $('select', 'lab-select', wrap); sel.setAttribute('aria-label', label); for (const [v, text] of opts) { const o = $('option', '', sel, text); o.value = v; }
  if (help) $('small', 'control-help', wrap, help);
  sel.addEventListener('change', () => {
    S[key] = ['dim','n','order','seed','steps','mesh','forceMesh','initialCutoff'].includes(key) ? Number(sel.value) : sel.value;
    if (key === 'dim') {
      S.n = Math.min(S.n, gridLimit(S.dim, 'n', extendedMode));
      S.mesh = Math.min(S.mesh, gridLimit(S.dim, 'mesh', extendedMode));
      S.forceMesh = Math.min(Math.max(S.forceMesh, S.n), gridLimit(S.dim, 'forceMesh', extendedMode));
    } else if (key === 'n') {
      S.forceMesh = Math.min(Math.max(S.forceMesh, S.n), gridLimit(S.dim, 'forceMesh', extendedMode));
    }
    if (key === 'n' || S.n <= 8 * S.initialCutoff) {
      S.initialCutoff = Math.max(1, Math.min(32, Math.floor((S.n - 1) / 8)));
    }
    S = normalizeState(S); controls.sync(); updateGridOptions(); changed(true);
  });
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
selectControl('n', 'Lagrangian vertices per axis', GRIDS.map(v=>[v,`${v} per axis`]), 'Supported maxima: 1D 1024, 2D 1024, 3D 64. Higher options are disabled for the selected dimension; imported unsupported settings receive an explicit core error, never a silent reduction.');
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
selectControl('ic', 'Initial conditions', [['waves','Deterministic modes'],['gaussian','Gaussian ΛCDM field']], 'Gaussian initial conditions draw mode-keyed random phases from the selected Eisenstein–Hu linear spectrum.');
selectControl('transfer', 'Linear transfer function', [['smooth','EH98 no-wiggle (smooth)'],['bao','EH98 with BAO wiggles']], 'Used only by Gaussian initial conditions. The full Eisenstein–Hu fit adds the baryon acoustic oscillations and Silk damping; the smooth fit keeps only the broad baryon suppression. The power view overlays the filtered linear input spectrum.');
numberControl('seed', 'Gaussian random seed', 0, 9999, 1, 'Integer mode-keyed seed, used only by Gaussian initial conditions.');
selectControl('order', 'LPT perturbative order', [[1,'1st order'],[2,'2nd order'],[3,'3rd order'],[4,'4th order']], 'Order used for the sheet prediction.');
activeControlGroup=controlSection('Evolution and observation grids');
selectControl('steps', 'PM integration steps', STEPS.map(v=>[v,`${v} steps`]), 'Number of √a-spaced canonical KDK time steps for the deconvolved TSC particle-mesh trajectory (≥ 64 recommended).');
selectControl('mesh', 'Observation grid cells per axis', GRIDS.map(v=>[v,`${v} cells`]), 'Eulerian grid resolution used to deposit density, compute PDFs, and evaluate FFT power spectra.');
selectControl('forceMesh', 'PM force grid cells per axis', GRIDS.map(v=>[v,`${v} cells`]), 'Deconvolved TSC Poisson force grid (filtered to the Lagrangian Brillouin zone when forceMesh > n); constrained ≥ vertex grid for high-fidelity PM.');
selectControl('backend', 'Compute backend', [['cpu','CPU reference'],['gpu','GPU experimental'],['auto','Automatic (CPU)']], 'CPU is the default and automatic selection uses the f64 CPU reference. A GPU request can construct nLPT fields on radix lattices, evolve PM trajectories, clip the exact P₀ sheet, and recover weak analytic modes with compensated arithmetic. CPU f64 recovery remains available when a GPU stage fails validation or a mode misses its accuracy gate. One-dimensional runs use CPU.');
{
  const extWrap = $('label', 'control-field', activeControlGroup);
  const extRow = $('span', 'control-top', extWrap);
  $('span', 'control-name', extRow, 'Extended long-running mode (larger 2D/3D grids)');
  const extCb = $('input', '', extRow);
  extCb.type = 'checkbox';
  extCb.checked = extendedMode;
  extCb.setAttribute('aria-label', 'Unlock extended long-running 2D and 3D simulation grids');
  $('small', 'control-help', extWrap, 'Unlocks larger multi-second / multi-minute browser simulations (1D/2D up to 2048², 3D up to 128³ vertices and 256³ meshes, up to 1024 PM steps, 3.5 GiB budget). Completed runs are automatically saved to the browser Replay Library and can be exported to disk (.json).');
  extCb.addEventListener('change', () => {
    extendedMode = extCb.checked;
    if (!extendedMode) {
      S.n = Math.min(S.n, gridLimit(S.dim, 'n', false));
      S.mesh = Math.min(S.mesh, gridLimit(S.dim, 'mesh', false));
      S.forceMesh = Math.min(Math.max(S.forceMesh, S.n), gridLimit(S.dim, 'forceMesh', false));
      S.steps = Math.min(S.steps, 512);
    }
    controls.sync();
    changed(true);
  });
  controlHandles._extendedMode = { set: () => { extCb.checked = extendedMode; }, el: extCb };

  const autoWrap = $('label', 'control-field', activeControlGroup);
  const autoRow = $('span', 'control-top', autoWrap);
  $('span', 'control-name', autoRow, 'Auto-download run to disk (.json) on completion');
  const autoCb = $('input', '', autoRow);
  autoCb.type = 'checkbox';
  autoCb.checked = autoSaveToDisk;
  autoCb.setAttribute('aria-label', 'Automatically download completed run to disk as JSON');
  $('small', 'control-help', autoWrap, 'Useful for unattended longer 2D/3D jobs: as soon as the worker finishes, the browser automatically downloads the compact replay JSON file to disk in addition to saving it in the Published Replay library.');
  autoCb.addEventListener('change', () => {
    autoSaveToDisk = autoCb.checked;
  });
}
activeControlGroup=controlSection('Density observation');
rangeControl('smoothing', 'Observation smoothing R/L', 0, .15, .0025, 'Gaussian measurement filter as a fraction of the box length. This is not initial-condition smoothing.');
selectControl('weight','Primary PDF weighting',[['volume','Volume weighted'],['mass','Mass weighted']],'Both PDFs are plotted; the selected weighting is emphasized, and the URL records the choice.');
const controls = { sync() { if (requiresExtendedLimits(S)) extendedMode = true; for (const [k, h] of Object.entries(controlHandles)) { if (k.startsWith('_')) h.set(); else h.set(S[k]); } updateGridOptions(); updateClockControls(); updateMemoryEstimate(); } };
function estimatePeakBytes(state=S, isExt=extendedMode){const N=state.n**state.dim,M=Math.max(state.mesh,state.forceMesh)**state.dim;if(isExt&&(N>262144||M>262144))return N*(state.dim<3?450:800)+M*(state.dim<3?96:110);return N*(state.dim<3?750:2400)+M*(state.dim<3?200:250);}
function currentBudgetBytes(isExt=extendedMode){return (isExt?3584:1024)*1024*1024;}
function updateGridOptions(){const budget=currentBudgetBytes(extendedMode);for(const key of ['n','mesh','forceMesh']){const select=controlHandles[key]?.el;if(!select)continue;const max=gridLimit(S.dim,key,extendedMode);for(const option of select.options){const val=Number(option.value),candidate={...S,[key]:val},tooLarge=estimatePeakBytes(candidate,extendedMode)>budget,subVertex=key==='forceMesh'&&val<S.n;option.disabled=val>max||tooLarge||subVertex;option.textContent=`${val}${val>max?(extendedMode?' · dimension limit':' · enable Extended mode'):tooLarge?' · allocation limit':subVertex?' · < vertex grid (under-resolved PM)':''}`;}}const stepsSel=controlHandles.steps?.el;if(stepsSel){for(const option of stepsSel.options){const val=Number(option.value);option.disabled=!extendedMode&&val>512;option.textContent=`${val} steps${!extendedMode&&val>512?' · enable Extended mode':''}`;}}}
function updateClockControls(){for(const key of ['fraction','scaleFactor']){const field=controlHandles[key]?.el;const active=key==='fraction'?S.epoch==='fraction':S.epoch==='scaleFactor';if(field){field.disabled=!active;field.closest('.control-field')?.classList.toggle('control-disabled',!active);}}}
const memoryEstimate=$('p','resource-estimate',sidebar);
function updateMemoryEstimate(){const budget=currentBudgetBytes(extendedMode),bytes=estimatePeakBytes(S,extendedMode),fits=bytes<=budget;memoryEstimate.textContent=`Conservative core allocation estimate: ${(bytes/1048576).toFixed(1)} MiB (budget ${(budget/1048576).toFixed(0)} MiB${extendedMode?' · Extended mode active':''}). ${fits?'The current grid combination is within the estimate.':'This selected combination exceeds the active budget; enable Extended mode or reduce a grid.'}`;memoryEstimate.classList.toggle('resource-over-limit',!fits);}
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
  if (lastResultKey === key && lastResult && !lastResult.error) {
    const missing = selectedEstimators.filter(k => !lastResult.customSeries?.[k]);
    const pmNeededNow = missing.some(k => k.startsWith('pm:')) && !lastResult._pmComputed;
    if (missing.length === 0) {
      cancelObsoleteForCachedResult();
      current = lastResult;
      execution = current.execution || null;
      computedFor = key;
      exportJsonButton.disabled = false;
      status.textContent = current?.guard?.ok === false ? 'Post-crossing · interpret cautiously' : 'Ready';
      render();
      return;
    }
    if (!pmNeededNow) {
      const isExt = Boolean(extendedMode || requiresExtendedLimits(S));
      const job = { key, estimatorsOnly: true, allEstimators: [...selectedEstimators], state: { ...S }, request: { ...coreRequest(S), estimators: missing, extended: isExt }, policy: executionPolicy(S) };
      if (activeRequestId != null) { pendingRun = job; cancelActiveRun(); return; }
      dispatchRun(job);
      return;
    }
  }
  const isExt = Boolean(extendedMode || requiresExtendedLimits(S));
  const job={key,estimatorsOnly:false,allEstimators:[...selectedEstimators],state:{...S},request:{...coreRequest(S),estimators:selectedEstimators,extended:isExt},policy:executionPolicy(S)};
  if(activeRequestId!=null){pendingRun=job;cancelActiveRun();return;}
  dispatchRun(job);
}

function ensureWorker(){
  if(worker)return worker;
  worker=new Worker('./js/worker.js?v=20261028d',{type:'module'});
  worker.onmessage=({data})=>{
    if(data.type==='phase'&&data.requestId===activeRequestId){activePhase=data.cancellable===false?`${data.phase||''}-sync`:data.phase||'';status.textContent=({preparing:'Preparing trajectories…',direct:'Evaluating direct modes…', 'direct-sync':'Evaluating CPU direct modes…','direct-recovery-sync':'Recovering direct modes on CPU…','finite-difference':'Evaluating GPU finite differences…','finite-difference-sync':'Checking the CPU finite difference…','lpt-construction':'Constructing nLPT fields on GPU…','lpt-recovery-sync':'Recovering nLPT construction on CPU…','pm-evolution':'Evolving PM trajectories on GPU…','pm-recovery-sync':'Recovering PM trajectories on CPU…','exact-deposit':'Clipping the exact sheet on GPU…','observations-sync':'Depositing the sheet and computing observables…','analytic-recovery':'Recovering weak analytic modes on GPU…',finalizing:'Finalizing observables…'}[activePhase]||'Computing…');notice.textContent=data.message||'Running the ΛCDM Lagrangian calculation in a background worker.';return;}
    if(data.type==='progress'&&data.requestId===activeRequestId){if(data.phase)activePhase=data.phase;const completed=data.completed??data.step??data.caseIndex,total=data.total??data.steps??data.cases;status.textContent=`${data.stage||'Evaluating GPU modes'}${completed!=null?` (${completed}/${total})`:''}…`;return;}
    if(data.type==='cancelled'&&data.requestId===activeRequestId){clearTimeout(cancelTimer);cancelTimer=0;activeRequestId=null;activePhase='';activeKey='';if(pendingRun){const next=pendingRun;pendingRun=null;dispatchRun(next);}return;}
    if(data.type==='shutdown-ack'){hardTerminateWorker();return;}
    if(data.type!=='result'||data.requestId!==activeRequestId)return;
    clearTimeout(cancelTimer);cancelTimer=0;activeRequestId=null;activePhase='';activeKey='';runMilliseconds=data.durationMilliseconds??null;
    if(pendingRun&&pendingRun.key!==data.key){const next=pendingRun;pendingRun=null;dispatchRun(next);return;}
    if(data.error){current={error:data.error,execution:data.execution||null};execution=data.execution||null;computedFor='';status.textContent='Run failed';}
    else if(data.estimatorsOnly && data.customSeries && lastResult && lastResultKey===data.key){
      lastResult.customSeries = { ...(lastResult.customSeries || {}), ...data.customSeries };
      current = lastResult;
      execution = current.execution || null;
      computedFor = data.key;
      status.textContent = current?.guard?.ok===false ? 'Post-crossing · interpret cautiously' : 'Ready';
      exportJsonButton.disabled = false;
    }
    else{
      if (lastResult && lastResultKey === data.key && lastResult.customSeries && data.result) {
        data.result.customSeries = { ...lastResult.customSeries, ...(data.result.customSeries || {}) };
      }
      if (data.result) {
        data.result._pmComputed = Boolean(
          !data.result?.execution?.pmEvolution?.skipped &&
          (selectedEstimators.some(k => k.startsWith('pm:')) || Object.keys(data.result?.customSeries || {}).some(k => k.startsWith('pm:')))
        );
      }
      current=data.result;execution=current?.execution||data.execution||null;lastResult=current;lastResultKey=data.key;computedFor=data.key;status.textContent=current?.guard?.ok===false?'Post-crossing · interpret cautiously':current?.error?'Run failed':'Ready';exportJsonButton.disabled=false;
      if (autoSaveToDisk || extendedMode || (runMilliseconds && runMilliseconds > 1500)) {
        void exportExperiment(autoSaveToDisk);
      }
    }
    render();
    if(pendingRun){const next=pendingRun;pendingRun=null;dispatchRun(next);}
    else if(lastResultKey===physicsKey()&&lastResult&&!lastResult.error&&selectedEstimators.some(k=>!lastResult.customSeries?.[k])){run();}
  };
  worker.onerror=(event)=>{clearTimeout(cancelTimer);cancelTimer=0;worker?.terminate();worker=null;activeRequestId=null;activePhase='';activeKey='';computedFor='';current={error:event.message||'Could not start the numerical worker.'};status.textContent='Run failed';render();};
  return worker;
}
function dispatchRun(job){const w=ensureWorker(),id=++requestId;activeRequestId=id;activePhase='queued';activeKey=job.key;runMilliseconds=null;if(!job.estimatorsOnly)execution=null;exportJsonButton.disabled=true;status.textContent=job.estimatorsOnly?'Updating estimators…':'Preparing…';notice.textContent=job.estimatorsOnly?'Depositing newly selected trajectory/estimator fields from retained particle positions.':'Preparing the ΛCDM trajectories and observations in a background worker.';w.postMessage({type:'run',requestId:id,key:job.key,estimatorsOnly:Boolean(job.estimatorsOnly),allEstimators:job.allEstimators||null,state:job.request,policy:job.policy});}
function cancelActiveRun(){if(!worker||activeRequestId==null)return;const id=activeRequestId;worker.postMessage({type:'cancel',requestId:id});status.textContent='Updating calculation…';clearTimeout(cancelTimer);cancelTimer=setTimeout(()=>{if(activeRequestId!==id)return;hardTerminateWorker();activeRequestId=null;activePhase='';activeKey='';if(pendingRun){const next=pendingRun;pendingRun=null;dispatchRun(next);}},8000);}
function cancelObsoleteForCachedResult(){pendingRun=null;if(activeRequestId!=null&&worker){const id=activeRequestId;activeRequestId=null;activePhase='';activeKey='';clearTimeout(cancelTimer);cancelTimer=0;worker.postMessage({type:'cancel',requestId:id});}}
function retireWorker(){clearTimeout(cancelTimer);cancelTimer=0;const old=worker;worker=null;pendingRun=null;activeRequestId=null;activePhase='';activeKey='';if(!old)return;old.onmessage=({data})=>{if(data?.type==='shutdown-ack')old.terminate();};old.onerror=()=>old.terminate();try{old.postMessage({type:'shutdown'});}catch(_){old.terminate();return;}setTimeout(()=>old.terminate(),5000);}
function hardTerminateWorker(){clearTimeout(cancelTimer);cancelTimer=0;const old=worker;worker=null;if(!old)return;old.onmessage=null;old.onerror=null;old.terminate();}

function explain() {
  const messages = {
    sheet: S.dim === 1
      ? 'Each Lagrangian label q maps to position x and velocity dx/da. The curve remains visible after orientation loss, with the single-stream interpretation explicitly flagged.'
      : S.dim === 2
        ? 'Each Lagrangian label q maps to a position (x₀,x₁). The triangulated plane shows the deformed sheet in position space; the velocity coordinates are not projected into this view.'
        : 'Each Lagrangian label q maps to a position (x₀,x₁,x₂). This wireframe projects the three-dimensional position sheet from the full six-dimensional (x,v) phase-space sheet.',
    density: 'The top pair of maps integrates the connected LPT sheet exactly using P₀ (piecewise-constant) and P₁ (piecewise-linear vertex Jacobian) elements, then applies the selected Gaussian observation filter. The bottom pair deposits the exact same LPT vertices as CIC and TSC point particles on the same mesh, so the difference isolates the density estimator.',
    spectra: `Compare Fourier estimates in physical units: k is h/Mpc and P(k) is (Mpc/h)^${S.dim}. The y-axis minimum is anchored to the measured sheet spectra so the initial Gaussian filter does not compress the dynamic range. Below the spectra, a separate panel analyzes the variance σ²(R) as a function of smoothing scale R across all density fields.`,
    variance: `Compare the variance σ²(R) (or RMS fluctuation σ(R)) of the continuous sheet (P₀ and P₁) and CIC/TSC particle density fields as a function of physical smoothing radius R in Mpc/h for both real-space Top-Hat and Gaussian filters.`,
    pdf: 'Volume weighting samples space uniformly. Mass weighting uses unsmoothed raw cell mass. Observation smoothing is separate from the initial-condition filter; smoothed density values with raw mass weights are not related by a simple ρ pV transform.',
    rsd: 'Map the phase-space sheet into redshift space s = x + (v_∥ / aH) n̂ along the line of sight. Compare the exact redshifted simplicial sheet (which shell-crosses in redshift space wherever ∂_∥ v_∥ < −aH) with a simplex velocity-gradient Gaussian packet construction that fits a d-dimensional Gaussian to each simplex using the exact velocity divergence and velocity-gradient tensor across the simplex.',
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
  if (current.guard?.ok === false) { const why = current.guard.reason || 'A simplex changed orientation.'; setNotice(`Post-crossing diagnostic: ${why} Minimum LPT vertex Jacobian: ${num(current.guard.minimumJacobian,4)}. Outputs are shown with caution; they no longer represent an everywhere one-to-one single-stream map.`, 'guard'); return hasScientificOutput(current); }
  if (current.error) { setNotice(current.error, 'error'); return false; }
  setNotice('Resolved LPT vertex and simplex orientations remained positive throughout the evolution.', 'success'); return true;
}
function render() {
  explain(); clearHost();
  if (S.view === 'published') { renderPublished(); return; }
  if (computedFor && computedFor !== physicsKey()) { setNotice('The controls changed. Waiting for the matching guarded calculation.'); return; }
  if (!guarded()) { showDiagnostics(current); return; }
  if (S.view === 'sheet') renderSheet(current);
  else if (S.view === 'density') renderDensity(current);
  else if (S.view === 'spectra') renderSpectra(current);
  else if (S.view === 'variance') renderVarianceScale(current);
  else if (S.view === 'pdf') renderPdfs(current);
  else if (S.view === 'rsd') renderRsd(current);
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
    const n=Number(d.sheetN||d.n||S.n),dim=Number(d.dim||S.dim),q=arr(d.q),x=arr(d.x),v=arr(d.v); const nv=Math.floor(Math.min(q.length,x.length,v.length)/(dim||1));
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
function resolveActiveSeries(d) {
  const custom = d?.customSeries || {};
  const ordTag = `${d?.request?.order || S.order || 4}lpt`;
  const trajMeta = Object.fromEntries(TRAJECTORIES.map(t => [t.id, t]));
  const estMeta = Object.fromEntries(ESTIMATORS.map(e => [e.id, e]));
  const keys = selectedEstimators.length ? selectedEstimators : DEFAULT_ESTIMATOR_SELECTION;
  const uniqueTrajs = new Set(keys.map(k => k.split(':')[0]));
  const uniqueEsts = new Set(keys.map(k => k.split(':')[1]));
  const multiBoth = uniqueTrajs.size > 1 && uniqueEsts.size > 1;
  const out = [];
  keys.forEach((key, idx) => {
    const [traj, est] = key.split(':');
    const tm = trajMeta[traj] || { id: traj, label: traj.toUpperCase(), color: '#0072B2' };
    const em = estMeta[est] || { id: est, short: est.toUpperCase(), label: est.toUpperCase(), width: 1.8 };
    const label = `${tm.label} · ${em.label}`;
    const shortLabel = `${tm.label} (${em.short})`;
    const color = multiBoth ? tm.color : SERIES_PALETTE[idx % SERIES_PALETTE.length];
    const dash = multiBoth ? em.dash : undefined;
    const width = em.width || 1.8;
    const cs = custom[key];
    const sp = d?.spectra || {}, vs = d?.varianceScale || {}, pdf = d?.pdf || {};
    if (cs) {
      out.push({
        key,
        fieldKey: `cs_${key.replace(':', '_')}`,
        traj,
        est,
        label,
        shortLabel,
        color,
        dash,
        width,
        rho: arr(cs.rho),
        massMean: cs.massMean,
        power: arr(cs.power),
        counts: arr(cs.counts).length ? arr(cs.counts) : arr(sp.counts),
        fineK: arr(cs.fineK).length ? arr(cs.fineK) : arr(sp.fineK),
        finePower: arr(cs.finePower),
        fineCounts: arr(cs.fineCounts).length ? arr(cs.fineCounts) : arr(sp.fineCounts),
        directShell: arr(cs.directShell),
        variance: cs.variance || {},
        pdf: cs.pdf || {},
      });
      return;
    }
    if ((traj === ordTag || !Object.keys(custom).length) && est === 'p0' && traj !== 'pm') {
      out.push({
        key, fieldKey: 'legacy_sheet_p0', traj, est, label, shortLabel, color, dash, width,
        rho: arr(d?.rho), massMean: d?.diagnostics?.mass, power: arr(sp.sheet),
        counts: arr(sp.counts), fineK: arr(sp.fineK), finePower: arr(sp.fineSheet), fineCounts: arr(sp.fineCounts),
        directShell: arr(sp.directShell).length ? arr(sp.directShell) : arr(sp.direct),
        variance: { unsmoothed: vs.unsmoothed?.sheet, tophat: arr(vs.tophat?.sheet), gaussian: arr(vs.gaussian?.sheet) },
        pdf: { logrho: arr(pdf.logrho), dy: pdf.dy, volume: arr(pdf.volume), mass: arr(pdf.mass), variance: pdf.variance, capturedVolume: pdf.capturedVolume, capturedMass: pdf.capturedMass, summary: pdf.summary },
      });
    } else if ((traj === ordTag || !Object.keys(custom).length) && est === 'p1' && traj !== 'pm') {
      out.push({
        key, fieldKey: 'legacy_sheet_p1', traj, est, label, shortLabel, color, dash, width,
        rho: arr(d?.rhoP1), massMean: d?.diagnostics?.p1Mass, power: arr(sp.sheetP1),
        counts: arr(sp.counts), fineK: arr(sp.fineK), finePower: arr(sp.fineSheetP1), fineCounts: arr(sp.fineCounts),
        directShell: [],
        variance: { unsmoothed: vs.unsmoothed?.sheetP1, tophat: arr(vs.tophat?.sheetP1), gaussian: arr(vs.gaussian?.sheetP1) },
        pdf: { logrho: arr(pdf.p1Logrho || pdf.logrho), dy: pdf.p1Dy || pdf.dy, volume: arr(pdf.p1Volume), mass: arr(pdf.p1Mass), variance: pdf.p1Variance, capturedVolume: pdf.p1CapturedVolume, capturedMass: pdf.p1CapturedMass, summary: pdf.p1Summary },
      });
    } else if (traj === 'pm' && est === 'p0') {
      out.push({
        key, fieldKey: 'legacy_pm_p0', traj, est, label, shortLabel, color, dash, width,
        rho: arr(d?.pmRho), massMean: d?.diagnostics?.pmMass, power: arr(sp.pm),
        counts: arr(sp.counts), fineK: arr(sp.fineK), finePower: arr(sp.finePm), fineCounts: arr(sp.fineCounts),
        directShell: arr(sp.directShellPM).length ? arr(sp.directShellPM) : arr(sp.directPM),
        variance: { unsmoothed: vs.unsmoothed?.pm, tophat: arr(vs.tophat?.pm), gaussian: arr(vs.gaussian?.pm) },
        pdf: { logrho: arr(pdf.pmLogrho || pdf.logrho), dy: pdf.pmDy || pdf.dy, volume: arr(pdf.pmVolume), mass: arr(pdf.pmMass), variance: pdf.pmVariance, capturedVolume: pdf.pmCapturedVolume, capturedMass: pdf.pmCapturedMass, summary: pdf.pmSummary },
      });
    } else if ((traj === ordTag || !Object.keys(custom).length) && est === 'cic') {
      out.push({
        key, fieldKey: 'legacy_cic', traj, est, label, shortLabel, color, dash, width,
        rho: arr(d?.cicRho), massMean: d?.diagnostics?.cicMass, power: arr(sp.cic),
        counts: arr(sp.counts), fineK: arr(sp.fineK), finePower: arr(sp.fineCic), fineCounts: arr(sp.fineCounts),
        directShell: [],
        variance: { unsmoothed: vs.unsmoothed?.cic, tophat: arr(vs.tophat?.cic), gaussian: arr(vs.gaussian?.cic) },
        pdf: { logrho: arr(pdf.cicLogrho || pdf.logrho), dy: pdf.cicDy || pdf.dy, volume: arr(pdf.cicVolume), mass: arr(pdf.cicMass), variance: pdf.cicVariance, capturedVolume: pdf.cicCapturedVolume, capturedMass: pdf.cicCapturedMass, summary: pdf.cicSummary },
      });
    } else if ((traj === ordTag || !Object.keys(custom).length) && est === 'tsc') {
      out.push({
        key, fieldKey: 'legacy_tsc', traj, est, label, shortLabel, color, dash, width,
        rho: arr(d?.tscRho), massMean: d?.diagnostics?.tscMass, power: arr(sp.tsc),
        counts: arr(sp.counts), fineK: arr(sp.fineK), finePower: arr(sp.fineTsc), fineCounts: arr(sp.fineCounts),
        directShell: [],
        variance: { unsmoothed: vs.unsmoothed?.tsc, tophat: arr(vs.tophat?.tsc), gaussian: arr(vs.gaussian?.tsc) },
        pdf: { volume: [], mass: [], variance: null, capturedVolume: null, capturedMass: null, summary: null },
      });
    }
  });
  return out;
}
function renderDensity(d) {
  const ratio=Number(d.mesh||S.mesh)/Number(d.n||S.n);
  const activeSeries=resolveActiveSeries(d);
  const p=panel('Continuous sheet vs traditional particle density', `Comparing the selected trajectories (${[...new Set(activeSeries.map(s=>s.traj.toUpperCase()))].join(', ')}) and density estimators (${[...new Set(activeSeries.map(s=>s.est.toUpperCase()))].join(', ')}) on the identical vertices. Gaussian observation filter R/L = ${num(S.smoothing,4)} (${num(S.smoothing*S.boxSize,4)} Mpc/h). All panels share the same spatial scale and ρ/ρ̄ color normalization (defaulting to a logarithmic [0.1, 10] Coolwarm scale where neutral white/grey marks the cosmic mean ρ/ρ̄ = 1). In voids, NGP, CIC, and TSC suffer from Poisson shot noise, sampling starvation, and empty cells (ρ=0), whereas the continuous P₀ and P₁ sheets smoothly expand simplices. For 3D, the view is the central x₀–x₁ slice at x₂/L≈0.5. Observation grid: ${S.mesh} cells per axis (${num(ratio,3)}× vertex spacing).`);
  const dim=Number(d.dim||S.dim), mesh=Number(d.mesh||S.mesh);
  let sum=0,count=0,sheetMax=0,sheetMin=Infinity;
  for(const s of activeSeries)for(let i=0;i<s.rho.length;i++){const value=Number(s.rho[i])||0;sum+=value;count++;if(value>0)sheetMin=Math.min(sheetMin,value);sheetMax=Math.max(sheetMax,value);}
  const mean=count?sum/count:1;
  const autoMin=densityViewOpts.scale==='log'?Math.max(0.05,+(Math.min(sheetMin/(mean||1),0.5)).toPrecision(2)):0;
  const autoMax=Math.max(1.5,+(sheetMax/(mean||1)).toPrecision(3));
  const effMin=Number.isFinite(densityViewOpts.vmin)?densityViewOpts.vmin:autoMin;
  const effMax=Number.isFinite(densityViewOpts.vmax)&&densityViewOpts.vmax>effMin?densityViewOpts.vmax:Math.max(effMin+0.1,autoMax);

  const bar=$('div','published-controls density-display-controls',p);
  const addSel=(label,opts,val,onChange)=>{const wrap=$('label','published-select',bar);$('span','',wrap,label);const s=$('select','lab-select',wrap);for(const [v,t] of opts){const o=$('option','',s,t);o.value=v;}s.value=val;s.addEventListener('change',()=>onChange(s.value));return s;};
  addSel('Colormap',[['coolwarm','Coolwarm (ρ/ρ̄ = 1 neutral)'],['viridis','Viridis'],['ylgnbu','Blue–Grey–Yellow (paper)'],['magma','Magma'],['inferno','Inferno'],['rdbu','RdBu'],['blueorange','Blue–Orange'],['gray','Grayscale']],densityViewOpts.cmap,(v)=>{densityViewOpts.cmap=v;render();});
  addSel('Color scale',[['log','Logarithmic (log₁₀ ρ / ρ̄)'],['linear','Linear (ρ / ρ̄)']],densityViewOpts.scale,(v)=>{densityViewOpts.scale=v;if(v==='log'&&!(densityViewOpts.vmin>0))densityViewOpts.vmin=0.1;render();});
  addSel('Rendering',[['smooth','Smooth (bilinear)'],['pixel','Crisp mesh cells']],densityViewOpts.interp,(v)=>{densityViewOpts.interp=v;render();});
  const addNum=(label,val,step,onChange)=>{const wrap=$('label','published-select',bar);$('span','',wrap,label);const inp=$('input','lab-number',wrap);inp.type='number';inp.step=step;inp.style.width='6.5rem';inp.value=String(val);inp.addEventListener('change',()=>onChange(parseFloat(inp.value)));return inp;};
  addNum('Min ρ/ρ̄',effMin,0.1,(v)=>{densityViewOpts.vmin=Number.isFinite(v)?Math.max(densityViewOpts.scale==='log'?1e-4:0,v):null;render();});
  addNum('Max ρ/ρ̄',effMax,0.5,(v)=>{densityViewOpts.vmax=Number.isFinite(v)&&v>0?v:null;render();});
  const stdBtn=$('button','qd-btn',bar,'0.1–10 Log Coolwarm');stdBtn.type='button';stdBtn.style.alignSelf='end';
  stdBtn.addEventListener('click',()=>{densityViewOpts.cmap='coolwarm';densityViewOpts.scale='log';densityViewOpts.vmin=0.1;densityViewOpts.vmax=10;render();});
  const resetBtn=$('button','qd-btn',bar,'Auto (field range)');resetBtn.type='button';resetBtn.style.alignSelf='end';
  resetBtn.addEventListener('click',()=>{densityViewOpts.vmin=null;densityViewOpts.vmax=null;render();});

  const maps=$('div','density-pair',p);
  for(const s of activeSeries){
    if(s.rho.length){
      drawDensityMap(maps,s.label,s.rho,`Observed density for ${s.label}`,dim,mesh,`density-${s.key.replace(':','-')}`,effMin,effMax,mean);
    }
  }
  const withRho=activeSeries.filter(s=>s.rho.length>0);
  if(withRho.length){
    const stats=(s)=>{
      const pre=d.densitySliceStats?.[s.fieldKey];
      if(pre&&pre.totalCells)return {empty:pre.empty,total:pre.totalCells,minimum:pre.minimum,maximum:pre.maximum};
      const values=s.rho;let empty=0,minimum=Infinity,maximum=0;
      for(let i=0;i<values.length;i++){const v=Number(values[i])||0;if(v<=1e-6)empty++;minimum=Math.min(minimum,v);maximum=Math.max(maximum,v);}
      return {empty,total:values.length,minimum,maximum};
    };
    const table=$('table','summary-table',p);$('caption','',table,'Cell statistics of the displayed (filtered) fields: a connected sheet never leaves an interior cell empty');
    const head=$('thead','',table),hr=$('tr','',head);for(const label of ['Trajectory · Estimator','Empty cells','Minimum ρ/mean','Maximum ρ/mean'])$('th','',hr,label);
    const body=$('tbody','',table);
    for(const s of withRho){const st=stats(s),tr=$('tr','',body);$('th','',tr,s.label);$('td','',tr,`${st.empty} of ${st.total}`);$('td','',tr,num(st.minimum/(mean||1),4));$('td','',tr,num(st.maximum/(mean||1),4));}
  }
}
function drawDensityMap(parent,title,values,label,dim,mesh,name,scaleMin,scaleMax,mean){
  const block=$('div','density-map',parent);$('h4','',block,title);
  const c=$('canvas',`lab-canvas ${dim===1?'canvas-wide':'canvas-square'}`,block);
  c.width=dim===1?960:680;c.height=dim===1?560:720;c.setAttribute('role','img');c.setAttribute('aria-label',label);
  const rho=arr(values);
  const isLog=densityViewOpts.scale==='log'&&scaleMin>0&&scaleMax>scaleMin;
  const normVal=(v)=>{
    const r=Math.max(0,Number(v)||0)/(mean||1);
    if(isLog){if(r<=0)return 0;return Math.max(0,Math.min(1,(Math.log10(r)-Math.log10(scaleMin))/(Math.log10(scaleMax)-Math.log10(scaleMin))));}
    return Math.max(0,Math.min(1,(r-scaleMin)/Math.max(1e-12,scaleMax-scaleMin)));
  };
  paintCanvas(c,(ctx,W,H)=>{
    const ink=getComputedStyle(c).color;ctx.fillStyle=ink;
    if(!rho.length){ctx.fillText('Density cells were not returned.',32,42);return;}
    const isSlice2d=(dim===3&&rho.length<mesh**3)||(dim===2&&rho.length<mesh*mesh);
    const n=isSlice2d?(Math.round(Math.sqrt(rho.length))||mesh):mesh,left=56,right=W-24,top=16,bottom=H-92;
    if(dim===1){for(let i=0;i<rho.length;i++){const t=normVal(rho[i]),height=t*(bottom-top);ctx.fillStyle=densityColor(t);const xStart=Math.round(left+i*(right-left)/rho.length),xEnd=Math.round(left+(i+1)*(right-left)/rho.length);ctx.fillRect(xStart,bottom-height,Math.max(1,xEnd-xStart),height);}}
    else {const side=Math.round(Math.min(right-left,bottom-top)),x0=Math.round(left+(right-left-side)/2),y0=Math.round(top+(bottom-top-side)/2),kMid=Math.floor(n/2);
      const off=document.createElement('canvas');off.width=n;off.height=n;
      const octx=off.getContext('2d'),img=octx.createImageData(n,n),buf=img.data;
      for(let i=0;i<n;i++)for(let j=0;j<n;j++){
        const index=(dim===3&&!isSlice2d)?(i*n+j)*n+kMid:i*n+j;
        const [r,g,b]=densityColorRgb(normVal(Number(rho[index]??0)));
        const p=4*((n-1-j)*n+i);
        buf[p]=r;buf[p+1]=g;buf[p+2]=b;buf[p+3]=255;
      }
      octx.putImageData(img,0,0);
      ctx.save();
      ctx.imageSmoothingEnabled=densityViewOpts.interp!=='pixel';
      if(ctx.imageSmoothingEnabled)ctx.imageSmoothingQuality='high';
      ctx.drawImage(off,x0,y0,side,side);
      ctx.restore();
      ctx.strokeStyle=ink;ctx.lineWidth=1;ctx.strokeRect(x0,y0,side,side);
      ctx.font='12px system-ui';
      for(const [v,x] of [['0',x0],['0.5',x0+side/2],['1',x0+side]]){ctx.fillStyle=ink;ctx.fillText(v,x-4,y0+side+16);ctx.beginPath();ctx.moveTo(x,y0+side);ctx.lineTo(x,y0+side+4);ctx.stroke();}
      for(const [v,y] of [['0',y0+side],['0.5',y0+side/2],['1',y0]])ctx.fillText(v,x0-26,y+4);
      ctx.fillStyle=ink;ctx.textAlign='center';ctx.fillText(`x₀/L  (L = ${num(S.boxSize,5)} Mpc/h)`,x0+side/2,y0+side+33);ctx.save();ctx.translate(x0-38,y0+side/2);ctx.rotate(-Math.PI/2);ctx.fillText('x₁/L',0,0);ctx.restore();ctx.textAlign='start';}
    ctx.fillStyle=ink;ctx.font='12px system-ui';
    const gw=Math.min(280,Math.max(140,(right-left)*0.52)),gx=left+(right-left-gw)/2,gy=H-26,gh=10,grad=ctx.createLinearGradient(gx,0,gx+gw,0);
    for(let i=0;i<=32;i++)grad.addColorStop(i/32,densityColor(i/32));
    ctx.fillStyle=grad;ctx.fillRect(gx,gy,gw,gh);ctx.strokeStyle=ink;ctx.lineWidth=0.75;ctx.strokeRect(gx,gy,gw,gh);
    if(scaleMin<1&&scaleMax>1){
      const tOne=normVal(mean||1),xOne=Math.round(gx+tOne*gw)+0.5;
      ctx.beginPath();ctx.moveTo(xOne,gy-2);ctx.lineTo(xOne,gy+gh+3);ctx.strokeStyle=ink;ctx.lineWidth=1.2;ctx.stroke();
      ctx.fillStyle=ink;ctx.textAlign='center';ctx.fillText('1',xOne,gy+gh+13);
    }
    ctx.fillStyle=ink;ctx.textAlign='right';ctx.fillText(num(scaleMin,3),gx-6,gy+9);
    ctx.textAlign='left';ctx.fillText(num(scaleMax,3),gx+gw+6,gy+9);
    ctx.textAlign='center';ctx.fillText(isLog?'ρ / ρ̄ (log scale)':'ρ / ρ̄',gx+gw/2,gy-5);ctx.textAlign='start';
  });addPngButton(c,name);
}
function densityColorRgb(t){
  const x=Math.max(0,Math.min(1,t));
  const cmap=densityViewOpts.cmap||'coolwarm';
  if(cmap==='blueorange'){
    const h=((218-205*x)%360+360)%360,s=(50+40*x)/100,l=(97-53*x)/100;
    const c=(1-Math.abs(2*l-1))*s,hp=h/60,x2=c*(1-Math.abs(hp%2-1)),m=l-c/2;
    let [r,g,b]=hp<1?[c,x2,0]:hp<2?[x2,c,0]:hp<3?[0,c,x2]:hp<4?[0,x2,c]:hp<5?[x2,0,c]:[c,0,x2];
    return [Math.round((r+m)*255),Math.round((g+m)*255),Math.round((b+m)*255)];
  }
  if(cmap==='ylgnbu'){
    const stops=[[0,[0,32,77]],[0.25,[49,73,107]],[0.5,[124,123,120]],[0.75,[190,175,111]],[1,[255,234,70]]];
    for(let i=0;i<stops.length-1;i++){
      const [t0,c0]=stops[i],[t1,c1]=stops[i+1];
      if(x<=t1||i===stops.length-2){const f=(x-t0)/Math.max(1e-9,t1-t0);return [Math.round(c0[0]+(c1[0]-c0[0])*f),Math.round(c0[1]+(c1[1]-c0[1])*f),Math.round(c0[2]+(c1[2]-c0[2])*f)];}
    }
  }
  const fn=COLORMAPS[cmap]||COLORMAPS.viridis;
  const [r,g,b]=fn(x);
  return [Math.round(r),Math.round(g),Math.round(b)];
}
function densityColor(t){
  const [r,g,b]=densityColorRgb(t);
  return `rgb(${r},${g},${b})`;
}
function plotHost(parent){return $('div','plot-host',parent);}
function addPlot(parent,title,x,seriesDefs,axes,d) {
  const box=$('section','plot-panel',parent);$('h4','',box,title); const host=plotHost(box), plot=new LinePlot(host,{width:720,height:390,interactive:true,binLabel:axes.binLabel,onBinsChange:axes.onBinsChange});
  const xs=arr(x); if(!xs.length){$('p','panel-note',box,'This run did not return data for this chart.');return null;}
  const series=[]; for(const def of seriesDefs){const ys=arr(d?.[def.key] ?? d?.spectra?.[def.key] ?? d?.pdf?.[def.key] ?? d?.gradient?.[def.key]);const sx=arr(def.xkey?(d?.[def.xkey]??d?.spectra?.[def.xkey]??d?.gradient?.[def.xkey]):xs);if(ys.length&&sx.length)series.push({...def,x:sx.slice(0,Math.min(sx.length,ys.length)),y:ys.slice(0,Math.min(sx.length,ys.length)),label:def.label});}
  if(!series.length){$('p','panel-note',box,'The selected output did not include the requested curves.');return null;}
  plot.setAxes({...axes,title,xlog:axes.xlog??false,ylog:axes.ylog??false});plot.setSeries(series);plot.draw();
  return plot;
}
function sheetPositiveRange(arrays) {
  let lo = Infinity, hi = 0;
  for (const a of arrays) {
    for (const raw of arr(a)) {
      const v = Number(raw);
      if (Number.isFinite(v) && v > 0) {
        lo = Math.min(lo, v);
        hi = Math.max(hi, v);
      }
    }
  }
  return Number.isFinite(lo) && hi > 0 ? [lo, hi] : null;
}
function rebinPowerSeries(kArr, pArr, countsArr, fineKArr, finePArr, fineCountsArr, targetBins) {
  const defK = arr(kArr).map(Number);
  const defP = arr(pArr).map(Number);
  const fK = arr(fineKArr).length ? arr(fineKArr).map(Number) : defK;
  const fP = arr(finePArr).length ? arr(finePArr).map(Number) : defP;
  const fC = arr(fineCountsArr).length
    ? arr(fineCountsArr).map(Number)
    : arr(countsArr).length
      ? arr(countsArr).map(Number)
      : null;
  if (!Number.isFinite(targetBins) || targetBins <= 0 || targetBins === defK.length) {
    return { x: defK, y: defP };
  }
  return resampleCurve(fK, fP, targetBins, { xlog: true, ylog: true, weights: fC });
}

function renderSpectra(d) {
  const p=panel('Power spectrum comparison','The primary chart compares isotropically binned FFT power spectra P(k) in physical units for all selected Trajectory × Estimator pairs, with the y-axis minimum anchored to the measured spectra rather than the Gaussian-damped linear input tail. Use the modes (− / N / +) control in the panel bar or plot toolbar to coarsen k-shells (reducing sample-variance noise) or refine up to 256 fine shells (resolving BAO wiggles and high-k roll-off).');
  const sp=d.spectra||{};
  const ks=arr(sp.k);
  const fineK=arr(sp.fineK);
  const fineCounts=arr(sp.fineCounts);
  const defaultKBins=ks.length||24;
  const activeKBins=spectraViewOpts.bins||defaultKBins;
  const activeSeries=resolveActiveSeries(d).filter(s=>s.power.length>0);

  const topBar=$('div','published-controls',p);
  const compWrap=$('label','published-select',topBar);$('span','',compWrap,'Companion spectral diagnostic');
  const compSel=$('select','lab-select',compWrap);
  const refSeries=activeSeries.find(s=>s.key==='4lpt:p0')||activeSeries.find(s=>s.est==='p0')||activeSeries[0];
  const refPower=refSeries?refSeries.power:arr(sp.sheet);
  const refFinePower=refSeries&&arr(refSeries.finePower).length?refSeries.finePower:arr(sp.fineSheet);
  const refName=refSeries?refSeries.shortLabel:'LPT (P₀)';
  const companionModes=[
    ['ratio_sheet',`Relative estimator ratio  P(k) / P_${refName}(k)`],
    ['transfer_realized','Nonlinear mode transfer  P(k) / P_lin,realized(k)'],
    ['dimensionless','Dimensionless power  Δ²(k) = C_d k^d P(k)'],
    ['direct_simplex','Mesh-free exact simplex Fourier spectrum vs isotropic FFT'],
  ];
  for(const [v,t] of companionModes){const o=$('option','',compSel,t);o.value=v;}
  compSel.value=spectraViewOpts.companion;
  compSel.addEventListener('change',()=>{spectraViewOpts.companion=compSel.value;render();});

  const kBinWrap=$('label','published-select',topBar);$('span','',kBinWrap,'k-shells / modes');
  const kBinRow=$('span','qd-pt-axis',kBinWrap);
  const kMinus=$('button','qd-btn',kBinRow,'−');kMinus.type='button';kMinus.style.padding='0.1rem 0.45rem';kMinus.title='Coarsen k-shells (fewer bins, lower noise)';
  const kInput=$('input','lab-number',kBinRow);kInput.type='number';kInput.min='6';kInput.max='320';kInput.step='4';kInput.style.width='4.5rem';kInput.value=String(activeKBins);
  const kPlus=$('button','qd-btn',kBinRow,'+');kPlus.type='button';kPlus.style.padding='0.1rem 0.45rem';kPlus.title='Refine k-shells (more bins, up to 256 native shells)';
  const kAuto=$('button','qd-btn',kBinRow,`Auto (${defaultKBins})`);kAuto.type='button';kAuto.style.padding='0.1rem 0.5rem';

  const specPlots=[];
  const syncSpecBins=(nb,sourcePlot=null)=>{
    spectraViewOpts.bins=nb;
    kInput.value=String(nb||defaultKBins);
    for(const pl of specPlots){
      if(pl&&pl!==sourcePlot)pl.setBins(nb,false);
    }
  };
  kMinus.addEventListener('click',()=>{if(specPlots[0])specPlots[0]._stepBins(-1);});
  kPlus.addEventListener('click',()=>{if(specPlots[0])specPlots[0]._stepBins(1);});
  kInput.addEventListener('change',()=>{
    const v=parseInt(kInput.value,10);
    if(Number.isFinite(v)&&v>=4){
      if(specPlots[0])specPlots[0].setBins(v,true);
      else{spectraViewOpts.bins=v;render();}
    }
  });
  kAuto.addEventListener('click',()=>{
    spectraViewOpts.bins=null;
    kInput.value=String(defaultKBins);
    for(const pl of specPlots)if(pl)pl.setBins(null,false);
  });

  const pkData={
    k:ks,
    realizedK:arr(sp.realizedK),
    realized:arr(sp.realized),
    linearK:arr(sp.linearK),
    linear:arr(sp.linear),
  };
  for(const s of activeSeries)pkData[s.fieldKey]=s.power;
  const sheetBounds=sheetPositiveRange(activeSeries.map(s=>s.power));
  const allBounds=sheetPositiveRange([...activeSeries.map(s=>s.power),sp.realized,sp.linear]);
  const pkYlim=sheetBounds&&allBounds?[sheetBounds[0]*0.45,Math.max(sheetBounds[1],allBounds[1])*2.2]:undefined;
  const pkDefs=[
    ...activeSeries.map(s=>({
      key:s.fieldKey,
      label:s.label,
      color:s.color,
      dash:s.dash,
      width:s.width,
      resample:(nb)=>rebinPowerSeries(ks,s.power,s.counts,s.fineK,s.finePower,s.fineCounts,nb),
    })),
    {
      key:'realized',
      xkey:'realizedK',
      label:'Linear input, this realization',
      color:'#7f8c8d',
      dash:'2 3',
      width:1.4,
      resample:(nb)=>rebinPowerSeries(sp.realizedK,sp.realized,sp.realizedCounts,sp.fineRealizedK,sp.fineRealized,sp.fineRealizedCounts,nb),
    },
    {
      key:'linear',
      xkey:'linearK',
      label:'Linear input, ensemble mean',
      color:'#34495e',
      dash:'5 3',
      width:1.5,
      fineX:arr(sp.fineLinearK).length?arr(sp.fineLinearK):arr(sp.linearK),
      fineY:arr(sp.fineLinear).length?arr(sp.fineLinear):arr(sp.linear),
    },
  ];
  const pMain=addPlot(p,'Isotropic FFT-bin power P(k)',ks,pkDefs,{xlabel:'k [h/Mpc]',ylabel:`P(k) [(Mpc/h)^${S.dim}]`,xlog:true,ylog:true,ylim:pkYlim,binLabel:'modes',bins:spectraViewOpts.bins?? undefined,minBins:6,maxBins:320,onBinsChange:(nb)=>syncSpecBins(nb,pMain)},pkData);
  if(pMain)specPlots.push(pMain);

  const pairRatioResample=(s,denomK,denomP,denomC,denomFineK,denomFineP,denomFineC)=>(nb)=>{
    const numCurve=rebinPowerSeries(ks,s.power,s.counts,s.fineK,s.finePower,s.fineCounts,nb);
    const denCurve=rebinPowerSeries(denomK,denomP,denomC,denomFineK,denomFineP,denomFineC,nb);
    if(!numCurve.x.length||!denCurve.x.length)return {x:[],y:[]};
    if(numCurve.x.length===denCurve.x.length){
      return {
        x:numCurve.x,
        y:numCurve.y.map((v,i)=>denCurve.y[i]>1e-30?v/denCurve.y[i]:NaN),
      };
    }
    const denInterp=pchipInterpolate(
      denCurve.x.map(v=>Math.log10(Math.max(1e-30,v))),
      denCurve.y,
      numCurve.x.map(v=>Math.log10(Math.max(1e-30,v)))
    );
    return {
      x:numCurve.x,
      y:numCurve.y.map((v,i)=>denInterp[i]>1e-30?v/denInterp[i]:NaN),
    };
  };

  if(spectraViewOpts.companion==='ratio_sheet'){
    const ratioToSheet=(vals)=>arr(vals).map((v,i)=>{const denom=Number(refPower[i]);return denom>1e-30?Number(v)/denom:NaN;});
    const ratioData={k:ks};
    for(const s of activeSeries)ratioData[s.fieldKey]=ratioToSheet(s.power);
    const ratioDefs=activeSeries.map(s=>({
      key:s.fieldKey,
      label:s===refSeries?`${s.label} ≡ 1`:`${s.label} / ${refName}`,
      color:s.color,
      dash:s.dash,
      width:s.width,
      resample:s===refSeries
        ?(nb)=>{const c=rebinPowerSeries(ks,s.power,s.counts,s.fineK,s.finePower,s.fineCounts,nb);return {x:c.x,y:c.x.map(()=>1)};}
        :pairRatioResample(s,ks,refPower,sp.counts,fineK,refFinePower,fineCounts),
    }));
    const pComp=addPlot(p,`Spectral ratio relative to ${refName}: P(k) / P_${refName}(k)`,ks,ratioDefs,{xlabel:'k [h/Mpc]',ylabel:`P(k) / P_${refName}(k)`,xlog:true,ylog:false,binLabel:'modes',bins:spectraViewOpts.bins?? undefined,minBins:6,maxBins:320,onBinsChange:(nb)=>syncSpecBins(nb,pComp)},ratioData);
    if(pComp)specPlots.push(pComp);
    $('p','panel-note',p,`Dividing by ${refName} isolates how the selected trajectories (1–4LPT, PM) and density estimators (P₀, P₁, NGP, CIC, TSC) compare across scale.`);
  } else if(spectraViewOpts.companion==='transfer_realized'){
    const rK=arr(sp.realizedK),rP=arr(sp.realized);
    const baseK=rK.length?rK:ks;
    const baseP=rP.length?rP:refPower;
    const ratioToRealized=(vals)=>baseK.map((_,i)=>{const denom=Number(baseP[i]);return denom>1e-30?Number(arr(vals)[i])/denom:NaN;});
    const trData={k:baseK,unit:baseK.map(()=>1)};
    for(const s of activeSeries)trData[s.fieldKey]=ratioToRealized(s.power);
    const trDefs=[
      ...activeSeries.map(s=>({
        key:s.fieldKey,
        label:s.label,
        color:s.color,
        dash:s.dash,
        width:s.width,
        resample:rP.length
          ?pairRatioResample(s,sp.realizedK,sp.realized,sp.realizedCounts,sp.fineRealizedK,sp.fineRealized,sp.fineRealizedCounts)
          :pairRatioResample(s,ks,refPower,sp.counts,fineK,refFinePower,fineCounts),
      })),
      {key:'unit',label:rP.length?'Linear realization ≡ 1':`${refName} ≡ 1`,color:'#7f8c8d',dash:'2 3',width:1.3,noResample:true},
    ];
    const pComp=addPlot(p,rP.length?'Nonlinear mode transfer relative to realized linear modes: P(k) / P_lin,realized(k)':`Ratio relative to ${refName} (deterministic wave run)`,baseK,trDefs,{xlabel:'k [h/Mpc]',ylabel:rP.length?'P(k) / [D(a)² |δ₀(k)|²]':'P(k) / P_ref(k)',xlog:true,ylog:true,binLabel:'modes',bins:spectraViewOpts.bins?? undefined,minBins:6,maxBins:320,onBinsChange:(nb)=>syncSpecBins(nb,pComp)},trData);
    if(pComp)specPlots.push(pComp);
    $('p','panel-note',p,'Because “Linear input, this realization” uses the exact random phases δ₀(k) of the run, dividing by D(a)²|δ₀(k)|² cancels cosmic variance on large scales and exposes nonlinear mode coupling into high-k modes above the initial smoothing scale.');
  } else if(spectraViewOpts.companion==='dimensionless'){
    const dim=Number(d.dim||S.dim);
    const pref=dim===1?1/Math.PI:dim===2?1/(2*Math.PI):1/(2*Math.PI*Math.PI);
    const toDelta2=(vals,kArr=ks)=>arr(vals).map((v,i)=>pref*Math.pow(Number(kArr[i]||0),dim)*Number(v));
    const d2Data={
      k:ks,
      realizedK:arr(sp.realizedK),
      realized:toDelta2(sp.realized,arr(sp.realizedK)),
      linearK:arr(sp.linearK),
      linear:toDelta2(sp.linear,arr(sp.linearK)),
    };
    const d2SeriesArrays=[];
    for(const s of activeSeries){
      const arrD2=toDelta2(s.power);
      d2Data[s.fieldKey]=arrD2;
      d2SeriesArrays.push(arrD2);
    }
    const d2Defs=[
      ...activeSeries.map(s=>({
        key:s.fieldKey,
        label:s.label,
        color:s.color,
        dash:s.dash,
        width:s.width,
        resample:(nb)=>{
          const c=rebinPowerSeries(ks,s.power,s.counts,s.fineK,s.finePower,s.fineCounts,nb);
          return {x:c.x,y:toDelta2(c.y,c.x)};
        },
      })),
      {
        key:'realized',
        xkey:'realizedK',
        label:'Linear input, this realization',
        color:'#7f8c8d',
        dash:'2 3',
        width:1.4,
        resample:(nb)=>{
          const c=rebinPowerSeries(sp.realizedK,sp.realized,sp.realizedCounts,sp.fineRealizedK,sp.fineRealized,sp.fineRealizedCounts,nb);
          return {x:c.x,y:toDelta2(c.y,c.x)};
        },
      },
      {
        key:'linear',
        xkey:'linearK',
        label:'Linear input, ensemble mean',
        color:'#34495e',
        dash:'5 3',
        width:1.5,
        fineX:arr(sp.fineLinearK).length?arr(sp.fineLinearK):arr(sp.linearK),
        fineY:toDelta2(arr(sp.fineLinear).length?arr(sp.fineLinear):arr(sp.linear),arr(sp.fineLinearK).length?arr(sp.fineLinearK):arr(sp.linearK)),
      },
    ];
    const d2SheetBounds=sheetPositiveRange(d2SeriesArrays);
    const d2AllBounds=sheetPositiveRange([...d2SeriesArrays,d2Data.realized,d2Data.linear]);
    const d2Ylim=d2SheetBounds&&d2AllBounds?[d2SheetBounds[0]*0.45,Math.max(d2SheetBounds[1],d2AllBounds[1])*2.2]:undefined;
    const pComp=addPlot(p,`Dimensionless power spectrum Δ²(k) in ${dim}D`,ks,d2Defs,{xlabel:'k [h/Mpc]',ylabel:'Δ²(k)  [variance per ln k]',xlog:true,ylog:true,ylim:d2Ylim,binLabel:'modes',bins:spectraViewOpts.bins?? undefined,minBins:6,maxBins:320,onBinsChange:(nb)=>syncSpecBins(nb,pComp)},d2Data);
    if(pComp)specPlots.push(pComp);
    $('p','panel-note',p,`Δ²(k) measures the contribution to the real-space variance per logarithmic wavenumber interval d ln k. Its integral over ln k weighted by W²(kR) yields the smoothed variance σ²(R) shown in the panel below.`);
  } else {
    const useShell=arr(sp.directShellK).length>0;
    const dk=useShell?arr(sp.directShellK):arr(sp.directK);
    const realKey=useShell?'directShellRealized':'directRealized';
    const realKKey=useShell?'directShellRealizedK':'directRealizedK';
    const linKey=useShell?'directShellLinear':'directLinear';
    const linKKey=useShell?'directShellLinearK':'directLinearK';
    const dirData={
      dk,
      k:ks,
      [realKKey]:arr(sp[realKKey]),
      [realKey]:arr(sp[realKey]),
      [linKKey]:arr(sp[linKKey]),
      [linKey]:arr(sp[linKey]),
    };
    const dirDefs=[];
    const dirArrays=[];
    const meshFreeSeries=activeSeries.filter(s=>s.directShell&&s.directShell.length>0);
    if(meshFreeSeries.length){
      for(const s of meshFreeSeries){
        const mfKey=`${s.fieldKey}_mf`;
        dirData[mfKey]=s.directShell;
        dirArrays.push(s.directShell);
        dirDefs.push({key:mfKey,label:`Mesh-free simplex (${s.shortLabel})`,color:s.color,width:2.1,points:true,radius:2.1,zOrder:10});
      }
    } else {
      const lptKey=useShell?'directShell':'direct',pmKey=useShell?'directShellPM':'directPM';
      dirData[lptKey]=arr(sp[lptKey]);dirData[pmKey]=arr(sp[pmKey]);
      dirArrays.push(dirData[lptKey],dirData[pmKey]);
      dirDefs.push({key:lptKey,label:'Mesh-free simplex Fourier (LPT P₀)',width:2.2,points:true,radius:2.2,zOrder:10},{key:pmKey,label:'Mesh-free simplex Fourier (PM P₀)',width:2.0,points:true,radius:2.0,zOrder:9});
    }
    for(const s of activeSeries){
      dirData[s.fieldKey]=s.power;
      dirArrays.push(s.power);
      dirDefs.push({key:s.fieldKey,xkey:'k',label:`Gridded FFT (${s.shortLabel})`,color:s.color,width:1.6,dash:s.dash||'6 3',opacity:0.85});
    }
    dirDefs.push(
      {key:realKey,xkey:realKKey,label:'Linear input, this realization',color:'#7f8c8d',width:1.4,dash:'2 3',opacity:0.85},
      {key:linKey,xkey:linKKey,label:'Linear input, ensemble mean',color:'#34495e',width:1.5,dash:'4 4',opacity:0.9},
    );
    const dirSheetBounds=sheetPositiveRange(dirArrays);
    const dirAllBounds=sheetPositiveRange([...dirArrays,sp[realKey],sp[linKey]]);
    const dirYlim=dirSheetBounds&&dirAllBounds?[dirSheetBounds[0]*0.45,Math.max(dirSheetBounds[1],dirAllBounds[1])*2.2]:undefined;
    addPlot(p,'Mesh-free exact simplex Fourier spectrum (shell-averaged) vs gridded FFT',dk,dirDefs,{xlabel:'k [h/Mpc]',ylabel:`P(k) [(Mpc/h)^${S.dim}]`,xlog:true,ylog:true,ylim:dirYlim},dirData);
    $('p','panel-note',p,'Unlike the FFT spectrum (which deposits mass onto an Eulerian mesh), the mesh-free curves evaluate the exact analytic simplex Fourier integral of exp(−i k·x(q)) over the Lagrangian simplices along crystal symmetry rays k = m k_f n̂ and average |δ̂(k)|² within each logarithmic |k| shell. At high k above the Eulerian mesh Nyquist frequency, the gridded FFT rolls off from the cell-average sinc² window whereas the mesh-free simplex Fourier transform resolves power without an Eulerian grid.');
  }

  const linear=arr(d.spectra?.linear);
  if(linear.length){
    $('p','panel-note',p,`Two linear references are drawn. “This realization” is D(a)² |δ₀(k)|² of the actual random modes, averaged over the modes of each bin, so any departure of the sheet curve from it is nonlinear evolution, the cell window, or estimator error, never sample variance. “Ensemble mean” is the ${S.transfer==='bao'?'Eisenstein–Hu transfer with baryon acoustic oscillations':'smooth Eisenstein–Hu transfer'}, normalized so the ${S.dim}D top-hat variance at R = 8 Mpc/h equals σ₈² and filtered solely by the physical Gaussian smoothing exp(−k²R²) with R = ${num(S.initialSmoothing,4)} Mpc/h, scaled by (D(a)/D(1))².`);
  }
  renderVarianceScale(d);
}
function renderVarianceScale(d) {
  const vs=d.varianceScale;
  const p=panel('Variance as a function of smoothing scale σ²(R)',`Smooths each selected density field across physical radius R in Mpc/h by integrating its full discrete Fourier spectrum against an isotropic ${S.dim}D window W²(kR), and compares against the exact continuum linear integral. This reveals how variance builds up from large linear scales down to the particle spacing Δq = L/n = ${num(vs?.particleScale ?? (S.boxSize/S.n),4)} Mpc/h and observation cell size Δx = L/mesh = ${num(vs?.cellScale ?? (S.boxSize/S.mesh),4)} Mpc/h.`);
  if(!vs||!arr(vs.radii).length){
    $('p','panel-note',p,'Variance-versus-scale curves are not available for this run.');
    return;
  }
  const bar=$('div','published-controls',p);
  const addSel=(label,opts,val,onChange)=>{
    const wrap=$('label','published-select',bar);$('span','',wrap,label);
    const sel=$('select','lab-select',wrap);
    for(const [v,t] of opts){const o=$('option','',sel,t);o.value=v;}
    sel.value=val;sel.addEventListener('change',()=>onChange(sel.value));
    return sel;
  };
  addSel('Smoothing filter W(kR)',[['tophat',`${S.dim}D real-space Top-Hat W_TH(kR)`],['gaussian','Gaussian filter exp(−k²R²/2)']],spectraViewOpts.filter,(v)=>{spectraViewOpts.filter=v;render();});
  addSel('Ordinate',[['sigma2','Variance σ²(R)'],['sigma','RMS fluctuation σ(R) = √σ²(R)']],spectraViewOpts.varianceOrdinate,(v)=>{spectraViewOpts.varianceOrdinate=v;render();});

  const radii=arr(vs.radii);
  const filterKey=spectraViewOpts.filter||'tophat';
  const fam=vs[filterKey]||vs.tophat||{};
  const isRms=spectraViewOpts.varianceOrdinate==='sigma';
  const tr=(vals)=>arr(vals).map(v=>{const n=Number(v);return Number.isFinite(n)&&n>=0?(isRms?Math.sqrt(n):n):NaN;});
  const activeSeries=resolveActiveSeries(d).filter(s=>arr(s.variance?.[filterKey]).length>0);
  const vData={
    radii,
    realized:arr(fam.realized).length?tr(fam.realized):undefined,
    linear:arr(fam.linear).length?tr(fam.linear):undefined,
  };
  const vSeriesArrays=[];
  for(const s of activeSeries){
    const curve=tr(s.variance[filterKey]);
    vData[s.fieldKey]=curve;
    vSeriesArrays.push(curve);
  }
  const sheetBounds=sheetPositiveRange(vSeriesArrays);
  const allBounds=sheetPositiveRange([...vSeriesArrays,vData.realized,vData.linear]);
  const vYlim=sheetBounds&&allBounds?[sheetBounds[0]*0.5,Math.max(sheetBounds[1],allBounds[1])*1.8]:undefined;
  const filterName=filterKey==='gaussian'?'Gaussian':'Top-Hat';
  const ordSymbol=isRms?'σ(R)':'σ²(R)';
  const seriesDefs=[
    ...activeSeries.map(s=>({key:s.fieldKey,label:s.label,color:s.color,dash:s.dash,width:s.width})),
    ...(vData.realized?[{key:'realized',label:'Linear input, this realization',color:'#7f8c8d',dash:'2 3',width:1.4}]:[]),
    ...(vData.linear?[{key:'linear',label:'Linear input, ensemble mean',color:'#34495e',dash:'5 3',width:1.5}]:[]),
  ];
  addPlot(p,`${filterName} ${isRms?'RMS density fluctuation σ(R)':'density variance σ²(R)'} vs smoothing scale R`,radii,seriesDefs,{xlabel:'Smoothing scale R [Mpc/h]',ylabel:`${ordSymbol}  (${filterName} window)`,xlog:true,ylog:true,ylim:vYlim},vData);

  const refSeries=activeSeries.find(s=>s.key==='4lpt:p0')||activeSeries.find(s=>s.est==='p0')||activeSeries[0];
  const refCurve=refSeries?vData[refSeries.fieldKey]:tr(fam.sheet);
  const refName=refSeries?refSeries.shortLabel:'LPT (P₀)';
  const toRatio=(vals)=>arr(vals).map((v,i)=>{const denom=Number(refCurve[i]);return denom>1e-30?Number(v)/denom:NaN;});
  const ratioData={
    radii,
    realized:vData.realized?toRatio(vData.realized):undefined,
    linear:vData.linear?toRatio(vData.linear):undefined,
  };
  for(const s of activeSeries)ratioData[s.fieldKey]=toRatio(vData[s.fieldKey]);
  const ratioDefs=seriesDefs.map(s=>refSeries&&s.key===refSeries.fieldKey?{...s,label:`${refSeries.label} ≡ 1`}:s);
  addPlot(p,`Scale-dependent ${isRms?'RMS':'variance'} ratio relative to ${refName}: ${ordSymbol} / ${ordSymbol}_${refName}`,radii,ratioDefs,{xlabel:'Smoothing scale R [Mpc/h]',ylabel:`${ordSymbol} / ${ordSymbol}_${refName}`,xlog:true,ylog:false},ratioData);

  const nearestIdx=(target)=>{
    let best=0,bestDist=Infinity;
    for(let i=0;i<radii.length;i++){
      const dist=Math.abs(Math.log((Number(radii[i])||1)/Math.max(1e-12,target)));
      if(dist<bestDist){bestDist=dist;best=i;}
    }
    return best;
  };
  const table=$('table','summary-table',p);
  $('caption','',table,`${filterName} smoothed ${isRms?'RMS fluctuation σ(R)':'variance σ²(R)'} at characteristic physical scales`);
  const head=$('thead','',table),hr=$('tr','',head);
  const cols=[
    ...activeSeries.map(s=>[s.fieldKey,s.shortLabel,s.variance?.unsmoothed]),
    ...(vData.realized?[['realized','Linear (realized)',vs.unsmoothed?.realized]]:[]),
    ...(vData.linear?[['linear','Linear (ensemble)',vs.unsmoothed?.linear]]:[]),
  ];
  $('th','',hr,'Scale R [Mpc/h]');
  for(const [,label] of cols)$('th','',hr,label);
  const body=$('tbody','',table);
  if(vs.unsmoothed){
    const tr0=$('tr','',body);
    $('th','',tr0,'R = 0 (unsmoothed mesh cells)');
    for(const [,,raw] of cols){
      const val=raw==null?null:(isRms?Math.sqrt(Math.max(0,Number(raw))):Number(raw));
      $('td','',tr0,num(val,4));
    }
  }
  const landmarks=[
    [`Δx_mesh = ${num(vs.cellScale,4)} Mpc/h`,Number(vs.cellScale)],
    ...(Math.abs(Number(vs.particleScale)-Number(vs.cellScale))>1e-6?[[`Δq_part = ${num(vs.particleScale,4)} Mpc/h`,Number(vs.particleScale)]]:[]),
    ['R = 8 Mpc/h (σ₈ scale)',8.0],
    ...(Number(vs.initialSmoothing)>0?[[`R_ini = ${num(vs.initialSmoothing,4)} Mpc/h`,Number(vs.initialSmoothing)]]:[]),
  ];
  for(const [label,targetR] of landmarks){
    if(!(targetR>=radii[0]*0.9&&targetR<=radii[radii.length-1]*1.1))continue;
    const idx=nearestIdx(targetR);
    const trRow=$('tr','',body);
    $('th','',trRow,`${label} (at ${num(radii[idx],4)})`);
    for(const [k] of cols)$('td','',trRow,num(vData[k]?.[idx],4));
  }
}
function pdfOrdinateMeta(mode) {
  if (mode === 'cdf') {
    return {
      shortKind: 'CDF',
      yLabelSuffix: 'CDF  P(≤ ρ/ρ̄)',
      ylog: false,
      ylim: [0, 1.02],
    };
  }
  if (mode === 'peaked_cdf') {
    return {
      shortKind: 'peaked CDF',
      yLabelSuffix: 'peaked CDF  min(P(≤ ρ/ρ̄), P(≥ ρ/ρ̄))',
      ylog: true,
      ylim: undefined,
    };
  }
  if (mode === 'log10') {
    return {
      shortKind: 'PDF',
      yLabelSuffix: 'dP / d log₁₀(ρ/ρ̄)',
      ylog: true,
      ylim: undefined,
    };
  }
  return {
    shortKind: 'PDF',
    yLabelSuffix: 'dP / d(ρ/ρ̄)',
    ylog: true,
    ylim: undefined,
  };
}
function buildPdfSeriesCurve(pdfObj, weightKey, mode, targetBins, fallbackLogrho = []) {
  const fine = pdfObj?.summary?.finePdf;
  const hasFine = fine && arr(fine[weightKey]).length > 0 && arr(fine.logrho).length > 0;
  const baseLogrho = (arr(pdfObj?.logrho).length ? arr(pdfObj.logrho) : arr(fallbackLogrho)).map(Number);
  const baseVals = arr(pdfObj?.[weightKey]).map(Number);
  const baseDy = Number(pdfObj?.dy) > 0
    ? Number(pdfObj.dy)
    : (baseLogrho.length > 1 ? Math.abs(baseLogrho[1] - baseLogrho[0]) : 3.6 / 80);

  const srcLogrho = hasFine ? arr(fine.logrho).map(Number) : baseLogrho;
  const srcVals = hasFine ? arr(fine[weightKey]).map(Number) : baseVals;
  const srcDy = hasFine && Number(fine.dy) > 0
    ? Number(fine.dy)
    : (srcLogrho.length > 1 ? Math.abs(srcLogrho[1] - srcLogrho[0]) : baseDy);

  if (!srcLogrho.length || !srcVals.length) return { x: [], y: [] };

  if (mode === 'cdf' || mode === 'peaked_cdf') {
    const m = Math.min(srcLogrho.length, srcVals.length);
    const probs = new Float64Array(m);
    let tot = 0, firstSrc = -1, lastSrc = -1;
    for (let i = 0; i < m; i++) {
      const p = Math.max(0, (Number.isFinite(srcVals[i]) ? srcVals[i] : 0) * srcDy);
      probs[i] = p;
      if (p > 0) {
        if (firstSrc < 0) firstSrc = i;
        lastSrc = i;
        tot += p;
      }
    }
    if (!(tot > 0) || firstSrc < 0) {
      return { x: [], y: [] };
    }

    const nb = Math.max(8, Math.round(Number.isFinite(targetBins) && targetBins >= 4 ? targetBins : 160));
    const spanBins = lastSrc - firstSrc + 1;
    const rebinCount = Math.max(4, Math.min(nb, spanBins));
    const uLo = srcLogrho[firstSrc] - 0.5 * srcDy;
    const uHi = srcLogrho[lastSrc] + 0.5 * srcDy;
    const dstDy = (uHi - uLo) / rebinCount;
    const dstProb = new Float64Array(rebinCount);
    for (let i = firstSrc; i <= lastSrc; i++) {
      const p = probs[i];
      if (!(p > 0)) continue;
      const a = srcLogrho[i] - 0.5 * srcDy;
      const b = a + srcDy;
      const j0 = Math.max(0, Math.floor((a - uLo) / dstDy));
      const j1 = Math.min(rebinCount - 1, Math.floor((b - 1e-14 - uLo) / dstDy));
      for (let j = j0; j <= j1; j++) {
        const c = uLo + j * dstDy;
        const d = c + dstDy;
        const overlap = Math.max(0, Math.min(b, d) - Math.max(a, c));
        dstProb[j] += p * (overlap / srcDy);
      }
    }

    let jFirst = -1, jLast = -1, dstTot = 0;
    for (let j = 0; j < rebinCount; j++) {
      if (dstProb[j] > 0) {
        if (jFirst < 0) jFirst = j;
        jLast = j;
        dstTot += dstProb[j];
      }
    }
    if (!(dstTot > 0) || jFirst < 0) return { x: [], y: [] };

    // Exact median crossing yMed from the fine source histogram
    let cumFine = 0, yMed = 0.5 * (uLo + uHi), foundMed = false;
    for (let i = firstSrc; i <= lastSrc; i++) {
      const p = probs[i];
      const fPrev = cumFine / tot;
      cumFine += p;
      const fNext = Math.min(1, cumFine / tot);
      if (!foundMed && fNext >= 0.5) {
        const uLeft = srcLogrho[i] - 0.5 * srcDy;
        const frac = fNext > fPrev ? (0.5 - fPrev) / (fNext - fPrev) : 0.5;
        yMed = uLeft + Math.max(0, Math.min(1, frac)) * srcDy;
        foundMed = true;
        break;
      }
    }

    // Build knots strictly from the first non-zero bin center to the last non-zero bin center
    const uKnots = [];
    const fLeKnots = [];
    const fGeKnots = [];
    let cumBefore = 0;
    for (let j = jFirst; j <= jLast; j++) {
      const p = dstProb[j];
      if (p > 0) {
        const uCenter = uLo + (j + 0.5) * dstDy;
        const fLe = (cumBefore + 0.5 * p) / dstTot;
        const fGe = Math.max(0, dstTot - cumBefore - 0.5 * p) / dstTot;
        if (fLe > 0 && (mode === 'cdf' || fGe > 0)) {
          uKnots.push(uCenter);
          fLeKnots.push(Math.min(1, fLe));
          fGeKnots.push(Math.min(1, fGe));
        }
      }
      cumBefore += p;
    }
    if (!uKnots.length) return { x: [], y: [] };

    // If nb > uKnots.length, upsample smoothly in log-probability space strictly within [uKnots[0], uKnots[last]]
    let uSample = uKnots.slice();
    let fLeSample = fLeKnots.slice();
    let fGeSample = fGeKnots.slice();
    if (nb > uKnots.length && uKnots.length >= 2) {
      const uStart = uKnots[0], uEnd = uKnots[uKnots.length - 1];
      uSample = new Array(nb);
      for (let k = 0; k < nb; k++) {
        uSample[k] = uStart + (k / (nb - 1)) * (uEnd - uStart);
      }
      if (mode === 'peaked_cdf' && foundMed && yMed > uStart && yMed < uEnd) {
        if (!uSample.some((u) => Math.abs(u - yMed) < 1e-7)) {
          uSample.push(yMed);
          uSample.sort((a, b) => a - b);
        }
      }
      const logLe = pchipInterpolate(uKnots, fLeKnots.map((v) => Math.log10(v)), uSample);
      const logGe = pchipInterpolate(uKnots, fGeKnots.map((v) => Math.log10(Math.max(1e-30, v))), uSample);
      fLeSample = logLe.map((lv) => Math.min(1, Math.pow(10, lv)));
      fGeSample = logGe.map((lv) => Math.min(1, Math.pow(10, lv)));
    } else if (mode === 'peaked_cdf' && foundMed && uSample.length >= 2) {
      const uStart = uSample[0], uEnd = uSample[uSample.length - 1];
      if (yMed > uStart && yMed < uEnd && !uSample.some((u) => Math.abs(u - yMed) < 1e-7)) {
        const insIdx = uSample.findIndex((u) => u > yMed);
        if (insIdx > 0) {
          uSample.splice(insIdx, 0, yMed);
          fLeSample.splice(insIdx, 0, 0.5);
          fGeSample.splice(insIdx, 0, 0.5);
        }
      }
    }

    const outX = new Array(uSample.length);
    const outY = new Array(uSample.length);
    for (let k = 0; k < uSample.length; k++) {
      const u = uSample[k];
      outX[k] = Math.pow(10, u);
      if (mode === 'cdf') {
        outY[k] = fLeSample[k];
      } else if (foundMed && Math.abs(u - yMed) < 1e-7) {
        outY[k] = 0.5;
      } else {
        outY[k] = Math.min(fLeSample[k], fGeSample[k]);
      }
    }
    return { x: outX, y: outY };
  }

  const defCount = baseLogrho.length || 80;
  const nb = Math.max(6, Math.round(Number.isFinite(targetBins) && targetBins >= 4 ? targetBins : defCount));
  if (nb === defCount && baseVals.length === defCount) {
    const outX = baseLogrho.map((y) => Math.pow(10, y));
    if (mode === 'log10') return { x: outX, y: baseVals };
    const sFactor = Math.pow(10, baseDy / 2) - Math.pow(10, -baseDy / 2);
    return {
      x: outX,
      y: baseVals.map((v, i) => (v * baseDy) / Math.max(1e-30, outX[i] * sFactor)),
    };
  }

  const m = Math.min(srcLogrho.length, srcVals.length);
  const rebinCount = Math.min(nb, m);
  const yLo = srcLogrho[0] - 0.5 * srcDy;
  const yHi = srcLogrho[m - 1] + 0.5 * srcDy;
  const dstDy = (yHi - yLo) / rebinCount;
  const dstProb = new Float64Array(rebinCount);
  for (let i = 0; i < m; i++) {
    const p = Math.max(0, (Number(srcVals[i]) || 0) * srcDy);
    if (!(p > 0)) continue;
    const a = srcLogrho[i] - 0.5 * srcDy;
    const b = a + srcDy;
    const j0 = Math.max(0, Math.floor((a - yLo) / dstDy));
    const j1 = Math.min(rebinCount - 1, Math.floor((b - 1e-14 - yLo) / dstDy));
    for (let j = j0; j <= j1; j++) {
      const c = yLo + j * dstDy;
      const d = c + dstDy;
      const overlap = Math.max(0, Math.min(b, d) - Math.max(a, c));
      dstProb[j] += p * (overlap / srcDy);
    }
  }
  const outX = new Array(rebinCount);
  const outY = new Array(rebinCount);
  const sFactor = Math.pow(10, dstDy / 2) - Math.pow(10, -dstDy / 2);
  for (let j = 0; j < rebinCount; j++) {
    const yCenter = yLo + (j + 0.5) * dstDy;
    const rho = Math.pow(10, yCenter);
    outX[j] = rho;
    const p = dstProb[j];
    outY[j] = mode === 'log10' ? p / dstDy : p / Math.max(1e-30, rho * sFactor);
  }
  if (nb > m) {
    return resampleCurve(outX, outY, nb, { xlog: true, ylog: true });
  }
  return { x: outX, y: outY };
}

function renderPdfs(d) {
  const activeSeries=resolveActiveSeries(d).filter(s=>arr(s.pdf?.volume).length>0);
  const p=panel('Density probability distributions',`Comparing probability distributions (PDF, CDF, and peaked CDF) for all selected Trajectory × Estimator pairs as a function of ρ/ρ̄. Continuous P₀ and P₁ sheet distributions are always evaluated directly from the Lagrangian simplices (intervals in 1D, triangles in 2D, Kuhn tetrahedra in 3D) up to 640 native bins without grid deposition, extending smoothly across deep voids and high-density caustics; only the grid-based particle estimators (NGP, CIC, TSC) are limited in density range by the Eulerian grid and zero-particle starvation. Primary weighting: ${S.weight}. Observation filter R/L = ${num(S.smoothing,4)}. Captured integrals are reported below.`);
  const logrho=arr(d.pdf?.logrho);
  const rhoAxis=logrho.map(y=>Math.pow(10,Number(y)));
  const isCdfMode=pdfViewOpts.ordinate==='cdf'||pdfViewOpts.ordinate==='peaked_cdf';
  const defaultPdfBins=isCdfMode?160:80;
  const activePdfBins=pdfViewOpts.bins||defaultPdfBins;

  const bar=$('div','published-controls',p);
  const wrap=$('label','published-select',bar);$('span','',wrap,'PDF ordinate');
  const sel=$('select','lab-select',wrap);
  for(const [v,t] of [
    ['linear','PDF: dP / d(ρ/ρ̄)  (probability density per unit ρ/ρ̄)'],
    ['log10','PDF: dP / d log₁₀(ρ/ρ̄)  (probability density per decade)'],
    ['cdf','CDF: P(≤ ρ/ρ̄)  (cumulative distribution function)'],
    ['peaked_cdf','Peaked CDF: min(P(≤ ρ/ρ̄), P(≥ ρ/ρ̄))  (folded two-sided tail probability)'],
  ]){const o=$('option','',sel,t);o.value=v;}
  sel.value=pdfViewOpts.ordinate;
  sel.addEventListener('change',()=>{pdfViewOpts.ordinate=sel.value;render();});

  const binWrap=$('label','published-select',bar);$('span','',binWrap,isCdfMode?'Curve points / quantiles':'Histogram bins');
  const binRow=$('span','qd-pt-axis',binWrap);
  const bMinus=$('button','qd-btn',binRow,'−');bMinus.type='button';bMinus.style.padding='0.1rem 0.45rem';bMinus.title='Fewer bins / points (reduce binning noise)';
  const bInput=$('input','lab-number',binRow);bInput.type='number';bInput.min='8';bInput.max='640';bInput.step='8';bInput.style.width='4.5rem';bInput.value=String(activePdfBins);
  const bPlus=$('button','qd-btn',binRow,'+');bPlus.type='button';bPlus.style.padding='0.1rem 0.45rem';bPlus.title='More bins / points (up to 640 exact simplex bins)';
  const bAuto=$('button','qd-btn',binRow,`Auto (${defaultPdfBins})`);bAuto.type='button';bAuto.style.padding='0.1rem 0.5rem';

  const pdfPlots=[];
  const syncPdfBins=(nb,sourcePlot=null)=>{
    pdfViewOpts.bins=nb;
    bInput.value=String(nb||defaultPdfBins);
    for(const pl of pdfPlots){
      if(pl&&pl!==sourcePlot)pl.setBins(nb,false);
    }
  };
  bMinus.addEventListener('click',()=>{if(pdfPlots[0])pdfPlots[0]._stepBins(-1);});
  bPlus.addEventListener('click',()=>{if(pdfPlots[0])pdfPlots[0]._stepBins(1);});
  bInput.addEventListener('change',()=>{
    const v=parseInt(bInput.value,10);
    if(Number.isFinite(v)&&v>=6){
      if(pdfPlots[0])pdfPlots[0].setBins(v,true);
      else{pdfViewOpts.bins=v;render();}
    }
  });
  bAuto.addEventListener('click',()=>{
    pdfViewOpts.bins=null;
    bInput.value=String(defaultPdfBins);
    for(const pl of pdfPlots)if(pl)pl.setBins(null,false);
  });

  $('p','panel-note',p,`Direct simplex evaluation active: P₀ and P₁ sheet distributions and quantiles are accumulated directly over all Lagrangian simplices (using exact Jacobian density 1/|J_T| for P₀ and analytical nodal vertex Jacobians 1/|J(q_i)| for P₁) at up to 640 logarithmic bins without depositing onto an Eulerian mesh. Use the bins (− / N / +) controls above or under each chart to refine or coarsen the binning.`);
  const pdfData={rhoAxis};
  const volumeDefs=[],massDefs=[];
  for(const s of activeSeries){
    const volCurve=buildPdfSeriesCurve(s.pdf,'volume',pdfViewOpts.ordinate,activePdfBins,logrho);
    const massCurve=buildPdfSeriesCurve(s.pdf,'mass',pdfViewOpts.ordinate,activePdfBins,logrho);
    const vxKey=`${s.fieldKey}_vx`,vKey=`${s.fieldKey}_vol`,mxKey=`${s.fieldKey}_mx`,mKey=`${s.fieldKey}_mass`;
    pdfData[vxKey]=volCurve.x;
    pdfData[vKey]=volCurve.y;
    pdfData[mxKey]=massCurve.x;
    pdfData[mKey]=massCurve.y;
    volumeDefs.push({
      key:vKey,
      xkey:vxKey,
      label:s.label,
      color:s.color,
      dash:s.dash,
      width:s.width,
      resample:(nb)=>buildPdfSeriesCurve(s.pdf,'volume',pdfViewOpts.ordinate,nb,logrho),
    });
    massDefs.push({
      key:mKey,
      xkey:mxKey,
      label:s.label,
      color:s.color,
      dash:s.dash,
      width:s.width,
      resample:(nb)=>buildPdfSeriesCurve(s.pdf,'mass',pdfViewOpts.ordinate,nb,logrho),
    });
  }
  const ordMeta=pdfOrdinateMeta(pdfViewOpts.ordinate);
  const binLabel=isCdfMode?'pts':'bins';
  if(S.weight==='mass'){
    const p1=addPlot(p,`PRIMARY · mass-weighted ${ordMeta.shortKind}`,rhoAxis,massDefs,{xlabel:'ρ / ρ̄',ylabel:`mass-weighted ${ordMeta.yLabelSuffix}`,xlog:true,ylog:ordMeta.ylog,ylim:ordMeta.ylim,binLabel,bins:activePdfBins,minBins:8,maxBins:640,onBinsChange:(nb)=>syncPdfBins(nb,p1)},pdfData);
    const p2=addPlot(p,`Volume-weighted ${ordMeta.shortKind}`,rhoAxis,volumeDefs,{xlabel:'ρ / ρ̄',ylabel:`volume-weighted ${ordMeta.yLabelSuffix}`,xlog:true,ylog:ordMeta.ylog,ylim:ordMeta.ylim,binLabel,bins:activePdfBins,minBins:8,maxBins:640,onBinsChange:(nb)=>syncPdfBins(nb,p2)},pdfData);
    if(p1)pdfPlots.push(p1);if(p2)pdfPlots.push(p2);
  } else {
    const p1=addPlot(p,`PRIMARY · volume-weighted ${ordMeta.shortKind}`,rhoAxis,volumeDefs,{xlabel:'ρ / ρ̄',ylabel:`volume-weighted ${ordMeta.yLabelSuffix}`,xlog:true,ylog:ordMeta.ylog,ylim:ordMeta.ylim,binLabel,bins:activePdfBins,minBins:8,maxBins:640,onBinsChange:(nb)=>syncPdfBins(nb,p1)},pdfData);
    const p2=addPlot(p,`Mass-weighted ${ordMeta.shortKind}`,rhoAxis,massDefs,{xlabel:'ρ / ρ̄',ylabel:`mass-weighted ${ordMeta.yLabelSuffix}`,xlog:true,ylog:ordMeta.ylog,ylim:ordMeta.ylim,binLabel,bins:activePdfBins,minBins:8,maxBins:640,onBinsChange:(nb)=>syncPdfBins(nb,p2)},pdfData);
    if(p1)pdfPlots.push(p1);if(p2)pdfPlots.push(p2);
  }
  renderPdfSummaries(p,d.pdf||{},activeSeries);
}
function renderPdfSummaries(parent,pdf,activeSeries){
  const withSum=(activeSeries||[]).filter(s=>s.pdf?.summary);
  if(!withSum.length)return;
  const table=$('table','summary-table',parent);$('caption','',table,'Full-field summaries and displayed-bin probability integrals');
  const head=$('thead','',table),hr=$('tr','',head);
  $('th','',hr,'Statistic');
  for(const s of withSum)$('th','',hr,s.shortLabel);
  const body=$('tbody','',table);
  const qps=arr(withSum[0].pdf.summary?.probabilities||pdf.summary?.probabilities);
  const rows=[
    ['Mean density',withSum.map(s=>s.pdf.summary?.mean)],
    ['Variance',withSum.map(s=>s.pdf.summary?.variance)],
    ['Reduced skewness S₃',withSum.map(s=>s.pdf.summary?.S3)],
    ['Reduced kurtosis S₄',withSum.map(s=>s.pdf.summary?.S4)],
    ['Captured volume-PDF probability',withSum.map(s=>s.pdf.capturedVolume)],
    ['Captured mass-PDF probability',withSum.map(s=>s.pdf.capturedMass)],
  ];
  for(let i=0;i<qps.length;i++){
    const suffix=` (${num(Number(qps[i])*100,3)}%)`;
    rows.push(
      [`Volume-weighted ρ quantile${suffix}`,withSum.map(s=>arr(s.pdf.summary?.volumeQuantiles)[i])],
      [`Mass-weighted ρ quantile${suffix}`,withSum.map(s=>arr(s.pdf.summary?.massQuantiles)[i])],
    );
  }
  for(const [statLabel,vals] of rows){
    const tr=$('tr','',body);
    $('th','',tr,statLabel);
    for(const v of vals)$('td','',tr,num(v,5));
  }
}
function renderRsd(d) {
  const rsd=d.rsd;
  if(!rsd){
    panel('Redshift-space distortions (RSD)','Redshift-space analysis was not returned for this run.');
    return;
  }
  const dim=Number(d.dim||S.dim),losAxis=Number(rsd.losAxis??(dim-1));
  const losName=dim===1?'x₀':dim===2?'x₁ (vertical)':'x₂ (z-axis)';
  const sum=rsd.summary||{};
  const percent=v=>Number.isFinite(Number(v))?`${num(100*Number(v),3)}%`:'—';

  // Panel 1: Redshifted Sheet Geometry, Velocity-Induced Shell Crossing, and Sampled Simplex Gaussian Packets
  const pSheet=panel(
    'Redshifted phase-space sheet & velocity-induced shell crossing',
    `Each Lagrangian vertex q maps to redshift space s = x + (v_∥ / aH) n̂ = x + a (dx_∥/da) n̂ along line-of-sight axis ${losName}. By the matrix determinant lemma, the redshift-space simplex Jacobian is J_{s,T} = J_T (1 + a ∂_∥ v_∥)_T. Consequently, the sheet shell-crosses in redshift space (J_{s,T} ≤ 0) wherever the line-of-sight velocity convergence exceeds the Hubble flow (a ∂_∥ v_∥ ≤ −1), even when the real-space sheet is still completely single-stream (J_T > 0). Fitting a ${dim}D Gaussian packet to each simplex with width set by the velocity gradient tensor across the simplex captures both large-scale Kaiser infall and small-scale Fingers-of-God radial dispersion.`
  );

  const strip=$('div','metric-strip',pSheet);
  for(const [label,val] of [
    ['Growth rate f(a) = d ln D / d ln a',num(sum.growthRateF,4)],
    ['Kaiser LOS boost (1 + f)²',num(sum.kaiserLosFactor,4)],
    ['Kaiser monopole factor',num(sum.kaiserMonopoleFactor,4)],
    ['Real-space min J_T',num(sum.minRealSimplexJacobian,4)],
    ['Redshift-space min J_{s,T}',num(sum.minRsdSimplexJacobian,4)],
    ['Real-space inverted mass',percent(sum.realInvertedMassFraction)],
    ['Redshift-space inverted mass',percent(sum.rsdInvertedMassFraction)],
    ['Pre-crossing RSD folded mass (J_T>0, J_{s,T}≤0)',percent(sum.rsdPreCrossingInvertedMass)],
    ['LOS RMS velocity σ_{v,∥}',`${num(sum.rmsVelocityLosKmS,4)} km/s`],
    ['Simplex velocity-gradient width RMS / max',`${num(sum.rmsSimplexVelocityWidthKmS,3)} / ${num(sum.maxSimplexVelocityWidthKmS,4)} km/s`],
  ]){
    const item=$('div','metric',strip);
    $('span','metric-label',item,label);$('strong','metric-value',item,val);
  }

  const sheetBar=$('div','published-controls',pSheet);
  const modeWrap=$('label','published-select',sheetBar);$('span','',modeWrap,'Sheet view');
  const modeSel=$('select','lab-select',modeWrap);
  for(const [v,t] of [
    ['rsd_crossing','Redshifted sheet s(q) · highlight RSD shell-crossed simplices (J_{s,T} ≤ 0 in crimson)'],
    ['overlay','Overlay: Real-space sheet x(q) (blue) vs Redshifted sheet s(q) (orange) + Gaussian samples'],
    ['real_vs_rsd_j','Simplex volume compression ratio J_{s,T} / J_T across the sheet'],
  ]){const o=$('option','',modeSel,t);o.value=v;}
  modeSel.value=rsdViewOpts.sheetMode;
  modeSel.addEventListener('change',()=>{rsdViewOpts.sheetMode=modeSel.value;render();});

  const sampWrap=$('label','published-select',sheetBar);$('span','',sampWrap,'Gaussian packet samples');
  const sampSel=$('select','lab-select',sampWrap);
  for(const [v,t] of [['1','Show sampled points from simplex Gaussians'],['0','Hide sampled points']]){const o=$('option','',sampSel,t);o.value=v;}
  sampSel.value=rsdViewOpts.showSamples?'1':'0';
  sampSel.addEventListener('change',()=>{rsdViewOpts.showSamples=sampSel.value==='1';render();});

  const wf=rsd.wireframe||{};
  const cSheet=canvasFor(pSheet,'Redshifted phase-space sheet and velocity-gradient Gaussian samples',true);
  const drawRsdSheet=()=>paintCanvas(cSheet,(ctx,W,H)=>{
    const nw=Number(wf.n||0),xr=arr(wf.xReal),xs=arr(wf.xRsd),uLos=arr(wf.uLos),jr=arr(wf.jReal),js=arr(wf.jRsd),samps=arr(wf.gaussSamples);
    const ink=getComputedStyle(cSheet).color;
    if(!nw||!xs.length){ctx.fillStyle=ink;ctx.fillText('Redshifted wireframe was not returned.',32,40);return;}
    const idx=(i,j=0,k=0)=>dim===1?i:dim===2?i*nw+j:(i*nw+j)*nw+k;
    const get=(a,id,axis)=>Number(a[id*dim+axis]??0);
    const pad=56,plotW=W-2*pad,plotH=H-2*pad;

    if(dim===1){
      let miny=Infinity,maxy=-Infinity;
      for(let i=0;i<nw;i++){
        const u=Number(uLos[i]||0);
        miny=Math.min(miny,u);maxy=Math.max(maxy,u);
      }
      if(!(maxy>miny)){miny-=0.05;maxy+=0.05;}
      const mapXY=(xVal,uVal)=>[pad+((xVal%1+1)%1)*plotW,H-pad-(uVal-miny)*plotH/(maxy-miny)];
      // Real-space curve (x, u_los) vs Redshifted curve (s, u_los)
      ctx.lineWidth=1.8;
      ctx.strokeStyle='#0072B2';
      ctx.beginPath();
      for(let i=0;i<nw;i++){
        const [px,py]=mapXY(get(xr,i,0),Number(uLos[i]||0));
        if(i===0)ctx.moveTo(px,py);else if(Math.abs(get(xr,i,0)-get(xr,i-1,0))<0.4)ctx.lineTo(px,py);else ctx.moveTo(px,py);
      }
      ctx.stroke();
      ctx.strokeStyle='#D55E00';
      ctx.beginPath();
      for(let i=0;i<nw;i++){
        const [px,py]=mapXY(get(xs,i,0),Number(uLos[i]||0));
        if(i===0)ctx.moveTo(px,py);else if(Math.abs(get(xs,i,0)-get(xs,i-1,0))<0.4)ctx.lineTo(px,py);else ctx.moveTo(px,py);
      }
      ctx.stroke();
      for(let i=0;i<nw;i++){
        const [px,py]=mapXY(get(xs,i,0),Number(uLos[i]||0));
        const crossed=Number(js[i])<=0;
        ctx.fillStyle=crossed?'#c0392b':'#D55E00';
        ctx.beginPath();ctx.arc(px,py,crossed?3.2:2.1,0,Math.PI*2);ctx.fill();
      }
      ctx.fillStyle=ink;ctx.font='14px system-ui';ctx.textAlign='center';
      ctx.fillText('position x/L (blue) and redshift coordinate s/L = x/L + u_∥ (orange; crimson where J_{s,T} ≤ 0)',W/2,H-14);
      ctx.save();ctx.translate(18,H/2);ctx.rotate(-Math.PI/2);ctx.fillText('LOS velocity displacement u_∥ = v_∥ / (a H L)',0,0);ctx.restore();ctx.textAlign='start';
      return;
    }

    const side=Math.min(plotW,plotH),cx=W/2,cy=H/2;
    const proj=(arrPos,id)=>{
      if(dim===2)return [cx+(get(arrPos,id,0)-0.5)*side,cy-(get(arrPos,id,1)-0.5)*side];
      return project3(get(arrPos,id,0)-0.5,get(arrPos,id,1)-0.5,get(arrPos,id,2)-0.5,S.rotate,S.tilt,side*0.82,cx,cy);
    };
    const drawEdge=(arrPos,a,b)=>{
      let d2=0;for(let k=0;k<dim;k++){const df=get(arrPos,a,k)-get(arrPos,b,k);d2+=df*df;}
      if(d2>0.2)return;
      const [x1,y1]=proj(arrPos,a),[x2,y2]=proj(arrPos,b);
      ctx.beginPath();ctx.moveTo(x1,y1);ctx.lineTo(x2,y2);ctx.stroke();
    };

    if(rsdViewOpts.showSamples&&samps.length>=dim){
      ctx.fillStyle='rgba(0, 158, 115, 0.36)';
      const nPts=Math.floor(samps.length/dim);
      for(let p=0;p<nPts;p++){
        let px,py;
        if(dim===2){px=cx+(Number(samps[p*2])-0.5)*side;py=cy-(Number(samps[p*2+1])-0.5)*side;}
        else{[px,py]=project3(Number(samps[p*3])-0.5,Number(samps[p*3+1])-0.5,Number(samps[p*3+2])-0.5,S.rotate,S.tilt,side*0.82,cx,cy);}
        ctx.beginPath();ctx.arc(px,py,1.6,0,Math.PI*2);ctx.fill();
      }
    }

    if(rsdViewOpts.sheetMode==='overlay'){
      ctx.strokeStyle='rgba(0, 114, 178, 0.38)';ctx.lineWidth=0.9;
      if(dim===2){
        for(let i=0;i+1<nw;i++)for(let j=0;j+1<nw;j++){
          const a=idx(i,j),b=idx(i+1,j),c=idx(i,j+1),e=idx(i+1,j+1);
          drawEdge(xr,a,b);drawEdge(xr,a,c);drawEdge(xr,a,e);
        }
      }else{
        for(let i=0;i+1<nw;i++)for(let j=0;j+1<nw;j++)for(let k=0;k+1<nw;k++){
          const a=idx(i,j,k),b=idx(i+1,j,k),c=idx(i,j+1,k),dd=idx(i,j,k+1);
          drawEdge(xr,a,b);drawEdge(xr,a,c);drawEdge(xr,a,dd);
        }
      }
    }

    // Draw Redshifted Sheet edges
    if(dim===2){
      for(let i=0;i+1<nw;i++)for(let j=0;j+1<nw;j++){
        const a=idx(i,j),b=idx(i+1,j),c=idx(i,j+1),e=idx(i+1,j+1);
        const minJ=Math.min(Number(js[a]),Number(js[b]),Number(js[c]),Number(js[e]));
        if(rsdViewOpts.sheetMode==='real_vs_rsd_j'){
          const ratio=Number(js[a])/Math.max(1e-3,Math.abs(Number(jr[a])));
          ctx.strokeStyle=ratio<=0?'#c0392b':ratio<0.6?'#d55e00':ratio<1.1?'#0072b2':'#56b4e9';
          ctx.lineWidth=ratio<=0?1.5:1.0;
        }else if(rsdViewOpts.sheetMode==='overlay'){
          ctx.strokeStyle=minJ<=0?'rgba(192, 57, 43, 0.85)':'rgba(213, 94, 0, 0.72)';
          ctx.lineWidth=minJ<=0?1.4:1.0;
        }else{
          ctx.strokeStyle=minJ<=0?'rgba(204, 37, 41, 0.9)':'rgba(0, 114, 178, 0.62)';
          ctx.lineWidth=minJ<=0?1.55:0.95;
        }
        drawEdge(xs,a,b);drawEdge(xs,a,c);drawEdge(xs,a,e);drawEdge(xs,b,e);drawEdge(xs,c,e);
      }
    }else{
      for(let i=0;i+1<nw;i++)for(let j=0;j+1<nw;j++)for(let k=0;k+1<nw;k++){
        const a=idx(i,j,k),b=idx(i+1,j,k),c=idx(i,j+1,k),dd=idx(i,j,k+1);
        const crossed=Math.min(Number(js[a]),Number(js[b]),Number(js[c]),Number(js[dd]))<=0;
        ctx.strokeStyle=crossed?'rgba(204, 37, 41, 0.78)':'rgba(0, 114, 178, 0.32)';
        ctx.lineWidth=crossed?1.2:0.65;
        drawEdge(xs,a,b);drawEdge(xs,a,c);drawEdge(xs,a,dd);
      }
    }

    ctx.fillStyle=ink;ctx.font='14px system-ui';ctx.textAlign='center';
    ctx.fillText(dim===2?'transverse x₀/L  (blue = single-stream J_{s,T}>0, crimson = RSD shell-crossed J_{s,T}≤0, green dots = simplex Gaussian samples)':'orthographic horizontal / L  (crimson = RSD shell-crossed simplices J_{s,T}≤0)',W/2,H-14);
    ctx.save();ctx.translate(18,H/2);ctx.rotate(-Math.PI/2);
    ctx.fillText(dim===2?'line-of-sight redshift coordinate s₁/L = x₁/L + v₁/(aHL)':'line-of-sight projection s/L',0,0);
    ctx.restore();ctx.textAlign='start';
  });
  drawRsdSheet();
  addPngButton(cSheet,'redshifted-phase-space-sheet');

  // Panel 2: Spatial Maps (Real Space, Exact Redshifted Sheet, Simplex Velocity-Gradient Gaussian Field, Sampled Field)
  const pMaps=panel(
    'Real-space vs Redshift-space density maps (Exact Sheet, Velocity-Gradient Gaussians, and Sampled Realization)',
    `All four density maps use the shared Coolwarm color scale ([${num(densityViewOpts.vmin??0.1,3)}, ${num(densityViewOpts.vmax??10,3)}]) with line-of-sight axis ${losName}. (1) Real-space P₀ sheet density ρ(x)/ρ̄; (2) Exact R3D-clipped redshifted sheet ρ_s(s)/ρ̄, showing sharp Kaiser pancakes and redshift-space caustics; (3) Simplex velocity-gradient Gaussian field ρ_{s,Gauss}(s)/ρ̄, where each simplex is fitted with a ${dim}D Gaussian whose line-of-sight dispersion is set by the velocity gradient tensor across the simplex; (4) Monte Carlo sampled realization drawn from the fitted simplex Gaussians, exhibiting realistic Fingers-of-God radial streaks.`
  );
  const meshRsd=Number(rsd.mesh||S.mesh);
  const maps=rsd.maps||{};
  const effMin=Number.isFinite(densityViewOpts.vmin)?densityViewOpts.vmin:0.1;
  const effMax=Number.isFinite(densityViewOpts.vmax)&&densityViewOpts.vmax>effMin?densityViewOpts.vmax:10;
  const mapGrid=$('div','density-pair',pMaps);
  drawDensityMap(mapGrid,'1 · Real-space P₀ sheet ρ(x)/ρ̄',maps.realRho,'Real-space P0 sheet density',dim,meshRsd,'rsd-real-sheet',effMin,effMax,1);
  drawDensityMap(mapGrid,'2 · Exact redshifted sheet ρ_s(s)/ρ̄ (R3D simplices)',maps.rsdSheetRho,'Exact redshifted sheet density',dim,meshRsd,'rsd-exact-sheet',effMin,effMax,1);
  drawDensityMap(mapGrid,`3 · Simplex velocity-gradient ${dim}D Gaussian field`,maps.rsdGaussRho,'Simplex velocity-gradient Gaussian RSD density',dim,meshRsd,'rsd-gaussian-packets',effMin,effMax,1);
  drawDensityMap(mapGrid,'4 · Sampled realization from simplex Gaussians',maps.rsdSampledRho,'Sampled simplex Gaussian RSD density',dim,meshRsd,'rsd-sampled-gaussians',effMin,effMax,1);

  // Panel 3: Anisotropic Power Spectra & Multipoles (Kaiser Boost vs Fingers-of-God Damping)
  const pSpec=panel(
    'Anisotropic redshift-space power spectra: Kaiser enhancement & Fingers-of-God damping',
    `In the linear Kaiser regime, coherent infall along the line of sight boosts Fourier modes by (1 + f μ²)², amplifying the angle-averaged monopole P₀(k) by ${num(sum.kaiserMonopoleFactor,4)}× and the line-of-sight modes (μ → 1) by (1 + f)² = ${num(sum.kaiserLosFactor,4)}×. At higher wavenumbers k, redshift-space shell crossing and the velocity-gradient dispersion across collapsing simplices damp line-of-sight power (the Fingers-of-God effect).`
  );
  const sp=rsd.spectra||{};
  const ks=arr(sp.k);
  const specBounds=sheetPositiveRange([sp.realMonopole,sp.rsdSheetMonopole,sp.rsdGaussMonopole,sp.kaiserMonopole,sp.rsdSheetLos]);
  const specYlim=specBounds?[specBounds[0]*0.45,specBounds[1]*2.2]:undefined;
  addPlot(
    pSpec,
    'Monopole P₀(k) & line-of-sight wedge P_∥(k) vs real-space P_r(k)',
    ks,
    [
      {key:'realMonopole',label:'Real-space monopole P_r(k)',color:'#34495e',dash:'5 3',width:1.8},
      {key:'kaiserMonopole',label:`Exact linear Kaiser monopole ⟨(1+fμ²)² P_r⟩ (${num(sum.kaiserMonopoleFactor,3)}×)`,color:'#7f8c8d',dash:'2 3',width:1.7},
      {key:'rsdSheetMonopole',label:'Redshifted P₀ sheet monopole P_{0,s}(k)',color:'#0072B2',width:2.2},
      {key:'rsdGaussMonopole',label:'Simplex velocity-gradient Gaussian P_{0,Gauss}(k)',color:'#009E73',dash:'7 3',width:2.1},
      {key:'rsdSheetLos',label:'Redshifted sheet LOS wedge P_∥(k) (μ² ≥ 0.5)',color:'#D55E00',width:1.9},
      {key:'rsdGaussLos',label:'Simplex Gaussian LOS wedge P_∥,Gauss(k) (FoG damped)',color:'#CC79A7',dash:'4 2.5',width:1.9},
    ],
    {xlabel:'k [h/Mpc]',ylabel:`P(k) [(Mpc/h)^${dim}]`,xlog:true,ylog:true,ylim:specYlim},
    sp
  );

  const specBar=$('div','published-controls',pSpec);
  const specWrap=$('label','published-select',specBar);$('span','',specWrap,'Anisotropy diagnostic');
  const specSel=$('select','lab-select',specWrap);
  for(const [v,t] of [
    ['ratio','Redshift-to-real power ratio P_s(k) / P_r(k) vs Kaiser plateau & FoG damping'],
    ['quadrupole','Quadrupole-to-monopole anisotropy ratio P₂(k) / P₀(k)'],
    ['wedges','Line-of-sight P_∥(k) vs Transverse P_⊥(k) ratio P_∥(k) / P_⊥(k)'],
  ]){const o=$('option','',specSel,t);o.value=v;}
  specSel.value=rsdViewOpts.specMode;
  specSel.addEventListener('change',()=>{rsdViewOpts.specMode=specSel.value;render();});

  const safeDiv=(a,b)=>arr(a).map((v,i)=>{const d0=Number(arr(b)[i]);return d0>1e-30?Number(v)/d0:NaN;});
  if(rsdViewOpts.specMode==='ratio'){
    const ratioData={
      k:ks,
      unity:ks.map(()=>1),
      kaiserMonoRatio:safeDiv(sp.kaiserMonopole,sp.realMonopole),
      kaiserLosRatio:safeDiv(sp.kaiserLos,sp.realMonopole),
      sheetMonoRatio:safeDiv(sp.rsdSheetMonopole,sp.realMonopole),
      gaussMonoRatio:safeDiv(sp.rsdGaussMonopole,sp.realMonopole),
      sheetLosRatio:safeDiv(sp.rsdSheetLos,sp.realMonopole),
      gaussLosRatio:safeDiv(sp.rsdGaussLos,sp.realMonopole),
    };
    addPlot(
      pSpec,
      'Redshift-to-real power ratio P(k) / P_r(k): Kaiser boost at low k vs Fingers-of-God suppression at high k',
      ks,
      [
        {key:'unity',label:'Real-space baseline (1.0)',color:'#7f8c8d',dash:'2 3',width:1.3},
        {key:'kaiserMonoRatio',label:'Exact linear Kaiser monopole ratio',color:'#34495e',dash:'5 3',width:1.6},
        {key:'kaiserLosRatio',label:'Exact linear Kaiser LOS wedge ratio',color:'#e69f00',dash:'2 2',width:1.5},
        {key:'sheetMonoRatio',label:'Redshifted sheet P_{0,s}(k) / P_r(k)',color:'#0072B2',width:2.1},
        {key:'gaussMonoRatio',label:'Simplex Gaussian P_{0,Gauss}(k) / P_r(k)',color:'#009E73',dash:'7 3',width:2.0},
        {key:'sheetLosRatio',label:'Redshifted sheet LOS P_∥(k) / P_r(k)',color:'#D55E00',width:1.9},
        {key:'gaussLosRatio',label:'Simplex Gaussian LOS P_∥,Gauss(k) / P_r(k) (FoG turnover)',color:'#CC79A7',dash:'4 2.5',width:1.9},
      ],
      {xlabel:'k [h/Mpc]',ylabel:'P(k) / P_r(k)',xlog:true,ylog:false},
      ratioData
    );
  }else if(rsdViewOpts.specMode==='quadrupole'){
    const quadData={
      k:ks,
      zero:ks.map(()=>0),
      kaiserQ2P0:safeDiv(sp.kaiserQuadrupole,sp.kaiserMonopole),
      sheetQ2P0:safeDiv(sp.rsdSheetQuadrupole,sp.rsdSheetMonopole),
      gaussQ2P0:safeDiv(sp.rsdGaussQuadrupole,sp.rsdGaussMonopole),
    };
    addPlot(
      pSpec,
      'Quadrupole-to-monopole anisotropy ratio P₂(k) / P₀(k)',
      ks,
      [
        {key:'zero',label:'Isotropic (P₂ / P₀ = 0)',color:'#7f8c8d',dash:'2 3',width:1.3},
        {key:'kaiserQ2P0',label:'Linear Kaiser quadrupole ratio P_{2,Kaiser} / P_{0,Kaiser}',color:'#34495e',dash:'5 3',width:1.7},
        {key:'sheetQ2P0',label:'Redshifted sheet P_{2,s}(k) / P_{0,s}(k)',color:'#0072B2',width:2.1},
        {key:'gaussQ2P0',label:'Simplex velocity-gradient Gaussian P_{2,Gauss}(k) / P_{0,Gauss}(k)',color:'#009E73',dash:'7 3',width:2.1},
      ],
      {xlabel:'k [h/Mpc]',ylabel:'P₂(k) / P₀(k)',xlog:true,ylog:false},
      quadData
    );
  }else{
    const wedgeData={
      k:ks,
      unity:ks.map(()=>1),
      kaiserWedge:safeDiv(sp.kaiserLos,sp.realTransverse),
      sheetWedge:safeDiv(sp.rsdSheetLos,sp.rsdSheetTransverse),
      gaussWedge:safeDiv(sp.rsdGaussLos,sp.rsdGaussTransverse),
    };
    addPlot(
      pSpec,
      'Line-of-sight to transverse power ratio P_∥(k) / P_⊥(k)',
      ks,
      [
        {key:'unity',label:'Isotropic real-space baseline (1.0)',color:'#7f8c8d',dash:'2 3',width:1.3},
        {key:'kaiserWedge',label:'Linear Kaiser P_∥ / P_⊥ expectation',color:'#34495e',dash:'5 3',width:1.7},
        {key:'sheetWedge',label:'Redshifted sheet P_∥,s(k) / P_⊥,s(k)',color:'#0072B2',width:2.1},
        {key:'gaussWedge',label:'Simplex Gaussian P_∥,Gauss(k) / P_⊥,Gauss(k)',color:'#009E73',dash:'7 3',width:2.1},
      ],
      {xlabel:'k [h/Mpc]',ylabel:'P_∥(k) / P_⊥(k)',xlog:true,ylog:false},
      wedgeData
    );
  }

  // Panel 4: Direct Simplex Density PDFs & Signed Simplex Jacobian Volume Distribution
  const pPdf=panel(
    'Direct simplex redshift-space density PDFs & signed simplex volume distribution',
    `Evaluating 1/|J_{s,T}| directly on the redshifted simplices without grid deposition captures the full caustic density tail induced by redshift-space shell crossing (J_{s,T} → 0) as well as the expanded void tail (1 + a ∂_∥ v_∥ > 1). The companion chart shows the Lagrangian mass distribution of signed simplex volumes J_T (real space) vs J_{s,T} (redshift space), where the tail at J_{s,T} ≤ 0 measures velocity-induced shell crossing.`
  );
  const rpdf=rsd.pdf||{};
  const rsdPdfBar=$('div','published-controls',pPdf);
  const rsdOrdWrap=$('label','published-select',rsdPdfBar);$('span','',rsdOrdWrap,'PDF ordinate');
  const rsdOrdSel=$('select','lab-select',rsdOrdWrap);
  for(const [v,t] of [
    ['linear','PDF: dP / d(ρ/ρ̄)  (probability density per unit ρ/ρ̄)'],
    ['log10','PDF: dP / d log₁₀(ρ/ρ̄)  (probability density per decade)'],
    ['cdf','CDF: P(≤ ρ/ρ̄)  (cumulative distribution function)'],
    ['peaked_cdf','Peaked CDF: min(P(≤ ρ/ρ̄), P(≥ ρ/ρ̄))  (folded two-sided tail probability)'],
  ]){const o=$('option','',rsdOrdSel,t);o.value=v;}
  rsdOrdSel.value=pdfViewOpts.ordinate;
  rsdOrdSel.addEventListener('change',()=>{pdfViewOpts.ordinate=rsdOrdSel.value;render();});
  const pdfPlotData={};
  const pdfDefs=[];
  const weightKey=S.weight==='mass'?'mass':'volume';
  const isRsdCdf=pdfViewOpts.ordinate==='cdf'||pdfViewOpts.ordinate==='peaked_cdf';
  const rsdPdfBins=pdfViewOpts.bins||(isRsdCdf?160:80);
  for(const [k,label,color,dash] of [
    ['real','Real-space P₀ simplices (1/|J_T|)','#34495e','5 3'],
    ['rsdSheet','Redshifted P₀ simplices (1/|J_{s,T}|)','#D55E00',undefined],
    ['rsdGauss','Simplex velocity-gradient Gaussian field','#009E73','7 3'],
  ]){
    const obj=rpdf[k];
    if(!obj)continue;
    const curve=buildPdfSeriesCurve(obj,weightKey,pdfViewOpts.ordinate,rsdPdfBins,obj.logrho);
    pdfPlotData[`${k}_x`]=curve.x;
    pdfPlotData[`${k}_y`]=curve.y;
    pdfDefs.push({
      key:`${k}_y`,
      xkey:`${k}_x`,
      label,
      color,
      dash,
      width:2.0,
      resample:(nb)=>buildPdfSeriesCurve(obj,weightKey,pdfViewOpts.ordinate,nb,obj.logrho),
    });
  }
  const rsdOrdMeta=pdfOrdinateMeta(pdfViewOpts.ordinate);
  addPlot(
    pPdf,
    `${S.weight==='mass'?'Mass':'Volume'}-weighted density ${rsdOrdMeta.shortKind}: Real-space simplices vs Redshifted simplices vs Velocity-gradient Gaussians`,
    pdfPlotData.rsdSheet_x||[],
    pdfDefs,
    {xlabel:'ρ / ρ̄',ylabel:`${S.weight}-weighted ${rsdOrdMeta.yLabelSuffix}`,xlog:true,ylog:rsdOrdMeta.ylog,ylim:rsdOrdMeta.ylim,binLabel:isRsdCdf?'pts':'bins',bins:rsdPdfBins,minBins:8,maxBins:640,onBinsChange:(nb)=>{pdfViewOpts.bins=nb;}},
    pdfPlotData
  );

  const jac=rpdf.jacobian||{};
  if(arr(jac.j).length){
    const box=$('section','plot-panel',pPdf);
    $('h4','',box,'Signed simplex volume ratio distribution p_M(J): Real space J_T vs Redshift space J_{s,T}');
    const host=plotHost(box),plot=new LinePlot(host,{width:720,height:370,interactive:true});
    plot.setAxes({
      title:'Signed simplex volume ratio distribution p_M(J): Real space J_T vs Redshift space J_{s,T}',
      xlabel:'Signed simplex volume ratio J = det(∂x/∂q) or det(∂s/∂q)  (J ≤ 0 is shell-crossed)',
      ylabel:'Mass probability density dP_M / dJ',
      xlog:false,
      ylog:false,
    });
    plot.setSeries([
      {x:arr(jac.j),y:arr(jac.realMassPdf),label:'Real-space simplex Jacobian J_T',color:'#0072B2',width:2.1},
      {x:arr(jac.j),y:arr(jac.rsdMassPdf),label:'Redshift-space simplex Jacobian J_{s,T} = J_T(1 + a ∂_∥ v_∥)',color:'#D55E00',width:2.1},
    ]);
    plot.setMarkers([{x:0,label:'Shell-crossing threshold J = 0',color:'#c0392b'}]);
    plot.draw();
  }
}
function renderGradient(d) {const p=panel('Initial-amplitude response at fixed epoch','This compares continuous P₀ Fourier power as initial amplitude A changes. The derivative is not taken through clipped or deposited density. The finite-difference check validates the exact sheet-LPT tangent.');addPlot(p,'Amplitude derivative by wavenumber',d.gradient?.k,[{key:'power',label:'P₀ power'},{key:'tangent',label:'Sheet-LPT tangent'},{key:'finiteDifference',label:'Sheet-LPT finite difference'},{key:'pmTangent',label:'PM-sheet tangent'}],{xlabel:'k [h/Mpc]',ylabel:`P₀ and ∂P₀/∂A [(Mpc/h)^${S.dim}]`,xlog:true},d);const err=arr(d.gradient?.relativeError);if(err.length)$('p','panel-note',p,`Maximum sheet-LPT tangent relative error: ${num(err.reduce((m,v)=>Math.max(m,Number(v)||0),0),4)}.`);}
function showDiagnostics(d) {
  if(!d)return;
  showExecution(d.execution||execution);
  const g=d.guard||{},m=d.diagnostics||{},p=$('div','metric-strip',diagnostics),fc=g.firstPmCrossing;
  const percent=v=>Number.isFinite(v)?`${num(100*v,3)}%`:'—';
  for(const [label,val] of [
    ['Output epoch a',num(d.a,6)],
    ['Redshift z',num(d.redshift,5)],
    ['Linear growth D(a)',num(d.D,6)],
    ['Growth D(1)',num(d.growthAtOne,6)],
    ['Crossing reference D',num(d.shellCrossing,6)],
    ['Crossing scale factor',num(d.aShellCrossing,6)],
    ['Minimum LPT vertex Jacobian',num(g.minimumJacobian,4)],
    ['Minimum PM drift Jacobian',num(g.pmMinimumSimplexJacobian,4)],
    ['LPT / PM final inverted mass',`${percent(g.lptFinalInvertedMass)} / ${percent(g.pmFinalInvertedMass)}`],
    ['First PM crossing bracket',fc?`step ${fc.step}: a=${num(fc.aStart,4)}..${num(fc.aEnd,4)} (J=${num(fc.minimumJacobian,4)})`:'none'],
    ['LPT P₀ mass ratio',num(m.mass,8)],
    ['LPT P₁ mass ratio',num(m.p1Mass,8)],
    ['PM P₀ mass ratio',num(m.pmMass,8)],
    ['CIC / TSC mass ratio',`${num(m.cicMass,6)} / ${num(m.tscMass,6)}`],
    ['Position RMS vs PM',`${num(m.positionRms*S.boxSize,4)} Mpc/h`],
    ['Position RMS / displacement RMS',num(m.positionRelativeRms,4)],
    ['Large-scale power vs PM',num(m.powerRelativeError,4)],
  ]) {
    const item=$('div','metric',p);
    $('span','metric-label',item,label);$('strong','metric-value',item,val);
  }
  if(d.diagnostics){$('p','panel-note',diagnostics,`Epoch a=${num(d.a,6)} (z=${num(d.redshift,5)}); D(a)=${num(d.D,6)} and D(1)=${num(d.growthAtOne,6)}. The crossing reference is ${d.crossingReference||'a diagnostic estimate'}, with Dsc=${num(d.shellCrossing,6)}${d.aShellCrossing==null?' (not reached within the growth-table range)':`; a_sc=${num(d.aShellCrossing,6)}`}. PM trajectories start at a_start=${num(d.aStart,4)} from 4LPT and evolve for ${d.request?.steps??S.steps} KDK steps on a ${d.forceMesh??S.forceMesh}^${S.dim} force mesh. Mass ratios compare deposited with initial mass.`);}
}
function showExecution(meta){
  if(!meta)return;
  backendMeta.replaceChildren();
  $('h3','control-subheading',backendMeta,'Run backend and precision');
  const grid=$('div','execution-meta',backendMeta);
  const compact=(label,value)=>{if(value==null||value==='')return;const wide=['Precision','GPU validation','Validation budget','nLPT construction','PM evolution','Exact deposit','Analytic recovery','Cache','PM validation','PM timings','Finite difference','FD validation','FD timings','Timings','Stage backends'].includes(label);const item=$('div',`execution-meta-item${wide?' execution-meta-item-wide':''}`,grid);$('span','metric-label',item,label);$('strong','metric-value',item,String(value));};
  const stageSummary=stage=>[stage.backend,stage.precision||stage.arithmeticPrecision||stage.provenance,stage.reused?'reused from cache':null,stage.fallbackReason?`recovery: ${stage.fallbackReason}`:null].filter(Boolean).join(' · ');
  compact('Requested backend',meta.requestedBackend);
  compact('Executed backend',meta.executedBackend);
  compact('Precision',meta.precision);
  if(meta.lptConstruction)compact('nLPT construction',stageSummary(meta.lptConstruction));
  const pm=meta.pmEvolution;
  if(pm){
    compact('PM evolution',stageSummary(pm));
    if(pm.validation)compact('PM validation',`${pm.validation.status} · ${pm.validation.rows?.length||0} FFT/force/trajectory checks`);
    if(pm.timings){const labels={compileMs:'setup',uploadMs:'upload',kernelMs:'force/KDK',diagnosticMs:'initial orientation',cpuCheckMs:'CPU orientation rechecks',readbackMs:'final readback',totalMs:'total'};compact('PM timings',Object.entries(pm.timings).filter(([,v])=>typeof v==='number'&&Number.isFinite(v)).map(([k,v])=>`${labels[k]||k} ${num(v,3)} ms`).join(' · '));}
  }
  if(meta.exactDeposit)compact('Exact deposit',stageSummary(meta.exactDeposit));
  if(meta.analyticRecovery)compact('Analytic recovery',stageSummary(meta.analyticRecovery));
  const cache=meta.cache;
  if(cache){
    const levels=['initial','pending','pm','observations'];
    const hits=levels.filter(name=>cache[name]?.status==='hit');
    const detail=levels.filter(name=>cache[name]?.status).map(name=>`${name} ${cache[name].status}${cache[name].bytes!=null?` ${(cache[name].bytes/1048576).toFixed(1)} MiB`:''}`).join(' · ');
    const retained=Number.isFinite(cache.bytes)?`${(cache.bytes/1048576).toFixed(1)} / ${(Number(cache.limitBytes||0)/1048576).toFixed(1)} MiB retained`:null;
    compact('Cache',[hits.length?`hit levels: ${hits.join(', ')}`:'no levels reused',detail,retained].filter(Boolean).join(' · '));
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
  const stageLabels={preparationMilliseconds:'preparation',directMilliseconds:device?'direct/setup':'CPU direct',finiteDifferenceMilliseconds:`${fd?.backend==='gpu'?'GPU':'CPU'} finite difference`,cpuRecoveryMilliseconds:'CPU recovery',finalizationMilliseconds:stages.cpuRecoveryMilliseconds>0||meta.analyticRecovery?.backend==='gpu'?'finalization (includes recovery)':'finalization'};
  const stageParts=Object.entries(stages).filter(([,v])=>typeof v==='number'&&Number.isFinite(v)).map(([k,v])=>`${stageLabels[k]||k.replace(/Milliseconds$/,'')} ${num(v,3)} ms`);
  if(typeof timings.kernelMs==='number'&&Number.isFinite(timings.kernelMs))stageParts.unshift(`GPU kernel ${num(timings.kernelMs,3)} ms`);
  if(typeof timings.totalMs==='number'&&Number.isFinite(timings.totalMs))stageParts.unshift(`GPU total ${num(timings.totalMs,3)} ms`);
  if(stageParts.length)compact('Timings',stageParts.join(' · '));

  const backends=meta.stageBackends||{};
  const direct=backends.directPrimal||{};
  const directSummary=['LPT','PM'].map((name)=>{const item=direct[name.toLowerCase()];return item?`${name} ${item.provenance||item.backend||'unknown'}`:null;}).filter(Boolean);
  const precisionSummary=[];
  if(backends.trajectories)precisionSummary.push(`nLPT ${backends.trajectories.lptBackend||'cpu'} fields f32 / growth-state f64 · PM ${backends.trajectories.pmBackend||'cpu'} ${backends.trajectories.pmStatePrecision||'f64'} state`);
  if(backends.exactDeposit)precisionSummary.push(`deposit ${backends.exactDeposit.backend||'cpu'} · ${backends.exactDeposit.arithmeticPrecision||'f64 geometry and mass accumulation'}`);
  if(backends.pdfs)precisionSummary.push('PDF f32 density→f64 stats');
  if(backends.griddedSpectra)precisionSummary.push('FFT f32 field→f64 transform/bins');
  if(backends.finiteDifference)precisionSummary.push(`FD ${backends.finiteDifference.backend} · ${backends.finiteDifference.precision}`);
  if(backends.finalization)precisionSummary.push('finalization f64');
  if(precisionSummary.length||directSummary.length)compact('Stage backends',[...precisionSummary,...directSummary].join(' · '));

  if(Array.isArray(meta.modeProvenance)&&meta.modeProvenance.length){
    const table=$('table','summary-table execution-mode-table',backendMeta),caption=$('caption','',table,'Direct-mode channel precision');
    const head=$('thead','',table),row=$('tr','',head);for(const label of ['Integer mode','LPT'])$('th','',row,label);
    const body=$('tbody','',table);
    for(const mode of meta.modeProvenance){const tr=$('tr','',body),integerMode=Array.isArray(mode.integerMode)?mode.integerMode.join(', '):String(mode.integerMode??'unknown');$('th','',tr,integerMode);$('td','',tr,mode.lpt||'unknown');}
  }

  const details=$('details','published-provenance execution-raw',backendMeta);$('summary','',details,'Full execution metadata');
  const pre=$('pre','execution-raw-json',details);pre.textContent=JSON.stringify(meta,null,2);
}
function renderTheory(){theory.replaceChildren();$('h2','section-heading',theory,'Interpretation and conventions');const text={sheet:'The resolved LPT vertex orientations and simplex Jacobian determinants are monitored throughout the evolution. Beyond the first orientation loss, the displayed density remains a sum of positive streams, while single-stream fluid interpretation and controlled pre-crossing LPT validity no longer apply.',density:'The sheet maps use exact cell integration of the P₀ and P₁ phase-space sheet following LPT trajectories. The CIC and TSC maps treat the same vertices as point particles; their empty cells and discreteness are properties of the estimator, not of the dynamics.',spectra:'The high-k end is where resolution, assignment, and perturbation-order differences become clearest. Compare Lagrangian resolution and observation-grid changes independently.',variance:'By Parseval’s theorem, the variance σ²(R) of a smoothed periodic field equals the sum over discrete Fourier modes weighted by W²(kR). At R = 8 Mpc/h with a real-space Top-Hat window, the unfiltered linear spectrum reproduces σ₈²(D(a)/D(1))² by construction.',pdf:'Mass weighting uses raw unsmoothed cell mass. At R > 0, the density value is smoothed, so the mass and volume PDFs are not related by a simple ρ pV transform.',gradient:'The tangent is the derivative of continuous P₀ Fourier power with respect to initial amplitude A at fixed epoch. It is not a derivative of clipped, deposited density.',published:'The archive panel replays the packaged reference measurements and records their simulation restrictions. It is separate from the live ΛCDM experiment.'};$('p','',theory,text[S.view]||text.sheet);const a=$('a','chapter-link',theory,'Open the chapter ↗');a.href=chapterRefs[S.view]||chapterRefs.sheet;}
async function loadPublished(){
  replayLibrary = await listReplayLibrary();
  if(!publishedData){
    status.textContent='Loading published replay…';
    notice.textContent='Loading packaged reference data.';
    try{
      if(!publishedPromise)publishedPromise=fetch('./data/published-results.json').then((res)=>{if(!res.ok)throw Error(`HTTP ${res.status}`);return res.json();});
      publishedData=await publishedPromise;
    }catch(e){
      publishedPromise=null;
      if(S.view==='published'&&activeReplaySourceId==='packaged'){
        current={publishedError:String(e.message||e)};
        exportJsonButton.disabled=true;
        status.textContent='Published data unavailable';
        renderPublished();
        return;
      }
    }
  }
  if(S.view!=='published')return;
  syncActivePublishedSource();
  exportJsonButton.disabled=false;
  status.textContent='Published data ready';
  renderPublished();
}
function syncActivePublishedSource(){
  if(activeReplaySourceId==='live'&&lastResult){
    const pub = lastResult.publishedReplay || buildPublishedReplayFromResult(lastResult, S);
    lastResult.publishedReplay = pub;
    current = { published: pub, savedRun: { id: 'live', label: 'Current live browser run', state: { ...S }, result: lastResult, run: { durationMilliseconds: runMilliseconds } } };
    return;
  }
  if(activeReplaySourceId!=='packaged'){
    const entry = replayLibrary.find((r)=>r.id===activeReplaySourceId);
    if(entry&&entry.result){
      const pub = entry.result.publishedReplay || buildPublishedReplayFromResult(entry.result, entry.state || S);
      entry.result.publishedReplay = pub;
      current = { published: pub, savedRun: entry };
      return;
    }
    activeReplaySourceId='packaged';
  }
  current={published:publishedData,savedRun:null};
}
function renderPublished(){
  explain();clearHost();renderTheory();
  syncActivePublishedSource();
  const data=current?.published;
  const savedEntry=current?.savedRun||null;

  // Replay Archive & Saved Runs Manager Panel
  const mgr=panel(
    'Replay Archive & Saved Browser Simulations',
    'Replay the packaged 300 Mpc/h DiscoDJ reference archive (List, Hahn & Abel 2026) or load and replay any saved 1D/2D/3D browser simulation from disk (.json) or the browser’s persistent Replay Library. Saved runs can also be hydrated across all 8 Laboratory views instantaneously without recomputing.'
  );
  const mgrBar=$('div','published-controls',mgr);
  const srcWrap=$('label','published-select',mgrBar);
  srcWrap.style.flex='1 1 24rem';
  $('span','',srcWrap,'Active replay dataset');
  const srcSel=$('select','lab-select',srcWrap);
  const addOpt=(val,txt)=>{const o=$('option','',srcSel,txt);o.value=val;};
  addOpt('packaged','Packaged reference archive — List, Hahn & Abel (2026) 300 Mpc/h DiscoDJ');
  if(lastResult&&!lastResult.published){
    addOpt('live',`[Live in memory] ${lastResult.dim||S.dim}D · N=${lastResult.n||S.n}^${lastResult.dim||S.dim} · mesh=${lastResult.mesh||S.mesh}^${lastResult.dim||S.dim} · a=${num(lastResult.a??S.scaleFactor,4)}`);
  }
  for(const item of replayLibrary){
    const dur=item.run?.durationMilliseconds?` · ${(item.run.durationMilliseconds/1000).toFixed(1)}s`:'';
    const ts=item.createdAt?` · ${new Date(item.createdAt).toLocaleTimeString([],{hour:'2-digit',minute:'2-digit'})}`:'';
    addOpt(item.id,`[Saved] ${item.label||item.id}${dur}${ts}`);
  }
  srcSel.value=activeReplaySourceId;
  srcSel.addEventListener('change',()=>{
    activeReplaySourceId=srcSel.value;
    syncActivePublishedSource();
    const d=current?.published;
    if(d){
      if(d.models?.length&&!d.models.some(m=>m.id===S.pubmodel))S.pubmodel=d.models[0].id;
      if(d.estimators?.length&&!d.estimators.some(e=>e.id===S.pubest))S.pubest=d.estimators[0].id;
    }
    renderPublished();
  });

  // Load from disk button inside Published Replay
  const diskLabel=$('label','import-json-button qd-btn',mgrBar,'Load run from disk (.json)');
  diskLabel.style.alignSelf='end';
  const diskInput=$('input','',diskLabel);
  diskInput.type='file';
  diskInput.accept='application/json,.json';
  diskInput.addEventListener('change',async()=>{
    const file=diskInput.files?.[0];
    if(!file)return;
    try{
      const manifest=JSON.parse(await file.text());
      await importExperimentManifest(manifest,file.name,'published');
    }catch(error){
      setNotice(`Could not load replay JSON: ${error?.message||error}`,'error');
    }finally{
      diskInput.value='';
    }
  });

  if(lastResult&&!lastResult.published){
    const saveLiveBtn=$('button','qd-btn',mgrBar,'Save live run to disk & library');
    saveLiveBtn.type='button';
    saveLiveBtn.style.alignSelf='end';
    saveLiveBtn.addEventListener('click',async()=>{
      const prev=current;
      current=lastResult;
      const saved=await exportExperiment(true);
      if(saved){
        activeReplaySourceId=saved.id;
        syncActivePublishedSource();
        renderPublished();
        setNotice(`Saved live simulation (${saved.label}) to disk and to the Replay Library.`,'success');
      }else{
        current=prev;
      }
    });
  }

  if(savedEntry&&savedEntry.result){
    const openAllBtn=$('button','qd-btn',mgrBar,'Inspect across all 8 Laboratory views (instant replay)');
    openAllBtn.type='button';
    openAllBtn.style.alignSelf='end';
    openAllBtn.addEventListener('click',()=>{
      if(savedEntry.state){
        S=normalizeState({...savedEntry.state,view:'sheet'});
        if(requiresExtendedLimits(S)||savedEntry.request?.extended)extendedMode=true;
      }else{
        S.view='sheet';
      }
      if(Array.isArray(savedEntry.request?.estimators)&&savedEntry.request.estimators.length){
        selectedEstimators=savedEntry.request.estimators;
        saveSelectedEstimators();
        syncEstimatorMatrixUI();
      }
      controls.sync();
      const key=physicsKey();
      lastResult=savedEntry.result;
      lastResultKey=key;
      computedFor=key;
      current=savedEntry.result;
      execution=savedEntry.result.execution||savedEntry.execution||null;
      runMilliseconds=savedEntry.run?.durationMilliseconds??null;
      exportJsonButton.disabled=false;
      selectView('sheet',false);
      updateURL();
      status.textContent='Replaying saved run';
      setNotice(`Instant replay active for “${savedEntry.label}”. Switch freely among Phase-space sheet, Density, Power spectra, Variance vs scale, PDFs, Redshift space (RSD), and Differentiability without recomputing.`,'success');
    });

    if(savedEntry.id!=='live'){
      const downloadSavedBtn=$('button','qd-btn',mgrBar,'Download .json to disk');
      downloadSavedBtn.type='button';
      downloadSavedBtn.style.alignSelf='end';
      downloadSavedBtn.addEventListener('click',()=>{
        const blob=new Blob([JSON.stringify(savedEntry)],{type:'application/json'});
        downloadBlob(blob,`precrossedsheet-replay-${savedEntry.id}.json`);
        flash(downloadSavedBtn,'Downloaded');
      });

      const delBtn=$('button','qd-btn',mgrBar,'Delete from library');
      delBtn.type='button';
      delBtn.style.alignSelf='end';
      delBtn.addEventListener('click',async()=>{
        await deleteReplayFromLibrary(savedEntry.id);
        replayLibrary=await listReplayLibrary();
        activeReplaySourceId='packaged';
        syncActivePublishedSource();
        renderPublished();
      });
    }
  }

  if(!data){setNotice(`Published replay is unavailable: ${current?.publishedError||'Loading packaged reference data.'}`,current?.publishedError?'error':'');return;}
  setNotice(
    savedEntry
      ? `Replaying saved simulation: ${savedEntry.label}. Use “Inspect across all 8 Laboratory views” above to explore its 3D sheet, density slices, RSD, and spectra interactively.`
      : 'Published reference replay loaded. Its archive and single-stream restrictions are listed below.',
    'success'
  );
  const p=panel(
    data.title||'Published density replay',
    savedEntry
      ? 'Replaying density PDFs, moments, quantiles, and single-stream diagnostics from the selected saved simulation.'
      : 'Reported density PDFs and moments from the packaged simulation archive. This package contains no published power-spectrum curves.'
  );
  const choose=$('div','published-controls',p);
  const availModels=(data.models||[]).map((m)=>[m.id,m.label]);
  const availEsts=(data.estimators||[]).map((e)=>[e.id,e.label]);
  const radii=(data.smoothing_radii_mpc_h||[7,14,28,42]);
  const activeModel=availModels.some(([id])=>id===S.pubmodel)?S.pubmodel:(availModels[0]?.[0]||S.pubmodel);
  const activeEst=availEsts.some(([id])=>id===S.pubest)?S.pubest:(availEsts[0]?.[0]||S.pubest);
  const activeR=radii.some((r)=>Number(r)===Number(S.pubr))?Number(S.pubr):Number(radii[0]??S.pubr);

  if(savedEntry){
    const addLocalSelect=(label,val,opts,onChange)=>{
      const wrap=$('label','published-select',choose);$('span','',wrap,label);
      const sel=$('select','lab-select',wrap);
      for(const [v,t] of opts){const o=$('option','',sel,t);o.value=String(v);}
      sel.value=String(val);
      sel.addEventListener('change',()=>onChange(sel.value));
    };
    addLocalSelect('Model',activeModel,availModels,(v)=>{S.pubmodel=v;renderPublished();});
    addLocalSelect('Estimator',activeEst,availEsts,(v)=>{S.pubest=v;renderPublished();});
    addLocalSelect('Smoothing radius R',activeR,radii.map((r)=>[r,`${r} Mpc/h`]),()=>renderPublished());
    if(publishedData){
      const ovWrap=$('label','published-select',choose);
      $('span','',ovWrap,'Packaged DiscoDJ reference overlay');
      const ovSel=$('select','lab-select',ovWrap);
      for(const [v,t] of [['1','Overlay packaged 300 Mpc/h DiscoDJ archive'],['0','Hide packaged reference overlay']]){const o=$('option','',ovSel,t);o.value=v;}
      ovSel.value=overlayPackagedInReplay?'1':'0';
      ovSel.addEventListener('change',()=>{overlayPackagedInReplay=ovSel.value==='1';renderPublished();});
    }
  }else{
    addPublishedSelect(choose,'Model','pubmodel',availModels);
    addPublishedSelect(choose,'Estimator','pubest',availEsts);
    addPublishedSelect(choose,'Smoothing radius R','pubr',radii.map((r)=>[r,`${r} Mpc/h`]));
    addPublishedSelect(choose,'Snapshot pairing','paired',[[true,'Phase-shifted companion pair'],[false,'Unpaired run']]);
  }

  const model=activeModel,est=activeEst,R=activeR;
  const runRec=(m,e,paired)=>data.runs?.find((r)=>r.model===m&&r.estimator===e&&(paired===undefined||r.paired===paired)&&r.radii?.some((x)=>Number(x.R)===R));
  const selectedRun=runRec(model,est,savedEntry?undefined:S.paired),selectedRadius=selectedRun?.radii?.find((x)=>Number(x.R)===R);
  if(!selectedRun||!selectedRadius){$('p','panel-note',p,'No packaged run matches this model, estimator, and smoothing radius.');return;}
  const weight=S.weight,otherModel=model==='nbody'?'4lpt':'nbody',otherRun=runRec(otherModel,est,selectedRun.paired),otherRadius=otherRun?.radii?.find((x)=>Number(x.R)===R);
  const estimatorOther=est==='sheet'?'cic':'sheet',pairedRun=selectedRun.paired?runRec(model,estimatorOther,true):null,pairedRadius=pairedRun?.radii?.find((x)=>Number(x.R)===R);
  const bins=arr(data.bin_edges_log10rho),centersLog=bins.slice(0,-1).map((x,i)=>(Number(x)+Number(bins[i+1]))/2),centersRho=centersLog.map((x)=>Math.pow(10,x)),ln10=Math.LN10;
  const toLinearPdf=(arrLog)=>arr(arrLog).map((v,i)=>{const r=centersRho[i];return r>0?Number(v)/(ln10*r):0;});
  const curves={selected:toLinearPdf(selectedRadius[weight]?.pdf||[])};
  const definitions=[{key:'selected',label:`${model.toUpperCase()} ${est} · ${weight}`,color:'#0072B2',width:2.2}];
  if(otherRadius){curves.comparison=toLinearPdf(otherRadius[weight]?.pdf||[]);definitions.push({key:'comparison',label:`${otherModel.toUpperCase()} ${est} · ${weight}`,color:'#D55E00',width:2.0});}
  if(pairedRadius){curves.paired=toLinearPdf(pairedRadius[weight]?.pdf||[]);definitions.push({key:'paired',label:`${model.toUpperCase()} ${estimatorOther} · ${weight}`,color:'#009E73',dash:'6 3',width:1.9});}

  // Optional overlay of packaged DiscoDJ reference when viewing a saved run
  if(savedEntry&&overlayPackagedInReplay&&publishedData){
    const pkgRadii=publishedData.smoothing_radii_mpc_h||[7,14,28,42];
    const nearestPkgR=pkgRadii.reduce((best,cand)=>Math.abs(cand-R)<Math.abs(best-R)?cand:best,pkgRadii[0]||14);
    const pkgEst=['sheet','cic'].includes(est)?est:'sheet';
    const pkgModel=['1lpt','2lpt','3lpt','4lpt','nbody'].includes(model)?model:'4lpt';
    const pkgRun=publishedData.runs?.find((r)=>r.model===pkgModel&&r.estimator===pkgEst&&r.paired===true&&r.radii?.some((x)=>Number(x.R)===nearestPkgR));
    const pkgRad=pkgRun?.radii?.find((x)=>Number(x.R)===nearestPkgR);
    if(pkgRad){
      const pkgBins=arr(publishedData.bin_edges_log10rho);
      const pkgCentersRho=pkgBins.slice(0,-1).map((x,i)=>Math.pow(10,(Number(x)+Number(pkgBins[i+1]))/2));
      curves.pkgX=pkgCentersRho;
      curves.pkgY=arr(pkgRad[weight]?.pdf||[]).map((v,i)=>{const r=pkgCentersRho[i];return r>0?Number(v)/(ln10*r):0;});
      definitions.push({key:'pkgY',xkey:'pkgX',label:`Packaged DiscoDJ ${pkgModel.toUpperCase()} ${pkgEst} (R=${nearestPkgR} Mpc/h)`,color:'#34495e',dash:'3 3',width:1.7});
    }
  }

  addPlot(p,'Published density PDF',centersRho,definitions,{xlabel:'ρ / ρ̄',ylabel:'dP / d(ρ / ρ̄)',xlog:true,ylog:true},curves);
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

  // If replaying a saved browser simulation, also render its saved Power Spectra, Variance vs Scale, and Diagnostics right inside Published Replay!
  if(savedEntry?.result){
    const sr=savedEntry.result;
    if(sr.spectra&&arr(sr.spectra.k).length){
      renderSpectra(sr);
    }
    showDiagnostics(sr);
  }
}
function addPublishedSelect(parent,label,key,options){const wrap=$('label','published-select',parent);$('span','',wrap,label);const sel=$('select','lab-select',wrap);sel.setAttribute('aria-label',label);for(const [value,text] of options){const o=$('option','',sel,text);o.value=String(value);}sel.value=String(S[key]);sel.addEventListener('change',()=>{S[key]=key==='pubr'?Number(sel.value):key==='paired'?sel.value==='true':sel.value;S=normalizeState(S);updateURL();renderPublished();});}
function addMetric(parent,label,value){const item=$('div','metric',parent);$('span','metric-label',item,label);$('strong','metric-value',item,value);}

function setDefaultView(){selectView(S.view, false);}
controls.sync(); setDefaultView(); updateURL(); run();
window.addEventListener('hashchange',()=>{const version=fragmentVersion(location.hash);unsupportedSchema=version!=null&&!(['2',String(STATE_VERSION)].includes(version));if(unsupportedSchema){if(!document.querySelector('.unsupported-link-note'))$('p','unsupported-link-note',intro,`This URL uses unsupported experiment schema v${version}. The experiment has not been run. Select a preset or edit a control to start a new version-${STATE_VERSION} run; the saved values cannot be safely interpreted.`);S=decodeHash('');controls.sync();run();return;}clearUnsupportedSchema();const next=decodeHashWithPreset(location.hash);if(encodeHash(next)===encodeHash(S))return;S=next;controls.sync();selectView(S.view,false);run();});
window.labExperimentState = () => ({...S});
