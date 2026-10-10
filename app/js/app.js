import { decodeHash, encodeHash, makeDefaults, normalizeState } from './state.js';
import { PRESETS, presetState } from './presets.js';
import { LinePlot } from './viz/plot.js';
import { copyText, downloadBlob, flash } from './viz/download.js';

let S = decodeHash(location.hash), worker = null, timer = 0, requestId = 0, current = null, computedFor = '', activeKey = '', publishedPromise = null, publishedData = null, lastResult = null, lastResultKey = '';
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
intro.innerHTML = '<p class="eyebrow">A Lagrangian view of structure formation</p><p>Track how a smooth sheet of matter moves through position–velocity space. The calculation stops at the first simplex inversion, where the single-stream description ceases to apply.</p>';
const presets = $('div', 'lab-presets', root);
$('span', 'control-caption', presets, 'Start with');
const presetSelect = $('select', 'lab-select', presets); presetSelect.setAttribute('aria-label', 'Choose an experiment preset');
const customOption = $('option', '', presetSelect, 'Current settings'); customOption.value = '';
customOption.disabled = true;
for (const p of PRESETS) { const o = $('option', '', presetSelect, p.label); o.value = p.id; }
const presetNote = $('span', 'preset-note', presets, PRESETS[0].note);
presetSelect.value = '';
presetNote.textContent = 'Adjust the controls or choose a guided experiment.';
presetSelect.addEventListener('change', () => { S = normalizeState(presetState(presetSelect.value)); controls.sync(); selectView(S.view, false); updateURL(); run(); presetNote.textContent = PRESETS.find((p) => p.id === presetSelect.value)?.note || ''; });

const layout = $('div', 'lab-layout', root), sidebar = $('aside', 'lab-sidebar', layout), main = $('section', 'lab-main', layout);
$('h2', 'section-heading', sidebar, 'Experiment controls');
const controlWrap = $('div', 'control-list', sidebar), controlHandles = {};
function selectControl(key, label, opts, help) {
  const wrap = $('label', 'control-field', controlWrap); $('span', 'control-name', wrap, label);
  const sel = $('select', 'lab-select', wrap); sel.setAttribute('aria-label', label); for (const [v, text] of opts) { const o = $('option', '', sel, text); o.value = v; }
  if (help) $('small', 'control-help', wrap, help);
  sel.addEventListener('change', () => { S[key] = ['dim','n','order','seed','steps','mesh','forceMesh'].includes(key) ? Number(sel.value) : sel.value; S = normalizeState(S); controls.sync(); changed(); });
  controlHandles[key] = { set: (v) => { sel.value = String(v); }, el: sel };
}
function rangeControl(key, label, min, max, step, help, format = (v) => String(+Number(v).toPrecision(3))) {
  const wrap = $('label', 'control-field', controlWrap); const top = $('span', 'control-top', wrap); $('span', 'control-name', top, label); const out = $('output', 'control-value', top); const input = $('input', '', wrap); input.type = 'range'; input.min = min; input.max = max; input.step = step; input.setAttribute('aria-label', label);
  if (help) $('small', 'control-help', wrap, help);
  const set = (v) => { input.value = String(v); out.value = format(Number(v)); out.textContent = format(Number(v)); };
  set(S[key]); input.addEventListener('input', () => { S[key] = Number(input.value); out.value = format(S[key]); out.textContent = format(S[key]); changed(false); }); input.addEventListener('change', () => changed(true));
  controlHandles[key] = { set, el: input };
}
selectControl('dim', 'Spatial dimension', [['1','1D curve'],['2','2D sheet'],['3','3D volume']], 'The rendered geometry changes with the number of spatial dimensions.');
selectControl('n', 'Lattice resolution', [[16,'16 per axis'],[24,'24 per axis'],[32,'32 per axis']], '3D runs are limited to 24 points per axis.');
rangeControl('fraction', 'Growth fraction', .1, .85, .01, 'Move toward the first shell crossing; the guard ends an invalid run.');
selectControl('order', 'LPT order', [[1,'1st order'],[2,'2nd order'],[3,'3rd order'],[4,'4th order']], 'Compare smoothness and accuracy as the perturbative order changes.');
selectControl('ic', 'Initial conditions', [['waves','Deterministic waves'],['gaussian','Gaussian field']], 'Gaussian fields are low-band filtered teaching initial conditions.');
rangeControl('smoothing', 'Smoothing scale', 0, .15, .005, 'Box units. This is an observation scale applied to the measured field.');
selectControl('steps', 'Evolution steps', [[32,'32'],[64,'64'],[128,'128']]);
selectControl('mesh', 'Observation mesh', [[32,'32 cells'],[48,'48 cells'],[64,'64 cells']], 'Controls the grid used to measure density and PDFs. 3D runs use 32 cells per axis.');
selectControl('forceMesh', 'Force mesh', [[16,'16 cells'],[24,'24 cells'],[32,'32 cells'],[48,'48 cells'],[64,'64 cells']], 'Start with a force mesh matching the particle lattice. Refine both together: a finer force grid alone can increase particle-load artifacts. 3D caps at 32 cells per axis.');
rangeControl('seed', 'Random seed', 0, 9999, 1, 'Used only by Gaussian initial conditions.', (v) => String(Math.round(v)));
const controls = { sync() { for (const [k, h] of Object.entries(controlHandles)) h.set(S[k]); } };

const chapterLink = $('a', 'chapter-link', sidebar, 'Read the chapter behind this view ↗'); chapterLink.href = chapterRefs[S.view];
const notice = $('div', 'lab-notice', main); notice.setAttribute('role', 'status'); notice.setAttribute('aria-live', 'polite');
const tabs = $('nav', 'view-tabs', main); tabs.setAttribute('aria-label', 'Choose an experiment view');
const viewButtons = {};
for (const [key, label] of VIEWS) { const b = $('button', 'view-tab', tabs, label); b.type = 'button'; b.setAttribute('aria-pressed', 'false'); b.addEventListener('click', () => selectView(key)); viewButtons[key] = b; }
const viewHelp = $('p', 'view-help', main);
const viewHost = $('div', 'view-host', main);
const diagnostics = $('div', 'diagnostics', main);
const theory = $('aside', 'lab-theory', root);

function selectView(view, doRun = true) { const changedView = S.view !== view; S.view = view; for (const [k, b] of Object.entries(viewButtons)) b.setAttribute('aria-pressed', String(k === view)); chapterLink.href = chapterRefs[view] || chapterRefs.sheet; updateURL(); if (changedView && doRun) run(); else render(); }
function updateURL() { history.replaceState(null, '', '#' + encodeHash(S)); }
function changed(immediate = false) { S = normalizeState(S); controls.sync(); presetSelect.value=''; presetNote.textContent='Custom experiment settings.'; updateURL(); selectView(S.view, false); if (timer) clearTimeout(timer); if (immediate) run(); else { status.textContent = 'Waiting for input…'; timer = setTimeout(run, 180); } }
function physicsKey() { const { view, rotate, tilt, pubmodel, pubest, pubr, ...physics } = S; return JSON.stringify(physics); }
function run() {
  if (timer) clearTimeout(timer); timer = 0;
  if (S.view === 'published') { if (worker) worker.terminate(); worker = null; activeKey=''; current = null; loadPublished(); return; }
  const key = physicsKey();
  if (worker && activeKey !== key) { worker.terminate(); worker = null; activeKey=''; }
  if (lastResultKey === key && lastResult) { current=lastResult; computedFor=key; status.textContent=current?.guard?.ok===false?'Stopped at shell crossing':'Ready'; render(); return; }
  if (computedFor === key && current && !current.published) { status.textContent=current?.guard?.ok===false?'Stopped at shell crossing':'Ready';render(); return; }
  if (worker && activeKey === key) { status.textContent = 'Computing…'; render(); return; }
  if (worker) worker.terminate();
  const id = ++requestId; worker = new Worker('./js/worker.js', { type: 'module' }); activeKey = key;
  status.textContent = 'Computing…'; notice.textContent = 'Running the guarded Lagrangian calculation in a background worker.';
  worker.onmessage = ({ data }) => { if (data.requestId !== id) return; worker.terminate(); worker = null; activeKey=''; if (data.error) { current = { error: data.error }; computedFor=''; status.textContent = 'Run failed'; } else { current = data.result; lastResult=current;lastResultKey=key;computedFor=key; status.textContent = current?.guard?.ok === false ? 'Stopped at shell crossing' : current?.error ? 'Run failed' : 'Ready'; } render(); };
  worker.onerror = (event) => { worker?.terminate(); worker = null; activeKey=''; computedFor=''; current = { error: event.message || 'Could not start the numerical worker.' }; status.textContent = 'Run failed'; render(); };
  const { pubmodel, pubest, pubr, ...coreState } = S;
  worker.postMessage({ type: 'run', requestId: id, state: coreState });
}

function explain() {
  const messages = {
    sheet: S.dim === 1
      ? 'Each Lagrangian label q maps to position x and velocity dx/da. The phase-space curve is shown only while its neighboring elements preserve orientation.'
      : S.dim === 2
        ? 'Each Lagrangian label q maps to a position (x₁,x₂). The triangulated plane shows the deformed sheet in position space; the velocity coordinates are not projected into this view.'
        : 'Each Lagrangian label q maps to a position (x₁,x₂,x₃). This wireframe projects the three-dimensional position sheet from the full six-dimensional (x,v) phase-space sheet.',
    density: 'Both maps integrate the exact P₀ sheet into cells: once along LPT trajectories and once along PM trajectories. The observation mesh and force mesh are separate; CIC particle assignment is compared in the spectrum view.',
    spectra: 'Compare the sheet-derived power with CIC, particle-mesh, and direct estimates. The direct curve is a reference calculation for this browser model.',
    pdf: 'Volume weighting samples space uniformly. Mass weighting uses unsmoothed raw density. At nonzero observation scale R, the volume PDF uses smoothed density, so ρ pV is not a simple transform between the two.',
    gradient: 'Compare the derivative of continuous P₀ Fourier power with respect to initial amplitude A at fixed epoch against a finite difference. This does not differentiate clipped or deposited density.',
    published: 'This panel replays packaged reference data. These curves are published-run results and are distinct from the simplified Einstein–de Sitter browser experiment.',
  };
  viewHelp.textContent = messages[S.view];
}
function setNotice(text, kind = '') { notice.textContent = text; notice.className = `lab-notice ${kind}`; }
function num(v, digits = 3) { return Number.isFinite(Number(v)) ? Number(v).toPrecision(digits) : '—'; }
function arr(a) { return ArrayBuffer.isView(a) ? Array.from(a) : Array.isArray(a) ? a : []; }
function clearHost() { viewHost.replaceChildren(); diagnostics.replaceChildren(); }
function panel(title, description = '') { const p = $('section', 'result-panel', viewHost); $('h3', '', p, title); if (description) $('p', 'panel-note', p, description); return p; }
function addPngButton(canvas, name) { const b = $('button', 'export-button', canvas.parentElement, 'Save PNG'); b.type = 'button'; b.setAttribute('aria-label', `Save ${name} image as PNG`); b.addEventListener('click', () => canvas.toBlob((blob) => { if (blob) { downloadBlob(blob, `${name}.png`); flash(b, 'Saved'); } })); }
function canvasFor(parent, label, wide = false) { const c = $('canvas', `lab-canvas${wide ? ' canvas-wide' : ''}`, parent); c.width = 960; c.height = wide ? 560 : 680; c.setAttribute('role', 'img'); c.setAttribute('aria-label', label); return c; }
function paintCanvas(c, draw) { const ctx = c.getContext('2d'); ctx.clearRect(0, 0, c.width, c.height); ctx.fillStyle = getComputedStyle(c).backgroundColor; ctx.fillRect(0,0,c.width,c.height); draw(ctx,c.width,c.height); }
function guarded() {
  if (current?.error || current?.errorMessage) { setNotice(`The numerical worker could not complete this run: ${current.error || current.errorMessage}`, 'error'); return false; }
  if (!current) { setNotice('Waiting for the calculation to finish.'); return false; }
  if (current.guard?.ok === false) { const why = current.guard.reason || 'A simplex changed orientation.'; setNotice(`Stopped before drawing scientific results: ${why} Minimum Jacobian: ${num(current.guard.minimumJacobian,4)}. Lower the growth fraction to return to the guarded regime.`, 'guard'); return false; }
  if (current.error) { setNotice(current.error, 'error'); return false; }
  setNotice('Guard passed: resolved LPT orientations and all PM drift orientations stayed positive.', 'success'); return true;
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
  const dimension=Number(d.dim||S.dim),caption=dimension===1?'A curve in position–velocity space: x horizontally and dx/da vertically.':dimension===2?'The triangulated sheet in the x₁–x₂ position plane.': 'A position projection of the 3D Lagrangian sheet embedded in six-dimensional (x,v) phase space. Drag to rotate the projection.';
  const p=panel(dimension===1?'One-dimensional phase-space curve':dimension===2?'Two-dimensional triangulated position sheet':'Three-dimensional tetrahedral position sheet', `Color follows the initial Lagrangian labels q. ${caption}`);
  const alt=dimension===1?'One-dimensional phase-space curve with position x and velocity dx/da':dimension===2?'Triangulated two-dimensional Lagrangian sheet in the x₁–x₂ position plane':'Rotatable wireframe position projection of the three-dimensional Lagrangian sheet embedded in six-dimensional phase space';
  const c=canvasFor(p,alt,true);
  const draw=()=>paintCanvas(c,(ctx,W,H)=>{
    const n=Number(d.n||S.n),dim=Number(d.dim||S.dim),q=arr(d.q),x=arr(d.x),v=arr(d.v); const nv=Math.floor(Math.min(q.length,x.length,v.length)/(dim||1));
    if (!nv) { ctx.fillStyle=getComputedStyle(c).color; ctx.fillText('Sheet coordinates were not returned.',32,40); return; }
    const get=(a,i,k)=>a[i*dim+k]??0;
    const pts=[]; for(let i=0;i<nv;i++) { let px,py; if(dim===1){px=get(x,i,0);py=get(v,i,0);} else if(dim===2){px=get(x,i,0);py=get(x,i,1);} else [px,py]=project3(get(x,i,0),get(x,i,1),get(x,i,2),S.rotate,S.tilt,1,0,0); pts.push([px,py]); }
    let minx=Infinity,maxx=-Infinity,miny=Infinity,maxy=-Infinity; for(const pt of pts){minx=Math.min(minx,pt[0]);maxx=Math.max(maxx,pt[0]);miny=Math.min(miny,pt[1]);maxy=Math.max(maxy,pt[1]);} if(minx===maxx){minx-=1;maxx+=1;} if(miny===maxy){miny-=1;maxy+=1;}
    const pad=55,sc=Math.min((W-2*pad)/(maxx-minx),(H-2*pad)/(maxy-miny)); const xy=dim===1?pts.map(([a,b])=>[pad+(a-minx)*(W-2*pad)/(maxx-minx),H-pad-(b-miny)*(H-2*pad)/(maxy-miny)]):pts.map(([a,b])=>[pad+(a-minx)*sc,H-pad-(b-miny)*sc]);
    ctx.strokeStyle='#8796a5';ctx.lineWidth=dim===3?.55:1;ctx.globalAlpha=dim===3?.17:.65;
    const grid=n; const idx=(i,j,k=0)=>i+grid*(j+grid*k);
    const edge=(a,b)=>{if(a<nv&&b<nv){ctx.beginPath();ctx.moveTo(...xy[a]);ctx.lineTo(...xy[b]);ctx.stroke();}};
    if(dim===1){ctx.beginPath();xy.forEach((pt,i)=>i?ctx.lineTo(...pt):ctx.moveTo(...pt));ctx.stroke();}
    else if(dim===2){for(let j=0;j<grid;j++)for(let i=0;i<grid;i++){const a=idx(i,j);if(i+1<grid)edge(a,idx(i+1,j));if(j+1<grid)edge(a,idx(i,j+1));if(i+1<grid&&j+1<grid){const b=idx(i+1,j),cc=idx(i,j+1),e=idx(i+1,j+1);edge(b,e);edge(cc,e);edge(a,e);}}}
    else {const step=Math.max(1,Math.ceil(grid/10));for(let k=0;k<grid;k+=step)for(let j=0;j<grid;j+=step)for(let i=0;i<grid;i+=step){const a=idx(i,j,k);if(i+step<grid)edge(a,idx(i+step,j,k));if(j+step<grid)edge(a,idx(i,j+step,k));if(k+step<grid)edge(a,idx(i,j,k+step));if(i+step<grid&&j+step<grid&&k+step<grid){const b=idx(i+step,j,k),cc=idx(i,j+step,k),dd=idx(i,j,k+step),e=idx(i+step,j+step,k),f=idx(i+step,j,k+step),g=idx(i,j+step,k+step),h=idx(i+step,j+step,k+step);edge(a,e);edge(a,f);edge(a,g);edge(b,e);edge(cc,e);edge(b,f);edge(dd,f);edge(cc,g);edge(dd,g);edge(e,h);edge(f,h);edge(g,h);}}}
    ctx.globalAlpha=1;const skip=dim===3?Math.max(1,Math.ceil(xy.length/850)):Math.max(1,Math.floor(xy.length/5000));for(let i=0;i<xy.length;i+=skip){const t=i/Math.max(1,xy.length-1);ctx.fillStyle=`hsl(${210-195*t} 76% 42%)`;ctx.beginPath();ctx.arc(xy[i][0],xy[i][1],2.2,0,Math.PI*2);ctx.fill();}
    ctx.fillStyle=getComputedStyle(c).color;ctx.font='15px system-ui';ctx.fillText(dim===1?'position x':`position x₁${dim===2?'':' / x₂ projection'}`,W/2-35,H-13);ctx.save();ctx.translate(18,H/2+28);ctx.rotate(-Math.PI/2);ctx.fillText(dim===1?'velocity dx/da':dim===2?'position x₂':'projected x₃',0,0);ctx.restore();
  }); draw(); addPngButton(c,'phase-space-sheet');
  if(S.dim===3){const controlsBox=$('div','rotation-controls',viewHost);$('span','control-caption',controlsBox,'Rotate 3D projection'); for(const [key,label,min,max] of [['rotate','Angle',-180,180],['tilt','Tilt',-80,80]]){const l=$('label','rotation-range',controlsBox);$('span','',l,label);const r=$('input','',l);r.type='range';r.min=min;r.max=max;r.value=S[key];r.setAttribute('aria-label',`${label} sheet angle`);r.addEventListener('input',()=>{S[key]=Number(r.value);updateURL();draw();});}}
}
function renderDensity(d) {
  const p=panel('Exact-deposit and PM-trajectory sheet density', 'Maps share a color scale. The exact map is cell-average density integrated from the sheet; the PM map evaluates the same P₀ sheet estimator along PM trajectories. CIC assignment appears separately in the spectrum view. For 3D, both show the central first-axis slice.');
  const maps=$('div','density-pair',p), dim=Number(d.dim||S.dim), mesh=Number(d.mesh||S.mesh);
  const exact=arr(d.rho),pm=arr(d.pmRho),all=[...exact,...pm],mean=all.length?all.reduce((m,v)=>m+(Number(v)||0),0)/all.length:1,sharedMax=all.reduce((m,v)=>Math.max(m,Number(v)||0),0)/(mean||1);
  drawDensityMap(maps,'Exact sheet deposit',exact,'Exact-deposit cell-average density map',dim,mesh,'exact-deposit-density',sharedMax,mean);
  drawDensityMap(maps,'P₀ sheet along PM trajectories',pm,'P zero sheet density sampled along particle-mesh trajectories',dim,mesh,'pm-trajectory-sheet-density',sharedMax,mean);
}
function drawDensityMap(parent,title,values,label,dim,mesh,name,scaleMax,mean){
  const block=$('div','density-map',parent);$('h4','',block,title);const c=canvasFor(block,label,true),rho=arr(values);
  paintCanvas(c,(ctx,W,H)=>{
    const ink=getComputedStyle(c).color;ctx.fillStyle=ink;
    if(!rho.length){ctx.fillText('Density cells were not returned.',32,42);return;}
    const n=dim===1?rho.length:Math.min(mesh,Math.floor(Math.sqrt(rho.length))),max=scaleMax||rho.reduce((a,b)=>Math.max(a,Number(b)||0),0)/(mean||1),left=48,right=W-40,top=24,bottom=H-98;
    if(dim===1){for(let i=0;i<rho.length;i++){const h=Math.max(0,rho[i]/((mean||1)*max))*(bottom-top);ctx.fillStyle=densityColor(rho[i]/((mean||1)*max));ctx.fillRect(left+i*(right-left)/rho.length,bottom-h,Math.max(1,(right-left)/rho.length),h);}}
    else {const stride=dim===3?mesh*mesh*Math.floor(mesh/2):0;for(let j=0;j<n;j++)for(let i=0;i<n;i++){const val=rho[(dim===3?stride:0)+j*n+i]??0;ctx.fillStyle=densityColor(val/((mean||1)*max));ctx.fillRect(left+i*(right-left)/n,top+(n-1-j)*(bottom-top)/n,(right-left)/n+1,(bottom-top)/n+1);}}
    ctx.fillStyle=ink;ctx.font='12px system-ui';
    const xt=[['0',left],['0.5',(left+right)/2],['1',right]];for(const [v,x] of xt){ctx.fillText(v,x-4,bottom+18);ctx.beginPath();ctx.moveTo(x,bottom);ctx.lineTo(x,bottom+4);ctx.strokeStyle=ink;ctx.stroke();}
    for(const [v,y] of [['0',bottom],['0.5',(top+bottom)/2],['1',top]])ctx.fillText(v,18,y+4);
    ctx.fillText(dim===1?'x/L':`x₁/L`,W/2-10,bottom+35);ctx.save();ctx.translate(14,(top+bottom)/2);ctx.rotate(-Math.PI/2);ctx.fillText(dim===1?'ρ':'x₂/L',0,0);ctx.restore();
    const gx=left+80,gy=H-32,gw=Math.max(80,right-left-145,80),gh=10,grad=ctx.createLinearGradient(gx,0,gx+gw,0);for(let i=0;i<=20;i++)grad.addColorStop(i/20,densityColor(i/20));ctx.fillStyle=grad;ctx.fillRect(gx,gy,gw,gh);ctx.fillStyle=ink;ctx.fillText('ρ / mean',left,gy+9);ctx.fillText('0',gx-12,gy-5);ctx.fillText(num(max,3),gx+gw+4,gy-5);
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
function renderSpectra(d) {const p=panel('Power spectrum comparison','The first chart compares isotropically binned FFT estimates. The second compares direct first-axis modes on their own deterministic k grid.');addPlot(p,'Isotropic FFT-bin power',d.spectra?.k,[{key:'sheet',label:'Exact sheet'},{key:'cic',label:'CIC particles'},{key:'pm',label:'PM trajectories'}],{xlabel:'wavenumber k',ylabel:'P(k)',xlog:true,ylog:true},d);addPlot(p,'Direct first-axis modes',d.spectra?.directK,[{key:'direct',label:'Direct sheet LPT'},{key:'directPM',label:'Direct PM'}],{xlabel:'first-axis wavenumber k',ylabel:'P(k)',xlog:true,ylog:true},d);}
function renderPdfs(d) {const p=panel('Density probability distributions','Both weighting conventions are shown. The selected convention is marked in its chart title.');$('p','panel-note',p,`Selected weighting: ${S.weight}. Mass weights use unsmoothed raw density; volume weighting uses the density at the selected observation scale.`);addPlot(p,`Volume-weighted PDF${S.weight==='volume'?' · selected':''}`,d.pdf?.logrho,[{key:'volume',label:'Sheet volume'},{key:'pmVolume',label:'PM volume'}],{xlabel:'log₁₀ ρ',ylabel:'probability density',ylim:[0,undefined]},d);addPlot(p,`Mass-weighted PDF${S.weight==='mass'?' · selected':''}`,d.pdf?.logrho,[{key:'mass',label:'Sheet mass'},{key:'pmMass',label:'PM mass'}],{xlabel:'log₁₀ ρ',ylabel:'mass probability',ylim:[0,undefined]},d);}
function renderGradient(d) {const p=panel('Initial-amplitude response at fixed epoch','This compares continuous P₀ Fourier power as initial amplitude A changes. The derivative is not taken through clipped or deposited density. The finite-difference check validates the sheet-LPT tangent; the PM tangent is shown as its own response.');addPlot(p,'Amplitude derivative by wavenumber',d.gradient?.k,[{key:'power',label:'P₀ power'},{key:'tangent',label:'Sheet-LPT tangent'},{key:'finiteDifference',label:'Sheet-LPT finite difference'},{key:'pmTangent',label:'PM tangent'}],{xlabel:'wavenumber k',ylabel:'P₀ response',xlog:true},d);const err=arr(d.gradient?.relativeError);if(err.length)$('p','panel-note',p,`Maximum sheet-LPT tangent relative error: ${num(err.reduce((m,v)=>Math.max(m,Number(v)||0),0),4)}.`);}
function showDiagnostics(d) {
  if(!d)return;
  const g=d.guard||{},m=d.diagnostics||{},p=$('div','metric-strip',diagnostics);
  const percent=v=>Number.isFinite(v)?`${num(100*v,3)}%`:'—';
  for(const [label,val] of [
    ['Minimum Jacobian',num(g.minimumJacobian,4)],
    ['Simplex Jacobian',num(g.minimumSimplexJacobian,4)],
    ['LPT mass ratio',num(m.mass,8)],
    ['PM mass ratio',num(m.pmMass,8)],
    ['Position RMS (box units)',num(m.positionRms,4)],
    ['Relative position RMS',percent(m.positionRelativeRms)],
    ['Max low-bin power residual',percent(m.powerRelativeError)],
  ]) {
    const item=$('div','metric',p);
    $('span','metric-label',item,label);$('strong','metric-value',item,val);
  }
  if(d.diagnostics)$('p','panel-note',diagnostics,'Mass ratios compare deposited with initial mass. Relative position RMS is normalized by the LPT displacement RMS. The power residual is max |P_LPT/P_PM − 1| over the first four populated FFT bins; weak-power bins can give large ratios. This finite-resolution PM run requires a convergence check.');
}
function renderTheory(){theory.replaceChildren();$('h2','section-heading',theory,'What to notice');const text={sheet:'Before shell crossing, the Lagrangian map is one-to-one. A positive Jacobian means neighboring mass elements have not flipped orientation.',density:'Both maps use exact cell integration of the P₀ sheet: one follows LPT trajectories and the other follows PM trajectories. CIC particle assignment is shown separately in the spectrum view.',spectra:'The high-k end is where resolution, assignment, and perturbation order differences become clearest.',pdf:'Mass weighting uses unsmoothed raw density. At R > 0, volume weighting uses smoothed density, so ρ pV is not a simple transform between these PDFs.',gradient:'The tangent is the derivative of continuous P₀ Fourier power with respect to initial amplitude A at fixed epoch. It is not a derivative of clipped, deposited density.',published:'The replayed run provides the stated publication reference. The browser calculation uses simplified EdS growth and should not be read as a new ΛCDM inference.'};$('p','',theory,text[S.view]||text.sheet);const a=$('a','chapter-link',theory,'Open the chapter ↗');a.href=chapterRefs[S.view]||chapterRefs.sheet;}
async function loadPublished(){if(publishedData){current={published:publishedData};renderPublished();return;}status.textContent='Loading published replay…';notice.textContent='Loading packaged reference data.';try{if(!publishedPromise)publishedPromise=fetch('./data/published-results.json').then((res)=>{if(!res.ok)throw Error(`HTTP ${res.status}`);return res.json();});const data=await publishedPromise;if(S.view!=='published')return;publishedData=data;current={published:data};status.textContent='Published data ready';renderPublished();}catch(e){if(S.view!=='published')return;publishedPromise=null;current={publishedError:String(e.message||e)};status.textContent='Published data unavailable';renderPublished();}}
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
  const crossing=selectedRun.crossing||{};addMetric(metrics,'Ever crossed mass fraction',num(Number(crossing.ever_mass_fraction)*100,3)+'%');addMetric(metrics,'Final crossed mass fraction',num(Number(crossing.final_mass_fraction)*100,3)+'%');
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
window.addEventListener('hashchange',()=>{const next=decodeHash(location.hash);if(encodeHash(next)===encodeHash(S))return;S=next;controls.sync();selectView(S.view,false);run();});
window.labExperimentState = () => ({...S});
