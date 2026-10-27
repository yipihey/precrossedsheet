import React, { useState, useEffect, useRef, useMemo } from 'https://esm.sh/react@18.3.1';
import { createRoot } from 'https://esm.sh/react-dom@18.3.1/client';
import htm from 'https://esm.sh/htm@3.1.1';
import initWasm, * as wasmApi from '../../app/pkg/hcc_core.js';
import { coreRequest, normalizeState } from '../../app/js/state.js';
import { LinePlot } from '../../app/js/viz/plot.js';

const html = htm.bind(React.createElement);

let wasmReadyPromise = null;
function ensureWasm() {
  if (!wasmReadyPromise) {
    wasmReadyPromise = initWasm().then(() => wasmApi);
  }
  return wasmReadyPromise;
}

const TRAJ_LIST = [
  { id: '1lpt', label: '1LPT', color: '#56B4E9' },
  { id: '2lpt', label: '2LPT', color: '#E69F00' },
  { id: '3lpt', label: '3LPT', color: '#009E73' },
  { id: '4lpt', label: '4LPT', color: '#0072B2' },
  { id: 'pm',   label: 'PM',   color: '#D55E00' },
];

const EST_LIST = [
  { id: 'p0',  label: 'P₀',  dash: undefined, width: 2.3 },
  { id: 'p1',  label: 'P₁',  dash: '7 3',     width: 2.0 },
  { id: 'ngp', label: 'NGP', dash: '2 3',     width: 1.7 },
  { id: 'cic', label: 'CIC', dash: '4 2.5',   width: 1.9 },
  { id: 'tsc', label: 'TSC', dash: '6 2.5 1.5 2.5', width: 1.8 },
];

function styleForKey(key) {
  const [tr, est] = String(key).split(':');
  const tObj = TRAJ_LIST.find((t) => t.id === tr) || TRAJ_LIST[3];
  const eObj = EST_LIST.find((e) => e.id === est) || EST_LIST[0];
  let color = tObj.color;
  if (tr === '4lpt' && est === 'p1') color = '#009E73';
  if (tr === '4lpt' && est === 'ngp') color = '#dc2626';
  if (tr === '4lpt' && est === 'cic') color = '#D55E00';
  if (tr === '4lpt' && est === 'tsc') color = '#7c3aed';
  return {
    color,
    width: eObj.width,
    dash: eObj.dash,
    label: `${tObj.label} (${eObj.label})`,
  };
}

function coolwarmRgb(u) {
  const x = Math.max(0, Math.min(1, u));
  const stops = [
    [0.0,  [59, 76, 192]],
    [0.25, [141, 176, 254]],
    [0.5,  [247, 247, 247]],
    [0.75, [244, 154, 123]],
    [1.0,  [180, 4, 38]],
  ];
  for (let i = 0; i < stops.length - 1; i++) {
    const [p0, c0] = stops[i];
    const [p1, c1] = stops[i + 1];
    if (x <= p1) {
      const f = (x - p0) / (p1 - p0 || 1);
      return [
        Math.round(c0[0] + f * (c1[0] - c0[0])),
        Math.round(c0[1] + f * (c1[1] - c0[1])),
        Math.round(c0[2] + f * (c1[2] - c0[2])),
      ];
    }
  }
  return [180, 4, 38];
}

function drawCoolwarmHeatmap(canvas, values, M) {
  if (!canvas || !values || !M) return;
  const ctx = canvas.getContext('2d');
  const W = canvas.width;
  const H = canvas.height;
  const off = document.createElement('canvas');
  off.width = M;
  off.height = M;
  const octx = off.getContext('2d');
  const img = octx.createImageData(M, M);
  const logMin = -1.0; // log10(0.1)
  const logMax = 1.0;  // log10(10.0)
  for (let iy = 0; iy < M; iy++) {
    for (let ix = 0; ix < M; ix++) {
      const val = Math.max(1e-4, Number(values[ix * M + iy] ?? 0));
      const lv = Math.log10(val);
      const u = Math.max(0, Math.min(1, (lv - logMin) / (logMax - logMin)));
      const [r, g, b] = coolwarmRgb(u);
      const idx = ((M - 1 - iy) * M + ix) * 4;
      img.data[idx] = r;
      img.data[idx + 1] = g;
      img.data[idx + 2] = b;
      img.data[idx + 3] = 255;
    }
  }
  octx.putImageData(img, 0, 0);
  ctx.clearRect(0, 0, W, H);
  ctx.imageSmoothingEnabled = true;
  ctx.imageSmoothingQuality = 'high';
  ctx.drawImage(off, 0, 0, W, H);
}

function summarizeField(values) {
  if (!values || !values.length) {
    return { emptyPct: 0, min: 0, max: 0, variance: 0 };
  }
  let empty = 0;
  let min = Infinity;
  let max = -Infinity;
  let sumSq = 0;
  const n = values.length;
  for (let i = 0; i < n; i++) {
    const v = Number(values[i]);
    if (v <= 1e-10) empty++;
    if (v < min) min = v;
    if (v > max) max = v;
    const d = v - 1.0;
    sumSq += d * d;
  }
  return {
    emptyPct: (100 * empty) / n,
    min,
    max,
    variance: sumSq / n,
  };
}

function renderSvgPlot(container, axes, series) {
  if (!container) return;
  container.innerHTML = '';
  const plot = new LinePlot(container, { width: 840, height: 280, interactive: false });
  plot.setAxes(axes);
  plot.setSeries(series);
  plot.draw();
}

function stopSlideEvents(e) {
  e.stopPropagation();
}

function TrajectoryEstimatorMatrix({ selected, onToggle, onPreset }) {
  return html`
    <div>
      <div className="rlab-section-label">Trajectories × Density Estimators</div>
      <table className="rlab-matrix">
        <thead>
          <tr>
            <th>Traj</th>
            ${EST_LIST.map((e) => html`<th key=${e.id}>${e.label}</th>`)}
          </tr>
        </thead>
        <tbody>
          ${TRAJ_LIST.map((t) => html`
            <tr key=${t.id}>
              <td className="traj-name" style=${{ color: t.color }}>
                ${t.label}
              </td>
              ${EST_LIST.map((e) => {
                const key = `${t.id}:${e.id}`;
                const isOn = selected.includes(key);
                return html`
                  <td key=${e.id}>
                    <button
                      type="button"
                      className=${`rlab-chip-btn ${isOn ? 'active' : ''}`}
                      onClick=${() => onToggle(key)}
                    >
                      ${e.label}
                    </button>
                  </td>
                `;
              })}
            </tr>
          `)}
        </tbody>
      </table>
      <div className="rlab-Quick-row" style=${{ marginTop: '5px' }}>
        <button type="button" className="rlab-mini-btn" onClick=${() => onPreset(['4lpt:p0', '4lpt:p1', '4lpt:ngp', '4lpt:cic'])}>
          4LPT Estimators
        </button>
        <button type="button" className="rlab-mini-btn" onClick=${() => onPreset(['1lpt:p0', '2lpt:p0', '3lpt:p0', '4lpt:p0', 'pm:p0'])}>
          1–4LPT + PM (P₀)
        </button>
        <button type="button" className="rlab-mini-btn" onClick=${() => onPreset(['4lpt:p0', '4lpt:cic', 'pm:p0', 'pm:cic'])}>
          4LPT vs PM
        </button>
      </div>
    </div>
  `;
}

function HeatmapCard({ title, color, values, M }) {
  const canvasRef = useRef(null);
  const stats = useMemo(() => summarizeField(values), [values]);
  useEffect(() => {
    if (canvasRef.current && values && M) {
      drawCoolwarmHeatmap(canvasRef.current, values, M);
    }
  }, [values, M]);

  return html`
    <div className="rlab-heatmap-card">
      <div className="rlab-heatmap-title">
        <span style=${{ color }}>${title}</span>
        <span style=${{ fontSize: '9.5px', color: stats.emptyPct > 0 ? '#dc2626' : '#0f766e' }}>
          ${stats.emptyPct.toFixed(1)}% empty
        </span>
      </div>
      <canvas ref=${canvasRef} width="220" height="220" className="rlab-heatmap-canvas"></canvas>
      <div className="rlab-heatmap-stats">
        <span>min: <strong>${stats.min.toFixed(2)}</strong></span>
        <span>max: <strong>${stats.max.toFixed(2)}</strong></span>
        <span>σ²: <strong>${stats.variance.toFixed(3)}</strong></span>
        <span>M: <strong>${M}²</strong></span>
      </div>
    </div>
  `;
}

function buildLabRequest({ dim = 2, n = 32, mesh = 64, forceMesh = 32, scaleFactor = 0.65, smoothing = 0.0, seed = 2650, transfer = 'bao', boxSize = 300, initialSmoothing = 14, estimators = ['4lpt:p0', '4lpt:cic'] }) {
  const st = normalizeState({
    dim,
    n,
    mesh,
    forceMesh,
    ic: 'gaussian',
    seed,
    initialCutoff: Math.max(3, Math.min(15, Math.floor(n / 6))),
    boxSize,
    initialSmoothing,
    smoothing,
    epoch: 'scaleFactor',
    scaleFactor,
    fraction: scaleFactor,
    order: 4,
    steps: 32,
    transfer,
    estimators,
  });
  return { ...coreRequest(st), estimators };
}

/* ============================================================================
   PANEL 1: Estimator & Density Laboratory (Slide 8)
   ============================================================================ */
function EstimatorDensityPanel() {
  const [selected, setSelected] = useState(['4lpt:p0', '4lpt:p1', '4lpt:cic', '4lpt:ngp']);
  const [nVertices, setNVertices] = useState(32);
  const [meshRatio, setMeshRatio] = useState(2);
  const [scaleFactor, setScaleFactor] = useState(0.65);
  const [smoothing, setSmoothing] = useState(0.0);
  const [viewTab, setViewTab] = useState('heatmaps');
  const [busy, setBusy] = useState(true);
  const [resData, setResData] = useState(null);
  const plotRef = useRef(null);

  const toggleKey = (key) => {
    setSelected((prev) => (prev.includes(key) ? (prev.length > 1 ? prev.filter((k) => k !== key) : prev) : [...prev, key]));
  };

  useEffect(() => {
    let cancelled = false;
    setBusy(true);
    ensureWasm().then((wasm) => {
      if (cancelled) return;
      const req = buildLabRequest({
        dim: 2,
        n: nVertices,
        mesh: nVertices * meshRatio,
        forceMesh: nVertices,
        scaleFactor,
        smoothing,
        estimators: selected,
      });
      const res = JSON.parse(wasm.lab_run(JSON.stringify(req)));
      if (!cancelled) {
        setResData(res);
        setBusy(false);
      }
    }).catch((err) => {
      console.error('EstimatorDensityPanel error:', err);
      if (!cancelled) setBusy(false);
    });
    return () => { cancelled = true; };
  }, [selected, nVertices, meshRatio, scaleFactor, smoothing]);

  useEffect(() => {
    if (viewTab !== 'slice' || !plotRef.current || !resData) return;
    const M = resData.mesh;
    const L = resData.boxSize || 300;
    const xs = Array.from({ length: M }, (_, i) => ((i + 0.5) * L) / M);
    const midY = Math.floor(M / 2);
    const series = [];
    for (const key of selected) {
      const entry = resData.customSeries?.[key];
      if (!entry?.rho) continue;
      const st = styleForKey(key);
      const ys = Array.from({ length: M }, (_, ix) => entry.rho[ix * M + midY]);
      series.push({
        label: st.label,
        x: xs,
        y: ys,
        color: st.color,
        width: st.width,
        dash: st.dash,
      });
    }
    renderSvgPlot(
      plotRef.current,
      {
        title: `Mid-plane 1D cross-section ρ(x, y = L/2) / ρ̄  (N = ${nVertices}², M = ${M}²)`,
        xlabel: 'Eulerian coordinate x [Mpc/h]',
        ylabel: 'ρ / ρ̄',
        xlog: false,
        ylog: false,
      },
      series
    );
  }, [viewTab, resData, selected, nVertices]);

  const crossed = Boolean(resData?.shellCrossing?.crossed);
  const minJ = resData?.shellCrossing?.minVertexJacobian ?? 1.0;

  return html`
    <div className="react-slide-lab" onKeyDown=${stopSlideEvents} onPointerDown=${stopSlideEvents}>
      <div className="rlab-header">
        <div className="rlab-title-group">
          <span className=${`rlab-live-dot ${busy ? 'busy' : ''}`}></span>
          <span className="rlab-title">Interactive React + Rust/Wasm Sheet Estimator Laboratory</span>
          <span className="rlab-subtitle">Coolwarm log₁₀(ρ/ρ̄) ∈ [−1, +1] ([0.1, 10], white = ρ̄)</span>
        </div>
        <a className="rlab-open-btn" href="../../app/index.html#v=3&dim=2&n=64&force=64&mesh=128&view=density" target="_blank" rel="noopener">
          Open Full Lab ↗
        </a>
      </div>

      <div className="rlab-layout">
        <div className="rlab-controls">
          <${TrajectoryEstimatorMatrix}
            selected=${selected}
            onToggle=${toggleKey}
            onPreset=${setSelected}
          />
          <div className="rlab-control-row">
            <label><span>Lagrangian vertices N</span><span>${nVertices}²</span></label>
            <input type="range" min="16" max="64" step="16" value=${nVertices} onChange=${(e) => setNVertices(Number(e.target.value))} />
          </div>
          <div className="rlab-control-row">
            <label><span>Eulerian mesh ratio M/N</span><span>${meshRatio}× (${nVertices * meshRatio}²)</span></label>
            <input type="range" min="1" max="3" step="1" value=${meshRatio} onChange=${(e) => setMeshRatio(Number(e.target.value))} />
          </div>
          <div className="rlab-control-row">
            <label><span>Scale factor a</span><span>${scaleFactor.toFixed(2)}</span></label>
            <input type="range" min="0.25" max="1.0" step="0.05" value=${scaleFactor} onChange=${(e) => setScaleFactor(Number(e.target.value))} />
          </div>
          <div className="rlab-control-row">
            <label><span>Observation smoothing R_s/L</span><span>${smoothing.toFixed(2)}</span></label>
            <input type="range" min="0" max="0.06" step="0.01" value=${smoothing} onChange=${(e) => setSmoothing(Number(e.target.value))} />
          </div>
        </div>

        <div className="rlab-main">
          <div style=${{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '6px' }}>
            <div className="rlab-tabs">
              <button type="button" className=${`rlab-tab ${viewTab === 'heatmaps' ? 'active' : ''}`} onClick=${() => setViewTab('heatmaps')}>
                2D Log-Density Maps (Coolwarm [0.1, 10])
              </button>
              <button type="button" className=${`rlab-tab ${viewTab === 'slice' ? 'active' : ''}`} onClick=${() => setViewTab('slice')}>
                Mid-Plane Cross-Section ρ(x)/ρ̄
              </button>
            </div>
            <div className="rlab-metrics-strip">
              <span className="rlab-metric-pill">Min vertex J: <strong>${Number(minJ).toFixed(3)}</strong></span>
              <span className="rlab-metric-pill">Regime: <strong>${!crossed ? 'Single-stream (J > 0)' : 'Post-crossing'}</strong></span>
            </div>
          </div>

          ${viewTab === 'heatmaps' ? html`
            <div className="rlab-canvas-grid">
              ${selected.slice(0, 4).map((key) => {
                const entry = resData?.customSeries?.[key];
                if (!entry?.rho) return null;
                const st = styleForKey(key);
                return html`
                  <${HeatmapCard}
                    key=${key}
                    title=${st.label}
                    color=${st.color}
                    values=${entry.rho}
                    M=${resData.mesh}
                  />
                `;
              })}
            </div>
          ` : html`
            <div className="rlab-plot-box" ref=${plotRef}></div>
          `}

          <div className="rlab-colorbar-bar">
            <span><strong>0.1 ρ̄</strong> (voids)</span>
            <div className="rlab-colorbar-grad">
              <div className="rlab-colorbar-mid" title="ρ / ρ̄ = 1"></div>
            </div>
            <span><strong>1.0 ρ̄</strong> (neutral white center)</span>
            <span>→ <strong>10 ρ̄</strong> (filaments/walls)</span>
          </div>
        </div>
      </div>
    </div>
  `;
}

/* ============================================================================
   PANEL 2: Simplex vs. Grid Fourier Spectra Laboratory (Slide 10)
   ============================================================================ */
function SimplexSpectraPanel() {
  const [selected, setSelected] = useState(['4lpt:p0', '4lpt:p1', '4lpt:cic', '4lpt:ngp']);
  const [specMode, setSpecMode] = useState('meshfree');
  const [nVertices, setNVertices] = useState(32);
  const [meshRatio, setMeshRatio] = useState(2);
  const [scaleFactor, setScaleFactor] = useState(0.65);
  const [busy, setBusy] = useState(true);
  const [resData, setResData] = useState(null);
  const plotRef = useRef(null);

  const toggleKey = (key) => {
    setSelected((prev) => (prev.includes(key) ? (prev.length > 1 ? prev.filter((k) => k !== key) : prev) : [...prev, key]));
  };

  useEffect(() => {
    let cancelled = false;
    setBusy(true);
    ensureWasm().then((wasm) => {
      if (cancelled) return;
      const req = buildLabRequest({
        dim: 2,
        n: nVertices,
        mesh: nVertices * meshRatio,
        forceMesh: nVertices,
        scaleFactor,
        estimators: selected,
      });
      const res = JSON.parse(wasm.lab_run(JSON.stringify(req)));
      if (!cancelled) {
        setResData(res);
        setBusy(false);
      }
    }).catch((err) => {
      console.error('SimplexSpectraPanel error:', err);
      if (!cancelled) setBusy(false);
    });
    return () => { cancelled = true; };
  }, [selected, nVertices, meshRatio, scaleFactor]);

  useEffect(() => {
    if (!plotRef.current || !resData) return;
    const sp = resData.spectra;
    const k = sp.k;
    const series = [];

    if (specMode === 'meshfree') {
      for (const key of selected) {
        const entry = resData.customSeries?.[key];
        if (!entry) continue;
        const st = styleForKey(key);
        series.push({
          label: `${st.label} (Gridded FFT)`,
          x: k,
          y: entry.power,
          color: st.color,
          width: st.width,
          dash: st.dash,
        });
        if (entry.directShell?.length && sp.directShellK?.length) {
          series.push({
            label: `${st.label} (16-ray Mesh-free Simplex)`,
            x: sp.directShellK,
            y: entry.directShell,
            color: '#7c3aed',
            width: 2.4,
            dash: '6 3',
          });
        }
      }
      renderSvgPlot(
        plotRef.current,
        {
          title: '16-Ray Brillouin-Zone Mesh-Free Exact Simplex Fourier Modes vs. Gridded FFT',
          xlabel: 'Wavenumber k [h/Mpc]',
          ylabel: 'P(k)',
          xlog: true,
          ylog: true,
        },
        series
      );
    } else if (specMode === 'fft') {
      for (const key of selected) {
        const entry = resData.customSeries?.[key];
        if (!entry) continue;
        const st = styleForKey(key);
        series.push({
          label: st.label,
          x: k,
          y: entry.power,
          color: st.color,
          width: st.width,
          dash: st.dash,
        });
      }
      if (sp.realized?.length) {
        series.push({
          label: 'Realized Linear D(a)² |δ₀(k)|²',
          x: sp.realizedK || k,
          y: sp.realized,
          color: '#64748b',
          width: 1.5,
          dash: '2 4',
        });
      }
      renderSvgPlot(
        plotRef.current,
        {
          title: 'Isotropic Shell-Averaged Power Spectra P(k)',
          xlabel: 'Wavenumber k [h/Mpc]',
          ylabel: 'P(k)',
          xlog: true,
          ylog: true,
        },
        series
      );
    } else if (specMode === 'transfer') {
      const denom = sp.realized?.length === k.length ? sp.realized : sp.linear;
      const mask = k.map((_, i) => denom[i] > 1e-12);
      const kf = k.filter((_, i) => mask[i]);
      const df = denom.filter((_, i) => mask[i]);
      for (const key of selected) {
        const entry = resData.customSeries?.[key];
        if (!entry) continue;
        const st = styleForKey(key);
        series.push({
          label: `${st.label} / P_lin,realized`,
          x: kf,
          y: entry.power.filter((_, i) => mask[i]).map((v, idx) => v / Math.max(df[idx], 1e-30)),
          color: st.color,
          width: st.width,
          dash: st.dash,
        });
      }
      renderSvgPlot(
        plotRef.current,
        {
          title: 'Cosmic-Variance-Free Nonlinear Response P(k) / [D(a)² |δ₀(k)|²]',
          xlabel: 'Wavenumber k [h/Mpc]',
          ylabel: 'P(k) / P_lin,realized(k)',
          xlog: true,
          ylog: false,
        },
        series
      );
    } else {
      const ref = resData.customSeries?.['4lpt:p0']?.power || sp.sheet;
      for (const key of selected) {
        const entry = resData.customSeries?.[key];
        if (!entry) continue;
        const st = styleForKey(key);
        series.push({
          label: `${st.label} / 4LPT (P₀)`,
          x: k,
          y: entry.power.map((v, i) => v / Math.max(ref[i], 1e-30)),
          color: st.color,
          width: st.width,
          dash: st.dash,
        });
      }
      renderSvgPlot(
        plotRef.current,
        {
          title: 'Relative Estimator Power Ratio P(k) / P_4LPT,P0(k)',
          xlabel: 'Wavenumber k [h/Mpc]',
          ylabel: 'Ratio to 4LPT (P₀)',
          xlog: true,
          ylog: false,
        },
        series
      );
    }
  }, [resData, specMode, selected]);

  return html`
    <div className="react-slide-lab" onKeyDown=${stopSlideEvents} onPointerDown=${stopSlideEvents}>
      <div className="rlab-header">
        <div className="rlab-title-group">
          <span className=${`rlab-live-dot ${busy ? 'busy' : ''}`}></span>
          <span className="rlab-title">Interactive Hermite–Genocchi Simplex & FFT Spectrum Analyzer</span>
          <span className="rlab-subtitle">I_T(k) = m_T d! exp[−ik·x₀, …, −ik·x_d]</span>
        </div>
        <a className="rlab-open-btn" href="../../app/index.html#v=3&dim=2&n=64&force=64&mesh=128&view=spectra" target="_blank" rel="noopener">
          Open Full Lab ↗
        </a>
      </div>

      <div className="rlab-layout">
        <div className="rlab-controls">
          <${TrajectoryEstimatorMatrix}
            selected=${selected}
            onToggle=${toggleKey}
            onPreset=${setSelected}
          />
          <div className="rlab-control-row">
            <label><span>Lagrangian vertices N</span><span>${nVertices}²</span></label>
            <input type="range" min="16" max="64" step="16" value=${nVertices} onChange=${(e) => setNVertices(Number(e.target.value))} />
          </div>
          <div className="rlab-control-row">
            <label><span>Eulerian mesh ratio M/N</span><span>${meshRatio}× (${nVertices * meshRatio}²)</span></label>
            <input type="range" min="1" max="3" step="1" value=${meshRatio} onChange=${(e) => setMeshRatio(Number(e.target.value))} />
          </div>
          <div className="rlab-control-row">
            <label><span>Scale factor a</span><span>${scaleFactor.toFixed(2)}</span></label>
            <input type="range" min="0.25" max="1.0" step="0.05" value=${scaleFactor} onChange=${(e) => setScaleFactor(Number(e.target.value))} />
          </div>
        </div>

        <div className="rlab-main">
          <div className="rlab-tabs">
            <button type="button" className=${`rlab-tab ${specMode === 'meshfree' ? 'active' : ''}`} onClick=${() => setSpecMode('meshfree')}>
              Mesh-Free Simplex vs. FFT
            </button>
            <button type="button" className=${`rlab-tab ${specMode === 'fft' ? 'active' : ''}`} onClick=${() => setSpecMode('fft')}>
              Isotropic FFT P(k)
            </button>
            <button type="button" className=${`rlab-tab ${specMode === 'transfer' ? 'active' : ''}`} onClick=${() => setSpecMode('transfer')}>
              Nonlinear Transfer P(k)/P_lin,realized
            </button>
            <button type="button" className=${`rlab-tab ${specMode === 'ratio' ? 'active' : ''}`} onClick=${() => setSpecMode('ratio')}>
              Ratio to 4LPT (P₀)
            </button>
          </div>
          <div className="rlab-plot-box" ref=${plotRef}></div>
        </div>
      </div>
    </div>
  `;
}

/* ============================================================================
   PANEL 3: LPT Order Convergence, Scale Variance σ²(R) & PDFs (Slide 13)
   ============================================================================ */
function LptConvergenceVariancePdfPanel() {
  const [selected, setSelected] = useState(['1lpt:p0', '2lpt:p0', '3lpt:p0', '4lpt:p0', 'pm:p0', '4lpt:cic']);
  const [activeTab, setActiveTab] = useState('variance_th');
  const [nVertices, setNVertices] = useState(32);
  const [scaleFactor, setScaleFactor] = useState(0.75);
  const [busy, setBusy] = useState(true);
  const [resData, setResData] = useState(null);
  const plotRef = useRef(null);

  const toggleKey = (key) => {
    setSelected((prev) => (prev.includes(key) ? (prev.length > 1 ? prev.filter((k) => k !== key) : prev) : [...prev, key]));
  };

  useEffect(() => {
    let cancelled = false;
    setBusy(true);
    ensureWasm().then((wasm) => {
      if (cancelled) return;
      const req = buildLabRequest({
        dim: 2,
        n: nVertices,
        mesh: nVertices * 2,
        forceMesh: nVertices,
        scaleFactor,
        estimators: selected,
      });
      const res = JSON.parse(wasm.lab_run(JSON.stringify(req)));
      if (!cancelled) {
        setResData(res);
        setBusy(false);
      }
    }).catch((err) => {
      console.error('LptConvergenceVariancePdfPanel error:', err);
      if (!cancelled) setBusy(false);
    });
    return () => { cancelled = true; };
  }, [selected, nVertices, scaleFactor]);

  useEffect(() => {
    if (!plotRef.current || !resData) return;
    const series = [];

    if (activeTab === 'variance_th' || activeTab === 'variance_g') {
      const vs = resData.varianceScale;
      const R = vs.radii;
      const isGauss = activeTab === 'variance_g';
      for (const key of selected) {
        const entry = resData.customSeries?.[key];
        if (!entry?.variance) continue;
        const st = styleForKey(key);
        series.push({
          label: st.label,
          x: R,
          y: isGauss ? entry.variance.gaussian : entry.variance.tophat,
          color: st.color,
          width: st.width,
          dash: st.dash,
        });
      }
      renderSvgPlot(
        plotRef.current,
        {
          title: isGauss
            ? 'Scale-Dependent Variance σ²_G(R) (Gaussian Window exp(−k²R²/2))'
            : 'Scale-Dependent Variance σ²_TH(R) (Real-Space Top-Hat Sphere R)',
          xlabel: 'Filter radius R [Mpc/h]',
          ylabel: 'Variance σ²(R)',
          xlog: true,
          ylog: true,
        },
        series
      );
    } else {
      const isMass = activeTab === 'pdf_mass_log';
      const rhoAxis = (resData.pdf?.logrho || []).map((lr) => Math.pow(10, lr));
      for (const key of selected) {
        const entry = resData.customSeries?.[key];
        if (!entry?.pdf) continue;
        const st = styleForKey(key);
        series.push({
          label: st.label,
          x: rhoAxis,
          y: isMass ? entry.pdf.mass : entry.pdf.volume,
          color: st.color,
          width: st.width,
          dash: st.dash,
        });
      }
      renderSvgPlot(
        plotRef.current,
        {
          title: isMass
            ? 'One-Point Mass-Weighted Log-Density PDF dP_M / d log₁₀(ρ/ρ̄)'
            : 'One-Point Volume-Weighted Log-Density PDF dP_V / d log₁₀(ρ/ρ̄)',
          xlabel: 'Normalized density ρ / ρ̄ (log scale)',
          ylabel: isMass ? 'dP_M / d log₁₀(ρ/ρ̄)' : 'dP_V / d log₁₀(ρ/ρ̄)',
          xlog: true,
          ylog: false,
        },
        series
      );
    }
  }, [resData, activeTab, selected]);

  return html`
    <div className="react-slide-lab" onKeyDown=${stopSlideEvents} onPointerDown=${stopSlideEvents}>
      <div className="rlab-header">
        <div className="rlab-title-group">
          <span className=${`rlab-live-dot ${busy ? 'busy' : ''}`}></span>
          <span className="rlab-title">Interactive LPT Order Convergence (1LPT → 4LPT vs PM Sheet)</span>
          <span className="rlab-subtitle">Isolating dynamical trajectory convergence from estimator noise</span>
        </div>
        <a className="rlab-open-btn" href="../../app/index.html#v=3&dim=2&n=64&force=64&mesh=128&view=variance" target="_blank" rel="noopener">
          Open Full Lab ↗
        </a>
      </div>

      <div className="rlab-layout">
        <div className="rlab-controls">
          <${TrajectoryEstimatorMatrix}
            selected=${selected}
            onToggle=${toggleKey}
            onPreset=${setSelected}
          />
          <div className="rlab-control-row">
            <label><span>Lagrangian vertices N</span><span>${nVertices}²</span></label>
            <input type="range" min="16" max="64" step="16" value=${nVertices} onChange=${(e) => setNVertices(Number(e.target.value))} />
          </div>
          <div className="rlab-control-row">
            <label><span>Scale factor a</span><span>${scaleFactor.toFixed(2)}</span></label>
            <input type="range" min="0.3" max="1.0" step="0.05" value=${scaleFactor} onChange=${(e) => setScaleFactor(Number(e.target.value))} />
          </div>
        </div>

        <div className="rlab-main">
          <div className="rlab-tabs">
            <button type="button" className=${`rlab-tab ${activeTab === 'variance_th' ? 'active' : ''}`} onClick=${() => setActiveTab('variance_th')}>
              Top-Hat Variance σ²_TH(R)
            </button>
            <button type="button" className=${`rlab-tab ${activeTab === 'variance_g' ? 'active' : ''}`} onClick=${() => setActiveTab('variance_g')}>
              Gaussian Variance σ²_G(R)
            </button>
            <button type="button" className=${`rlab-tab ${activeTab === 'pdf_vol_log' ? 'active' : ''}`} onClick=${() => setActiveTab('pdf_vol_log')}>
              Volume PDF dP_V/d log₁₀ρ
            </button>
            <button type="button" className=${`rlab-tab ${activeTab === 'pdf_mass_log' ? 'active' : ''}`} onClick=${() => setActiveTab('pdf_mass_log')}>
              Mass PDF dP_M/d log₁₀ρ
            </button>
          </div>
          <div className="rlab-plot-box" ref=${plotRef}></div>
        </div>
      </div>
    </div>
  `;
}

/* ============================================================================
   PANEL 4: BAO Preservation & Analytic JVP Inference Explorer (Slide 20)
   ============================================================================ */
function BaoJvpInferencePanel() {
  const [mode, setMode] = useState('bao');
  const [nVertices, setNVertices] = useState(48);
  const [scaleFactor, setScaleFactor] = useState(0.75);
  const [seed, setSeed] = useState(2650);
  const [busy, setBusy] = useState(true);
  const [baoData, setBaoData] = useState(null);
  const plotRef = useRef(null);

  useEffect(() => {
    let cancelled = false;
    setBusy(true);
    ensureWasm().then((wasm) => {
      if (cancelled) return;
      const ests = ['4lpt:p0', '4lpt:p1', '4lpt:cic'];
      const reqBao = buildLabRequest({
        dim: 2,
        n: nVertices,
        mesh: nVertices * 2,
        forceMesh: nVertices,
        scaleFactor,
        seed,
        transfer: 'bao',
        boxSize: 1000,
        initialSmoothing: 10,
        estimators: ests,
      });
      const reqSmooth = buildLabRequest({
        dim: 2,
        n: nVertices,
        mesh: nVertices * 2,
        forceMesh: nVertices,
        scaleFactor,
        seed,
        transfer: 'smooth',
        boxSize: 1000,
        initialSmoothing: 10,
        estimators: ests,
      });
      const wiggle = JSON.parse(wasm.lab_run(JSON.stringify(reqBao)));
      const noWiggle = JSON.parse(wasm.lab_run(JSON.stringify(reqSmooth)));
      if (!cancelled) {
        setBaoData({ wiggle, noWiggle });
        setBusy(false);
      }
    }).catch((err) => {
      console.error('BaoJvpInferencePanel error:', err);
      if (!cancelled) setBusy(false);
    });
    return () => { cancelled = true; };
  }, [nVertices, scaleFactor, seed]);

  useEffect(() => {
    if (!plotRef.current || !baoData) return;
    const { wiggle, noWiggle } = baoData;

    if (mode === 'bao') {
      const k = wiggle.spectra.k;
      const mask = k.map((kv) => kv >= 0.015 && kv <= 0.30);
      const kf = k.filter((_, i) => mask[i]);
      const ratio = (numArr, denArr) =>
        numArr.filter((_, i) => mask[i]).map((v, idx) => {
          const d = denArr.filter((_, j) => mask[j])[idx];
          return v / Math.max(d, 1e-30);
        });

      const series = [
        {
          label: 'Linear Theory P_BAO(k) / P_smooth(k)',
          x: kf,
          y: ratio(wiggle.spectra.linear, noWiggle.spectra.linear),
          color: '#64748b',
          width: 1.8,
          dash: '2 3',
        },
        {
          label: '4LPT Sheet (P₀) BAO / No-Wiggle Ratio',
          x: kf,
          y: ratio(wiggle.spectra.sheet, noWiggle.spectra.sheet),
          color: '#0072B2',
          width: 2.5,
        },
        {
          label: '4LPT Sheet (P₁) BAO / No-Wiggle Ratio',
          x: kf,
          y: ratio(wiggle.spectra.sheetP1, noWiggle.spectra.sheetP1),
          color: '#009E73',
          width: 2.1,
          dash: '7 3',
        },
        {
          label: '4LPT Particles (CIC) BAO / No-Wiggle Ratio',
          x: kf,
          y: ratio(wiggle.spectra.cic, noWiggle.spectra.cic),
          color: '#D55E00',
          width: 1.9,
          dash: '4 2.5',
        },
      ];
      renderSvgPlot(
        plotRef.current,
        {
          title: 'Phase-Matched BAO Modulation Ratio P_BAO(k) / P_smooth(k) (L = 1000 Mpc/h)',
          xlabel: 'Wavenumber k [h/Mpc]',
          ylabel: 'P_BAO(k) / P_smooth(k)',
          xlog: false,
          ylog: false,
        },
        series
      );
    } else {
      const g = wiggle.gradient;
      if (g?.k?.length) {
        renderSvgPlot(
          plotRef.current,
          {
            title: 'Exact Analytic Confluent Divided-Difference JVP ∂P(k)/∂A vs. Central Finite Difference',
            xlabel: 'Wavenumber k [h/Mpc]',
            ylabel: '∂P(k) / ∂A',
            xlog: false,
            ylog: false,
          },
          [
            {
              label: 'Analytic Confluent Divided-Difference JVP ∂P(k)/∂A',
              x: g.k,
              y: g.tangent,
              color: '#0072B2',
              width: 2.5,
            },
            {
              label: 'Centered Finite Difference [P(A+h) − P(A−h)] / 2h',
              x: g.k,
              y: g.finiteDifference,
              color: '#D55E00',
              width: 2.0,
              dash: '5 3',
            },
          ]
        );
      }
    }
  }, [baoData, mode]);

  const maxRelErr = baoData?.wiggle?.diagnostics?.powerRelativeError ?? 0;

  return html`
    <div className="react-slide-lab" onKeyDown=${stopSlideEvents} onPointerDown=${stopSlideEvents}>
      <div className="rlab-header">
        <div className="rlab-title-group">
          <span className=${`rlab-live-dot ${busy ? 'busy' : ''}`}></span>
          <span className="rlab-title">Interactive BAO Acoustic Preservation & Differentiable Forward Model (JVP)</span>
          <span className="rlab-subtitle">Confluent divided-difference identity ∂/∂s_i exp[s₀,…,s_d] = exp[s₀,…,s_d,s_i]</span>
        </div>
        <a className="rlab-open-btn" href="../../app/index.html#v=3&dim=2&n=256&force=256&mesh=256&ic=gaussian&transfer=bao&cutoff=31&L=1000&Rini=14&smoothing=0&epoch=scaleFactor&a=1&order=4&steps=64&view=spectra" target="_blank" rel="noopener">
          Open Full Lab ↗
        </a>
      </div>

      <div className="rlab-layout">
        <div className="rlab-controls">
          <div className="rlab-section-label">Inference & BAO Controls</div>
          <div className="rlab-control-row">
            <label><span>Lagrangian vertices N</span><span>${nVertices}²</span></label>
            <input type="range" min="32" max="64" step="16" value=${nVertices} onChange=${(e) => setNVertices(Number(e.target.value))} />
          </div>
          <div className="rlab-control-row">
            <label><span>Scale factor a</span><span>${scaleFactor.toFixed(2)}</span></label>
            <input type="range" min="0.4" max="1.0" step="0.05" value=${scaleFactor} onChange=${(e) => setScaleFactor(Number(e.target.value))} />
          </div>
          <div className="rlab-control-row">
            <label><span>Realization seed</span><span>#${seed}</span></label>
            <input type="range" min="2600" max="2700" step="1" value=${seed} onChange=${(e) => setSeed(Number(e.target.value))} />
          </div>
          <div className="rlab-metrics-strip" style=${{ marginTop: '6px' }}>
            <span className="rlab-metric-pill">Box L: <strong>1000 Mpc/h</strong></span>
            <span className="rlab-metric-pill">JVP rel. err: <strong>${Number(maxRelErr).toExponential(2)}</strong></span>
          </div>
        </div>

        <div className="rlab-main">
          <div className="rlab-tabs">
            <button type="button" className=${`rlab-tab ${mode === 'bao' ? 'active' : ''}`} onClick=${() => setMode('bao')}>
              Phase-Matched BAO / No-Wiggle Ratio P_BAO(k)/P_smooth(k)
            </button>
            <button type="button" className=${`rlab-tab ${mode === 'jvp' ? 'active' : ''}`} onClick=${() => setMode('jvp')}>
              Analytic JVP ∂P(k)/∂A vs. Finite Difference
            </button>
          </div>
          <div className="rlab-plot-box" ref=${plotRef}></div>
        </div>
      </div>
    </div>
  `;
}

const PANEL_COMPONENTS = {
  'estimator-density': EstimatorDensityPanel,
  'simplex-spectra': SimplexSpectraPanel,
  'lpt-convergence-variance-pdf': LptConvergenceVariancePdfPanel,
  'bao-jvp-inference': BaoJvpInferencePanel,
};

function mountAllSlidePanels() {
  const containers = document.querySelectorAll('.react-slide-lab-mount');
  containers.forEach((el) => {
    if (el.dataset.mounted === 'true') return;
    const panelType = el.dataset.panel || 'estimator-density';
    const Comp = PANEL_COMPONENTS[panelType] || EstimatorDensityPanel;
    el.dataset.mounted = 'true';
    const root = createRoot(el);
    root.render(html`<${Comp} />`);
  });
}

if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', mountAllSlidePanels);
} else {
  mountAllSlidePanels();
}
