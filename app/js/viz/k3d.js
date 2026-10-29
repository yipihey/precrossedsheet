// K3D (k3d-jupyter 3.2.0) standalone WebGL2 / Three.js 3D visualization bridge
// for the Cosmological Phase-Space Sheet Laboratory.
// Runs 100% client-side in the browser without Python or Jupyter.

import { COLORMAPS } from './colormaps.js';
import { downloadBlob, flash } from './download.js';

let k3dModulePromise = null;
const activeInstances = new Set();

const LOCAL_STANDALONE_URL = './vendor/k3d/standalone.js';
const CDN_STANDALONE_URL = 'https://cdn.jsdelivr.net/npm/k3d@3.2.0/dist/standalone.js';

function injectScriptWithAmdCapture(src) {
  return new Promise((resolve, reject) => {
    if (window.__precrossedK3DLib) {
      resolve(window.__precrossedK3DLib);
      return;
    }
    const prevDefine = window.define;
    let captured = null;
    const amdShim = function (name, deps, factory) {
      const fn =
        typeof name === 'function'
          ? name
          : typeof deps === 'function'
            ? deps
            : factory;
      try {
        const mod = typeof fn === 'function' ? fn() : fn;
        if (mod && (mod.K3D || name === 'k3d')) {
          captured = mod;
          window.__precrossedK3DLib = mod;
        }
      } catch (err) {
        reject(err);
      }
      if (typeof prevDefine === 'function') {
        try {
          prevDefine.apply(this, arguments);
        } catch (_) {
          // ignore downstream AMD registry conflicts
        }
      }
    };
    amdShim.amd = true;
    window.define = amdShim;

    const script = document.createElement('script');
    script.src = src;
    script.async = true;
    script.onload = () => {
      if (captured || window.__precrossedK3DLib) {
        resolve(captured || window.__precrossedK3DLib);
      } else {
        reject(new Error(`K3D bundle loaded from ${src} but did not register the k3d module.`));
      }
    };
    script.onerror = () => {
      script.remove();
      reject(new Error(`Failed to load K3D standalone bundle from ${src}.`));
    };
    document.head.appendChild(script);
  });
}

export function loadK3D() {
  if (window.__precrossedK3DLib) return Promise.resolve(window.__precrossedK3DLib);
  if (!k3dModulePromise) {
    k3dModulePromise = injectScriptWithAmdCapture(LOCAL_STANDALONE_URL).catch(() =>
      injectScriptWithAmdCapture(CDN_STANDALONE_URL),
    );
  }
  return k3dModulePromise;
}

export function disposeK3DInstance(instance) {
  if (!instance) return;
  activeInstances.delete(instance);
  try {
    instance.disable?.();
  } catch (_) {
    // best-effort WebGL cleanup
  }
}

export function disposeAllK3DInstances() {
  for (const inst of Array.from(activeInstances)) {
    disposeK3DInstance(inst);
  }
  activeInstances.clear();
}

function isDarkMode() {
  return Boolean(
    window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches,
  );
}

function rgbAt(cmapName, t) {
  const x = Math.max(0, Math.min(1, t));
  if (cmapName === 'blueorange') {
    const h = (((218 - 205 * x) % 360) + 360) % 360;
    const s = (50 + 40 * x) / 100;
    const l = (97 - 53 * x) / 100;
    const c = (1 - Math.abs(2 * l - 1)) * s;
    const hp = h / 60;
    const x2 = c * (1 - Math.abs((hp % 2) - 1));
    const m = l - c / 2;
    const [r, g, b] =
      hp < 1
        ? [c, x2, 0]
        : hp < 2
          ? [x2, c, 0]
          : hp < 3
            ? [0, c, x2]
            : hp < 4
              ? [0, x2, c]
              : hp < 5
                ? [x2, 0, c]
                : [c, 0, x2];
    return [r + m, g + m, b + m];
  }
  if (cmapName === 'ylgnbu') {
    const stops = [
      [0, [0, 32, 77]],
      [0.25, [49, 73, 107]],
      [0.5, [124, 123, 120]],
      [0.75, [190, 175, 111]],
      [1, [255, 234, 70]],
    ];
    for (let i = 0; i < stops.length - 1; i++) {
      const [t0, c0] = stops[i];
      const [t1, c1] = stops[i + 1];
      if (x <= t1 || i === stops.length - 2) {
        const f = (x - t0) / Math.max(1e-9, t1 - t0);
        return [
          (c0[0] + (c1[0] - c0[0]) * f) / 255,
          (c0[1] + (c1[1] - c0[1]) * f) / 255,
          (c0[2] + (c1[2] - c0[2]) * f) / 255,
        ];
      }
    }
  }
  const fn = COLORMAPS[cmapName] || COLORMAPS.coolwarm || COLORMAPS.viridis;
  const [r, g, b] = fn(x);
  return [r / 255, g / 255, b / 255];
}

export function buildK3DColorMap(cmapName = 'coolwarm', steps = 33) {
  const data = new Float32Array(steps * 4);
  for (let i = 0; i < steps; i++) {
    const t = i / (steps - 1);
    const [r, g, b] = rgbAt(cmapName, t);
    data[i * 4] = t;
    data[i * 4 + 1] = r;
    data[i * 4 + 2] = g;
    data[i * 4 + 3] = b;
  }
  return { data, shape: [steps, 4] };
}

function identityMatrix4() {
  return {
    data: new Float32Array([
      1, 0, 0, 0,
      0, 1, 0, 0,
      0, 0, 1, 0,
      0, 0, 0, 1,
    ]),
    shape: [4, 4],
  };
}

function boxVolumeMatrix4(L, zScale = L, zOffset = L * 0.5) {
  return {
    data: new Float32Array([
      L, 0, 0, L * 0.5,
      0, L, 0, L * 0.5,
      0, 0, zScale, zOffset,
      0, 0, 0, 1,
    ]),
    shape: [4, 4],
  };
}

function hexInt(cssHex) {
  if (typeof cssHex === 'number') return cssHex;
  const s = String(cssHex || '#0072B2').replace(/^#/, '');
  const n = parseInt(s, 16);
  return Number.isFinite(n) ? n : 0x0072b2;
}

function arr(a) {
  return ArrayBuffer.isView(a) ? a : Array.isArray(a) ? a : [];
}

function wrapPeriodicDelta(dx) {
  let d = dx;
  if (d > 0.5) d -= 1.0;
  else if (d < -0.5) d += 1.0;
  return d;
}

function estimateVertexJacobians2D(xArr, n) {
  const out = new Float32Array(n * n);
  const dx0 = 1 / n;
  for (let i = 0; i < n; i++) {
    const ip = (i + 1) % n;
    const im = (i + n - 1) % n;
    for (let j = 0; j < n; j++) {
      const jp = (j + 1) % n;
      const jm = (j + n - 1) % n;
      const idXp = (ip * n + j) * 2;
      const idXm = (im * n + j) * 2;
      const idYp = (i * n + jp) * 2;
      const idYm = (i * n + jm) * 2;
      const dxdq0 = wrapPeriodicDelta(Number(xArr[idXp]) - Number(xArr[idXm])) / (2 * dx0);
      const dydq0 = wrapPeriodicDelta(Number(xArr[idXp + 1]) - Number(xArr[idXm + 1])) / (2 * dx0);
      const dxdq1 = wrapPeriodicDelta(Number(xArr[idYp]) - Number(xArr[idYm])) / (2 * dx0);
      const dydq1 = wrapPeriodicDelta(Number(xArr[idYp + 1]) - Number(xArr[idYm + 1])) / (2 * dx0);
      out[i * n + j] = dxdq0 * dydq1 - dydq0 * dxdq1;
    }
  }
  return out;
}

function estimateVertexJacobians3D(xArr, n) {
  const out = new Float32Array(n * n * n);
  const dx0 = 1 / n;
  const idx = (i, j, k) => ((i * n + j) * n + k) * 3;
  for (let i = 0; i < n; i++) {
    const ip = (i + 1) % n;
    const im = (i + n - 1) % n;
    for (let j = 0; j < n; j++) {
      const jp = (j + 1) % n;
      const jm = (j + n - 1) % n;
      for (let k = 0; k < n; k++) {
        const kp = (k + 1) % n;
        const km = (k + n - 1) % n;
        const ixp = idx(ip, j, k), ixm = idx(im, j, k);
        const iyp = idx(i, jp, k), iym = idx(i, jm, k);
        const izp = idx(i, j, kp), izm = idx(i, j, km);
        const a00 = wrapPeriodicDelta(Number(xArr[ixp]) - Number(xArr[ixm])) / (2 * dx0);
        const a10 = wrapPeriodicDelta(Number(xArr[ixp + 1]) - Number(xArr[ixm + 1])) / (2 * dx0);
        const a20 = wrapPeriodicDelta(Number(xArr[ixp + 2]) - Number(xArr[ixm + 2])) / (2 * dx0);
        const a01 = wrapPeriodicDelta(Number(xArr[iyp]) - Number(xArr[iym])) / (2 * dx0);
        const a11 = wrapPeriodicDelta(Number(xArr[iyp + 1]) - Number(xArr[iym + 1])) / (2 * dx0);
        const a21 = wrapPeriodicDelta(Number(xArr[iyp + 2]) - Number(xArr[iym + 2])) / (2 * dx0);
        const a02 = wrapPeriodicDelta(Number(xArr[izp]) - Number(xArr[izm])) / (2 * dx0);
        const a12 = wrapPeriodicDelta(Number(xArr[izp + 1]) - Number(xArr[izm + 1])) / (2 * dx0);
        const a22 = wrapPeriodicDelta(Number(xArr[izp + 2]) - Number(xArr[izm + 2])) / (2 * dx0);
        const det =
          a00 * (a11 * a22 - a12 * a21) -
          a01 * (a10 * a22 - a12 * a20) +
          a02 * (a10 * a21 - a11 * a20);
        out[(i * n + j) * n + k] = det;
      }
    }
  }
  return out;
}

function percentileRange(values, pLo = 0.02, pHi = 0.98, fallback = [-1, 1]) {
  const finite = [];
  for (let i = 0; i < values.length; i++) {
    const v = Number(values[i]);
    if (Number.isFinite(v)) finite.push(v);
  }
  if (!finite.length) return fallback;
  finite.sort((a, b) => a - b);
  let lo = finite[Math.floor((finite.length - 1) * pLo)];
  let hi = finite[Math.ceil((finite.length - 1) * pHi)];
  if (!(hi > lo)) {
    lo -= 0.5;
    hi += 0.5;
  }
  return [lo, hi];
}

export function attachK3DExportBar(toolbarEl, getK3DInstance, baseName = 'precrossedsheet-k3d') {
  const mkBtn = (label, title, onClick) => {
    const b = document.createElement('button');
    b.type = 'button';
    b.className = 'qd-btn k3d-action-btn';
    b.textContent = label;
    b.title = title;
    b.addEventListener('click', () => onClick(b));
    toolbarEl.appendChild(b);
    return b;
  };

  mkBtn('Reset 3D camera', 'Auto-fit the 3D camera to the scene bounds', (btn) => {
    const inst = getK3DInstance();
    if (!inst) return;
    inst.resetCamera();
    inst.render();
    flash(btn, 'Reset');
  });

  mkBtn('Save 3D PNG', 'Capture a high-resolution PNG screenshot of the 3D viewport', async (btn) => {
    const inst = getK3DInstance();
    if (!inst) return;
    const canvas = await inst.getScreenshot(2, false);
    canvas.toBlob((blob) => {
      if (blob) {
        downloadBlob(blob, `${baseName}.png`);
        flash(btn, 'Saved PNG');
      }
    });
  });

  mkBtn('Export standalone .html', 'Download a self-contained interactive 3D HTML file that opens in any browser', (btn) => {
    const inst = getK3DInstance();
    if (!inst) return;
    const html = inst.getHTMLSnapshot(9);
    const blob = new Blob([html], { type: 'text/html;charset=utf-8' });
    downloadBlob(blob, `${baseName}.html`);
    flash(btn, 'Exported HTML');
  });

  mkBtn('Export .k3d', 'Download a compressed K3D binary snapshot (.k3d)', (btn) => {
    const inst = getK3DInstance();
    if (!inst) return;
    const snap = inst.getSnapshot(9);
    const blob = new Blob([snap], { type: 'application/octet-stream' });
    downloadBlob(blob, `${baseName}.k3d`);
    flash(btn, 'Saved .k3d');
  });

  mkBtn('Fullscreen', 'Toggle fullscreen on the 3D K3D viewport', () => {
    const inst = getK3DInstance();
    const node = inst?.getWorld?.()?.targetDOMNode;
    if (!node) return;
    if (!document.fullscreenElement) {
      node.requestFullscreen?.();
    } else {
      document.exitFullscreen?.();
    }
  });
}

async function createK3DViewport(hostEl, params = {}) {
  const lib = await loadK3D();
  hostEl.replaceChildren();
  const dark = isDarkMode();
  const k3d = new lib.K3D(lib.ThreeJsProvider, hostEl, {
    cameraAutoFit: true,
    gridAutoFit: false,
    gridVisible: true,
    gridColor: dark ? 0x384654 : 0xdbe2e8,
    labelColor: dark ? 0xd5dfe8 : 0x24313e,
    clearColor: dark ? 0x161c22 : 0xfbfcfd,
    antialias: 1,
    logarithmicDepthBuffer: true,
    lighting: 1.65,
    depthPeels: 4,
    renderer: 'simple',
    cameraMode: 'trackball',
    standaloneGUI: true,
    menuVisibility: true,
    snapshotType: 'online',
    name: 'precrossedsheet-k3d',
    ...params,
  });
  activeInstances.add(k3d);
  return k3d;
}

/**
 * Mount an interactive 3D K3D Phase-Space Sheet scene (1D, 2D, and 3D).
 */
export async function mountK3DSheetScene(hostEl, d, state, opts = {}) {
  const dim = Number(d.dim || state.dim || 2);
  const L = Number(d.boxSize || state.boxSize || 100);
  const q = arr(d.q);
  const x = arr(d.x).length ? arr(d.x) : arr(d.rsd?.wireframe?.xReal);
  const v = arr(d.v);
  const pmX = arr(d.pmX);
  const nv = Math.floor(Math.min(x.length, Math.max(q.length, x.length)) / Math.max(1, dim));
  const n = Number(d.sheetN || (dim === 1 ? nv : dim === 2 ? Math.round(Math.sqrt(nv)) : Math.round(Math.cbrt(nv))) || d.n || state.n);

  const zMode = opts.zMode || (dim === 2 ? 'v_los' : 'position');
  const colorMode = opts.colorMode || 'log_rho';
  const cmap = opts.cmap || 'coolwarm';
  const opacity = Number.isFinite(opts.opacity) ? opts.opacity : 0.82;
  const showWireframe = opts.showWireframe ?? true;
  const showVectors = opts.showVectors ?? false;
  const showPmOverlay = opts.showPmOverlay ?? false;
  const renderer = opts.renderer || 'simple';
  const depthPeels = Number.isFinite(opts.depthPeels) ? opts.depthPeels : 4;
  const sliceMode3D = opts.sliceMode3D || 'tri_planes';
  const sliceIndex3D = Number.isFinite(opts.sliceIndex3D) ? opts.sliceIndex3D : 0.5;

  const objects = [];
  const colorMap = buildK3DColorMap(cmap);
  let axes = ['x₀ [Mpc/h]', 'x₁ [Mpc/h]', 'x₂ [Mpc/h]'];
  let grid = [0, 0, 0, L, L, L];

  if (dim === 1 && nv >= 2) {
    axes = ['x [Mpc/h]', 'dx/da (velocity)', 'q [Mpc/h]'];
    const pos = new Float32Array(nv * 3);
    const attr = new Float32Array(nv);
    const segs = [];
    let vMin = Infinity, vMax = -Infinity;
    for (let i = 0; i < nv; i++) {
      const vi = Number(v[i] || 0);
      vMin = Math.min(vMin, vi);
      vMax = Math.max(vMax, vi);
    }
    const vSpan = Math.max(1e-6, vMax - vMin);
    for (let i = 0; i < nv; i++) {
      const xi = Number(x[i] || 0) * L;
      const vi = ((Number(v[i] || 0) - vMin) / vSpan) * L;
      const qi = (q.length ? Number(q[i]) : i / nv) * L;
      pos[i * 3] = xi;
      pos[i * 3 + 1] = vi;
      pos[i * 3 + 2] = qi;
      const ip = (i + 1) % nv;
      const dx = wrapPeriodicDelta(Number(x[ip]) - Number(x[i])) * nv;
      attr[i] = Math.log10(1 / Math.max(1e-4, Math.abs(dx)));
      if (i + 1 < nv && Math.abs(Number(x[i + 1]) - Number(x[i])) < 0.45) {
        segs.push(i, i + 1);
      }
    }
    const cRange = percentileRange(attr, 0.02, 0.98, [-1, 1.5]);
    objects.push({
      type: 'Lines',
      name: '1D Phase-Space Curve (x, v, q)',
      visible: true,
      model_matrix: identityMatrix4(),
      vertices: { data: pos, shape: [nv, 3] },
      indices: { data: new Uint32Array(segs), shape: [segs.length / 2, 2] },
      indices_type: 'segment',
      shader: 'simple',
      color: 0x0072b2,
      opacity: 1.0,
    });
    objects.push({
      type: 'Points',
      name: 'Lagrangian Vertices',
      visible: true,
      model_matrix: identityMatrix4(),
      positions: { data: pos, shape: [nv, 3] },
      attribute: { data: attr, shape: [nv] },
      color_map: colorMap,
      color_range: cRange,
      color: 0xffffff,
      point_size: L * 0.014,
      shader: '3d',
      opacity: 1.0,
    });
  } else if (dim === 2 && n >= 2) {
    const maxN = 128;
    const stride = n > maxN ? Math.ceil(n / maxN) : 1;
    const m = Math.floor(n / stride);
    const subX = new Float32Array(m * m * 2);
    const subV = new Float32Array(m * m * 2);
    const subQ = new Float32Array(m * m * 2);
    for (let i = 0; i < m; i++) {
      const ii = i * stride;
      for (let j = 0; j < m; j++) {
        const jj = j * stride;
        const src = (ii * n + jj) * 2;
        const dst = (i * m + j) * 2;
        subX[dst] = Number(x[src] ?? ii / n);
        subX[dst + 1] = Number(x[src + 1] ?? jj / n);
        subV[dst] = Number(v[src] ?? 0);
        subV[dst + 1] = Number(v[src + 1] ?? 0);
        subQ[dst] = q.length > src ? Number(q[src]) : ii / n;
        subQ[dst + 1] = q.length > src + 1 ? Number(q[src + 1]) : jj / n;
      }
    }
    const jac = estimateVertexJacobians2D(subX, m);
    const count = m * m;
    const zRaw = new Float32Array(count);
    const attr = new Float32Array(count);

    for (let p = 0; p < count; p++) {
      const jVal = jac[p];
      const logRho = Math.log10(1 / Math.max(1e-4, Math.abs(jVal)));
      const vx = subV[p * 2];
      const vy = subV[p * 2 + 1];
      const vMag = Math.hypot(vx, vy);
      const disp = Math.hypot(
        wrapPeriodicDelta(subX[p * 2] - subQ[p * 2]),
        wrapPeriodicDelta(subX[p * 2 + 1] - subQ[p * 2 + 1]),
      );

      if (zMode === 'v_los') zRaw[p] = vy;
      else if (zMode === 'v_mag') zRaw[p] = vMag;
      else if (zMode === 'log_rho') zRaw[p] = logRho;
      else if (zMode === 'jacobian') zRaw[p] = jVal;
      else zRaw[p] = 0;

      if (colorMode === 'jacobian') attr[p] = jVal;
      else if (colorMode === 'stream') attr[p] = jVal <= 0 ? -1 : 1;
      else if (colorMode === 'v_los') attr[p] = vy;
      else if (colorMode === 'disp') attr[p] = disp * L;
      else attr[p] = logRho;
    }

    const [zLo, zHi] = percentileRange(zRaw, 0.01, 0.99, [-1, 1]);
    const zSpan = Math.max(1e-6, zHi - zLo);
    const zHeight = zMode === 'flat' ? L * 0.08 : L * 0.42;
    const vertices = new Float32Array(count * 3);
    for (let p = 0; p < count; p++) {
      vertices[p * 3] = ((subX[p * 2] % 1 + 1) % 1) * L;
      vertices[p * 3 + 1] = ((subX[p * 2 + 1] % 1 + 1) % 1) * L;
      vertices[p * 3 + 2] =
        zMode === 'flat'
          ? 0
          : ((zRaw[p] - 0.5 * (zLo + zHi)) / zSpan) * zHeight;
    }

    const zLabel =
      zMode === 'v_los'
        ? 'v₁ (phase-space height)'
        : zMode === 'v_mag'
          ? '|v| speed'
          : zMode === 'log_rho'
            ? 'log₁₀(1/|J|)'
            : zMode === 'jacobian'
              ? 'Jacobian J(q)'
              : 'z = 0';
    axes = ['x₀ [Mpc/h]', 'x₁ [Mpc/h]', zLabel];
    grid = [0, 0, -0.35 * L, L, L, 0.35 * L];

    const tris = [];
    const segs = [];
    const maxEdgeSq = (0.42 * L) ** 2;
    const distSq = (a, b) => {
      const dx = vertices[a * 3] - vertices[b * 3];
      const dy = vertices[a * 3 + 1] - vertices[b * 3 + 1];
      return dx * dx + dy * dy;
    };
    for (let i = 0; i + 1 < m; i++) {
      for (let j = 0; j + 1 < m; j++) {
        const a = i * m + j;
        const b = (i + 1) * m + j;
        const c = i * m + (j + 1);
        const e = (i + 1) * m + (j + 1);
        if (distSq(a, b) < maxEdgeSq && distSq(a, e) < maxEdgeSq && distSq(b, e) < maxEdgeSq) {
          tris.push(a, b, e);
        }
        if (distSq(a, c) < maxEdgeSq && distSq(a, e) < maxEdgeSq && distSq(c, e) < maxEdgeSq) {
          tris.push(a, e, c);
        }
        if (showWireframe) {
          if (distSq(a, b) < maxEdgeSq) segs.push(a, b);
          if (distSq(a, c) < maxEdgeSq) segs.push(a, c);
          if (distSq(a, e) < maxEdgeSq) segs.push(a, e);
        }
      }
    }

    const cRange =
      colorMode === 'stream'
        ? [-1, 1]
        : percentileRange(attr, 0.02, 0.98, [-1, 1.5]);

    if (tris.length) {
      objects.push({
        type: 'Mesh',
        name: `LPT Deformed Sheet (${m}×${m})`,
        visible: true,
        model_matrix: identityMatrix4(),
        vertices: { data: vertices, shape: [count, 3] },
        indices: { data: new Uint32Array(tris), shape: [tris.length / 3, 3] },
        attribute: { data: attr, shape: [count] },
        color_map: colorMap,
        color_range: cRange,
        color: 0xffffff,
        opacity,
        alpha_mode: opacity < 1 ? 'blend' : 'opaque',
        flat_shading: false,
        wireframe: false,
        side: 'double',
        roughness: 0.35,
        metalness: 0.05,
      });
    }

    if (showWireframe && segs.length) {
      objects.push({
        type: 'Lines',
        name: 'Lagrangian Simplex Edges',
        visible: true,
        model_matrix: identityMatrix4(),
        vertices: { data: vertices, shape: [count, 3] },
        indices: { data: new Uint32Array(segs), shape: [segs.length / 2, 2] },
        indices_type: 'segment',
        shader: 'simple',
        color: isDarkMode() ? 0x99b3cc : 0x1f2d3a,
        opacity: 0.36,
      });
    }

    if (showPmOverlay && pmX.length >= n * n * 2) {
      const pmVerts = new Float32Array(count * 3);
      const pmSegs = [];
      for (let i = 0; i < m; i++) {
        const ii = i * stride;
        for (let j = 0; j < m; j++) {
          const jj = j * stride;
          const src = (ii * n + jj) * 2;
          const p = i * m + j;
          pmVerts[p * 3] = ((Number(pmX[src]) % 1 + 1) % 1) * L;
          pmVerts[p * 3 + 1] = ((Number(pmX[src + 1]) % 1 + 1) % 1) * L;
          pmVerts[p * 3 + 2] = vertices[p * 3 + 2];
        }
      }
      const pmDistSq = (a, b) => {
        const dx = pmVerts[a * 3] - pmVerts[b * 3];
        const dy = pmVerts[a * 3 + 1] - pmVerts[b * 3 + 1];
        return dx * dx + dy * dy;
      };
      for (let i = 0; i + 1 < m; i++) {
        for (let j = 0; j + 1 < m; j++) {
          const a = i * m + j;
          const b = (i + 1) * m + j;
          const c = i * m + (j + 1);
          if (pmDistSq(a, b) < maxEdgeSq) pmSegs.push(a, b);
          if (pmDistSq(a, c) < maxEdgeSq) pmSegs.push(a, c);
        }
      }
      if (pmSegs.length) {
        objects.push({
          type: 'Lines',
          name: 'PM N-body Wireframe Overlay',
          visible: true,
          model_matrix: identityMatrix4(),
          vertices: { data: pmVerts, shape: [count, 3] },
          indices: { data: new Uint32Array(pmSegs), shape: [pmSegs.length / 2, 2] },
          indices_type: 'segment',
          shader: 'simple',
          color: 0xd55e00,
          opacity: 0.78,
        });
      }
    }

    if (showVectors) {
      const vecStep = Math.max(1, Math.floor(m / 20));
      const orig = [];
      const vecs = [];
      let maxV = 1e-6;
      for (let p = 0; p < count; p++) {
        maxV = Math.max(maxV, Math.hypot(subV[p * 2], subV[p * 2 + 1]));
      }
      const vScale = (0.075 * L) / maxV;
      for (let i = 0; i < m; i += vecStep) {
        for (let j = 0; j < m; j += vecStep) {
          const p = i * m + j;
          orig.push(vertices[p * 3], vertices[p * 3 + 1], vertices[p * 3 + 2]);
          vecs.push(subV[p * 2] * vScale, subV[p * 2 + 1] * vScale, 0);
        }
      }
      if (orig.length) {
        objects.push({
          type: 'Vectors',
          name: 'Velocity Field v(q)',
          visible: true,
          model_matrix: identityMatrix4(),
          origins: { data: new Float32Array(orig), shape: [orig.length / 3, 3] },
          vectors: { data: new Float32Array(vecs), shape: [vecs.length / 3, 3] },
          origin_color: 0x0072b2,
          head_color: 0xd55e00,
          use_head: true,
          head_size: 1.0,
          line_width: 0.003,
          labels: [],
          label_size: 1.0,
        });
      }
    }
  } else if (dim === 3 && n >= 2) {
    const maxN = 32;
    const stride = n > maxN ? Math.ceil(n / maxN) : 1;
    const m = Math.floor(n / stride);
    const count = m * m * m;
    const subX = new Float32Array(count * 3);
    const subV = new Float32Array(count * 3);
    for (let i = 0; i < m; i++) {
      const ii = i * stride;
      for (let j = 0; j < m; j++) {
        const jj = j * stride;
        for (let k = 0; k < m; k++) {
          const kk = k * stride;
          const src = ((ii * n + jj) * n + kk) * 3;
          const dst = ((i * m + j) * m + k) * 3;
          subX[dst] = ((Number(x[src] ?? ii / n) % 1) + 1) % 1;
          subX[dst + 1] = ((Number(x[src + 1] ?? jj / n) % 1) + 1) % 1;
          subX[dst + 2] = ((Number(x[src + 2] ?? kk / n) % 1) + 1) % 1;
          subV[dst] = Number(v[src] ?? 0);
          subV[dst + 1] = Number(v[src + 1] ?? 0);
          subV[dst + 2] = Number(v[src + 2] ?? 0);
        }
      }
    }
    const jac = estimateVertexJacobians3D(subX, m);
    const vertices = new Float32Array(count * 3);
    const attr = new Float32Array(count);
    for (let p = 0; p < count; p++) {
      vertices[p * 3] = subX[p * 3] * L;
      vertices[p * 3 + 1] = subX[p * 3 + 1] * L;
      vertices[p * 3 + 2] = subX[p * 3 + 2] * L;
      const jVal = jac[p];
      if (colorMode === 'jacobian') attr[p] = jVal;
      else if (colorMode === 'stream') attr[p] = jVal <= 0 ? -1 : 1;
      else if (colorMode === 'v_los') attr[p] = subV[p * 3 + 2];
      else attr[p] = Math.log10(1 / Math.max(1e-4, Math.abs(jVal)));
    }

    const maxEdgeSq = (0.42 * L) ** 2;
    const distSq = (a, b) => {
      const dx = vertices[a * 3] - vertices[b * 3];
      const dy = vertices[a * 3 + 1] - vertices[b * 3 + 1];
      const dz = vertices[a * 3 + 2] - vertices[b * 3 + 2];
      return dx * dx + dy * dy + dz * dz;
    };
    const idx = (i, j, k) => (i * m + j) * m + k;
    const tris = [];
    const segs = [];

    const addSheetPlane = (planeAxis, fixedIdx) => {
      for (let u = 0; u + 1 < m; u++) {
        for (let w = 0; w + 1 < m; w++) {
          const getP = (du, dw) =>
            planeAxis === 0
              ? idx(fixedIdx, u + du, w + dw)
              : planeAxis === 1
                ? idx(u + du, fixedIdx, w + dw)
                : idx(u + du, w + dw, fixedIdx);
          const a = getP(0, 0);
          const b = getP(1, 0);
          const c = getP(0, 1);
          const e = getP(1, 1);
          if (distSq(a, b) < maxEdgeSq && distSq(a, e) < maxEdgeSq && distSq(b, e) < maxEdgeSq) {
            tris.push(a, b, e);
          }
          if (distSq(a, c) < maxEdgeSq && distSq(a, e) < maxEdgeSq && distSq(c, e) < maxEdgeSq) {
            tris.push(a, e, c);
          }
          if (showWireframe) {
            if (distSq(a, b) < maxEdgeSq) segs.push(a, b);
            if (distSq(a, c) < maxEdgeSq) segs.push(a, c);
          }
        }
      }
    };

    const kCenter = Math.max(0, Math.min(m - 1, Math.round(sliceIndex3D * (m - 1))));
    if (sliceMode3D === 'single_plane') {
      addSheetPlane(2, kCenter);
    } else if (sliceMode3D === 'multi_stack') {
      const step = Math.max(1, Math.floor(m / 4));
      for (let k = Math.floor(step / 2); k < m; k += step) {
        addSheetPlane(2, k);
      }
    } else if (sliceMode3D === 'caustic_only') {
      for (let i = 0; i + 1 < m; i++) {
        for (let j = 0; j + 1 < m; j++) {
          for (let k = 0; k + 1 < m; k++) {
            const a = idx(i, j, k);
            const b = idx(i + 1, j, k);
            const c = idx(i, j + 1, k);
            const e = idx(i + 1, j + 1, k);
            if (Math.min(jac[a], jac[b], jac[c], jac[e]) <= 0.15) {
              if (distSq(a, b) < maxEdgeSq && distSq(a, e) < maxEdgeSq && distSq(b, e) < maxEdgeSq) {
                tris.push(a, b, e);
              }
              if (distSq(a, c) < maxEdgeSq && distSq(a, e) < maxEdgeSq && distSq(c, e) < maxEdgeSq) {
                tris.push(a, e, c);
              }
            }
          }
        }
      }
      if (!tris.length) addSheetPlane(2, kCenter);
    } else {
      // tri_planes: 3 orthogonal Lagrangian planes intersecting at kCenter
      addSheetPlane(0, kCenter);
      addSheetPlane(1, kCenter);
      addSheetPlane(2, kCenter);
    }

    const cRange =
      colorMode === 'stream'
        ? [-1, 1]
        : percentileRange(attr, 0.02, 0.98, [-1, 1.5]);

    if (tris.length) {
      objects.push({
        type: 'Mesh',
        name: `3D Deformed Lagrangian Sheet (${sliceMode3D})`,
        visible: true,
        model_matrix: identityMatrix4(),
        vertices: { data: vertices, shape: [count, 3] },
        indices: { data: new Uint32Array(tris), shape: [tris.length / 3, 3] },
        attribute: { data: attr, shape: [count] },
        color_map: colorMap,
        color_range: cRange,
        color: 0xffffff,
        opacity,
        alpha_mode: opacity < 1 ? 'blend' : 'opaque',
        flat_shading: false,
        wireframe: false,
        side: 'double',
        roughness: 0.35,
        metalness: 0.05,
      });
    }

    if (showWireframe && segs.length) {
      objects.push({
        type: 'Lines',
        name: '3D Lagrangian Lattice Wireframe',
        visible: true,
        model_matrix: identityMatrix4(),
        vertices: { data: vertices, shape: [count, 3] },
        indices: { data: new Uint32Array(segs), shape: [segs.length / 2, 2] },
        indices_type: 'segment',
        shader: 'simple',
        color: isDarkMode() ? 0x8aa4bd : 0x24313e,
        opacity: 0.32,
      });
    }

    if (showVectors) {
      const vecStep = Math.max(1, Math.floor(m / 8));
      const orig = [];
      const vecs = [];
      let maxV = 1e-6;
      for (let p = 0; p < count; p++) {
        maxV = Math.max(maxV, Math.hypot(subV[p * 3], subV[p * 3 + 1], subV[p * 3 + 2]));
      }
      const vScale = (0.09 * L) / maxV;
      for (let i = 0; i < m; i += vecStep) {
        for (let j = 0; j < m; j += vecStep) {
          for (let k = 0; k < m; k += vecStep) {
            const p = idx(i, j, k);
            orig.push(vertices[p * 3], vertices[p * 3 + 1], vertices[p * 3 + 2]);
            vecs.push(subV[p * 3] * vScale, subV[p * 3 + 1] * vScale, subV[p * 3 + 2] * vScale);
          }
        }
      }
      if (orig.length) {
        objects.push({
          type: 'Vectors',
          name: '3D Velocity Field v(q)',
          visible: true,
          model_matrix: identityMatrix4(),
          origins: { data: new Float32Array(orig), shape: [orig.length / 3, 3] },
          vectors: { data: new Float32Array(vecs), shape: [vecs.length / 3, 3] },
          origin_color: 0x0072b2,
          head_color: 0xd55e00,
          use_head: true,
          head_size: 1.0,
          line_width: 0.003,
          labels: [],
          label_size: 1.0,
        });
      }
    }
  }

  const k3d = await createK3DViewport(hostEl, {
    axes,
    grid,
    renderer,
    depthPeels,
    name: `precrossedsheet-sheet-${dim}d`,
  });
  if (objects.length) {
    await k3d.load({ objects });
  }
  return k3d;
}

function extract3DVolumeOrSlice(seriesItem, d, dim, mesh) {
  const raw = arr(seriesItem?.rho3d).length ? arr(seriesItem.rho3d) : arr(seriesItem?.rho);
  if (!raw.length) return null;
  if (dim === 3) {
    const n3 = Math.round(Math.cbrt(raw.length));
    if (n3 * n3 * n3 === raw.length && n3 >= 2) {
      return { is3D: true, n: n3, data: raw };
    }
    const n2 = Math.round(Math.sqrt(raw.length));
    if (n2 * n2 === raw.length && n2 >= 2) {
      return { is3D: false, n: n2, data: raw };
    }
  }
  if (dim === 2) {
    const n2 = Math.round(Math.sqrt(raw.length)) || mesh;
    if (n2 >= 2) return { is3D: false, n: n2, data: raw };
  }
  return null;
}

/**
 * Mount an interactive 3D K3D Density Field scene (Volume raycasting, Marching Cubes
 * isosurfaces, VolumeSlice 3-axis tomography in 3D; 3D Surface relief in 2D).
 */
export async function mountK3DDensityScene(hostEl, d, activeSeries, state, densityViewOpts, k3dOpts = {}) {
  const dim = Number(d.dim || state.dim || 2);
  const mesh = Number(d.mesh || state.mesh || 32);
  const L = Number(d.boxSize || state.boxSize || 100);
  const isLog = densityViewOpts.scale === 'log';
  const cmap = densityViewOpts.cmap || 'coolwarm';
  const colorMap = buildK3DColorMap(cmap);

  const primarySeries =
    activeSeries.find((s) => s.key === k3dOpts.primaryKey) ||
    activeSeries.find((s) => s.rho?.length || s.rho3d?.length) ||
    activeSeries[0];
  const compareSeries =
    k3dOpts.compareKey && k3dOpts.compareKey !== 'none'
      ? activeSeries.find((s) => s.key === k3dOpts.compareKey)
      : null;

  const vmin = Number.isFinite(densityViewOpts.vmin) ? densityViewOpts.vmin : isLog ? 0.1 : 0;
  const vmax = Number.isFinite(densityViewOpts.vmax) && densityViewOpts.vmax > vmin ? densityViewOpts.vmax : 10;
  const rangeLo = isLog ? Math.log10(Math.max(1e-4, vmin)) : vmin;
  const rangeHi = isLog ? Math.log10(Math.max(1e-3, vmax)) : vmax;
  const isoRho = Number.isFinite(k3dOpts.isoLevel) ? k3dOpts.isoLevel : 2.5;
  const isoVal = isLog ? Math.log10(Math.max(1e-4, isoRho)) : isoRho;
  const mode3D = k3dOpts.mode3D || 'volume_iso';
  const renderer = k3dOpts.renderer || 'simple';
  const depthPeels = Number.isFinite(k3dOpts.depthPeels) ? k3dOpts.depthPeels : 4;

  const transformVal = (v, mean = 1) => {
    const r = Math.max(0, Number(v) || 0) / (mean || 1);
    return isLog ? Math.log10(Math.max(1e-4, r)) : r;
  };

  const objects = [];
  let axes = ['x₀ [Mpc/h]', 'x₁ [Mpc/h]', 'x₂ [Mpc/h]'];
  let grid = [0, 0, 0, L, L, L];

  const extracted = primarySeries ? extract3DVolumeOrSlice(primarySeries, d, dim, mesh) : null;
  if (extracted && extracted.is3D) {
    const m = extracted.n;
    const src = extracted.data;
    const mean = Number(primarySeries.massMean) || 1;
    // Reorder (i, j, k) -> (k, j, i) so K3D's (x=width, y=height, z=length) matches (x0, x1, x2)
    const vol = new Float32Array(m * m * m);
    for (let i = 0; i < m; i++) {
      for (let j = 0; j < m; j++) {
        for (let k = 0; k < m; k++) {
          vol[(k * m + j) * m + i] = transformVal(src[(i * m + j) * m + k], mean);
        }
      }
    }
    const volMatrix = boxVolumeMatrix4(L, L, L * 0.5);
    const opacityRamp = {
      data: new Float32Array([
        rangeLo, 0.0,
        rangeLo + 0.35 * (rangeHi - rangeLo), 0.04,
        rangeLo + 0.55 * (rangeHi - rangeLo), 0.22,
        rangeLo + 0.8 * (rangeHi - rangeLo), 0.65,
        rangeHi, 0.95,
      ]),
      shape: [5, 2],
    };

    if (mode3D === 'volume' || mode3D === 'volume_iso') {
      objects.push({
        type: 'Volume',
        name: `3D Volume · ${primarySeries.shortLabel}`,
        visible: true,
        model_matrix: volMatrix,
        volume: { data: vol, shape: [m, m, m] },
        color_map: colorMap,
        opacity_function: opacityRamp,
        color_range: [rangeLo, rangeHi],
        samples: 512.0,
        alpha_coef: Number(k3dOpts.alphaCoef) || 40.0,
        gradient_step: 0.005,
        shadow: 'off',
        shadow_delay: 500,
        shadow_res: 128,
        interpolation: densityViewOpts.interp !== 'pixel',
        mask: { data: new Uint8Array(0), shape: [0] },
        mask_opacities: { data: new Float32Array(0), shape: [0] },
      });
    }

    if (mode3D === 'mip') {
      objects.push({
        type: 'MIP',
        name: `3D Maximum Intensity Projection · ${primarySeries.shortLabel}`,
        visible: true,
        model_matrix: volMatrix,
        volume: { data: vol, shape: [m, m, m] },
        color_map: colorMap,
        opacity_function: opacityRamp,
        color_range: [rangeLo, rangeHi],
        samples: 512.0,
        gradient_step: 0.005,
        interpolation: densityViewOpts.interp !== 'pixel',
        mask: { data: new Uint8Array(0), shape: [0] },
        mask_opacities: { data: new Float32Array(0), shape: [0] },
      });
    }

    if (mode3D === 'slices' || mode3D === 'slices_iso') {
      const mid = Math.floor(m / 2);
      objects.push({
        type: 'VolumeSlice',
        name: `3-Axis Orthogonal Slices · ${primarySeries.shortLabel}`,
        visible: true,
        model_matrix: volMatrix,
        volume: { data: vol, shape: [m, m, m] },
        color_map: colorMap,
        opacity_function: { data: new Float32Array([rangeLo, 1.0, rangeHi, 1.0]), shape: [2, 2] },
        color_range: [rangeLo, rangeHi],
        slice_x: mid,
        slice_y: mid,
        slice_z: mid,
        opacity: 0.92,
        interpolation: densityViewOpts.interp === 'pixel' ? 0 : 1,
        mask: { data: new Uint8Array(0), shape: [0] },
        color_map_masks: { data: new Float32Array(0), shape: [0] },
        active_masks: { data: new Float32Array(0), shape: [0] },
      });
    }

    if (mode3D === 'iso' || mode3D === 'volume_iso' || mode3D === 'slices_iso') {
      objects.push({
        type: 'MarchingCubes',
        name: `Isosurface ρ/ρ̄ = ${isoRho} · ${primarySeries.shortLabel}`,
        visible: true,
        model_matrix: volMatrix,
        scalar_field: { data: vol, shape: [m, m, m] },
        level: isoVal,
        color: hexInt(primarySeries.color || '#0072B2'),
        opacity: compareSeries ? 0.62 : 0.78,
        wireframe: false,
        flat_shading: false,
        roughness: 0.28,
        metalness: 0.06,
      });
    }

    if (compareSeries && compareSeries !== primarySeries) {
      const compExt = extract3DVolumeOrSlice(compareSeries, d, dim, mesh);
      if (compExt && compExt.is3D) {
        const mc = compExt.n;
        const cSrc = compExt.data;
        const cMean = Number(compareSeries.massMean) || 1;
        const cVol = new Float32Array(mc * mc * mc);
        for (let i = 0; i < mc; i++) {
          for (let j = 0; j < mc; j++) {
            for (let k = 0; k < mc; k++) {
              cVol[(k * mc + j) * mc + i] = transformVal(cSrc[(i * mc + j) * mc + k], cMean);
            }
          }
        }
        objects.push({
          type: 'MarchingCubes',
          name: `Comparison Isosurface ρ/ρ̄ = ${isoRho} · ${compareSeries.shortLabel}`,
          visible: true,
          model_matrix: volMatrix,
          scalar_field: { data: cVol, shape: [mc, mc, mc] },
          level: isoVal,
          color: hexInt(compareSeries.color || '#D55E00'),
          opacity: 0.58,
          wireframe: true,
          flat_shading: false,
          roughness: 0.35,
          metalness: 0.05,
        });
      }
    }
  } else if (extracted) {
    // 2D Density Map (or 2D slice of 3D): render as a 3D Surface height-field
    const m = extracted.n;
    const src = extracted.data;
    const mean = Number(primarySeries.massMean) || 1;
    const heights = new Float32Array(m * m);
    const zSpan = Math.max(1e-6, rangeHi - rangeLo);
    const reliefHeight = L * 0.36;
    for (let i = 0; i < m; i++) {
      for (let j = 0; j < m; j++) {
        const val = transformVal(src[i * m + j], mean);
        heights[j * m + i] = ((val - rangeLo) / zSpan - 0.5) * reliefHeight;
      }
    }
    const attr = new Float32Array(m * m);
    for (let i = 0; i < m; i++) {
      for (let j = 0; j < m; j++) {
        attr[j * m + i] = transformVal(src[i * m + j], mean);
      }
    }
    axes = [
      'x₀ [Mpc/h]',
      'x₁ [Mpc/h]',
      isLog ? 'log₁₀(ρ/ρ̄) relief' : 'ρ/ρ̄ relief',
    ];
    grid = [0, 0, -0.25 * L, L, L, 0.25 * L];

    objects.push({
      type: 'Surface',
      name: `3D Density Relief · ${primarySeries.shortLabel}`,
      visible: true,
      model_matrix: boxVolumeMatrix4(L, 1.0, 0.0),
      heights: { data: heights, shape: [m, m] },
      attribute: { data: attr, shape: [m, m] },
      color_map: colorMap,
      color_range: [rangeLo, rangeHi],
      color: 0xffffff,
      opacity: compareSeries ? 0.78 : 0.94,
      flat_shading: densityViewOpts.interp === 'pixel',
      wireframe: false,
      roughness: 0.32,
      metalness: 0.04,
    });

    if (compareSeries && compareSeries !== primarySeries) {
      const compExt = extract3DVolumeOrSlice(compareSeries, d, dim, mesh);
      if (compExt && !compExt.is3D) {
        const mc = compExt.n;
        const cSrc = compExt.data;
        const cMean = Number(compareSeries.massMean) || 1;
        const cHeights = new Float32Array(mc * mc);
        for (let i = 0; i < mc; i++) {
          for (let j = 0; j < mc; j++) {
            const val = transformVal(cSrc[i * mc + j], cMean);
            cHeights[j * mc + i] = ((val - rangeLo) / zSpan - 0.5) * reliefHeight;
          }
        }
        objects.push({
          type: 'Surface',
          name: `Comparison Wireframe · ${compareSeries.shortLabel}`,
          visible: true,
          model_matrix: boxVolumeMatrix4(L, 1.0, 0.0),
          heights: { data: cHeights, shape: [mc, mc] },
          color: hexInt(compareSeries.color || '#D55E00'),
          opacity: 0.68,
          flat_shading: true,
          wireframe: true,
        });
      }
    }
  }

  const k3d = await createK3DViewport(hostEl, {
    axes,
    grid,
    renderer,
    depthPeels,
    name: `precrossedsheet-density-${dim}d`,
  });
  if (objects.length) {
    await k3d.load({ objects });
  }
  return k3d;
}

/**
 * Mount an interactive 3D K3D Redshift-Space Distortions (RSD) scene:
 * - Redshifted phase-space sheet s(q) colored by J_{s,T} or log10(1/|J_{s,T}|)
 * - Real-space sheet x(q) overlay
 * - Kaiser / Finger-of-God line-of-sight displacement vectors s(q) - x(q)
 * - Simplex velocity-gradient Gaussian packet Monte Carlo sample cloud
 */
export async function mountK3DRsdScene(hostEl, d, state, rsdViewOpts, densityViewOpts, k3dOpts = {}) {
  const rsd = d?.rsd || {};
  const wf = rsd.wireframe || {};
  const dim = Number(d.dim || state.dim || 2);
  const L = Number(d.boxSize || state.boxSize || 100);
  const nw = Number(wf.n || 0);
  const xr = arr(wf.xReal);
  const xs = arr(wf.xRsd);
  const uLos = arr(wf.uLos);
  const jr = arr(wf.jReal);
  const js = arr(wf.jRsd);
  const samps = arr(wf.gaussSamples);

  const renderer = k3dOpts.renderer || 'simple';
  const depthPeels = Number.isFinite(k3dOpts.depthPeels) ? k3dOpts.depthPeels : 4;
  const showVectors = k3dOpts.showVectors ?? true;
  const showVolume = k3dOpts.showVolume ?? false;
  const colorMap = buildK3DColorMap(densityViewOpts?.cmap || 'coolwarm');
  const objects = [];
  let axes = ['s₀ [Mpc/h]', 's₁ (LOS) [Mpc/h]', 'v_∥ [phase-space height]'];
  let grid = [0, 0, -0.3 * L, L, L, 0.3 * L];

  if (dim === 2 && nw >= 2 && xs.length >= nw * nw * 2) {
    const count = nw * nw;
    const [uLo, uHi] = percentileRange(uLos, 0.01, 0.99, [-0.1, 0.1]);
    const uSpan = Math.max(1e-6, uHi - uLo);
    const zHeight = L * 0.38;

    const rsdVerts = new Float32Array(count * 3);
    const realVerts = new Float32Array(count * 3);
    const attr = new Float32Array(count);

    for (let p = 0; p < count; p++) {
      const zVal = ((Number(uLos[p] || 0) - 0.5 * (uLo + uHi)) / uSpan) * zHeight;
      rsdVerts[p * 3] = ((Number(xs[p * 2]) % 1 + 1) % 1) * L;
      rsdVerts[p * 3 + 1] = ((Number(xs[p * 2 + 1]) % 1 + 1) % 1) * L;
      rsdVerts[p * 3 + 2] = zVal;

      realVerts[p * 3] = ((Number(xr[p * 2]) % 1 + 1) % 1) * L;
      realVerts[p * 3 + 1] = ((Number(xr[p * 2 + 1]) % 1 + 1) % 1) * L;
      realVerts[p * 3 + 2] = zVal;

      const jVal = Number(js[p] ?? 1);
      const jRealVal = Math.max(1e-3, Math.abs(Number(jr[p] ?? 1)));
      if (rsdViewOpts.sheetMode === 'real_vs_rsd_j') {
        attr[p] = jVal / jRealVal;
      } else {
        attr[p] = Math.log10(1 / Math.max(1e-4, Math.abs(jVal))) * (jVal <= 0 ? -1 : 1);
      }
    }

    const maxEdgeSq = (0.42 * L) ** 2;
    const distSq = (verts, a, b) => {
      const dx = verts[a * 3] - verts[b * 3];
      const dy = verts[a * 3 + 1] - verts[b * 3 + 1];
      return dx * dx + dy * dy;
    };

    const rsdTris = [];
    const rsdSegs = [];
    const realSegs = [];
    for (let i = 0; i + 1 < nw; i++) {
      for (let j = 0; j + 1 < nw; j++) {
        const a = i * nw + j;
        const b = (i + 1) * nw + j;
        const c = i * nw + (j + 1);
        const e = (i + 1) * nw + (j + 1);
        if (distSq(rsdVerts, a, b) < maxEdgeSq && distSq(rsdVerts, a, e) < maxEdgeSq && distSq(rsdVerts, b, e) < maxEdgeSq) {
          rsdTris.push(a, b, e);
        }
        if (distSq(rsdVerts, a, c) < maxEdgeSq && distSq(rsdVerts, a, e) < maxEdgeSq && distSq(rsdVerts, c, e) < maxEdgeSq) {
          rsdTris.push(a, e, c);
        }
        if (distSq(rsdVerts, a, b) < maxEdgeSq) rsdSegs.push(a, b);
        if (distSq(rsdVerts, a, c) < maxEdgeSq) rsdSegs.push(a, c);
        if (distSq(realVerts, a, b) < maxEdgeSq) realSegs.push(a, b);
        if (distSq(realVerts, a, c) < maxEdgeSq) realSegs.push(a, c);
      }
    }

    const cRange = percentileRange(attr, 0.02, 0.98, [-1.5, 1.5]);
    if (rsdTris.length) {
      objects.push({
        type: 'Mesh',
        name: 'Redshifted Phase-Space Sheet s(q)',
        visible: true,
        model_matrix: identityMatrix4(),
        vertices: { data: rsdVerts, shape: [count, 3] },
        indices: { data: new Uint32Array(rsdTris), shape: [rsdTris.length / 3, 3] },
        attribute: { data: attr, shape: [count] },
        color_map: colorMap,
        color_range: cRange,
        color: 0xffffff,
        opacity: 0.78,
        alpha_mode: 'blend',
        flat_shading: false,
        wireframe: false,
        side: 'double',
        roughness: 0.35,
        metalness: 0.05,
      });
    }

    if (rsdSegs.length) {
      objects.push({
        type: 'Lines',
        name: 'Redshifted Sheet Wireframe s(q)',
        visible: true,
        model_matrix: identityMatrix4(),
        vertices: { data: rsdVerts, shape: [count, 3] },
        indices: { data: new Uint32Array(rsdSegs), shape: [rsdSegs.length / 2, 2] },
        indices_type: 'segment',
        shader: 'simple',
        color: 0xd55e00,
        opacity: 0.55,
      });
    }

    if (rsdViewOpts.sheetMode === 'overlay' && realSegs.length) {
      objects.push({
        type: 'Lines',
        name: 'Real-Space Sheet Wireframe x(q)',
        visible: true,
        model_matrix: identityMatrix4(),
        vertices: { data: realVerts, shape: [count, 3] },
        indices: { data: new Uint32Array(realSegs), shape: [realSegs.length / 2, 2] },
        indices_type: 'segment',
        shader: 'simple',
        color: 0x0072b2,
        opacity: 0.45,
      });
    }

    if (showVectors) {
      const step = Math.max(1, Math.floor(nw / 18));
      const orig = [];
      const vecs = [];
      for (let i = 0; i < nw; i += step) {
        for (let j = 0; j < nw; j += step) {
          const p = i * nw + j;
          const dx = rsdVerts[p * 3] - realVerts[p * 3];
          const dy = rsdVerts[p * 3 + 1] - realVerts[p * 3 + 1];
          if (dx * dx + dy * dy < maxEdgeSq) {
            orig.push(realVerts[p * 3], realVerts[p * 3 + 1], realVerts[p * 3 + 2]);
            vecs.push(dx, dy, 0);
          }
        }
      }
      if (orig.length) {
        objects.push({
          type: 'Vectors',
          name: 'RSD Displacement Vectors s(q) − x(q)',
          visible: true,
          model_matrix: identityMatrix4(),
          origins: { data: new Float32Array(orig), shape: [orig.length / 3, 3] },
          vectors: { data: new Float32Array(vecs), shape: [vecs.length / 3, 3] },
          origin_color: 0x0072b2,
          head_color: 0xcc2529,
          use_head: true,
          head_size: 1.0,
          line_width: 0.003,
          labels: [],
          label_size: 1.0,
        });
      }
    }

    if (rsdViewOpts.showSamples && samps.length >= 2) {
      const nPts = Math.floor(samps.length / 2);
      const pts = new Float32Array(nPts * 3);
      for (let p = 0; p < nPts; p++) {
        pts[p * 3] = ((Number(samps[p * 2]) % 1 + 1) % 1) * L;
        pts[p * 3 + 1] = ((Number(samps[p * 2 + 1]) % 1 + 1) % 1) * L;
        pts[p * 3 + 2] = 0;
      }
      objects.push({
        type: 'Points',
        name: 'Simplex Velocity-Gradient Gaussian Samples',
        visible: true,
        model_matrix: identityMatrix4(),
        positions: { data: pts, shape: [nPts, 3] },
        color: 0x009e73,
        point_size: L * 0.01,
        shader: '3d',
        opacity: 0.72,
      });
    }
  } else if (dim === 3 && nw >= 2 && xs.length >= nw * nw * nw * 3) {
    axes = ['s₀ [Mpc/h]', 's₁ [Mpc/h]', 's₂ (LOS) [Mpc/h]'];
    grid = [0, 0, 0, L, L, L];
    const count = nw * nw * nw;
    const rsdVerts = new Float32Array(count * 3);
    const realVerts = new Float32Array(count * 3);
    const attr = new Float32Array(count);

    for (let p = 0; p < count; p++) {
      rsdVerts[p * 3] = ((Number(xs[p * 3]) % 1 + 1) % 1) * L;
      rsdVerts[p * 3 + 1] = ((Number(xs[p * 3 + 1]) % 1 + 1) % 1) * L;
      rsdVerts[p * 3 + 2] = ((Number(xs[p * 3 + 2]) % 1 + 1) % 1) * L;
      realVerts[p * 3] = ((Number(xr[p * 3]) % 1 + 1) % 1) * L;
      realVerts[p * 3 + 1] = ((Number(xr[p * 3 + 1]) % 1 + 1) % 1) * L;
      realVerts[p * 3 + 2] = ((Number(xr[p * 3 + 2]) % 1 + 1) % 1) * L;
      const jVal = Number(js[p] ?? 1);
      attr[p] = Math.log10(1 / Math.max(1e-4, Math.abs(jVal))) * (jVal <= 0 ? -1 : 1);
    }

    const idx = (i, j, k) => (i * nw + j) * nw + k;
    const maxEdgeSq = (0.42 * L) ** 2;
    const distSq = (verts, a, b) => {
      const dx = verts[a * 3] - verts[b * 3];
      const dy = verts[a * 3 + 1] - verts[b * 3 + 1];
      const dz = verts[a * 3 + 2] - verts[b * 3 + 2];
      return dx * dx + dy * dy + dz * dz;
    };

    const rsdTris = [];
    const rsdSegs = [];
    const mid = Math.floor(nw / 2);
    for (const planeAxis of [0, 1, 2]) {
      for (let u = 0; u + 1 < nw; u++) {
        for (let w = 0; w + 1 < nw; w++) {
          const getP = (du, dw) =>
            planeAxis === 0
              ? idx(mid, u + du, w + dw)
              : planeAxis === 1
                ? idx(u + du, mid, w + dw)
                : idx(u + du, w + dw, mid);
          const a = getP(0, 0), b = getP(1, 0), c = getP(0, 1), e = getP(1, 1);
          if (distSq(rsdVerts, a, b) < maxEdgeSq && distSq(rsdVerts, a, e) < maxEdgeSq && distSq(rsdVerts, b, e) < maxEdgeSq) {
            rsdTris.push(a, b, e);
          }
          if (distSq(rsdVerts, a, c) < maxEdgeSq && distSq(rsdVerts, a, e) < maxEdgeSq && distSq(rsdVerts, c, e) < maxEdgeSq) {
            rsdTris.push(a, e, c);
          }
          if (distSq(rsdVerts, a, b) < maxEdgeSq) rsdSegs.push(a, b);
          if (distSq(rsdVerts, a, c) < maxEdgeSq) rsdSegs.push(a, c);
        }
      }
    }

    const cRange = percentileRange(attr, 0.02, 0.98, [-1.5, 1.5]);
    if (rsdTris.length) {
      objects.push({
        type: 'Mesh',
        name: '3D Redshifted Sheet Slices s(q)',
        visible: true,
        model_matrix: identityMatrix4(),
        vertices: { data: rsdVerts, shape: [count, 3] },
        indices: { data: new Uint32Array(rsdTris), shape: [rsdTris.length / 3, 3] },
        attribute: { data: attr, shape: [count] },
        color_map: colorMap,
        color_range: cRange,
        color: 0xffffff,
        opacity: 0.72,
        alpha_mode: 'blend',
        flat_shading: false,
        wireframe: false,
        side: 'double',
        roughness: 0.35,
        metalness: 0.05,
      });
    }
    if (rsdSegs.length) {
      objects.push({
        type: 'Lines',
        name: '3D Redshifted Wireframe',
        visible: true,
        model_matrix: identityMatrix4(),
        vertices: { data: rsdVerts, shape: [count, 3] },
        indices: { data: new Uint32Array(rsdSegs), shape: [rsdSegs.length / 2, 2] },
        indices_type: 'segment',
        shader: 'simple',
        color: 0xd55e00,
        opacity: 0.45,
      });
    }

    if (showVectors) {
      const step = Math.max(1, Math.floor(nw / 6));
      const orig = [];
      const vecs = [];
      for (let i = 0; i < nw; i += step) {
        for (let j = 0; j < nw; j += step) {
          for (let k = 0; k < nw; k += step) {
            const p = idx(i, j, k);
            const dx = rsdVerts[p * 3] - realVerts[p * 3];
            const dy = rsdVerts[p * 3 + 1] - realVerts[p * 3 + 1];
            const dz = rsdVerts[p * 3 + 2] - realVerts[p * 3 + 2];
            if (dx * dx + dy * dy + dz * dz < maxEdgeSq) {
              orig.push(realVerts[p * 3], realVerts[p * 3 + 1], realVerts[p * 3 + 2]);
              vecs.push(dx, dy, dz);
            }
          }
        }
      }
      if (orig.length) {
        objects.push({
          type: 'Vectors',
          name: '3D Line-of-Sight Kaiser / FoG Shifts s − x',
          visible: true,
          model_matrix: identityMatrix4(),
          origins: { data: new Float32Array(orig), shape: [orig.length / 3, 3] },
          vectors: { data: new Float32Array(vecs), shape: [vecs.length / 3, 3] },
          origin_color: 0x0072b2,
          head_color: 0xcc2529,
          use_head: true,
          head_size: 1.0,
          line_width: 0.003,
          labels: [],
          label_size: 1.0,
        });
      }
    }

    if (rsdViewOpts.showSamples && samps.length >= 3) {
      const nPts = Math.floor(samps.length / 3);
      const pts = new Float32Array(nPts * 3);
      for (let p = 0; p < nPts; p++) {
        pts[p * 3] = ((Number(samps[p * 3]) % 1 + 1) % 1) * L;
        pts[p * 3 + 1] = ((Number(samps[p * 3 + 1]) % 1 + 1) % 1) * L;
        pts[p * 3 + 2] = ((Number(samps[p * 3 + 2]) % 1 + 1) % 1) * L;
      }
      objects.push({
        type: 'Points',
        name: '3D Simplex Velocity-Gradient Gaussian Samples',
        visible: true,
        model_matrix: identityMatrix4(),
        positions: { data: pts, shape: [nPts, 3] },
        color: 0x009e73,
        point_size: L * 0.012,
        shader: '3d',
        opacity: 0.68,
      });
    }

    if (showVolume) {
      const volRaw = arr(rsd.maps?.rsdGaussRho3d).length
        ? arr(rsd.maps.rsdGaussRho3d)
        : arr(rsd.maps?.rsdGaussRho);
      const mVol = Math.round(Math.cbrt(volRaw.length));
      if (mVol * mVol * mVol === volRaw.length && mVol >= 2) {
        const vol = new Float32Array(mVol * mVol * mVol);
        for (let i = 0; i < mVol; i++) {
          for (let j = 0; j < mVol; j++) {
            for (let k = 0; k < mVol; k++) {
              vol[(k * mVol + j) * mVol + i] = Math.log10(
                Math.max(1e-4, Number(volRaw[(i * mVol + j) * mVol + k]) || 1e-4),
              );
            }
          }
        }
        objects.push({
          type: 'Volume',
          name: '3D RSD Simplex Gaussian Density Volume',
          visible: true,
          model_matrix: boxVolumeMatrix4(L, L, L * 0.5),
          volume: { data: vol, shape: [mVol, mVol, mVol] },
          color_map: colorMap,
          opacity_function: {
            data: new Float32Array([-1.0, 0.0, 0.0, 0.05, 0.5, 0.35, 1.0, 0.85]),
            shape: [4, 2],
          },
          color_range: [-1.0, 1.0],
          samples: 512.0,
          alpha_coef: 32.0,
          gradient_step: 0.005,
          shadow: 'off',
          shadow_delay: 500,
          shadow_res: 128,
          interpolation: true,
          mask: { data: new Uint8Array(0), shape: [0] },
          mask_opacities: { data: new Float32Array(0), shape: [0] },
        });
      }
    }
  }

  const k3d = await createK3DViewport(hostEl, {
    axes,
    grid,
    renderer,
    depthPeels,
    name: `precrossedsheet-rsd-${dim}d`,
  });
  if (objects.length) {
    await k3d.load({ objects });
  }
  return k3d;
}
