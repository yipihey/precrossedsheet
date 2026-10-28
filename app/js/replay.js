// Disk-backed simulation archive, compact typed-array serializer, IndexedDB
// replay library, and Published-Replay adapter for long-running 2D/3D runs.

const DB_NAME = 'precrossedsheet-replay-db';
const DB_VERSION = 1;
const STORE_NAME = 'runs';
const MAX_SAVED_RUNS = 25;

function isTypedArray(v) {
  return ArrayBuffer.isView(v) && !(v instanceof DataView);
}

function roundNum(x, sig = 7) {
  const n = Number(x);
  if (!Number.isFinite(n)) return null;
  if (n === 0) return 0;
  return Number(n.toPrecision(sig));
}

function toCompactNumberArray(arrLike, sig = 6) {
  if (!arrLike) return [];
  const len = arrLike.length ?? 0;
  const out = new Array(len);
  for (let i = 0; i < len; i++) {
    out[i] = roundNum(arrLike[i], sig);
  }
  return out;
}

/**
 * Extract a compact 2D display slice (for 3D grids or very large 2D grids)
 * together with full-field cell statistics so that saved JSON files on disk
 * remain a few MiB even for 128^3 / 256^3 3D or 2048^2 2D simulations.
 */
function compactDensityField(rhoLike, dim, mesh) {
  if (!rhoLike || !rhoLike.length) return { slice: [], stats: null };
  const len = rhoLike.length;
  let empty = 0;
  let minimum = Infinity;
  let maximum = 0;
  let sum = 0;
  for (let i = 0; i < len; i++) {
    const v = Number(rhoLike[i]) || 0;
    sum += v;
    if (v <= 1e-6) empty++;
    if (v < minimum) minimum = v;
    if (v > maximum) maximum = v;
  }
  const stats = {
    totalCells: len,
    empty,
    minimum: Number.isFinite(minimum) ? roundNum(minimum, 6) : 0,
    maximum: roundNum(maximum, 6),
    mean: len > 0 ? roundNum(sum / len, 6) : 1,
  };

  if (dim === 1) {
    return { slice: toCompactNumberArray(rhoLike, 6), stats, sliceMesh: len, isSlice2d: false };
  }
  if (dim === 2) {
    const n = Math.round(Math.sqrt(len)) || mesh;
    const maxSide = 512;
    if (n <= maxSide) {
      return { slice: toCompactNumberArray(rhoLike, 5), stats, sliceMesh: n, isSlice2d: true };
    }
    const stride = Math.ceil(n / maxSide);
    const side = Math.floor(n / stride);
    const out = new Array(side * side);
    for (let i = 0; i < side; i++) {
      for (let j = 0; j < side; j++) {
        out[i * side + j] = roundNum(rhoLike[(i * stride) * n + (j * stride)], 5);
      }
    }
    return { slice: out, stats, sliceMesh: side, isSlice2d: true };
  }
  // dim === 3: extract central x2 = floor(n/2) plane (or if already a 2D slice, keep it)
  const n3 = Math.round(Math.cbrt(len));
  if (n3 * n3 * n3 === len) {
    const kMid = Math.floor(n3 / 2);
    const maxSide = 256;
    const stride = n3 > maxSide ? Math.ceil(n3 / maxSide) : 1;
    const side = Math.floor(n3 / stride);
    const out = new Array(side * side);
    for (let i = 0; i < side; i++) {
      for (let j = 0; j < side; j++) {
        const ii = i * stride;
        const jj = j * stride;
        out[i * side + j] = roundNum(rhoLike[(ii * n3 + jj) * n3 + kMid], 5);
      }
    }
    return { slice: out, stats, sliceMesh: side, isSlice2d: true };
  }
  const n2 = Math.round(Math.sqrt(len));
  if (n2 * n2 === len) {
    return { slice: toCompactNumberArray(rhoLike, 5), stats, sliceMesh: n2, isSlice2d: true };
  }
  return { slice: toCompactNumberArray(rhoLike.slice(0, Math.min(len, 65536)), 5), stats, sliceMesh: mesh, isSlice2d: true };
}

/**
 * Decimate vertex coordinate arrays (q, x, v) for large 2D/3D lattices to match
 * the display wireframe budget so disk exports stay small and fast.
 */
function compactVertexField(vecLike, dim, n) {
  if (!vecLike || !vecLike.length) return { data: [], sheetN: n };
  const nv = Math.floor(vecLike.length / dim);
  const expectedNv = Math.pow(n, dim);
  if (nv !== expectedNv) {
    const nInferred = dim === 1 ? nv : dim === 2 ? Math.round(Math.sqrt(nv)) : Math.round(Math.cbrt(nv));
    return { data: toCompactNumberArray(vecLike, 6), sheetN: nInferred || n };
  }
  const budget = dim === 1 ? 2048 : dim === 2 ? 16384 : 15625;
  if (nv <= budget) {
    return { data: toCompactNumberArray(vecLike, 6), sheetN: n };
  }
  const stride = Math.max(1, Math.ceil(Math.pow(nv / budget, 1 / dim)));
  const axis = [];
  for (let i = 0; i < n; i += stride) axis.push(i);
  if (axis[axis.length - 1] !== n - 1) axis.push(n - 1);
  const m = axis.length;
  const idx = (i, j = 0, k = 0) => (dim === 1 ? i : dim === 2 ? i * n + j : (i * n + j) * n + k);
  const out = [];
  if (dim === 1) {
    for (const i of axis) {
      out.push(roundNum(vecLike[i], 6));
    }
  } else if (dim === 2) {
    for (const i of axis) {
      for (const j of axis) {
        const id = idx(i, j);
        out.push(roundNum(vecLike[id * 2], 6), roundNum(vecLike[id * 2 + 1], 6));
      }
    }
  } else {
    for (const i of axis) {
      for (const j of axis) {
        for (const k of axis) {
          const id = idx(i, j, k);
          out.push(
            roundNum(vecLike[id * 3], 6),
            roundNum(vecLike[id * 3 + 1], 6),
            roundNum(vecLike[id * 3 + 2], 6),
          );
        }
      }
    }
  }
  return { data: out, sheetN: m };
}

function normalizeJsonValue(value) {
  if (value == null) return value;
  if (typeof value === 'number') return Number.isFinite(value) ? value : null;
  if (typeof value === 'string' || typeof value === 'boolean') return value;
  if (isTypedArray(value)) return toCompactNumberArray(value, 7);
  if (Array.isArray(value)) return value.map(normalizeJsonValue);
  if (typeof value === 'object') {
    const out = {};
    for (const [k, v] of Object.entries(value)) {
      out[k] = normalizeJsonValue(v);
    }
    return out;
  }
  return null;
}

/**
 * Build a Published-Replay-compatible dataset from a live laboratory result so
 * any saved simulation can be explored directly inside the "Published replay"
 * panel (and compared against the packaged DiscoDJ reference archive).
 */
export function buildPublishedReplayFromResult(result, state) {
  if (!result || result.published) return result?.published || null;
  const dim = Number(result.dim || state?.dim || 3);
  const n = Number(result.n || state?.n || 32);
  const mesh = Number(result.mesh || state?.mesh || n);
  const forceMesh = Number(result.forceMesh || state?.forceMesh || n);
  const boxSize = Number(state?.boxSize || result.request?.boxSize || 300);
  const smoothingFrac = Number(state?.smoothing ?? result.request?.smoothing ?? 0);
  const R = Math.round(smoothingFrac * boxSize * 100) / 100;
  const order = Number(state?.order || result.request?.order || 4);

  const pdfRoot = result.pdf || {};
  let minLog = Infinity;
  let maxLog = -Infinity;
  const scanCenters = (centers, dyHint) => {
    const arrC = isTypedArray(centers) || Array.isArray(centers) ? Array.from(centers, Number) : [];
    if (!arrC.length) return;
    const step = Number(dyHint) > 0 ? Number(dyHint) : arrC.length > 1 ? Math.abs(arrC[1] - arrC[0]) : 3.6 / 80;
    const lo = arrC[0] - step / 2;
    const hi = arrC[arrC.length - 1] + step / 2;
    if (Number.isFinite(lo) && lo < minLog) minLog = lo;
    if (Number.isFinite(hi) && hi > maxLog) maxLog = hi;
  };
  scanCenters(pdfRoot.logrho, pdfRoot.dy);
  scanCenters(pdfRoot.p1Logrho, pdfRoot.p1Dy);
  scanCenters(pdfRoot.pmLogrho, pdfRoot.pmDy);
  scanCenters(pdfRoot.cicLogrho, pdfRoot.cicDy);
  if (result.customSeries) {
    for (const cs of Object.values(result.customSeries)) {
      if (cs?.pdf) scanCenters(cs.pdf.logrho, cs.pdf.dy);
    }
  }
  const nBins = (isTypedArray(pdfRoot.logrho) || Array.isArray(pdfRoot.logrho)) && pdfRoot.logrho.length
    ? pdfRoot.logrho.length
    : 80;
  const defaultDy = Number.isFinite(minLog) && Number.isFinite(maxLog) && maxLog > minLog
    ? (maxLog - minLog) / nBins
    : Number(pdfRoot.dy) > 0
      ? Number(pdfRoot.dy)
      : 3.6 / nBins;
  const defaultLogrho = Number.isFinite(minLog) && Number.isFinite(maxLog) && maxLog > minLog
    ? Array.from({ length: nBins }, (_, i) => roundNum(minLog + (i + 0.5) * defaultDy, 6))
    : isTypedArray(pdfRoot.logrho) || Array.isArray(pdfRoot.logrho)
      ? Array.from(pdfRoot.logrho, Number)
      : [];

  const edgesFromCenters = (centers, dy) => {
    if (!centers || !centers.length) return [];
    const half = (dy || 3.6 / 80) / 2;
    const edges = centers.map((c) => roundNum(Number(c) - half, 6));
    edges.push(roundNum(Number(centers[centers.length - 1]) + half, 6));
    return edges;
  };

  const binEdges = edgesFromCenters(defaultLogrho, defaultDy);
  const qProbs = pdfRoot.summary?.probabilities
    ? Array.from(pdfRoot.summary.probabilities, Number)
    : [0.001, 0.01, 0.1, 0.5, 0.9, 0.99, 0.999];

  const modelLabels = {
    '1lpt': '1LPT (Zel’dovich)',
    '2lpt': '2LPT',
    '3lpt': '3LPT',
    '4lpt': '4LPT',
    'nbody': `PM N-body (${forceMesh}^${dim} force, ${state?.steps || result.request?.steps || 64} steps)`,
  };
  const estLabels = {
    'sheet': 'P₀ exact simplicial sheet',
    'p1': 'P₁ vertex-Jacobian sheet',
    'ngp': 'NGP particles',
    'cic': 'CIC particles',
    'tsc': 'TSC particles',
  };

  const resampleLogPdf = (srcCenters, srcPdf, targetCenters) => {
    const sc = Array.from(srcCenters || [], Number);
    const sp = Array.from(srcPdf || [], Number);
    if (!sc.length || !sp.length) return [];
    if (!targetCenters.length || (sc.length === targetCenters.length && Math.abs(sc[0] - targetCenters[0]) < 1e-4 && Math.abs(sc[sc.length - 1] - targetCenters[targetCenters.length - 1]) < 1e-4)) {
      return toCompactNumberArray(sp, 6);
    }
    return targetCenters.map((y) => {
      if (y < sc[0] || y > sc[sc.length - 1]) return 0;
      let lo = 0;
      let hi = sc.length - 1;
      while (hi - lo > 1) {
        const mid = (lo + hi) >> 1;
        if (sc[mid] <= y) lo = mid;
        else hi = mid;
      }
      const span = sc[hi] - sc[lo];
      const t = span > 1e-12 ? (y - sc[lo]) / span : 0;
      return roundNum((1 - t) * sp[lo] + t * sp[hi], 6);
    });
  };

  const runs = [];
  const seenModelEst = new Set();
  const guard = result.guard || {};

  const addRunEntry = (modelId, estId, pdfObj) => {
    if (!pdfObj || (!pdfObj.volume?.length && !pdfObj.mass?.length)) return;
    const key = `${modelId}:${estId}`;
    if (seenModelEst.has(key)) return;
    seenModelEst.add(key);

    const centers = (pdfObj.logrho && pdfObj.logrho.length) ? pdfObj.logrho : defaultLogrho;
    const volPdf = resampleLogPdf(centers, pdfObj.volume, defaultLogrho);
    const massPdf = resampleLogPdf(centers, pdfObj.mass, defaultLogrho);
    const sum = pdfObj.summary || {};
    const isPm = modelId === 'nbody';
    const finalCrossed = isPm ? guard.pmFinalInvertedMass : guard.lptFinalInvertedMass;

    for (const pairedVal of [true, false]) {
      runs.push({
        model: modelId,
        estimator: estId,
        paired: pairedVal,
        crossing: {
          ever_mass_fraction: finalCrossed ?? null,
          final_mass_fraction: finalCrossed ?? null,
          belongs_to: `${modelLabels[modelId] || modelId.toUpperCase()} (${n}^${dim} vertices)`,
        },
        radii: [
          {
            R,
            variance: roundNum(sum.variance ?? pdfObj.variance, 6),
            S3: roundNum(sum.S3, 6),
            S4: roundNum(sum.S4, 6),
            volume: {
              pdf: volPdf,
              quantiles_rho: toCompactNumberArray(sum.volumeQuantiles || [], 6),
              captured_probability: roundNum(pdfObj.capturedVolume, 6),
            },
            mass: {
              pdf: massPdf,
              quantiles_rho: toCompactNumberArray(sum.massQuantiles || [], 6),
              captured_probability: roundNum(pdfObj.capturedMass, 6),
            },
          },
        ],
      });
    }
  };

  // Add all series from customSeries first
  const custom = result.customSeries || {};
  for (const [seriesKey, cs] of Object.entries(custom)) {
    const [rawTraj, rawEst] = seriesKey.split(':');
    const modelId = rawTraj === 'pm' ? 'nbody' : rawTraj;
    const estId = rawEst === 'p0' ? 'sheet' : rawEst;
    if (cs?.pdf) addRunEntry(modelId, estId, cs.pdf);
  }

  // Ensure legacy primary fields are also registered
  const ordModel = `${order}lpt`;
  addRunEntry(ordModel, 'sheet', {
    logrho: pdfRoot.logrho,
    volume: pdfRoot.volume,
    mass: pdfRoot.mass,
    variance: pdfRoot.variance,
    capturedVolume: pdfRoot.capturedVolume,
    capturedMass: pdfRoot.capturedMass,
    summary: pdfRoot.summary,
  });
  addRunEntry(ordModel, 'p1', {
    logrho: pdfRoot.p1Logrho || pdfRoot.logrho,
    volume: pdfRoot.p1Volume,
    mass: pdfRoot.p1Mass,
    variance: pdfRoot.p1Variance,
    capturedVolume: pdfRoot.p1CapturedVolume,
    capturedMass: pdfRoot.p1CapturedMass,
    summary: pdfRoot.p1Summary,
  });
  addRunEntry('nbody', 'sheet', {
    logrho: pdfRoot.pmLogrho || pdfRoot.logrho,
    volume: pdfRoot.pmVolume,
    mass: pdfRoot.pmMass,
    variance: pdfRoot.pmVariance,
    capturedVolume: pdfRoot.pmCapturedVolume,
    capturedMass: pdfRoot.pmCapturedMass,
    summary: pdfRoot.pmSummary,
  });
  addRunEntry(ordModel, 'cic', {
    logrho: pdfRoot.cicLogrho || pdfRoot.logrho,
    volume: pdfRoot.cicVolume,
    mass: pdfRoot.cicMass,
    variance: pdfRoot.cicVariance,
    capturedVolume: pdfRoot.cicCapturedVolume,
    capturedMass: pdfRoot.cicMass ? pdfRoot.cicCapturedMass : null,
    summary: pdfRoot.cicSummary,
  });

  const presentModels = [...new Set(runs.map((r) => r.model))];
  const presentEsts = [...new Set(runs.map((r) => r.estimator))];

  return {
    title: `Saved browser simulation replay (${dim}D, N=${n}^${dim}, mesh=${mesh}^${dim}, a=${roundNum(result.a ?? state?.scaleFactor, 4)})`,
    isSavedRunReplay: true,
    bin_edges_log10rho: binEdges,
    quantile_probabilities: qProbs,
    smoothing_radii_mpc_h: [R],
    models: presentModels.map((id) => ({ id, label: modelLabels[id] || id.toUpperCase() })),
    estimators: presentEsts.map((id) => ({ id, label: estLabels[id] || id.toUpperCase() })),
    runs,
    provenance: {
      upstream_repo: 'Interactive Browser Laboratory (hcc_core Wasm / WebGPU)',
      upstream_commit: state?.backend || 'cpu',
      conditions: {
        dimension: dim,
        verticesPerAxis: n,
        observationMesh: mesh,
        forceMesh,
        steps: state?.steps ?? result.request?.steps,
        boxSizeMpcH: boxSize,
        scaleFactorA: roundNum(result.a ?? state?.scaleFactor, 5),
        growthFactorD: roundNum(result.D, 5),
        omegaM: state?.omegaM,
        omegaB: state?.omegaB,
        h: state?.h,
        ns: state?.ns,
        sigma8: state?.sigma8,
        initialSmoothingMpcH: state?.initialSmoothing,
        observationSmoothingMpcH: R,
        seed: state?.seed,
        ic: state?.ic,
        transfer: state?.transfer,
      },
      restrictions: [
        R === 0
          ? 'Direct simplex PDF evaluation (R = 0): P₀ and P₁ sheet PDFs are integrated directly over the Lagrangian simplices without grid deposition.'
          : `Smoothed on ${mesh}^${dim} observation mesh with Gaussian kernel R = ${R} Mpc/h (R/L = ${smoothingFrac}).`,
        `PM trajectory evolved for ${state?.steps ?? result.request?.steps ?? 64} √a-spaced KDK steps on a ${forceMesh}^${dim} deconvolved TSC force grid.`,
      ],
      field_meanings: {
        pdf: R === 0
          ? 'Unfiltered P₀ and P₁ sheet PDFs are evaluated directly from the simplex Jacobians 1/|J_T|; particle estimators use their Eulerian grid.'
          : `Density PDFs evaluated at smoothing scale R = ${R} Mpc/h.`,
      },
    },
  };
}

/**
 * Create a compact, self-contained experiment & replay manifest safe for
 * JSON.stringify and disk storage even for 128^3 / 256^3 3D or 2048^2 2D runs.
 */
export function buildCompactManifest({ state, request, result, execution, build, runMilliseconds }) {
  const dim = Number(result?.dim || state?.dim || 2);
  const n = Number(result?.n || state?.n || 32);
  const mesh = Number(result?.mesh || state?.mesh || 32);

  if (result?.published) {
    return {
      manifestFormat: 'precrossedsheet-experiment',
      schemaVersion: state?.schemaVersion || 3,
      kind: 'archiveReplay',
      createdAt: new Date().toISOString(),
      state: { ...state },
      request: null,
      execution: null,
      build: build || null,
      run: {
        durationMilliseconds: runMilliseconds ?? null,
        guard: null,
        epoch: state?.epoch,
        scaleFactor: state?.scaleFactor,
        growthFactor: null,
        shellCrossingReference: null,
      },
      source: result.published?.provenance || null,
      result: { published: normalizeJsonValue(result.published) },
    };
  }

  const qComp = compactVertexField(result?.q, dim, n);
  const xComp = compactVertexField(result?.x, dim, n);
  const vComp = compactVertexField(result?.v, dim, n);
  const pmXComp = compactVertexField(result?.pmX, dim, n);
  const pmVComp = compactVertexField(result?.pmV, dim, n);

  const rhoComp = compactDensityField(result?.rho, dim, mesh);
  const rhoP1Comp = compactDensityField(result?.rhoP1, dim, mesh);
  const pmRhoComp = compactDensityField(result?.pmRho, dim, mesh);
  const cicRhoComp = compactDensityField(result?.cicRho, dim, mesh);
  const tscRhoComp = compactDensityField(result?.tscRho, dim, mesh);

  const customSeriesOut = {};
  const densitySliceStats = {};
  if (rhoComp.stats) densitySliceStats['legacy_sheet_p0'] = rhoComp.stats;
  if (rhoP1Comp.stats) densitySliceStats['legacy_sheet_p1'] = rhoP1Comp.stats;
  if (pmRhoComp.stats) densitySliceStats['legacy_pm_p0'] = pmRhoComp.stats;
  if (cicRhoComp.stats) densitySliceStats['legacy_cic'] = cicRhoComp.stats;
  if (tscRhoComp.stats) densitySliceStats['legacy_tsc'] = tscRhoComp.stats;

  if (result?.customSeries) {
    for (const [k, entry] of Object.entries(result.customSeries)) {
      if (!entry) continue;
      const cRho = compactDensityField(entry.rho, dim, mesh);
      if (cRho.stats) densitySliceStats[`cs_${k.replace(':', '_')}`] = cRho.stats;
      customSeriesOut[k] = {
        massMean: roundNum(entry.massMean, 7),
        rho: cRho.slice,
        sliceMesh: cRho.sliceMesh,
        power: toCompactNumberArray(entry.power, 6),
        directShell: toCompactNumberArray(entry.directShell, 6),
        variance: normalizeJsonValue(entry.variance),
        pdf: normalizeJsonValue(entry.pdf),
      };
    }
  }

  let rsdOut = null;
  if (result?.rsd) {
    const rsdMesh = Number(result.rsd.mesh || mesh);
    const maps = result.rsd.maps || {};
    rsdOut = {
      losAxis: result.rsd.losAxis,
      mesh: rsdMesh,
      summary: normalizeJsonValue(result.rsd.summary),
      wireframe: normalizeJsonValue(result.rsd.wireframe),
      maps: {
        realRho: compactDensityField(maps.realRho, dim, rsdMesh).slice,
        rsdSheetRho: compactDensityField(maps.rsdSheetRho, dim, rsdMesh).slice,
        rsdGaussRho: compactDensityField(maps.rsdGaussRho, dim, rsdMesh).slice,
        rsdSampledRho: compactDensityField(maps.rsdSampledRho, dim, rsdMesh).slice,
      },
      spectra: normalizeJsonValue(result.rsd.spectra),
      pdf: normalizeJsonValue(result.rsd.pdf),
    };
  }

  const publishedReplay = buildPublishedReplayFromResult(result, state);

  const compactResult = {
    dim,
    n,
    sheetN: xComp.sheetN,
    mesh,
    sliceMesh: rhoComp.sliceMesh,
    forceMesh: result?.forceMesh ?? state?.forceMesh,
    a: result?.a,
    aStart: result?.aStart,
    redshift: result?.redshift,
    D: result?.D,
    growthAtOne: result?.growthAtOne,
    shellCrossing: result?.shellCrossing,
    aShellCrossing: result?.aShellCrossing,
    crossingReference: result?.crossingReference,
    request: normalizeJsonValue(result?.request || request),
    guard: normalizeJsonValue(result?.guard),
    diagnostics: normalizeJsonValue(result?.diagnostics),
    q: qComp.data,
    x: xComp.data,
    v: vComp.data,
    pmX: pmXComp.data,
    pmV: pmVComp.data,
    rho: rhoComp.slice,
    rhoP1: rhoP1Comp.slice,
    pmRho: pmRhoComp.slice,
    cicRho: cicRhoComp.slice,
    tscRho: tscRhoComp.slice,
    densitySliceStats,
    spectra: normalizeJsonValue(result?.spectra),
    varianceScale: normalizeJsonValue(result?.varianceScale),
    pdf: normalizeJsonValue(result?.pdf),
    rsd: rsdOut,
    gradient: normalizeJsonValue(result?.gradient),
    customSeries: customSeriesOut,
    execution: normalizeJsonValue(result?.execution || execution),
    publishedReplay,
  };

  const label = `${dim}D · N=${n}^${dim} · mesh=${mesh}^${dim} · force=${compactResult.forceMesh}^${dim} · ${state?.ic === 'gaussian' ? `ΛCDM (${state?.transfer || 'smooth'}, seed ${state?.seed})` : 'waves'} · a=${roundNum(result?.a ?? state?.scaleFactor, 4)}`;

  return {
    id: `run-${Date.now()}-${Math.random().toString(36).slice(2, 7)}`,
    label,
    manifestFormat: 'precrossedsheet-experiment',
    schemaVersion: state?.schemaVersion || 3,
    kind: 'liveNumericalRun',
    createdAt: new Date().toISOString(),
    state: { ...state },
    request: normalizeJsonValue(result?.request || request),
    execution: normalizeJsonValue(result?.execution || execution || null),
    build: build || null,
    run: {
      durationMilliseconds: runMilliseconds ?? null,
      guard: normalizeJsonValue(result?.guard || null),
      epoch: result?.request?.epoch ?? state?.epoch,
      scaleFactor: result?.a ?? state?.scaleFactor,
      growthFactor: result?.D ?? null,
      shellCrossingReference: result?.shellCrossing ?? null,
    },
    source: null,
    result: compactResult,
  };
}

function openReplayDb() {
  if (typeof indexedDB === 'undefined') return Promise.resolve(null);
  return new Promise((resolve) => {
    try {
      const req = indexedDB.open(DB_NAME, DB_VERSION);
      req.onupgradeneeded = () => {
        const db = req.result;
        if (!db.objectStoreNames.contains(STORE_NAME)) {
          const store = db.createObjectStore(STORE_NAME, { keyPath: 'id' });
          store.createIndex('createdAt', 'createdAt', { unique: false });
        }
      };
      req.onsuccess = () => resolve(req.result);
      req.onerror = () => resolve(null);
    } catch (_) {
      resolve(null);
    }
  });
}

export async function saveReplayToLibrary(manifest) {
  const db = await openReplayDb();
  if (!db || !manifest) return false;
  const entry = {
    ...manifest,
    id: manifest.id || `run-${Date.now()}-${Math.random().toString(36).slice(2, 7)}`,
    createdAt: manifest.createdAt || new Date().toISOString(),
  };
  return new Promise((resolve) => {
    try {
      const tx = db.transaction(STORE_NAME, 'readwrite');
      const store = tx.objectStore(STORE_NAME);
      store.put(entry);
      const getAllReq = store.getAll();
      getAllReq.onsuccess = () => {
        const items = Array.isArray(getAllReq.result) ? getAllReq.result : [];
        if (items.length > MAX_SAVED_RUNS) {
          items.sort((a, b) => String(b.createdAt).localeCompare(String(a.createdAt)));
          for (const old of items.slice(MAX_SAVED_RUNS)) {
            store.delete(old.id);
          }
        }
      };
      tx.oncomplete = () => resolve(true);
      tx.onerror = () => resolve(false);
    } catch (_) {
      resolve(false);
    }
  });
}

export async function listReplayLibrary() {
  const db = await openReplayDb();
  if (!db) return [];
  return new Promise((resolve) => {
    try {
      const tx = db.transaction(STORE_NAME, 'readonly');
      const store = tx.objectStore(STORE_NAME);
      const req = store.getAll();
      req.onsuccess = () => {
        const items = Array.isArray(req.result) ? req.result : [];
        items.sort((a, b) => String(b.createdAt).localeCompare(String(a.createdAt)));
        resolve(items);
      };
      req.onerror = () => resolve([]);
    } catch (_) {
      resolve([]);
    }
  });
}

export async function deleteReplayFromLibrary(id) {
  const db = await openReplayDb();
  if (!db || !id) return false;
  return new Promise((resolve) => {
    try {
      const tx = db.transaction(STORE_NAME, 'readwrite');
      tx.objectStore(STORE_NAME).delete(id);
      tx.oncomplete = () => resolve(true);
      tx.onerror = () => resolve(false);
    } catch (_) {
      resolve(false);
    }
  });
}
