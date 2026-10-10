// A persistent worker owns the Wasm facade and (when requested) its GPU device.
// Jobs are serialized. A newer request replaces the queued request and asks an
// asynchronous GPU run to stop between submitted batches.
import init, * as wasmApi from '../pkg/hcc_core.js';

let wasmReady;
let gpuContext = null;
let gpuContextNeedsReinit = false;
let active = null;
let queued = null;
let shuttingDown = false;

const send = (message) => self.postMessage(message);
const yieldToMessages = () => new Promise((resolve) => setTimeout(resolve, 0));
function phase(task, name, message, cancellable = false) {
  send({ type: 'phase', requestId: task.requestId, phase: name, message, cancellable });
}
function cancelled(task) { return task.cancelled || task.controller?.signal.aborted || shuttingDown; }

self.onmessage = ({ data }) => {
  if (!data) return;
  if (data.type === 'shutdown') {
    shuttingDown = true;
    queued = null;
    if (active) { active.cancelled = true; active.controller?.abort(); return; }
    disposeGpu().finally(() => send({ type: 'shutdown-ack' }));
    return;
  }
  if (data.type === 'cancel') {
    if (active && (data.requestId == null || data.requestId === active.requestId)) {
      active.cancelled = true;
      active.controller?.abort();
    }
    return;
  }
  if (data.type !== 'run' || shuttingDown) return;
  if (active) {
    queued = data;
    active.cancelled = true;
    active.controller?.abort();
    return;
  }
  start(data);
};

function start(data) {
  if (shuttingDown) return;
  const task = { requestId: data.requestId, key: data.key, state: data.state, policy: data.policy || { requested: 'cpu' }, cancelled: false, controller: null };
  active = task;
  void execute(task).then((result) => {
    if (cancelled(task)) send({ type: 'cancelled', requestId: task.requestId });
    else send({ type: 'result', requestId: task.requestId, key: task.key, ...result });
  }).catch((error) => {
    if (cancelled(task)) send({ type: 'cancelled', requestId: task.requestId });
    else send({ type: 'result', requestId: task.requestId, key: task.key, error: String(error?.message || error), execution: task.execution || null });
  }).finally(async () => {
    active = null;
    if (shuttingDown) {
      await disposeGpu();
      send({ type: 'shutdown-ack' });
      return;
    }
    const next = queued;
    queued = null;
    if (next) start(next);
  });
}

async function execute(task) {
  const started = performance.now();
  if (!wasmReady) wasmReady = init();
  try { await wasmReady; } catch (error) { wasmReady = null; throw error; }
  const wasm = wasmApi;
  if (cancelled(task)) return {};

  phase(task, 'preparing', 'Preparing trajectories, tangents, and exact observations.');
  const preparationStarted = performance.now();
  const preparedCtor = wasm.PreparedLabRun;
  if (typeof preparedCtor !== 'function') {
    // Older cached deployments can still run the CPU reference until the new
    // split Wasm facade has been rebuilt and published.
    phase(task, 'direct', 'Running the CPU reference through the compatibility facade.', false);
    const raw = wasm.lab_run(JSON.stringify(task.state));
    const result = typeof raw === 'string' ? JSON.parse(raw) : raw;
    if (result?.error) throw new Error(result.error);
    const execution = cpuExecution(task.policy.requested, 'Prepared GPU boundary is unavailable in this Wasm build.');
    result.execution = execution;
    task.execution = execution;
    return { result, durationMilliseconds: performance.now() - started, execution };
  }

  const prepared = new preparedCtor(JSON.stringify(task.state));
  try {
  let metadata = parseJson(prepared.metadata(), 'prepared metadata');
  if (!Number.isInteger(metadata.modeCount) || !Number.isInteger(metadata.dim) || !Number.isInteger(metadata.n)) {
    throw new Error('Wasm prepared metadata is missing the dimension, grid, or mode count.');
  }
  const preparationMilliseconds = performance.now() - preparationStarted;
  // Give cancel/run messages queued during synchronous Rust preparation a
  // chance to run before selecting the next expensive stage.
  await yieldToMessages();
  if (cancelled(task)) return {};

  let channels;
  let execMeta;
  const stageTimes = { preparationMilliseconds, directMilliseconds: 0, finiteDifferenceMilliseconds: null, cpuRecoveryMilliseconds: 0, finalizationMilliseconds: 0 };
  const directStarted = performance.now();
  const requested = task.policy.requested || 'cpu';
  let attemptedContext = null;
  let attemptedTimings = null;
  let attemptedDiagnostics = null;
  let attemptedValidation = null;
  if (requested === 'auto') {
    phase(task, 'direct', 'Automatic backend policy is using the f64 CPU reference.', false);
    channels = copyF64(prepared.direct_cpu());
    execMeta = cpuExecution(requested, 'Automatic selection is held on CPU until a device and workload pass the release validation fixtures.');
  } else if (requested === 'gpu' && metadata.dim === 1) {
    phase(task, 'direct', 'The 1D direct-mode reference is evaluated on the CPU.', false);
    channels = copyF64(prepared.direct_cpu());
    execMeta = cpuExecution(requested, 'The GPU direct-mode kernel supports 2D and 3D; 1D remains on the CPU reference.');
  } else if (requested === 'gpu') {
    try {
      task.controller = new AbortController();
      phase(task, 'direct', 'Checking the worker GPU device and evaluating fixed-mass Fourier modes.', true);
      const forcedFailure = task.policy?.forceGpuFailure;
      if (forcedFailure === 'missing-api') throw forcedGpuError(forcedFailure);
      const context = await getGpuContext(task);
      attemptedContext = context;
      if (['device-destroy','destroy-device','device-loss'].includes(forcedFailure)) { context.device.destroy(); throw forcedGpuError(forcedFailure); }
      if (['limits','shader'].includes(forcedFailure)) throw forcedGpuError(forcedFailure);
      if (cancelled(task)) return {};
      const { runDirect } = await import('./gpu/direct.js');
      const input = {
        dim: metadata.dim,
        n: metadata.n,
        modes: copyI32(prepared.mode_records()),
        lptPsi: copyF32(prepared.lpt_displacement()),
        lptDirection: copyF32(prepared.lpt_direction()),
        pmPsi: copyF32(prepared.pm_displacement()),
        pmDirection: copyF32(prepared.pm_direction()),
      };
      let gpu = await runDirect(context.device, input, {
        signal: task.controller.signal,
        shouldCancel: () => cancelled(task),
        maxGpuBytes: 512 * 1024 * 1024,
        onProgress: (progress) => send({ type: 'progress', requestId: task.requestId, phase: 'direct', completed: progress.batch, total: progress.batches }),
      });
      attemptedTimings = gpu?.timings || null;
      attemptedDiagnostics = gpu?.diagnostics || null;
      if (cancelled(task)) return {};
      if (['no-result','no-result-gpu'].includes(forcedFailure)) { gpu = null; throw forcedGpuError(forcedFailure); }
      if (forcedFailure === 'readback') throw forcedGpuError(forcedFailure);
      channels = copyF64(gpu.channels);
      if (['nonfinite','channels-nan','channel-nan','nan'].includes(forcedFailure) && channels.length) channels[0] = Number.NaN;
      if (channels.length !== metadata.modeCount * 8 || !allFinite(channels)) throw forcedFailure ? forcedGpuError(forcedFailure) : new Error('GPU direct-mode channels were malformed or non-finite.');
      execMeta = {
        requestedBackend: requested, executedBackend: 'gpu', fallbackReason: null,
        precision: 'f32 GPU inputs and kernel with f64 CPU finalization',
        channelProvenance: { lpt: 'gpu-f32', pm: 'gpu-f32', finiteDifference: 'cpu-f64' },
        modeProvenance: metadata.modes.map((mode) => ({ integerMode: mode, lpt: 'gpu-f32', pm: 'gpu-f32' })),
        device: deviceMetadata(context),
        validation: context.validation,
        diagnostics: gpu.diagnostics || null, timings: gpu.timings || null,
      };
    } catch (error) {
      if (cancelled(task)) return {};
      attemptedValidation = error?.gpuValidation || attemptedContext?.validation || null;
      if (attemptedContext && attemptedContext !== gpuContext) await disposeContext(attemptedContext);
      await disposeGpu();
      const fallbackStarted = performance.now();
      channels = copyF64(prepared.direct_cpu());
      stageTimes.directMilliseconds += performance.now() - fallbackStarted;
      execMeta = cpuExecution(requested, `GPU unavailable or failed: ${String(error?.message || error)}`);
      execMeta.gpuAttempt = { device: deviceMetadata(attemptedContext), validation: attemptedValidation, diagnostics: attemptedDiagnostics, timings: attemptedTimings };
    }
  } else {
    phase(task, 'direct', 'Evaluating direct modes with the f64 CPU reference.', false);
    channels = copyF64(prepared.direct_cpu());
    execMeta = cpuExecution(requested, null);
  }
  stageTimes.directMilliseconds = performance.now() - directStarted;
  if (cancelled(task)) return {};
  if (channels.length !== metadata.modeCount * 8 || !allFinite(channels)) throw new Error('Direct-mode channels were malformed or non-finite.');

  phase(task, 'finite-difference', 'Evaluating the f64 CPU finite-difference check.', false);
  const fdStarted = performance.now();
  if (typeof prepared.finite_difference_cpu === 'function') {
    const fd = copyF64(prepared.finite_difference_cpu());
    if (fd.length !== metadata.modeCount || !allFinite(fd)) throw new Error('CPU finite-difference channels were malformed or non-finite.');
    stageTimes.finiteDifferenceMilliseconds = performance.now() - fdStarted;
  }
  if (execMeta.executedBackend === 'gpu') {
    execMeta.diagnostics = { ...(execMeta.diagnostics || {}), cpuFiniteDifferencePrecomputed: stageTimes.finiteDifferenceMilliseconds != null };
  }
  execMeta.timings = { ...(execMeta.timings || {}), stages: stageTimes };

  phase(task, 'finalizing', 'Finalizing f64 observables and the run manifest.');
  const finalizationStarted = performance.now();
  let result = finish(prepared, channels, execMeta);
  if (execMeta.executedBackend === 'gpu') {
    const failedModes = failedDerivativeModes(result, metadata.modeCount);
    if (failedModes == null) {
      const gpuAttempt = execMeta;
      const fallbackStarted = performance.now();
      channels = copyF64(prepared.direct_cpu());
      stageTimes.directMilliseconds += performance.now() - fallbackStarted;
      execMeta = cpuExecution(requested, 'GPU run omitted the required finite-difference comparison.');
      execMeta.gpuAttempt = gpuAttempt;
      execMeta.timings = { stages: stageTimes };
      result = finish(prepared, channels, execMeta);
    } else if (failedModes.length) {
      const recoveryStarted = performance.now();
      const failedIndices = Uint32Array.from(failedModes);
      if (typeof prepared.direct_lpt_cpu_modes !== 'function') {
        const gpuAttempt = execMeta;
        const fallbackStarted = performance.now();
        channels = copyF64(prepared.direct_cpu());
        stageTimes.directMilliseconds += performance.now() - fallbackStarted;
        execMeta = cpuExecution(requested, 'Selective weak-mode recovery is unavailable in this Wasm build.');
        execMeta.gpuAttempt = gpuAttempt;
        execMeta.timings = { stages: stageTimes };
        result = finish(prepared, channels, execMeta);
      } else {
        phase(task, 'direct-recovery', 'Recovering weak LPT derivative modes with the f64 CPU reference.', false);
        const recovered = copyF64(prepared.direct_lpt_cpu_modes(failedIndices));
        if (recovered.length !== failedModes.length * 4 || !allFinite(recovered)) {
          throw new Error('Selective CPU recovery returned malformed direct-mode channels.');
        }
        for (let ri = 0; ri < failedModes.length; ri++) {
          const modeIndex = failedModes[ri];
          channels.set(recovered.subarray(4 * ri, 4 * ri + 4), modeIndex * 8);
          execMeta.modeProvenance[modeIndex].lpt = 'cpu-f64';
        }
        stageTimes.cpuRecoveryMilliseconds = performance.now() - recoveryStarted;
        const labels = failedModes.map((i) => (metadata.modes?.[i] || [i]).join(','));
        execMeta = {
          ...execMeta,
          executedBackend: 'hybrid',
          fallbackReason: `Weak LPT mode derivative errors exceeded 1e-5; those ${failedModes.length} LPT mode(s) used the f64 CPU reference (${labels.join('; ')}).`,
          precision: 'f32 GPU modes with selective f64 CPU LPT recovery and f64 CPU finalization',
          channelProvenance: { lpt: 'mixed-cpu-f64-and-gpu-f32', pm: 'gpu-f32', finiteDifference: 'cpu-f64' },
          diagnostics: { ...(execMeta.diagnostics || {}), cpuRecoveredLptModes: failedModes.map((i) => metadata.modes?.[i] || [i]), derivativeGate: 'relativeError <= 1e-5 per mode' },
        };
        result = finish(prepared, channels, execMeta);
        const unrecoveredModes = failedDerivativeModes(result, metadata.modeCount);
        if (unrecoveredModes == null || unrecoveredModes.length) {
          const hybridAttempt = execMeta;
          const fallbackStarted = performance.now();
          channels = copyF64(prepared.direct_cpu());
          stageTimes.directMilliseconds += performance.now() - fallbackStarted;
          execMeta = cpuExecution(requested, 'The selective f64 LPT recovery did not satisfy the per-mode finite-difference gate; the complete f64 CPU reference was used.');
          execMeta.gpuAttempt = hybridAttempt;
          execMeta.timings = { stages: stageTimes };
          result = finish(prepared, channels, execMeta);
        }
      }
    }
  }
  if (result?.error) throw new Error(result.error);
  stageTimes.finalizationMilliseconds = performance.now() - finalizationStarted;
  if (result.execution) result.execution.timings = { ...(result.execution.timings || {}), stages: stageTimes };
  if (result.execution?.gpuAttempt?.timings) result.execution.gpuAttempt.timings.stages = stageTimes;
  task.execution = result.execution || execMeta;
  return { result, durationMilliseconds: performance.now() - started, execution: task.execution };
  } finally {
    prepared.free?.();
  }
}

async function getGpuContext(task) {
  if (gpuContext && !gpuContextNeedsReinit) return gpuContext;
  if (gpuContext) await disposeGpu();
  const { createGpuContext } = await import('./gpu/context.js');
  let ownedContext = null;
  const validationStarted = performance.now();
  const context = await createGpuContext({ onDeviceLost: () => {
    gpuContextNeedsReinit = true;
    if (gpuContext === ownedContext) gpuContext = null;
  } });
  ownedContext = context;
  try {
    const { runGpuSelfTests } = await import('./gpu/selftest.js');
    const validation = await runGpuSelfTests({
      device: context.device,
      signal: task.controller?.signal,
      schedules: ['chunk-mode', 'cell-tile'],
      onProgress: (progress) => send({ type: 'progress', requestId: task.requestId, phase: 'direct', stage: `GPU validation: ${progress.stage || progress.schedule || 'running'}`, completed: progress.caseIndex, total: progress.cases }),
    });
    context.validation = {
      status: validation.status === 'pass' ? 'pass-provisional' : validation.status,
      version: 'gpu-direct-selftest-p0',
      budget: validation.budget || null,
      errors: validation.numericalErrors || [],
      timings: { totalMilliseconds: performance.now() - validationStarted, schedules: ['chunk-mode', 'cell-tile'] },
    };
    if (validation.status !== 'pass') {
      const error = new Error(`GPU device validation ${validation.status}: ${validation.reason || 'one or more independent fixtures failed'}`);
      error.gpuValidation = context.validation;
      throw error;
    }
    gpuContext = context;
    gpuContextNeedsReinit = false;
    return context;
  } catch (error) {
    try { await context.dispose?.(); } catch (_) { /* failed validation device cleanup is best effort */ }
    gpuContext = null;
    gpuContextNeedsReinit = true;
    if (!error.gpuValidation) error.gpuValidation = context.validation || { status: 'error', version: 'gpu-direct-selftest-p0', budget: null, errors: [{ message: String(error?.message || error) }], timings: { totalMilliseconds: performance.now() - validationStarted, schedules: ['chunk-mode', 'cell-tile'] } };
    throw error;
  }
}
async function disposeContext(context) {
  if (context?.dispose) { try { await context.dispose(); } catch (_) { /* cleanup is best effort */ } }
}
async function disposeGpu() {
  const context = gpuContext;
  gpuContext = null;
  if (context) await disposeContext(context);
}
function stageBackends(execution) {
  const channels = execution.channelProvenance || {};
  const directStage = (key) => {
    const provenance = channels[key] || 'cpu-f64';
    return { provenance, precision: provenance.startsWith('mixed-') ? 'f32 GPU input/kernel with f64 CPU mode recovery' : provenance.includes('f32') ? 'f32 GPU input/kernel' : 'f64 CPU reference' };
  };
  execution.stageBackends = {
    trajectories: { backend: 'cpu', spatialFieldPrecision: 'f32 nLPT tensors', statePrecision: 'f64', tangentPrecision: 'f64', gpuPackedInputPrecision: 'f32' },
    exactDeposit: { backend: 'cpu', inputPrecision: 'f32 positions', arithmeticPrecision: 'f64 geometry and mass accumulation', outputPrecision: 'f32 density field' },
    pdfs: { backend: 'cpu', inputPrecision: 'f32 density fields', arithmeticPrecision: 'f64 histogram and moment accumulation', outputPrecision: 'f64 summaries' },
    griddedSpectra: { backend: 'cpu', inputPrecision: 'f32 density field', arithmeticPrecision: 'f64 complex FFT and binning', outputPrecision: 'f64 spectrum bins' },
    directPrimal: { lpt: directStage('lpt'), pm: directStage('pm') },
    directJvp: { lpt: directStage('lpt'), pm: directStage('pm') },
    finiteDifference: { backend: 'cpu', precision: 'f64' },
    finalization: { backend: 'cpu', precision: 'f64' },
  };
  return execution;
}
function finish(prepared, channels, execution) {
  stageBackends(execution);
  const raw = prepared.finish(channels, JSON.stringify(execution));
  return typeof raw === 'string' ? JSON.parse(raw) : raw;
}
function cpuExecution(requestedBackend, fallbackReason) {
  return {
    requestedBackend,
    executedBackend: 'cpu',
    fallbackReason,
    precision: 'f64 CPU reference and f64 finalization',
    channelProvenance: { lpt: 'cpu-f64', pm: 'cpu-f64', finiteDifference: 'cpu-f64' },
  };
}
function parseJson(value, label) {
  if (typeof value === 'string') return JSON.parse(value);
  if (value && typeof value === 'object') return value;
  throw new Error(`Wasm ${label} was not valid JSON.`);
}
function copyF64(value) { return value instanceof Float64Array ? new Float64Array(value) : Float64Array.from(value || []); }
function copyF32(value) { return value instanceof Float32Array ? new Float32Array(value) : Float32Array.from(value || []); }
function copyI32(value) { return value instanceof Int32Array ? new Int32Array(value) : Int32Array.from(value || []); }
function allFinite(values) { for (const value of values) if (!Number.isFinite(value)) return false; return true; }
function plainRecord(value) {
  if (value == null) return null;
  if (typeof value === 'string' || typeof value === 'number' || typeof value === 'boolean') return value;
  try { return JSON.parse(JSON.stringify(value)); } catch (_) { return String(value); }
}
function pickLimits(limits) {
  if (!limits) return null;
  const keys = ['maxBufferSize','maxStorageBufferBindingSize','maxComputeWorkgroupsPerDimension','maxComputeInvocationsPerWorkgroup','maxComputeWorkgroupSizeX','maxComputeWorkgroupSizeY'];
  return Object.fromEntries(keys.filter((key) => limits[key] != null).map((key) => [key, Number(limits[key])]));
}
function deviceMetadata(context) {
  if (!context) return null;
  return { adapterInfo: plainRecord(context.adapterInfo), provenance: plainRecord(context.provenance), validation: plainRecord(context.validation), limits: pickLimits(context.limits), features: Array.from(context.features || []) };
}
function failedDerivativeModes(result, modeCount) {
  const values = result?.gradient?.relativeError;
  if (!values || values.length !== modeCount) return null;
  const failed=[];
  for (let i=0;i<values.length;i++) {
    const value=values[i];
    const number = Number(value);
    if (!Number.isFinite(number) || Math.abs(number)>1e-5) failed.push(i);
  }
  return failed;
}
function forcedGpuError(kind) { return new Error(`forced GPU failure test hook: ${kind}`); }
