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

  const physical = await preparePhysical(task, wasm, preparedCtor);
  if (!physical || cancelled(task)) { physical?.prepared?.free?.(); return {}; }
  const { prepared, pmExecution } = physical;
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
      task.controller ||= new AbortController();
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

  const fdStarted = performance.now();
  let fdValues = null;
  let fdExecution = { backend: 'cpu', provenance: 'cpu-f64', precision: 'f64', h: metadata.h };
  const cpuFd = (reason = null, attempt = null) => {
    phase(task, 'finite-difference', 'Evaluating the retained f64 CPU finite-difference check.', false);
    if (typeof prepared.finite_difference_cpu === 'function') {
      fdValues = copyF64(prepared.finite_difference_cpu());
      if (fdValues.length !== metadata.modeCount || !allFinite(fdValues)) throw new Error('CPU finite-difference channels were malformed or non-finite.');
    }
    fdExecution = { backend: 'cpu', provenance: 'cpu-f64', precision: 'f64', h: metadata.h, fallbackReason: reason, ...(attempt ? { gpuAttempt: attempt } : {}) };
  };
  if (execMeta.executedBackend === 'gpu' && !task.policy.disableFd && typeof prepared.finish_with_finite_difference === 'function') {
    let attempt = null;
    try {
      phase(task, 'finite-difference', 'Evaluating both centered sheet endpoints with compensated GPU arithmetic.', true);
      const context = await getGpuContext(task);
      const validation = await validateFdContext(context, task);
      attempt = { device: deviceMetadata(context), validation };
      if (validation.status !== 'pass' || task.policy.forceFdFailure === 'validation') throw new Error('The GPU centered finite-difference self-check did not pass.');
      if (cancelled(task)) return {};
      if (task.policy.forceFdFailure === 'device-loss') { context.device.destroy(); throw new Error('Forced finite-difference device loss.'); }
      const { runFiniteDifference } = await import('./gpu/finite-difference.js');
      const fd = await runFiniteDifference(context.device, {
        dim: metadata.dim, n: metadata.n, modes: copyI32(prepared.mode_records()), h: metadata.h,
        plusHi: copyF32(prepared.finite_difference_plus_hi()), plusLo: copyF32(prepared.finite_difference_plus_lo()),
        minusHi: copyF32(prepared.finite_difference_minus_hi()), minusLo: copyF32(prepared.finite_difference_minus_lo()),
      }, { signal: task.controller.signal, shouldCancel: () => cancelled(task), batchChunks: task.policy.fdBatchChunks ?? task.policy.forceFdBatchChunks,
        maxGpuBytes: task.policy.forceFdFailure === 'limits' ? 1 : 512 * 1024 * 1024,
        onProgress: async (progress) => { send({ type: 'progress', requestId: task.requestId, phase: 'finite-difference', ...progress }); await yieldToMessages(); } });
      if (cancelled(task)) return {};
      attempt = { ...attempt, timings: fd.timings, diagnostics: fd.diagnostics, kernelProvenance: fd.provenance };
      if (task.policy.forceFdFailure === 'readback') throw new Error('Forced finite-difference readback failure.');
      fdValues = copyF64(fd.values);
      if (task.policy.forceFdFailure === 'nonfinite' && fdValues.length) fdValues[0] = Number.NaN;
      if (fdValues.length !== metadata.modeCount || !allFinite(fdValues)) throw new Error('GPU finite-difference channels were malformed or non-finite.');
      fdExecution = { backend: 'gpu', provenance: 'gpu-compensated-f32', precision: 'high/low f32 endpoint inputs and compensated GPU integrals; f64 host power difference', h: metadata.h, ...attempt };
    } catch (error) {
      if (cancelled(task)) return {};
      await disposeGpu();
      cpuFd(String(error?.message || error), attempt);
    }
  } else {
    cpuFd(task.policy.disableFd ? 'GPU finite differences disabled for an explicit reference comparison.' : null);
  }
  if (cancelled(task)) return {};
  stageTimes.finiteDifferenceMilliseconds = performance.now() - fdStarted;
  const finalize = () => {
    execMeta.finiteDifference = fdExecution;
    execMeta.channelProvenance = { ...execMeta.channelProvenance, finiteDifference: fdExecution.provenance };
    return finish(prepared, channels, execMeta, fdValues);
  };
  execMeta.pmEvolution = pmExecution;
  execMeta.timings = { ...(execMeta.timings || {}), stages: stageTimes };

  phase(task, 'finalizing', 'Finalizing f64 observables and the run manifest.');
  const finalizationStarted = performance.now();
  let result = finalize();
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
      result = finalize();
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
        result = finalize();
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
          channelProvenance: { lpt: 'mixed-cpu-f64-and-gpu-f32', pm: 'gpu-f32', finiteDifference: fdExecution.provenance },
          diagnostics: { ...(execMeta.diagnostics || {}), cpuRecoveredLptModes: failedModes.map((i) => metadata.modes?.[i] || [i]), derivativeGate: 'relativeError <= 1e-5 per mode' },
        };
        result = finalize();
        let unrecoveredModes = failedDerivativeModes(result, metadata.modeCount);
        if ((unrecoveredModes == null || unrecoveredModes.length) && fdExecution.backend === 'gpu') {
          const recoveryStarted = performance.now();
          cpuFd('GPU finite differences did not satisfy the unchanged amplitude-derivative gate after f64 LPT recovery.', fdExecution);
          stageTimes.finiteDifferenceMilliseconds += performance.now() - recoveryStarted;
          result = finalize();
          unrecoveredModes = failedDerivativeModes(result, metadata.modeCount);
        }
        if (unrecoveredModes == null || unrecoveredModes.length) {
          const hybridAttempt = execMeta;
          const fallbackStarted = performance.now();
          channels = copyF64(prepared.direct_cpu());
          stageTimes.directMilliseconds += performance.now() - fallbackStarted;
          execMeta = cpuExecution(requested, 'The selective f64 LPT recovery did not satisfy the per-mode finite-difference gate; the complete f64 CPU reference was used.');
          execMeta.gpuAttempt = hybridAttempt;
          execMeta.timings = { stages: stageTimes };
          result = finalize();
        }
      }
    }
  }
  // Direct-mode fallback does not replace an already completed GPU trajectory.
  // Record those two decisions separately, including on complete direct recovery.
  execMeta.pmEvolution = pmExecution;
  if (result?.error) throw new Error(result.error);
  result.execution.pmEvolution = pmExecution;
  stageBackends(result.execution);
  if (pmExecution.backend === 'gpu' && result.execution.executedBackend === 'cpu') {
    result.execution.executedBackend = 'hybrid';
    result.execution.precision = 'f32 GPU PM evolution with f64 CPU direct modes and finalization';
  }
  if (fdExecution.backend === 'gpu' && result.execution.executedBackend === 'cpu') {
    result.execution.executedBackend = 'hybrid';
    result.execution.precision = 'compensated GPU finite differences with f64 CPU direct modes and finalization';
  }
  stageTimes.finalizationMilliseconds = performance.now() - finalizationStarted;
  if (result.execution) result.execution.timings = { ...(result.execution.timings || {}), stages: stageTimes };
  if (result.execution?.gpuAttempt?.timings) result.execution.gpuAttempt.timings.stages = stageTimes;
  task.execution = result.execution || execMeta;
  return { result, durationMilliseconds: performance.now() - started, execution: task.execution };
  } finally {
    prepared.free?.();
  }
}

async function preparePhysical(task, wasm, preparedCtor) {
  const cpu = (reason) => ({ prepared: new preparedCtor(JSON.stringify(task.state)), pmExecution: { backend: 'cpu', precision: 'f64', fallbackReason: reason } });
  const requested = task.policy.requested || 'cpu';
  if (requested !== 'gpu' || task.state.dim === 1) return cpu(null);
  // Failure hooks intentionally test the original complete CPU recovery path.
  if (task.policy.forceGpuFailure) return cpu('Resident PM bypassed for a forced GPU failure check.');
  if (task.policy.disablePm) return cpu('Resident PM disabled for an explicit observer-only comparison.');
  if (typeof wasm.PendingLabRun !== 'function') return cpu('Deferred PM boundary unavailable in this Wasm build.');
  const pending = new wasm.PendingLabRun(JSON.stringify(task.state));
  let attempt = null;
  try {
    const meta = parseJson(pending.metadata(), 'pending PM metadata');
    await yieldToMessages();
    if (cancelled(task)) return null;
    const meshN = meta.forceMesh;
    if (!Number.isInteger(meshN) || meshN < 2 || (meshN & (meshN - 1)) !== 0) {
      return { prepared: pending.finish_cpu(), pmExecution: { backend: 'cpu', precision: 'f64', fallbackReason: `Resident force FFT requires a power-of-two force mesh; requested mesh ${meshN} was retained on CPU.` } };
    }
    task.controller = new AbortController();
    phase(task, 'pm-evolution', 'Validating and evolving the cosmological PM state and tangent on GPU.', true);
    const context = await getGpuContext(task);
    if (cancelled(task)) return null;
    const forcedPmFailure = task.policy.forcePmFailure;
    attempt = { device: deviceMetadata(context), validation: null };
    if (forcedPmFailure === 'validation') throw new Error('Forced PM device-validation failure.');
    const validation = await validatePmContext(context, task);
    attempt = { device: deviceMetadata(context), validation };
    if (validation.status !== 'pass') throw new Error('The GPU cosmological PM self-check did not pass.');
    if (forcedPmFailure === 'device-loss') {
      context.device.destroy();
      throw new Error('Forced PM device loss.');
    }
    const clocksFlat = copyF64(pending.pm_clocks());
    const steps = clocksFlat.length / 5;
    if (!Number.isInteger(steps) || steps !== meta.steps) throw new Error('Malformed Rust PM clock schedule.');
    const clocks = { kick0: new Float64Array(steps), drift: new Float64Array(steps), kick1: new Float64Array(steps) };
    for (let i = 0; i < steps; i++) { clocks.kick0[i] = clocksFlat[5*i]; clocks.drift[i] = clocksFlat[5*i+1]; clocks.kick1[i] = clocksFlat[5*i+2]; }
    const { runPm } = await import('./gpu/pm.js');
    const pm = await runPm(context.device, {
      dim: meta.dim, n: meta.n, meshN, boxSize: 1, omegaM: meta.omegaM,
      initialMinimum: Math.min(meta.pmInitialMinimumJacobianF64, meta.pmInitialMinimumJacobianDisplacementF32),
      positionRepresentation: 'lattice-displacement',
      x: copyF32(pending.pm_displacement()), p: copyF32(pending.pm_p()),
      dx: copyF32(pending.pm_dx()), dp: copyF32(pending.pm_dp()), clocks,
      forceKernel: 'rust-spectral-cic-v1',
    }, {
      signal: task.controller.signal, shouldCancel: () => cancelled(task), maxGpuBytes: forcedPmFailure === 'limits' ? 1 : 512 * 1024 * 1024, allowCrossing: true,
      checkDrift: (start, end) => wasm.pm_drift_minimum(meta.dim, meta.n, start, end),
      onProgress: (progress) => send({ type: 'progress', requestId: task.requestId, phase: 'pm-evolution', ...progress }),
    });
    if (cancelled(task)) return null;
    attempt = { ...attempt, diagnostics: pm.diagnostics, timings: pm.timings };
    if (forcedPmFailure === 'readback') throw new Error('Forced PM readback failure.');
    if (forcedPmFailure === 'nonfinite') pm.x[0] = Number.NaN;
    phase(task, 'observations', 'Depositing the evolved sheet exactly and computing CPU density statistics.');
    const observationStarted = performance.now();
    const prepared = pending.finish_gpu_pm(pm.x, pm.p, pm.dx, pm.dp, Float64Array.from(pm.driftMinima));
    return { prepared, pmExecution: { backend: 'gpu', precision: 'f32 state, tangent, CIC and FFT; Rust f64 clocks rounded to f32; near-zero drift checks in CPU f64', ...attempt, observationMilliseconds: performance.now() - observationStarted } };
  } catch (error) {
    if (cancelled(task)) return null;
    phase(task, 'pm-recovery', 'Recovering cosmological PM evolution with the retained f64 CPU initial state.');
    // A failed resident stage must release its device resources before reuse.
    await disposeGpu();
    return { prepared: pending.finish_cpu(), pmExecution: { backend: 'cpu', precision: 'f64', fallbackReason: String(error?.message || error), gpuAttempt: attempt } };
  } finally { pending.free?.(); }
}

async function validatePmContext(context, task) {
  if (context.pmValidation) return context.pmValidation;
  const { runPmSelfTests } = await import('./gpu/resident-check.js');
  const validation = await runPmSelfTests({ device: context.device, signal: task.controller.signal,
    onProgress: (progress) => send({ type: 'progress', requestId: task.requestId, phase: 'pm-evolution', stage: 'PM device validation', ...progress }) });
  if (cancelled(task)) throw new Error('PM validation cancelled.');
  if (validation.status === 'pass') context.pmValidation = validation;
  return validation;
}

async function validateFdContext(context, task) {
  if (context.fdValidation) return context.fdValidation;
  const { runFiniteDifferenceSelfTest } = await import('./gpu/fd-check.js');
  const validation = await runFiniteDifferenceSelfTest({ device: context.device, signal: task.controller.signal,
    onProgress: (progress) => send({ type: 'progress', requestId: task.requestId, phase: 'finite-difference', stage: 'Centered FD device validation', ...progress }) });
  if (cancelled(task)) throw new Error('Finite-difference validation cancelled.');
  if (validation.status === 'pass') context.fdValidation = validation;
  return validation;
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
    trajectories: { backend: execution.pmEvolution?.backend === 'gpu' ? 'mixed' : 'cpu', lptBackend: 'cpu', pmBackend: execution.pmEvolution?.backend || 'cpu', spatialFieldPrecision: 'f32 nLPT tensors', statePrecision: execution.pmEvolution?.backend === 'gpu' ? 'LPT f64, PM f32' : 'f64', lptStatePrecision: 'f64', pmStatePrecision: execution.pmEvolution?.backend === 'gpu' ? 'f32' : 'f64', tangentPrecision: execution.pmEvolution?.backend === 'gpu' ? 'LPT f64, PM f32' : 'f64', gpuPackedInputPrecision: 'f32' },
    exactDeposit: { backend: 'cpu', inputPrecision: 'f32 positions', arithmeticPrecision: 'f64 geometry and mass accumulation', outputPrecision: 'f32 density field' },
    pdfs: { backend: 'cpu', inputPrecision: 'f32 density fields', arithmeticPrecision: 'f64 histogram and moment accumulation', outputPrecision: 'f64 summaries' },
    griddedSpectra: { backend: 'cpu', inputPrecision: 'f32 density field', arithmeticPrecision: 'f64 complex FFT and binning', outputPrecision: 'f64 spectrum bins' },
    directPrimal: { lpt: directStage('lpt'), pm: directStage('pm') },
    directJvp: { lpt: directStage('lpt'), pm: directStage('pm') },
    finiteDifference: { backend: execution.finiteDifference?.backend || 'cpu', precision: execution.finiteDifference?.precision || 'f64', provenance: channels.finiteDifference || 'cpu-f64' },
    finalization: { backend: 'cpu', precision: 'f64' },
  };
  return execution;
}
function finish(prepared, channels, execution, fdValues = null) {
  stageBackends(execution);
  const raw = fdValues && typeof prepared.finish_with_finite_difference === 'function'
    ? prepared.finish_with_finite_difference(channels, fdValues, JSON.stringify(execution))
    : prepared.finish(channels, JSON.stringify(execution));
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
