// A persistent worker owns the Wasm facade and (when requested) its GPU device.
// Jobs are serialized. A newer request replaces the queued request and asks an
// asynchronous GPU run to stop between submitted batches.
import init, * as wasmApi from '../pkg/hcc_core.js?v=20261027h';

let wasmReady;
let gpuContext = null;
let labSession = null;
let lastPhysical = null;
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
    disposeGpu().finally(() => { disposeSession(); send({ type: 'shutdown-ack' }); });
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
  const task = { requestId: data.requestId, key: data.key, state: data.state, allEstimators: data.allEstimators || null, estimatorsOnly: Boolean(data.estimatorsOnly), eftCounterterms: Boolean(data.eftCounterterms), policy: data.policy || { requested: 'cpu' }, cancelled: false, controller: null };
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
      disposeSession();
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
  const inBrowserWorker = typeof WorkerGlobalScope !== 'undefined' && self instanceof WorkerGlobalScope;
  if (!wasmReady) wasmReady = inBrowserWorker ? init({ module_or_path: new URL('../pkg/hcc_core_bg.wasm?v=20261029disp', import.meta.url) }) : init();
  try { await wasmReady; } catch (error) { wasmReady = null; throw error; }
  const wasm = wasmApi;
  if (cancelled(task)) return {};

  if (task.estimatorsOnly && labSession && typeof labSession.compute_estimators_cached === 'function' && !task.policy?.disableCache) {
    phase(task, 'observations', 'Constructing requested LPT/estimator density fields from retained trajectories.', false);
    const reqJson = JSON.stringify(task.state);
    const candidateSources = lastPhysical?.source ? [lastPhysical.source, lastPhysical.source === 'gpu-f32' ? 'cpu-f64' : 'gpu-f32'] : ['cpu-f64', 'gpu-f32'];
    const candidateArith = lastPhysical?.arithmetic ? [lastPhysical.arithmetic, lastPhysical.arithmetic === 'f32' ? 'f64' : 'f32'] : ['f64', 'f32'];
    let upd = null;
    for (const src of candidateSources) {
      for (const ar of candidateArith) {
        try {
          upd = labSession.compute_estimators_cached(reqJson, ar, src);
          if (upd) break;
        } catch (_) {
          upd = null;
        }
      }
      if (upd) break;
    }
    if (upd) {
      try {
        const parsed = parseJson(upd.json(), 'estimator update');
        if (parsed.customSeries && typeof upd.custom_rho_f32 === 'function') {
          for (const [k, entry] of Object.entries(parsed.customSeries)) {
            if (entry && (!entry.rho || entry.rho.length === 0)) {
              entry.rho = upd.custom_rho_f32(k);
            }
          }
        }
        return { estimatorsOnly: true, customSeries: parsed.customSeries || {}, durationMilliseconds: performance.now() - started };
      } finally {
        upd.free?.();
      }
    }
    if (Array.isArray(task.allEstimators) && task.allEstimators.length) {
      task.state = { ...task.state, estimators: task.allEstimators };
    }
  }

  if (task.eftCounterterms) {
    phase(task, 'direct', 'Comparing dual-smoothing P₁ sheet spectra and fitting the sound-speed counterterm.', false);
    if (typeof wasm.eft_counterterms_run !== 'function') {
      throw new Error('This Wasm build does not expose eft_counterterms_run; rebuild with make wasm.');
    }
    const uv = Number(task.state.uvSmoothing ?? 5);
    const irRaw = Number(task.state.irSmoothing ?? -1);
    const kMax = Number(task.state.kMaxFit ?? 0.08);
    const req = { ...task.state };
    delete req.uvSmoothing;
    delete req.irSmoothing;
    delete req.kMaxFit;
    delete req.view;
    delete req.weight;
    delete req.backend;
    delete req.rotate;
    delete req.tilt;
    delete req.pubmodel;
    delete req.pubest;
    delete req.pubr;
    delete req.paired;
    delete req.schemaVersion;
    req.ic = 'gaussian';
    req.estimators = [];
    const raw = wasm.eft_counterterms_run(JSON.stringify(req), uv, irRaw, kMax);
    const result = typeof raw === 'string' ? JSON.parse(raw) : raw;
    if (result?.error) throw new Error(result.error);
    const execution = cpuExecution(task.policy?.requested || 'cpu', 'Dual-smoothing counterterm comparison uses the f64 CPU P₁ sheet path.');
    result.execution = execution;
    result.guard = { ok: true, crossed: false, reason: 'Counterterm comparison reports inverted fractions per filter; it does not use the single-run orientation guard.' };
    task.execution = execution;
    return { result, durationMilliseconds: performance.now() - started, execution };
  }

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
  const { prepared, pmExecution, lptExecution, depositExecution } = physical;
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
      if (gpuContextNeedsReinit || ['device-destroy','destroy-device','device-loss'].includes(task.policy?.forceGpuFailure)) await disposeGpu();
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
      if (gpuContextNeedsReinit || task.policy.forceFdFailure === 'device-loss') await disposeGpu();
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
  execMeta.lptConstruction = lptExecution;
  execMeta.exactDeposit = depositExecution;
  execMeta.timings = { ...(execMeta.timings || {}), stages: stageTimes };

  phase(task, 'finalizing', 'Finalizing f64 observables and the run manifest.');
  const finalizationStarted = performance.now();
  let result = finalize();
  let analyticExecution = { backend: 'cpu', provenance: 'cpu-f64', fallbackReason: task.policy.disableCompensated ? 'Compensated GPU analytic recovery disabled for an explicit reference comparison.' : null };
  if (execMeta.executedBackend === 'gpu') {
    let failedModes = failedDerivativeModes(result, metadata.modeCount);
    if (failedModes != null) {
      const targets = task.policy.forceCompensatedFailure ? Array.from({length:metadata.modeCount},(_,i)=>i) : failedModes;
      if (!task.policy.disableCompensated && targets.length && typeof prepared.lpt_displacement_hi === 'function') {
        let attempt = null;
        try {
          phase(task,'analytic-recovery','Evaluating weak LPT modes and their analytic amplitude response with compensated GPU arithmetic.',true);
          const context=await getGpuContext(task),validation=await validateCompensatedContext(context,task);
          attempt={device:deviceMetadata(context),validation};
          const failure=task.policy.forceCompensatedFailure;
          if(validation.status!=='pass'||failure==='validation')throw new Error('GPU analytic-recovery validation did not pass.');
          const records=copyI32(prepared.mode_records()),modes=new Int32Array(targets.length*4);
          targets.forEach((m,i)=>modes.set(records.subarray(m*4,m*4+4),i*4));
          const {runCompensatedDirect}=await import('./gpu/compensated-direct.js');
          const gpu=await runCompensatedDirect(context.device,{dim:metadata.dim,n:metadata.n,modes,
            psiHi:copyF32(prepared.lpt_displacement_hi()),psiLo:copyF32(prepared.lpt_displacement_lo()),
            directionHi:copyF32(prepared.lpt_direction_hi()),directionLo:copyF32(prepared.lpt_direction_lo())}, {
            signal:task.controller.signal,shouldCancel:()=>cancelled(task),maxGpuBytes:failure==='limits'?1:512*1024*1024,
            onProgress:p=>send({type:'progress',requestId:task.requestId,phase:'analytic-recovery',...p})});
          if(cancelled(task))return {};
          attempt={...attempt,timings:gpu.timings,diagnostics:gpu.diagnostics,kernelProvenance:gpu.provenance};
          if(failure==='readback')throw new Error('Forced analytic-recovery readback failure.');
          const values=copyF64(gpu.channels);
          if(failure==='nonfinite')values[0]=NaN;
          if(values.length!==targets.length*4||!allFinite(values))throw new Error('GPU analytic recovery returned malformed or nonfinite channels.');
          targets.forEach((m,i)=>{channels.set(values.subarray(i*4,i*4+4),m*8);execMeta.modeProvenance[m].lpt='gpu-compensated-f32';});
          analyticExecution={backend:'gpu',provenance:'gpu-compensated-f32',modes:targets.map(i=>metadata.modes[i]),...attempt};
          execMeta.channelProvenance.lpt=targets.length===metadata.modeCount?'gpu-compensated-f32':'mixed-gpu-f32-and-gpu-compensated-f32';
          execMeta.analyticRecovery=analyticExecution;
          result=finalize();
          failedModes=failedDerivativeModes(result,metadata.modeCount);
        }catch(error){
          if(cancelled(task))return {};
          analyticExecution={backend:'cpu',provenance:'cpu-f64',fallbackReason:String(error?.message||error),gpuAttempt:attempt};
          // A forced failure tests the reference recovery even if the ordinary
          // observer happened to pass this particular request's derivative gate.
          if(task.policy.forceCompensatedFailure)failedModes=targets;
        }
      } else if (!task.policy.disableCompensated && !targets.length) analyticExecution={backend:'unused',provenance:null};
    }
    if (failedModes?.length && typeof prepared.direct_lpt_cpu_modes === 'function') {
      phase(task,'direct-recovery','Recovering the remaining LPT modes with the f64 CPU reference.',false);
      const recoveryStarted=performance.now(),values=copyF64(prepared.direct_lpt_cpu_modes(Uint32Array.from(failedModes)));
      if(values.length!==failedModes.length*4||!allFinite(values))throw new Error('CPU recovery returned malformed direct-mode channels.');
      failedModes.forEach((m,i)=>{channels.set(values.subarray(i*4,i*4+4),m*8);execMeta.modeProvenance[m].lpt='cpu-f64';});
      stageTimes.cpuRecoveryMilliseconds=performance.now()-recoveryStarted;
      execMeta.executedBackend='hybrid';
      execMeta.fallbackReason=`${failedModes.length} LPT mode(s) required f64 CPU recovery at the unchanged 1e-5 amplitude-derivative gate.`;
      execMeta.channelProvenance.lpt='mixed-cpu-f64-and-gpu-f32';
      execMeta.diagnostics={...(execMeta.diagnostics||{}),cpuRecoveredLptModes:failedModes.map(i=>metadata.modes[i]),derivativeGate:'relativeError <= 1e-5 per mode'};
      result=finalize();
      failedModes=failedDerivativeModes(result,metadata.modeCount);
      if((failedModes==null||failedModes.length)&&fdExecution.backend==='gpu'){
        const recoveryStarted=performance.now();
        cpuFd('GPU finite differences did not satisfy the unchanged amplitude-derivative gate after f64 LPT recovery.',fdExecution);
        stageTimes.finiteDifferenceMilliseconds+=performance.now()-recoveryStarted;
        result=finalize();failedModes=failedDerivativeModes(result,metadata.modeCount);
      }
    }
    if(failedModes==null||failedModes.length){
      const gpuAttempt=execMeta,fallbackStarted=performance.now();
      channels=copyF64(prepared.direct_cpu());
      stageTimes.cpuRecoveryMilliseconds+=performance.now()-fallbackStarted;
      execMeta=cpuExecution(requested,'GPU observer recovery did not satisfy the per-mode derivative gate; the complete f64 CPU observer was used.');
      execMeta.gpuAttempt=gpuAttempt;execMeta.timings={stages:stageTimes};
      result=finalize();
    }
  }
  execMeta.analyticRecovery=analyticExecution;
  // Direct-mode fallback does not replace an already completed GPU trajectory.
  // Record those two decisions separately, including on complete direct recovery.
  execMeta.pmEvolution = pmExecution;
  execMeta.lptConstruction = lptExecution;
  execMeta.exactDeposit = depositExecution;
  if (result?.error) throw new Error(result.error);
  result.execution.pmEvolution = pmExecution;
  result.execution.lptConstruction = lptExecution;
  result.execution.exactDeposit = depositExecution;
  result.execution.cache = result.cache || metadata.cache || null;
  result.execution.analyticRecovery = analyticExecution;
  stageBackends(result.execution);
  if ([pmExecution,lptExecution,depositExecution].some(stage=>stage.backend==='gpu') && result.execution.executedBackend === 'cpu') {
    result.execution.executedBackend = 'hybrid';
    result.execution.precision = 'GPU physical stages with f64 CPU direct modes and finalization';
  }
  if (fdExecution.backend === 'gpu' && result.execution.executedBackend === 'cpu') {
    result.execution.executedBackend = 'hybrid';
    result.execution.precision = 'compensated GPU finite differences with f64 CPU direct modes and finalization';
  }
  stageTimes.finalizationMilliseconds = performance.now() - finalizationStarted;
  attachBinaryFields(prepared, result);
  if (result.execution) result.execution.timings = { ...(result.execution.timings || {}), stages: stageTimes };
  if (result.execution?.gpuAttempt?.timings) result.execution.gpuAttempt.timings.stages = stageTimes;
  task.execution = result.execution || execMeta;
  return { result, durationMilliseconds: performance.now() - started, execution: task.execution };
  } finally {
    prepared.free?.();
  }
}

// Native caches own f64 state and validity diagnostics; JS retains only the
// provenance of the last physical run. Policies that exercise failure paths
// bypass the completed-run cache so the requested stage is actually reached.
const DEPOSIT_SOURCE = 'gpu-f32-geometry-overlap';
function disposeSession() { labSession?.free?.(); labSession = null; lastPhysical = null; }
function reusedStage(record, fallback) {
  const value = record || fallback;
  return { ...value, reused: true, previousRunTimings: value.timings || value.previousRunTimings || null, timings: null };
}
function physicalKey(state) { const { smoothing, mesh, estimators, ...physical } = state; return JSON.stringify(physical); }
async function preparePhysical(task, wasm, preparedCtor) {
  if (typeof wasm.LabSession !== 'function') return { prepared: new preparedCtor(JSON.stringify(task.state)),
    lptExecution: { backend: 'cpu', precision: 'f32 spatial tensors; f64 growth/state' },
    pmExecution: { backend: 'cpu', precision: 'f64' }, depositExecution: { backend: 'cpu', provenance: 'cpu-exact', arithmeticPrecision: 'f64 geometry and accumulation' } };
  const ephemeral = Boolean(task.policy.disableCache);
  const session = ephemeral ? new wasm.LabSession() : (labSession ||= new wasm.LabSession());
  let pending = null, seed = null;
  const request = JSON.stringify(task.state), requestedGpu = task.policy.requested === 'gpu' && task.state.dim > 1 && !task.policy.forceGpuFailure;
  const radix = (n) => Number.isInteger(n) && n >= 2 && (n & (n-1)) === 0;
  const gridCells = (task.state.n || 0) ** (task.state.dim || 1);
  const meshCells = (task.state.mesh || 0) ** (task.state.dim || 1);
  const maxGpuLptCells = task.state.extended ? 4194304 : 262144;
  const maxGpuPmCells = task.state.extended ? 262144 : 13824;
  const needPm = !Array.isArray(task.state.estimators) || task.state.estimators.length === 0 || task.state.estimators.some(e => String(e).startsWith('pm:'));
  const gpuLpt = requestedGpu && !task.policy.disableLpt && radix(task.state.n) && gridCells <= maxGpuLptCells && typeof wasm.GpuLptSeed === 'function';
  const gpuPm = needPm && requestedGpu && !task.policy.disablePm && radix(task.state.forceMesh) && gridCells <= maxGpuPmCells;
  const gpuDeposit = requestedGpu && !task.policy.disableDeposit && gridCells <= 65536 && meshCells <= 65536;
  const forcePhysical = ['forceLptFailure','forcePmFailure','forceDepositFailure'].some(k => task.policy[k]);
  let lptExecution = { backend: 'cpu', precision: 'f32 nLPT tensors; f64 cosmological growth/state', fallbackReason: task.policy.disableLpt ? 'GPU nLPT disabled for an explicit reference comparison.' : requestedGpu && !gpuLpt ? (gridCells > 262144 ? 'Large lattice uses CPU f64 LPT construction.' : 'GPU nLPT requires a radix lattice in 2D/3D.') : null };
  let pmExecution = !needPm
    ? { backend: 'cpu', precision: 'skipped (LPT-only selection)', skipped: true, fallbackReason: 'No PM trajectory/estimator selected in the comparison matrix; skipped PM simulation for fast LPT evaluation.' }
    : { backend: 'cpu', precision: 'f64', fallbackReason: task.policy.disablePm ? 'Resident PM disabled for an explicit reference comparison.' : requestedGpu && !gpuPm ? (gridCells > 13824 ? 'Large-grid laboratory omits PM evolution.' : 'Resident PM requires a power-of-two force mesh.') : null };
  let depositExecution = { backend: 'cpu', provenance: 'cpu-exact', arithmeticPrecision: 'f64 geometry and accumulation', fallbackReason: task.policy.disableDeposit ? 'GPU exact deposition disabled for an explicit reference comparison.' : requestedGpu && !gpuDeposit ? 'Large observation grid uses CPU exact deposition.' : null };
  const source = gpuLpt ? 'gpu-f32' : 'cpu-f64', arithmetic = gpuPm ? 'f32' : 'f64', depositSource = gpuDeposit ? DEPOSIT_SOURCE : 'cpu-exact';
  const key = physicalKey(task.state);
  try {
    if (!ephemeral && !forcePhysical) {
      const cached = session.try_cached_observed_source(request, arithmetic, source, depositSource);
      if (cached) {
        const previous = lastPhysical?.key === key && lastPhysical.source === source && lastPhysical.arithmetic === arithmetic && lastPhysical.depositSource === depositSource ? lastPhysical : null;
        return { prepared: cached,
          lptExecution: reusedStage(previous?.lptExecution, { ...lptExecution, backend: gpuLpt ? 'gpu' : 'cpu' }),
          pmExecution: reusedStage(previous?.pmExecution, { ...pmExecution, backend: gpuPm ? 'gpu' : 'cpu' }),
          depositExecution: parseJson(cached.metadata(),'cached metadata').cache?.observations?.status === 'hit'
            ? reusedStage(previous?.depositExecution, { ...depositExecution, backend: gpuDeposit ? 'gpu' : 'cpu', provenance: depositSource })
            : { ...depositExecution, reused: false } };
      }
    }
    if (gpuLpt) {
      let attempt = null;
      try {
        if (session.prepare_initial_slot(request, source) && !task.policy.forceLptFailure) {
          pending = session.prepare_pending_source(request, source);
          lptExecution = reusedStage(lastPhysical?.source === source ? lastPhysical.lptExecution : null, { backend: 'gpu', precision: 'f32 spectral nLPT fields; f64 cosmological growth/state' });
        } else {
          task.controller ||= new AbortController();
          phase(task, 'lpt-construction', 'Constructing spectral nLPT sources and displacement tensors on GPU.', true);
          const context = await getGpuContext(task), validation = await validateLptContext(context, task);
          attempt = { device: deviceMetadata(context), validation };
          if (validation.status !== 'pass' || task.policy.forceLptFailure === 'validation') throw new Error('GPU nLPT validation did not pass.');
          seed = new wasm.GpuLptSeed(request);
          const meta = parseJson(seed.metadata(), 'GPU nLPT seed metadata');
          const { runLpt } = await import('./gpu/lpt.js');
          const lpt = await runLpt(context.device, { dim: meta.dim, n: meta.n, deltaHat: copyF32(seed.delta_hat()), specs: meta.specs }, {
            signal: task.controller.signal, shouldCancel: () => cancelled(task), maxGpuBytes: task.policy.forceLptFailure === 'limits' ? 1 : 512*1024*1024,
            onProgress: p => send({ type: 'progress', requestId: task.requestId, phase: 'lpt-construction', ...p }) });
          if (cancelled(task)) return null;
          attempt = { ...attempt, timings: lpt.timings, diagnostics: lpt.diagnostics, kernelProvenance: lpt.provenance };
          if (task.policy.forceLptFailure === 'readback') throw new Error('Forced nLPT readback failure.');
          if (task.policy.forceLptFailure === 'nonfinite') lpt.fields[0] = NaN;
          pending = seed.finish_gpu(session, lpt.fields);
          lptExecution = { backend: 'gpu', precision: 'f32 spectral nLPT fields; f64 cosmological growth/state', ...attempt };
        }
      } catch (error) {
        if (cancelled(task)) return null;
        lptExecution = { ...lptExecution, backend: 'cpu', fallbackReason: String(error?.message || error), gpuAttempt: attempt };
        phase(task, 'lpt-recovery', 'Recovering nLPT construction with the retained CPU algorithm.');
        pending = session.prepare_pending(request);
      }
    } else pending = session.prepare_pending(request);
    const meta = parseJson(pending.metadata(), 'pending physical metadata');
    await yieldToMessages();
    if (cancelled(task)) return null;
    let pm = null;
    const cachedValue = !ephemeral && !task.policy.forcePmFailure ? session.cached_pm_positions(pending, arithmetic) : null;
    const cachedPmPositions = cachedValue == null ? null : copyF32(cachedValue);
    if (cachedPmPositions) {
      if (cachedPmPositions.length !== meta.n**meta.dim*4 || !allFinite(cachedPmPositions)) throw new Error('Malformed cached PM positions.');
      pmExecution = reusedStage(lastPhysical?.arithmetic === arithmetic ? lastPhysical.pmExecution : null, { ...pmExecution, backend: gpuPm ? 'gpu' : 'cpu' });
    }
    if (gpuPm && !cachedPmPositions) {
      let attempt = null;
      try {
        task.controller ||= new AbortController();
        phase(task, 'pm-evolution', 'Evolving the cosmological PM state and tangent on GPU.', true);
        const context = await getGpuContext(task), validation = await validatePmContext(context, task);
        attempt = { device: deviceMetadata(context), validation };
        const failure = task.policy.forcePmFailure;
        if (validation.status !== 'pass' || failure === 'validation') throw new Error('GPU PM device validation did not pass.');
        if (failure === 'device-loss') { context.device.destroy(); throw new Error('Forced PM device loss.'); }
        const flat = copyF64(pending.pm_clocks()), steps = flat.length / 5;
        if (!Number.isInteger(steps) || steps !== meta.steps) throw new Error('Malformed Rust PM clock schedule.');
        const clocks = { kick0: new Float64Array(steps), drift: new Float64Array(steps), kick1: new Float64Array(steps) };
        for (let i=0;i<steps;i++) { clocks.kick0[i]=flat[5*i]; clocks.drift[i]=flat[5*i+1]; clocks.kick1[i]=flat[5*i+2]; }
        const { runPm } = await import('./gpu/pm.js');
        pm = await runPm(context.device, { dim:meta.dim,n:meta.n,meshN:meta.forceMesh,boxSize:1,omegaM:meta.omegaM,
          initialMinimum:Math.min(meta.pmInitialMinimumJacobianF64,meta.pmInitialMinimumJacobianDisplacementF32),positionRepresentation:'lattice-displacement',
          x:copyF32(pending.pm_displacement()),p:copyF32(pending.pm_p()),dx:copyF32(pending.pm_dx()),dp:copyF32(pending.pm_dp()),clocks,forceKernel:'rust-spectral-cic-v1' }, {
          signal:task.controller.signal,shouldCancel:()=>cancelled(task),maxGpuBytes:failure==='limits'?1:512*1024*1024,allowCrossing:true,
          checkDrift:(start,end)=>wasm.pm_drift_minimum(meta.dim,meta.n,start,end),
          onProgress:p=>send({type:'progress',requestId:task.requestId,phase:'pm-evolution',...p}) });
        if (cancelled(task)) return null;
        attempt = { ...attempt, timings:pm.timings,diagnostics:pm.diagnostics };
        if (failure==='readback') throw new Error('Forced PM readback failure.');
        if (failure==='nonfinite') pm.x[0]=NaN;
        if (![pm.x,pm.p,pm.dx,pm.dp].every(allFinite)) throw new Error('GPU PM state is nonfinite.');
        pmExecution={backend:'gpu',precision:'f32 state/tangent, CIC and FFT; Rust f64 clocks; near-zero drift checks in CPU f64',...attempt};
      } catch(error) {
        if (cancelled(task)) return null;
        pm=null;
        pmExecution={backend:'cpu',precision:'f64',fallbackReason:String(error?.message||error),gpuAttempt:attempt};
        if (gpuContextNeedsReinit || task.policy.forcePmFailure === 'device-loss') await disposeGpu();
      }
    }
    let prepared = null;
    if (gpuDeposit) {
      let attempt = null;
      try {
        task.controller ||= new AbortController();
        phase(task,'exact-deposit','Clipping sheet simplices against Eulerian cells on GPU.',true);
        const context=await getGpuContext(task),validation=await validateDepositContext(context,task,wasm);
        attempt={device:deviceMetadata(context),validation};
        const failure=task.policy.forceDepositFailure;
        if(validation.status!=='pass'||failure==='validation')throw new Error('GPU exact-deposition validation did not pass.');
        const {runExactDeposit}=await import('./gpu/deposit.js');
        const options={signal:task.controller.signal,shouldCancel:()=>cancelled(task),maxGpuBytes:failure==='limits'?1:512*1024*1024,
          onProgress:p=>send({type:'progress',requestId:task.requestId,phase:'exact-deposit',...p})};
        const count=meta.n**meta.dim*4;
        const input={dim:meta.dim,n:meta.n,meshN:task.state.mesh,boxSize:1};
        const lpt=await runExactDeposit(context.device,{...input,positions:copyF32(pending.lpt_positions())},options);
        const pmDensity=(!needPm||gridCells>13824)?lpt:await runExactDeposit(context.device,{...input,positions:cachedPmPositions || (pm?pm.x:copyF32(pending.pm_reference()).subarray(0,count))},options);
        if(cancelled(task))return null;
        attempt={...attempt,timings:{lpt:lpt.timings,pm:pmDensity.timings},diagnostics:{lpt:lpt.diagnostics,pm:pmDensity.diagnostics},kernelProvenance:lpt.provenance};
        if(failure==='readback')throw new Error('Forced exact-deposition readback failure.');
        if(failure==='nonfinite')lpt.density[0]=NaN;
        phase(task,'observations','Computing f64 spectra and PDF statistics from exact cell averages.');
        prepared=cachedPmPositions?session.finish_cached_observed(pending,arithmetic,lpt.density,pmDensity.density):pm?session.finish_gpu_pm_observed(pending,pm.x,pm.p,pm.dx,pm.dp,Float64Array.from(pm.driftMinima),lpt.density,pmDensity.density):session.finish_cpu_observed(pending,lpt.density,pmDensity.density);
        depositExecution={backend:'gpu',provenance:DEPOSIT_SOURCE,arithmeticPrecision:'f32 cell-local simplex clipping and compensated cell-owner sums',...attempt};
      }catch(error){
        if(cancelled(task))return null;
        depositExecution={backend:'cpu',provenance:'cpu-exact',arithmeticPrecision:'f64 geometry and mass accumulation',fallbackReason:String(error?.message||error),gpuAttempt:attempt};
      }
    }
    if(!prepared){
      phase(task,'observations','Depositing exact cell averages and computing f64 density statistics.');
      prepared=cachedPmPositions?session.finish_cached(pending,arithmetic):pm?session.finish_gpu_pm(pending,pm.x,pm.p,pm.dx,pm.dp,Float64Array.from(pm.driftMinima)):session.finish_cpu(pending);
      if (!prepared) throw new Error('A matched cached PM state became unavailable before finalization.');
    }
    if(!ephemeral)lastPhysical={key,source:lptExecution.backend==='gpu'?'gpu-f32':'cpu-f64',arithmetic:pmExecution.backend==='gpu'?'f32':'f64',depositSource:depositExecution.provenance,lptExecution,pmExecution,depositExecution};
    return {prepared,lptExecution,pmExecution,depositExecution};
  }finally { pending?.free?.();seed?.free?.();if(ephemeral)session.free(); }
}

function compactValidation(validation,version){
  const rows=validation.rows||[],failed=rows.filter(r=>r.pass===false||r.ok===false);
  let maximumAbsolute=0,maximumBudgetFraction=0;
  for(const row of rows){if(Number.isFinite(row.absoluteError))maximumAbsolute=Math.max(maximumAbsolute,row.absoluteError);if(Number.isFinite(row.absoluteError)&&row.tolerance>0)maximumBudgetFraction=Math.max(maximumBudgetFraction,row.absoluteError/row.tolerance);}
  return {status:validation.status,version,budget:validation.budget,fixtures:validation.fixtures,checkCount:rows.length,failedCount:failed.length,maximumAbsolute,maximumBudgetFraction,
    rows:[{label:'Independent closed-form nLPT component checks',pass:validation.status==='pass',components:rows.length,maximumAbsolute,maximumBudgetFraction}],failures:failed.slice(0,16)};
}
async function validateLptContext(context,task){
  if(context.lptValidation)return context.lptValidation;
  const {runLptSelfTests}=await import('./gpu/lpt-check.js');
  const validation=await runLptSelfTests(context.device,{signal:task.controller.signal});
  if(cancelled(task))throw new Error('nLPT validation cancelled.');
  const summary=compactValidation(validation,'gpu-nlpt-axis-and-quadratic-v1');
  if(summary.status==='pass')context.lptValidation=summary;
  return summary;
}
async function validateDepositContext(context,task,wasm){
  if(context.depositValidation)return context.depositValidation;
  const {runExactDepositChecks}=await import('./gpu/deposit-check.js');
  const validation=await runExactDepositChecks(context.device,{signal:task.controller.signal,
    cpuReference:input=>copyF32(wasm.exact_sheet_density(input.dim,input.n,input.meshN,input.positions))});
  if(cancelled(task))throw new Error('Exact-deposition validation cancelled.');
  if(validation.status==='pass')context.depositValidation=validation;
  return validation;
}
async function validateCompensatedContext(context,task){
  if(context.compensatedValidation)return context.compensatedValidation;
  const {runCompensatedDirectSelfTests}=await import('./gpu/compensated-check.js');
  const validation=await runCompensatedDirectSelfTests(context.device,{signal:task.controller.signal});
  if(cancelled(task))throw new Error('Analytic recovery validation cancelled.');
  if(validation.status==='pass')context.compensatedValidation=validation;
  return validation;
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
    return { provenance, precision: provenance.startsWith('mixed-cpu') ? 'f32 GPU input/kernel with f64 CPU mode recovery' : provenance.includes('compensated') ? 'high/low f32 inputs and compensated GPU arithmetic' : provenance.includes('f32') ? 'f32 GPU input/kernel' : 'f64 CPU reference' };
  };
  execution.stageBackends = {
    trajectories: { backend: execution.pmEvolution?.backend === 'gpu' ? 'mixed' : 'cpu', lptBackend: execution.lptConstruction?.backend || 'cpu', pmBackend: execution.pmEvolution?.backend || 'cpu', spatialFieldPrecision: 'f32 nLPT tensors', statePrecision: execution.pmEvolution?.backend === 'gpu' ? 'LPT f64, PM f32' : 'f64', lptStatePrecision: 'f64', pmStatePrecision: execution.pmEvolution?.backend === 'gpu' ? 'f32' : 'f64', tangentPrecision: execution.pmEvolution?.backend === 'gpu' ? 'LPT f64, PM f32' : 'f64', gpuPackedInputPrecision: 'f32' },
    exactDeposit: { backend: execution.exactDeposit?.backend || 'cpu', provenance: execution.exactDeposit?.provenance || 'cpu-exact', inputPrecision: 'f32 positions', arithmeticPrecision: execution.exactDeposit?.arithmeticPrecision || 'f64 geometry and mass accumulation', outputPrecision: 'f32 density field' },
    pdfs: { backend: 'cpu', inputPrecision: 'f32 density fields', arithmeticPrecision: 'f64 histogram and moment accumulation', outputPrecision: 'f64 summaries' },
    griddedSpectra: { backend: 'cpu', inputPrecision: 'f32 density field', arithmeticPrecision: 'f64 complex FFT and binning', outputPrecision: 'f64 spectrum bins' },
    directPrimal: { lpt: directStage('lpt'), pm: directStage('pm') },
    directJvp: { lpt: directStage('lpt'), pm: directStage('pm') },
    analyticRecovery: { backend: execution.analyticRecovery?.backend || 'unused', provenance: execution.analyticRecovery?.provenance || null },
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
function attachBinaryFields(prepared, out) {
  if (!out || typeof prepared?.rho_f32 !== 'function') return out;
  if (!out.q || out.q.length === 0) out.q = prepared.q_f64();
  if (!out.x || out.x.length === 0) out.x = prepared.x_f64();
  if (!out.v || out.v.length === 0) out.v = prepared.v_f64();
  if (!out.pmX || out.pmX.length === 0) out.pmX = prepared.pm_x_f64();
  if (!out.pmV || out.pmV.length === 0) out.pmV = prepared.pm_v_f64();
  if (!out.rho || out.rho.length === 0) out.rho = prepared.rho_f32();
  if (!out.rhoP1 || out.rhoP1.length === 0) out.rhoP1 = prepared.rho_p1_f32();
  if (!out.pmRho || out.pmRho.length === 0) out.pmRho = prepared.pm_rho_f32();
  if (!out.cicRho || out.cicRho.length === 0) out.cicRho = prepared.cic_rho_f32();
  if (!out.tscRho || out.tscRho.length === 0) out.tscRho = prepared.tsc_rho_f32();
  if (out.customSeries && typeof prepared.custom_rho_f32 === 'function') {
    for (const [k, entry] of Object.entries(out.customSeries)) {
      if (entry && (!entry.rho || entry.rho.length === 0)) {
        entry.rho = prepared.custom_rho_f32(k);
      }
    }
  }
  return out;
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
