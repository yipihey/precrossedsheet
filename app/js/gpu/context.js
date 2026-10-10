// Worker-safe compute-only WebGPU capability and device setup. No DOM globals.
export const CONTEXT_PROVENANCE = Object.freeze({
  module: 'precrossedsheet/app/js/gpu/context.js',
  policy: 'compute-only; one owned device per caller unless injected',
  revision: 'precrossedsheet-gpu-p0-2026-10-10',
});

function withTimeout(promise, timeoutMs, label) {
  let timer;
  return Promise.race([
    promise,
    new Promise((_, reject) => { timer=setTimeout(()=>reject(new Error(`${label} timed out after ${timeoutMs} ms`)),timeoutMs); }),
  ]).finally(()=>clearTimeout(timer));
}

export async function createGpuContext(options = {}) {
  const nav = options.navigator ?? globalThis.navigator;
  const started = performance.now();
  const timeoutMs = options.timeoutMs ?? 15000;
  const reasons = [];
  let adapter;
  try {
    if (options.forceFailure === 'missing-api') throw new Error('forced: missing-api');
    if (!nav?.gpu) throw new Error('WebGPU is unavailable in this worker context');
    const ac = new AbortController();
    const timer = setTimeout(() => ac.abort(), timeoutMs);
    try {
      adapter = await Promise.race([
        nav.gpu.requestAdapter({ powerPreference: options.powerPreference ?? 'high-performance' }),
        new Promise((_, reject) => ac.signal.addEventListener('abort', () => reject(new Error(`adapter request timed out after ${timeoutMs} ms`)), { once: true })),
      ]);
    } finally { clearTimeout(timer); }
    if (!adapter) throw new Error('WebGPU returned no adapter');
    const requested = options.requiredLimits ?? {};
    for (const [name, value] of Object.entries(requested)) {
      if ((adapter.limits?.[name] ?? 0) < value) throw new Error(`adapter limit ${name} is below required value ${value}`);
    }
    const device = await withTimeout(adapter.requestDevice({
      label: 'precrossedsheet direct P0',
      requiredFeatures: options.requiredFeatures ?? [],
      requiredLimits: requested,
    }),timeoutMs,'WebGPU device request');
    const info = adapter.info ?? {};
    const limitNames=['maxTextureDimension1D','maxTextureDimension2D','maxTextureDimension3D','maxBindGroups','maxBindingsPerBindGroup','maxDynamicUniformBuffersPerPipelineLayout','maxDynamicStorageBuffersPerPipelineLayout','maxSampledTexturesPerShaderStage','maxSamplersPerShaderStage','maxStorageBuffersPerShaderStage','maxStorageTexturesPerShaderStage','maxUniformBuffersPerShaderStage','maxUniformBufferBindingSize','maxStorageBufferBindingSize','maxBufferSize','maxVertexBuffers','maxVertexAttributes','maxVertexBufferArrayStride','maxInterStageShaderComponents','maxInterStageShaderVariables','maxColorAttachments','maxColorAttachmentBytesPerSample','maxComputeWorkgroupStorageSize','maxComputeInvocationsPerWorkgroup','maxComputeWorkgroupSizeX','maxComputeWorkgroupSizeY','maxComputeWorkgroupSizeZ','maxComputeWorkgroupsPerDimension'];
    const limits=Object.fromEntries(limitNames.map((name)=>[name,device.limits?.[name]]).filter(([,v])=>typeof v==='number'));
    let disposed = false;
    device.addEventListener?.('uncapturederror', (event) => options.onUncapturedError?.(event.error));
    const lost = device.lost.then((reason) => {
      if (!disposed) options.onDeviceLost?.({ reason: reason?.message ?? 'device lost' });
      return reason;
    });
    const context = {
      device,
      adapterInfo: { vendor: info.vendor ?? '', architecture: info.architecture ?? '', device: info.device ?? '', description: info.description ?? '' },
      limits,
      features: [...(device.features ?? [])],
      provenance: { ...CONTEXT_PROVENANCE, setupMs: performance.now() - started },
      lost,
      dispose() { if (!disposed) { disposed = true; device.destroy(); } },
    };
    return context;
  } catch (error) {
    reasons.push(error?.message ?? String(error));
    error.gpuFallbackReason = reasons.join('; ');
    throw error;
  }
}
