// Small worker-safe WebGPU utilities adapted from Hopf–Cole Cosmology.
// Copyright (c) 2026 Tom Abel. See reuse-manifest.json for the pinned source.
export const STORAGE = 0x80 | 0x04 | 0x08;
export const MAP_READ = 0x01 | 0x08;
export const UNIFORM = 0x40 | 0x08;

export function makePipeline(device, code, entryPoint, entries, label) {
  const module = device.createShaderModule({ label, code });
  return module.getCompilationInfo().then((info) => {
    const errors = info.messages.filter((m) => m.type === 'error');
    if (errors.length) throw new Error(`${label} shader compilation failed: ${errors.map((m) => m.message).join('; ')}`);
    const bgl = device.createBindGroupLayout({ label: `${label} layout`, entries });
    const pipeline = device.createComputePipeline({
      label,
      layout: device.createPipelineLayout({ bindGroupLayouts: [bgl] }),
      compute: { module, entryPoint },
    });
    return { pipeline, bgl, module };
  });
}

export function bindGroup(device, layout, buffers, label) {
  return device.createBindGroup({
    label, layout,
    entries: buffers.map((buffer, binding) => ({ binding, resource: { buffer } })),
  });
}

export function dispatch1d(pass, count, workgroupSize = 64) {
  const groups = Math.max(1, Math.ceil(count / workgroupSize));
  const max = 65535;
  if (groups <= max) pass.dispatchWorkgroups(groups);
  else pass.dispatchWorkgroups(max, Math.ceil(groups / max));
}

export async function readBuffer(device, source, byteLength, Type = Float32Array) {
  const size = Math.max(4, (byteLength + 3) & ~3);
  const staging = device.createBuffer({ label: 'direct readback', size, usage: MAP_READ });
  try {
    const encoder = device.createCommandEncoder({ label: 'direct readback copy' });
    encoder.copyBufferToBuffer(source, 0, staging, 0, size);
    device.queue.submit([encoder.finish()]);
    await staging.mapAsync(1, 0, size);
    return new Type(staging.getMappedRange(0, byteLength).slice(0));
  } finally {
    try { if (staging.mapState === 'mapped') staging.unmap(); } catch { /* device may be lost */ }
    staging.destroy();
  }
}
