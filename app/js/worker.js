// Keep every calculation off the page thread. The caller terminates this worker
// when a newer control state arrives, which also cancels a running WASM call.
import init, { lab_run } from '../pkg/hcc_core.js';

let ready;
self.onmessage = async ({ data }) => {
  if (!data || data.type !== 'run') return;
  const { requestId, state } = data;
  try {
    if (!ready) ready = init();
    await ready;
    const result = lab_run(JSON.stringify(state));
    const parsed = typeof result === 'string' ? JSON.parse(result) : result;
    self.postMessage({ requestId, result: parsed });
  } catch (error) {
    self.postMessage({ requestId, error: String(error?.message || error) });
  }
};
