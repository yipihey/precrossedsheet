#!/usr/bin/env node
/** GPU browser checks with no npm dependencies. Requires an installed Chrome.
 * A missing adapter is reported as a skip; --require-gpu turns it into failure.
 * Outputs the exact self-test report, including adapter and precision metadata.
 */
import {spawn} from 'node:child_process';
import {createServer} from 'node:http';
import {readFile, mkdtemp, rm, writeFile} from 'node:fs/promises';
import {existsSync} from 'node:fs';
import {tmpdir,cpus,totalmem,platform,arch,release} from 'node:os';
import {resolve, join, extname, sep} from 'node:path';
import {fileURLToPath} from 'node:url';

const root=resolve(fileURLToPath(new URL('..',import.meta.url)));
const args=new Set(process.argv.slice(2));
const chromePath=process.env.CHROME_PATH || (process.platform==='darwin'
  ? '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome'
  : '/usr/bin/google-chrome');
if(!existsSync(chromePath)) {
  console.log(JSON.stringify({status:'skip',reason:'Chrome executable unavailable; set CHROME_PATH.'}));
  process.exit(args.has('--require-gpu')?1:0);
}
const profile=await mkdtemp(join(tmpdir(),'precrossedsheet-gpu-'));
const mime={'.js':'text/javascript','.mjs':'text/javascript','.wasm':'application/wasm','.html':'text/html','.css':'text/css','.json':'application/json'};
const server=createServer(async(req,res)=>{
  try {
    const path=resolve(root,'.'+decodeURIComponent(new URL(req.url,'http://localhost').pathname));
    if(!path.startsWith(root+sep)) throw new Error('Invalid path');
    const data=await readFile(path);
    res.writeHead(200,{'content-type':mime[extname(path)]||'application/octet-stream','cache-control':'no-store'});
    res.end(data);
  } catch {res.writeHead(404);res.end('Not found');}
});
await new Promise((resolve,reject)=>{server.once('error',reject);server.listen(0,'127.0.0.1',resolve);});
let chrome,ws;
try {
  const chromeArgs=['--headless=new','--remote-debugging-port=0',`--user-data-dir=${profile}`,'--no-first-run','--no-default-browser-check'];
  if(process.env.GPU_UNSAFE==='1') chromeArgs.push('--enable-unsafe-webgpu');
  if(process.env.CHROME_NO_SANDBOX==='1') chromeArgs.push('--no-sandbox');
  chrome=spawn(chromePath,chromeArgs,{stdio:['ignore','ignore','pipe']});
  const endpoint=await new Promise((ok,fail)=>{
    let stderr='';const timeout=setTimeout(()=>fail(new Error('Chrome startup timeout')),20000);
    chrome.stderr.on('data',b=>{stderr=(stderr+b).slice(-20000);const m=stderr.match(/DevTools listening on (ws:\/\/[^\s]+)/);if(m){clearTimeout(timeout);ok(m[1]);}});
    chrome.on('error',fail);chrome.on('exit',code=>{clearTimeout(timeout);fail(new Error(`Chrome exited ${code}: ${stderr.slice(-2000)}`));});
  });
  ws=new WebSocket(endpoint);
  await new Promise((ok,fail)=>{ws.onopen=ok;ws.onerror=fail;});
  let serial=0;const pending=new Map(),events=[];
  ws.onmessage=e=>{const m=JSON.parse(e.data);if(m.id){const p=pending.get(m.id);pending.delete(m.id);if(p){clearTimeout(p.timer);m.error?p.reject(new Error(JSON.stringify(m.error))):p.resolve(m.result);}}else if(m.method==='Runtime.exceptionThrown')events.push(m.params.exceptionDetails);};
  const send=(method,params={},sessionId)=>new Promise((resolve,reject)=>{const id=++serial;const timer=setTimeout(()=>{pending.delete(id);reject(new Error(`Browser command timed out: ${method}`));},120000);pending.set(id,{resolve,reject,timer});ws.send(JSON.stringify({id,method,params,...(sessionId?{sessionId}:{})}));});
  const target=await send('Target.createTarget',{url:'about:blank'});
  const {sessionId}=await send('Target.attachToTarget',{targetId:target.targetId,flatten:true});
  await send('Runtime.enable',{},sessionId);
  const query=new URLSearchParams({autorun:'1'});
  if(args.has('--fd-probe'))query.set('fd-probe','1');
  if(args.has('--fd-self'))query.set('fd-self','1');
  if(args.has('--full'))query.set('full','1');
  if(args.has('--benchmark'))query.set('benchmark','1');
  if(args.has('--schedule-benchmark'))query.set('schedule-benchmark','1');
  if(args.has('--resident')||args.has('--resident-benchmark'))query.set('resident','1');
  if(args.has('--resident-benchmark'))query.set('resident-benchmark','1');
  if(args.has('--resident-worker'))query.set('resident-worker','1');
  if(args.has('--fd-repeat'))query.set('fd-repeat','1');
  if(args.has('--fd')||args.has('--fd-benchmark'))query.set('fd','1');
  if(args.has('--fd-benchmark'))query.set('fd-benchmark','1');
  const url=`http://127.0.0.1:${server.address().port}/app/gpu-test.html?${query}`;
  await send('Page.navigate',{url},sessionId);
  const timeoutMs=Number(process.env.GPU_TEST_TIMEOUT_MS||900000),start=Date.now();
  let report,lastStage;
  while(Date.now()-start<timeoutMs){
    const result=await send('Runtime.evaluate',{expression:'({report:window.gpuTestReport || null,stage:document.querySelector("#status")?.textContent})',returnByValue:true},sessionId);
    report=result.result?.value?.report;
    const stage=result.result?.value?.stage;
    if(stage&&stage!==lastStage){lastStage=stage;if(args.has('--resident-benchmark')||args.has('--fd-benchmark'))console.error(stage);}
    if(report&&['pass','fail','skip'].includes(report.status))break;
    if(events.length)throw new Error(JSON.stringify(events));
    await new Promise(r=>setTimeout(r,500));
  }
  if(!report)throw new Error('GPU test page did not finish before timeout');
  report={...report,browser:await send('Browser.getVersion'),runner:{chromePath,flags:chromeArgs.filter(x=>!x.startsWith('--user-data-dir')),wallMilliseconds:Date.now()-start},
    machine:{cpu:cpus()[0]?.model??null,logicalCpus:cpus().length,memoryBytes:totalmem(),platform:platform(),architecture:arch(),release:release()}};
  const outputFlag=process.argv.indexOf('--output');
  if(outputFlag>=0){if(!process.argv[outputFlag+1])throw new Error('--output requires a path');await writeFile(resolve(process.argv[outputFlag+1]),JSON.stringify(report,null,2)+'\n');}
  const median=values=>{const sorted=[...values].filter(Number.isFinite).sort((a,b)=>a-b);return sorted.length?sorted[Math.floor(sorted.length/2)]:null;};
  const fdTimings=(report.fd?.timings??[]).map(t=>({request:t.request??null,cpuFiniteDifferenceMs:t.cpuFiniteDifferenceMs??null,gpuFiniteDifferenceWallMs:t.gpuFiniteDifferenceWallMs??null,gpuKernelMs:t.gpu?.kernelMs??null,repeats:t.repeats??null,cpuFdMedianMs:median(t.cpuFdWallMs??t.cpuMs??[]),gpuFdMedianMs:median(t.gpuFdWallMs??t.gpuWallMs??[]),gpuKernelMedianMs:median(t.gpuKernelMs??[]),pairedAccuracy:t.pairedAccuracy?.map(p=>({repeat:p.repeat,pass:p.pass,maximumWeakScaledRelative:p.comparison?.maximumWeakScaledRelative}))??null}));
  const concise={status:report.status,reason:report.reason??report.resident?.reason,browser:report.browser.product,adapter:report.deviceInfo?.adapter,
    kernelRows:report.kernel?.rows.length??report.rows?.length,labRows:report.lab?.rows.length,
    residentRows:report.resident?.rows?.length,residentBenchmarks:report.resident?.timings,
    probe:report.probe,fdRepeat:report.fdRepeat?{status:report.fdRepeat.status,repeatsPerCase:report.fdRepeat.repeatsPerCase,totalRepeats:report.fdRepeat.totalRepeats,endpointRepeatBudget:report.fdRepeat.endpointRepeatBudget,cases:report.fdRepeat.cases?.map(({label,request,repeats,status,inputHashes,outputsStable,runs})=>({label,request,repeats,status,inputHashes,cpuFiniteDifferenceHash:inputHashes?.cpuFiniteDifference,outputsStable,maximumComponentAbsoluteDifferenceVsFirst:Math.max(0,...(runs??[]).map(run=>run.componentRepeatError?.maximumAbsolute??Infinity)),runs:runs?.map(({caseLabel,repeat,finite,inputsUnchanged,sameAsFirst,accuracy,componentRepeatError,outputHashes})=>({caseLabel,repeat,finite,inputsUnchanged,sameAsFirst,accuracy,componentRepeatError,outputHashes}))})),runs:report.fdRepeat.runs?.map(({caseLabel,repeat,finite,inputsUnchanged,sameAsFirst,accuracy,componentRepeatError,outputHashes})=>({caseLabel,repeat,finite,inputsUnchanged,sameAsFirst,accuracy,componentRepeatError,outputHashes}))}:null,fdRows:report.fd?.rows?.length,fdSelfTest:{status:report.fd?.selfTest?.status,rows:report.fd?.selfTest?.rows?.map(r=>({label:r.label,pass:r.pass,maximumWeakScaledRelative:r.maximumWeakScaledRelative,maximumAbsoluteError:r.maximumAbsoluteError,failedModes:r.failedModes}))},fdNumericalCases:report.fd?.cases?.map(c=>({label:c.label,request:c.request,pass:c.pass,maximumWeakScaledRelative:c.accuracy?.maximumWeakScaledRelative,failedModes:c.accuracy?.failedModes,finalizerGatePass:c.finalizer?.gatePass,crossing:c.crossing})),fdTimings,
    gpuEligibleRequests:report.lab?.cases?.filter(c=>c.gpuEligible).length,totalRequests:report.lab?.cases?.length,
    failures:report.rows?.filter(r=>r.ok===false||r.pass===false),
    benchmarks:report.lab?.timings?.filter(t=>t.benchmark).map(t=>t.endToEndWorker
      ? {request:t.request,cpu:t.cpu,hybrid:t.hybrid,description:t.description}
      : {request:t.request,cpuDirectMs:t.cpuDirectMs,gpuDirectMs:t.gpuDirectMs,cpuFdMs:t.cpuFdMs,cpuTotalMs:t.cpuTotalMs,hybridTotalMs:t.hybridTotalMs,description:t.description}),
    scheduleBenchmarks:report.scheduleBenchmark?.rows.map(({samples,...row})=>row),wallMilliseconds:report.runner.wallMilliseconds};
  console.log(JSON.stringify(args.has('--verbose')?report:concise,null,2));
  if(report.status==='fail'||(args.has('--require-gpu')&&report.status==='skip'))process.exitCode=1;
} catch(error){console.error(error.stack||String(error));process.exitCode=1;}
finally {
  ws?.close();chrome?.kill('SIGTERM');
  await new Promise(r=>server.close(r));
  // Chromium may still have a log file open briefly; removal is best effort.
  await rm(profile,{recursive:true,force:true,maxRetries:4,retryDelay:250}).catch(()=>{});
}
