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
  ws.onmessage=e=>{const m=JSON.parse(e.data);if(m.id){const p=pending.get(m.id);pending.delete(m.id);if(p)m.error?p.reject(new Error(JSON.stringify(m.error))):p.resolve(m.result);}else if(m.method==='Runtime.exceptionThrown')events.push(m.params.exceptionDetails);};
  const send=(method,params={},sessionId)=>new Promise((resolve,reject)=>{const id=++serial;pending.set(id,{resolve,reject});ws.send(JSON.stringify({id,method,params,...(sessionId?{sessionId}:{})}));});
  const target=await send('Target.createTarget',{url:'about:blank'});
  const {sessionId}=await send('Target.attachToTarget',{targetId:target.targetId,flatten:true});
  await send('Runtime.enable',{},sessionId);
  const query=new URLSearchParams({autorun:'1'});
  if(args.has('--full'))query.set('full','1');
  if(args.has('--benchmark'))query.set('benchmark','1');
  if(args.has('--schedule-benchmark'))query.set('schedule-benchmark','1');
  const url=`http://127.0.0.1:${server.address().port}/app/gpu-test.html?${query}`;
  await send('Page.navigate',{url},sessionId);
  const timeoutMs=Number(process.env.GPU_TEST_TIMEOUT_MS||900000),start=Date.now();
  let report;
  while(Date.now()-start<timeoutMs){
    const result=await send('Runtime.evaluate',{expression:'window.gpuTestReport || null',returnByValue:true},sessionId);
    report=result.result?.value;
    if(report&&['pass','fail','skip'].includes(report.status))break;
    if(events.length)throw new Error(JSON.stringify(events));
    await new Promise(r=>setTimeout(r,500));
  }
  if(!report)throw new Error('GPU test page did not finish before timeout');
  report={...report,browser:await send('Browser.getVersion'),runner:{chromePath,flags:chromeArgs.filter(x=>!x.startsWith('--user-data-dir')),wallMilliseconds:Date.now()-start},
    machine:{cpu:cpus()[0]?.model??null,logicalCpus:cpus().length,memoryBytes:totalmem(),platform:platform(),architecture:arch(),release:release()}};
  const outputFlag=process.argv.indexOf('--output');
  if(outputFlag>=0){if(!process.argv[outputFlag+1])throw new Error('--output requires a path');await writeFile(resolve(process.argv[outputFlag+1]),JSON.stringify(report,null,2)+'\n');}
  const concise={status:report.status,reason:report.reason,browser:report.browser.product,adapter:report.deviceInfo?.adapter,
    kernelRows:report.kernel?.rows.length??report.rows?.length,labRows:report.lab?.rows.length,
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
