"""Benchmark a complete NanoDet detector graph on physical iPhone Safari / ORT Web."""
from __future__ import annotations
import base64
import hashlib
import inspect
import json
import math
import os
import statistics
import time
import urllib.error
import urllib.request
from pathlib import Path
import torch
from mldb_v2.src.evaluation.evaluate_interface import EvaluationCandidate

RUNNER = 'http://192.168.11.22:8877'
ORT_DIST = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.27.0/dist/'
TIMEOUT = 300

PAGE = r"""<!doctype html><meta name="viewport" content="width=device-width,initial-scale=1">
<pre id="status">loading</pre><script type="module">
let ort;
const statusNode=document.getElementById('status');
async function report(data) {
 const q=new URLSearchParams(location.search);
 const url='/v1/jobs/'+encodeURIComponent(q.get('runner_job_id'))+'/result?token='+encodeURIComponent(q.get('runner_token'));
 const r=await fetch(url,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(data)});
 if(!r.ok) throw Error('result callback '+r.status);
}
function decode(b64) {
 const bin=atob(b64), out=new Uint8Array(bin.length);
 for(let i=0;i<bin.length;i++) out[i]=bin.charCodeAt(i);
 return out;
}
try {
 ort=await import(__ORT_MODULE__);
 ort.env.wasm.proxy=false;ort.env.wasm.simd=true;ort.env.wasm.numThreads=1;
 ort.env.wasm.wasmPaths=__ORT_DIST__;
 const sess=await ort.InferenceSession.create(decode(__MODEL__),{executionProviders:['wasm'],graphOptimizationLevel:'all',executionMode:'sequential'});
 if(sess.inputNames.length!==1||sess.outputNames.length!==1)throw Error('input/output count');
 const ik=sess.inputNames[0],ok=sess.outputNames[0], n=__POSITIONS__;
 const input=new ort.Tensor('float32',new Float32Array(3*320*320),[1,3,320,320]);
 const check=(o)=>{const v=o[ok];if(!v||v.dims.join(',')!==('1,'+n+',33'))throw Error('unexpected output '+JSON.stringify(v?.dims));return v;};
 statusNode.textContent='parity';
 let result=check(await sess.run({[ik]:input}));
 const positions=[0,1,32,33,64,97,193,391,997,1999,5999,9999,19999,29999,n*33-33,n*33-1].filter(i=>i<result.data.length);
 const sample=positions.map(i=>Number(result.data[i]));
 statusNode.textContent='warmup';
 for(let i=0;i<__WARMUP__;i++)check(await sess.run({[ik]:input}));
 let vals=[],totals=[];
 statusNode.textContent='measuring';
 for(let b=0;b<__BLOCKS__;b++){
  const t=performance.now();
  for(let i=0;i<__RUNS__;i++)result=check(await sess.run({[ik]:input}));
  const delta=performance.now()-t;totals.push(delta);vals.push(delta/__RUNS__);
 }
 await report({ok:true,parity:{dims:[1,n,33],indices:positions,values:sample},benchmark:{
   batch_size:1,warmup_runs:__WARMUP__,runs_per_block:__RUNS__,measure_blocks:__BLOCKS__,
   total_measure_runs:__BLOCKS__*__RUNS__,samples_ms:vals,block_totals_ms:totals},
   environment:{provider:'wasm-simd',num_threads:1,wasm_proxy:false,
      hardware_concurrency:navigator.hardwareConcurrency||1,
      user_agent:navigator.userAgent,secure_context:globalThis.isSecureContext===true}});
 statusNode.textContent='completed';
 await sess.release();
} catch(e) {
 statusNode.textContent=String(e);
 try{await report({ok:false,error:String(e)});}catch(_){}
}
</script>"""

def _json_request(method: str, url: str, payload=None):
    data=None if payload is None else json.dumps(payload).encode()
    hdr={} if payload is None else {'Content-Type':'application/json'}
    req=urllib.request.Request(url,data=data,headers=hdr,method=method)
    with urllib.request.urlopen(req,timeout=20) as resp:
        return json.loads(resp.read().decode())

def _run_page(html: str):
    root=os.environ.get('MLDB_IPHONE_BROWSER_RUNNER_URL',RUNNER).rstrip('/')
    health=_json_request('GET',root+'/healthz')
    if health.get('ok') is not True:
        raise RuntimeError('iPhone browser runner cannot execute: device not seen')
    job=_json_request('POST',root+'/v1/jobs',{'document':html,'result_timeout_sec':TIMEOUT})
    if not isinstance(job.get('id'),str):raise RuntimeError('runner did not return id')
    stop=time.monotonic()+TIMEOUT+30
    while time.monotonic()<stop:
        state=_json_request('GET',root+'/v1/jobs/'+job['id'])
        if state.get('state')=='completed':
            result=state.get('result')
            if not isinstance(result,dict) or result.get('ok') is not True:
                raise RuntimeError('browser result failed: '+str(result))
            return result,health
        if state.get('state')=='failed':raise RuntimeError(str(state.get('error')))
        if state.get('state') not in ('queued','launching','launched'):
            raise RuntimeError('runner state '+str(state.get('state')))
        time.sleep(.25)
    raise RuntimeError('iPhone runner timeout')

def _percentile(values,p):
    values=sorted(values)
    return float(values[max(0,math.ceil(p*len(values))-1)])

def evaluate(context):
    params=context.parameters
    warmup=int(params['warmup_runs']);runs=int(params['runs_per_block']);blocks=int(params['measure_blocks'])
    if min(warmup,runs,blocks)<1:raise ValueError('benchmark counts must be positive')
    root=Path(context.work_dir);root.mkdir(parents=True,exist_ok=True)
    model=context.model.module.to('cpu').eval()
    x=torch.zeros((1,3,320,320),dtype=torch.float32)
    with torch.no_grad(): raw=model(x)
    if raw.ndim!=3 or raw.shape[0]!=1 or raw.shape[2]!=33:
        raise RuntimeError('NanoDet dense output contract does not match [1,N,33]')
    expected=torch.cat((raw[...,:1].sigmoid(),raw[...,1:]),dim=-1).contiguous().view(-1)
    onnx_path=root/'detector.onnx'
    kwargs={'opset_version':16,'input_names':['images'],'output_names':['dense_detections'],'do_constant_folding':True}
    if 'dynamo' in inspect.signature(torch.onnx.export).parameters:kwargs['dynamo']=False
    torch.onnx.export(model,x,str(onnx_path),**kwargs)
    import onnx
    onnx.checker.check_model(onnx.load(str(onnx_path)))
    data=onnx_path.read_bytes()
    html=PAGE
    mapping={'__ORT_MODULE__':json.dumps(ORT_DIST+'ort.wasm.min.mjs'),'__ORT_DIST__':json.dumps(ORT_DIST),
        '__MODEL__':json.dumps(base64.b64encode(data).decode()),'__POSITIONS__':str(int(raw.shape[1])),
        '__WARMUP__':str(warmup),'__RUNS__':str(runs),'__BLOCKS__':str(blocks)}
    for k,v in mapping.items():html=html.replace(k,v)
    result,health=_run_page(html)
    par=result.get('parity',{})
    if par.get('dims')!=[1,int(raw.shape[1]),33]:raise RuntimeError('parity shape mismatch')
    indices=par.get('indices',[]);vals=par.get('values',[])
    if len(indices)!=len(vals) or len(indices)<8:raise RuntimeError('parity sample mismatch')
    differences=[]
    for i,v in zip(indices,vals):
        actual=float(v);reference=float(expected[int(i)])
        if not math.isfinite(actual) or abs(reference-actual)>0.003+0.003*abs(reference):
            raise RuntimeError(f'ORT Web detector parity mismatch at {i}')
        differences.append(abs(reference-actual))
    bench=result['benchmark']
    if any(bench.get(k)!=v for k,v in {'batch_size':1,'warmup_runs':warmup,'runs_per_block':runs,
      'measure_blocks':blocks,'total_measure_runs':runs*blocks}.items()):
        raise RuntimeError('browser benchmark counts differ')
    samples=bench.get('samples_ms')
    if not isinstance(samples,list) or len(samples)!=blocks or not all(
      isinstance(v,(int,float)) and not isinstance(v,bool) and math.isfinite(v) and v>0 for v in samples):
        raise RuntimeError('invalid iPhone samples')
    env=result.get('environment',{})
    if env.get('provider')!='wasm-simd' or env.get('num_threads')!=1 or env.get('wasm_proxy') is not False:
        raise RuntimeError('unexpected browser execution provider')
    metrics={'latency_p50_ms':float(statistics.median(samples)),
      'latency_p95_ms':_percentile(samples,.95),'latency_mean_ms':float(statistics.fmean(samples))}
    report={'schema':'mjtensu.nanodet/iphone-detector-ort-web-latency/v1',
      'scope':'full detector backbone + GhostPAN + NanoDet head, ORT Web session.run; no NMS/preprocessing',
      'strides':list(model.head.strides),'dense_positions':int(raw.shape[1]),
      'model_sha256':hashlib.sha256(data).hexdigest(),'model_bytes':len(data),
      'benchmark':bench,'parity_max_abs_error':max(differences),
      'environment':{**env,'runner_device_sha256':hashlib.sha256(str(health.get('udid')).encode()).hexdigest()}}
    report_path=root/'iphone-detector-latency.json';report_path.write_text(json.dumps(report,indent=2)+'\n')
    return EvaluationCandidate(metrics=metrics,artifacts={'onnx_model':onnx_path,'latency_report':report_path})
