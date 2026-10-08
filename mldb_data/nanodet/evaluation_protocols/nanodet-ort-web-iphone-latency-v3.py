"""iPhone Safari ORT Web latency from an accepted ONNX artifact, without PyTorch."""
from __future__ import annotations
import base64
import hashlib
import json
import math
import os
import statistics
import time
import urllib.request
from pathlib import Path
from mldb_v2.src.evaluation.evaluate_interface import EvaluationCandidate

RUNNER = "http://192.168.11.22:8877"
ORT_DIST = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.27.0/dist/"
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
 const ik=sess.inputNames[0],ok=sess.outputNames[0];
 const input=new ort.Tensor('float32',new Float32Array(3*320*320),[1,3,320,320]);
 const first=await sess.run({[ik]:input});
 const dims=first[ok]?.dims;
 if(!dims||dims.length!==3||dims[0]!==1||dims[2]!==33||dims[1]<=0)throw Error('unexpected ONNX detector output');
 const n=dims[1];
 const check=(o)=>{const v=o[ok];if(!v||v.dims.join(',')!==('1,'+n+',33'))throw Error('unexpected output '+JSON.stringify(v?.dims));return v;};
 statusNode.textContent='parity';
 let result=check(first);
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
    if context.model is not None or not context.inputs or "onnx_model" not in context.inputs:
        raise ValueError("iPhone Eval requires the declared ONNX stage input")
    if type(context.inputs["onnx_model"].data) is not bytes:
        raise ValueError("NanoDet ONNX-only iPhone Eval requires verified ONNX input")
    if not context.inputs["onnx_model"].data or not context.inputs["onnx_model"].source_evaluation_result:
        raise ValueError("source ONNX EvaluationResult missing")
    p = context.parameters
    warmup, runs, blocks = (int(p[k]) for k in ("warmup_runs", "runs_per_block", "measure_blocks"))
    if min(warmup, runs, blocks) < 1:
        raise ValueError("benchmark counts must be positive")
    html = PAGE
    mapping = {
        "__ORT_MODULE__": json.dumps(ORT_DIST + "ort.wasm.min.mjs"),
        "__ORT_DIST__": json.dumps(ORT_DIST),
        "__MODEL__": json.dumps(base64.b64encode(context.inputs["onnx_model"].data).decode()),
        "__WARMUP__": str(warmup),
        "__RUNS__": str(runs),
        "__BLOCKS__": str(blocks),
    }
    for key, value in mapping.items():
        html = html.replace(key, value)
    browser, health = _run_page(html)
    parity = browser.get("parity")
    if not isinstance(parity, dict):
        raise ValueError("detector output shape evidence missing")
    dims = parity.get("dims")
    if (type(dims) is not list or len(dims) != 3
            or dims[0] != 1 or type(dims[1]) is not int or dims[1] <= 0 or dims[2] != 33):
        raise ValueError("detector ONNX shape mismatch")
    indices, values = parity.get("indices"), parity.get("values")
    if (type(indices) is not list or type(values) is not list
            or len(indices) != len(values) or len(indices) < 8
            or any(type(v) not in (int, float) or not math.isfinite(v) for v in values)):
        raise ValueError("detector ONNX sample validation failed")
    bench = browser.get("benchmark")
    if not isinstance(bench, dict) or any(
        bench.get(k) != v for k, v in {
            "batch_size": 1, "warmup_runs": warmup, "runs_per_block": runs,
            "measure_blocks": blocks, "total_measure_runs": runs * blocks
        }.items()
    ):
        raise ValueError("benchmark count mismatch")
    samples = bench.get("samples_ms")
    if (type(samples) is not list or len(samples) != blocks
            or any(type(v) not in (int, float) or not math.isfinite(v) or v <= 0 for v in samples)):
        raise ValueError("benchmark sample values invalid")
    env = browser.get("environment")
    if (not isinstance(env, dict) or env.get("provider") != "wasm-simd"
            or env.get("num_threads") != 1 or env.get("wasm_proxy") is not False):
        raise ValueError("unexpected iPhone runtime")
    metrics = {
        "latency_p50_ms": float(statistics.median(samples)),
        "latency_p95_ms": _percentile(samples, .95),
        "latency_mean_ms": float(statistics.fmean(samples)),
    }
    report = {
        "schema": "mjtensu.nanodet/iphone-onnx-input-latency/v1",
        "source_evaluation_result": context.inputs["onnx_model"].source_evaluation_result,
        "onnx_sha256": hashlib.sha256(context.inputs["onnx_model"].data).hexdigest(),
        "onnx_bytes": len(context.inputs["onnx_model"].data),
        "dense_positions": dims[1],
        "benchmark": bench, "metrics_ms": metrics,
        "environment": {
            **env, "runner_device_sha256": hashlib.sha256(str(health.get("udid")).encode()).hexdigest()
        },
        "verification": "browser shape/finite check; export parity checked by source EvaluationResult",
    }
    work_dir = Path(context.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    path = work_dir / "iphone-detector-onnx-latency.json"
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return EvaluationCandidate(metrics=metrics, artifacts={"latency_report": path})
