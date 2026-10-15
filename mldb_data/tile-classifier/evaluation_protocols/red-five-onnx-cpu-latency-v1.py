from __future__ import annotations
import hashlib,json,math,statistics,time
from pathlib import Path
import numpy as np
import onnxruntime as ort
from mldb_v2.src.evaluation.evaluate_interface import EvaluationCandidate

def evaluate(context):
    if context.model is not None or not context.inputs or 'onnx_model' not in context.inputs:
        raise ValueError('RED5 ONNX-only CPU Eval requires producer artifact')
    src=context.inputs['onnx_model']; p=context.parameters
    batch=int(p['batch_size']); warmup=int(p['warmup_runs']); runs=int(p['measure_runs']); intra=int(p['intra_op_threads']); inter=int(p['inter_op_threads'])
    if batch<1 or warmup<1 or runs<10 or intra<1 or inter<1: raise ValueError('invalid benchmark parameters')
    opts=ort.SessionOptions(); opts.intra_op_num_threads=intra; opts.inter_op_num_threads=inter; opts.execution_mode=ort.ExecutionMode.ORT_SEQUENTIAL; opts.graph_optimization_level=ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session=ort.InferenceSession(src.data,sess_options=opts,providers=['CPUExecutionProvider'])
    if session.get_providers()!=['CPUExecutionProvider'] or len(session.get_inputs())!=1 or len(session.get_outputs())!=1: raise ValueError('invalid ORT session')
    name=session.get_inputs()[0].name; x=np.zeros((batch,3,64,64),dtype=np.float32); check=session.run(None,{name:x})[0]
    if check.shape!=(batch,2): raise ValueError(f'unexpected RED5 output shape {check.shape}')
    for _ in range(warmup): session.run(None,{name:x})
    vals=[]
    for _ in range(runs):
        t=time.perf_counter_ns(); session.run(None,{name:x}); vals.append((time.perf_counter_ns()-t)/1e6)
    vals.sort(); metrics={'latency_p50_ms':float(statistics.median(vals)),'latency_p95_ms':float(vals[max(0,math.ceil(.95*len(vals))-1)]),'latency_mean_ms':float(statistics.fmean(vals))}
    report={'schema':'mjtensu.recognition/red-five-onnx-cpu-latency/v1','source_evaluation_result':src.source_evaluation_result,'onnx_sha256':hashlib.sha256(src.data).hexdigest(),'onnx_bytes':len(src.data),'input_shape':list(x.shape),'output_shape':list(check.shape),'metrics_ms':metrics,'samples_ms':vals,'warmup_runs':warmup,'environment':{'onnxruntime_version':ort.__version__,'active_providers':session.get_providers(),'intra_op_threads':intra,'inter_op_threads':inter,'execution_mode':'ORT_SEQUENTIAL','graph_optimization_level':'ORT_ENABLE_ALL'},'measurement_scope':'ORT session.run only'}
    out=Path(context.work_dir); out.mkdir(parents=True,exist_ok=True); path=out/'red-five-onnx-cpu-latency.json'; path.write_text(json.dumps(report,indent=2)+'\n')
    return EvaluationCandidate(metrics=metrics,artifacts={'latency_report':path})
