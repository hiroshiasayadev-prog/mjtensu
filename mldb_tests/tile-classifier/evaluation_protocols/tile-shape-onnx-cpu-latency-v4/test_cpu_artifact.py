from pathlib import Path
from types import SimpleNamespace
import torch
from torch import nn
from mldb_v2.src.evaluation.evaluate_interface import _load_evaluation_callable
ROOT = Path(__file__).resolve().parents[4] / "mldb_data"

def test_cpu_receives_onnx_without_loading_torch_model(tmp_path):
    model = nn.Sequential(nn.Flatten(),nn.Linear(4096,35))
    path=tmp_path/"test.onnx"
    torch.onnx.export(model,torch.zeros(1,1,64,64),str(path),opset_version=16,
        input_names=["images"],output_names=["logits"],dynamic_axes={"images":{0:"batch"}})
    run = _load_evaluation_callable(ROOT,"tile-classifier/tile-shape-onnx-cpu-latency-v4")
    context = SimpleNamespace(model=None, inputs={"onnx_model":SimpleNamespace(
        data=path.read_bytes(),source_evaluation_result="demo/export-v1",artifact="onnx_model")},
        parameters={"batch_size":1,"warmup_runs":1,"measure_runs":10,"intra_op_threads":1,
            "inter_op_threads":1},work_dir=tmp_path)
    result=run(context)
    assert result.metrics["latency_p50_ms"]>0
    assert result.artifacts["latency_report"].is_file()
