from pathlib import Path
from types import SimpleNamespace
import torch
from torch import nn
from mldb_v2.src.evaluation.evaluate_interface import _load_evaluation_callable
ROOT = Path(__file__).resolve().parents[4] / "mldb_data"

class TinyRGBClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Flatten(), nn.Linear(3 * 64 * 64, 2))
    def forward(self, x):
        return self.net(x)

def test_red_five_export_and_cpu_parity(tmp_path):
    run = _load_evaluation_callable(ROOT, "tile-classifier/red-five-onnx-export-v1")
    context = SimpleNamespace(model=SimpleNamespace(module=TinyRGBClassifier(), definition={"id":"demo/red-five-v1"}), work_dir=tmp_path)
    result = run(context)
    assert result.artifacts["onnx_model"].is_file()
    assert result.metrics["onnx_parity_max_abs_error"] < 1e-3
