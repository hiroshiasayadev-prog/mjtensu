from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn

from mldb_v2.src.evaluation.evaluate_interface import _load_evaluation_callable

ROOT = Path(__file__).resolve().parents[4] / "mldb_data"


class TinyDetector(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 33, 1)

    def forward(self, images):
        output = self.conv(images).mean(dim=(2, 3))
        return output.unsqueeze(1).repeat(1, 10, 1)


def test_export_and_cpu_parity(tmp_path):
    evaluate = _load_evaluation_callable(ROOT, "nanodet/nanodet-onnx-export-v1")
    context = SimpleNamespace(model=SimpleNamespace(module=TinyDetector(), definition={"id": "demo/model-v1"}), work_dir=tmp_path)
    result = evaluate(context)
    assert result.artifacts["onnx_model"].is_file()
    assert result.metrics["onnx_parity_max_abs_error"] < 1e-3
