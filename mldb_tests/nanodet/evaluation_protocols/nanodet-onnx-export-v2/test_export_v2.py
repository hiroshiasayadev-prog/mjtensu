from pathlib import Path
from types import SimpleNamespace
import torch
from torch import nn
from mldb_v2.src.evaluation.evaluate_interface import _load_evaluation_callable
ROOT=Path(__file__).resolve().parents[4]/'mldb_data'
class TinyNanoHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv=nn.Conv2d(3,33,1)
    def forward(self, x):
        raw=self.conv(x).mean(dim=(2,3)).unsqueeze(1).repeat(1,10,1)
        if torch.onnx.is_in_onnx_export():
            return torch.cat((raw[..., :1].sigmoid(),raw[...,1:]),dim=-1)
        return raw

def test_onnx_export_head_sigmoid_matches(tmp_path):
    fn=_load_evaluation_callable(ROOT,'nanodet/nanodet-onnx-export-v2')
    result=fn(SimpleNamespace(model=SimpleNamespace(module=TinyNanoHead(),definition={'id':'demo/tiny-nano'}),work_dir=tmp_path))
    assert result.metrics['onnx_parity_max_abs_error'] < 0.0001
    assert result.artifacts['onnx_model'].is_file()
