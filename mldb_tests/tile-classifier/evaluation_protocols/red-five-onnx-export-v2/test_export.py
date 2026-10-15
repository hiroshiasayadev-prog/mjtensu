from pathlib import Path
from types import SimpleNamespace
import torch
from mldb_v2.src.catalog.architecture_build import _load_architecture_build
from mldb_v2.src.evaluation.evaluate_interface import _load_evaluation_callable
ROOT=Path(__file__).resolve().parents[4]/'mldb_data'

def test_red_feature_export_includes_preprocessing(tmp_path):
    model=_load_architecture_build(ROOT,'tile-classifier/red-five-c8-rgb-exr-dw222-fields6-12-24-48-v1')()
    model.set_input_normalization((0.5,0.5,0.5,0.2),(0.2,0.2,0.2,0.3))
    run=_load_evaluation_callable(ROOT,'tile-classifier/red-five-onnx-export-v2')
    result=run(SimpleNamespace(model=SimpleNamespace(module=model,definition={'id':'demo/red5-exr'}),work_dir=tmp_path))
    assert result.artifacts['onnx_model'].is_file()
    assert result.metrics['onnx_parity_max_abs_error'] < 1e-3
