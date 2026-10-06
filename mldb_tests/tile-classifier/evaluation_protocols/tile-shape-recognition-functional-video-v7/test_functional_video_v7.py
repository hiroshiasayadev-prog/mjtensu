from pathlib import Path
import csv
import importlib.util
import sys
from mldb_v2.src.evaluation.evaluate_interface import _load_evaluation_callable

ROOT = Path(__file__).resolve().parents[4] / "mldb_data"
IMPLEMENTATION = ROOT / "tile-classifier" / "evaluation_protocols" / "tile-shape-recognition-functional-video-v7.py"

def _module():
    spec = importlib.util.spec_from_file_location("recognition_functional_video_v7", IMPLEMENTATION)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module

def test_evaluate_callable_contract() -> None:
    assert callable(_load_evaluation_callable(ROOT, "tile-classifier/tile-shape-recognition-functional-video-v7"))

def test_per_take_csv_artifacts(tmp_path: Path) -> None:
    module = _module()
    functional = tmp_path / "functional.csv"
    module._write_functional_take_summary_csv(functional,[{"id":"t5","evaluations":20,"eligible_frames":15,"exact_frame_rate":0.1,"completed_hand_exact_rate":0.9,"dora_exact_rate":0.05,"meld_exact_rate":0.02,"first_gt_streak3":None,"first_product_confirm":{"eval_index":4,"video_time_sec":0.4,"exact":False}}])
    rows=list(csv.DictReader(functional.open(encoding="utf-8")))
    assert rows[0]["take_id"] == "t5"
    assert float(rows[0]["eligible_rate"]) == 0.75
    assert rows[0]["first_gt_streak3_time_sec"] == ""
    detector = tmp_path / "detector.csv"
    summary={"frames":10,"gt_tiles":20,"predicted_tiles":21,"correct":18,"substitutions":1,"insertions":2,"deletions":1,"invalid_detections":3,"fallback_order_frames":4,"precision":0.85,"recall":0.9,"f1":0.875,"exact_sequence_rate":0.5}
    module._write_detector_semantic_take_summary_csv(detector,{"t5":{region:dict(summary) for region in module._DIAGNOSTIC_REGIONS}})
    rows=list(csv.DictReader(detector.open(encoding="utf-8")))
    assert len(rows) == 3
