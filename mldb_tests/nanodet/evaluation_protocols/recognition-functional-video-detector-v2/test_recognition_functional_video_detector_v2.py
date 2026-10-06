from pathlib import Path
from mldb_v2.src.evaluation.evaluate_interface import _load_evaluation_callable
import importlib.util
import sys
import csv

ROOT = Path(__file__).resolve().parents[4] / "mldb_data"
IMPLEMENTATION = ROOT / "nanodet" / "evaluation_protocols" / "recognition-functional-video-detector-v2.py"

def _module():
    spec = importlib.util.spec_from_file_location("recognition_functional_video_detector_v2", IMPLEMENTATION)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module

def test_evaluate_callable_contract() -> None:
    assert callable(_load_evaluation_callable(ROOT, "nanodet/recognition-functional-video-detector-v2"))

def test_per_take_csv_artifacts(tmp_path: Path) -> None:
    module = _module()
    functional = tmp_path / "functional.csv"
    module._write_functional_take_summary_csv(functional, [{"id":"t1","evaluations":10,"eligible_frames":8,"exact_frame_rate":0.6,"completed_hand_exact_rate":0.7,"dora_exact_rate":0.8,"meld_exact_rate":0.4,"first_gt_streak3":{"eval_index":2,"video_time_sec":0.2},"first_product_confirm":{"eval_index":3,"video_time_sec":0.3,"exact":True}}])
    rows=list(csv.DictReader(functional.open(encoding="utf-8")))
    assert rows[0]["take_id"] == "t1"
    assert float(rows[0]["eligible_rate"]) == 0.8
    detector = tmp_path / "detector.csv"
    summary={"frames":10,"gt_tiles":20,"predicted_tiles":21,"correct":18,"substitutions":1,"insertions":2,"deletions":1,"invalid_detections":3,"fallback_order_frames":4,"precision":0.85,"recall":0.9,"f1":0.875,"exact_sequence_rate":0.5}
    module._write_detector_semantic_take_summary_csv(detector,{"t1":{region:dict(summary) for region in module._DIAGNOSTIC_REGIONS}})
    rows=list(csv.DictReader(detector.open(encoding="utf-8")))
    assert len(rows) == 3
    assert next(row for row in rows if row["region"]=="melds")["insertions"] == "2"
