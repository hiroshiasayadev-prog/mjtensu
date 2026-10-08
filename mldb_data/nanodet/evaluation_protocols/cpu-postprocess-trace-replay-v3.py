"""CPU-only fixed-trace counterfactual, no detector forward or classifier rerun."""
from __future__ import annotations

import json
from pathlib import Path

from mldb_v2.src.evaluation.evaluate_interface import EvaluationCandidate
from mldb_data.nanodet.lib import nanodet_postprocess_trace_replay_v3 as replay

_ALLOWED = frozenset(name for name, _, _ in replay.CASES)


def evaluate(context):
    choice = str(context.parameters["postprocess_condition"])
    if choice not in _ALLOWED:
        raise ValueError(f"unsupported postprocess condition {choice!r}")
    trace = Path(context.corpus.root) / "prediction_trace"
    if not trace.is_file():
        raise FileNotFoundError(f"sealed trace corpus absent: {trace}")
    out = Path(context.work_dir) / "cpu-postprocess-replay"
    replay.run(trace, out)
    report = json.loads((out / "comparison.json").read_text(encoding="utf-8"))
    if report["frames"] != 1497:
        raise ValueError("unexpected trace length")
    parts = {row["region"]: row for row in report["conditions"] if row["condition"] == choice}
    if set(parts) != set(replay.REGIONS):
        raise ValueError("missing replay regions")
    hand, meld, dora = (parts[r] for r in replay.REGIONS)
    if choice == "baseline":
        if any(parts[r]["removed_boxes"] for r in replay.REGIONS):
            raise ValueError("baseline unexpectedly suppresses detections")
    summary = {
        "schema": "mjtensu.nanodet/cpu-postprocess-trace-replay-result/v1",
        "postprocess_condition": choice,
        "source_corpus": str(context.corpus.definition.id),
        "source_model": str(context.model.definition.id) if context.model is not None else None,
        "scope": "CPU replay on already-postprocessed 1497-frame trace; classifier results and meld groups frozen",
        "recomputed": "Levenshtein semantic scores after applying additional postprocess suppression",
        "not_recomputed": ["model inference", "classifier", "meld regroup", "product stabilization"],
        "regions": parts,
    }
    report_path = out / "selected_result.json"
    report_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    context.telemetry.report_scalar(group="postprocess", series="hand_exact_rate", value=float(hand["exact_frame_rate"]), step=0)
    return EvaluationCandidate(
        metrics={
            "hand_exact_rate_proxy": float(hand["exact_frame_rate"]),
            "hand_semantic_f1_proxy": float(hand["semantic_f1_proxy"]),
            "hand_overlap_frames": int(hand["remaining_iou50_overlap_frames"]),
            "hand_boxes_removed": int(hand["removed_boxes"]),
            "hand_exact_lost": int(hand["exact_lost"]),
            "meld_exact_rate_proxy": float(meld["exact_frame_rate"]),
            "meld_semantic_f1_proxy": float(meld["semantic_f1_proxy"]),
            "meld_overlap_frames": int(meld["remaining_iou50_overlap_frames"]),
            "meld_exact_lost": int(meld["exact_lost"]),
            "dora_exact_rate_proxy": float(dora["exact_frame_rate"]),
            "dora_exact_lost": int(dora["exact_lost"]),
        },
        artifacts={
            "selected_result": report_path,
            "comparison_csv": out / "comparison.csv",
            "exact_frame_changes": out / "exact_frame_changes.jsonl",
        },
    )
