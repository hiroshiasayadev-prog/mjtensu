"""Formal INV-041 Soft-NMS + union-coverage replay on an upstream functional trace."""
from __future__ import annotations

import json
from pathlib import Path

from mldb_v2.src.evaluation.evaluate_interface import EvaluationCandidate
from mldb_data.nanodet.lib import nanodet_postprocess_overlay_v2 as overlay
from mldb_data.nanodet.lib import nanodet_postprocess_soft_union_v1 as replay


def evaluate(context):
    if not context.inputs or "prediction_trace" not in context.inputs:
        raise ValueError("soft+union replay requires upstream prediction_trace artifact")
    source = context.inputs["prediction_trace"]
    work = Path(context.work_dir) / "soft-union-replay"
    work.mkdir(parents=True, exist_ok=True)
    trace = work / "prediction_trace.jsonl"
    trace.write_bytes(source.data)

    report = replay.evaluate_trace(trace, work)
    hand = report["regions"]["completed-hand"]
    meld = report["regions"]["melds"]
    dora = report["regions"]["dora-indicators"]

    overlay_path = work / "soft-union-overlay.mp4"
    video = overlay.render_condition_video(
        corpus_root=Path(context.corpus.root),
        trace_path=trace,
        condition=replay.CONDITION,
        destination=overlay_path,
    )

    result_path = work / "result.json"
    enriched = json.loads(result_path.read_text(encoding="utf-8"))
    enriched.update({
        "source_trace_evaluation_result": source.source_evaluation_result,
        "source_trace_artifact": source.artifact,
        "source_trace_sha256": source.ref["sha256"],
        "source_model": context.model.definition["id"] if context.model is not None else None,
        "source_corpus": context.corpus.definition["id"],
        "rendered_video": video,
        "not_recomputed": ["classifier inference", "meld regroup", "product stabilization"],
    })
    result_path.write_text(
        json.dumps(enriched, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    context.telemetry.report_scalar(
        group="postprocess",
        series="hand_exact_rate_proxy",
        value=float(hand["exact_frame_rate"]),
        step=0,
    )
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
            "t4_hatsu_overcount_frames": int(report["t4_hatsu_overcount_frames"]),
            "union_suppression_count": int(report["union_suppression_count"]),
        },
        artifacts={
            "selected_result": result_path,
            "comparison_csv": work / "comparison.csv",
            "exact_frame_changes": work / "exact_frame_changes.jsonl",
            "union_suppressions": work / "union_suppressions.jsonl",
            "overlay_video": overlay_path,
        },
    )
