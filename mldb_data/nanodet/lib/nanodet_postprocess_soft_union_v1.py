"""INV-041 supplemental Soft-NMS + confidence-gated union-coverage replay."""
from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

from mldb_data.nanodet.lib import nanodet_postprocess_trace_replay_v3 as base

CONDITION = "gaussian_soft_nms_sigma_0.50_union_coverage_0.70"
SOFT_NMS_SIGMA = 0.50
UNION_COVERAGE_THRESHOLD = 0.70
REGIONS = base.REGIONS


def _intersection_rect(left: dict, right: dict) -> tuple[float, float, float, float] | None:
    x0 = max(float(left["x"]), float(right["x"]))
    y0 = max(float(left["y"]), float(right["y"]))
    x1 = min(float(left["x"]) + float(left["width"]), float(right["x"]) + float(right["width"]))
    y1 = min(float(left["y"]) + float(left["height"]), float(right["y"]) + float(right["height"]))
    if x1 <= x0 or y1 <= y0:
        return None
    return (x0, y0, x1, y1)


def _rect_union_area(rectangles: list[tuple[float, float, float, float]]) -> float:
    if not rectangles:
        return 0.0
    xs = sorted({x for rect in rectangles for x in (rect[0], rect[2])})
    total = 0.0
    for left, right in zip(xs, xs[1:]):
        if right <= left:
            continue
        intervals = sorted(
            (y0, y1) for x0, y0, x1, y1 in rectangles
            if x0 < right and x1 > left and y1 > y0
        )
        if not intervals:
            continue
        covered = 0.0
        start, end = intervals[0]
        for y0, y1 in intervals[1:]:
            if y0 <= end:
                end = max(end, y1)
            else:
                covered += end - start
                start, end = y0, y1
        covered += end - start
        total += (right - left) * covered
    return total


def union_coverage(candidate: dict, others: list[dict]) -> float:
    box = candidate["sourceBox"]
    area = base.area(box)
    if area <= 0.0:
        return 0.0
    clipped = [
        rect for other in others
        if (rect := _intersection_rect(box, other["sourceBox"])) is not None
    ]
    return _rect_union_area(clipped) / area


def suppress_union_confidence(
    detections: list[dict],
    threshold: float = UNION_COVERAGE_THRESHOLD,
) -> tuple[list[dict], list[dict]]:
    active = list(detections)
    suppressed_ids: set[str] = set()
    decisions: list[dict] = []
    ordered = sorted(
        active,
        key=lambda d: (-float(d["confidence"]), int(d["detectionIndex"]), str(d["id"])),
    )
    for candidate in ordered:
        cid = str(candidate["id"])
        if cid in suppressed_ids:
            continue
        others = [
            other for other in active
            if str(other["id"]) not in suppressed_ids
            and other is not candidate
            and base.intersect(candidate["sourceBox"], other["sourceBox"]) > 0.0
        ]
        if not others:
            continue
        confidence = float(candidate["confidence"])
        if not all(confidence < float(other["confidence"]) for other in others):
            continue
        coverage = union_coverage(candidate, others)
        if coverage < threshold:
            continue
        suppressed_ids.add(cid)
        decisions.append({
            "id": cid,
            "confidence": confidence,
            "union_coverage": coverage,
            "covering_ids": [str(other["id"]) for other in others],
            "covering_confidences": [float(other["confidence"]) for other in others],
        })
    kept = [d for d in detections if str(d["id"]) not in suppressed_ids]
    return kept, decisions


def apply(detections: list[dict]) -> tuple[list[dict], dict[str, object]]:
    after_soft = base.suppress(detections, "soft", SOFT_NMS_SIGMA)
    kept, union_decisions = suppress_union_confidence(after_soft)
    return kept, {
        "input_boxes": len(detections),
        "after_soft_nms_boxes": len(after_soft),
        "after_union_boxes": len(kept),
        "soft_nms_removed": len(detections) - len(after_soft),
        "union_removed": len(after_soft) - len(kept),
        "union_decisions": union_decisions,
    }


def _region_row(row: dict, region: str, selected: list[dict]) -> dict[str, object]:
    gt = row["detector_semantic"][region]["gt"]
    before = row["detector_semantic"][region]["predicted"]
    after = base.ordered_predictions(row, region, selected)
    counts = base.align(gt, after)
    precision, recall, f1 = base.f1_from_counts(counts)
    return {
        "gt": gt,
        "before": before,
        "after": after,
        "before_exact": before == gt,
        "after_exact": after == gt,
        "alignment": dict(counts),
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def evaluate_trace(trace_path: Path, output_path: Path) -> dict[str, object]:
    output_path.mkdir(parents=True, exist_ok=True)
    counters = {region: Counter() for region in REGIONS}
    frame_changes: list[dict] = []
    union_decisions: list[dict] = []
    frames = 0
    t4_hatsu_overcount_frames = 0
    t4_hatsu_overcount_baseline_frames = 0

    with trace_path.open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            frames += 1
            by_region = {
                region: [d for d in row["detections"] if d["region"] == region]
                for region in REGIONS
            }
            for region in REGIONS:
                original = base.ordered_predictions(row, region, by_region[region])
                expected = row["detector_semantic"][region]["predicted"]
                if original != expected:
                    raise RuntimeError(
                        f"baseline ordering mismatch {row['take_id']} "
                        f"{row['eval_index']} {region}: {original!r} != {expected!r}"
                    )

                selected, stats = apply(by_region[region])
                detail = _region_row(row, region, selected)
                c = counters[region]
                c.update(detail["alignment"])
                c["frames"] += 1
                c["exact_frames"] += int(detail["after_exact"])
                c["baseline_exact_frames"] += int(detail["before_exact"])
                c["exact_gained"] += int(detail["after_exact"] and not detail["before_exact"])
                c["exact_lost"] += int(detail["before_exact"] and not detail["after_exact"])
                c["removed_boxes"] += len(by_region[region]) - len(selected)
                c["soft_nms_removed"] += int(stats["soft_nms_removed"])
                c["union_removed"] += int(stats["union_removed"])
                c["remaining_overlap_pairs"] += base.candidate_metrics(selected)
                c["remaining_overlap_frames"] += int(base.candidate_metrics(selected) > 0)
                for decision in stats["union_decisions"]:
                    union_decisions.append({
                        "take_id": row["take_id"],
                        "eval_index": row["eval_index"],
                        "video_time_sec": row["video_time_sec"],
                        "region": region,
                        **decision,
                    })
                if detail["after_exact"] != detail["before_exact"]:
                    frame_changes.append({
                        "take_id": row["take_id"],
                        "eval_index": row["eval_index"],
                        "video_time_sec": row["video_time_sec"],
                        "region": region,
                        "before": detail["before"],
                        "after": detail["after"],
                        "gt": detail["gt"],
                        "became_exact": detail["after_exact"],
                    })

                if row["take_id"] == "t4" and region == "completed-hand":
                    before_tokens = [base.tile_token(d) for d in by_region[region]]
                    after_tokens = [base.tile_token(d) for d in selected]
                    t4_hatsu_overcount_baseline_frames += int(before_tokens.count("6z") >= 3)
                    t4_hatsu_overcount_frames += int(after_tokens.count("6z") >= 3)

    if frames != 1497:
        raise ValueError(f"unexpected trace length: {frames}")

    rows: list[dict[str, object]] = []
    regions: dict[str, dict[str, object]] = {}
    for region in REGIONS:
        c = counters[region]
        precision, recall, f1 = base.f1_from_counts(c)
        summary = {
            "frames": int(c["frames"]),
            "exact_frames": int(c["exact_frames"]),
            "exact_frame_rate": c["exact_frames"] / c["frames"],
            "baseline_exact_frames": int(c["baseline_exact_frames"]),
            "exact_gained": int(c["exact_gained"]),
            "exact_lost": int(c["exact_lost"]),
            "removed_boxes": int(c["removed_boxes"]),
            "soft_nms_removed": int(c["soft_nms_removed"]),
            "union_removed": int(c["union_removed"]),
            "remaining_iou50_overlap_pairs": int(c["remaining_overlap_pairs"]),
            "remaining_iou50_overlap_frames": int(c["remaining_overlap_frames"]),
            "semantic_precision_proxy": precision,
            "semantic_recall_proxy": recall,
            "semantic_f1_proxy": f1,
            "correct": int(c["match"]),
            "substitutions": int(c["substitution"]),
            "insertions": int(c["insertion"]),
            "deletions": int(c["deletion"]),
        }
        regions[region] = summary
        rows.append({"condition": CONDITION, "region": region, **summary})

    report = {
        "schema": "mjtensu.nanodet/soft-union-trace-replay/v1",
        "condition": CONDITION,
        "frames": frames,
        "soft_nms_sigma": SOFT_NMS_SIGMA,
        "union_coverage_threshold": UNION_COVERAGE_THRESHOLD,
        "regions": regions,
        "t4_hatsu_overcount_baseline_frames": t4_hatsu_overcount_baseline_frames,
        "t4_hatsu_overcount_frames": t4_hatsu_overcount_frames,
        "union_suppression_count": len(union_decisions),
        "scope": "saved detector/classifier trace; no model or classifier inference; no regroup/stabilization",
    }
    (output_path / "result.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    with (output_path / "comparison.csv").open("w", newline="", encoding="utf-8") as fd:
        writer = csv.DictWriter(fd, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (output_path / "exact_frame_changes.jsonl").open("w", encoding="utf-8") as fd:
        for item in frame_changes:
            fd.write(json.dumps(item, ensure_ascii=False) + "\n")
    with (output_path / "union_suppressions.jsonl").open("w", encoding="utf-8") as fd:
        for item in union_decisions:
            fd.write(json.dumps(item, ensure_ascii=False) + "\n")
    return report
