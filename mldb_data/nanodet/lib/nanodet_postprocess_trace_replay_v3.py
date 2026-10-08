#!/usr/bin/env python3
"""CPU-only postprocessor replay on a verified, already-inferred recognition trace.

Does not run Torch, ONNX or reclassify tiles. Reuses immutable detections and
classifications; adjusts suppression and re-evaluates region ordered semantics.
This is a trace-replay counterfactual, NOT a production stabilization rerun.
"""
from __future__ import annotations

import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

CASES = (
    ("baseline", "coverage", .80),
    ("coverage_0.75", "coverage", .75),
    ("coverage_0.70", "coverage", .70),
    ("nms_iou_0.55", "nms", .55),
    ("nms_iou_0.50", "nms", .50),
    ("gaussian_soft_nms_sigma_0.50", "soft", .50),
)
REGIONS = ("completed-hand", "melds", "dora-indicators")
MIN_SCORE = .35


def area(b):
    return max(0.0, b["width"]) * max(0.0, b["height"])


def intersect(a, b):
    return (
        max(0.0, min(a["x"] + a["width"], b["x"] + b["width"]) - max(a["x"], b["x"]))
        * max(0.0, min(a["y"] + a["height"], b["y"] + b["height"]) - max(a["y"], b["y"]))
    )


def iou(a, b):
    x = intersect(a, b)
    denom = area(a) + area(b) - x
    return x / denom if denom > 0 else 0.0


def coverage(a, b):
    denom = min(area(a), area(b))
    return intersect(a, b) / denom if denom > 0 else 0.0


def suppress(ds, mode, threshold):
    """Apply a stricter replay to *already* product-NMS-filtered detections."""
    if mode == "coverage":
        kept = []
        for d in sorted(ds, key=lambda d: (-d["confidence"], d["detectionIndex"])):
            if all(coverage(d["sourceBox"], a["sourceBox"]) < threshold for a in kept):
                kept.append(d)
        return kept

    pending = [(d, float(d["confidence"])) for d in ds]
    kept = []
    while pending:
        best = max(range(len(pending)), key=lambda i: pending[i][1])
        winner, conf = pending.pop(best)
        if conf < MIN_SCORE:
            break
        kept.append(winner)
        next_pending = []
        for d, score in pending:
            overlap = iou(winner["sourceBox"], d["sourceBox"])
            if mode == "nms":
                if overlap >= threshold:
                    continue
            elif mode == "soft":
                score *= math.exp(-(overlap * overlap) / threshold)
                if score < MIN_SCORE:
                    continue
            else:
                raise ValueError(mode)
            next_pending.append((d, score))
        pending = next_pending
    return kept


def tile_token(d):
    c = d.get("classification", {})
    if c.get("kind") != "tile":
        return None
    v = c["tile"]
    return str(v["kind"]) + ("R" if v.get("red") else "")


def ordered_predictions(row, region, selected):
    """Existing layout order with dropped IDs removed. Does NOT regenerate groups."""
    valid = [d for d in selected if tile_token(d) is not None]
    if region != "melds":
        return [tile_token(d) for d in sorted(valid, key=lambda d: (d["sourceBox"]["x"], d["sourceBox"]["y"]))]
    by_id = {d["id"]: d for d in valid}
    ordered = []
    used = set()
    groups = row["snapshot"]["meldGroups"]
    if isinstance(groups, list) and groups:
        for group in groups:
            for member in group.get("memberObservationIds", []):
                if member in by_id and member not in used:
                    ordered.append(by_id[member])
                    used.add(member)
    remain = [d for d in valid if d["id"] not in used]
    if remain:
        heights = sorted(d["sourceBox"]["height"] for d in remain)
        threshold = max(1e-9, heights[len(heights)//2] * .55)
        pending = sorted(remain, key=lambda d: (d["sourceBox"]["y"] + d["sourceBox"]["height"]/2, d["sourceBox"]["x"], d["id"]))
        rows = []
        cy_means = []
        for d in pending:
            cy = d["sourceBox"]["y"] + d["sourceBox"]["height"]/2
            if not rows or abs(cy - cy_means[-1]) > threshold:
                rows.append([d])
                cy_means.append(cy)
            else:
                rows[-1].append(d)
                cy_means[-1] = sum(q["sourceBox"]["y"] + q["sourceBox"]["height"]/2 for q in rows[-1]) / len(rows[-1])
        for group in rows:
            ordered.extend(sorted(group, key=lambda d: (d["sourceBox"]["x"] + d["sourceBox"]["width"]/2, d["sourceBox"]["y"], d["id"])))
    return [tile_token(d) for d in ordered]


def align(gt, predicted):
    """Canonical evaluator's Levenshtein cost and tie-breaking rules."""
    rows, cols = len(gt) + 1, len(predicted) + 1
    cost = [[0] * cols for _ in range(rows)]
    op = [[""] * cols for _ in range(rows)]
    for i in range(1, rows):
        cost[i][0], op[i][0] = i, "deletion"
    for j in range(1, cols):
        cost[0][j], op[0][j] = j, "insertion"
    rank = {"match": 0, "substitution": 1, "deletion": 2, "insertion": 3}
    for i in range(1, rows):
        for j in range(1, cols):
            kind = "match" if gt[i-1] == predicted[j-1] else "substitution"
            candidates = [
                (cost[i-1][j-1] + (kind != "match"), kind),
                (cost[i-1][j] + 1, "deletion"),
                (cost[i][j-1] + 1, "insertion"),
            ]
            cost[i][j], op[i][j] = min(candidates, key=lambda item: (item[0], rank[item[1]]))
    i, j = len(gt), len(predicted)
    counts = Counter()
    while i or j:
        kind = op[i][j]
        if kind in ("match", "substitution"):
            i -= 1; j -= 1
        elif kind == "deletion":
            i -= 1
        else:
            j -= 1
        counts[kind] += 1
    return counts


def f1_from_counts(c):
    matches, sub, ins, dele = (c[k] for k in ("match", "substitution", "insertion", "deletion"))
    p = matches / (matches + sub + ins) if matches + sub + ins else 1.0
    r = matches / (matches + sub + dele) if matches + sub + dele else 1.0
    return p, r, 2*p*r/(p+r) if p+r else 0.0


def candidate_metrics(ds):
    n = 0
    for i, a in enumerate(ds):
        for b in ds[i+1:]:
            n += iou(a["sourceBox"], b["sourceBox"]) >= .5
    return n


def run(trace_path: Path, output_path: Path) -> None:
    output_path.mkdir(parents=True, exist_ok=True)
    counters = {name: {r: Counter() for r in REGIONS} for name, _, _ in CASES}
    frame_deltas = []
    nframes = 0

    with trace_path.open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            nframes += 1
            by_region = {r: [d for d in row["detections"] if d["region"] == r] for r in REGIONS}
            for region in REGIONS:
                original = ordered_predictions(row, region, by_region[region])
                expected = row["detector_semantic"][region]["predicted"]
                if original != expected:
                    raise RuntimeError(
                        f"baseline ordering mismatch {row['take_id']} {row['eval_index']} {region}: "
                        f"{original!r} != {expected!r}"
                    )
            for name, mode, threshold in CASES:
                for region in REGIONS:
                    ds = by_region[region]
                    kept = ds if name == "baseline" else suppress(ds, mode, threshold)
                    pred = ordered_predictions(row, region, kept)
                    gt = row["detector_semantic"][region]["gt"]
                    origpred = row["detector_semantic"][region]["predicted"]
                    original_exact = gt == origpred
                    new_exact = gt == pred
                    original_tokens = Counter(tile_token(d) for d in ds if tile_token(d) is not None)
                    new_tokens = Counter(tile_token(d) for d in kept if tile_token(d) is not None)
                    origpairs = candidate_metrics(ds)
                    newpairs = candidate_metrics(kept)
                    c = counters[name][region]
                    c.update(align(gt, pred))
                    c["frames"] += 1
                    c["gt_tiles"] += len(gt)
                    c["predicted_tiles"] += len(pred)
                    c["exact_frames"] += int(new_exact)
                    c["original_exact_frames"] += int(original_exact)
                    c["exact_gained"] += int(new_exact and not original_exact)
                    c["exact_lost"] += int(original_exact and not new_exact)
                    c["removed_boxes"] += len(ds) - len(kept)
                    c["removed_classified_tiles"] += sum(original_tokens.values()) - sum(new_tokens.values())
                    c["overlap_pairs"] += newpairs
                    c["overlap_frames"] += int(newpairs > 0)
                    c["baseline_overlap_pairs"] += origpairs
                    c["baseline_overlap_frames"] += int(origpairs > 0)
                    if new_exact != original_exact:
                        frame_deltas.append({
                            "condition": name, "region": region, "take": row["take_id"],
                            "eval_index": row["eval_index"], "video_time_sec": row["video_time_sec"],
                            "gt": gt, "before": origpred, "after": pred,
                            "became_exact": new_exact,
                        })
    rows = []
    for name, method, threshold in CASES:
        for region in REGIONS:
            c = counters[name][region]
            p, r, f = f1_from_counts(c)
            rows.append({
                "condition": name, "method": method, "value": threshold, "region": region,
                "frames": c["frames"], "removed_boxes": c["removed_boxes"],
                "removed_classified_tiles": c["removed_classified_tiles"],
                "remaining_iou50_overlap_pairs": c["overlap_pairs"],
                "remaining_iou50_overlap_frames": c["overlap_frames"],
                "exact_frames": c["exact_frames"], "exact_frame_rate": c["exact_frames"]/c["frames"],
                "exact_gained": c["exact_gained"], "exact_lost": c["exact_lost"],
                "semantic_precision_proxy": p, "semantic_recall_proxy": r,
                "semantic_f1_proxy": f, "correct": c["match"],
                "substitutions": c["substitution"], "insertions": c["insertion"],
                "deletions": c["deletion"],
            })
    assert nframes == 1497, f"unexpected replay frames: {nframes}"
    with (output_path / "comparison.csv").open("w", newline="", encoding="utf-8") as fd:
        writer = csv.DictWriter(fd, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    (output_path / "comparison.json").write_text(json.dumps({
        "schema": "mjtensu.nanodet/cpu-postprocess-trace-replay/v1",
        "source_trace": str(trace_path),
        "frames": nframes,
        "scope": "already-postprocessed detections; immutable classifier results; no model execution",
        "limitation": "Meld groups and product stabilization are frozen from original trace; not full E2E rerun.",
        "conditions": rows,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    with (output_path / "exact_frame_changes.jsonl").open("w", encoding="utf-8") as fd:
        for delta in frame_deltas:
            fd.write(json.dumps(delta, ensure_ascii=False) + "\n")
    print("CPU_TRACE_REPLAY_COMPLETED", nframes, "frames", len(rows)//len(REGIONS), "conditions")
    for row in rows:
        if row["region"] in ("completed-hand", "melds"):
            print(row["condition"],row["region"],
                  "removed",row["removed_boxes"],
                  "remaining_overlap_frames",row["remaining_iou50_overlap_frames"],
                  "exact_rate",round(row["exact_frame_rate"],5),
                  "exact+/-",row["exact_gained"],row["exact_lost"],
                  "semantic_f1_proxy",round(row["semantic_f1_proxy"],5))
    print("RESULT",output_path.resolve())


# Imported only through MLDB Evaluation. Never execute CLI on worker source load.
