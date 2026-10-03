from __future__ import annotations

import csv
import gzip
import heapq
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw

from mldb_v2.src.evaluation.evaluate_interface import EvaluationCandidate

EXPECTED_LABELS = (
    "1m","2m","3m","4m","5m","6m","7m","8m","9m",
    "1p","2p","3p","4p","5p","6p","7p","8p","9p",
    "1s","2s","3s","4s","5s","6s","7s","8s","9s",
    "east","south","west","north","white","green","red","invalid",
)
VALID_LABELS = EXPECTED_LABELS[:-1]
ORIENTATION_BUCKETS = ("portrait", "square", "landscape")


def _orientation(width: int, height: int) -> str:
    if width >= height * 1.15:
        return "landscape"
    if height >= width * 1.15:
        return "portrait"
    return "square"


def _write_csv(path: Path, fieldnames: tuple[str, ...], rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _plotly(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n", encoding="utf-8")


def _confusion_plot(confusion: np.ndarray) -> dict[str, Any]:
    return {
        "data": [{
            "type": "heatmap",
            "x": list(EXPECTED_LABELS),
            "y": list(VALID_LABELS),
            "z": confusion.tolist(),
            "colorscale": "Blues",
            "hovertemplate": "GT=%{y}<br>pred=%{x}<br>count=%{z}<extra></extra>",
        }],
        "layout": {
            "title": "All-real-crop confusion counts",
            "xaxis": {"title": "Predicted class"},
            "yaxis": {"title": "True class", "autorange": "reversed"},
            "height": 900,
        },
    }


def _orientation_heatmap(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by = {(r["label"], r["orientation"]): r["recall"] for r in rows}
    z = [[by.get((label, orient)) for orient in ORIENTATION_BUCKETS] for label in VALID_LABELS]
    return {
        "data": [{
            "type": "heatmap",
            "x": list(ORIENTATION_BUCKETS),
            "y": list(VALID_LABELS),
            "z": z,
            "zmin": 0.0,
            "zmax": 1.0,
            "colorscale": "RdYlGn",
            "hovertemplate": "class=%{y}<br>orientation=%{x}<br>recall=%{z:.4f}<extra></extra>",
        }],
        "layout": {
            "title": "All-real-crop class × orientation recall",
            "xaxis": {"title": "Crop aspect orientation"},
            "yaxis": {"title": "True class", "autorange": "reversed"},
            "height": 900,
        },
    }


def _contact_sheet(path: Path, selected: dict[str, list[tuple[float, int, bytes, str, str, str]]]) -> None:
    cell_w, cell_h = 250, 110
    cols = 4
    rows = math.ceil(len(VALID_LABELS) / cols)
    sheet = Image.new("RGB", (cols * cell_w, rows * cell_h), "white")
    draw = ImageDraw.Draw(sheet)
    for class_index, label in enumerate(VALID_LABELS):
        x = (class_index % cols) * cell_w
        y = (class_index // cols) * cell_h
        draw.text((x + 4, y + 3), label, fill="black")
        entries = sorted(selected.get(label, []), reverse=True)
        for j, (_conf, _ord, raw, pred, crop_id, orient) in enumerate(entries[:2]):
            img = Image.frombytes("L", (64, 64), raw).convert("RGB")
            px = x + 4 + j * 120
            py = y + 20
            sheet.paste(img, (px, py))
            draw.text((px, py + 66), f"->{pred} {orient}", fill="black")
            draw.text((px, py + 80), crop_id[-16:], fill="black")
    sheet.save(path)


def evaluate(context):
    batch_size = int(context.parameters["batch_size"])
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    slice_min_samples = int(context.parameters["slice_min_samples"])
    if slice_min_samples < 1:
        raise ValueError("slice_min_samples must be positive")

    root = context.corpus.root
    index = json.loads((root / "index.json").read_text(encoding="utf-8"))
    if tuple(index["labels"]) != EXPECTED_LABELS:
        raise ValueError("all-real-crop Corpus labels do not match tile task")
    mean = float(index["normalization"]["mean"])
    std = float(index["normalization"]["std"])
    if not math.isfinite(mean) or not math.isfinite(std) or std <= 0:
        raise ValueError("invalid normalization")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = context.model.module.to(device)
    model.eval()

    class_total = np.zeros(len(VALID_LABELS), dtype=np.int64)
    class_correct = np.zeros(len(VALID_LABELS), dtype=np.int64)
    confusion = np.zeros((len(VALID_LABELS), len(EXPECTED_LABELS)), dtype=np.int64)
    source_total: defaultdict[tuple[str, str], int] = defaultdict(int)
    source_correct: defaultdict[tuple[str, str], int] = defaultdict(int)
    orient_total: defaultdict[tuple[int, str], int] = defaultdict(int)
    orient_correct: defaultdict[tuple[int, str], int] = defaultdict(int)
    invalid_predictions = 0
    total = 0
    correct = 0
    corrected_total = 0
    corrected_correct = 0
    serial = 0
    selected: dict[str, list[tuple[float, int, bytes, str, str, str]]] = defaultdict(list)

    work_dir = context.work_dir
    work_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = work_dir / "all-real-crop-predictions.jsonl.gz"
    errors_path = work_dir / "all-real-crop-errors.jsonl.gz"

    shard_paths = sorted(root.glob("shard-*.npz"))
    if not shard_paths:
        raise ValueError("all-real-crop Corpus has no shards")

    with gzip.open(predictions_path, "wt", encoding="utf-8") as pred_out, gzip.open(
        errors_path, "wt", encoding="utf-8"
    ) as err_out, torch.inference_mode():
        for shard_index, shard_path in enumerate(shard_paths):
            meta_path = shard_path.with_suffix(".jsonl.gz")
            with np.load(shard_path, allow_pickle=False) as data:
                images_np = np.asarray(data["images"], dtype=np.uint8)
                targets_np = np.asarray(data["labels"], dtype=np.int64)
                widths = np.asarray(data["widths"], dtype=np.int64)
                heights = np.asarray(data["heights"], dtype=np.int64)
                corrected_flags = np.asarray(data["corrected_flags"], dtype=np.uint8)
                annotation_angles = np.asarray(data["annotation_angles_deg"], dtype=np.float32)
                expected_rotations = np.asarray(data["expected_rotations_deg"], dtype=np.int16)
            with gzip.open(meta_path, "rt", encoding="utf-8") as handle:
                metadata = [json.loads(line) for line in handle]
            n = int(images_np.shape[0])
            if not (
                len(metadata) == n
                == len(targets_np)
                == len(widths)
                == len(heights)
                == len(corrected_flags)
                == len(annotation_angles)
                == len(expected_rotations)
            ):
                raise ValueError(f"shard alignment mismatch: {shard_path.name}")

            for start in range(0, n, batch_size):
                stop = min(start + batch_size, n)
                batch_np = images_np[start:stop]
                batch = torch.from_numpy(batch_np).float().unsqueeze(1).mul_(1.0 / 255.0)
                batch = batch.to(device, non_blocking=device.type == "cuda")
                logits = model(batch.sub(mean).div(std))
                if not isinstance(logits, torch.Tensor) or logits.ndim != 2 or logits.shape[1] != len(EXPECTED_LABELS):
                    raise ValueError(f"classifier logits shape invalid: {tuple(logits.shape)}")
                probs = torch.softmax(logits.float(), dim=1)
                top_probs, top_idx = probs.topk(2, dim=1)
                top_probs = top_probs.cpu().numpy()
                top_idx = top_idx.cpu().numpy()

                for local in range(stop - start):
                    i = start + local
                    target = int(targets_np[i])
                    pred = int(top_idx[local, 0])
                    confidence = float(top_probs[local, 0])
                    second = int(top_idx[local, 1])
                    second_conf = float(top_probs[local, 1])
                    margin = confidence - second_conf
                    is_correct = pred == target
                    row = metadata[i]
                    source = str(row["source"])
                    partition = str(row["source_partition"])
                    orientation = _orientation(int(widths[i]), int(heights[i]))
                    crop_id = str(row["crop_id"])

                    total += 1
                    correct += int(is_correct)
                    class_total[target] += 1
                    class_correct[target] += int(is_correct)
                    confusion[target, pred] += 1
                    source_total[(source, partition)] += 1
                    source_correct[(source, partition)] += int(is_correct)
                    orient_total[(target, orientation)] += 1
                    orient_correct[(target, orientation)] += int(is_correct)
                    if pred == len(EXPECTED_LABELS) - 1:
                        invalid_predictions += 1
                    if int(corrected_flags[i]):
                        corrected_total += 1
                        corrected_correct += int(is_correct)

                    payload = {
                        "crop_id": crop_id,
                        "source": source,
                        "source_partition": partition,
                        "source_annotation_id": row["source_annotation_id"],
                        "true_label": VALID_LABELS[target],
                        "predicted_label": EXPECTED_LABELS[pred],
                        "confidence": confidence,
                        "margin": margin,
                        "second_label": EXPECTED_LABELS[second],
                        "second_confidence": second_conf,
                        "correct": bool(is_correct),
                        "corrected_annotation": bool(corrected_flags[i]),
                        "crop_width": int(widths[i]),
                        "crop_height": int(heights[i]),
                        "orientation": orientation,
                        "annotation_angle_deg": float(annotation_angles[i]),
                        "expected_rotation_deg": int(expected_rotations[i]),
                        "source_image_path": row.get("source_image_path"),
                    }
                    pred_out.write(json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n")
                    if not is_correct:
                        err_out.write(json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n")
                        serial += 1
                        entry = (
                            confidence,
                            serial,
                            bytes(batch_np[local].reshape(-1)),
                            EXPECTED_LABELS[pred],
                            crop_id,
                            orientation,
                        )
                        heap = selected[VALID_LABELS[target]]
                        if len(heap) < 2:
                            heapq.heappush(heap, entry)
                        elif entry[0] > heap[0][0]:
                            heapq.heapreplace(heap, entry)

            context.telemetry.report_scalar(
                group="classifier/all_real_crop_recall",
                series="running_accuracy",
                value=correct / max(total, 1),
                step=shard_index,
            )

    if total != int(index["counts"]["included"]):
        raise ValueError(f"evaluated {total} samples but Corpus index declares {index['counts']['included']}")

    class_rows: list[dict[str, Any]] = []
    recalls: list[float] = []
    for idx, label in enumerate(VALID_LABELS):
        count = int(class_total[idx])
        recall = float(class_correct[idx] / count) if count else 0.0
        recalls.append(recall)
        wrong = confusion[idx].copy()
        wrong[idx] = 0
        dominant = int(wrong.argmax()) if int(wrong.sum()) else idx
        class_rows.append({
            "class_index": idx,
            "label": label,
            "sample_count": count,
            "correct_count": int(class_correct[idx]),
            "recall": recall,
            "error_count": count - int(class_correct[idx]),
            "dominant_wrong_label": EXPECTED_LABELS[dominant] if int(wrong.sum()) else "",
            "dominant_wrong_count": int(wrong[dominant]) if int(wrong.sum()) else 0,
        })

    source_rows: list[dict[str, Any]] = []
    for key in sorted(source_total):
        n = source_total[key]
        source_rows.append({
            "source": key[0],
            "source_partition": key[1],
            "sample_count": n,
            "correct_count": source_correct[key],
            "accuracy": source_correct[key] / n,
        })

    orientation_rows: list[dict[str, Any]] = []
    eligible_orientation_recalls: list[float] = []
    for idx, label in enumerate(VALID_LABELS):
        for orient in ORIENTATION_BUCKETS:
            n = orient_total[(idx, orient)]
            c = orient_correct[(idx, orient)]
            recall = (c / n) if n else None
            orientation_rows.append({
                "class_index": idx,
                "label": label,
                "orientation": orient,
                "sample_count": n,
                "correct_count": c,
                "recall": recall,
            })
            if n >= slice_min_samples and recall is not None:
                eligible_orientation_recalls.append(float(recall))

    jp_n = sum(n for (src, _), n in source_total.items() if src == "jp")
    jp_c = sum(source_correct[k] for k in source_total if k[0] == "jp")
    manual_n = sum(n for (src, _), n in source_total.items() if src == "manual")
    manual_c = sum(source_correct[k] for k in source_total if k[0] == "manual")

    metrics: dict[str, int | float] = {
        "accuracy": correct / total,
        "macro_recall": float(sum(recalls) / len(recalls)),
        "worst_class_recall": float(min(recalls)),
        "worst_class_orientation_recall": float(min(eligible_orientation_recalls)),
        "class_recall_below_0p90_count": int(sum(r < 0.90 for r in recalls)),
        "class_recall_below_0p95_count": int(sum(r < 0.95 for r in recalls)),
        "invalid_prediction_rate": invalid_predictions / total,
        "jp_accuracy": jp_c / jp_n,
        "manual_accuracy": manual_c / manual_n,
    }

    class_path = work_dir / "all-real-crop-class-recall.csv"
    source_path = work_dir / "all-real-crop-source-recall.csv"
    orient_path = work_dir / "all-real-crop-class-orientation-recall.csv"
    confusion_path = work_dir / "all-real-crop-confusion.plotly.json"
    heatmap_path = work_dir / "all-real-crop-orientation-recall.plotly.json"
    contact_path = work_dir / "all-real-crop-high-confidence-errors.png"
    report_path = work_dir / "all-real-crop-report.json"

    _write_csv(class_path, tuple(class_rows[0]), class_rows)
    _write_csv(source_path, tuple(source_rows[0]), source_rows)
    _write_csv(orient_path, tuple(orientation_rows[0]), orientation_rows)
    _plotly(confusion_path, _confusion_plot(confusion))
    _plotly(heatmap_path, _orientation_heatmap(orientation_rows))
    _contact_sheet(contact_path, selected)

    report = {
        "schema": "mjtensu.recognition/tile-all-real-crop-recall-report/v1",
        "population": index["counts"],
        "normalization": index["normalization"],
        "preprocess": index["preprocess"],
        "metrics": metrics,
        "slice_min_samples": slice_min_samples,
        "orientation_rule": {
            "landscape": "crop_width >= 1.15 * crop_height",
            "portrait": "crop_height >= 1.15 * crop_width",
            "square": "otherwise",
        },
        "corrected_annotation_subset": {
            "sample_count": corrected_total,
            "correct_count": corrected_correct,
            "accuracy": corrected_correct / corrected_total if corrected_total else None,
        },
        "notes": {
            "purpose": "coverage/failure inventory, not an independent holdout generalization estimate",
            "train_overlap": "JP train and manual crops may overlap training data by design",
            "invalid": "all corpus rows are valid tile crops; prediction of invalid is counted as an error",
            "red_five": "red5m/red5p/red5s are folded into base 5m/5p/5s labels",
        },
        "telemetry": {
            "group": "classifier/all_real_crop_recall",
            "series": "running_accuracy",
            "cadence": "after each completed shard",
            "step": "zero-based shard index",
        },
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    return EvaluationCandidate(
        metrics=metrics,
        artifacts={
            "class_recall_table": class_path,
            "source_recall_table": source_path,
            "class_orientation_recall_table": orient_path,
            "confusion_matrix": confusion_path,
            "class_orientation_recall_heatmap": heatmap_path,
            "high_confidence_error_contact_sheet": contact_path,
            "error_inventory": errors_path,
            "per_sample_predictions": predictions_path,
            "report": report_path,
        },
    )
