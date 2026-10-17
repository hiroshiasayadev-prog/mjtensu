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
            "title": "Blender v6 balanced 34-class confusion counts",
            "xaxis": {"title": "Predicted class"},
            "yaxis": {"title": "True class", "autorange": "reversed"},
            "height": 900,
        },
    }


def _contact_sheet(path: Path, selected: dict[str, list[tuple[float, int, bytes, str, str]]]) -> None:
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
        for j, (_conf, _ord, raw, pred, crop_id) in enumerate(entries[:2]):
            img = Image.frombytes("L", (64, 64), raw).convert("RGB")
            px = x + 4 + j * 120
            py = y + 20
            sheet.paste(img, (px, py))
            draw.text((px, py + 66), f"->{pred}", fill="black")
            draw.text((px, py + 80), crop_id[-16:], fill="black")
    sheet.save(path)


def evaluate(context):
    batch_size = int(context.parameters["batch_size"])
    recall_gate = float(context.parameters["recall_gate"])
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if not (0.0 <= recall_gate <= 1.0):
        raise ValueError("recall_gate must be within [0,1]")

    root = context.corpus.root
    index = json.loads((root / "index.json").read_text(encoding="utf-8"))
    if tuple(index["labels"]) != EXPECTED_LABELS:
        raise ValueError("Corpus output labels do not match tile task")
    if tuple(index["true_labels"]) != VALID_LABELS:
        raise ValueError("Corpus true labels do not match 34 valid base tile classes")
    if int(index["counts"]["per_class"]) != 100 or int(index["counts"]["classes"]) != len(VALID_LABELS):
        raise ValueError("Corpus is not the authored 34x100 balanced population")
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
    invalid_predictions = 0
    total = 0
    correct = 0
    serial = 0
    selected: dict[str, list[tuple[float, int, bytes, str, str]]] = defaultdict(list)

    work_dir = context.work_dir
    work_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = work_dir / "blender-balanced-predictions.jsonl.gz"
    errors_path = work_dir / "blender-balanced-errors.jsonl.gz"
    shard_paths = sorted(root.glob("shard-*.npz"))
    if not shard_paths:
        raise ValueError("balanced Blender Corpus has no shards")

    with gzip.open(predictions_path, "wt", encoding="utf-8") as pred_out, gzip.open(
        errors_path, "wt", encoding="utf-8"
    ) as err_out, torch.inference_mode():
        for shard_index, shard_path in enumerate(shard_paths):
            meta_path = shard_path.with_suffix(".jsonl.gz")
            with np.load(shard_path, allow_pickle=False) as data:
                images_np = np.asarray(data["images"], dtype=np.uint8)
                targets_np = np.asarray(data["labels"], dtype=np.int64)
            with gzip.open(meta_path, "rt", encoding="utf-8") as handle:
                metadata = [json.loads(line) for line in handle]
            if len(metadata) != len(images_np) or len(targets_np) != len(images_np):
                raise ValueError(f"shard alignment mismatch: {shard_path.name}")

            for start in range(0, len(images_np), batch_size):
                stop = min(start + batch_size, len(images_np))
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
                    crop_id = str(row["crop_id"])

                    total += 1
                    correct += int(is_correct)
                    class_total[target] += 1
                    class_correct[target] += int(is_correct)
                    confusion[target, pred] += 1
                    if pred == len(EXPECTED_LABELS) - 1:
                        invalid_predictions += 1

                    payload = {
                        "crop_id": crop_id,
                        "source_annotation_id": row["source_annotation_id"],
                        "source_image_path": row["source_image_path"],
                        "region": row.get("region"),
                        "tile_identity": row.get("tile_identity"),
                        "true_label": VALID_LABELS[target],
                        "predicted_label": EXPECTED_LABELS[pred],
                        "confidence": confidence,
                        "margin": margin,
                        "second_label": EXPECTED_LABELS[second],
                        "second_confidence": second_conf,
                        "correct": bool(is_correct),
                    }
                    pred_out.write(json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n")
                    if not is_correct:
                        err_out.write(json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n")
                        serial += 1
                        entry = (confidence, serial, bytes(batch_np[local].reshape(-1)), EXPECTED_LABELS[pred], crop_id)
                        heap = selected[VALID_LABELS[target]]
                        if len(heap) < 2:
                            heapq.heappush(heap, entry)
                        elif entry[0] > heap[0][0]:
                            heapq.heapreplace(heap, entry)

            context.telemetry.report_scalar(
                group="classifier/blender_balanced_class_recall",
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
        if count != 100:
            raise ValueError(f"class {label} has {count} samples, expected 100")
        recall = float(class_correct[idx] / count)
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
            "passes_gate": recall >= recall_gate,
            "error_count": count - int(class_correct[idx]),
            "dominant_wrong_label": EXPECTED_LABELS[dominant] if int(wrong.sum()) else "",
            "dominant_wrong_count": int(wrong[dominant]) if int(wrong.sum()) else 0,
        })

    metrics: dict[str, int | float] = {
        "accuracy": correct / total,
        "macro_recall": float(sum(recalls) / len(recalls)),
        "worst_class_recall": float(min(recalls)),
        "class_recall_below_gate_count": int(sum(r < recall_gate for r in recalls)),
        "invalid_prediction_rate": invalid_predictions / total,
    }

    class_path = work_dir / "blender-balanced-class-recall.csv"
    confusion_path = work_dir / "blender-balanced-confusion.plotly.json"
    contact_path = work_dir / "blender-balanced-high-confidence-errors.png"
    report_path = work_dir / "blender-balanced-report.json"
    _write_csv(class_path, tuple(class_rows[0]), class_rows)
    _plotly(confusion_path, _confusion_plot(confusion))
    _contact_sheet(contact_path, selected)

    report = {
        "schema": "mjtensu.recognition/blender-balanced-class-recall-report/v1",
        "population": index["counts"],
        "normalization": index["normalization"],
        "preprocess": index["preprocess"],
        "selection": index["selection"],
        "recall_gate": recall_gate,
        "metrics": metrics,
        "notes": {
            "purpose": "balanced synthetic feasibility gate for known-tile identity classification",
            "domain": "Blender v6 synthetic_val; this does not replace real-domain robustness evaluation",
            "invalid": "invalid has no GT rows and remains a prediction-only error destination",
            "red_five": "red5m/red5p/red5s are folded into base 5m/5p/5s labels",
        },
        "telemetry": {
            "group": "classifier/blender_balanced_class_recall",
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
            "confusion_matrix": confusion_path,
            "high_confidence_error_contact_sheet": contact_path,
            "error_inventory": errors_path,
            "per_sample_predictions": predictions_path,
            "report": report_path,
        },
    )
