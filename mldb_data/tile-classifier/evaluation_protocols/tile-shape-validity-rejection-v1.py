from __future__ import annotations

import csv
import json
import math
import sqlite3
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from PIL import Image, ImageDraw

from mldb_v2.src.evaluation.evaluate_interface import EvaluationCandidate


EXPECTED_LABELS = (
    "1m", "2m", "3m", "4m", "5m", "6m", "7m", "8m", "9m",
    "1p", "2p", "3p", "4p", "5p", "6p", "7p", "8p", "9p",
    "1s", "2s", "3s", "4s", "5s", "6s", "7s", "8s", "9s",
    "east", "south", "west", "north", "white", "green", "red", "invalid",
)
VALID_LABELS = EXPECTED_LABELS[:-1]
INVALID_LABEL = "invalid"
INVALID_INDEX = len(EXPECTED_LABELS) - 1
IMAGE_SIZE = 64
SPLIT = "manual_val"
MAX_ERRORS_PER_KIND = 12


def _load_metadata_and_normalization(database: Path) -> tuple[list[str], float, float]:
    with sqlite3.connect(database) as connection:
        metadata = dict(connection.execute("SELECT key, value FROM experiment_metadata"))
        labels = [str(value) for value in json.loads(metadata["base_labels"])]
        pixel_sum = 0.0
        pixel_sq_sum = 0.0
        pixel_count = 0
        for (payload,) in connection.execute(
            "SELECT image_gray_u8 FROM sample WHERE split='train'"
        ):
            values = np.frombuffer(payload, dtype=np.uint8).astype(np.float64) / 255.0
            pixel_sum += float(values.sum())
            pixel_sq_sum += float(np.square(values).sum())
            pixel_count += int(values.size)
    if pixel_count == 0:
        raise ValueError("Corpus train split is empty")
    mean = pixel_sum / pixel_count
    variance = max(pixel_sq_sum / pixel_count - mean * mean, 0.0)
    return labels, mean, max(math.sqrt(variance), 1.0 / 255.0)


def _load_rows(database: Path, labels: Sequence[str]) -> list[dict[str, Any]]:
    with sqlite3.connect(database) as connection:
        rows = connection.execute(
            """
            SELECT sample_id, base_label, class_index, image_gray_u8,
                   original_width, original_height, source, capture_id,
                   layout_id, region, source_image_path,
                   detector_review_decision, invalid_reason
            FROM sample
            WHERE split=?
            ORDER BY class_index, sample_id
            """,
            (SPLIT,),
        ).fetchall()
    counts = {label: 0 for label in labels}
    result: list[dict[str, Any]] = []
    for row in rows:
        class_index = int(row[2])
        label = str(row[1])
        if class_index < 0 or class_index >= len(labels):
            raise ValueError(f"manual_val class_index out of range: {class_index}")
        if labels[class_index] != label:
            raise ValueError(
                f"manual_val label/index mismatch: label={label!r} index={class_index}"
            )
        counts[label] += 1
        result.append(
            {
                "sample_id": str(row[0]),
                "base_label": label,
                "class_index": class_index,
                "image": np.frombuffer(row[3], dtype=np.uint8)
                .copy()
                .reshape(IMAGE_SIZE, IMAGE_SIZE),
                "original_width": int(row[4]),
                "original_height": int(row[5]),
                "source": str(row[6]),
                "capture_id": None if row[7] is None else str(row[7]),
                "layout_id": None if row[8] is None else str(row[8]),
                "region": None if row[9] is None else str(row[9]),
                "source_image_path": str(row[10]),
                "detector_review_decision": (
                    None if row[11] is None else str(row[11])
                ),
                "invalid_reason": None if row[12] is None else str(row[12]),
            }
        )
    missing = [label for label, count in counts.items() if count == 0]
    if missing:
        raise ValueError(f"manual_val must cover every classifier class; missing={missing}")
    return result


def _infer(
    model,
    images_01: torch.Tensor,
    *,
    mean: float,
    std: float,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    outputs: list[torch.Tensor] = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, images_01.shape[0], batch_size):
            batch = images_01[start : start + batch_size].to(
                device,
                non_blocking=device.type == "cuda",
            )
            logits = model(batch.sub(mean).div(std))
            if not isinstance(logits, torch.Tensor):
                raise TypeError("Classifier must return a logits tensor")
            outputs.append(logits.detach().float().cpu())
    return torch.cat(outputs, dim=0)


def _roc_auc(invalid_scores: torch.Tensor, valid_scores: torch.Tensor) -> float:
    if invalid_scores.numel() == 0 or valid_scores.numel() == 0:
        raise ValueError("ROC AUC requires both valid and invalid samples")
    comparisons = invalid_scores[:, None] - valid_scores[None, :]
    wins = (comparisons > 0).to(torch.float64).sum()
    ties = (comparisons == 0).to(torch.float64).sum()
    denominator = invalid_scores.numel() * valid_scores.numel()
    return float((wins + 0.5 * ties) / denominator)


def _summarize_rejection(
    logits: torch.Tensor,
    rows: Sequence[dict[str, Any]],
    labels: Sequence[str],
) -> dict[str, Any]:
    if tuple(labels) != EXPECTED_LABELS:
        raise ValueError("Classifier labels do not match tile-shape-classification-35-v1")
    expected_shape = (len(rows), len(labels))
    if logits.ndim != 2 or tuple(logits.shape) != expected_shape:
        raise ValueError(
            f"Classifier logits must have shape {expected_shape}; got {tuple(logits.shape)}"
        )

    targets = torch.tensor([int(row["class_index"]) for row in rows], dtype=torch.long)
    valid_mask = targets != INVALID_INDEX
    invalid_mask = targets == INVALID_INDEX
    if not bool(valid_mask.any()) or not bool(invalid_mask.any()):
        raise ValueError("Validity rejection evaluation requires both valid and invalid samples")

    valid_logits = logits[:, :INVALID_INDEX]
    best_valid_logits, best_valid_indices = valid_logits.max(dim=1)
    invalid_logits = logits[:, INVALID_INDEX]
    rejection_scores = invalid_logits - best_valid_logits
    reject = rejection_scores >= 0.0
    accept = ~reject

    valid_count = int(valid_mask.sum())
    invalid_count = int(invalid_mask.sum())
    valid_accept_count = int((valid_mask & accept).sum())
    valid_false_reject_count = int((valid_mask & reject).sum())
    invalid_reject_count = int((invalid_mask & reject).sum())
    invalid_false_accept_count = int((invalid_mask & accept).sum())

    valid_accept_recall = valid_accept_count / valid_count
    invalid_reject_recall = invalid_reject_count / invalid_count
    valid_false_reject_rate = valid_false_reject_count / valid_count
    invalid_false_accept_rate = invalid_false_accept_count / invalid_count
    balanced_accuracy = 0.5 * (valid_accept_recall + invalid_reject_recall)
    roc_auc = _roc_auc(rejection_scores[invalid_mask], rejection_scores[valid_mask])

    per_valid_class: list[dict[str, Any]] = []
    for class_index, label in enumerate(VALID_LABELS):
        mask = targets == class_index
        sample_count = int(mask.sum())
        false_reject_count = int((mask & reject).sum())
        per_valid_class.append(
            {
                "class_index": class_index,
                "label": label,
                "sample_count": sample_count,
                "false_reject_count": false_reject_count,
                "false_reject_rate": (
                    false_reject_count / sample_count if sample_count else None
                ),
            }
        )

    details: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        true_invalid = bool(invalid_mask[index])
        rejected = bool(reject[index])
        if true_invalid and rejected:
            outcome = "true_reject"
        elif true_invalid:
            outcome = "false_accept"
        elif rejected:
            outcome = "false_reject"
        else:
            outcome = "true_accept"
        argmax_index = int(logits[index].argmax())
        details.append(
            {
                "row_index": index,
                "sample_id": row["sample_id"],
                "true_label": row["base_label"],
                "true_group": "invalid" if true_invalid else "valid",
                "decision": "reject" if rejected else "accept",
                "outcome": outcome,
                "argmax_label": labels[argmax_index],
                "best_valid_label": labels[int(best_valid_indices[index])],
                "invalid_logit": float(invalid_logits[index]),
                "best_valid_logit": float(best_valid_logits[index]),
                "rejection_score": float(rejection_scores[index]),
            }
        )

    return {
        "balanced_accuracy": float(balanced_accuracy),
        "valid_accept_recall": float(valid_accept_recall),
        "valid_false_reject_rate": float(valid_false_reject_rate),
        "invalid_reject_recall": float(invalid_reject_recall),
        "invalid_false_accept_rate": float(invalid_false_accept_rate),
        "roc_auc": float(roc_auc),
        "valid_count": valid_count,
        "invalid_count": invalid_count,
        "valid_accept_count": valid_accept_count,
        "valid_false_reject_count": valid_false_reject_count,
        "invalid_reject_count": invalid_reject_count,
        "invalid_false_accept_count": invalid_false_accept_count,
        "per_valid_class": per_valid_class,
        "details": details,
    }


def _write_csv(
    path: Path,
    fieldnames: Sequence[str],
    rows: Sequence[dict[str, Any]],
) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_plotly(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )


def _binary_confusion_plot(summary: dict[str, Any]) -> dict[str, Any]:
    counts = [
        [summary["valid_accept_count"], summary["valid_false_reject_count"]],
        [summary["invalid_false_accept_count"], summary["invalid_reject_count"]],
    ]
    rates = [
        [
            summary["valid_accept_recall"],
            summary["valid_false_reject_rate"],
        ],
        [
            summary["invalid_false_accept_rate"],
            summary["invalid_reject_recall"],
        ],
    ]
    return {
        "data": [
            {
                "type": "heatmap",
                "x": ["accept / valid", "reject / invalid"],
                "y": ["true valid", "true invalid"],
                "z": rates,
                "customdata": counts,
                "zmin": 0,
                "zmax": 1,
                "colorscale": "Blues",
                "hovertemplate": (
                    "%{y}<br>%{x}<br>row-normalized rate=%{z:.2%}"
                    "<br>raw count=%{customdata}<extra></extra>"
                ),
            }
        ],
        "layout": {
            "title": (
                "Validity rejection confusion — row-normalized rates "
                f"(valid n={summary['valid_count']}, invalid n={summary['invalid_count']})"
            ),
            "xaxis": {"title": "binary classifier decision"},
            "yaxis": {"title": "true validity group", "autorange": "reversed"},
        },
    }


def _per_valid_class_plot(summary: dict[str, Any]) -> dict[str, Any]:
    rows = summary["per_valid_class"]
    return {
        "data": [
            {
                "type": "bar",
                "name": "valid → invalid false-reject rate",
                "x": [row["label"] for row in rows],
                "y": [row["false_reject_rate"] for row in rows],
                "customdata": [
                    [row["sample_count"], row["false_reject_count"]]
                    for row in rows
                ],
                "hovertemplate": (
                    "tile=%{x}<br>false-reject rate=%{y:.2%}"
                    "<br>samples=%{customdata[0]}"
                    "<br>false rejects=%{customdata[1]}<extra></extra>"
                ),
            }
        ],
        "layout": {
            "title": "False rejection of real tiles by true tile class",
            "xaxis": {"title": "true tile class"},
            "yaxis": {"title": "false-reject rate", "range": [0, 1]},
        },
    }


def _roc_curve_plot(summary: dict[str, Any]) -> dict[str, Any]:
    details = summary["details"]
    scores = sorted(
        {float(row["rejection_score"]) for row in details},
        reverse=True,
    )
    thresholds = [float("inf"), *scores, float("-inf")]
    valid_count = summary["valid_count"]
    invalid_count = summary["invalid_count"]
    fpr: list[float] = []
    tpr: list[float] = []
    threshold_text: list[str] = []
    for threshold in thresholds:
        rejected = [
            row
            for row in details
            if float(row["rejection_score"]) >= threshold
        ]
        false_rejects = sum(
            row["true_group"] == "valid" for row in rejected
        )
        true_rejects = sum(
            row["true_group"] == "invalid" for row in rejected
        )
        fpr.append(false_rejects / valid_count)
        tpr.append(true_rejects / invalid_count)
        if math.isinf(threshold):
            threshold_text.append("+inf" if threshold > 0 else "-inf")
        else:
            threshold_text.append(f"{threshold:.6g}")
    return {
        "data": [
            {
                "type": "scatter",
                "mode": "lines",
                "name": f"ROC (AUC={summary['roc_auc']:.4f})",
                "x": fpr,
                "y": tpr,
                "customdata": threshold_text,
                "hovertemplate": (
                    "valid false-reject rate=%{x:.2%}"
                    "<br>invalid reject recall=%{y:.2%}"
                    "<br>threshold=%{customdata}<extra></extra>"
                ),
            },
            {
                "type": "scatter",
                "mode": "markers",
                "name": "runtime boundary: score ≥ 0",
                "x": [summary["valid_false_reject_rate"]],
                "y": [summary["invalid_reject_recall"]],
                "hovertemplate": (
                    "score threshold=0"
                    "<br>valid false-reject rate=%{x:.2%}"
                    "<br>invalid reject recall=%{y:.2%}<extra></extra>"
                ),
            },
        ],
        "layout": {
            "title": (
                "Validity rejection ROC — score = invalid_logit − max(valid logits)"
            ),
            "xaxis": {
                "title": "valid false-reject rate",
                "range": [0, 1],
            },
            "yaxis": {
                "title": "invalid reject recall",
                "range": [0, 1],
            },
        },
    }


def _select_errors(summary: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    false_accepts = sorted(
        [
            row
            for row in summary["details"]
            if row["outcome"] == "false_accept"
        ],
        key=lambda row: (float(row["rejection_score"]), str(row["sample_id"])),
    )[:MAX_ERRORS_PER_KIND]
    false_rejects = sorted(
        [
            row
            for row in summary["details"]
            if row["outcome"] == "false_reject"
        ],
        key=lambda row: (-float(row["rejection_score"]), str(row["sample_id"])),
    )[:MAX_ERRORS_PER_KIND]
    return {
        "false_accepts": false_accepts,
        "false_rejects": false_rejects,
    }


def _row_image(row: dict[str, Any], size: int = 96) -> Image.Image:
    return Image.fromarray(row["image"], mode="L").convert("RGB").resize(
        (size, size),
        Image.Resampling.NEAREST,
    )


def _error_contact_sheet(
    path: Path,
    rows: Sequence[dict[str, Any]],
    selections: dict[str, list[dict[str, Any]]],
) -> None:
    columns = 4
    panel_width = 285
    panel_height = 155
    header_height = 34
    sections = [
        ("FALSE ACCEPTS: true invalid → accepted tile", selections["false_accepts"]),
        ("FALSE REJECTS: true valid tile → rejected invalid", selections["false_rejects"]),
    ]
    section_heights: list[int] = []
    for _title, items in sections:
        item_rows = max(1, math.ceil(len(items) / columns))
        section_heights.append(header_height + item_rows * panel_height)
    sheet = Image.new(
        "RGB",
        (columns * panel_width, sum(section_heights)),
        "white",
    )
    draw = ImageDraw.Draw(sheet)
    y_base = 0
    for (title, items), section_height in zip(
        sections, section_heights, strict=True
    ):
        draw.text((10, y_base + 8), title, fill="black")
        if not items:
            draw.text((10, y_base + header_height + 10), "No errors.", fill="gray")
        for position, item in enumerate(items):
            grid_y, grid_x = divmod(position, columns)
            x0 = grid_x * panel_width
            y0 = y_base + header_height + grid_y * panel_height
            row = rows[int(item["row_index"])]
            sheet.paste(_row_image(row), (x0 + 8, y0 + 30))
            draw.text(
                (x0 + 8, y0 + 6),
                f"{row['base_label']} → {item['decision']} | score={item['rejection_score']:.3f}",
                fill="black",
            )
            draw.text(
                (x0 + 112, y0 + 36),
                f"best valid: {item['best_valid_label']}",
                fill="black",
            )
            draw.text(
                (x0 + 112, y0 + 54),
                f"invalid logit: {item['invalid_logit']:.3f}",
                fill="black",
            )
            draw.text(
                (x0 + 112, y0 + 72),
                f"best valid logit: {item['best_valid_logit']:.3f}",
                fill="black",
            )
            if row["invalid_reason"]:
                draw.text(
                    (x0 + 112, y0 + 90),
                    f"reason: {row['invalid_reason']}",
                    fill="black",
                )
            draw.text(
                (x0 + 8, y0 + 132),
                str(row["sample_id"])[-42:],
                fill="gray",
            )
        y_base += section_height
    sheet.save(path)


def _provenance(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    invalid_rows = [row for row in rows if row["base_label"] == INVALID_LABEL]
    valid_rows = [row for row in rows if row["base_label"] != INVALID_LABEL]
    return {
        "valid_source_counts": dict(
            sorted(Counter(str(row["source"]) for row in valid_rows).items())
        ),
        "invalid_source_counts": dict(
            sorted(Counter(str(row["source"]) for row in invalid_rows).items())
        ),
        "invalid_review_decision_counts": dict(
            sorted(
                Counter(
                    str(row["detector_review_decision"])
                    for row in invalid_rows
                ).items()
            )
        ),
        "invalid_reason_counts": dict(
            sorted(Counter(str(row["invalid_reason"]) for row in invalid_rows).items())
        ),
    }


def evaluate(context):
    batch_size = int(context.parameters["batch_size"])
    if batch_size < 1:
        raise ValueError("batch_size must be positive")

    database = context.corpus.root / "dataset.sqlite"
    labels, mean, std = _load_metadata_and_normalization(database)
    if tuple(labels) != EXPECTED_LABELS:
        raise ValueError(
            "Corpus base_labels do not match tile-shape-classification-35-v1"
        )
    rows = _load_rows(database, labels)
    images = (
        torch.from_numpy(np.stack([row["image"] for row in rows]))
        .float()
        .unsqueeze(1)
        .mul(1.0 / 255.0)
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logits = _infer(
        context.model.module.to(device),
        images,
        mean=mean,
        std=std,
        batch_size=batch_size,
        device=device,
    )
    summary = _summarize_rejection(logits, rows, labels)
    metrics: dict[str, int | float] = {
        "balanced_accuracy": summary["balanced_accuracy"],
        "valid_accept_recall": summary["valid_accept_recall"],
        "valid_false_reject_rate": summary["valid_false_reject_rate"],
        "invalid_reject_recall": summary["invalid_reject_recall"],
        "invalid_false_accept_rate": summary["invalid_false_accept_rate"],
        "roc_auc": summary["roc_auc"],
    }

    work_dir = context.work_dir
    work_dir.mkdir(parents=True, exist_ok=True)
    summary_path = work_dir / "validity-rejection-summary.csv"
    confusion_path = work_dir / "validity-rejection-confusion.plotly.json"
    per_class_path = work_dir / "valid-tile-false-rejects.csv"
    per_class_plot_path = work_dir / "valid-tile-false-rejects.plotly.json"
    errors_path = work_dir / "validity-rejection-errors.png"
    samples_path = work_dir / "validity-rejection-per-sample.jsonl"
    roc_path = work_dir / "validity-rejection-roc.plotly.json"
    report_path = work_dir / "validity-rejection-report.json"

    summary_row = {
        "sample_count": len(rows),
        "valid_count": summary["valid_count"],
        "invalid_count": summary["invalid_count"],
        "valid_accept_count": summary["valid_accept_count"],
        "valid_false_reject_count": summary["valid_false_reject_count"],
        "invalid_reject_count": summary["invalid_reject_count"],
        "invalid_false_accept_count": summary["invalid_false_accept_count"],
        **metrics,
        "valid_group_weight": 0.5,
        "invalid_group_weight": 0.5,
    }
    _write_csv(summary_path, tuple(summary_row), [summary_row])
    _write_csv(
        per_class_path,
        (
            "class_index",
            "label",
            "sample_count",
            "false_reject_count",
            "false_reject_rate",
        ),
        summary["per_valid_class"],
    )
    _write_plotly(confusion_path, _binary_confusion_plot(summary))
    _write_plotly(per_class_plot_path, _per_valid_class_plot(summary))
    _write_plotly(roc_path, _roc_curve_plot(summary))

    selections = _select_errors(summary)
    _error_contact_sheet(errors_path, rows, selections)

    by_sample_id = {row["sample_id"]: row for row in rows}
    with samples_path.open("w", encoding="utf-8") as handle:
        for detail in summary["details"]:
            row = by_sample_id[detail["sample_id"]]
            payload = {
                **detail,
                "source": row["source"],
                "capture_id": row["capture_id"],
                "layout_id": row["layout_id"],
                "region": row["region"],
                "source_image_path": row["source_image_path"],
                "detector_review_decision": row["detector_review_decision"],
                "invalid_reason": row["invalid_reason"],
            }
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")

    provenance = _provenance(rows)
    report = {
        "schema": "mjtensu.recognition/tile-validity-rejection-report/v1",
        "split": SPLIT,
        "labels": list(labels),
        "sample_population": {
            "sample_count": len(rows),
            "valid_count": summary["valid_count"],
            "invalid_count": summary["invalid_count"],
            "selection": "all authored samples in corpus split manual_val",
            "perturbations": "none",
            "provenance": provenance,
            "invalid_population_interpretation": (
                "Invalid examples are authored human-reviewed detector candidate crops "
                "retained as unusable/non-tile classifier inputs; no synthetic negatives "
                "are added by this Evaluation Protocol."
            ),
        },
        "metrics": metrics,
        "metric_formulas": {
            "valid_accept_recall": "valid accepted / all true valid",
            "valid_false_reject_rate": "valid rejected / all true valid = 1 - valid_accept_recall",
            "invalid_reject_recall": "invalid rejected / all true invalid",
            "invalid_false_accept_rate": "invalid accepted / all true invalid = 1 - invalid_reject_recall",
            "balanced_accuracy": (
                "0.5 * valid_accept_recall + 0.5 * invalid_reject_recall"
            ),
            "roc_auc": (
                "threshold-free rank separation for rejection_score with true invalid "
                "as the positive class; ties receive half credit"
            ),
        },
        "score_semantics": {
            "formula": "invalid_logit - max(valid_class_logits)",
            "runtime_boundary": "reject when score >= 0",
            "calibration_note": (
                "Raw score magnitude is not treated as calibrated probability and "
                "must not be compared across architectures as an absolute confidence."
            ),
        },
        "error_inspection_selection": {
            "false_accepts": selections["false_accepts"],
            "false_rejects": selections["false_rejects"],
            "rule": (
                "False accepts sort by ascending rejection_score; false rejects sort "
                "by descending rejection_score; sample_id breaks ties; at most 12 each."
            ),
        },
        "telemetry": {
            "source": "automatic terminal Evaluation metric projection",
            "cadence": "once after successful Evaluation result validation",
            "step": 0,
        },
    }
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    return EvaluationCandidate(
        metrics=metrics,
        artifacts={
            "rejection_summary_table": summary_path,
            "binary_confusion_matrix": confusion_path,
            "per_valid_class_false_reject_table": per_class_path,
            "per_valid_class_false_reject_plot": per_class_plot_path,
            "error_inspection_contact_sheet": errors_path,
            "per_sample_details": samples_path,
            "rejection_roc_curve": roc_path,
            "rejection_report": report_path,
        },
    )
