from __future__ import annotations

import argparse
import json
import sqlite3
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

import numpy as np


SCORE_FUNCTIONS: dict[str, Callable[[np.ndarray], float]] = {}


def score_function(name: str) -> Callable[[Callable[[np.ndarray], float]], Callable[[np.ndarray], float]]:
    def decorator(function: Callable[[np.ndarray], float]) -> Callable[[np.ndarray], float]:
        SCORE_FUNCTIONS[name] = function
        return function
    return decorator


@score_function("mean_red_dominance")
def mean_red_dominance(image: np.ndarray) -> float:
    rgb = image.astype(np.int16, copy=False)
    dominance = rgb[:, :, 0] - np.maximum(rgb[:, :, 1], rgb[:, :, 2])
    return float(np.maximum(dominance, 0).mean() / 255.0)


@score_function("red_pixel_fraction_margin20")
def red_pixel_fraction_margin20(image: np.ndarray) -> float:
    rgb = image.astype(np.int16, copy=False)
    dominance = rgb[:, :, 0] - np.maximum(rgb[:, :, 1], rgb[:, :, 2])
    return float(np.mean(dominance > 20))


def parse_args() -> argparse.Namespace:
    repository_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate simple redness-only baselines for red-five classification. "
            "Thresholds are fitted only on jp_val, then reported on jp_test and manual_val."
        )
    )
    parser.add_argument("--repository-root", type=Path, default=repository_root)
    parser.add_argument(
        "--database",
        type=Path,
        help=(
            "Defaults to <repository-root>/.local/recognition/red_five_datasets/"
            "rgb64_binary_jp5000_seed42.sqlite."
        ),
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        help="Defaults to <database stem>.redness_baseline.json.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repository_root = args.repository_root.resolve()
    database = (
        args.database.resolve()
        if args.database is not None
        else repository_root
        / ".local"
        / "recognition"
        / "red_five_datasets"
        / "rgb64_binary_jp5000_seed42.sqlite"
    )
    output_json = (
        args.output_json.resolve()
        if args.output_json is not None
        else database.with_suffix(".redness_baseline.json")
    )
    result = evaluate_database(database)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    result["output_json"] = str(output_json)
    print(json.dumps(result, ensure_ascii=False, indent=2))


def evaluate_database(database: Path) -> dict[str, Any]:
    database = database.resolve()
    if not database.is_file():
        raise FileNotFoundError(database)

    connection = sqlite3.connect(sqlite_readonly_uri(database), uri=True, timeout=60)
    connection.row_factory = sqlite3.Row
    try:
        validate_schema(connection)
        rows_by_split = {
            split: load_split_rows(connection, split)
            for split in ("jp_val", "jp_test", "manual_val")
        }
    finally:
        connection.close()

    if not rows_by_split["jp_val"]:
        raise ValueError("jp_val is empty; cannot fit redness threshold")

    results: dict[str, Any] = {}
    for score_name, function in SCORE_FUNCTIONS.items():
        scored = {
            split: score_rows(rows, function)
            for split, rows in rows_by_split.items()
        }
        fit_scores = np.asarray([item["score"] for item in scored["jp_val"]], dtype=np.float64)
        fit_labels = np.asarray([item["is_red"] for item in scored["jp_val"]], dtype=np.int8)
        threshold, fit_metrics = fit_threshold(fit_scores, fit_labels)

        score_result: dict[str, Any] = {
            "threshold_fit_split": "jp_val",
            "threshold": threshold,
            "jp_val_fit_metrics": fit_metrics,
            "splits": {},
        }
        for split, items in scored.items():
            score_result["splits"][split] = evaluate_scored_rows(items, threshold)
        results[score_name] = score_result

    best_score_name = max(
        results,
        key=lambda name: (
            results[name]["splits"]["manual_val"]["overall"]["balanced_accuracy"]
            if results[name]["splits"]["manual_val"]["overall"]["sample_count"] > 0
            else results[name]["splits"]["jp_test"]["overall"]["balanced_accuracy"]
        ),
    )
    return {
        "status": "completed",
        "database": str(database),
        "threshold_policy": "maximize_balanced_accuracy_on_jp_val",
        "prediction_rule": "red_if_score_greater_than_or_equal_to_threshold",
        "scores": results,
        "best_by_manual_val_balanced_accuracy": best_score_name,
    }


def validate_schema(connection: sqlite3.Connection) -> None:
    table = connection.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='sample'"
    ).fetchone()
    if table is None:
        raise ValueError("Database has no sample table")
    required = {
        "split", "suit", "is_red", "image_size", "image_rgb_u8",
        "source", "brightness", "shadow", "crop_id",
    }
    columns = {str(row[1]) for row in connection.execute("PRAGMA table_info(sample)")}
    missing = required - columns
    if missing:
        raise ValueError(f"sample table is missing columns: {sorted(missing)}")


def load_split_rows(connection: sqlite3.Connection, split: str) -> list[sqlite3.Row]:
    return connection.execute(
        """
        SELECT crop_id, split, source, suit, is_red, image_size, image_rgb_u8,
               COALESCE(brightness, '') AS brightness,
               COALESCE(shadow, '') AS shadow
        FROM sample
        WHERE split=?
        ORDER BY crop_id
        """,
        (split,),
    ).fetchall()


def score_rows(
    rows: list[sqlite3.Row],
    function: Callable[[np.ndarray], float],
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for row in rows:
        image_size = int(row["image_size"])
        raw = bytes(row["image_rgb_u8"])
        expected = image_size * image_size * 3
        if len(raw) != expected:
            raise ValueError(
                f"Invalid RGB byte count for {row['crop_id']}: {len(raw)} != {expected}"
            )
        image = np.frombuffer(raw, dtype=np.uint8).reshape(image_size, image_size, 3)
        result.append(
            {
                "crop_id": str(row["crop_id"]),
                "suit": str(row["suit"]),
                "is_red": int(row["is_red"]),
                "brightness": str(row["brightness"]),
                "shadow": str(row["shadow"]),
                "score": function(image),
            }
        )
    return result


def fit_threshold(scores: np.ndarray, labels: np.ndarray) -> tuple[float, dict[str, Any]]:
    if scores.ndim != 1 or labels.ndim != 1 or scores.shape != labels.shape:
        raise ValueError("scores and labels must be one-dimensional arrays with equal length")
    if scores.size == 0:
        raise ValueError("Cannot fit threshold on an empty dataset")
    if not np.any(labels == 0) or not np.any(labels == 1):
        raise ValueError("Threshold fit requires both normal and red samples")

    unique = np.unique(scores)
    if unique.size == 1:
        candidates = np.asarray([unique[0]], dtype=np.float64)
    else:
        midpoints = (unique[:-1] + unique[1:]) / 2.0
        epsilon = max(1e-12, float(unique[-1] - unique[0]) * 1e-12)
        candidates = np.concatenate(
            ([unique[0] - epsilon], midpoints, [unique[-1] + epsilon])
        )

    best_threshold = float(candidates[0])
    best_metrics = binary_metrics(labels, scores >= best_threshold)
    best_key = (
        best_metrics["balanced_accuracy"],
        best_metrics["f1"],
        best_metrics["accuracy"],
    )
    for candidate in candidates[1:]:
        metrics = binary_metrics(labels, scores >= candidate)
        key = (metrics["balanced_accuracy"], metrics["f1"], metrics["accuracy"])
        if key > best_key:
            best_key = key
            best_threshold = float(candidate)
            best_metrics = metrics
    return best_threshold, best_metrics


def evaluate_scored_rows(items: list[dict[str, Any]], threshold: float) -> dict[str, Any]:
    if not items:
        return {
            "overall": empty_metrics(),
            "by_suit": {},
            "by_condition": {},
        }

    overall = metrics_for_items(items, threshold)
    by_suit: dict[str, Any] = {}
    for suit in "mps":
        subset = [item for item in items if item["suit"] == suit]
        if subset:
            by_suit[suit] = metrics_for_items(subset, threshold)

    by_condition: dict[str, Any] = {}
    groups: defaultdict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for item in items:
        if item["brightness"] or item["shadow"]:
            groups[(item["brightness"], item["shadow"])].append(item)
    for (brightness, shadow), subset in sorted(groups.items()):
        key = f"brightness={brightness}|shadow={shadow}"
        by_condition[key] = metrics_for_items(subset, threshold)

    return {
        "overall": overall,
        "by_suit": by_suit,
        "by_condition": by_condition,
    }


def metrics_for_items(items: list[dict[str, Any]], threshold: float) -> dict[str, Any]:
    labels = np.asarray([item["is_red"] for item in items], dtype=np.int8)
    scores = np.asarray([item["score"] for item in items], dtype=np.float64)
    metrics = binary_metrics(labels, scores >= threshold)
    metrics["score_normal_mean"] = safe_mean(scores[labels == 0])
    metrics["score_red_mean"] = safe_mean(scores[labels == 1])
    metrics["score_normal_p95"] = safe_percentile(scores[labels == 0], 95)
    metrics["score_red_p05"] = safe_percentile(scores[labels == 1], 5)
    return metrics


def binary_metrics(labels: np.ndarray, predictions: np.ndarray) -> dict[str, Any]:
    labels = labels.astype(bool, copy=False)
    predictions = predictions.astype(bool, copy=False)
    tp = int(np.sum(predictions & labels))
    tn = int(np.sum(~predictions & ~labels))
    fp = int(np.sum(predictions & ~labels))
    fn = int(np.sum(~predictions & labels))
    positives = tp + fn
    negatives = tn + fp
    total = positives + negatives

    recall = tp / positives if positives else 0.0
    specificity = tn / negatives if negatives else 0.0
    precision = tp / (tp + fp) if tp + fp else 0.0
    accuracy = (tp + tn) / total if total else 0.0
    balanced_accuracy = (
        (recall + specificity) / 2.0
        if positives and negatives
        else recall if positives else specificity if negatives else 0.0
    )
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "sample_count": total,
        "normal_count": negatives,
        "red_count": positives,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": f1,
    }


def empty_metrics() -> dict[str, Any]:
    return {
        "sample_count": 0,
        "normal_count": 0,
        "red_count": 0,
        "tp": 0,
        "tn": 0,
        "fp": 0,
        "fn": 0,
        "accuracy": 0.0,
        "balanced_accuracy": 0.0,
        "precision": 0.0,
        "recall": 0.0,
        "specificity": 0.0,
        "f1": 0.0,
    }


def safe_mean(values: np.ndarray) -> float | None:
    return None if values.size == 0 else float(np.mean(values))


def safe_percentile(values: np.ndarray, percentile: float) -> float | None:
    return None if values.size == 0 else float(np.percentile(values, percentile))


def sqlite_readonly_uri(path: Path) -> str:
    return f"{path.resolve().as_uri()}?mode=ro"


if __name__ == "__main__":
    main()
