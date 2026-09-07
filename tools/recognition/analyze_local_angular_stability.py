from __future__ import annotations

"""Fine-angle stability probe for PRODUCT-INV-RECOGNITION-009."""

import argparse
import csv
import gc
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

try:
    from run_rotation_classifier_experiment import (
        ExperimentCache,
        SplitCache,
        fetch_batch,
        load_cache,
        load_checkpoint_model,
        rotate_batch,
    )
except ModuleNotFoundError:  # package-style import used by tests/tools
    from tools.recognition.run_rotation_classifier_experiment import (
        ExperimentCache,
        SplitCache,
        fetch_batch,
        load_cache,
        load_checkpoint_model,
        rotate_batch,
    )


def parse_args() -> argparse.Namespace:
    repository_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(
        description="Compare fine local angular stability of Plain random360 and production C8."
    )
    parser.add_argument(
        "--database",
        type=Path,
        default=(
            repository_root
            / ".local"
            / "recognition"
            / "tile_classifier_datasets"
            / "gray35_jp500_seed42_v3_jp189.sqlite"
        ),
    )
    parser.add_argument("--plain-checkpoint", type=Path, required=True)
    parser.add_argument("--c8-checkpoint", type=Path, required=True)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=(
            repository_root
            / ".local"
            / "recognition"
            / "plain_random360_local_angular_stability"
        ),
    )
    parser.add_argument("--min-angle", type=int, default=-10)
    parser.add_argument("--max-angle", type=int, default=10)
    parser.add_argument("--angle-step", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=256)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the INV-009 checkpoint comparison")
    if args.angle_step <= 0 or args.max_angle < args.min_angle:
        raise ValueError("invalid angle range")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive")

    angles = tuple(
        float(value)
        for value in range(args.min_angle, args.max_angle + 1, args.angle_step)
    )
    if 0.0 not in angles:
        raise ValueError("INV-009 local sweep must include 0 degrees")

    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    cache = load_cache(args.database.resolve(), cache_device="cpu")
    split = cache.splits["manual_val"]

    results: dict[str, dict[str, Any]] = {}
    prediction_matrices: dict[str, np.ndarray] = {}
    for name, architecture, checkpoint in (
        ("plain-random360-e150", "plain", args.plain_checkpoint.resolve()),
        ("c8-production", "c8", args.c8_checkpoint.resolve()),
    ):
        print(f"[model] {name}: {checkpoint}", flush=True)
        model = load_checkpoint_model(
            checkpoint,
            architecture=architecture,
            class_count=len(cache.class_labels),
            image_size=cache.image_size,
            device="cuda",
        )
        predictions = predict_angle_matrix(
            model,
            split,
            cache=cache,
            angles=angles,
            batch_size=int(args.batch_size),
        )
        metrics = summarize_predictions(
            predictions,
            labels=split.labels.detach().cpu().numpy().astype(np.int64),
            angles=angles,
        )
        results[name] = metrics
        prediction_matrices[name] = predictions
        del model
        gc.collect()
        torch.cuda.empty_cache()

    write_per_angle_csv(output_root / "per_angle.csv", angles, results)
    write_per_sample_csv(
        output_root / "per_sample.csv",
        split=split,
        class_labels=cache.class_labels,
        angles=angles,
        predictions=prediction_matrices,
    )
    summary = {
        "status": "completed",
        "investigation": "PRODUCT-INV-RECOGNITION-009",
        "database": str(args.database.resolve()),
        "split": "manual_val",
        "sample_count": split.count,
        "angles_deg": list(angles),
        "models": {
            "plain-random360-e150": {
                "checkpoint": str(args.plain_checkpoint.resolve()),
                **results["plain-random360-e150"]["summary"],
            },
            "c8-production": {
                "checkpoint": str(args.c8_checkpoint.resolve()),
                **results["c8-production"]["summary"],
            },
        },
    }
    (output_root / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    write_svg_plot(output_root / "local_angular_stability.svg", angles, results)

    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    print(f"outputs: {output_root}", flush=True)


def predict_angle_matrix(
    model: torch.nn.Module,
    split: SplitCache,
    *,
    cache: ExperimentCache,
    angles: Sequence[float],
    batch_size: int,
) -> np.ndarray:
    device = torch.device("cuda")
    predictions = np.empty((len(angles), split.count), dtype=np.int64)
    model.eval()
    with torch.inference_mode():
        for angle_index, angle in enumerate(angles):
            correct = 0
            for start in range(0, split.count, batch_size):
                stop = min(split.count, start + batch_size)
                indices = np.arange(start, stop, dtype=np.int64)
                images, targets = fetch_batch(split, indices, device=device)
                images = images.float().unsqueeze(1).mul_(1.0 / 255.0)
                if abs(float(angle)) > 1.0e-12:
                    angle_tensor = torch.full(
                        (images.shape[0],),
                        float(angle),
                        device=device,
                        dtype=torch.float32,
                    )
                    images = rotate_batch(images, angle_tensor)
                images = images.sub(cache.mean).div(cache.std)
                logits = model(images)
                batch_predictions = logits.argmax(dim=1)
                predictions[angle_index, start:stop] = (
                    batch_predictions.detach().cpu().numpy().astype(np.int64)
                )
                correct += int((batch_predictions == targets).sum().item())
            print(
                f"[angle] {angle:+6.1f} deg accuracy={correct / split.count:.6f}",
                flush=True,
            )
    return predictions


def summarize_predictions(
    predictions: np.ndarray,
    *,
    labels: np.ndarray,
    angles: Sequence[float],
) -> dict[str, Any]:
    if predictions.shape != (len(angles), len(labels)):
        raise ValueError("prediction matrix shape does not match angle/sample dimensions")
    accuracies = np.mean(predictions == labels[None, :], axis=1)
    flip_rates = np.full((len(angles),), np.nan, dtype=np.float64)
    if len(angles) > 1:
        flip_rates[1:] = np.mean(predictions[1:] != predictions[:-1], axis=1)
    zero_index = list(angles).index(0.0)
    zero_correct = predictions[zero_index] == labels
    all_correct = np.all(predictions == labels[None, :], axis=0)
    stable_prediction = np.all(predictions == predictions[0:1, :], axis=0)
    distinct_counts = np.asarray(
        [len(np.unique(predictions[:, index])) for index in range(predictions.shape[1])],
        dtype=np.int64,
    )
    zero_correct_count = int(np.count_nonzero(zero_correct))
    conditional_local_robustness = (
        float(np.count_nonzero(all_correct) / zero_correct_count)
        if zero_correct_count > 0
        else 0.0
    )
    return {
        "per_angle": [
            {
                "angle_deg": float(angle),
                "accuracy": float(accuracies[index]),
                "flip_rate_from_previous_angle": (
                    None if index == 0 else float(flip_rates[index])
                ),
            }
            for index, angle in enumerate(angles)
        ],
        "summary": {
            "zero_degree_accuracy": float(accuracies[zero_index]),
            "mean_accuracy_over_local_window": float(np.mean(accuracies)),
            "worst_accuracy_over_local_window": float(np.min(accuracies)),
            "worst_angle_deg": float(angles[int(np.argmin(accuracies))]),
            "mean_adjacent_angle_flip_rate": float(np.nanmean(flip_rates)),
            "max_adjacent_angle_flip_rate": float(np.nanmax(flip_rates)),
            "stable_prediction_fraction": float(np.mean(stable_prediction)),
            "correct_at_every_angle_fraction": float(np.mean(all_correct)),
            "conditional_all_angle_correct_given_zero_correct": conditional_local_robustness,
            "mean_distinct_predicted_classes_per_sample": float(np.mean(distinct_counts)),
            "max_distinct_predicted_classes_per_sample": int(np.max(distinct_counts)),
        },
    }


def write_per_angle_csv(
    path: Path,
    angles: Sequence[float],
    results: dict[str, dict[str, Any]],
) -> None:
    fieldnames = ["angle_deg"]
    for name in results:
        fieldnames.extend((f"{name}_accuracy", f"{name}_flip_rate_from_previous_angle"))
    with path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        for index, angle in enumerate(angles):
            row: dict[str, Any] = {"angle_deg": angle}
            for name, result in results.items():
                metric = result["per_angle"][index]
                row[f"{name}_accuracy"] = metric["accuracy"]
                row[f"{name}_flip_rate_from_previous_angle"] = metric[
                    "flip_rate_from_previous_angle"
                ]
            writer.writerow(row)


def write_per_sample_csv(
    path: Path,
    *,
    split: SplitCache,
    class_labels: Sequence[str],
    angles: Sequence[float],
    predictions: dict[str, np.ndarray],
) -> None:
    labels = split.labels.detach().cpu().numpy().astype(np.int64)
    zero_index = list(angles).index(0.0)
    names = list(predictions)
    fieldnames = ["sample_id", "target_index", "target_label"]
    for name in names:
        fieldnames.extend(
            (
                f"{name}_zero_prediction",
                f"{name}_distinct_class_count",
                f"{name}_stable_prediction",
                f"{name}_zero_correct",
                f"{name}_all_angles_correct",
                f"{name}_prediction_sequence",
            )
        )
    with path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        for sample_index, sample_id in enumerate(split.sample_ids):
            target_index = int(labels[sample_index])
            row: dict[str, Any] = {
                "sample_id": sample_id,
                "target_index": target_index,
                "target_label": class_labels[target_index],
            }
            for name in names:
                sequence = predictions[name][:, sample_index]
                zero_prediction = int(sequence[zero_index])
                row[f"{name}_zero_prediction"] = class_labels[zero_prediction]
                row[f"{name}_distinct_class_count"] = len(np.unique(sequence))
                row[f"{name}_stable_prediction"] = bool(np.all(sequence == sequence[0]))
                row[f"{name}_zero_correct"] = zero_prediction == target_index
                row[f"{name}_all_angles_correct"] = bool(np.all(sequence == target_index))
                row[f"{name}_prediction_sequence"] = "|".join(
                    class_labels[int(value)] for value in sequence
                )
            writer.writerow(row)


def write_svg_plot(
    path: Path,
    angles: Sequence[float],
    results: dict[str, dict[str, Any]],
) -> None:
    width = 1000
    height = 720
    left = 85
    right = 30
    plot_width = width - left - right
    panel_height = 250
    top_accuracy = 70
    top_flip = 405
    angle_min = float(min(angles))
    angle_max = float(max(angles))
    palette = {
        "plain-random360-e150": "#2563eb",
        "c8-production": "#dc2626",
    }

    def x_for(angle: float) -> float:
        if angle_max == angle_min:
            return float(left)
        return left + (float(angle) - angle_min) / (angle_max - angle_min) * plot_width

    def y_for(value: float, top: float, y_min: float, y_max: float) -> float:
        if y_max <= y_min:
            return top + panel_height / 2
        return top + panel_height - (value - y_min) / (y_max - y_min) * panel_height

    all_accuracies = [
        float(metric["accuracy"])
        for result in results.values()
        for metric in result["per_angle"]
    ]
    accuracy_min = max(0.0, min(all_accuracies) - 0.03)
    accuracy_max = min(1.0, max(all_accuracies) + 0.01)
    all_flip_rates = [
        float(metric["flip_rate_from_previous_angle"])
        for result in results.values()
        for metric in result["per_angle"]
        if metric["flip_rate_from_previous_angle"] is not None
    ]
    flip_max = max(0.02, max(all_flip_rates, default=0.0) * 1.15)

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="500" y="28" text-anchor="middle" font-family="sans-serif" font-size="20" font-weight="700">INV-009 local angular stability</text>',
    ]

    def add_panel(top: float, title: str, y_min: float, y_max: float, metric_key: str) -> None:
        lines.append(
            f'<text x="{left}" y="{top - 20}" font-family="sans-serif" font-size="16" font-weight="700">{title}</text>'
        )
        lines.append(
            f'<rect x="{left}" y="{top}" width="{plot_width}" height="{panel_height}" fill="none" stroke="#444" stroke-width="1"/>'
        )
        for tick in range(6):
            fraction = tick / 5
            value = y_min + (y_max - y_min) * fraction
            y = y_for(value, top, y_min, y_max)
            lines.append(
                f'<line x1="{left}" y1="{y:.2f}" x2="{left + plot_width}" y2="{y:.2f}" stroke="#e5e7eb" stroke-width="1"/>'
            )
            lines.append(
                f'<text x="{left - 10}" y="{y + 4:.2f}" text-anchor="end" font-family="monospace" font-size="11">{value:.3f}</text>'
            )
        for angle in angles:
            if int(angle) % 5 != 0:
                continue
            x = x_for(angle)
            lines.append(
                f'<line x1="{x:.2f}" y1="{top}" x2="{x:.2f}" y2="{top + panel_height}" stroke="#f3f4f6" stroke-width="1"/>'
            )
            lines.append(
                f'<text x="{x:.2f}" y="{top + panel_height + 18}" text-anchor="middle" font-family="monospace" font-size="11">{angle:+g}</text>'
            )
        for name, result in results.items():
            points: list[str] = []
            for metric in result["per_angle"]:
                value = metric[metric_key]
                if value is None:
                    continue
                points.append(
                    f'{x_for(float(metric["angle_deg"])):.2f},{y_for(float(value), top, y_min, y_max):.2f}'
                )
            lines.append(
                f'<polyline points="{" ".join(points)}" fill="none" stroke="{palette.get(name, "#111827")}" stroke-width="2.5"/>'
            )
        lines.append(
            f'<text x="{left + plot_width / 2}" y="{top + panel_height + 42}" text-anchor="middle" font-family="sans-serif" font-size="12">local rotation (degrees)</text>'
        )

    add_panel(top_accuracy, "Accuracy", accuracy_min, accuracy_max, "accuracy")
    add_panel(top_flip, "Adjacent 1-degree argmax flip rate", 0.0, flip_max, "flip_rate_from_previous_angle")

    legend_y = 52
    legend_x = 650
    for offset, name in enumerate(results):
        x = legend_x + offset * 170
        color = palette.get(name, "#111827")
        lines.extend(
            [
                f'<line x1="{x}" y1="{legend_y}" x2="{x + 24}" y2="{legend_y}" stroke="{color}" stroke-width="3"/>',
                f'<text x="{x + 30}" y="{legend_y + 4}" font-family="sans-serif" font-size="11">{name}</text>',
            ]
        )

    lines.append("</svg>")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
