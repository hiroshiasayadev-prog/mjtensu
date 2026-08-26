from __future__ import annotations

import argparse
import base64
import csv
import html
import io
import json
import math
import os
import sqlite3
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from tile_shape_classifier import build_model


DEFAULT_ANGLES = (0.0, 15.0, 30.0, 45.0)


def parse_args() -> argparse.Namespace:
    repository_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(
        description=(
            "Inspect tile-shape classifier validation errors. Produces CSV and an HTML "
            "contact sheet grouped by crop, including expected/original label, prediction, "
            "confidence, capture metadata, and exact evaluation-angle renderings."
        )
    )
    parser.add_argument("--repository-root", type=Path, default=repository_root)
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--split",
        choices=("train", "manual_val", "jp_val"),
        default="manual_val",
    )
    parser.add_argument(
        "--angles",
        type=float,
        nargs="+",
        default=list(DEFAULT_ANGLES),
    )
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help="0 means include all crops with at least one error.",
    )
    parser.add_argument(
        "--audit-per-class-source",
        type=int,
        default=12,
        help=(
            "Write a deterministic label-audit gallery with up to this many examples "
            "per (base class, source). 0 disables the audit gallery."
        ),
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cuda", "cpu"),
        default="auto",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    database = args.database.resolve()
    checkpoint = args.checkpoint.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not database.is_file():
        raise FileNotFoundError(database)
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive")
    if not args.angles:
        raise ValueError("--angles must not be empty")

    device = resolve_device(args.device)
    payload = torch.load(checkpoint, map_location="cpu")
    config = payload.get("config")
    if not isinstance(config, dict):
        raise ValueError("Checkpoint does not contain a config dictionary")

    model_info = config.get("model") or {}
    model_name = str(model_info.get("name", "c8"))
    c8_fields = model_info.get("c8_fields") or [8, 16, 32, 64]
    class_labels = tuple(str(value) for value in config.get("class_labels", []))
    if len(class_labels) not in (34, 35):
        raise ValueError(
            f"Expected 34 or 35 class labels in checkpoint config, got {len(class_labels)}"
        )
    if len(class_labels) == 35 and class_labels[-1] != "invalid":
        raise ValueError("35-class checkpoint must use 'invalid' as the final class")

    normalization = config.get("normalization") or {}
    mean = float(normalization["mean"])
    std = float(normalization["std"])

    model = build_model(
        model_name,
        class_count=len(class_labels),
        c8_fields=tuple(int(value) for value in c8_fields),
    )
    model.load_state_dict(payload["model_state_dict"])
    model.to(device)
    model.eval()

    samples, image_size = load_split(database, args.split)
    print(
        f"[inspect] split={args.split} samples={len(samples)} image_size={image_size} "
        f"checkpoint_epoch={payload.get('epoch')} device={device}"
    )

    angle_results = evaluate_angles(
        model,
        samples,
        image_size=image_size,
        class_labels=class_labels,
        angles=tuple(float(value) for value in args.angles),
        batch_size=int(args.batch_size),
        mean=mean,
        std=std,
        device=device,
    )

    grouped = build_grouped_results(samples, angle_results)
    suspicious = [item for item in grouped if item["wrong_angle_count"] > 0]
    suspicious.sort(
        key=lambda item: (
            -int(item["wrong_angle_count"]),
            -float(item["max_wrong_confidence"]),
            str(item["sample_id"]),
        )
    )
    if int(args.max_samples) > 0:
        suspicious = suspicious[: int(args.max_samples)]

    write_error_csv(output_dir / "errors.csv", suspicious)
    write_sample_summary_csv(output_dir / "sample_summary.csv", grouped)
    write_html(
        output_dir / "errors.html",
        suspicious,
        class_labels=class_labels,
        angles=tuple(float(value) for value in args.angles),
        split=args.split,
        checkpoint=checkpoint,
        checkpoint_epoch=payload.get("epoch"),
    )
    if int(args.audit_per_class_source) > 0:
        write_label_audit_html(
            output_dir / "label_audit.html",
            samples,
            per_class_source=int(args.audit_per_class_source),
            split=args.split,
        )

    summary = build_summary(
        grouped,
        angles=tuple(float(value) for value in args.angles),
        split=args.split,
        checkpoint=checkpoint,
        checkpoint_epoch=payload.get("epoch"),
    )
    atomic_write_json(output_dir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"[inspect] open: {output_dir / 'errors.html'}")
    if int(args.audit_per_class_source) > 0:
        print(f"[inspect] audit: {output_dir / 'label_audit.html'}")


def resolve_device(requested: str) -> torch.device:
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_split(database: Path, split: str) -> tuple[list[dict[str, Any]], int]:
    connection = sqlite3.connect(f"{database.as_uri()}?mode=ro", uri=True, timeout=60)
    connection.row_factory = sqlite3.Row
    try:
        metadata = {
            str(row["key"]): str(row["value"])
            for row in connection.execute("SELECT key, value FROM experiment_metadata")
        }
        image_size = int(metadata["image_size"])
        rows = connection.execute(
            """
            SELECT
                sample_id,
                source,
                source_partition,
                base_label,
                class_index,
                original_label,
                crop_id,
                image_gray_u8,
                original_width,
                original_height,
                source_image_path,
                source_image_id,
                source_annotation_id,
                capture_id,
                layout_id,
                region,
                brightness,
                shadow,
                annotation_angle_deg,
                expected_rotation_deg,
                detector_candidate_id,
                detector_review_decision,
                invalid_reason
            FROM sample
            WHERE split = ?
            ORDER BY sample_id
            """,
            (split,),
        ).fetchall()
    finally:
        connection.close()

    expected_bytes = image_size * image_size
    samples: list[dict[str, Any]] = []
    for row in rows:
        raw = bytes(row["image_gray_u8"])
        if len(raw) != expected_bytes:
            raise ValueError(
                f"{row['sample_id']} has {len(raw)} gray bytes; expected {expected_bytes}"
            )
        sample = {key: row[key] for key in row.keys() if key != "image_gray_u8"}
        sample["image_u8"] = np.frombuffer(raw, dtype=np.uint8).reshape(image_size, image_size).copy()
        samples.append(sample)
    if not samples:
        raise ValueError(f"Split is empty: {split}")
    return samples, image_size


def evaluate_angles(
    model: torch.nn.Module,
    samples: Sequence[dict[str, Any]],
    *,
    image_size: int,
    class_labels: Sequence[str],
    angles: Sequence[float],
    batch_size: int,
    mean: float,
    std: float,
    device: torch.device,
) -> dict[float, list[dict[str, Any]]]:
    base_images = torch.from_numpy(
        np.stack([sample["image_u8"] for sample in samples], axis=0)
    )
    targets = torch.tensor(
        [int(sample["class_index"]) for sample in samples], dtype=torch.long
    )

    results: dict[float, list[dict[str, Any]]] = {}
    with torch.inference_mode():
        for angle in angles:
            predictions: list[int] = []
            confidences: list[float] = []
            expected_confidences: list[float] = []
            rendered: list[np.ndarray] = []

            for start in range(0, len(samples), batch_size):
                end = min(len(samples), start + batch_size)
                images = base_images[start:end].to(device=device, non_blocking=False)
                batch_targets = targets[start:end].to(device=device, non_blocking=False)
                images = images.float().unsqueeze(1).mul_(1.0 / 255.0)
                if abs(float(angle)) > 1.0e-9:
                    angle_tensor = torch.full(
                        (images.shape[0],),
                        float(angle),
                        dtype=torch.float32,
                        device=device,
                    )
                    images = rotate_batch(images, angle_tensor)

                rendered_batch = (
                    images.squeeze(1)
                    .clamp(0.0, 1.0)
                    .mul(255.0)
                    .round()
                    .to(torch.uint8)
                    .cpu()
                    .numpy()
                )
                normalized = images.sub(mean).div(std)
                logits = model(normalized)
                probabilities = torch.softmax(logits.float(), dim=1)
                confidence, prediction = probabilities.max(dim=1)
                expected_confidence = probabilities.gather(1, batch_targets[:, None]).squeeze(1)

                predictions.extend(int(value) for value in prediction.cpu().tolist())
                confidences.extend(float(value) for value in confidence.cpu().tolist())
                expected_confidences.extend(
                    float(value) for value in expected_confidence.cpu().tolist()
                )
                rendered.extend(rendered_batch)

            per_sample: list[dict[str, Any]] = []
            for index, prediction in enumerate(predictions):
                expected = int(samples[index]["class_index"])
                per_sample.append(
                    {
                        "angle": float(angle),
                        "prediction_index": prediction,
                        "prediction_label": class_labels[prediction],
                        "confidence": confidences[index],
                        "expected_confidence": expected_confidences[index],
                        "correct": prediction == expected,
                        "rendered_u8": rendered[index],
                    }
                )
            results[float(angle)] = per_sample
            correct = sum(1 for value in per_sample if value["correct"])
            print(
                f"[inspect] angle={format_angle(angle)} accuracy={correct / len(samples):.6f} "
                f"errors={len(samples) - correct}/{len(samples)}"
            )
    return results


def rotate_batch(images: torch.Tensor, angles_deg: torch.Tensor) -> torch.Tensor:
    radians = angles_deg.to(dtype=torch.float32) * (math.pi / 180.0)
    cosine = torch.cos(radians)
    sine = torch.sin(radians)
    theta = torch.zeros(
        (images.shape[0], 2, 3),
        device=images.device,
        dtype=torch.float32,
    )
    theta[:, 0, 0] = cosine
    theta[:, 0, 1] = -sine
    theta[:, 1, 0] = sine
    theta[:, 1, 1] = cosine
    grid = F.affine_grid(theta, images.shape, align_corners=False)
    return F.grid_sample(
        images,
        grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=False,
    )


def build_grouped_results(
    samples: Sequence[dict[str, Any]],
    angle_results: dict[float, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    grouped: list[dict[str, Any]] = []
    angles = list(angle_results)
    for index, sample in enumerate(samples):
        evaluations = {angle: angle_results[angle][index] for angle in angles}
        wrong = [value for value in evaluations.values() if not value["correct"]]
        item = dict(sample)
        item.pop("image_u8", None)
        item["evaluations"] = evaluations
        item["wrong_angle_count"] = len(wrong)
        item["max_wrong_confidence"] = max(
            (float(value["confidence"]) for value in wrong),
            default=0.0,
        )
        item["min_expected_confidence"] = min(
            float(value["expected_confidence"]) for value in evaluations.values()
        )
        grouped.append(item)
    return grouped


def write_error_csv(path: Path, suspicious: Sequence[dict[str, Any]]) -> None:
    fieldnames = [
        "sample_id",
        "crop_id",
        "base_label",
        "original_label",
        "angle_deg",
        "prediction_label",
        "confidence",
        "expected_confidence",
        "wrong_angle_count",
        "capture_id",
        "brightness",
        "shadow",
        "region",
        "source_image_path",
        "source_annotation_id",
        "annotation_angle_deg",
        "expected_rotation_deg",
        "detector_candidate_id",
        "invalid_reason",
    ]
    with path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        for item in suspicious:
            for angle, evaluation in item["evaluations"].items():
                if evaluation["correct"]:
                    continue
                writer.writerow(
                    {
                        "sample_id": item["sample_id"],
                        "crop_id": item["crop_id"],
                        "base_label": item["base_label"],
                        "original_label": item["original_label"],
                        "angle_deg": angle,
                        "prediction_label": evaluation["prediction_label"],
                        "confidence": evaluation["confidence"],
                        "expected_confidence": evaluation["expected_confidence"],
                        "wrong_angle_count": item["wrong_angle_count"],
                        "capture_id": item.get("capture_id"),
                        "brightness": item.get("brightness"),
                        "shadow": item.get("shadow"),
                        "region": item.get("region"),
                        "source_image_path": item.get("source_image_path"),
                        "source_annotation_id": item.get("source_annotation_id"),
                        "annotation_angle_deg": item.get("annotation_angle_deg"),
                        "expected_rotation_deg": item.get("expected_rotation_deg"),
                        "detector_candidate_id": item.get("detector_candidate_id"),
                        "invalid_reason": item.get("invalid_reason"),
                    }
                )


def write_sample_summary_csv(path: Path, grouped: Sequence[dict[str, Any]]) -> None:
    fieldnames = [
        "sample_id",
        "crop_id",
        "base_label",
        "original_label",
        "wrong_angle_count",
        "max_wrong_confidence",
        "min_expected_confidence",
        "capture_id",
        "brightness",
        "shadow",
        "region",
        "source_image_path",
        "source_annotation_id",
    ]
    with path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        for item in sorted(
            grouped,
            key=lambda value: (
                -int(value["wrong_angle_count"]),
                -float(value["max_wrong_confidence"]),
                str(value["sample_id"]),
            ),
        ):
            writer.writerow({field: item.get(field) for field in fieldnames})


def write_html(
    path: Path,
    suspicious: Sequence[dict[str, Any]],
    *,
    class_labels: Sequence[str],
    angles: Sequence[float],
    split: str,
    checkpoint: Path,
    checkpoint_epoch: Any,
) -> None:
    rows: list[str] = []
    for item in suspicious:
        image_cells: list[str] = []
        for angle in angles:
            evaluation = item["evaluations"][float(angle)]
            image_uri = image_data_uri(evaluation["rendered_u8"])
            status = "ok" if evaluation["correct"] else "ng"
            image_cells.append(
                "<td class='angle-cell {status}'>"
                "<div class='angle-title'>{angle}</div>"
                "<img src='{image_uri}' width='128' height='128' loading='lazy'>"
                "<div><b>{pred}</b> {confidence:.1%}</div>"
                "<div class='expected-confidence'>expected conf {expected_confidence:.1%}</div>"
                "</td>".format(
                    status=status,
                    angle=html.escape(format_angle(angle)),
                    image_uri=image_uri,
                    pred=html.escape(str(evaluation["prediction_label"])),
                    confidence=float(evaluation["confidence"]),
                    expected_confidence=float(evaluation["expected_confidence"]),
                )
            )

        metadata = "<br>".join(
            [
                f"sample: {html.escape(str(item['sample_id']))}",
                f"crop: {html.escape(str(item['crop_id']))}",
                f"source: {html.escape(str(item.get('source') or ''))}",
                f"detector review: {html.escape(str(item.get('detector_review_decision') or 'unreviewed-validity'))}",
                f"capture: {html.escape(str(item.get('capture_id') or ''))}",
                f"brightness: {html.escape(str(item.get('brightness') or ''))}",
                f"shadow: {html.escape(str(item.get('shadow') or ''))}",
                f"region: {html.escape(str(item.get('region') or ''))}",
                f"ann angle: {html.escape(str(item.get('annotation_angle_deg') or ''))}",
                f"expected rot: {html.escape(str(item.get('expected_rotation_deg') or ''))}",
                f"detector candidate: {html.escape(str(item.get('detector_candidate_id') or ''))}",
                f"invalid reason: {html.escape(str(item.get('invalid_reason') or ''))}",
                f"source ann: {html.escape(str(item.get('source_annotation_id') or ''))}",
                f"source path: {html.escape(str(item.get('source_image_path') or ''))}",
            ]
        )
        rows.append(
            "<tr>"
            "<td class='metadata'>"
            f"<div class='expected'>expected: <b>{html.escape(str(item['base_label']))}</b></div>"
            f"<div>original label: <b>{html.escape(str(item['original_label']))}</b></div>"
            f"<div>wrong angles: <b>{int(item['wrong_angle_count'])}</b></div>"
            f"<div>max wrong conf: {float(item['max_wrong_confidence']):.1%}</div>"
            f"<hr>{metadata}"
            "</td>"
            + "".join(image_cells)
            + "</tr>"
        )

    document = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Tile classifier errors</title>
<style>
body {{ font-family: system-ui, sans-serif; margin: 20px; background: #f5f5f5; }}
table {{ border-collapse: collapse; width: 100%; background: white; }}
th, td {{ border: 1px solid #ccc; padding: 8px; vertical-align: top; }}
th {{ position: sticky; top: 0; background: #eee; z-index: 2; }}
.metadata {{ min-width: 340px; font-size: 13px; line-height: 1.35; }}
.expected {{ font-size: 18px; margin-bottom: 4px; }}
.angle-cell {{ text-align: center; min-width: 150px; }}
.angle-cell img {{ image-rendering: auto; border: 2px solid #aaa; }}
.angle-cell.ok img {{ border-color: #3a3; }}
.angle-cell.ng {{ background: #ffeaea; }}
.angle-cell.ng img {{ border-color: #d33; }}
.angle-title {{ font-weight: 700; margin-bottom: 4px; }}
.expected-confidence {{ color: #666; font-size: 12px; }}
.summary {{ margin-bottom: 16px; }}
</style>
</head>
<body>
<h1>Tile classifier validation errors</h1>
<div class="summary">
<div>split: <b>{html.escape(split)}</b></div>
<div>checkpoint: <code>{html.escape(str(checkpoint))}</code></div>
<div>checkpoint epoch: <b>{html.escape(str(checkpoint_epoch))}</b></div>
<div>crops with at least one error: <b>{len(suspicious)}</b></div>
<div>Rows are sorted by number of wrong angles, then wrong-prediction confidence. Red cells are mistakes.</div>
</div>
<table>
<thead><tr><th>metadata</th>{''.join(f'<th>{html.escape(format_angle(a))}</th>' for a in angles)}</tr></thead>
<tbody>
{''.join(rows)}
</tbody>
</table>
</body>
</html>
"""
    path.write_text(document, encoding="utf-8")


def write_label_audit_html(
    path: Path,
    samples: Sequence[dict[str, Any]],
    *,
    per_class_source: int,
    split: str,
) -> None:
    groups: defaultdict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for sample in samples:
        groups[(str(sample["base_label"]), str(sample["source"]))].append(sample)

    sections: list[str] = []
    for (base_label, source), values in sorted(groups.items()):
        ordered = sorted(values, key=lambda item: str(item["sample_id"]))
        if len(ordered) > per_class_source:
            # Deterministic spread across the full sorted group, rather than taking only
            # neighboring samples from one capture/file region.
            positions = np.linspace(0, len(ordered) - 1, per_class_source, dtype=int)
            selected = [ordered[int(position)] for position in positions]
        else:
            selected = ordered

        cards: list[str] = []
        for item in selected:
            uri = image_data_uri(item["image_u8"])
            metadata = "<br>".join(
                [
                    f"original: <b>{html.escape(str(item['original_label']))}</b>",
                    f"crop: {html.escape(str(item['crop_id']))}",
                    f"capture: {html.escape(str(item.get('capture_id') or ''))}",
                    f"brightness: {html.escape(str(item.get('brightness') or ''))}",
                    f"shadow: {html.escape(str(item.get('shadow') or ''))}",
                    f"region: {html.escape(str(item.get('region') or ''))}",
                    f"source ann: {html.escape(str(item.get('source_annotation_id') or ''))}",
                ]
            )
            cards.append(
                "<div class='card'>"
                f"<img src='{uri}' width='128' height='128' loading='lazy'>"
                f"<div>{metadata}</div>"
                "</div>"
            )
        sections.append(
            f"<h2>{html.escape(base_label)} / {html.escape(source)} "
            f"({len(values)} total, showing {len(selected)})</h2>"
            f"<div class='grid'>{''.join(cards)}</div>"
        )

    document = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Tile classifier label audit</title>
<style>
body {{ font-family: system-ui, sans-serif; margin: 20px; background: #f5f5f5; }}
h2 {{ margin-top: 32px; }}
.grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(180px, 1fr)); gap: 10px; }}
.card {{ background: white; border: 1px solid #ccc; padding: 8px; font-size: 12px; overflow-wrap: anywhere; }}
.card img {{ display: block; margin: 0 auto 6px auto; border: 1px solid #888; }}
</style>
</head>
<body>
<h1>Tile classifier label audit</h1>
<p>split: <b>{html.escape(split)}</b>. Samples are grouped by expected base label and source. This gallery is independent of model correctness and is intended to spot label/preprocessing mistakes.</p>
{''.join(sections)}
</body>
</html>
"""
    path.write_text(document, encoding="utf-8")


def image_data_uri(array: np.ndarray) -> str:
    image = Image.fromarray(np.asarray(array, dtype=np.uint8), mode="L")
    output = io.BytesIO()
    image.save(output, format="PNG", optimize=False)
    encoded = base64.b64encode(output.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def build_summary(
    grouped: Sequence[dict[str, Any]],
    *,
    angles: Sequence[float],
    split: str,
    checkpoint: Path,
    checkpoint_epoch: Any,
) -> dict[str, Any]:
    by_angle: dict[str, Any] = {}
    for angle in angles:
        evaluations = [item["evaluations"][float(angle)] for item in grouped]
        errors = [value for value in evaluations if not value["correct"]]
        by_angle[format_angle(angle)] = {
            "count": len(evaluations),
            "correct": len(evaluations) - len(errors),
            "errors": len(errors),
            "accuracy": (len(evaluations) - len(errors)) / len(evaluations),
        }

    wrong_count_histogram: defaultdict[str, int] = defaultdict(int)
    for item in grouped:
        wrong_count_histogram[str(int(item["wrong_angle_count"]))] += 1

    repeat_errors = [
        {
            "sample_id": item["sample_id"],
            "base_label": item["base_label"],
            "original_label": item["original_label"],
            "wrong_angle_count": item["wrong_angle_count"],
            "max_wrong_confidence": item["max_wrong_confidence"],
            "capture_id": item.get("capture_id"),
            "brightness": item.get("brightness"),
            "shadow": item.get("shadow"),
            "region": item.get("region"),
            "source_image_path": item.get("source_image_path"),
        }
        for item in sorted(
            (value for value in grouped if int(value["wrong_angle_count"]) >= 2),
            key=lambda value: (
                -int(value["wrong_angle_count"]),
                -float(value["max_wrong_confidence"]),
            ),
        )[:50]
    ]
    return {
        "status": "completed",
        "split": split,
        "checkpoint": str(checkpoint),
        "checkpoint_epoch": checkpoint_epoch,
        "sample_count": len(grouped),
        "crops_with_any_error": sum(1 for item in grouped if item["wrong_angle_count"] > 0),
        "by_angle": by_angle,
        "wrong_angle_count_histogram": dict(sorted(wrong_count_histogram.items())),
        "repeat_errors": repeat_errors,
    }



def format_angle(value: float) -> str:
    value = float(value)
    return f"{int(value)}deg" if value.is_integer() else f"{value:g}deg"


def atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent, text=True
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as output:
            json.dump(payload, output, ensure_ascii=False, indent=2)
            output.write("\n")
        os.replace(temporary_name, path)
    except Exception:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


if __name__ == "__main__":
    main()
