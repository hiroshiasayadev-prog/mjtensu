from __future__ import annotations

import argparse
import base64
import csv
import heapq
import html
import io
import json
import sqlite3
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from build_tile_classifier_dataset import (
    BASE_LABEL_TO_INDEX,
    QualityReview,
    base_label,
    effective_quality_label,
    load_quality_reviews,
    preprocess_gray_u8,
    sqlite_readonly_uri,
)
from tile_shape_classifier import DEFAULT_C8_FIELDS, build_model
from train_tile_shape_classifier import rotate_batch


DEFAULT_ANGLES = (0.0, 15.0, 30.0, 45.0)
DEFAULT_BATCH_SIZE = 2048
DEFAULT_WORKERS = 12
DEFAULT_CANDIDATE_CONFIDENCE = 0.50
DEFAULT_MAX_CONSENSUS_CANDIDATES = 50_000
DEFAULT_HTML_LIMIT = 2_000

RAW_COLUMNS = (
    "crop_id",
    "source",
    "source_partition",
    "tile_label",
    "image_png",
    "source_image_path",
    "source_image_id",
    "source_annotation_id",
    "capture_id",
    "layout_id",
    "layout_ordinal",
    "region",
    "group_name",
    "group_ordinal",
    "tile_ordinal",
    "brightness",
    "shadow",
    "annotation_angle_deg",
    "expected_rotation_deg",
)


def parse_args() -> argparse.Namespace:
    repository_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(
        description=(
            "Audit labels in the persistent tile-crop dataset with a trained 34-class "
            "shape classifier. Training crops from the compact experiment DB are excluded "
            "by default. A cheap 0-degree pass finds disagreements; only suspicious crops "
            "receive the full multi-angle consensus pass."
        )
    )
    parser.add_argument(
        "--database",
        type=Path,
        default=repository_root / ".local" / "recognition" / "tile_crop_dataset" / "dataset.sqlite",
        help="Persistent source crop database.",
    )
    parser.add_argument(
        "--experiment-database",
        type=Path,
        required=True,
        help="Compact classifier database used to identify training crop IDs to exclude.",
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--quality-audit-database",
        type=Path,
        help=(
            "Optional review sidecar from review_tile_crop_label_audit.py. "
            "label_error uses corrected_label, false_detection keeps the source label, "
            "and unusable_crop/background are excluded from evaluation."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=repository_root / ".local" / "recognition" / "tile_crop_label_audit",
    )
    parser.add_argument(
        "--exclude-splits",
        nargs="+",
        default=["train"],
        choices=("train", "manual_val", "jp_val"),
        help="Experiment splits whose crop IDs must not be audited. Defaults to train only.",
    )
    parser.add_argument(
        "--sources",
        nargs="+",
        choices=("jp", "manual"),
        default=["jp", "manual"],
    )
    parser.add_argument("--angles", type=float, nargs="+", default=list(DEFAULT_ANGLES))
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument(
        "--candidate-confidence",
        type=float,
        default=DEFAULT_CANDIDATE_CONFIDENCE,
        help=(
            "0-degree wrong-label predictions below this confidence are counted but do not "
            "enter the expensive consensus pass. Defaults to 0.50."
        ),
    )
    parser.add_argument(
        "--max-consensus-candidates",
        type=int,
        default=DEFAULT_MAX_CONSENSUS_CANDIDATES,
        help="Keep at most this many highest-confidence 0-degree disagreements.",
    )
    parser.add_argument(
        "--html-limit",
        type=int,
        default=DEFAULT_HTML_LIMIT,
        help="Maximum number of ranked candidates embedded into candidates.html.",
    )
    parser.add_argument("--progress-every", type=int, default=50_000)
    parser.add_argument("--limit", type=int, default=0, help="Optional raw-row limit for smoke tests.")
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument(
        "--amp",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use CUDA AMP when CUDA is selected.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    validate_args(args)

    database = args.database.resolve()
    experiment_database = args.experiment_database.resolve()
    checkpoint = args.checkpoint.resolve()
    quality_audit_database = (
        None if args.quality_audit_database is None else args.quality_audit_database.resolve()
    )
    output_dir = args.output_dir.resolve()
    for path in (database, experiment_database, checkpoint):
        if not path.is_file():
            raise FileNotFoundError(path)
    if quality_audit_database is not None and not quality_audit_database.is_file():
        raise FileNotFoundError(quality_audit_database)
    output_dir.mkdir(parents=True, exist_ok=True)

    quality_reviews, quality_summary = load_quality_reviews(quality_audit_database)
    device = resolve_device(str(args.device))
    payload = torch.load(checkpoint, map_location="cpu")
    config = payload.get("config")
    if not isinstance(config, dict):
        raise ValueError("Checkpoint is missing config")
    class_labels = tuple(str(value) for value in config["class_labels"])
    if len(class_labels) != 34:
        raise ValueError(f"Expected 34 classifier labels, found {len(class_labels)}")
    label_to_index = {label: index for index, label in enumerate(class_labels)}
    if label_to_index != BASE_LABEL_TO_INDEX:
        raise ValueError(
            "Checkpoint class ordering does not match build_tile_classifier_dataset.BASE_LABELS"
        )

    model_config = config.get("model", {})
    model_name = str(model_config.get("name", "c8"))
    c8_fields = model_config.get("c8_fields") or list(DEFAULT_C8_FIELDS)
    model = build_model(
        model_name,
        class_count=len(class_labels),
        c8_fields=tuple(int(value) for value in c8_fields),
    )
    model.load_state_dict(payload["model_state_dict"])
    model.to(device).eval()

    normalization = config["normalization"]
    mean = float(normalization["mean"])
    std = float(normalization["std"])
    image_size = int(config["image_size"])
    use_amp = bool(args.amp) and device.type == "cuda"

    excluded_crop_ids = load_excluded_crop_ids(
        experiment_database,
        splits=tuple(str(value) for value in args.exclude_splits),
    )
    print(
        f"[audit] device={device} checkpoint_epoch={payload.get('epoch')} "
        f"excluded_training_crops={len(excluded_crop_ids)}"
    )

    started = time.perf_counter()
    first_pass = scan_disagreements(
        database,
        model=model,
        device=device,
        class_labels=class_labels,
        label_to_index=label_to_index,
        mean=mean,
        std=std,
        image_size=image_size,
        batch_size=int(args.batch_size),
        workers=int(args.workers),
        sources=tuple(str(value) for value in args.sources),
        excluded_crop_ids=excluded_crop_ids,
        quality_reviews=quality_reviews,
        confidence_threshold=float(args.candidate_confidence),
        max_candidates=int(args.max_consensus_candidates),
        progress_every=int(args.progress_every),
        limit=int(args.limit),
        amp=use_amp,
    )

    candidates = first_pass["candidates"]
    print(
        f"[audit] first pass complete: scanned={first_pass['scanned_count']} "
        f"wrong={first_pass['disagreement_count']} "
        f"consensus_candidates={len(candidates)}"
    )

    run_consensus(
        candidates,
        model=model,
        device=device,
        class_labels=class_labels,
        label_to_index=label_to_index,
        mean=mean,
        std=std,
        image_size=image_size,
        batch_size=int(args.batch_size),
        workers=int(args.workers),
        angles=tuple(float(value) for value in args.angles),
        amp=use_amp,
    )
    ranked = rank_candidates(candidates, angle_count=len(args.angles))

    write_candidates_csv(output_dir / "candidates.csv", ranked, angles=args.angles)
    write_candidates_html(
        output_dir / "candidates.html",
        ranked[: int(args.html_limit)],
        angles=args.angles,
    )
    summary = make_summary(
        ranked,
        first_pass=first_pass,
        checkpoint=checkpoint,
        checkpoint_epoch=payload.get("epoch"),
        database=database,
        experiment_database=experiment_database,
        quality_audit_database=quality_audit_database,
        quality_summary=quality_summary,
        excluded_splits=tuple(str(value) for value in args.exclude_splits),
        angles=tuple(float(value) for value in args.angles),
        elapsed_seconds=time.perf_counter() - started,
    )
    atomic_write_json(output_dir / "summary.json", summary)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"[audit] open: {output_dir / 'candidates.html'}")


def validate_args(args: argparse.Namespace) -> None:
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive")
    if args.workers < 1:
        raise ValueError("--workers must be positive")
    if not 0.0 <= args.candidate_confidence <= 1.0:
        raise ValueError("--candidate-confidence must be in [0,1]")
    if args.max_consensus_candidates < 1:
        raise ValueError("--max-consensus-candidates must be positive")
    if args.html_limit < 1:
        raise ValueError("--html-limit must be positive")
    if args.progress_every < 1:
        raise ValueError("--progress-every must be positive")
    if args.limit < 0:
        raise ValueError("--limit must be non-negative")
    if not args.angles:
        raise ValueError("--angles must not be empty")


def resolve_device(requested: str) -> torch.device:
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_excluded_crop_ids(database: Path, *, splits: Sequence[str]) -> set[str]:
    placeholders = ",".join("?" for _ in splits)
    with sqlite3.connect(database) as connection:
        rows = connection.execute(
            f"SELECT crop_id FROM sample WHERE split IN ({placeholders})",
            tuple(splits),
        ).fetchall()
    return {str(row[0]) for row in rows}


def scan_disagreements(
    database: Path,
    *,
    model: torch.nn.Module,
    device: torch.device,
    class_labels: Sequence[str],
    label_to_index: dict[str, int],
    mean: float,
    std: float,
    image_size: int,
    batch_size: int,
    workers: int,
    sources: Sequence[str],
    excluded_crop_ids: set[str],
    quality_reviews: dict[str, QualityReview],
    confidence_threshold: float,
    max_candidates: int,
    progress_every: int,
    limit: int,
    amp: bool,
) -> dict[str, Any]:
    source_placeholders = ",".join("?" for _ in sources)
    sql = (
        f"SELECT {', '.join(RAW_COLUMNS)} FROM tile_crop "
        f"WHERE source IN ({source_placeholders}) ORDER BY rowid"
    )
    scanned_count = 0
    excluded_count = 0
    disagreement_count = 0
    below_threshold_count = 0
    unsupported_label_count = 0
    quality_excluded_count = 0
    heap: list[tuple[float, int, dict[str, Any]]] = []
    sequence = 0

    with sqlite3.connect(sqlite_readonly_uri(database), uri=True, timeout=60) as connection:
        connection.row_factory = sqlite3.Row
        cursor = connection.execute(sql, tuple(sources))
        with ThreadPoolExecutor(max_workers=workers) as executor:
            while True:
                rows = cursor.fetchmany(batch_size * 2)
                if not rows:
                    break
                filtered: list[tuple[sqlite3.Row, str]] = []
                for row in rows:
                    crop_id = str(row["crop_id"])
                    if crop_id in excluded_crop_ids:
                        excluded_count += 1
                        continue
                    source_label = str(row["tile_label"])
                    effective_label, _quality_decision = effective_quality_label(
                        crop_id,
                        source_label,
                        quality_reviews,
                    )
                    if effective_label is None:
                        quality_excluded_count += 1
                        continue
                    try:
                        expected_label = base_label(effective_label)
                    except ValueError:
                        unsupported_label_count += 1
                        continue
                    if expected_label not in label_to_index:
                        unsupported_label_count += 1
                        continue
                    filtered.append((row, expected_label))
                    if limit and scanned_count + len(filtered) >= limit:
                        break
                if not filtered:
                    if limit and scanned_count >= limit:
                        break
                    continue

                prepared = list(
                    executor.map(
                        lambda item: preprocess_row(
                            item[0],
                            expected_label=item[1],
                            image_size=image_size,
                        ),
                        filtered,
                        chunksize=32,
                    )
                )
                images = np.stack([item[0] for item in prepared], axis=0)
                expected_indices = np.asarray(
                    [label_to_index[item[1]] for item in prepared], dtype=np.int64
                )
                prediction, confidence, expected_confidence = infer_batch(
                    model,
                    images,
                    expected_indices=expected_indices,
                    device=device,
                    mean=mean,
                    std=std,
                    angle_deg=0.0,
                    amp=amp,
                )

                for index, (row, _expected_label) in enumerate(filtered):
                    scanned_count += 1
                    expected_index = int(expected_indices[index])
                    predicted_index = int(prediction[index])
                    if predicted_index == expected_index:
                        if limit and scanned_count >= limit:
                            break
                        continue
                    disagreement_count += 1
                    predicted_confidence = float(confidence[index])
                    if predicted_confidence < confidence_threshold:
                        below_threshold_count += 1
                        if limit and scanned_count >= limit:
                            break
                        continue

                    candidate = row_to_candidate(
                        row,
                        expected_label=class_labels[expected_index],
                        zero_prediction=class_labels[predicted_index],
                        zero_confidence=predicted_confidence,
                        zero_expected_confidence=float(expected_confidence[index]),
                    )
                    item = (predicted_confidence, sequence, candidate)
                    sequence += 1
                    if len(heap) < max_candidates:
                        heapq.heappush(heap, item)
                    elif predicted_confidence > heap[0][0]:
                        heapq.heapreplace(heap, item)

                    if limit and scanned_count >= limit:
                        break

                if scanned_count % progress_every < len(filtered):
                    print(
                        f"[audit] scanned={scanned_count:,} wrong={disagreement_count:,} "
                        f"kept={len(heap):,}"
                    )
                if limit and scanned_count >= limit:
                    break

    candidates = [item[2] for item in sorted(heap, key=lambda item: item[:2], reverse=True)]
    return {
        "candidates": candidates,
        "scanned_count": scanned_count,
        "excluded_count": excluded_count,
        "disagreement_count": disagreement_count,
        "below_candidate_confidence_count": below_threshold_count,
        "unsupported_label_count": unsupported_label_count,
        "quality_excluded_count": quality_excluded_count,
        "retained_candidate_count": len(candidates),
        "candidate_cap_reached": disagreement_count - below_threshold_count > max_candidates,
    }


def preprocess_row(
    row: sqlite3.Row,
    *,
    expected_label: str,
    image_size: int,
) -> tuple[np.ndarray, str]:
    raw = preprocess_gray_u8(bytes(row["image_png"]), image_size=image_size)
    image = np.frombuffer(raw, dtype=np.uint8).reshape(image_size, image_size).copy()
    return image, expected_label


def infer_batch(
    model: torch.nn.Module,
    images_u8: np.ndarray,
    *,
    expected_indices: np.ndarray,
    device: torch.device,
    mean: float,
    std: float,
    angle_deg: float,
    amp: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    images = torch.from_numpy(images_u8).to(device, non_blocking=True)
    images = images.float().unsqueeze(1).mul_(1.0 / 255.0)
    if abs(float(angle_deg)) > 1.0e-9:
        angles = torch.full(
            (images.shape[0],),
            float(angle_deg),
            device=device,
            dtype=torch.float32,
        )
        images = rotate_batch(images, angles)
    images = images.sub(mean).div(std)
    targets = torch.from_numpy(expected_indices).to(device, non_blocking=True)

    with torch.inference_mode():
        with torch.cuda.amp.autocast(enabled=amp):
            logits = model(images)
        probabilities = F.softmax(logits.float(), dim=1)
        confidence, prediction = probabilities.max(dim=1)
        expected_confidence = probabilities.gather(1, targets[:, None]).squeeze(1)
    return (
        prediction.cpu().numpy(),
        confidence.cpu().numpy(),
        expected_confidence.cpu().numpy(),
    )


def row_to_candidate(
    row: sqlite3.Row,
    *,
    expected_label: str,
    zero_prediction: str,
    zero_confidence: float,
    zero_expected_confidence: float,
) -> dict[str, Any]:
    return {
        "crop_id": str(row["crop_id"]),
        "source": str(row["source"]),
        "source_partition": str(row["source_partition"]),
        "original_label": str(row["tile_label"]),
        "expected_label": expected_label,
        "image_png": bytes(row["image_png"]),
        "source_image_path": str(row["source_image_path"]),
        "source_image_id": none_or_str(row["source_image_id"]),
        "source_annotation_id": str(row["source_annotation_id"]),
        "capture_id": none_or_str(row["capture_id"]),
        "layout_id": none_or_str(row["layout_id"]),
        "layout_ordinal": none_or_int(row["layout_ordinal"]),
        "region": none_or_str(row["region"]),
        "group_name": none_or_str(row["group_name"]),
        "group_ordinal": none_or_int(row["group_ordinal"]),
        "tile_ordinal": none_or_int(row["tile_ordinal"]),
        "brightness": none_or_str(row["brightness"]),
        "shadow": none_or_str(row["shadow"]),
        "annotation_angle_deg": float(row["annotation_angle_deg"]),
        "expected_rotation_deg": int(row["expected_rotation_deg"]),
        "zero_prediction": zero_prediction,
        "zero_confidence": zero_confidence,
        "zero_expected_confidence": zero_expected_confidence,
        "angles": {},
    }


def run_consensus(
    candidates: list[dict[str, Any]],
    *,
    model: torch.nn.Module,
    device: torch.device,
    class_labels: Sequence[str],
    label_to_index: dict[str, int],
    mean: float,
    std: float,
    image_size: int,
    batch_size: int,
    workers: int,
    angles: Sequence[float],
    amp: bool,
) -> None:
    if not candidates:
        return
    with ThreadPoolExecutor(max_workers=workers) as executor:
        for start in range(0, len(candidates), batch_size):
            batch = candidates[start : start + batch_size]
            images = np.stack(
                list(
                    executor.map(
                        lambda item: np.frombuffer(
                            preprocess_gray_u8(item["image_png"], image_size=image_size),
                            dtype=np.uint8,
                        )
                        .reshape(image_size, image_size)
                        .copy(),
                        batch,
                        chunksize=32,
                    )
                ),
                axis=0,
            )
            expected_indices = np.asarray(
                [label_to_index[item["expected_label"]] for item in batch], dtype=np.int64
            )
            for angle in angles:
                prediction, confidence, expected_confidence = infer_batch(
                    model,
                    images,
                    expected_indices=expected_indices,
                    device=device,
                    mean=mean,
                    std=std,
                    angle_deg=float(angle),
                    amp=amp,
                )
                key = angle_key(float(angle))
                for index, item in enumerate(batch):
                    item["angles"][key] = {
                        "prediction": class_labels[int(prediction[index])],
                        "confidence": float(confidence[index]),
                        "expected_confidence": float(expected_confidence[index]),
                    }

            for item in batch:
                summarize_consensus(item, angle_count=len(angles))
            print(
                f"[audit] consensus={min(start + len(batch), len(candidates)):,}/"
                f"{len(candidates):,}"
            )


def summarize_consensus(item: dict[str, Any], *, angle_count: int) -> None:
    angle_results = list(item["angles"].values())
    counts = Counter(str(result["prediction"]) for result in angle_results)
    max_count = max(counts.values())
    tied = [label for label, count in counts.items() if count == max_count]

    def confidence_for(label: str) -> float:
        values = [
            float(result["confidence"])
            for result in angle_results
            if result["prediction"] == label
        ]
        return float(np.mean(values)) if values else 0.0

    consensus_label = max(tied, key=lambda label: (confidence_for(label), label))
    consensus_count = counts[consensus_label]
    consensus_confidence = confidence_for(consensus_label)
    expected_mean_confidence = float(
        np.mean([float(result["expected_confidence"]) for result in angle_results])
    )
    expected_label = str(item["expected_label"])
    wrong_consensus = consensus_label != expected_label

    if wrong_consensus and consensus_count == angle_count and consensus_confidence >= 0.95:
        tier = 1
    elif wrong_consensus and consensus_count >= max(3, angle_count - 1) and consensus_confidence >= 0.90:
        tier = 2
    else:
        tier = 3

    item["consensus_prediction"] = consensus_label
    item["consensus_count"] = int(consensus_count)
    item["consensus_confidence"] = consensus_confidence
    item["expected_mean_confidence"] = expected_mean_confidence
    item["tier"] = tier


def rank_candidates(
    candidates: Iterable[dict[str, Any]],
    *,
    angle_count: int,
) -> list[dict[str, Any]]:
    return sorted(
        candidates,
        key=lambda item: (
            int(item["tier"]),
            -int(item["consensus_count"]),
            -float(item["consensus_confidence"]),
            float(item["expected_mean_confidence"]),
            -float(item["zero_confidence"]),
            str(item["crop_id"]),
        ),
    )


def write_candidates_csv(
    path: Path,
    candidates: Sequence[dict[str, Any]],
    *,
    angles: Sequence[float],
) -> None:
    angle_fields: list[str] = []
    for angle in angles:
        key = angle_key(float(angle))
        angle_fields.extend(
            [f"{key}_prediction", f"{key}_confidence", f"{key}_expected_confidence"]
        )
    fields = [
        "tier",
        "crop_id",
        "source",
        "source_partition",
        "original_label",
        "expected_label",
        "consensus_prediction",
        "consensus_count",
        "consensus_confidence",
        "expected_mean_confidence",
        "zero_prediction",
        "zero_confidence",
        "zero_expected_confidence",
        "source_image_path",
        "source_image_id",
        "source_annotation_id",
        "capture_id",
        "layout_id",
        "layout_ordinal",
        "region",
        "group_name",
        "group_ordinal",
        "tile_ordinal",
        "brightness",
        "shadow",
        "annotation_angle_deg",
        "expected_rotation_deg",
        *angle_fields,
    ]
    with path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for item in candidates:
            row = {key: value for key, value in item.items() if key != "image_png" and key != "angles"}
            for angle in angles:
                key = angle_key(float(angle))
                result = item["angles"][key]
                row[f"{key}_prediction"] = result["prediction"]
                row[f"{key}_confidence"] = result["confidence"]
                row[f"{key}_expected_confidence"] = result["expected_confidence"]
            writer.writerow(row)


def write_candidates_html(
    path: Path,
    candidates: Sequence[dict[str, Any]],
    *,
    angles: Sequence[float],
) -> None:
    cards: list[str] = []
    for item in candidates:
        angle_lines = []
        for angle in angles:
            key = angle_key(float(angle))
            result = item["angles"][key]
            css = "wrong" if result["prediction"] != item["expected_label"] else "right"
            angle_lines.append(
                f"<div class='{css}'><b>{html.escape(key)}</b>: "
                f"{html.escape(str(result['prediction']))} "
                f"({float(result['confidence']):.4f}); expected-conf="
                f"{float(result['expected_confidence']):.4f}</div>"
            )
        metadata = [
            f"crop: {item['crop_id']}",
            f"source: {item['source']}/{item['source_partition']}",
            f"original label: {item['original_label']}",
            f"expected base: {item['expected_label']}",
            f"consensus: {item['consensus_prediction']} ({item['consensus_count']}/{len(angles)}, {item['consensus_confidence']:.4f})",
            f"source image: {item['source_image_path']}",
            f"source ann: {item['source_annotation_id']}",
            f"capture: {item['capture_id'] or ''}",
            f"layout: {item['layout_id'] or ''}",
            f"region: {item['region'] or ''}",
            f"slot: {item['group_name'] or ''}/{item['group_ordinal']}/{item['tile_ordinal']}",
            f"condition: {item['brightness'] or ''}/{item['shadow'] or ''}",
        ]
        cards.append(
            f"<article class='card tier{int(item['tier'])}'>"
            f"<h2>Tier {int(item['tier'])}: {html.escape(item['expected_label'])} → "
            f"{html.escape(item['consensus_prediction'])}</h2>"
            f"<img src='{image_data_uri(item['image_png'])}' loading='lazy'>"
            f"<div class='angles'>{''.join(angle_lines)}</div>"
            f"<pre>{html.escape(chr(10).join(str(value) for value in metadata))}</pre>"
            "</article>"
        )

    document = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Tile crop label audit</title>
<style>
body {{ font-family: system-ui, sans-serif; margin: 20px; background: #f3f3f3; }}
.grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(340px, 1fr)); gap: 12px; }}
.card {{ background: white; border: 2px solid #aaa; padding: 10px; overflow-wrap: anywhere; }}
.card.tier1 {{ border-width: 4px; }}
.card img {{ display: block; width: 160px; height: 160px; object-fit: contain; margin: 0 auto 8px; image-rendering: auto; }}
.card h2 {{ font-size: 18px; margin: 0 0 8px; }}
.angles {{ font-family: ui-monospace, monospace; font-size: 13px; }}
.wrong {{ font-weight: 700; }}
.right {{ opacity: 0.65; }}
pre {{ white-space: pre-wrap; font-size: 11px; }}
</style>
</head>
<body>
<h1>Tile crop label audit</h1>
<p>Tier 1 = all evaluated angles agree on the same non-expected class with mean prediction confidence ≥ 0.95. Tier 2 = strong near-unanimous disagreement. These are review candidates, not automatic relabel decisions.</p>
<div class="grid">{''.join(cards)}</div>
</body>
</html>
"""
    path.write_text(document, encoding="utf-8")


def make_summary(
    candidates: Sequence[dict[str, Any]],
    *,
    first_pass: dict[str, Any],
    checkpoint: Path,
    checkpoint_epoch: Any,
    database: Path,
    experiment_database: Path,
    quality_audit_database: Path | None,
    quality_summary: dict[str, Any],
    excluded_splits: Sequence[str],
    angles: Sequence[float],
    elapsed_seconds: float,
) -> dict[str, Any]:
    tier_counts = Counter(int(item["tier"]) for item in candidates)
    source_counts = Counter(str(item["source"]) for item in candidates)
    pair_counts = Counter(
        f"{item['expected_label']}->{item['consensus_prediction']}" for item in candidates
    )
    top_pairs = dict(pair_counts.most_common(100))
    return {
        "status": "completed",
        "database": str(database),
        "experiment_database": str(experiment_database),
        "quality_audit_database": None if quality_audit_database is None else str(quality_audit_database),
        "quality_audit": quality_summary,
        "excluded_splits": list(excluded_splits),
        "checkpoint": str(checkpoint),
        "checkpoint_epoch": checkpoint_epoch,
        "angles": list(angles),
        "scanned_count": int(first_pass["scanned_count"]),
        "excluded_count": int(first_pass["excluded_count"]),
        "zero_degree_disagreement_count": int(first_pass["disagreement_count"]),
        "below_candidate_confidence_count": int(first_pass["below_candidate_confidence_count"]),
        "quality_excluded_count": int(first_pass["quality_excluded_count"]),
        "retained_candidate_count": len(candidates),
        "candidate_cap_reached": bool(first_pass["candidate_cap_reached"]),
        "tier_counts": {str(key): value for key, value in sorted(tier_counts.items())},
        "candidate_counts_by_source": dict(sorted(source_counts.items())),
        "top_expected_to_consensus_pairs": top_pairs,
        "elapsed_seconds": elapsed_seconds,
    }


def image_data_uri(image_png: bytes) -> str:
    with Image.open(io.BytesIO(image_png)) as source:
        image = source.convert("RGB")
        image.thumbnail((256, 256), Image.Resampling.LANCZOS)
        output = io.BytesIO()
        image.save(output, format="PNG", compress_level=1)
    encoded = base64.b64encode(output.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def angle_key(angle: float) -> str:
    value = float(angle)
    if value.is_integer():
        return f"{int(value)}deg"
    return f"{value:g}deg"


def none_or_str(value: Any) -> str | None:
    return None if value is None else str(value)


def none_or_int(value: Any) -> int | None:
    return None if value is None else int(value)


def atomic_write_json(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as output:
        json.dump(payload, output, ensure_ascii=False, indent=2)
        output.write("\n")
    temporary.replace(path)


if __name__ == "__main__":
    main()
