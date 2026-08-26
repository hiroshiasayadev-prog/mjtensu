from __future__ import annotations

import argparse
import json
import os
import sqlite3
import tempfile
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import torch

try:
    from build_red_five_classifier_dataset import preprocess_rgb_u8
    from red_five_classifier import build_model
    from train_red_five_classifier import (
        normalize_tensor,
        rgb_u8_to_input_torch,
        rotate_batch,
    )
except ModuleNotFoundError:  # package import path used by unit tests
    from tools.recognition.build_red_five_classifier_dataset import preprocess_rgb_u8
    from tools.recognition.red_five_classifier import build_model
    from tools.recognition.train_red_five_classifier import (
        normalize_tensor,
        rgb_u8_to_input_torch,
        rotate_batch,
    )


DEFAULT_INPUT_MODES = ("rgb", "cr", "ycr")
SOURCE_LABELS = ("5m", "red5m", "5p", "red5p", "5s", "red5s")
DEFAULT_BATCH_SIZE = 1024
DEFAULT_WORKERS = min(12, os.cpu_count() or 1)


@dataclass(frozen=True)
class SourceRow:
    crop_id: str
    source: str
    source_partition: str
    suit: str
    is_red: int
    source_label: str
    image_png: bytes
    source_image_path: str
    source_image_id: str | None
    source_annotation_id: str
    capture_id: str | None
    brightness: str
    shadow: str


@dataclass
class LoadedModel:
    input_mode: str
    checkpoint_path: Path
    model: torch.nn.Module
    mean: tuple[float, ...]
    std: tuple[float, ...]
    epoch: int


class BinaryCounter:
    __slots__ = ("tp", "tn", "fp", "fn")

    def __init__(self) -> None:
        self.tp = 0
        self.tn = 0
        self.fp = 0
        self.fn = 0

    def add(self, target: int, prediction: int) -> None:
        if target == 1 and prediction == 1:
            self.tp += 1
        elif target == 0 and prediction == 0:
            self.tn += 1
        elif target == 0 and prediction == 1:
            self.fp += 1
        elif target == 1 and prediction == 0:
            self.fn += 1
        else:
            raise ValueError(f"Binary target/prediction expected, got {target}/{prediction}")

    def to_json(self) -> dict[str, Any]:
        normal = self.tn + self.fp
        red = self.tp + self.fn
        count = normal + red
        recall = self.tp / red if red else 0.0
        specificity = self.tn / normal if normal else 0.0
        precision = self.tp / (self.tp + self.fp) if self.tp + self.fp else 0.0
        accuracy = (self.tp + self.tn) / count if count else 0.0
        balanced = 0.5 * (recall + specificity) if normal and red else accuracy
        f1 = (
            2.0 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )
        return {
            "sample_count": count,
            "normal_count": normal,
            "red_count": red,
            "tp": self.tp,
            "tn": self.tn,
            "fp": self.fp,
            "fn": self.fn,
            "accuracy": accuracy,
            "balanced_accuracy": balanced,
            "precision": precision,
            "recall": recall,
            "specificity": specificity,
            "f1": f1,
        }


class MetricAccumulator:
    def __init__(self) -> None:
        self.overall = BinaryCounter()
        self.by_membership: defaultdict[str, BinaryCounter] = defaultdict(BinaryCounter)
        self.by_source_partition: defaultdict[str, BinaryCounter] = defaultdict(BinaryCounter)
        self.by_source_label: defaultdict[str, BinaryCounter] = defaultdict(BinaryCounter)
        self.by_suit: defaultdict[str, BinaryCounter] = defaultdict(BinaryCounter)
        self.by_manual_condition: defaultdict[str, BinaryCounter] = defaultdict(BinaryCounter)

    def add(
        self,
        *,
        row: SourceRow,
        membership: str,
        prediction: int,
    ) -> None:
        target = row.is_red
        self.overall.add(target, prediction)
        self.by_membership[membership].add(target, prediction)
        self.by_source_partition[f"{row.source}/{row.source_partition}"].add(
            target, prediction
        )
        self.by_source_label[row.source_label].add(target, prediction)
        self.by_suit[row.suit].add(target, prediction)
        if row.source == "manual":
            condition = f"brightness={row.brightness}|shadow={row.shadow}"
            self.by_manual_condition[condition].add(target, prediction)

    def to_json(self) -> dict[str, Any]:
        return {
            "overall": self.overall.to_json(),
            "by_experiment_membership": counters_to_json(self.by_membership),
            "by_source_partition": counters_to_json(self.by_source_partition),
            "by_source_label": counters_to_json(self.by_source_label),
            "by_suit": counters_to_json(self.by_suit),
            "by_manual_condition": counters_to_json(self.by_manual_condition),
        }


def parse_args() -> argparse.Namespace:
    repository_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate RGB, Cr, and Y+Cr best C8 red-five checkpoints against every "
            "sample in red_five_all.sqlite. The compact experiment DB is used only "
            "to label whether each crop was train/validation/test/unselected."
        )
    )
    parser.add_argument("--repository-root", type=Path, default=repository_root)
    parser.add_argument(
        "--source-database",
        type=Path,
        help=(
            "Defaults to <repository-root>/.local/recognition/red_five_datasets/"
            "red_five_all.sqlite."
        ),
    )
    parser.add_argument(
        "--experiment-database",
        type=Path,
        help=(
            "Defaults to <repository-root>/.local/recognition/red_five_datasets/"
            "rgb64_binary_jp5000_seed42.sqlite."
        ),
    )
    parser.add_argument(
        "--run-root",
        type=Path,
        help=(
            "Defaults to <repository-root>/.local/recognition/red_five_runs/"
            "c8_rgb_cr_ycr_seed42."
        ),
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        help="Defaults to <run-root>/all_samples_evaluation.json.",
    )
    parser.add_argument(
        "--errors-jsonl",
        type=Path,
        help="Defaults to <run-root>/all_samples_errors.jsonl.",
    )
    parser.add_argument(
        "--input-modes",
        nargs="+",
        choices=DEFAULT_INPUT_MODES,
        default=list(DEFAULT_INPUT_MODES),
    )
    parser.add_argument(
        "--angles",
        type=float,
        nargs="+",
        default=[0.0],
        help=(
            "Evaluation rotations in degrees. Default is only 0 degrees because this "
            "audit targets the actual stored crops. Use 0 15 30 45 for a full angle sweep."
        ),
    )
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument(
        "--amp",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive")
    if args.workers < 1:
        raise ValueError("--workers must be positive")
    if not args.angles:
        raise ValueError("--angles must not be empty")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this evaluator")

    root = args.repository_root.resolve()
    if args.source_database is not None:
        source_database = args.source_database.resolve()
    else:
        compact_source = (
            root / ".local" / "recognition" / "red_five_datasets" / "red_five_all.sqlite"
        )
        persistent_source = (
            root / ".local" / "recognition" / "tile_crop_dataset" / "dataset.sqlite"
        )
        source_database = compact_source if compact_source.is_file() else persistent_source
    experiment_database = (
        args.experiment_database.resolve()
        if args.experiment_database is not None
        else root
        / ".local"
        / "recognition"
        / "red_five_datasets"
        / "rgb64_binary_jp5000_seed42.sqlite"
    )
    run_root = (
        args.run_root.resolve()
        if args.run_root is not None
        else root
        / ".local"
        / "recognition"
        / "red_five_runs"
        / "c8_rgb_cr_ycr_seed42"
    )
    output_json = (
        args.output_json.resolve()
        if args.output_json is not None
        else run_root / "all_samples_evaluation.json"
    )
    errors_jsonl = (
        args.errors_jsonl.resolve()
        if args.errors_jsonl is not None
        else run_root / "all_samples_errors.jsonl"
    )

    for path in (source_database, experiment_database):
        if not path.is_file():
            raise FileNotFoundError(path)
    run_root.mkdir(parents=True, exist_ok=True)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    errors_jsonl.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda")
    torch.backends.cudnn.benchmark = True
    models = load_models(
        run_root,
        input_modes=tuple(args.input_modes),
        device=device,
    )
    membership = load_experiment_membership(experiment_database)
    source_kind = detect_source_kind(source_database)
    source_count = count_source_rows(source_database, source_kind=source_kind)
    print(
        f"[all-red-five-eval] source_kind={source_kind} source_samples={source_count} "
        f"experiment_membership={len(membership)} models={list(models)} "
        f"angles={[float(v) for v in args.angles]}"
    )

    metrics: dict[str, dict[str, MetricAccumulator]] = {
        mode: {angle_key(angle): MetricAccumulator() for angle in args.angles}
        for mode in models
    }
    error_counts: defaultdict[str, int] = defaultdict(int)
    started = time.perf_counter()
    completed = 0

    with errors_jsonl.open("w", encoding="utf-8", newline="\n") as error_output:
        with ThreadPoolExecutor(max_workers=int(args.workers)) as executor:
            for rows in source_batches(
                source_database,
                source_kind=source_kind,
                batch_size=int(args.batch_size),
            ):
                rgb_numpy = np.stack(
                    list(
                        executor.map(
                            decode_and_preprocess,
                            rows,
                            chunksize=max(1, len(rows) // max(1, int(args.workers) * 4)),
                        )
                    ),
                    axis=0,
                )
                rgb_u8 = torch.from_numpy(rgb_numpy).to(device=device, non_blocking=False)

                for mode, loaded in models.items():
                    base_input = rgb_u8_to_input_torch(rgb_u8, input_mode=mode)
                    for angle in args.angles:
                        key = angle_key(angle)
                        images = base_input
                        if abs(float(angle)) > 1.0e-9:
                            angle_tensor = torch.full(
                                (images.shape[0],),
                                float(angle),
                                device=device,
                                dtype=torch.float32,
                            )
                            images = rotate_batch(images, angle_tensor)
                        images = normalize_tensor(
                            images,
                            mean=loaded.mean,
                            std=loaded.std,
                        )
                        with torch.inference_mode(), torch.cuda.amp.autocast(
                            enabled=bool(args.amp)
                        ):
                            logits = loaded.model(images)
                            probabilities = torch.softmax(logits, dim=1)
                        predictions = logits.argmax(dim=1).cpu().numpy()
                        red_probabilities = probabilities[:, 1].float().cpu().numpy()

                        accumulator = metrics[mode][key]
                        for index, row in enumerate(rows):
                            member = classify_membership(row, membership)
                            prediction = int(predictions[index])
                            accumulator.add(
                                row=row,
                                membership=member,
                                prediction=prediction,
                            )
                            if prediction != row.is_red:
                                error_counts[f"{mode}/{key}"] += 1
                                error_output.write(
                                    json.dumps(
                                        {
                                            "input_mode": mode,
                                            "angle_deg": float(angle),
                                            "checkpoint_epoch": loaded.epoch,
                                            "crop_id": row.crop_id,
                                            "experiment_membership": member,
                                            "source": row.source,
                                            "source_partition": row.source_partition,
                                            "source_label": row.source_label,
                                            "suit": row.suit,
                                            "target_is_red": row.is_red,
                                            "prediction_is_red": prediction,
                                            "red_probability": float(red_probabilities[index]),
                                            "source_image_path": row.source_image_path,
                                            "source_image_id": row.source_image_id,
                                            "source_annotation_id": row.source_annotation_id,
                                            "capture_id": row.capture_id,
                                            "brightness": row.brightness,
                                            "shadow": row.shadow,
                                        },
                                        ensure_ascii=False,
                                        separators=(",", ":"),
                                    )
                                    + "\n"
                                )

                completed += len(rows)
                elapsed = time.perf_counter() - started
                print(
                    f"[all-red-five-eval] {completed}/{source_count} "
                    f"({completed / max(elapsed, 1e-9):.1f} samples/s source-crops)"
                )

    elapsed = time.perf_counter() - started
    payload = {
        "status": "completed",
        "source_database": str(source_database),
        "source_kind": source_kind,
        "experiment_database": str(experiment_database),
        "run_root": str(run_root),
        "source_sample_count": source_count,
        "experiment_membership_count": len(membership),
        "angles": [float(value) for value in args.angles],
        "batch_size": int(args.batch_size),
        "workers": int(args.workers),
        "amp": bool(args.amp),
        "elapsed_seconds": elapsed,
        "models": {
            mode: {
                "checkpoint": str(loaded.checkpoint_path),
                "checkpoint_epoch": loaded.epoch,
                "normalization": {
                    "mean": list(loaded.mean),
                    "std": list(loaded.std),
                },
                "angles": {
                    key: accumulator.to_json()
                    for key, accumulator in metrics[mode].items()
                },
            }
            for mode, loaded in models.items()
        },
        "error_counts": dict(sorted(error_counts.items())),
        "errors_jsonl": str(errors_jsonl),
        "output_json": str(output_json),
    }
    atomic_write_json(output_json, payload)
    print(json.dumps(compact_console_summary(payload), ensure_ascii=False, indent=2))


def load_models(
    run_root: Path,
    *,
    input_modes: tuple[str, ...],
    device: torch.device,
) -> dict[str, LoadedModel]:
    result: dict[str, LoadedModel] = {}
    for mode in input_modes:
        checkpoint_path = run_root / f"c8_{mode}_rot22p5_seed42" / "best.pt"
        if not checkpoint_path.is_file():
            raise FileNotFoundError(checkpoint_path)
        checkpoint = torch.load(checkpoint_path, map_location=device)
        config = checkpoint.get("config")
        if not isinstance(config, dict):
            raise ValueError(f"Checkpoint has no config dict: {checkpoint_path}")
        configured_mode = str(config.get("input_mode", ""))
        if configured_mode != mode:
            raise ValueError(
                f"Checkpoint input mode mismatch: expected {mode}, got {configured_mode}"
            )
        model_config = config.get("model", {})
        c8_fields = tuple(int(value) for value in model_config.get("c8_fields", (8, 16, 32, 64)))
        model = build_model(mode, c8_fields=c8_fields).to(device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        normalization = config.get("normalization", {})
        mean = tuple(float(value) for value in normalization["mean"])
        std = tuple(float(value) for value in normalization["std"])
        result[mode] = LoadedModel(
            input_mode=mode,
            checkpoint_path=checkpoint_path,
            model=model,
            mean=mean,
            std=std,
            epoch=int(checkpoint.get("epoch", 0)),
        )
        print(
            f"[all-red-five-eval] loaded {mode}: epoch={result[mode].epoch} "
            f"checkpoint={checkpoint_path}"
        )
    return result


def load_experiment_membership(database: Path) -> dict[str, tuple[str, str]]:
    result: dict[str, tuple[str, str]] = {}
    with sqlite3.connect(sqlite_readonly_uri(database), uri=True, timeout=60) as connection:
        connection.row_factory = sqlite3.Row
        for row in connection.execute(
            "SELECT crop_id, split, source FROM sample ORDER BY crop_id"
        ):
            crop_id = str(row["crop_id"])
            if crop_id in result:
                raise ValueError(f"Duplicate crop_id in experiment DB: {crop_id}")
            result[crop_id] = (str(row["split"]), str(row["source"]))
    return result


def classify_membership(
    row: SourceRow,
    membership: dict[str, tuple[str, str]],
) -> str:
    selected = membership.get(row.crop_id)
    if selected is not None:
        split, source = selected
        if split == "train":
            return f"train_{source}"
        return split
    if row.source == "jp" and row.source_partition == "train":
        return "jp_train_unselected"
    return "not_in_experiment_db"


def detect_source_kind(database: Path) -> str:
    with sqlite3.connect(sqlite_readonly_uri(database), uri=True, timeout=60) as connection:
        tables = {
            str(row[0])
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        }
        if "sample" in tables:
            columns = {
                str(row[1])
                for row in connection.execute("PRAGMA table_info(sample)")
            }
            if {"suit", "is_red", "source_label", "image_png"}.issubset(columns):
                return "red_five_all"
        if "tile_crop" in tables:
            columns = {
                str(row[1])
                for row in connection.execute("PRAGMA table_info(tile_crop)")
            }
            if {"tile_label", "image_png", "source", "source_partition"}.issubset(columns):
                return "tile_crop_dataset"
    raise ValueError(f"Unsupported red-five source database schema: {database}")


def count_source_rows(database: Path, *, source_kind: str) -> int:
    with sqlite3.connect(sqlite_readonly_uri(database), uri=True, timeout=60) as connection:
        if source_kind == "red_five_all":
            return int(connection.execute("SELECT COUNT(*) FROM sample").fetchone()[0])
        if source_kind == "tile_crop_dataset":
            placeholders = ",".join("?" for _ in SOURCE_LABELS)
            return int(
                connection.execute(
                    f"SELECT COUNT(*) FROM tile_crop WHERE tile_label IN ({placeholders})",
                    SOURCE_LABELS,
                ).fetchone()[0]
            )
    raise ValueError(f"Unsupported source_kind: {source_kind}")


def source_batches(
    database: Path,
    *,
    source_kind: str,
    batch_size: int,
) -> Iterable[list[SourceRow]]:
    connection = sqlite3.connect(sqlite_readonly_uri(database), uri=True, timeout=60)
    connection.row_factory = sqlite3.Row
    try:
        if source_kind == "red_five_all":
            cursor = connection.execute(
                """
                SELECT crop_id, source, source_partition, suit, is_red, source_label,
                       image_png, source_image_path, source_image_id, source_annotation_id,
                       capture_id, COALESCE(brightness, '') AS brightness,
                       COALESCE(shadow, '') AS shadow
                FROM sample
                ORDER BY rowid
                """
            )
            converter = source_row_from_red_five_all
        elif source_kind == "tile_crop_dataset":
            placeholders = ",".join("?" for _ in SOURCE_LABELS)
            cursor = connection.execute(
                f"""
                SELECT crop_id, source, source_partition, tile_label AS source_label,
                       image_png, source_image_path, source_image_id, source_annotation_id,
                       capture_id, COALESCE(brightness, '') AS brightness,
                       COALESCE(shadow, '') AS shadow
                FROM tile_crop
                WHERE tile_label IN ({placeholders})
                ORDER BY rowid
                """,
                SOURCE_LABELS,
            )
            converter = source_row_from_tile_crop
        else:
            raise ValueError(f"Unsupported source_kind: {source_kind}")

        while True:
            raw_rows = cursor.fetchmany(batch_size)
            if not raw_rows:
                break
            yield [converter(row) for row in raw_rows]
    finally:
        connection.close()


def source_row_from_red_five_all(row: sqlite3.Row) -> SourceRow:
    return SourceRow(
        crop_id=str(row["crop_id"]),
        source=str(row["source"]),
        source_partition=str(row["source_partition"]),
        suit=str(row["suit"]),
        is_red=int(row["is_red"]),
        source_label=str(row["source_label"]),
        image_png=bytes(row["image_png"]),
        source_image_path=str(row["source_image_path"]),
        source_image_id=(
            None if row["source_image_id"] is None else str(row["source_image_id"])
        ),
        source_annotation_id=str(row["source_annotation_id"]),
        capture_id=(None if row["capture_id"] is None else str(row["capture_id"])),
        brightness=str(row["brightness"]),
        shadow=str(row["shadow"]),
    )


def source_row_from_tile_crop(row: sqlite3.Row) -> SourceRow:
    label = str(row["source_label"])
    if label not in SOURCE_LABELS:
        raise ValueError(f"Unexpected tile label in filtered source: {label}")
    return SourceRow(
        crop_id=str(row["crop_id"]),
        source=str(row["source"]),
        source_partition=str(row["source_partition"]),
        suit=label[-1],
        is_red=1 if label.startswith("red5") else 0,
        source_label=label,
        image_png=bytes(row["image_png"]),
        source_image_path=str(row["source_image_path"]),
        source_image_id=(
            None if row["source_image_id"] is None else str(row["source_image_id"])
        ),
        source_annotation_id=str(row["source_annotation_id"]),
        capture_id=(None if row["capture_id"] is None else str(row["capture_id"])),
        brightness=str(row["brightness"]),
        shadow=str(row["shadow"]),
    )


def decode_and_preprocess(row: SourceRow) -> np.ndarray:
    raw = preprocess_rgb_u8(row.image_png, image_size=64)
    return np.frombuffer(raw, dtype=np.uint8).reshape(64, 64, 3).copy()


def counters_to_json(counters: dict[str, BinaryCounter]) -> dict[str, Any]:
    return {
        key: counters[key].to_json()
        for key in sorted(counters)
    }


def angle_key(angle: float) -> str:
    value = float(angle)
    return f"{int(value)}deg" if value.is_integer() else f"{value:g}deg"


def compact_console_summary(payload: dict[str, Any]) -> dict[str, Any]:
    models: dict[str, Any] = {}
    for mode, model_payload in payload["models"].items():
        models[mode] = {
            "checkpoint_epoch": model_payload["checkpoint_epoch"],
            "angles": {
                angle: {
                    "overall": metrics["overall"],
                    "jp_train_unselected": metrics["by_experiment_membership"].get(
                        "jp_train_unselected"
                    ),
                    "manual_all": metrics["by_source_partition"].get("manual/capture"),
                }
                for angle, metrics in model_payload["angles"].items()
            },
        }
    return {
        "status": payload["status"],
        "source_sample_count": payload["source_sample_count"],
        "elapsed_seconds": payload["elapsed_seconds"],
        "models": models,
        "error_counts": payload["error_counts"],
        "output_json": payload["output_json"],
        "errors_jsonl": payload["errors_jsonl"],
    }


def sqlite_readonly_uri(path: Path) -> str:
    return f"{path.resolve().as_uri()}?mode=ro"


def atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
        text=True,
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
