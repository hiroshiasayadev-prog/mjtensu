from __future__ import annotations

import argparse
import hashlib
import io
import json
import sqlite3
import time
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

try:
    from .build_tile_classifier_dataset import BASE_LABELS, preprocess_gray_u8, sqlite_readonly_uri
    from .detector_duplicate_groups import load_duplicate_plan
    from .tile_shape_classifier import DEFAULT_C8_FIELDS, build_model
except ImportError:  # direct script execution
    from build_tile_classifier_dataset import BASE_LABELS, preprocess_gray_u8, sqlite_readonly_uri
    from detector_duplicate_groups import load_duplicate_plan
    from tile_shape_classifier import DEFAULT_C8_FIELDS, build_model


INVALID_LABEL = "invalid"
EXPECTED_LABELS = tuple(BASE_LABELS) + (INVALID_LABEL,)
AUDIT_SCHEMA = """
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS audit_metadata (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS classifier_prediction (
    candidate_id        TEXT NOT NULL,
    model_key           TEXT NOT NULL,
    checkpoint          TEXT NOT NULL,
    checkpoint_sha256   TEXT NOT NULL,
    predicted_label     TEXT NOT NULL,
    confidence          REAL NOT NULL,
    invalid_probability REAL NOT NULL,
    predicted_at        TEXT NOT NULL,
    PRIMARY KEY(candidate_id, model_key)
);

CREATE INDEX IF NOT EXISTS idx_classifier_prediction_model_label
ON classifier_prediction(model_key, predicted_label, confidence DESC);

CREATE INDEX IF NOT EXISTS idx_classifier_prediction_model_invalid
ON classifier_prediction(model_key, invalid_probability DESC);
"""


def parse_args() -> argparse.Namespace:
    repository_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(
        description=(
            "Run a trained gray35 classifier over NanoDet crops kept by detector-side "
            "postprocessing and write predictions to a separate audit sidecar. Predictions are review hints; "
            "they are never written into the human review DB or used as training truth."
        )
    )
    parser.add_argument("--repository-root", type=Path, default=repository_root)
    parser.add_argument(
        "--database",
        type=Path,
        help="Defaults to .local/recognition/detector_crop_dataset/dataset.sqlite.",
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--output-database",
        type=Path,
        help="Defaults to .local/recognition/detector_crop_dataset/classifier_audit.sqlite.",
    )
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--progress-every", type=int, default=20_000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if int(args.batch_size) < 1:
        raise ValueError("--batch-size must be positive")
    if int(args.limit) < 0:
        raise ValueError("--limit must be non-negative")
    if int(args.progress_every) < 1:
        raise ValueError("--progress-every must be positive")

    repository_root = args.repository_root.resolve()
    default_audit_root = repository_root / ".local" / "recognition" / "detector_crop_dataset"
    database = (args.database or default_audit_root / "dataset.sqlite").resolve()
    audit_root = database.parent if args.database is not None else default_audit_root
    checkpoint = args.checkpoint.resolve()
    output_database = (args.output_database or audit_root / "classifier_audit.sqlite").resolve()
    for path in (database, checkpoint):
        if not path.is_file():
            raise FileNotFoundError(path)
    output_database.parent.mkdir(parents=True, exist_ok=True)

    device = resolve_device(str(args.device))
    # This is our own full training checkpoint, not an untrusted weights-only artifact.
    # PyTorch 2.6+ defaults torch.load(..., weights_only=True), which rejects metadata
    # such as TorchVersion stored in these checkpoints.
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    config = payload.get("config")
    if not isinstance(config, dict):
        raise ValueError("Checkpoint is missing config")
    class_labels = tuple(str(value) for value in config.get("class_labels", ()))
    if class_labels != EXPECTED_LABELS:
        raise ValueError(
            "Checkpoint must use canonical gray35 class order: 34 base tiles followed by invalid"
        )
    invalid_index = class_labels.index(INVALID_LABEL)

    model_config = config.get("model", {})
    model_name = str(model_config.get("name", "c8"))
    c8_fields = tuple(
        int(value)
        for value in (model_config.get("c8_fields") or list(DEFAULT_C8_FIELDS))
    )
    model = build_model(
        model_name,
        class_count=len(class_labels),
        c8_fields=c8_fields,
    )
    model.load_state_dict(payload["model_state_dict"])
    model.to(device).eval()

    image_size = int(config["image_size"])
    normalization = config["normalization"]
    mean = float(normalization["mean"])
    std = float(normalization["std"])
    use_amp = bool(args.amp) and device.type == "cuda"
    checkpoint_sha256 = sha256_file(checkpoint)
    model_key = f"gray35:{checkpoint_sha256[:16]}"
    predicted_at = datetime.now(timezone.utc).isoformat()

    source_detector_run_key = detector_run_key(database)

    connection = sqlite3.connect(output_database, timeout=60)
    connection.row_factory = sqlite3.Row
    try:
        connection.execute("PRAGMA journal_mode = WAL")
        connection.execute("PRAGMA synchronous = NORMAL")
        connection.executescript(AUDIT_SCHEMA)
        existing_run_key = get_metadata(connection, "source_detector_run_key")
        if existing_run_key is not None and existing_run_key != source_detector_run_key:
            raise ValueError(
                "Output classifier audit DB belongs to another detector run: "
                f"stored={existing_run_key}, requested={source_detector_run_key}"
            )
        set_metadata(connection, "source_detector_run_key", source_detector_run_key)
        set_metadata(connection, "latest_model_key", model_key)
        set_metadata(connection, f"model.{model_key}.checkpoint", str(checkpoint))
        set_metadata(connection, f"model.{model_key}.checkpoint_sha256", checkpoint_sha256)
        set_metadata(connection, f"model.{model_key}.source_database", str(database))
        set_metadata(connection, f"model.{model_key}.checkpoint_epoch", str(payload.get("epoch", "")))
        set_metadata(connection, f"model.{model_key}.predicted_at", predicted_at)
        connection.commit()

        started = time.perf_counter()
        result = run_inference(
            database,
            connection,
            model=model,
            model_key=model_key,
            checkpoint=checkpoint,
            checkpoint_sha256=checkpoint_sha256,
            predicted_at=predicted_at,
            class_labels=class_labels,
            invalid_index=invalid_index,
            image_size=image_size,
            mean=mean,
            std=std,
            device=device,
            batch_size=int(args.batch_size),
            amp=use_amp,
            limit=int(args.limit),
            progress_every=int(args.progress_every),
        )
        set_metadata(connection, f"model.{model_key}.candidate_count", str(result["candidate_count"]))
        set_metadata(connection, f"model.{model_key}.elapsed_seconds", repr(time.perf_counter() - started))
        connection.commit()
        connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    finally:
        connection.close()

    summary = {
        "status": "completed",
        "database": str(database),
        "output_database": str(output_database),
        "checkpoint": str(checkpoint),
        "checkpoint_epoch": payload.get("epoch"),
        "model_key": model_key,
        "source_detector_run_key": source_detector_run_key,
        "device": str(device),
        "amp": use_amp,
        **result,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def run_inference(
    source_database: Path,
    output: sqlite3.Connection,
    *,
    model: torch.nn.Module,
    model_key: str,
    checkpoint: Path,
    checkpoint_sha256: str,
    predicted_at: str,
    class_labels: Sequence[str],
    invalid_index: int,
    image_size: int,
    mean: float,
    std: float,
    device: torch.device,
    batch_size: int,
    amp: bool,
    limit: int,
    progress_every: int,
) -> dict[str, Any]:
    duplicate_plan = load_duplicate_plan(source_database)
    winner_ids = duplicate_plan.winner_candidate_ids
    total_available = len(winner_ids)
    requested = total_available if not limit else min(total_available, limit)
    processed = 0
    label_counts = {label: 0 for label in class_labels}
    invalid_prob_sum = 0.0
    started = time.perf_counter()

    insert_sql = """
        INSERT INTO classifier_prediction(
            candidate_id, model_key, checkpoint, checkpoint_sha256,
            predicted_label, confidence, invalid_probability, predicted_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(candidate_id, model_key) DO UPDATE SET
            checkpoint = excluded.checkpoint,
            checkpoint_sha256 = excluded.checkpoint_sha256,
            predicted_label = excluded.predicted_label,
            confidence = excluded.confidence,
            invalid_probability = excluded.invalid_probability,
            predicted_at = excluded.predicted_at
    """

    with closing(
        sqlite3.connect(sqlite_readonly_uri(source_database), uri=True, timeout=60)
    ) as source:
        source.row_factory = sqlite3.Row
        all_rows = source.execute(
            "SELECT candidate_id, image_png FROM candidate ORDER BY candidate_id"
        ).fetchall()
        rows_to_process = [
            row for row in all_rows if str(row["candidate_id"]) in winner_ids
        ]
        if limit:
            rows_to_process = rows_to_process[:limit]
        for batch_start in range(0, len(rows_to_process), batch_size):
            rows = rows_to_process[batch_start : batch_start + batch_size]
            images = np.stack(
                [
                    np.frombuffer(
                        preprocess_gray_u8(bytes(row["image_png"]), image_size=image_size),
                        dtype=np.uint8,
                    )
                    .reshape(image_size, image_size)
                    .copy()
                    for row in rows
                ],
                axis=0,
            )
            tensor = torch.from_numpy(images).to(device, non_blocking=True)
            tensor = tensor.float().unsqueeze(1).mul_(1.0 / 255.0)
            tensor = tensor.sub(mean).div(std)
            with torch.inference_mode():
                with torch.cuda.amp.autocast(enabled=amp):
                    logits = model(tensor)
                probabilities = F.softmax(logits.float(), dim=1)
                confidence, prediction = probabilities.max(dim=1)
                invalid_probability = probabilities[:, invalid_index]
            prediction_np = prediction.cpu().numpy()
            confidence_np = confidence.cpu().numpy()
            invalid_np = invalid_probability.cpu().numpy()

            values: list[tuple[Any, ...]] = []
            for index, row in enumerate(rows):
                label = str(class_labels[int(prediction_np[index])])
                confidence_value = float(confidence_np[index])
                invalid_value = float(invalid_np[index])
                values.append(
                    (
                        str(row["candidate_id"]),
                        model_key,
                        str(checkpoint),
                        checkpoint_sha256,
                        label,
                        confidence_value,
                        invalid_value,
                        predicted_at,
                    )
                )
                label_counts[label] += 1
                invalid_prob_sum += invalid_value
            output.executemany(insert_sql, values)
            output.commit()
            processed += len(rows)
            if processed % progress_every < len(rows) or processed == requested:
                elapsed = time.perf_counter() - started
                rate = processed / max(elapsed, 1.0e-9)
                print(
                    f"[gray35-audit] {processed:,}/{requested:,} "
                    f"({rate:,.1f} crops/s)"
                )

    return {
        "candidate_count": processed,
        "available_candidate_count": total_available,
        "prediction_counts_by_label": {key: value for key, value in label_counts.items() if value},
        "mean_invalid_probability": invalid_prob_sum / max(1, processed),
        "elapsed_seconds": time.perf_counter() - started,
    }


def candidate_count(database: Path) -> int:
    return len(load_duplicate_plan(database).winner_candidate_ids)


def resolve_device(requested: str) -> torch.device:
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def detector_run_key(database: Path) -> str:
    with closing(
        sqlite3.connect(sqlite_readonly_uri(database), uri=True, timeout=30)
    ) as connection:
        row = connection.execute(
            "SELECT value FROM dataset_metadata WHERE key='detector_run_key'"
        ).fetchone()
    if row is None:
        raise ValueError(f"Candidate DB has no detector_run_key metadata: {database}")
    return str(row[0])


def get_metadata(connection: sqlite3.Connection, key: str) -> str | None:
    row = connection.execute(
        "SELECT value FROM audit_metadata WHERE key = ?", (key,)
    ).fetchone()
    return None if row is None else str(row[0])


def set_metadata(connection: sqlite3.Connection, key: str, value: str) -> None:
    connection.execute(
        """
        INSERT INTO audit_metadata(key, value) VALUES (?, ?)
        ON CONFLICT(key) DO UPDATE SET value = excluded.value
        """,
        (key, value),
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    main()
