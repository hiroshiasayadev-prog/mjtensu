from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
import sqlite3
import tempfile
from collections import Counter, defaultdict
from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, wait
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Iterable, Iterator, Sequence, TypeVar

from PIL import Image, ImageOps


BUILDER_VERSION = "1"
JP_SPLITS = ("train", "valid", "test")
MANUAL_REGION_COLUMNS = {
    "completed_hand": "hand_crop_path",
    "dora_indicators": "dora_crop_path",
    "melds": "meld_crop_path",
}

# The bulk portion of Mahjong-jp v2 uses numeric class names. Classes 0-29
# are suit-major, with the red five inserted immediately after the normal five.
# Classes 30-36 were verified from representative crops on 2026-08-06 as
# east, south, west, north, white, green, and red respectively.
JP_NUMERIC_TILE_LABELS = (
    "1m",
    "2m",
    "3m",
    "4m",
    "5m",
    "red5m",
    "6m",
    "7m",
    "8m",
    "9m",
    "1p",
    "2p",
    "3p",
    "4p",
    "5p",
    "red5p",
    "6p",
    "7p",
    "8p",
    "9p",
    "1s",
    "2s",
    "3s",
    "4s",
    "5s",
    "red5s",
    "6s",
    "7s",
    "8s",
    "9s",
    "east",
    "south",
    "west",
    "north",
    "white",
    "green",
    "red",
)
EXPLICIT_JP_TILE_LABELS = frozenset(
    [f"{number}{suit}" for suit in "mps" for number in range(1, 10)]
    + ["east", "south", "west", "north", "white", "green", "red"]
    + ["5mr", "5pr", "5sr", "red5m", "red5p", "red5s"]
)
JP_LABEL_ALIASES = {
    "5mr": "red5m",
    "5pr": "red5p",
    "5sr": "red5s",
}

SCHEMA = """
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS dataset_metadata (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS tile_crop (
    crop_id                  TEXT PRIMARY KEY,
    source                   TEXT NOT NULL CHECK (source IN ('jp', 'manual')),
    source_partition         TEXT NOT NULL,
    tile_label               TEXT NOT NULL,
    raw_category_name        TEXT NOT NULL,
    raw_category_id          INTEGER,
    image_format             TEXT NOT NULL,
    image_width              INTEGER NOT NULL,
    image_height             INTEGER NOT NULL,
    image_png                BLOB NOT NULL,
    source_image_path        TEXT NOT NULL,
    source_image_id          TEXT,
    source_annotation_id     TEXT NOT NULL,
    bbox_json                TEXT NOT NULL,
    capture_id               TEXT,
    layout_id                TEXT,
    layout_ordinal           INTEGER,
    region                   TEXT,
    group_name               TEXT,
    group_ordinal            INTEGER,
    tile_ordinal             INTEGER,
    brightness               TEXT,
    shadow                   TEXT,
    annotation_angle_deg     REAL NOT NULL DEFAULT 0,
    expected_rotation_deg    INTEGER NOT NULL DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_tile_crop_source_partition
ON tile_crop(source, source_partition);

CREATE INDEX IF NOT EXISTS idx_tile_crop_source_label
ON tile_crop(source, tile_label);

CREATE INDEX IF NOT EXISTS idx_tile_crop_manual_layout
ON tile_crop(source, layout_id, brightness, shadow);
"""

INSERT_SQL = """
INSERT INTO tile_crop(
    crop_id,
    source,
    source_partition,
    tile_label,
    raw_category_name,
    raw_category_id,
    image_format,
    image_width,
    image_height,
    image_png,
    source_image_path,
    source_image_id,
    source_annotation_id,
    bbox_json,
    capture_id,
    layout_id,
    layout_ordinal,
    region,
    group_name,
    group_ordinal,
    tile_ordinal,
    brightness,
    shadow,
    annotation_angle_deg,
    expected_rotation_deg
) VALUES (
    :crop_id,
    :source,
    :source_partition,
    :tile_label,
    :raw_category_name,
    :raw_category_id,
    'png',
    :image_width,
    :image_height,
    :image_png,
    :source_image_path,
    :source_image_id,
    :source_annotation_id,
    :bbox_json,
    :capture_id,
    :layout_id,
    :layout_ordinal,
    :region,
    :group_name,
    :group_ordinal,
    :tile_ordinal,
    :brightness,
    :shadow,
    :annotation_angle_deg,
    :expected_rotation_deg
)
"""


@dataclass(frozen=True)
class ExpectedManualSlot:
    group_name: str
    group_ordinal: int
    tile_ordinal: int
    tile_label: str
    expected_rotation_deg: int


@dataclass(frozen=True)
class AssignedManualBox:
    box: dict[str, Any]
    slot: ExpectedManualSlot


@dataclass(frozen=True)
class JpAnnotationJob:
    annotation_id: str
    raw_category_name: str
    raw_category_id: int
    tile_label: str
    bbox: tuple[float, float, float, float]


@dataclass(frozen=True)
class JpImageJob:
    split: str
    image_path: str
    source_image_path: str
    image_id: str
    expected_width: int
    expected_height: int
    annotations: tuple[JpAnnotationJob, ...]


@dataclass(frozen=True)
class ManualRegionJob:
    region: str
    image_path: str
    source_image_path: str
    assignments: tuple[AssignedManualBox, ...]


@dataclass(frozen=True)
class ManualCaptureJob:
    capture_id: str
    layout_id: str
    layout_ordinal: int
    brightness: str
    shadow: str
    regions: tuple[ManualRegionJob, ...]


@dataclass(frozen=True)
class ProcessedImage:
    source_image_id: str
    rows: tuple[dict[str, Any], ...]


JobT = TypeVar("JobT")
ResultT = TypeVar("ResultT")


def parse_args() -> argparse.Namespace:
    repository_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(
        description=(
            "Build a persistent, lossless tile-crop SQLite dataset from Mahjong-jp v2 "
            "and completed manual capture annotations. Unchanged sources are skipped on "
            "subsequent runs, so color-threshold trials only read the generated database."
        )
    )
    parser.add_argument("--repository-root", type=Path, default=repository_root)
    parser.add_argument(
        "--jp-root",
        type=Path,
        help="Defaults to <repository-root>/data/coco_mahjong_jp_v2.",
    )
    parser.add_argument(
        "--manual-storage-root",
        type=Path,
        help="Defaults to <repository-root>/.local/recognition/capture_dataset.",
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        help="Defaults to <repository-root>/.local/recognition/tile_crop_dataset.",
    )
    parser.add_argument(
        "--sources",
        nargs="+",
        choices=("jp", "manual"),
        default=("jp", "manual"),
    )
    parser.add_argument(
        "--jp-splits",
        nargs="+",
        choices=JP_SPLITS,
        default=JP_SPLITS,
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rebuild selected sources even when their input fingerprint is unchanged.",
    )
    parser.add_argument(
        "--commit-interval",
        type=int,
        default=5000,
        help="Commit after this many generated crops. Defaults to 5000.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=min(8, os.cpu_count() or 1),
        help=(
            "Worker processes used for image decode, crop, and PNG encoding. "
            "SQLite writes remain in the parent process. Defaults to min(8, CPU count)."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.commit_interval < 1:
        raise ValueError("commit-interval must be positive")
    if args.workers < 1:
        raise ValueError("workers must be positive")

    repository_root = args.repository_root.resolve()
    jp_root = (
        args.jp_root.resolve()
        if args.jp_root is not None
        else repository_root / "data" / "coco_mahjong_jp_v2"
    )
    manual_storage_root = (
        args.manual_storage_root.resolve()
        if args.manual_storage_root is not None
        else repository_root / ".local" / "recognition" / "capture_dataset"
    )
    output_directory = (
        args.output_directory.resolve()
        if args.output_directory is not None
        else repository_root / ".local" / "recognition" / "tile_crop_dataset"
    )
    output_directory.mkdir(parents=True, exist_ok=True)
    output_database = output_directory / "dataset.sqlite"

    connection = sqlite3.connect(output_database, timeout=60)
    connection.row_factory = sqlite3.Row
    try:
        configure_output_database(connection)
        connection.executescript(SCHEMA)
        set_metadata(connection, "schema_version", "1")
        set_metadata(connection, "builder_version", BUILDER_VERSION)
        set_metadata(
            connection,
            "jp_numeric_tile_labels",
            json.dumps(JP_NUMERIC_TILE_LABELS, ensure_ascii=False),
        )
        set_metadata(
            connection,
            "jp_numeric_honor_mapping_status",
            "resolved_by_contact_sheet_2026-08-06",
        )
        connection.commit()

        build_events: list[dict[str, Any]] = []
        if "jp" in args.sources:
            for split in args.jp_splits:
                build_events.append(
                    build_jp_split(
                        connection,
                        repository_root=repository_root,
                        jp_root=jp_root,
                        split=split,
                        force=bool(args.force),
                        commit_interval=int(args.commit_interval),
                        workers=int(args.workers),
                    )
                )

        if "manual" in args.sources:
            build_events.append(
                build_manual_source(
                    connection,
                    repository_root=repository_root,
                    storage_root=manual_storage_root,
                    force=bool(args.force),
                    commit_interval=int(args.commit_interval),
                    workers=int(args.workers),
                )
            )

        connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        summary = build_summary(connection, output_database, build_events)
        summary_path = output_directory / "summary.json"
        atomic_write_json(summary_path, summary)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    finally:
        connection.close()


def configure_output_database(connection: sqlite3.Connection) -> None:
    connection.execute("PRAGMA journal_mode = WAL")
    connection.execute("PRAGMA synchronous = NORMAL")
    connection.execute("PRAGMA temp_store = MEMORY")
    connection.execute("PRAGMA cache_size = -262144")


def build_jp_split(
    connection: sqlite3.Connection,
    *,
    repository_root: Path,
    jp_root: Path,
    split: str,
    force: bool,
    commit_interval: int,
    workers: int = 1,
) -> dict[str, Any]:
    annotation_path = jp_root / split / "_annotations.coco.json"
    image_directory = jp_root / split
    if not annotation_path.is_file():
        raise FileNotFoundError(annotation_path)

    metadata_prefix = f"source.jp.{split}"
    fingerprint = files_fingerprint([annotation_path])
    if not force and source_is_current(connection, metadata_prefix, fingerprint):
        count = source_partition_count(connection, "jp", split)
        print(f"[jp/{split}] unchanged; reusing {count} crops")
        return {
            "source": "jp",
            "partition": split,
            "action": "reused",
            "crop_count": count,
        }

    can_resume = not force and source_can_resume(
        connection,
        metadata_prefix,
        fingerprint,
    )
    if can_resume:
        existing_annotation_ids = source_annotation_ids(connection, "jp", split)
        action = "resumed"
        print(
            f"[jp/{split}] resuming from {len(existing_annotation_ids)} existing crops "
            f"with {workers} workers"
        )
    else:
        existing_annotation_ids = set()
        action = "rebuilt"
        connection.execute(
            "DELETE FROM tile_crop WHERE source = 'jp' AND source_partition = ?",
            (split,),
        )
        connection.commit()
        print(f"[jp/{split}] building with {workers} workers")

    mark_source_building(connection, metadata_prefix, fingerprint)

    payload = load_json(annotation_path)
    categories = category_map(payload, annotation_path)
    images = payload.get("images")
    annotations = payload.get("annotations")
    if not isinstance(images, list) or not isinstance(annotations, list):
        raise ValueError(f"Invalid COCO images or annotations: {annotation_path}")

    annotations_by_image: defaultdict[int, list[dict[str, Any]]] = defaultdict(list)
    for annotation in annotations:
        if not isinstance(annotation, dict):
            raise ValueError(f"Non-object annotation in {annotation_path}")
        annotations_by_image[int(annotation["image_id"])].append(annotation)

    jobs = iter_jp_image_jobs(
        images=images,
        annotations_by_image=annotations_by_image,
        categories=categories,
        annotation_path=annotation_path,
        image_directory=image_directory,
        repository_root=repository_root,
        split=split,
        existing_annotation_ids=existing_annotation_ids,
    )

    new_generated = 0
    processed_images = 0
    batch: list[dict[str, Any]] = []
    for result in bounded_process_map(process_jp_image_job, jobs, workers):
        processed_images += 1
        new_generated += len(result.rows)
        batch.extend(result.rows)
        if len(batch) >= commit_interval:
            insert_crop_batch(connection, batch)
            batch.clear()
            total = len(existing_annotation_ids) + new_generated
            print(
                f"[jp/{split}] generated {total} crops "
                f"({processed_images} pending source images processed)"
            )

    if batch:
        insert_crop_batch(connection, batch)

    total_count = source_partition_count(connection, "jp", split)
    label_counts = source_label_counts(connection, "jp", split)
    mark_source_complete(
        connection,
        metadata_prefix,
        fingerprint,
        total_count,
        label_counts,
    )
    connection.commit()
    return {
        "source": "jp",
        "partition": split,
        "action": action,
        "workers": workers,
        "new_crop_count": new_generated,
        "crop_count": total_count,
        "label_counts": dict(sorted(label_counts.items())),
    }


def iter_jp_image_jobs(
    *,
    images: Sequence[Any],
    annotations_by_image: dict[int, list[dict[str, Any]]],
    categories: dict[int, str],
    annotation_path: Path,
    image_directory: Path,
    repository_root: Path,
    split: str,
    existing_annotation_ids: set[str],
) -> Iterator[JpImageJob]:
    for image_record in images:
        if not isinstance(image_record, dict):
            raise ValueError(f"Non-object image record in {annotation_path}")
        image_id = int(image_record["id"])
        file_name = safe_relative_path(str(image_record["file_name"]))
        image_path = image_directory / file_name
        if not image_path.is_file():
            raise FileNotFoundError(image_path)

        pending_annotations: list[JpAnnotationJob] = []
        for annotation in annotations_by_image.get(image_id, []):
            annotation_id = str(annotation["id"])
            if annotation_id in existing_annotation_ids:
                continue
            category_id = int(annotation["category_id"])
            raw_category_name = categories.get(category_id)
            if raw_category_name is None:
                raise ValueError(
                    f"Annotation {annotation_id} references unknown category {category_id}"
                )
            tile_label = normalize_jp_tile_label(raw_category_name)
            if tile_label is None:
                continue
            bbox = validated_coco_bbox(annotation)
            pending_annotations.append(
                JpAnnotationJob(
                    annotation_id=annotation_id,
                    raw_category_name=raw_category_name,
                    raw_category_id=category_id,
                    tile_label=tile_label,
                    bbox=tuple(bbox),
                )
            )

        if pending_annotations:
            yield JpImageJob(
                split=split,
                image_path=str(image_path),
                source_image_path=repository_relative_path(image_path, repository_root),
                image_id=str(image_id),
                expected_width=int(image_record["width"]),
                expected_height=int(image_record["height"]),
                annotations=tuple(pending_annotations),
            )


def process_jp_image_job(job: JpImageJob) -> ProcessedImage:
    image_path = Path(job.image_path)
    loaded = load_rgb_image(image_path)
    expected_size = (job.expected_width, job.expected_height)
    if loaded.size != expected_size:
        raise ValueError(
            f"COCO image size mismatch for {image_path}: "
            f"annotation={expected_size}, loaded={loaded.size}"
        )

    rows: list[dict[str, Any]] = []
    for annotation in job.annotations:
        crop = crop_axis_aligned_bbox(loaded, annotation.bbox)
        rows.append(
            crop_row(
                crop_id=f"jp:{job.split}:{annotation.annotation_id}",
                source="jp",
                source_partition=job.split,
                tile_label=annotation.tile_label,
                raw_category_name=annotation.raw_category_name,
                raw_category_id=annotation.raw_category_id,
                crop=crop,
                source_image_path=job.source_image_path,
                source_image_id=job.image_id,
                source_annotation_id=annotation.annotation_id,
                bbox={"kind": "coco_xywh", "value": annotation.bbox},
            )
        )
    return ProcessedImage(source_image_id=job.image_id, rows=tuple(rows))


def build_manual_source(
    connection: sqlite3.Connection,
    *,
    repository_root: Path,
    storage_root: Path,
    force: bool,
    commit_interval: int,
    workers: int = 1,
) -> dict[str, Any]:
    source_database = storage_root / "dataset.sqlite"
    if not source_database.is_file():
        raise FileNotFoundError(source_database)

    source_connection = sqlite3.connect(source_database, timeout=60)
    source_connection.row_factory = sqlite3.Row
    try:
        rows = source_connection.execute(
            """
            SELECT
                capture.id AS capture_id,
                capture.original_path,
                capture.hand_crop_path,
                capture.dora_crop_path,
                capture.meld_crop_path,
                capture_task.layout_id,
                capture_task.layout_ordinal,
                capture_task.brightness,
                capture_task.shadow,
                capture_task.task_json,
                capture_annotation.annotation_json
            FROM capture
            JOIN capture_task ON capture_task.id = capture.task_id
            JOIN capture_annotation ON capture_annotation.capture_id = capture.id
            WHERE capture_annotation.status = 'complete'
            ORDER BY capture_task.task_order
            """
        ).fetchall()
    finally:
        source_connection.close()

    fingerprint = manual_rows_fingerprint(rows, storage_root)
    metadata_prefix = "source.manual"
    if not force and source_is_current(connection, metadata_prefix, fingerprint):
        count = source_partition_count(connection, "manual", "capture")
        print(f"[manual] unchanged; reusing {count} crops")
        return {
            "source": "manual",
            "partition": "capture",
            "action": "reused",
            "crop_count": count,
        }

    can_resume = not force and source_can_resume(
        connection,
        metadata_prefix,
        fingerprint,
    )
    if can_resume:
        existing_crop_ids = source_crop_ids(connection, "manual", "capture")
        action = "resumed"
        print(
            f"[manual] resuming from {len(existing_crop_ids)} existing crops "
            f"with {workers} workers"
        )
    else:
        existing_crop_ids = set()
        action = "rebuilt"
        connection.execute("DELETE FROM tile_crop WHERE source = 'manual'")
        connection.commit()
        print(f"[manual] building with {workers} workers")

    mark_source_building(connection, metadata_prefix, fingerprint)
    jobs = iter_manual_capture_jobs(
        rows=rows,
        storage_root=storage_root,
        repository_root=repository_root,
        existing_crop_ids=existing_crop_ids,
    )

    new_generated = 0
    processed_captures = 0
    batch: list[dict[str, Any]] = []
    for result in bounded_process_map(process_manual_capture_job, jobs, workers):
        processed_captures += 1
        new_generated += len(result.rows)
        batch.extend(result.rows)
        if len(batch) >= commit_interval:
            insert_crop_batch(connection, batch)
            batch.clear()
            total = len(existing_crop_ids) + new_generated
            print(
                f"[manual] generated {total} crops "
                f"({processed_captures} pending captures processed)"
            )

    if batch:
        insert_crop_batch(connection, batch)

    total_count = source_partition_count(connection, "manual", "capture")
    label_counts = source_label_counts(connection, "manual", "capture")
    mark_source_complete(
        connection,
        metadata_prefix,
        fingerprint,
        total_count,
        label_counts,
    )
    connection.commit()
    return {
        "source": "manual",
        "partition": "capture",
        "action": action,
        "workers": workers,
        "capture_count": len(rows),
        "new_crop_count": new_generated,
        "crop_count": total_count,
        "label_counts": dict(sorted(label_counts.items())),
    }


def iter_manual_capture_jobs(
    *,
    rows: Sequence[sqlite3.Row],
    storage_root: Path,
    repository_root: Path,
    existing_crop_ids: set[str],
) -> Iterator[ManualCaptureJob]:
    for row in rows:
        capture_id = str(row["capture_id"])
        task = json.loads(row["task_json"])
        annotation_document = json.loads(row["annotation_json"])
        boxes_by_region = annotation_document.get("boxes")
        if not isinstance(boxes_by_region, dict):
            raise ValueError(f"Capture {capture_id} annotation has no boxes object")

        region_jobs: list[ManualRegionJob] = []
        for region, path_column in MANUAL_REGION_COLUMNS.items():
            region_boxes = boxes_by_region.get(region)
            if not isinstance(region_boxes, list):
                raise ValueError(f"Capture {capture_id} boxes.{region} is not an array")
            assigned = assign_manual_boxes(task, region, region_boxes)
            pending_assignments = tuple(
                assignment
                for assignment in assigned
                if (
                    f"manual:{capture_id}:{region}:{assignment.box['id']}"
                    not in existing_crop_ids
                )
            )
            if not pending_assignments:
                continue

            path_value = row[path_column]
            if path_value is None:
                raise ValueError(
                    f"Capture {capture_id} has annotated {region} boxes but no region image"
                )
            region_path = storage_root / safe_relative_path(str(path_value))
            if not region_path.is_file():
                raise FileNotFoundError(region_path)
            region_jobs.append(
                ManualRegionJob(
                    region=region,
                    image_path=str(region_path),
                    source_image_path=repository_relative_path(
                        region_path,
                        repository_root,
                    ),
                    assignments=pending_assignments,
                )
            )

        if region_jobs:
            yield ManualCaptureJob(
                capture_id=capture_id,
                layout_id=str(row["layout_id"]),
                layout_ordinal=int(row["layout_ordinal"]),
                brightness=str(row["brightness"]),
                shadow=str(row["shadow"]),
                regions=tuple(region_jobs),
            )


def process_manual_capture_job(job: ManualCaptureJob) -> ProcessedImage:
    rows: list[dict[str, Any]] = []
    for region_job in job.regions:
        region_image = load_rgb_image(Path(region_job.image_path))
        for assignment in region_job.assignments:
            box = assignment.box
            slot = assignment.slot
            box_id = str(box["id"])
            crop = extract_rotated_crop(region_image, box)
            rows.append(
                crop_row(
                    crop_id=f"manual:{job.capture_id}:{region_job.region}:{box_id}",
                    source="manual",
                    source_partition="capture",
                    tile_label=slot.tile_label,
                    raw_category_name=slot.tile_label,
                    raw_category_id=None,
                    crop=crop,
                    source_image_path=region_job.source_image_path,
                    source_image_id=job.capture_id,
                    source_annotation_id=box_id,
                    bbox={
                        "kind": "rotated_center_wh",
                        "value": {
                            "centerX": float(box["centerX"]),
                            "centerY": float(box["centerY"]),
                            "width": float(box["width"]),
                            "height": float(box["height"]),
                            "angleDeg": float(box["angleDeg"]),
                        },
                    },
                    capture_id=job.capture_id,
                    layout_id=job.layout_id,
                    layout_ordinal=job.layout_ordinal,
                    region=region_job.region,
                    group_name=slot.group_name,
                    group_ordinal=slot.group_ordinal,
                    tile_ordinal=slot.tile_ordinal,
                    brightness=job.brightness,
                    shadow=job.shadow,
                    annotation_angle_deg=float(box["angleDeg"]),
                    expected_rotation_deg=slot.expected_rotation_deg,
                )
            )
    return ProcessedImage(source_image_id=job.capture_id, rows=tuple(rows))


def bounded_process_map(
    worker: Callable[[JobT], ResultT],
    jobs: Iterable[JobT],
    workers: int,
) -> Iterator[ResultT]:
    if workers < 1:
        raise ValueError("workers must be positive")
    if workers == 1:
        for job in jobs:
            yield worker(job)
        return

    job_iterator = iter(jobs)
    max_in_flight = max(workers * 2, workers + 1)
    with ProcessPoolExecutor(max_workers=workers) as executor:
        pending: set[Future[ResultT]] = set()
        for _ in range(max_in_flight):
            try:
                job = next(job_iterator)
            except StopIteration:
                break
            pending.add(executor.submit(worker, job))

        while pending:
            completed, still_pending = wait(
                pending,
                return_when=FIRST_COMPLETED,
            )
            pending = set(still_pending)
            for future in completed:
                result = future.result()
                try:
                    next_job = next(job_iterator)
                except StopIteration:
                    pass
                else:
                    pending.add(executor.submit(worker, next_job))
                yield result


def normalize_jp_tile_label(raw_category_name: str) -> str | None:
    if raw_category_name == "mahjong-tiles":
        return None
    if raw_category_name.isdecimal():
        index = int(raw_category_name)
        if not 0 <= index < len(JP_NUMERIC_TILE_LABELS):
            raise ValueError(f"Unsupported numeric Mahjong-jp class: {raw_category_name}")
        return JP_NUMERIC_TILE_LABELS[index]
    if raw_category_name not in EXPLICIT_JP_TILE_LABELS:
        raise ValueError(f"Unsupported Mahjong-jp category: {raw_category_name}")
    return JP_LABEL_ALIASES.get(raw_category_name, raw_category_name)


def assign_manual_boxes(
    task: dict[str, Any],
    region: str,
    boxes: Sequence[dict[str, Any]],
) -> list[AssignedManualBox]:
    groups = expected_manual_groups(task, region)
    expected_count = sum(len(group_slots) for _, _, group_slots in groups)
    if len(boxes) != expected_count:
        raise ValueError(
            f"Manual annotation count mismatch for {task.get('id')}/{region}: "
            f"expected {expected_count}, found {len(boxes)}"
        )

    if str(task.get("campaignId", "")).startswith("tile-catalog") and region == "melds":
        ordered = order_catalog_boxes(boxes)
    else:
        ordered = sorted(
            boxes,
            key=lambda box: (float(box["centerY"]), float(box["centerX"])),
        )
    assigned: list[AssignedManualBox] = []
    cursor = 0
    for group_name, group_ordinal, slots in groups:
        group_boxes = sorted(
            ordered[cursor : cursor + len(slots)],
            key=lambda box: float(box["centerX"]),
        )
        cursor += len(slots)
        for box, slot in zip(group_boxes, slots):
            assigned.append(
                AssignedManualBox(
                    box=box,
                    slot=ExpectedManualSlot(
                        group_name=group_name,
                        group_ordinal=group_ordinal,
                        tile_ordinal=int(slot["ordinal"]),
                        tile_label=str(slot["tile"]),
                        expected_rotation_deg=int(slot["rotation"]),
                    ),
                )
            )
    return assigned


def order_catalog_boxes(boxes: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    slope = estimate_catalog_row_slope(boxes)
    return sorted(
        boxes,
        key=lambda box: (
            float(box["centerY"]) - slope * float(box["centerX"]),
            float(box["centerX"]),
        ),
    )


def estimate_catalog_row_slope(boxes: Sequence[dict[str, Any]]) -> float:
    if len(boxes) < 2:
        return 0.0
    widths = sorted(float(box["width"]) for box in boxes)
    heights = sorted(float(box["height"]) for box in boxes)
    median_width = widths[len(widths) // 2]
    median_height = heights[len(heights) // 2]
    slopes: list[float] = []

    for left_index, left in enumerate(boxes):
        for right in boxes[left_index + 1 :]:
            delta_x = float(right["centerX"]) - float(left["centerX"])
            delta_y = float(right["centerY"]) - float(left["centerY"])
            absolute_delta_x = abs(delta_x)
            if (
                absolute_delta_x < median_width * 0.45
                or absolute_delta_x > median_width * 2.4
                or abs(delta_y) > median_height * 0.7
            ):
                continue
            slopes.append(delta_y / delta_x)

    if not slopes:
        return 0.0
    slopes.sort()
    return slopes[len(slopes) // 2]


def expected_manual_groups(
    task: dict[str, Any],
    region: str,
) -> list[tuple[str, int, list[dict[str, Any]]]]:
    if region == "completed_hand":
        slots = front_slots(task["hand"])
        return [] if not slots else [("hand", 0, slots)]
    if region == "dora_indicators":
        groups: list[tuple[str, int, list[dict[str, Any]]]] = []
        visible = front_slots(task["dora"]["visible"])
        ura = front_slots(task["dora"]["ura"])
        if visible:
            groups.append(("dora-visible", 0, visible))
        if ura:
            groups.append(("dora-ura", 1, ura))
        return groups
    if region == "melds":
        groups = []
        for meld in task["melds"]:
            slots = front_slots(meld["tiles"])
            if slots:
                groups.append((str(meld["kind"]), int(meld["ordinal"]), slots))
        return groups
    raise ValueError(f"Unknown manual annotation region: {region}")


def front_slots(slots: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    return [slot for slot in slots if str(slot["face"]) == "front"]


def crop_axis_aligned_bbox(image: Image.Image, bbox: Sequence[float]) -> Image.Image:
    x, y, width, height = (float(value) for value in bbox)
    left = max(0, math.floor(x))
    top = max(0, math.floor(y))
    right = min(image.width, math.ceil(x + width))
    bottom = min(image.height, math.ceil(y + height))
    if right <= left or bottom <= top:
        raise ValueError(f"Bounding box has no visible pixels: {bbox}, image={image.size}")
    return image.crop((left, top, right, bottom)).convert("RGB")


def extract_rotated_crop(image: Image.Image, box: dict[str, Any]) -> Image.Image:
    width = max(1, int(math.floor(float(box["width"]) + 0.5)))
    height = max(1, int(math.floor(float(box["height"]) + 0.5)))
    center_x = float(box["centerX"])
    center_y = float(box["centerY"])
    angle = math.radians(float(box["angleDeg"]))
    cosine = math.cos(angle)
    sine = math.sin(angle)

    # Pillow's affine transform maps output coordinates back to the input image.
    # This is the inverse of the editor's localToWorld transform and therefore
    # deskews the arbitrary-angle annotation rectangle without adding a padded
    # axis-aligned border around it.
    affine = (
        cosine,
        -sine,
        center_x - cosine * width / 2 + sine * height / 2,
        sine,
        cosine,
        center_y - sine * width / 2 - cosine * height / 2,
    )
    return image.transform(
        (width, height),
        Image.Transform.AFFINE,
        affine,
        resample=Image.Resampling.BICUBIC,
    ).convert("RGB")


def crop_row(
    *,
    crop_id: str,
    source: str,
    source_partition: str,
    tile_label: str,
    raw_category_name: str,
    raw_category_id: int | None,
    crop: Image.Image,
    source_image_path: str,
    source_image_id: str | None,
    source_annotation_id: str,
    bbox: dict[str, Any],
    capture_id: str | None = None,
    layout_id: str | None = None,
    layout_ordinal: int | None = None,
    region: str | None = None,
    group_name: str | None = None,
    group_ordinal: int | None = None,
    tile_ordinal: int | None = None,
    brightness: str | None = None,
    shadow: str | None = None,
    annotation_angle_deg: float = 0.0,
    expected_rotation_deg: int = 0,
) -> dict[str, Any]:
    encoded = encode_png(crop)
    return {
        "crop_id": crop_id,
        "source": source,
        "source_partition": source_partition,
        "tile_label": tile_label,
        "raw_category_name": raw_category_name,
        "raw_category_id": raw_category_id,
        "image_width": crop.width,
        "image_height": crop.height,
        "image_png": encoded,
        "source_image_path": source_image_path,
        "source_image_id": source_image_id,
        "source_annotation_id": source_annotation_id,
        "bbox_json": json.dumps(bbox, ensure_ascii=False, separators=(",", ":")),
        "capture_id": capture_id,
        "layout_id": layout_id,
        "layout_ordinal": layout_ordinal,
        "region": region,
        "group_name": group_name,
        "group_ordinal": group_ordinal,
        "tile_ordinal": tile_ordinal,
        "brightness": brightness,
        "shadow": shadow,
        "annotation_angle_deg": annotation_angle_deg,
        "expected_rotation_deg": expected_rotation_deg,
    }


def encode_png(image: Image.Image) -> bytes:
    output = io.BytesIO()
    image.save(output, format="PNG", compress_level=1, optimize=False)
    return output.getvalue()


def insert_crop_batch(
    connection: sqlite3.Connection,
    rows: Sequence[dict[str, Any]],
) -> None:
    connection.executemany(INSERT_SQL, rows)
    connection.commit()


def validated_coco_bbox(annotation: dict[str, Any]) -> list[float]:
    bbox = annotation.get("bbox")
    if not isinstance(bbox, list) or len(bbox) != 4:
        raise ValueError(f"Invalid bbox for annotation {annotation.get('id')}")
    values = [float(value) for value in bbox]
    if not all(math.isfinite(value) for value in values):
        raise ValueError(f"Non-finite bbox for annotation {annotation.get('id')}: {bbox}")
    if values[2] <= 0 or values[3] <= 0:
        raise ValueError(f"Non-positive bbox for annotation {annotation.get('id')}: {bbox}")
    return values


def category_map(payload: dict[str, Any], path: Path) -> dict[int, str]:
    categories = payload.get("categories")
    if not isinstance(categories, list):
        raise ValueError(f"Invalid COCO categories: {path}")
    result: dict[int, str] = {}
    for category in categories:
        if not isinstance(category, dict):
            raise ValueError(f"Non-object category in {path}")
        category_id = int(category["id"])
        if category_id in result:
            raise ValueError(f"Duplicate category id {category_id} in {path}")
        result[category_id] = str(category["name"])
    return result


def load_rgb_image(path: Path) -> Image.Image:
    with Image.open(path) as source:
        source.load()
        oriented = ImageOps.exif_transpose(source)
        oriented.load()
        return oriented.convert("RGB")


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as source:
        payload = json.load(source)
    if not isinstance(payload, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return payload


def safe_relative_path(value: str) -> Path:
    pure = PurePosixPath(value.replace("\\", "/"))
    if pure.is_absolute() or ".." in pure.parts:
        raise ValueError(f"Unsafe relative path: {value}")
    return Path(*pure.parts)


def repository_relative_path(path: Path, repository_root: Path) -> str:
    resolved = path.resolve()
    try:
        relative = resolved.relative_to(repository_root.resolve())
    except ValueError:
        return str(resolved)
    return relative.as_posix()


def files_fingerprint(paths: Sequence[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted((candidate.resolve() for candidate in paths), key=str):
        stat = path.stat()
        digest.update(str(path).encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(stat.st_size).encode("ascii"))
        digest.update(b"\0")
        digest.update(str(stat.st_mtime_ns).encode("ascii"))
        digest.update(b"\0")
    return digest.hexdigest()


def manual_rows_fingerprint(
    rows: Sequence[sqlite3.Row],
    storage_root: Path,
) -> str:
    digest = hashlib.sha256()
    for row in rows:
        for column in (
            "capture_id",
            "hand_crop_path",
            "dora_crop_path",
            "meld_crop_path",
            "layout_id",
            "layout_ordinal",
            "brightness",
            "shadow",
            "task_json",
            "annotation_json",
        ):
            value = row[column]
            digest.update(column.encode("utf-8"))
            digest.update(b"=")
            digest.update(("" if value is None else str(value)).encode("utf-8"))
            digest.update(b"\0")

        for path_column in MANUAL_REGION_COLUMNS.values():
            path_value = row[path_column]
            if path_value is None:
                continue
            path = storage_root / safe_relative_path(str(path_value))
            stat = path.stat()
            digest.update(path_column.encode("utf-8"))
            digest.update(b"=")
            digest.update(str(path.resolve()).encode("utf-8"))
            digest.update(b"\0")
            digest.update(str(stat.st_size).encode("ascii"))
            digest.update(b"\0")
            digest.update(str(stat.st_mtime_ns).encode("ascii"))
            digest.update(b"\0")
    return digest.hexdigest()


def source_is_current(
    connection: sqlite3.Connection,
    metadata_prefix: str,
    fingerprint: str,
) -> bool:
    return (
        get_metadata(connection, f"{metadata_prefix}.status") == "complete"
        and get_metadata(connection, f"{metadata_prefix}.fingerprint") == fingerprint
        and get_metadata(connection, f"{metadata_prefix}.builder_version")
        == BUILDER_VERSION
    )


def source_can_resume(
    connection: sqlite3.Connection,
    metadata_prefix: str,
    fingerprint: str,
) -> bool:
    return (
        get_metadata(connection, f"{metadata_prefix}.status") == "building"
        and get_metadata(connection, f"{metadata_prefix}.fingerprint") == fingerprint
    )


def source_annotation_ids(
    connection: sqlite3.Connection,
    source: str,
    partition: str,
) -> set[str]:
    return {
        str(row["source_annotation_id"])
        for row in connection.execute(
            """
            SELECT source_annotation_id
            FROM tile_crop
            WHERE source = ? AND source_partition = ?
            """,
            (source, partition),
        )
    }


def source_crop_ids(
    connection: sqlite3.Connection,
    source: str,
    partition: str,
) -> set[str]:
    return {
        str(row["crop_id"])
        for row in connection.execute(
            """
            SELECT crop_id
            FROM tile_crop
            WHERE source = ? AND source_partition = ?
            """,
            (source, partition),
        )
    }


def source_label_counts(
    connection: sqlite3.Connection,
    source: str,
    partition: str,
) -> Counter[str]:
    return Counter(
        {
            str(row["tile_label"]): int(row["crop_count"])
            for row in connection.execute(
                """
                SELECT tile_label, COUNT(*) AS crop_count
                FROM tile_crop
                WHERE source = ? AND source_partition = ?
                GROUP BY tile_label
                """,
                (source, partition),
            ).fetchall()
        }
    )


def mark_source_building(
    connection: sqlite3.Connection,
    metadata_prefix: str,
    fingerprint: str,
) -> None:
    set_metadata(connection, f"{metadata_prefix}.status", "building")
    set_metadata(connection, f"{metadata_prefix}.fingerprint", fingerprint)
    set_metadata(connection, f"{metadata_prefix}.builder_version", BUILDER_VERSION)
    connection.commit()


def mark_source_complete(
    connection: sqlite3.Connection,
    metadata_prefix: str,
    fingerprint: str,
    count: int,
    label_counts: Counter[str],
) -> None:
    set_metadata(connection, f"{metadata_prefix}.fingerprint", fingerprint)
    set_metadata(connection, f"{metadata_prefix}.crop_count", str(count))
    set_metadata(
        connection,
        f"{metadata_prefix}.label_counts",
        json.dumps(dict(sorted(label_counts.items())), ensure_ascii=False),
    )
    set_metadata(connection, f"{metadata_prefix}.status", "complete")


def set_metadata(connection: sqlite3.Connection, key: str, value: str) -> None:
    connection.execute(
        """
        INSERT INTO dataset_metadata(key, value)
        VALUES (?, ?)
        ON CONFLICT(key) DO UPDATE SET value = excluded.value
        """,
        (key, value),
    )


def get_metadata(connection: sqlite3.Connection, key: str) -> str | None:
    row = connection.execute(
        "SELECT value FROM dataset_metadata WHERE key = ?",
        (key,),
    ).fetchone()
    return None if row is None else str(row["value"])


def source_partition_count(
    connection: sqlite3.Connection,
    source: str,
    partition: str,
) -> int:
    return int(
        connection.execute(
            """
            SELECT COUNT(*)
            FROM tile_crop
            WHERE source = ? AND source_partition = ?
            """,
            (source, partition),
        ).fetchone()[0]
    )


def build_summary(
    connection: sqlite3.Connection,
    output_database: Path,
    build_events: list[dict[str, Any]],
) -> dict[str, Any]:
    total = int(connection.execute("SELECT COUNT(*) FROM tile_crop").fetchone()[0])
    by_source = {
        str(row["source"]): int(row["crop_count"])
        for row in connection.execute(
            """
            SELECT source, COUNT(*) AS crop_count
            FROM tile_crop
            GROUP BY source
            ORDER BY source
            """
        ).fetchall()
    }
    by_source_and_label: dict[str, dict[str, int]] = defaultdict(dict)
    for row in connection.execute(
        """
        SELECT source, tile_label, COUNT(*) AS crop_count
        FROM tile_crop
        GROUP BY source, tile_label
        ORDER BY source, tile_label
        """
    ).fetchall():
        by_source_and_label[str(row["source"])][str(row["tile_label"])] = int(
            row["crop_count"]
        )
    return {
        "status": "completed",
        "database": str(output_database),
        "database_bytes": output_database.stat().st_size,
        "crop_count": total,
        "counts_by_source": by_source,
        "counts_by_source_and_label": dict(by_source_and_label),
        "build_events": build_events,
        "storage": {
            "kind": "sqlite_png_blob",
            "table": "tile_crop",
            "image_column": "image_png",
            "lossless": True,
            "png_compress_level": 1,
        },
    }


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
