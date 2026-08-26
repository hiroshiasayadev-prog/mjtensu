from __future__ import annotations

import argparse
import json
import shutil
import sqlite3
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Iterable


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_STORAGE_ROOT = REPOSITORY_ROOT / ".local" / "recognition" / "capture_dataset"
DEFAULT_CAMPAIGN_ID = "tile-catalog-warm-4-v2"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Delete every saved capture in one tile-catalog campaign and return all "
            "of its tasks to pending. A consistent SQLite backup and archived image "
            "copy are created before the reset is committed."
        )
    )
    parser.add_argument("--storage-root", type=Path, default=DEFAULT_STORAGE_ROOT)
    parser.add_argument("--campaign-id", default=DEFAULT_CAMPAIGN_ID)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    storage_root = args.storage_root.resolve()
    database_path = storage_root / "dataset.sqlite"
    if not database_path.is_file():
        raise FileNotFoundError(database_path)

    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    backup_directory = storage_root / "backups"
    backup_directory.mkdir(parents=True, exist_ok=True)
    database_backup = backup_directory / f"dataset-before-{args.campaign_id}-reset-{timestamp}.sqlite"
    archive_root = backup_directory / f"{args.campaign_id}-reset-{timestamp}-files"

    with sqlite3.connect(database_path) as source, sqlite3.connect(database_backup) as destination:
        source.backup(destination)

    connection = sqlite3.connect(database_path, timeout=30)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    moved: list[tuple[Path, Path]] = []
    try:
        rows = connection.execute(
            """
            SELECT
                capture.id,
                capture.original_path,
                capture.composite_path,
                capture.hand_crop_path,
                capture.dora_crop_path,
                capture.meld_crop_path
            FROM capture
            JOIN capture_task ON capture_task.id = capture.task_id
            WHERE capture_task.campaign_id = ?
            ORDER BY capture_task.task_order
            """,
            (args.campaign_id,),
        ).fetchall()
        if not rows:
            print(
                json.dumps(
                    {
                        "status": "nothing-to-reset",
                        "campaignId": args.campaign_id,
                        "captureCount": 0,
                        "databaseBackup": str(database_backup),
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )
            return 0

        relative_paths = sorted(
            {
                str(path)
                for row in rows
                for path in _row_paths(row)
                if path is not None
            }
        )
        for relative_path in relative_paths:
            source_path = resolve_storage_path(storage_root, relative_path)
            if not source_path.exists():
                continue
            destination_path = archive_root.joinpath(*PurePosixPath(relative_path).parts)
            destination_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(source_path), str(destination_path))
            moved.append((source_path, destination_path))

        connection.execute("BEGIN IMMEDIATE")
        deleted = connection.execute(
            """
            DELETE FROM capture
            WHERE task_id IN (
                SELECT id FROM capture_task WHERE campaign_id = ?
            )
            """,
            (args.campaign_id,),
        ).rowcount
        reset_tasks = connection.execute(
            """
            UPDATE capture_task
            SET status = 'pending', completed_at = NULL
            WHERE campaign_id = ?
            """,
            (args.campaign_id,),
        ).rowcount
        connection.commit()
    except Exception:
        connection.rollback()
        for original_path, archived_path in reversed(moved):
            if archived_path.exists():
                original_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(archived_path), str(original_path))
        raise
    finally:
        connection.close()

    print(
        json.dumps(
            {
                "status": "reset",
                "campaignId": args.campaign_id,
                "captureCount": len(rows),
                "deletedCaptureCount": int(deleted),
                "resetTaskCount": int(reset_tasks),
                "databaseBackup": str(database_backup),
                "archivedFiles": str(archive_root),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


def _row_paths(row: sqlite3.Row) -> Iterable[str | None]:
    for key in (
        "original_path",
        "composite_path",
        "hand_crop_path",
        "dora_crop_path",
        "meld_crop_path",
    ):
        yield row[key]


def resolve_storage_path(storage_root: Path, relative_path: str) -> Path:
    pure = PurePosixPath(relative_path.replace("\\", "/"))
    if pure.is_absolute() or ".." in pure.parts:
        raise ValueError(f"Unsafe storage-relative path: {relative_path}")
    candidate = storage_root.joinpath(*pure.parts).resolve()
    try:
        candidate.relative_to(storage_root)
    except ValueError as error:
        raise ValueError(f"Stored path escapes storage root: {relative_path}") from error
    return candidate


if __name__ == "__main__":
    raise SystemExit(main())
