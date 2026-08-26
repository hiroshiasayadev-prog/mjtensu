from __future__ import annotations

import json
import sqlite3
import sys
from datetime import UTC, datetime
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from tools.recognition.capture_dataset_api.campaign import CAMPAIGN_ID
from tools.recognition.capture_dataset_api.database import CaptureDatabase


LAYOUT_ID = "layout-024"
CORRECTED_HAND = ["5p", "5p", "5p", "5p", "6p", "6p"]
CORRECTION_REASON = (
    "The captured layout used four 5p tiles instead of the planned three, "
    "increasing the completed-hand count from five to six."
)


def main() -> int:
    database_path = (
        REPOSITORY_ROOT
        / ".local"
        / "recognition"
        / "capture_dataset"
        / "dataset.sqlite"
    )
    if not database_path.is_file():
        raise FileNotFoundError(database_path)

    backup_directory = database_path.parent / "backups"
    backup_directory.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    backup_path = backup_directory / f"dataset-before-layout-024-hand-fix-{timestamp}.sqlite"
    _backup_sqlite(database_path, backup_path)

    database = CaptureDatabase(database_path)
    result = database.replace_layout_hand_expectation(
        CAMPAIGN_ID,
        LAYOUT_ID,
        CORRECTED_HAND,
        correction_reason=CORRECTION_REASON,
    )
    result["backupPath"] = str(backup_path)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


def _backup_sqlite(source_path: Path, destination_path: Path) -> None:
    with sqlite3.connect(source_path) as source:
        with sqlite3.connect(destination_path) as destination:
            source.backup(destination)


if __name__ == "__main__":
    raise SystemExit(main())
