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


CORRECTIONS = {
    "layout-015-open-kan": {
        "layoutId": "layout-015",
        "meldOrdinal": 0,
        "previous": ["red", "red", "red", "red"],
        "replacement": ["white", "white", "white", "white"],
        "reason": (
            "Classifier audit and manual review confirmed that the captured open kan "
            "used white dragons, while the task definition expected red dragons."
        ),
    },
    "layout-024-dora-visible-1": {
        "layoutId": "layout-024",
        "row": "visible",
        "tileOrdinal": 1,
        "previous": "1s",
        "replacement": "1p",
        "reason": (
            "Classifier audit and manual review confirmed that dora-visible slot 1 "
            "was physically captured as 1p, while the task definition expected 1s."
        ),
    },
    "layout-028-closed-kan": {
        "layoutId": "layout-028",
        "meldOrdinal": 0,
        "previous": ["3s", "3s", "3s", "3s"],
        "replacement": ["3m", "3m", "3m", "3m"],
        "reason": (
            "Classifier audit and manual review confirmed that the captured closed kan "
            "used 3m tiles, while the task definition expected 3s tiles."
        ),
    },
}


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
    backup_path = (
        backup_directory
        / f"dataset-before-classifier-audit-label-fixes-{timestamp}.sqlite"
    )
    _backup_sqlite(database_path, backup_path)

    database = CaptureDatabase(database_path)
    results: list[dict[str, object]] = []

    layout_015 = CORRECTIONS["layout-015-open-kan"]
    results.append(
        database.replace_layout_meld_tile_expectation(
            CAMPAIGN_ID,
            str(layout_015["layoutId"]),
            int(layout_015["meldOrdinal"]),
            list(layout_015["replacement"]),
            expected_previous_tile_codes=list(layout_015["previous"]),
            correction_reason=str(layout_015["reason"]),
        )
    )

    layout_024 = CORRECTIONS["layout-024-dora-visible-1"]
    results.append(
        database.replace_layout_dora_tile_expectation(
            CAMPAIGN_ID,
            str(layout_024["layoutId"]),
            str(layout_024["row"]),
            int(layout_024["tileOrdinal"]),
            str(layout_024["replacement"]),
            expected_previous_tile_code=str(layout_024["previous"]),
            correction_reason=str(layout_024["reason"]),
        )
    )

    layout_028 = CORRECTIONS["layout-028-closed-kan"]
    results.append(
        database.replace_layout_meld_tile_expectation(
            CAMPAIGN_ID,
            str(layout_028["layoutId"]),
            int(layout_028["meldOrdinal"]),
            list(layout_028["replacement"]),
            expected_previous_tile_codes=list(layout_028["previous"]),
            correction_reason=str(layout_028["reason"]),
        )
    )

    verification = _verify(database_path)
    payload = {
        "status": "completed",
        "database": str(database_path),
        "backupPath": str(backup_path),
        "results": results,
        "verification": verification,
        "nextStep": (
            "Rebuild only the manual source of tile_crop_dataset, then rebuild the "
            "compact classifier dataset before retraining."
        ),
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


def _verify(database_path: Path) -> dict[str, object]:
    checks = {
        "layout-015": {
            "sql": """
                SELECT task.id, slot.tile_ordinal, slot.tile_code
                FROM capture_task AS task
                JOIN task_tile_slot AS slot ON slot.task_id = task.id
                WHERE task.campaign_id = ?
                  AND task.layout_id = 'layout-015'
                  AND slot.region = 'meld'
                  AND slot.group_ordinal = 0
                ORDER BY task.environment_ordinal, slot.tile_ordinal
            """,
            "expected": ["white", "white", "white", "white"],
        },
        "layout-024": {
            "sql": """
                SELECT task.id, slot.tile_ordinal, slot.tile_code
                FROM capture_task AS task
                JOIN task_tile_slot AS slot ON slot.task_id = task.id
                WHERE task.campaign_id = ?
                  AND task.layout_id = 'layout-024'
                  AND slot.region = 'dora-visible'
                  AND slot.row_ordinal = 0
                  AND slot.tile_ordinal = 1
                ORDER BY task.environment_ordinal
            """,
            "expected": ["1p"],
        },
        "layout-028": {
            "sql": """
                SELECT task.id, slot.tile_ordinal, slot.tile_code
                FROM capture_task AS task
                JOIN task_tile_slot AS slot ON slot.task_id = task.id
                WHERE task.campaign_id = ?
                  AND task.layout_id = 'layout-028'
                  AND slot.region = 'meld'
                  AND slot.group_ordinal = 0
                ORDER BY task.environment_ordinal, slot.tile_ordinal
            """,
            "expected": ["3m", "3m", "3m", "3m"],
        },
    }

    result: dict[str, object] = {}
    with sqlite3.connect(database_path) as connection:
        connection.row_factory = sqlite3.Row
        for name, check in checks.items():
            rows = connection.execute(str(check["sql"]), (CAMPAIGN_ID,)).fetchall()
            by_task: dict[str, list[str]] = {}
            for row in rows:
                by_task.setdefault(str(row["id"]), []).append(str(row["tile_code"]))
            expected = list(check["expected"])
            mismatches = {
                task_id: values
                for task_id, values in by_task.items()
                if values != expected
            }
            if len(by_task) != 4 or mismatches:
                raise RuntimeError(
                    f"Verification failed for {name}: taskCount={len(by_task)}, "
                    f"expected={expected}, mismatches={mismatches}"
                )
            result[name] = {
                "taskCount": len(by_task),
                "expectedPerTask": expected,
                "verified": True,
            }
    return result


def _backup_sqlite(source_path: Path, destination_path: Path) -> None:
    with sqlite3.connect(source_path) as source:
        with sqlite3.connect(destination_path) as destination:
            source.backup(destination)


if __name__ == "__main__":
    raise SystemExit(main())
