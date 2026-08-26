from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterator

from .campaign import CAMPAIGN_ID, VISIBLE_TILE_CODES, task_slots


SCHEMA = """
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS campaign (
    id                  TEXT PRIMARY KEY,
    name                TEXT NOT NULL,
    definition_sha256   TEXT NOT NULL,
    definition_json     TEXT NOT NULL,
    created_at          TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS capture_task (
    id                  TEXT PRIMARY KEY,
    campaign_id         TEXT NOT NULL REFERENCES campaign(id),
    layout_id           TEXT NOT NULL,
    layout_ordinal      INTEGER NOT NULL,
    environment_ordinal INTEGER NOT NULL,
    brightness          TEXT NOT NULL,
    shadow              TEXT NOT NULL,
    repetition          INTEGER NOT NULL,
    expected_hand       INTEGER NOT NULL,
    expected_dora       INTEGER NOT NULL,
    expected_meld       INTEGER NOT NULL,
    task_order          INTEGER NOT NULL UNIQUE,
    task_json           TEXT NOT NULL,
    status              TEXT NOT NULL DEFAULT 'pending' CHECK (status IN ('pending', 'completed')),
    completed_at        TEXT,
    UNIQUE(campaign_id, layout_id, brightness, shadow, repetition)
);

CREATE INDEX IF NOT EXISTS idx_capture_task_next
ON capture_task(campaign_id, status, task_order);

CREATE TABLE IF NOT EXISTS task_tile_slot (
    slot_key            TEXT PRIMARY KEY,
    task_id             TEXT NOT NULL REFERENCES capture_task(id) ON DELETE CASCADE,
    region              TEXT NOT NULL,
    row_ordinal         INTEGER,
    group_ordinal       INTEGER,
    tile_ordinal        INTEGER NOT NULL,
    tile_code           TEXT NOT NULL,
    face                TEXT NOT NULL CHECK (face IN ('front', 'back')),
    rotation            INTEGER NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_task_tile_slot_task
ON task_tile_slot(task_id, region, row_ordinal, group_ordinal, tile_ordinal);

CREATE TABLE IF NOT EXISTS capture (
    id                    TEXT PRIMARY KEY,
    task_id               TEXT NOT NULL UNIQUE REFERENCES capture_task(id),
    captured_at           TEXT NOT NULL,
    stored_at             TEXT NOT NULL,
    original_path         TEXT NOT NULL,
    composite_path        TEXT NOT NULL,
    hand_crop_path        TEXT,
    dora_crop_path        TEXT,
    meld_crop_path        TEXT,
    original_width        INTEGER NOT NULL,
    original_height       INTEGER NOT NULL,
    model_sha256          TEXT NOT NULL,
    model_name            TEXT NOT NULL,
    layout_version        TEXT NOT NULL,
    confidence_threshold  REAL NOT NULL,
    nms_iou_threshold     REAL NOT NULL,
    provider              TEXT NOT NULL,
    camera_json           TEXT NOT NULL,
    telemetry_json        TEXT NOT NULL,
    preview_json          TEXT NOT NULL,
    region_rects_json     TEXT NOT NULL,
    manifest_json         TEXT NOT NULL,
    upload_client_id      TEXT NOT NULL UNIQUE
);

CREATE INDEX IF NOT EXISTS idx_capture_task
ON capture(task_id, captured_at);

CREATE UNIQUE INDEX IF NOT EXISTS idx_capture_unique_task
ON capture(task_id);

CREATE TABLE IF NOT EXISTS detection (
    capture_id         TEXT NOT NULL REFERENCES capture(id) ON DELETE CASCADE,
    detection_index    INTEGER NOT NULL,
    region             TEXT NOT NULL,
    confidence         REAL NOT NULL,
    composite_x        REAL NOT NULL,
    composite_y        REAL NOT NULL,
    composite_width    REAL NOT NULL,
    composite_height   REAL NOT NULL,
    original_x         REAL,
    original_y         REAL,
    original_width     REAL,
    original_height    REAL,
    preview_x          REAL,
    preview_y          REAL,
    preview_width      REAL,
    preview_height     REAL,
    PRIMARY KEY (capture_id, detection_index)
);

CREATE TABLE IF NOT EXISTS capture_annotation (
    capture_id         TEXT PRIMARY KEY REFERENCES capture(id) ON DELETE CASCADE,
    status             TEXT NOT NULL CHECK (status IN ('draft', 'complete')),
    schema_version     INTEGER NOT NULL,
    annotation_json    TEXT NOT NULL,
    updated_at         TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_capture_annotation_status
ON capture_annotation(status, updated_at);

CREATE TABLE IF NOT EXISTS detection_refresh (
    capture_id          TEXT PRIMARY KEY REFERENCES capture(id) ON DELETE CASCADE,
    model_sha256        TEXT NOT NULL,
    model_name          TEXT NOT NULL,
    confidence_threshold REAL NOT NULL,
    nms_iou_threshold   REAL NOT NULL,
    provider            TEXT NOT NULL,
    refreshed_at        TEXT NOT NULL
);

CREATE VIEW IF NOT EXISTS capture_expected_tile_slot AS
SELECT
    capture.id AS capture_id,
    capture.task_id,
    slot.region,
    slot.row_ordinal,
    slot.group_ordinal,
    slot.tile_ordinal,
    slot.tile_code,
    slot.face,
    CASE WHEN slot.face = 'back' THEN 'back' ELSE slot.tile_code END AS visible_class,
    slot.rotation
FROM capture
JOIN task_tile_slot AS slot ON slot.task_id = capture.task_id;
"""


class CaptureDatabase:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as connection:
            connection.executescript(SCHEMA)
            self._migrate_schema(connection)

    @staticmethod
    def _migrate_schema(connection: sqlite3.Connection) -> None:
        capture_columns = {
            str(row["name"])
            for row in connection.execute("PRAGMA table_info(capture)").fetchall()
        }
        if "preview_json" not in capture_columns:
            connection.execute(
                "ALTER TABLE capture ADD COLUMN preview_json TEXT NOT NULL DEFAULT '{}'"
            )

        detection_columns = {
            str(row["name"])
            for row in connection.execute("PRAGMA table_info(detection)").fetchall()
        }
        for column in ("preview_x", "preview_y", "preview_width", "preview_height"):
            if column not in detection_columns:
                connection.execute(f"ALTER TABLE detection ADD COLUMN {column} REAL")

        connection.execute("DROP VIEW IF EXISTS capture_expected_tile_slot")
        connection.execute(
            """
            CREATE VIEW capture_expected_tile_slot AS
            SELECT
                capture.id AS capture_id,
                capture.task_id,
                slot.region,
                slot.row_ordinal,
                slot.group_ordinal,
                slot.tile_ordinal,
                slot.tile_code,
                slot.face,
                CASE WHEN slot.face = 'back' THEN 'back' ELSE slot.tile_code END AS visible_class,
                slot.rotation
            FROM capture
            JOIN task_tile_slot AS slot ON slot.task_id = capture.task_id
            """
        )

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.path, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA journal_mode = WAL")
        connection.execute("PRAGMA synchronous = NORMAL")
        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def seed_campaign(self, campaign: dict[str, Any]) -> None:
        definition_sha256 = str(campaign["definitionSha256"])
        definition_json = json.dumps(campaign, ensure_ascii=False, sort_keys=True)
        with self.connect() as connection:
            existing = connection.execute(
                "SELECT definition_sha256 FROM campaign WHERE id = ?",
                (campaign["id"],),
            ).fetchone()
            if existing is not None:
                if existing["definition_sha256"] != definition_sha256:
                    capture_count = connection.execute(
                        "SELECT COUNT(*) FROM capture_task WHERE campaign_id = ? AND status = 'completed'",
                        (campaign["id"],),
                    ).fetchone()[0]
                    if capture_count > 0:
                        raise RuntimeError(
                            "Campaign definition changed after captures were recorded. "
                            f"Use a new campaign ID instead of mutating {CAMPAIGN_ID}."
                        )
                    connection.execute(
                        "DELETE FROM capture_task WHERE campaign_id = ?",
                        (campaign["id"],),
                    )
                    connection.execute("DELETE FROM campaign WHERE id = ?", (campaign["id"],))
                else:
                    return

            connection.execute(
                """
                INSERT INTO campaign(id, name, definition_sha256, definition_json, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    campaign["id"],
                    campaign["name"],
                    definition_sha256,
                    definition_json,
                    _now_iso(),
                ),
            )
            task_order_offset = int(
                connection.execute(
                    "SELECT COALESCE(MAX(task_order) + 1, 0) FROM capture_task"
                ).fetchone()[0]
            )
            for task in campaign["tasks"]:
                connection.execute(
                    """
                    INSERT INTO capture_task(
                        id, campaign_id, layout_id, layout_ordinal,
                        environment_ordinal, brightness, shadow, repetition,
                        expected_hand, expected_dora, expected_meld,
                        task_order, task_json, status
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'pending')
                    """,
                    (
                        task["id"],
                        task["campaignId"],
                        task["layoutId"],
                        task["layoutOrdinal"],
                        task["environmentOrdinal"],
                        task["environment"]["brightness"],
                        task["environment"]["shadow"],
                        int(task.get("repetition", 0)),
                        task["expected"]["hand"],
                        task["expected"]["dora"],
                        task["expected"]["meld"],
                        task_order_offset + int(task["taskOrder"]),
                        json.dumps(task, ensure_ascii=False, sort_keys=True),
                    ),
                )
                connection.executemany(
                    """
                    INSERT INTO task_tile_slot(
                        slot_key, task_id, region, row_ordinal, group_ordinal,
                        tile_ordinal, tile_code, face, rotation
                    ) VALUES (
                        :slot_key, :task_id, :region, :row_ordinal, :group_ordinal,
                        :tile_ordinal, :tile_code, :face, :rotation
                    )
                    """,
                    list(task_slots(task)),
                )

    def campaign_overview(self, campaign_id: str) -> dict[str, Any] | None:
        with self.connect() as connection:
            campaign = connection.execute(
                "SELECT id, name, definition_json FROM campaign WHERE id = ?",
                (campaign_id,),
            ).fetchone()
            if campaign is None:
                return None
            counts = connection.execute(
                """
                SELECT
                    COUNT(*) AS total_tasks,
                    SUM(CASE WHEN status = 'completed' THEN 1 ELSE 0 END) AS completed_tasks,
                    COUNT(DISTINCT layout_id) AS total_layouts
                FROM capture_task
                WHERE campaign_id = ?
                """,
                (campaign_id,),
            ).fetchone()
            completed_layouts = connection.execute(
                """
                SELECT COUNT(*)
                FROM (
                    SELECT layout_id
                    FROM capture_task
                    WHERE campaign_id = ?
                    GROUP BY layout_id
                    HAVING SUM(CASE WHEN status = 'completed' THEN 1 ELSE 0 END) = COUNT(*)
                )
                """,
                (campaign_id,),
            ).fetchone()[0]

            coverage: dict[str, int] = {tile: 0 for tile in VISIBLE_TILE_CODES}
            coverage["back"] = 0
            rows = connection.execute(
                """
                SELECT
                    CASE WHEN slot.face = 'back' THEN 'back' ELSE slot.tile_code END AS visible_class,
                    COUNT(*) AS capture_count
                FROM task_tile_slot AS slot
                JOIN capture_task AS task ON task.id = slot.task_id
                WHERE task.campaign_id = ? AND task.status = 'completed'
                GROUP BY visible_class
                """,
                (campaign_id,),
            ).fetchall()
            for row in rows:
                coverage[str(row["visible_class"])] = int(row["capture_count"])

            total_tasks = int(counts["total_tasks"] or 0)
            completed_tasks = int(counts["completed_tasks"] or 0)
            return {
                "campaignId": campaign["id"],
                "name": campaign["name"],
                "totalTasks": total_tasks,
                "completedTasks": completed_tasks,
                "pendingTasks": total_tasks - completed_tasks,
                "completedLayouts": int(completed_layouts),
                "totalLayouts": int(counts["total_layouts"] or 0),
                "coverage": coverage,
            }

    def next_task(self, campaign_id: str) -> dict[str, Any] | None:
        with self.connect() as connection:
            row = connection.execute(
                """
                SELECT task_json
                FROM capture_task
                WHERE campaign_id = ? AND status = 'pending'
                ORDER BY task_order
                LIMIT 1
                """,
                (campaign_id,),
            ).fetchone()
            return None if row is None else json.loads(row["task_json"])

    def task(self, task_id: str) -> dict[str, Any] | None:
        with self.connect() as connection:
            row = connection.execute(
                "SELECT task_json FROM capture_task WHERE id = ?",
                (task_id,),
            ).fetchone()
            return None if row is None else json.loads(row["task_json"])

    def capture_for_task(self, task_id: str) -> dict[str, Any] | None:
        with self.connect() as connection:
            row = connection.execute(
                "SELECT id, upload_client_id FROM capture WHERE task_id = ?",
                (task_id,),
            ).fetchone()
            if row is None:
                return None
            return {
                "captureId": row["id"],
                "uploadClientId": row["upload_client_id"],
            }

    def existing_capture(self, upload_client_id: str) -> dict[str, Any] | None:
        with self.connect() as connection:
            row = connection.execute(
                "SELECT id, task_id FROM capture WHERE upload_client_id = ?",
                (upload_client_id,),
            ).fetchone()
            if row is None:
                return None
            return {
                "captureId": row["id"],
                "taskId": row["task_id"],
            }

    def insert_capture(
        self,
        capture_id: str,
        manifest: dict[str, Any],
        paths: dict[str, str | None],
    ) -> None:
        original = manifest["original"]
        model = manifest["model"]
        with self.connect() as connection:
            connection.execute(
                """
                INSERT INTO capture(
                    id, task_id, captured_at, stored_at,
                    original_path, composite_path, hand_crop_path, dora_crop_path, meld_crop_path,
                    original_width, original_height,
                    model_sha256, model_name, layout_version,
                    confidence_threshold, nms_iou_threshold, provider,
                    camera_json, telemetry_json, preview_json, region_rects_json,
                    manifest_json, upload_client_id
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    capture_id,
                    manifest["taskId"],
                    manifest["capturedAt"],
                    _now_iso(),
                    paths["original"],
                    paths["composite"],
                    paths["hand_crop"],
                    paths["dora_crop"],
                    paths["meld_crop"],
                    int(original["width"]),
                    int(original["height"]),
                    model["sha256"],
                    model["name"],
                    manifest["layoutVersion"],
                    float(manifest["confidenceThreshold"]),
                    float(manifest["nmsIouThreshold"]),
                    manifest["provider"],
                    json.dumps(manifest["camera"], ensure_ascii=False, sort_keys=True),
                    json.dumps(manifest["telemetry"], ensure_ascii=False, sort_keys=True),
                    json.dumps(manifest["preview"], ensure_ascii=False, sort_keys=True),
                    json.dumps(manifest["regionRects"], ensure_ascii=False, sort_keys=True),
                    json.dumps(manifest, ensure_ascii=False, sort_keys=True),
                    manifest["uploadClientId"],
                ),
            )
            for detection in manifest["detections"]:
                composite = detection["composite"]
                original_rect = detection.get("original")
                preview_rect = detection.get("preview")
                connection.execute(
                    """
                    INSERT INTO detection(
                        capture_id, detection_index, region, confidence,
                        composite_x, composite_y, composite_width, composite_height,
                        original_x, original_y, original_width, original_height,
                        preview_x, preview_y, preview_width, preview_height
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        capture_id,
                        int(detection["detectionIndex"]),
                        detection["region"],
                        float(detection["confidence"]),
                        float(composite["x"]),
                        float(composite["y"]),
                        float(composite["width"]),
                        float(composite["height"]),
                        None if original_rect is None else float(original_rect["x"]),
                        None if original_rect is None else float(original_rect["y"]),
                        None if original_rect is None else float(original_rect["width"]),
                        None if original_rect is None else float(original_rect["height"]),
                        None if preview_rect is None else float(preview_rect["x"]),
                        None if preview_rect is None else float(preview_rect["y"]),
                        None if preview_rect is None else float(preview_rect["width"]),
                        None if preview_rect is None else float(preview_rect["height"]),
                    ),
                )
            connection.execute(
                """
                UPDATE capture_task
                SET status = 'completed', completed_at = COALESCE(completed_at, ?)
                WHERE id = ?
                """,
                (_now_iso(), manifest["taskId"]),
            )

    def undo_last_capture(self, campaign_id: str) -> dict[str, Any] | None:
        with self.connect() as connection:
            row = connection.execute(
                """
                SELECT
                    capture.id AS capture_id,
                    capture.task_id,
                    capture.original_path,
                    capture.composite_path,
                    capture.hand_crop_path,
                    capture.dora_crop_path,
                    capture.meld_crop_path
                FROM capture
                JOIN capture_task ON capture_task.id = capture.task_id
                WHERE capture_task.campaign_id = ?
                ORDER BY capture.stored_at DESC, capture.id DESC
                LIMIT 1
                """,
                (campaign_id,),
            ).fetchone()
            if row is None:
                return None

            connection.execute("DELETE FROM capture WHERE id = ?", (row["capture_id"],))
            connection.execute(
                """
                UPDATE capture_task
                SET status = 'pending', completed_at = NULL
                WHERE id = ?
                """,
                (row["task_id"],),
            )
            return {
                "captureId": str(row["capture_id"]),
                "taskId": str(row["task_id"]),
                "paths": [
                    str(path)
                    for path in (
                        row["original_path"],
                        row["composite_path"],
                        row["hand_crop_path"],
                        row["dora_crop_path"],
                        row["meld_crop_path"],
                    )
                    if path is not None
                ],
            }

    def annotation_campaigns(self) -> list[dict[str, Any]]:
        with self.connect() as connection:
            rows = connection.execute(
                """
                SELECT
                    campaign.id AS campaign_id,
                    campaign.name,
                    COUNT(capture.id) AS capture_count,
                    SUM(CASE WHEN capture_annotation.status = 'complete' THEN 1 ELSE 0 END) AS complete_count,
                    SUM(CASE WHEN capture_annotation.status = 'draft' THEN 1 ELSE 0 END) AS draft_count,
                    MAX(capture.stored_at) AS latest_capture_at
                FROM campaign
                JOIN capture_task ON capture_task.campaign_id = campaign.id
                JOIN capture ON capture.task_id = capture_task.id
                LEFT JOIN capture_annotation ON capture_annotation.capture_id = capture.id
                GROUP BY campaign.id, campaign.name
                ORDER BY latest_capture_at DESC, campaign.id
                """
            ).fetchall()
            return [
                {
                    "campaignId": str(row["campaign_id"]),
                    "name": str(row["name"]),
                    "captureCount": int(row["capture_count"] or 0),
                    "completeCount": int(row["complete_count"] or 0),
                    "draftCount": int(row["draft_count"] or 0),
                }
                for row in rows
            ]

    def annotation_capture_list(self, campaign_id: str) -> list[dict[str, Any]]:
        with self.connect() as connection:
            rows = connection.execute(
                """
                SELECT
                    capture.id AS capture_id,
                    capture.task_id,
                    capture.captured_at,
                    capture_task.layout_id,
                    capture_task.layout_ordinal,
                    capture_task.brightness,
                    capture_task.shadow,
                    capture_task.task_order,
                    capture_task.task_json,
                    COALESCE(capture_annotation.status, 'unannotated') AS annotation_status,
                    capture_annotation.updated_at
                FROM capture
                JOIN capture_task ON capture_task.id = capture.task_id
                LEFT JOIN capture_annotation ON capture_annotation.capture_id = capture.id
                WHERE capture_task.campaign_id = ?
                ORDER BY capture_task.task_order
                """,
                (campaign_id,),
            ).fetchall()
            result: list[dict[str, Any]] = []
            for row in rows:
                task = json.loads(row["task_json"])
                result.append(
                    {
                        "captureId": str(row["capture_id"]),
                        "taskId": str(row["task_id"]),
                        "capturedAt": str(row["captured_at"]),
                        "layoutId": str(row["layout_id"]),
                        "layoutOrdinal": int(row["layout_ordinal"]),
                        "environment": {
                            "brightness": str(row["brightness"]),
                            "shadow": str(row["shadow"]),
                            "label": task.get("environment", {}).get("label"),
                        },
                        "taskOrder": int(task.get("taskOrder", row["task_order"])),
                        "annotationStatus": str(row["annotation_status"]),
                        "annotationUpdatedAt": (
                            None if row["updated_at"] is None else str(row["updated_at"])
                        ),
                    }
                )
            return result

    def annotation_capture(self, capture_id: str) -> dict[str, Any] | None:
        with self.connect() as connection:
            row = connection.execute(
                """
                SELECT
                    capture.id AS capture_id,
                    capture.task_id,
                    capture.original_path,
                    capture.composite_path,
                    capture.hand_crop_path,
                    capture.dora_crop_path,
                    capture.meld_crop_path,
                    capture.original_width,
                    capture.original_height,
                    capture.manifest_json,
                    capture_task.campaign_id,
                    capture_task.task_json,
                    capture_task.task_order,
                    capture_annotation.status AS annotation_status,
                    capture_annotation.schema_version,
                    capture_annotation.annotation_json,
                    capture_annotation.updated_at
                FROM capture
                JOIN capture_task ON capture_task.id = capture.task_id
                LEFT JOIN capture_annotation ON capture_annotation.capture_id = capture.id
                WHERE capture.id = ?
                """,
                (capture_id,),
            ).fetchone()
            if row is None:
                return None
            detections = connection.execute(
                """
                SELECT
                    detection_index,
                    region,
                    confidence,
                    original_x,
                    original_y,
                    original_width,
                    original_height
                FROM detection
                WHERE capture_id = ?
                ORDER BY detection_index
                """,
                (capture_id,),
            ).fetchall()
            task = json.loads(row["task_json"])
            return {
                "captureId": str(row["capture_id"]),
                "taskId": str(row["task_id"]),
                "campaignId": str(row["campaign_id"]),
                "taskOrder": int(task.get("taskOrder", row["task_order"])),
                "task": task,
                "manifest": json.loads(row["manifest_json"]),
                "original": {
                    "path": str(row["original_path"]),
                    "width": int(row["original_width"]),
                    "height": int(row["original_height"]),
                },
                "regionPaths": {
                    "completed_hand": row["hand_crop_path"],
                    "dora_indicators": row["dora_crop_path"],
                    "melds": row["meld_crop_path"],
                },
                "detections": [
                    {
                        "detectionIndex": int(detection["detection_index"]),
                        "region": str(detection["region"]),
                        "confidence": float(detection["confidence"]),
                        "original": None
                        if detection["original_x"] is None
                        else {
                            "x": float(detection["original_x"]),
                            "y": float(detection["original_y"]),
                            "width": float(detection["original_width"]),
                            "height": float(detection["original_height"]),
                        },
                    }
                    for detection in detections
                ],
                "annotation": None
                if row["annotation_json"] is None
                else {
                    "status": str(row["annotation_status"]),
                    "schemaVersion": int(row["schema_version"]),
                    "updatedAt": str(row["updated_at"]),
                    "document": json.loads(row["annotation_json"]),
                },
            }

    def unannotated_captures(
        self,
        campaign_id: str,
        *,
        minimum_layout_ordinal: int = 0,
        include_drafts: bool = False,
    ) -> list[dict[str, Any]]:
        with self.connect() as connection:
            rows = connection.execute(
                """
                SELECT
                    capture.id AS capture_id,
                    capture.composite_path,
                    capture.manifest_json,
                    capture_task.layout_id,
                    capture_task.layout_ordinal,
                    capture_task.brightness,
                    capture_task.shadow,
                    capture_task.task_order
                FROM capture
                JOIN capture_task ON capture_task.id = capture.task_id
                LEFT JOIN capture_annotation ON capture_annotation.capture_id = capture.id
                WHERE
                    capture_task.campaign_id = ?
                    AND capture_task.layout_ordinal >= ?
                    AND (
                        capture_annotation.capture_id IS NULL
                        OR (? = 1 AND capture_annotation.status = 'draft')
                    )
                ORDER BY capture_task.task_order
                """,
                (campaign_id, minimum_layout_ordinal, int(include_drafts)),
            ).fetchall()
            return [
                {
                    "captureId": str(row["capture_id"]),
                    "compositePath": str(row["composite_path"]),
                    "manifest": json.loads(row["manifest_json"]),
                    "layoutId": str(row["layout_id"]),
                    "layoutOrdinal": int(row["layout_ordinal"]),
                    "environment": {
                        "brightness": str(row["brightness"]),
                        "shadow": str(row["shadow"]),
                    },
                    "taskOrder": int(row["task_order"]),
                }
                for row in rows
            ]

    def replace_unannotated_detections(
        self,
        capture_id: str,
        detections: list[dict[str, Any]],
        *,
        model_sha256: str,
        model_name: str,
        confidence_threshold: float,
        nms_iou_threshold: float,
        provider: str,
        allow_draft: bool = False,
    ) -> bool:
        with self.connect() as connection:
            capture = connection.execute(
                "SELECT 1 FROM capture WHERE id = ?",
                (capture_id,),
            ).fetchone()
            if capture is None:
                raise KeyError(capture_id)
            annotation = connection.execute(
                "SELECT status FROM capture_annotation WHERE capture_id = ?",
                (capture_id,),
            ).fetchone()
            if annotation is not None and not (
                allow_draft and str(annotation["status"]) == "draft"
            ):
                return False

            connection.execute("DELETE FROM detection WHERE capture_id = ?", (capture_id,))
            for detection in detections:
                composite = detection["composite"]
                original_rect = detection.get("original")
                preview_rect = detection.get("preview")
                connection.execute(
                    """
                    INSERT INTO detection(
                        capture_id, detection_index, region, confidence,
                        composite_x, composite_y, composite_width, composite_height,
                        original_x, original_y, original_width, original_height,
                        preview_x, preview_y, preview_width, preview_height
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        capture_id,
                        int(detection["detectionIndex"]),
                        str(detection["region"]),
                        float(detection["confidence"]),
                        float(composite["x"]),
                        float(composite["y"]),
                        float(composite["width"]),
                        float(composite["height"]),
                        None if original_rect is None else float(original_rect["x"]),
                        None if original_rect is None else float(original_rect["y"]),
                        None if original_rect is None else float(original_rect["width"]),
                        None if original_rect is None else float(original_rect["height"]),
                        None if preview_rect is None else float(preview_rect["x"]),
                        None if preview_rect is None else float(preview_rect["y"]),
                        None if preview_rect is None else float(preview_rect["width"]),
                        None if preview_rect is None else float(preview_rect["height"]),
                    ),
                )
            connection.execute(
                """
                INSERT INTO detection_refresh(
                    capture_id, model_sha256, model_name,
                    confidence_threshold, nms_iou_threshold, provider, refreshed_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(capture_id) DO UPDATE SET
                    model_sha256 = excluded.model_sha256,
                    model_name = excluded.model_name,
                    confidence_threshold = excluded.confidence_threshold,
                    nms_iou_threshold = excluded.nms_iou_threshold,
                    provider = excluded.provider,
                    refreshed_at = excluded.refreshed_at
                """,
                (
                    capture_id,
                    model_sha256,
                    model_name,
                    confidence_threshold,
                    nms_iou_threshold,
                    provider,
                    _now_iso(),
                ),
            )
            return True

    def save_annotation(
        self,
        capture_id: str,
        status: str,
        schema_version: int,
        document: dict[str, Any],
    ) -> None:
        with self.connect() as connection:
            if connection.execute(
                "SELECT 1 FROM capture WHERE id = ?",
                (capture_id,),
            ).fetchone() is None:
                raise KeyError(capture_id)
            connection.execute(
                """
                INSERT INTO capture_annotation(
                    capture_id, status, schema_version, annotation_json, updated_at
                ) VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(capture_id) DO UPDATE SET
                    status = excluded.status,
                    schema_version = excluded.schema_version,
                    annotation_json = excluded.annotation_json,
                    updated_at = excluded.updated_at
                """,
                (
                    capture_id,
                    status,
                    schema_version,
                    json.dumps(document, ensure_ascii=False, sort_keys=True),
                    _now_iso(),
                ),
            )

    def replace_layout_hand_expectation(
        self,
        campaign_id: str,
        layout_id: str,
        tile_codes: list[str],
        *,
        correction_reason: str,
    ) -> dict[str, Any]:
        if not tile_codes:
            raise ValueError("At least one hand tile is required")
        unknown = sorted(set(tile_codes) - set(VISIBLE_TILE_CODES))
        if unknown:
            raise ValueError(f"Unknown hand tile codes: {unknown}")

        applied_at = _now_iso()
        with self.connect() as connection:
            rows = connection.execute(
                """
                SELECT id, task_json
                FROM capture_task
                WHERE campaign_id = ? AND layout_id = ?
                ORDER BY environment_ordinal
                """,
                (campaign_id, layout_id),
            ).fetchall()
            if not rows:
                raise KeyError(f"Unknown layout: {campaign_id}/{layout_id}")

            changed_tasks = 0
            previous_hands: set[tuple[str, ...]] = set()
            for row in rows:
                task_id = str(row["id"])
                task = json.loads(row["task_json"])
                previous_hand = tuple(str(slot["tile"]) for slot in task["hand"])
                previous_hands.add(previous_hand)
                if previous_hand == tuple(tile_codes):
                    continue

                replacement = [
                    {
                        "ordinal": ordinal,
                        "tile": tile_code,
                        "face": "front",
                        "rotation": 0,
                    }
                    for ordinal, tile_code in enumerate(tile_codes)
                ]
                task["hand"] = replacement
                task["expected"]["hand"] = len(replacement)
                task["captureCorrection"] = {
                    "kind": "replace-expected-hand",
                    "reason": correction_reason,
                    "previousHand": list(previous_hand),
                    "replacementHand": list(tile_codes),
                    "appliedAt": applied_at,
                }

                connection.execute(
                    """
                    UPDATE capture_task
                    SET expected_hand = ?, task_json = ?
                    WHERE id = ?
                    """,
                    (
                        len(replacement),
                        json.dumps(task, ensure_ascii=False, sort_keys=True),
                        task_id,
                    ),
                )
                connection.execute(
                    "DELETE FROM task_tile_slot WHERE task_id = ? AND region = 'hand'",
                    (task_id,),
                )
                connection.executemany(
                    """
                    INSERT INTO task_tile_slot(
                        slot_key, task_id, region, row_ordinal, group_ordinal,
                        tile_ordinal, tile_code, face, rotation
                    ) VALUES (?, ?, 'hand', NULL, NULL, ?, ?, 'front', 0)
                    """,
                    [
                        (
                            f"{task_id}:hand:-:-:{ordinal}",
                            task_id,
                            ordinal,
                            tile_code,
                        )
                        for ordinal, tile_code in enumerate(tile_codes)
                    ],
                )
                changed_tasks += 1

            downgraded_annotations = 0
            if changed_tasks > 0:
                downgraded_annotations = connection.execute(
                    """
                    UPDATE capture_annotation
                    SET status = 'draft', updated_at = ?
                    WHERE status = 'complete'
                      AND capture_id IN (
                          SELECT capture.id
                          FROM capture
                          JOIN capture_task ON capture_task.id = capture.task_id
                          WHERE capture_task.campaign_id = ?
                            AND capture_task.layout_id = ?
                      )
                    """,
                    (applied_at, campaign_id, layout_id),
                ).rowcount

            return {
                "campaignId": campaign_id,
                "layoutId": layout_id,
                "taskCount": len(rows),
                "changedTaskCount": changed_tasks,
                "downgradedAnnotationCount": int(downgraded_annotations),
                "previousHands": [list(hand) for hand in sorted(previous_hands)],
                "replacementHand": list(tile_codes),
            }

    def replace_layout_meld_tile_expectation(
        self,
        campaign_id: str,
        layout_id: str,
        meld_ordinal: int,
        tile_codes: list[str],
        *,
        expected_previous_tile_codes: list[str],
        correction_reason: str,
    ) -> dict[str, Any]:
        if meld_ordinal < 0:
            raise ValueError("Meld ordinal must be non-negative")
        if not tile_codes:
            raise ValueError("At least one meld tile is required")
        if len(tile_codes) != len(expected_previous_tile_codes):
            raise ValueError("Replacement and previous melds must have the same size")
        unknown = sorted(
            (set(tile_codes) | set(expected_previous_tile_codes))
            - set(VISIBLE_TILE_CODES)
        )
        if unknown:
            raise ValueError(f"Unknown meld tile codes: {unknown}")

        replacement = tuple(tile_codes)
        expected_previous = tuple(expected_previous_tile_codes)
        applied_at = _now_iso()
        with self.connect() as connection:
            rows = connection.execute(
                """
                SELECT id, task_json
                FROM capture_task
                WHERE campaign_id = ? AND layout_id = ?
                ORDER BY environment_ordinal
                """,
                (campaign_id, layout_id),
            ).fetchall()
            if not rows:
                raise KeyError(f"Unknown layout: {campaign_id}/{layout_id}")

            changed_tasks = 0
            observed_melds: set[tuple[str, ...]] = set()
            for row in rows:
                task_id = str(row["id"])
                task = json.loads(row["task_json"])
                matching_melds = [
                    meld
                    for meld in task["melds"]
                    if int(meld["ordinal"]) == meld_ordinal
                ]
                if len(matching_melds) != 1:
                    raise ValueError(
                        f"{task_id} has {len(matching_melds)} melds with ordinal "
                        f"{meld_ordinal}"
                    )
                meld = matching_melds[0]
                ordered_slots = sorted(
                    meld["tiles"],
                    key=lambda slot: int(slot["ordinal"]),
                )
                current = tuple(str(slot["tile"]) for slot in ordered_slots)
                observed_melds.add(current)
                if current == replacement:
                    continue
                if current != expected_previous:
                    raise ValueError(
                        f"Refusing to change unexpected meld in {task_id}: "
                        f"expected {list(expected_previous)}, found {list(current)}"
                    )

                for slot, tile_code in zip(ordered_slots, replacement):
                    slot["tile"] = tile_code
                task["captureCorrection"] = {
                    "kind": "replace-expected-meld-tiles",
                    "reason": correction_reason,
                    "meldOrdinal": meld_ordinal,
                    "previousTiles": list(expected_previous),
                    "replacementTiles": list(replacement),
                    "appliedAt": applied_at,
                }
                connection.execute(
                    "UPDATE capture_task SET task_json = ? WHERE id = ?",
                    (
                        json.dumps(task, ensure_ascii=False, sort_keys=True),
                        task_id,
                    ),
                )
                for slot, tile_code in zip(ordered_slots, replacement):
                    updated = connection.execute(
                        """
                        UPDATE task_tile_slot
                        SET tile_code = ?
                        WHERE task_id = ?
                          AND region = 'meld'
                          AND group_ordinal = ?
                          AND tile_ordinal = ?
                        """,
                        (
                            tile_code,
                            task_id,
                            meld_ordinal,
                            int(slot["ordinal"]),
                        ),
                    ).rowcount
                    if updated != 1:
                        raise RuntimeError(
                            f"Expected one normalized meld slot for {task_id}, "
                            f"meld {meld_ordinal}, tile {slot['ordinal']}; updated {updated}"
                        )
                changed_tasks += 1

            return {
                "campaignId": campaign_id,
                "layoutId": layout_id,
                "meldOrdinal": meld_ordinal,
                "taskCount": len(rows),
                "changedTaskCount": changed_tasks,
                "preservedAnnotationStatuses": True,
                "observedMelds": [list(meld) for meld in sorted(observed_melds)],
                "replacementMeld": list(replacement),
            }

    def replace_layout_dora_tile_expectation(
        self,
        campaign_id: str,
        layout_id: str,
        row_key: str,
        tile_ordinal: int,
        replacement_tile_code: str,
        *,
        expected_previous_tile_code: str,
        correction_reason: str,
    ) -> dict[str, Any]:
        if row_key not in {"visible", "ura"}:
            raise ValueError("Dora row must be 'visible' or 'ura'")
        if tile_ordinal < 0:
            raise ValueError("Dora tile ordinal must be non-negative")
        unknown = sorted(
            {replacement_tile_code, expected_previous_tile_code}
            - set(VISIBLE_TILE_CODES)
        )
        if unknown:
            raise ValueError(f"Unknown dora tile codes: {unknown}")

        region = "dora-visible" if row_key == "visible" else "dora-ura"
        row_ordinal = 0 if row_key == "visible" else 1
        applied_at = _now_iso()
        with self.connect() as connection:
            rows = connection.execute(
                """
                SELECT id, task_json
                FROM capture_task
                WHERE campaign_id = ? AND layout_id = ?
                ORDER BY environment_ordinal
                """,
                (campaign_id, layout_id),
            ).fetchall()
            if not rows:
                raise KeyError(f"Unknown layout: {campaign_id}/{layout_id}")

            changed_tasks = 0
            observed_tiles: set[str] = set()
            for row in rows:
                task_id = str(row["id"])
                task = json.loads(row["task_json"])
                matching_slots = [
                    slot
                    for slot in task["dora"][row_key]
                    if int(slot["ordinal"]) == tile_ordinal
                ]
                if len(matching_slots) != 1:
                    raise ValueError(
                        f"{task_id} has {len(matching_slots)} dora {row_key} slots "
                        f"with ordinal {tile_ordinal}"
                    )
                slot = matching_slots[0]
                current = str(slot["tile"])
                observed_tiles.add(current)
                if current == replacement_tile_code:
                    continue
                if current != expected_previous_tile_code:
                    raise ValueError(
                        f"Refusing to change unexpected dora tile in {task_id}: "
                        f"expected {expected_previous_tile_code}, found {current}"
                    )

                slot["tile"] = replacement_tile_code
                correction = {
                    "kind": "replace-expected-dora-tile",
                    "reason": correction_reason,
                    "row": row_key,
                    "tileOrdinal": tile_ordinal,
                    "previousTile": expected_previous_tile_code,
                    "replacementTile": replacement_tile_code,
                    "appliedAt": applied_at,
                }
                corrections = task.get("captureCorrections")
                if not isinstance(corrections, list):
                    corrections = []
                corrections.append(correction)
                task["captureCorrections"] = corrections

                connection.execute(
                    "UPDATE capture_task SET task_json = ? WHERE id = ?",
                    (
                        json.dumps(task, ensure_ascii=False, sort_keys=True),
                        task_id,
                    ),
                )
                updated = connection.execute(
                    """
                    UPDATE task_tile_slot
                    SET tile_code = ?
                    WHERE task_id = ?
                      AND region = ?
                      AND row_ordinal = ?
                      AND tile_ordinal = ?
                    """,
                    (
                        replacement_tile_code,
                        task_id,
                        region,
                        row_ordinal,
                        tile_ordinal,
                    ),
                ).rowcount
                if updated != 1:
                    raise RuntimeError(
                        f"Expected one normalized dora slot for {task_id}, "
                        f"{row_key} tile {tile_ordinal}; updated {updated}"
                    )
                changed_tasks += 1

            return {
                "campaignId": campaign_id,
                "layoutId": layout_id,
                "row": row_key,
                "tileOrdinal": tile_ordinal,
                "taskCount": len(rows),
                "changedTaskCount": changed_tasks,
                "preservedAnnotationStatuses": True,
                "observedTiles": sorted(observed_tiles),
                "replacementTile": replacement_tile_code,
            }

    def next_task_id(self, campaign_id: str) -> str | None:
        with self.connect() as connection:
            row = connection.execute(
                """
                SELECT id FROM capture_task
                WHERE campaign_id = ? AND status = 'pending'
                ORDER BY task_order LIMIT 1
                """,
                (campaign_id,),
            ).fetchone()
            return None if row is None else str(row["id"])


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()
