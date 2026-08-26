from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path

from tools.recognition.capture_dataset_api.campaign import CAMPAIGN_ID, generate_campaign
from tools.recognition.capture_dataset_api.database import CaptureDatabase
from tools.recognition.capture_dataset_api.tile_catalog_campaign import (
    CAMPAIGN_ID as TILE_CATALOG_CAMPAIGN_ID,
    generate_tile_catalog_campaign,
)


class DatabaseTest(unittest.TestCase):
    def test_initial_and_tile_catalog_campaigns_seed_together(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            database = CaptureDatabase(Path(temporary_directory) / "dataset.sqlite")
            database.seed_campaign(generate_campaign())
            database.seed_campaign(generate_tile_catalog_campaign())

            initial = database.campaign_overview(CAMPAIGN_ID)
            catalog = database.campaign_overview(TILE_CATALOG_CAMPAIGN_ID)
            self.assertIsNotNone(initial)
            self.assertIsNotNone(catalog)
            assert initial is not None
            assert catalog is not None
            self.assertEqual(120, initial["totalTasks"])
            self.assertEqual(4, catalog["totalTasks"])

            with database.connect() as connection:
                rows = connection.execute(
                    """
                    SELECT brightness, shadow, repetition, task_order
                    FROM capture_task
                    WHERE campaign_id = ?
                    ORDER BY environment_ordinal
                    """,
                    (TILE_CATALOG_CAMPAIGN_ID,),
                ).fetchall()
            self.assertEqual(4, len(rows))
            self.assertEqual([0, 1, 2, 3], [row["repetition"] for row in rows])
            self.assertEqual([120, 121, 122, 123], [row["task_order"] for row in rows])

    def test_annotation_round_trip_and_progress(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            database = CaptureDatabase(Path(temporary_directory) / "dataset.sqlite")
            campaign = generate_campaign()
            database.seed_campaign(campaign)
            task = campaign["tasks"][0]
            database.insert_capture(
                "cap_annotation",
                {
                    "uploadClientId": "00000000-0000-4000-8000-000000000077",
                    "taskId": task["id"],
                    "campaignId": CAMPAIGN_ID,
                    "capturedAt": "2026-08-05T08:00:00+00:00",
                    "original": {"width": 1920, "height": 1080},
                    "preview": {},
                    "model": {"name": "model.onnx", "sha256": "a" * 64},
                    "layoutVersion": "PRODUCT-ADR-RECOGNITION-002-v1",
                    "confidenceThreshold": 0.3,
                    "nmsIouThreshold": 0.6,
                    "provider": "webgl",
                    "camera": {},
                    "telemetry": {},
                    "regionRects": {},
                    "detections": [],
                },
                {
                    "original": "originals/cap_annotation.jpg",
                    "composite": "composites/cap_annotation.png",
                    "hand_crop": "regions/hand/cap_annotation.png",
                    "dora_crop": "regions/dora/cap_annotation.png",
                    "meld_crop": None,
                },
            )
            document = {
                "schemaVersion": 1,
                "captureId": "cap_annotation",
                "boxes": {
                    "completed_hand": [],
                    "dora_indicators": [],
                    "melds": [],
                },
            }
            database.save_annotation("cap_annotation", "draft", 1, document)

            campaigns = database.annotation_campaigns()
            self.assertEqual(1, len(campaigns))
            self.assertEqual(1, campaigns[0]["captureCount"])
            self.assertEqual(1, campaigns[0]["draftCount"])

            captures = database.annotation_capture_list(CAMPAIGN_ID)
            self.assertEqual("draft", captures[0]["annotationStatus"])
            detail = database.annotation_capture("cap_annotation")
            self.assertIsNotNone(detail)
            assert detail is not None
            self.assertEqual(document, detail["annotation"]["document"])

            database.save_annotation("cap_annotation", "complete", 1, document)
            updated = database.annotation_capture_list(CAMPAIGN_ID)
            self.assertEqual("complete", updated[0]["annotationStatus"])

    def test_undo_last_capture_reopens_task(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            database = CaptureDatabase(Path(temporary_directory) / "dataset.sqlite")
            campaign = generate_campaign()
            database.seed_campaign(campaign)
            first_task = campaign["tasks"][0]
            database.insert_capture(
                "cap_undo",
                {
                    "uploadClientId": "00000000-0000-4000-8000-000000000088",
                    "taskId": first_task["id"],
                    "campaignId": CAMPAIGN_ID,
                    "capturedAt": "2026-08-05T08:00:00+00:00",
                    "original": {"width": 1920, "height": 1080},
                    "preview": {},
                    "model": {"name": "model.onnx", "sha256": "a" * 64},
                    "layoutVersion": "PRODUCT-ADR-RECOGNITION-002-v1",
                    "confidenceThreshold": 0.3,
                    "nmsIouThreshold": 0.6,
                    "provider": "webgl",
                    "camera": {},
                    "telemetry": {},
                    "regionRects": {},
                    "detections": [],
                },
                {
                    "original": "originals/cap_undo.jpg",
                    "composite": "composites/cap_undo.png",
                    "hand_crop": "regions/hand/cap_undo.png",
                    "dora_crop": "regions/dora/cap_undo.png",
                    "meld_crop": None,
                },
            )

            undone = database.undo_last_capture(CAMPAIGN_ID)
            self.assertIsNotNone(undone)
            assert undone is not None
            self.assertEqual("cap_undo", undone["captureId"])
            self.assertEqual(first_task["id"], undone["taskId"])
            self.assertIn("originals/cap_undo.jpg", undone["paths"])

            overview = database.campaign_overview(CAMPAIGN_ID)
            self.assertIsNotNone(overview)
            assert overview is not None
            self.assertEqual(0, overview["completedTasks"])
            reopened = database.next_task(CAMPAIGN_ID)
            self.assertIsNotNone(reopened)
            assert reopened is not None
            self.assertEqual(first_task["id"], reopened["id"])

    def test_new_campaign_coexists_with_captured_legacy_campaign(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            database = CaptureDatabase(Path(temporary_directory) / "dataset.sqlite")
            current_campaign = generate_campaign()
            legacy_task = copy.deepcopy(current_campaign["tasks"][0])
            legacy_task["id"] = "initial-240-layout-001-bright-none-r1"
            legacy_task["campaignId"] = "initial-240"
            legacy_campaign = {
                "id": "initial-240",
                "name": "Legacy capture campaign",
                "definitionSha256": "0" * 64,
                "tasks": [legacy_task],
            }
            database.seed_campaign(legacy_campaign)
            database.insert_capture(
                "cap_legacy",
                {
                    "uploadClientId": "00000000-0000-4000-8000-000000000099",
                    "taskId": legacy_task["id"],
                    "campaignId": "initial-240",
                    "capturedAt": "2026-08-05T08:00:00+00:00",
                    "original": {"width": 1920, "height": 1080},
                    "preview": {},
                    "model": {"name": "legacy.onnx", "sha256": "b" * 64},
                    "layoutVersion": "PRODUCT-ADR-RECOGNITION-002-v1",
                    "confidenceThreshold": 0.3,
                    "nmsIouThreshold": 0.6,
                    "provider": "webgl",
                    "camera": {},
                    "telemetry": {},
                    "regionRects": {},
                    "detections": [],
                },
                {
                    "original": "originals/cap_legacy.jpg",
                    "composite": "composites/cap_legacy.png",
                    "hand_crop": "regions/hand/cap_legacy.png",
                    "dora_crop": "regions/dora/cap_legacy.png",
                    "meld_crop": None,
                },
            )

            database.seed_campaign(current_campaign)

            legacy_overview = database.campaign_overview("initial-240")
            current_overview = database.campaign_overview(CAMPAIGN_ID)
            self.assertIsNotNone(legacy_overview)
            self.assertIsNotNone(current_overview)
            assert legacy_overview is not None
            assert current_overview is not None
            self.assertEqual(1, legacy_overview["completedTasks"])
            self.assertEqual(120, current_overview["totalTasks"])
            self.assertEqual(0, current_overview["completedTasks"])

    def test_seed_and_next_task(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            database = CaptureDatabase(Path(temporary_directory) / "dataset.sqlite")
            campaign = generate_campaign()
            database.seed_campaign(campaign)
            database.seed_campaign(campaign)

            overview = database.campaign_overview(CAMPAIGN_ID)
            self.assertIsNotNone(overview)
            assert overview is not None
            self.assertEqual(120, overview["totalTasks"])
            self.assertEqual(0, overview["completedTasks"])
            self.assertEqual(30, overview["totalLayouts"])

            next_task = database.next_task(CAMPAIGN_ID)
            self.assertIsNotNone(next_task)
            assert next_task is not None
            self.assertEqual(0, next_task["taskOrder"])

            manifest = {
                "uploadClientId": "00000000-0000-4000-8000-000000000001",
                "taskId": next_task["id"],
                "campaignId": CAMPAIGN_ID,
                "capturedAt": "2026-08-05T08:00:00+00:00",
                "original": {"width": 1920, "height": 1080},
                "preview": {
                    "width": 844,
                    "height": 390,
                    "devicePixelRatio": 3,
                    "videoElement": {"x": 0, "y": 0, "width": 844, "height": 390},
                    "sourceToDisplayScale": 0.4395833333,
                    "sourceDisplayOffsetX": 0,
                    "sourceDisplayOffsetY": -42.375,
                },
                "model": {"name": "model.onnx", "sha256": "a" * 64},
                "layoutVersion": "PRODUCT-ADR-RECOGNITION-002-v1",
                "confidenceThreshold": 0.3,
                "nmsIouThreshold": 0.6,
                "provider": "webgl",
                "camera": {},
                "telemetry": {},
                "regionRects": {},
                "detections": [],
            }
            database.insert_capture(
                "cap_test",
                manifest,
                {
                    "original": "originals/cap_test.jpg",
                    "composite": "composites/cap_test.png",
                    "hand_crop": "regions/hand/cap_test.jpg",
                    "dora_crop": "regions/dora/cap_test.jpg",
                    "meld_crop": None,
                },
            )
            updated = database.campaign_overview(CAMPAIGN_ID)
            self.assertIsNotNone(updated)
            assert updated is not None
            self.assertEqual(1, updated["completedTasks"])
            expected_back_count = sum(
                1
                for meld in next_task["melds"]
                for slot in meld["tiles"]
                if slot["face"] == "back"
            )
            self.assertEqual(expected_back_count, updated["coverage"]["back"])
            following_task = database.next_task(CAMPAIGN_ID)
            self.assertIsNotNone(following_task)
            assert following_task is not None
            self.assertEqual(1, following_task["taskOrder"])

            with database.connect() as connection:
                visible_classes = {
                    row["visible_class"]
                    for row in connection.execute(
                        "SELECT visible_class FROM capture_expected_tile_slot WHERE capture_id = ?",
                        ("cap_test",),
                    ).fetchall()
                }
            self.assertTrue(visible_classes)
            if expected_back_count > 0:
                self.assertIn("back", visible_classes)


    def test_detection_refresh_only_replaces_unannotated_capture(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            database = CaptureDatabase(Path(temporary_directory) / "dataset.sqlite")
            campaign = generate_campaign()
            database.seed_campaign(campaign)
            tasks = campaign["tasks"][:2]

            for index, task in enumerate(tasks):
                capture_id = f"cap_refresh_{index}"
                database.insert_capture(
                    capture_id,
                    {
                        "uploadClientId": f"00000000-0000-4000-8000-{index + 1:012d}",
                        "taskId": task["id"],
                        "campaignId": CAMPAIGN_ID,
                        "capturedAt": "2026-08-05T08:00:00+00:00",
                        "original": {"width": 1920, "height": 1080},
                        "preview": {},
                        "model": {"name": "old.onnx", "sha256": "a" * 64},
                        "layoutVersion": "PRODUCT-ADR-RECOGNITION-002-v1",
                        "confidenceThreshold": 0.3,
                        "nmsIouThreshold": 0.6,
                        "provider": "webgl",
                        "camera": {},
                        "telemetry": {},
                        "regionRects": {},
                        "detections": [
                            {
                                "detectionIndex": 0,
                                "region": "invalid",
                                "confidence": 0.2,
                                "composite": {"x": 1, "y": 1, "width": 2, "height": 2},
                                "original": None,
                                "preview": None,
                            }
                        ],
                    },
                    {
                        "original": f"originals/{capture_id}.jpg",
                        "composite": f"composites/{capture_id}.png",
                        "hand_crop": f"regions/hand/{capture_id}.png",
                        "dora_crop": f"regions/dora/{capture_id}.png",
                        "meld_crop": None,
                    },
                )

            draft_document = {
                "schemaVersion": 1,
                "captureId": "cap_refresh_1",
                "boxes": {
                    "completed_hand": [],
                    "dora_indicators": [],
                    "melds": [],
                },
            }
            database.save_annotation("cap_refresh_1", "draft", 1, draft_document)

            selected = database.unannotated_captures(CAMPAIGN_ID)
            self.assertEqual(["cap_refresh_0"], [capture["captureId"] for capture in selected])

            refreshed_detection = {
                "detectionIndex": 0,
                "region": "invalid",
                "confidence": 0.9,
                "composite": {"x": 5, "y": 6, "width": 7, "height": 8},
                "original": None,
                "preview": None,
            }
            self.assertTrue(
                database.replace_unannotated_detections(
                    "cap_refresh_0",
                    [refreshed_detection],
                    model_sha256="b" * 64,
                    model_name="new.onnx",
                    confidence_threshold=0.35,
                    nms_iou_threshold=0.6,
                    provider="onnxruntime-cpu-refresh",
                )
            )
            self.assertFalse(
                database.replace_unannotated_detections(
                    "cap_refresh_1",
                    [refreshed_detection],
                    model_sha256="b" * 64,
                    model_name="new.onnx",
                    confidence_threshold=0.35,
                    nms_iou_threshold=0.6,
                    provider="onnxruntime-cpu-refresh",
                )
            )
            draft_candidates = database.unannotated_captures(
                CAMPAIGN_ID,
                include_drafts=True,
            )
            self.assertEqual(
                ["cap_refresh_0", "cap_refresh_1"],
                [capture["captureId"] for capture in draft_candidates],
            )
            self.assertTrue(
                database.replace_unannotated_detections(
                    "cap_refresh_1",
                    [refreshed_detection],
                    model_sha256="b" * 64,
                    model_name="new.onnx",
                    confidence_threshold=0.35,
                    nms_iou_threshold=0.6,
                    provider="onnxruntime-cpu-refresh",
                    allow_draft=True,
                )
            )

            with database.connect() as connection:
                unannotated_confidence = connection.execute(
                    "SELECT confidence FROM detection WHERE capture_id = 'cap_refresh_0'"
                ).fetchone()[0]
                draft_confidence = connection.execute(
                    "SELECT confidence FROM detection WHERE capture_id = 'cap_refresh_1'"
                ).fetchone()[0]
                refresh_rows = connection.execute(
                    "SELECT COUNT(*) FROM detection_refresh"
                ).fetchone()[0]
            self.assertEqual(0.9, unannotated_confidence)
            self.assertEqual(0.9, draft_confidence)
            self.assertEqual(2, refresh_rows)
            detail = database.annotation_capture("cap_refresh_1")
            self.assertIsNotNone(detail)
            assert detail is not None
            self.assertEqual("draft", detail["annotation"]["status"])
            self.assertEqual(draft_document, detail["annotation"]["document"])

    def test_replace_layout_hand_expectation_updates_all_environments(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            database = CaptureDatabase(Path(temporary_directory) / "dataset.sqlite")
            campaign = generate_campaign()
            database.seed_campaign(campaign)
            task = campaign["tasks"][23 * 4]
            database.insert_capture(
                "cap_layout_024",
                {
                    "uploadClientId": "00000000-0000-4000-8000-000000000024",
                    "taskId": task["id"],
                    "campaignId": CAMPAIGN_ID,
                    "capturedAt": "2026-08-06T08:00:00+00:00",
                    "original": {"width": 1920, "height": 1080},
                    "preview": {},
                    "model": {"name": "model.onnx", "sha256": "a" * 64},
                    "layoutVersion": "PRODUCT-ADR-RECOGNITION-002-v1",
                    "confidenceThreshold": 0.3,
                    "nmsIouThreshold": 0.6,
                    "provider": "webgl",
                    "camera": {},
                    "telemetry": {},
                    "regionRects": {},
                    "detections": [],
                },
                {
                    "original": "originals/cap_layout_024.jpg",
                    "composite": "composites/cap_layout_024.png",
                    "hand_crop": "regions/hand/cap_layout_024.png",
                    "dora_crop": "regions/dora/cap_layout_024.png",
                    "meld_crop": "regions/meld/cap_layout_024.png",
                },
            )
            document = {
                "schemaVersion": 1,
                "captureId": "cap_layout_024",
                "boxes": {
                    "completed_hand": [],
                    "dora_indicators": [],
                    "melds": [],
                },
            }
            database.save_annotation("cap_layout_024", "complete", 1, document)

            corrected_hand = ["5p", "5p", "5p", "5p", "6p", "6p"]
            result = database.replace_layout_hand_expectation(
                CAMPAIGN_ID,
                "layout-024",
                corrected_hand,
                correction_reason="Captured one additional 5p.",
            )
            self.assertEqual(4, result["taskCount"])
            self.assertEqual(4, result["changedTaskCount"])
            self.assertEqual(1, result["downgradedAnnotationCount"])

            with database.connect() as connection:
                rows = connection.execute(
                    """
                    SELECT id, expected_hand, task_json
                    FROM capture_task
                    WHERE campaign_id = ? AND layout_id = 'layout-024'
                    ORDER BY environment_ordinal
                    """,
                    (CAMPAIGN_ID,),
                ).fetchall()
                self.assertEqual(4, len(rows))
                for row in rows:
                    self.assertEqual(6, row["expected_hand"])
                    updated_task = json.loads(row["task_json"])
                    self.assertEqual(
                        corrected_hand,
                        [slot["tile"] for slot in updated_task["hand"]],
                    )
                    hand_slots = connection.execute(
                        """
                        SELECT tile_code
                        FROM task_tile_slot
                        WHERE task_id = ? AND region = 'hand'
                        ORDER BY tile_ordinal
                        """,
                        (row["id"],),
                    ).fetchall()
                    self.assertEqual(
                        corrected_hand,
                        [slot["tile_code"] for slot in hand_slots],
                    )
                annotation_status = connection.execute(
                    """
                    SELECT status
                    FROM capture_annotation
                    WHERE capture_id = 'cap_layout_024'
                    """
                ).fetchone()[0]
            self.assertEqual("draft", annotation_status)

            repeated = database.replace_layout_hand_expectation(
                CAMPAIGN_ID,
                "layout-024",
                corrected_hand,
                correction_reason="Captured one additional 5p.",
            )
            self.assertEqual(0, repeated["changedTaskCount"])
            self.assertEqual(0, repeated["downgradedAnnotationCount"])

    def test_replace_layout_meld_tile_expectation_preserves_geometry_and_status(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            database = CaptureDatabase(Path(temporary_directory) / "dataset.sqlite")
            campaign = generate_campaign()
            database.seed_campaign(campaign)
            task = campaign["tasks"][28 * 4]
            self.assertEqual(
                ["6m", "7m", "8m"],
                [slot["tile"] for slot in task["melds"][0]["tiles"]],
            )
            original_rotations = [
                slot["rotation"] for slot in task["melds"][0]["tiles"]
            ]
            database.insert_capture(
                "cap_layout_029",
                {
                    "uploadClientId": "00000000-0000-4000-8000-000000000029",
                    "taskId": task["id"],
                    "campaignId": CAMPAIGN_ID,
                    "capturedAt": "2026-08-06T08:00:00+00:00",
                    "original": {"width": 1920, "height": 1080},
                    "preview": {},
                    "model": {"name": "model.onnx", "sha256": "a" * 64},
                    "layoutVersion": "PRODUCT-ADR-RECOGNITION-002-v1",
                    "confidenceThreshold": 0.3,
                    "nmsIouThreshold": 0.6,
                    "provider": "webgl",
                    "camera": {},
                    "telemetry": {},
                    "regionRects": {},
                    "detections": [],
                },
                {
                    "original": "originals/cap_layout_029.jpg",
                    "composite": "composites/cap_layout_029.png",
                    "hand_crop": "regions/hand/cap_layout_029.png",
                    "dora_crop": "regions/dora/cap_layout_029.png",
                    "meld_crop": "regions/meld/cap_layout_029.png",
                },
            )
            document = {
                "schemaVersion": 1,
                "captureId": "cap_layout_029",
                "boxes": {
                    "completed_hand": [],
                    "dora_indicators": [],
                    "melds": [],
                },
            }
            database.save_annotation("cap_layout_029", "complete", 1, document)

            corrected_meld = ["6m", "8m", "7m"]
            result = database.replace_layout_meld_tile_expectation(
                CAMPAIGN_ID,
                "layout-029",
                0,
                corrected_meld,
                expected_previous_tile_codes=["6m", "7m", "8m"],
                correction_reason="The photographed 7m and 8m positions were reversed.",
            )
            self.assertEqual(4, result["taskCount"])
            self.assertEqual(4, result["changedTaskCount"])
            self.assertTrue(result["preservedAnnotationStatuses"])

            with database.connect() as connection:
                rows = connection.execute(
                    """
                    SELECT id, task_json
                    FROM capture_task
                    WHERE campaign_id = ? AND layout_id = 'layout-029'
                    ORDER BY environment_ordinal
                    """,
                    (CAMPAIGN_ID,),
                ).fetchall()
                self.assertEqual(4, len(rows))
                for row in rows:
                    updated_task = json.loads(row["task_json"])
                    updated_meld = updated_task["melds"][0]["tiles"]
                    self.assertEqual(
                        corrected_meld,
                        [slot["tile"] for slot in updated_meld],
                    )
                    self.assertEqual(
                        original_rotations,
                        [slot["rotation"] for slot in updated_meld],
                    )
                    normalized = connection.execute(
                        """
                        SELECT tile_code, rotation
                        FROM task_tile_slot
                        WHERE task_id = ?
                          AND region = 'meld'
                          AND group_ordinal = 0
                        ORDER BY tile_ordinal
                        """,
                        (row["id"],),
                    ).fetchall()
                    self.assertEqual(
                        corrected_meld,
                        [slot["tile_code"] for slot in normalized],
                    )
                    self.assertEqual(
                        original_rotations,
                        [slot["rotation"] for slot in normalized],
                    )
                annotation_status = connection.execute(
                    """
                    SELECT status
                    FROM capture_annotation
                    WHERE capture_id = 'cap_layout_029'
                    """
                ).fetchone()[0]
            self.assertEqual("complete", annotation_status)

            repeated = database.replace_layout_meld_tile_expectation(
                CAMPAIGN_ID,
                "layout-029",
                0,
                corrected_meld,
                expected_previous_tile_codes=["6m", "7m", "8m"],
                correction_reason="The photographed 7m and 8m positions were reversed.",
            )
            self.assertEqual(0, repeated["changedTaskCount"])

    def test_replace_layout_dora_tile_expectation_preserves_status(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            database = CaptureDatabase(Path(temporary_directory) / "dataset.sqlite")
            campaign = generate_campaign()
            database.seed_campaign(campaign)
            task = campaign["tasks"][23 * 4]
            original_tile = str(task["dora"]["visible"][1]["tile"])
            self.assertEqual("1s", original_tile)

            database.insert_capture(
                "cap_layout_024_dora",
                {
                    "uploadClientId": "00000000-0000-4000-8000-000000002024",
                    "taskId": task["id"],
                    "campaignId": CAMPAIGN_ID,
                    "capturedAt": "2026-08-06T08:00:00+00:00",
                    "original": {"width": 1920, "height": 1080},
                    "preview": {},
                    "model": {"name": "model.onnx", "sha256": "a" * 64},
                    "layoutVersion": "PRODUCT-ADR-RECOGNITION-002-v1",
                    "confidenceThreshold": 0.3,
                    "nmsIouThreshold": 0.6,
                    "provider": "webgl",
                    "camera": {},
                    "telemetry": {},
                    "regionRects": {},
                    "detections": [],
                },
                {
                    "original": "originals/cap_layout_024_dora.jpg",
                    "composite": "composites/cap_layout_024_dora.png",
                    "hand_crop": "regions/hand/cap_layout_024_dora.png",
                    "dora_crop": "regions/dora/cap_layout_024_dora.png",
                    "meld_crop": "regions/meld/cap_layout_024_dora.png",
                },
            )
            document = {
                "schemaVersion": 1,
                "captureId": "cap_layout_024_dora",
                "boxes": {
                    "completed_hand": [],
                    "dora_indicators": [],
                    "melds": [],
                },
            }
            database.save_annotation("cap_layout_024_dora", "complete", 1, document)

            result = database.replace_layout_dora_tile_expectation(
                CAMPAIGN_ID,
                "layout-024",
                "visible",
                1,
                "1p",
                expected_previous_tile_code="1s",
                correction_reason="The photographed indicator was 1p, not 1s.",
            )
            self.assertEqual(4, result["taskCount"])
            self.assertEqual(4, result["changedTaskCount"])
            self.assertTrue(result["preservedAnnotationStatuses"])

            with database.connect() as connection:
                rows = connection.execute(
                    """
                    SELECT id, task_json
                    FROM capture_task
                    WHERE campaign_id = ? AND layout_id = 'layout-024'
                    ORDER BY environment_ordinal
                    """,
                    (CAMPAIGN_ID,),
                ).fetchall()
                self.assertEqual(4, len(rows))
                for row in rows:
                    updated_task = json.loads(row["task_json"])
                    self.assertEqual("1p", updated_task["dora"]["visible"][1]["tile"])
                    normalized = connection.execute(
                        """
                        SELECT tile_code
                        FROM task_tile_slot
                        WHERE task_id = ?
                          AND region = 'dora-visible'
                          AND row_ordinal = 0
                          AND tile_ordinal = 1
                        """,
                        (row["id"],),
                    ).fetchone()
                    self.assertIsNotNone(normalized)
                    assert normalized is not None
                    self.assertEqual("1p", normalized["tile_code"])
                annotation_status = connection.execute(
                    """
                    SELECT status
                    FROM capture_annotation
                    WHERE capture_id = 'cap_layout_024_dora'
                    """
                ).fetchone()[0]
            self.assertEqual("complete", annotation_status)

            repeated = database.replace_layout_dora_tile_expectation(
                CAMPAIGN_ID,
                "layout-024",
                "visible",
                1,
                "1p",
                expected_previous_tile_code="1s",
                correction_reason="The photographed indicator was 1p, not 1s.",
            )
            self.assertEqual(0, repeated["changedTaskCount"])


if __name__ == "__main__":
    unittest.main()
