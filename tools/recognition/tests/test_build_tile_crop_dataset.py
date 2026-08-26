from __future__ import annotations

import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from tools.recognition.build_tile_crop_dataset import (
    JP_NUMERIC_TILE_LABELS,
    SCHEMA,
    assign_manual_boxes,
    build_jp_split,
    build_manual_source,
    configure_output_database,
    crop_axis_aligned_bbox,
    extract_rotated_crop,
    normalize_jp_tile_label,
)


class NormalizeJpTileLabelTests(unittest.TestCase):
    def test_numeric_labels_follow_suit_major_order_with_red_fives(self) -> None:
        self.assertEqual(37, len(JP_NUMERIC_TILE_LABELS))
        self.assertEqual("1m", normalize_jp_tile_label("0"))
        self.assertEqual("5m", normalize_jp_tile_label("4"))
        self.assertEqual("red5m", normalize_jp_tile_label("5"))
        self.assertEqual("1p", normalize_jp_tile_label("10"))
        self.assertEqual("red5p", normalize_jp_tile_label("15"))
        self.assertEqual("1s", normalize_jp_tile_label("20"))
        self.assertEqual("red5s", normalize_jp_tile_label("25"))
        self.assertEqual("east", normalize_jp_tile_label("30"))
        self.assertEqual("south", normalize_jp_tile_label("31"))
        self.assertEqual("west", normalize_jp_tile_label("32"))
        self.assertEqual("north", normalize_jp_tile_label("33"))
        self.assertEqual("white", normalize_jp_tile_label("34"))
        self.assertEqual("green", normalize_jp_tile_label("35"))
        self.assertEqual("red", normalize_jp_tile_label("36"))

    def test_explicit_red_five_aliases_use_mjtensu_codes(self) -> None:
        self.assertEqual("red5m", normalize_jp_tile_label("5mr"))
        self.assertEqual("red5p", normalize_jp_tile_label("5pr"))
        self.assertEqual("red5s", normalize_jp_tile_label("5sr"))

    def test_supercategory_is_not_a_tile(self) -> None:
        self.assertIsNone(normalize_jp_tile_label("mahjong-tiles"))


class ManualAssignmentTests(unittest.TestCase):
    def test_hand_boxes_are_assigned_left_to_right(self) -> None:
        task = {
            "id": "task-1",
            "hand": [
                {"ordinal": 0, "tile": "1m", "face": "front", "rotation": 0},
                {"ordinal": 1, "tile": "2m", "face": "front", "rotation": 0},
            ],
            "dora": {"visible": [], "ura": []},
            "melds": [],
        }
        boxes = [
            _box("right", 30, 10),
            _box("left", 10, 10),
        ]

        assigned = assign_manual_boxes(task, "completed_hand", boxes)

        self.assertEqual(["left", "right"], [item.box["id"] for item in assigned])
        self.assertEqual(["1m", "2m"], [item.slot.tile_label for item in assigned])

    def test_catalog_rows_follow_sloped_row_direction(self) -> None:
        task = {
            "id": "catalog-task",
            "campaignId": "tile-catalog-warm-4-v2",
            "hand": [],
            "dora": {"visible": [], "ura": []},
            "melds": [
                {
                    "ordinal": 0,
                    "kind": "catalog-row",
                    "tiles": [
                        {"ordinal": 0, "tile": "1m", "face": "front", "rotation": 0},
                        {"ordinal": 1, "tile": "2m", "face": "front", "rotation": 0},
                    ],
                },
                {
                    "ordinal": 1,
                    "kind": "catalog-row",
                    "tiles": [
                        {"ordinal": 0, "tile": "1p", "face": "front", "rotation": 0},
                        {"ordinal": 1, "tile": "2p", "face": "front", "rotation": 0},
                    ],
                },
            ],
        }
        boxes = [
            _box("m1", 10, 10),
            _box("m2", 30, 20),
            _box("p1", 10, 35),
            _box("p2", 30, 45),
        ]

        assigned = assign_manual_boxes(task, "melds", boxes)

        self.assertEqual(["m1", "m2", "p1", "p2"], [item.box["id"] for item in assigned])
        self.assertEqual(["1m", "2m", "1p", "2p"], [item.slot.tile_label for item in assigned])

    def test_dora_rows_are_partitioned_top_to_bottom(self) -> None:
        task = {
            "id": "task-2",
            "hand": [],
            "dora": {
                "visible": [
                    {"ordinal": 0, "tile": "east", "face": "front", "rotation": 0},
                    {"ordinal": 1, "tile": "south", "face": "front", "rotation": 0},
                ],
                "ura": [
                    {"ordinal": 0, "tile": "west", "face": "front", "rotation": 0},
                ],
            },
            "melds": [],
        }
        boxes = [
            _box("ura", 10, 40),
            _box("visible-right", 30, 10),
            _box("visible-left", 10, 10),
        ]

        assigned = assign_manual_boxes(task, "dora_indicators", boxes)

        self.assertEqual(
            ["visible-left", "visible-right", "ura"],
            [item.box["id"] for item in assigned],
        )
        self.assertEqual(
            ["east", "south", "west"],
            [item.slot.tile_label for item in assigned],
        )
        self.assertEqual(
            ["dora-visible", "dora-visible", "dora-ura"],
            [item.slot.group_name for item in assigned],
        )


class DatasetBuildTests(unittest.TestCase):
    def test_jp_split_is_built_once_and_reused(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            repository_root = Path(temporary_directory)
            split_directory = repository_root / "data" / "coco_mahjong_jp_v2" / "train"
            split_directory.mkdir(parents=True)
            Image.new("RGB", (20, 20), (240, 240, 240)).save(
                split_directory / "image.png"
            )
            (split_directory / "_annotations.coco.json").write_text(
                json.dumps(
                    {
                        "images": [
                            {
                                "id": 1,
                                "file_name": "image.png",
                                "width": 20,
                                "height": 20,
                            }
                        ],
                        "annotations": [
                            {
                                "id": 10,
                                "image_id": 1,
                                "category_id": 1,
                                "bbox": [2, 3, 5, 7],
                            },
                            {
                                "id": 11,
                                "image_id": 1,
                                "category_id": 1,
                                "bbox": [10, 8, 4, 6],
                            },
                        ],
                        "categories": [{"id": 1, "name": "0"}],
                    }
                ),
                encoding="utf-8",
            )
            connection = _output_connection(repository_root / "output.sqlite")
            try:
                first = build_jp_split(
                    connection,
                    repository_root=repository_root,
                    jp_root=repository_root / "data" / "coco_mahjong_jp_v2",
                    split="train",
                    force=False,
                    commit_interval=10,
                    workers=2,
                )
                second = build_jp_split(
                    connection,
                    repository_root=repository_root,
                    jp_root=repository_root / "data" / "coco_mahjong_jp_v2",
                    split="train",
                    force=False,
                    commit_interval=10,
                    workers=2,
                )
                connection.execute(
                    "DELETE FROM tile_crop WHERE source_annotation_id = '11'"
                )
                connection.execute(
                    """
                    UPDATE dataset_metadata
                    SET value = 'building'
                    WHERE key = 'source.jp.train.status'
                    """
                )
                connection.commit()
                resumed = build_jp_split(
                    connection,
                    repository_root=repository_root,
                    jp_root=repository_root / "data" / "coco_mahjong_jp_v2",
                    split="train",
                    force=False,
                    commit_interval=10,
                    workers=2,
                )
                row = connection.execute(
                    """
                    SELECT tile_label, image_width, image_height
                    FROM tile_crop
                    WHERE source_annotation_id = '10'
                    """
                ).fetchone()
                crop_count = connection.execute(
                    "SELECT COUNT(*) FROM tile_crop"
                ).fetchone()[0]
            finally:
                connection.close()

            self.assertEqual("rebuilt", first["action"])
            self.assertEqual("reused", second["action"])
            self.assertEqual("resumed", resumed["action"])
            self.assertEqual(1, resumed["new_crop_count"])
            self.assertEqual(2, crop_count)
            self.assertEqual(("1m", 5, 7), tuple(row))

    def test_completed_manual_annotation_is_deskewed_and_stored(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            repository_root = Path(temporary_directory)
            storage_root = repository_root / ".local" / "recognition" / "capture_dataset"
            storage_root.mkdir(parents=True)
            region_path = storage_root / "regions" / "hand.png"
            region_path.parent.mkdir(parents=True)
            Image.new("RGB", (30, 30), (250, 250, 250)).save(region_path)

            task = {
                "id": "task-1",
                "hand": [
                    {"ordinal": 0, "tile": "1p", "face": "front", "rotation": 0}
                ],
                "dora": {"visible": [], "ura": []},
                "melds": [],
            }
            annotation = {
                "schemaVersion": 1,
                "captureId": "capture-1",
                "boxes": {
                    "completed_hand": [_box("box-1", 15, 15)],
                    "dora_indicators": [],
                    "melds": [],
                },
            }
            source_database = storage_root / "dataset.sqlite"
            source = sqlite3.connect(source_database)
            try:
                source.executescript(
                    """
                    CREATE TABLE capture_task(
                        id TEXT PRIMARY KEY,
                        layout_id TEXT,
                        layout_ordinal INTEGER,
                        brightness TEXT,
                        shadow TEXT,
                        task_order INTEGER,
                        task_json TEXT
                    );
                    CREATE TABLE capture(
                        id TEXT PRIMARY KEY,
                        task_id TEXT,
                        original_path TEXT,
                        hand_crop_path TEXT,
                        dora_crop_path TEXT,
                        meld_crop_path TEXT
                    );
                    CREATE TABLE capture_annotation(
                        capture_id TEXT PRIMARY KEY,
                        status TEXT,
                        annotation_json TEXT
                    );
                    """
                )
                source.execute(
                    "INSERT INTO capture_task VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (
                        "task-1",
                        "layout-1",
                        0,
                        "bright",
                        "none",
                        0,
                        json.dumps(task),
                    ),
                )
                source.execute(
                    "INSERT INTO capture VALUES (?, ?, ?, ?, ?, ?)",
                    (
                        "capture-1",
                        "task-1",
                        "original.png",
                        "regions/hand.png",
                        None,
                        None,
                    ),
                )
                source.execute(
                    "INSERT INTO capture_annotation VALUES (?, 'complete', ?)",
                    ("capture-1", json.dumps(annotation)),
                )
                source.commit()
            finally:
                source.close()

            output = _output_connection(repository_root / "output.sqlite")
            try:
                result = build_manual_source(
                    output,
                    repository_root=repository_root,
                    storage_root=storage_root,
                    force=False,
                    commit_interval=10,
                    workers=2,
                )
                reused = build_manual_source(
                    output,
                    repository_root=repository_root,
                    storage_root=storage_root,
                    force=False,
                    commit_interval=10,
                    workers=2,
                )
                row = output.execute(
                    """
                    SELECT tile_label, source, brightness, image_width, image_height
                    FROM tile_crop
                    """
                ).fetchone()
            finally:
                output.close()

            self.assertEqual(1, result["crop_count"])
            self.assertEqual("reused", reused["action"])
            self.assertEqual(("1p", "manual", "bright", 10, 15), tuple(row))


class CropGeometryTests(unittest.TestCase):
    def test_axis_aligned_crop_rounds_outward(self) -> None:
        image = Image.new("RGB", (10, 10), (0, 0, 0))
        result = crop_axis_aligned_bbox(image, [1.2, 2.8, 3.1, 4.1])
        self.assertEqual((4, 5), result.size)

    def test_zero_angle_rotated_crop_has_requested_size_and_center(self) -> None:
        image = Image.new("RGB", (20, 20), (0, 0, 0))
        for y in range(6, 14):
            for x in range(7, 13):
                image.putpixel((x, y), (255, 0, 0))
        result = extract_rotated_crop(
            image,
            {
                "centerX": 10,
                "centerY": 10,
                "width": 6,
                "height": 8,
                "angleDeg": 0,
            },
        )
        self.assertEqual((6, 8), result.size)
        red, green, blue = result.getpixel((3, 4))
        self.assertGreater(red, 240)
        self.assertLess(green, 10)
        self.assertLess(blue, 10)


def _output_connection(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(path)
    connection.row_factory = sqlite3.Row
    configure_output_database(connection)
    connection.executescript(SCHEMA)
    return connection


def _box(box_id: str, center_x: float, center_y: float) -> dict[str, float | str]:
    return {
        "id": box_id,
        "centerX": center_x,
        "centerY": center_y,
        "width": 10,
        "height": 15,
        "angleDeg": 0,
    }


if __name__ == "__main__":
    unittest.main()
