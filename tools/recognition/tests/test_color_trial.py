from __future__ import annotations

import io
import sqlite3
import tempfile
import unittest
from contextlib import closing
from pathlib import Path

import numpy as np
from PIL import Image

from tools.recognition.build_color_trial_sample import build_sample_database
from tools.recognition.run_lab_threshold_trial import (
    Thresholds,
    classify_lab,
    rgb_to_lab,
    run_trial,
    threshold_image,
)
from tools.recognition.run_sauvola_trial import (
    SauvolaParameters,
    filter_red_ink_components,
    run_trial as run_sauvola_trial,
    threshold_image as threshold_sauvola_image,
)


SOURCE_SCHEMA = """
CREATE TABLE dataset_metadata (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE TABLE tile_crop (
    crop_id              TEXT PRIMARY KEY,
    source               TEXT NOT NULL,
    source_partition     TEXT NOT NULL,
    tile_label           TEXT NOT NULL,
    image_width          INTEGER NOT NULL,
    image_height         INTEGER NOT NULL,
    image_png            BLOB NOT NULL,
    source_image_path    TEXT NOT NULL,
    source_image_id      TEXT,
    capture_id           TEXT,
    layout_id            TEXT,
    region               TEXT,
    brightness           TEXT,
    shadow               TEXT
);
CREATE INDEX idx_tile_crop_source_label
ON tile_crop(source, tile_label);
CREATE INDEX idx_tile_crop_source_partition
ON tile_crop(source, source_partition);
"""


class ColorTrialSampleTests(unittest.TestCase):
    def test_jp_train_is_deterministically_sampled_and_manual_is_copied(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source_database = root / "source.sqlite"
            output_database = root / "sample.sqlite"
            create_source_database(source_database)

            first = build_sample_database(
                source_database=source_database,
                output_database=output_database,
                seed=42,
                jp_samples_per_label=3,
            )
            first_ids = selected_crop_ids(output_database)
            reused = build_sample_database(
                source_database=source_database,
                output_database=output_database,
                seed=42,
                jp_samples_per_label=3,
            )
            rebuilt = build_sample_database(
                source_database=source_database,
                output_database=output_database,
                seed=42,
                jp_samples_per_label=3,
                force=True,
            )
            rebuilt_ids = selected_crop_ids(output_database)

            self.assertEqual("built", first["action"])
            self.assertEqual("reused", reused["action"])
            self.assertEqual("built", rebuilt["action"])
            self.assertEqual(9, first["crop_count"])
            self.assertEqual({"jp": 6, "manual": 3}, first["counts_by_source"])
            self.assertEqual(first_ids, rebuilt_ids)
            self.assertFalse(any(crop_id.startswith("jp-valid") for crop_id in first_ids))
            self.assertEqual(
                ["manual-0", "manual-1", "manual-2"],
                sorted(crop_id for crop_id in first_ids if crop_id.startswith("manual")),
            )


class SauvolaThresholdTests(unittest.TestCase):
    def test_uneven_gray_background_keeps_relative_dark_stroke(self) -> None:
        width = 21
        height = 21
        gradient = np.linspace(80, 220, width, dtype=np.float32)
        gray = np.repeat(gradient[None, :], height, axis=0)
        gray[height // 2, :] *= 0.40
        rgb = np.repeat(np.rint(gray)[..., None], 3, axis=2).astype(np.uint8)

        result = threshold_sauvola_image(
            Image.fromarray(rgb),
            SauvolaParameters(window_size=9),
        )
        final = np.asarray(result.ternary_image)
        stroke_black = np.all(final[height // 2] == (0, 0, 0), axis=1)
        background_white = np.all(
            np.delete(final, height // 2, axis=0) == (255, 255, 255),
            axis=2,
        )

        self.assertGreater(float(np.mean(stroke_black)), 0.80)
        self.assertGreater(float(np.mean(background_white)), 0.80)

    def test_red_stays_red_and_green_collapses_to_black(self) -> None:
        rgb = np.full((15, 15, 3), 220, dtype=np.uint8)
        rgb[2:7, 2:7] = (180, 25, 25)
        rgb[8:13, 8:13] = (25, 140, 25)

        result = threshold_sauvola_image(
            Image.fromarray(rgb),
            SauvolaParameters(window_size=7),
        )
        final = np.asarray(result.ternary_image)

        self.assertTrue(np.all(final[4, 4] == (255, 0, 0)))
        self.assertTrue(np.all(final[10, 10] == (0, 0, 0)))

    def test_warm_neutral_reference_recovers_darker_red(self) -> None:
        rgb = np.full((31, 31, 3), (170, 140, 110), dtype=np.uint8)
        rgb[11:20, 11:20] = (125, 65, 55)

        result = threshold_sauvola_image(
            Image.fromarray(rgb),
            SauvolaParameters(window_size=9, colorfulness_min=0.30),
        )
        final = np.asarray(result.ternary_image)

        self.assertEqual("relative_lab", result.metrics["red_detection_mode"])
        self.assertEqual(1, result.metrics["neutral_reference_reliable"])
        self.assertTrue(np.all(final[15, 15] == (255, 0, 0)))
        self.assertFalse(np.all(final[3, 3] == (255, 0, 0)))

    def test_red_field_is_rejected_but_inner_ink_component_is_kept(self) -> None:
        field = np.ones((20, 20), dtype=bool)
        accepted, rejected, largest_ratio, rejected_count, field_rejected = (
            filter_red_ink_components(
                field,
                field_max_fraction=0.40,
                component_max_fraction=0.30,
                component_max_border_sides=1,
            )
        )

        self.assertFalse(np.any(accepted))
        self.assertTrue(np.all(rejected))
        self.assertEqual(1.0, largest_ratio)
        self.assertEqual(1, rejected_count)
        self.assertTrue(field_rejected)

        ink = np.zeros((20, 20), dtype=bool)
        ink[7:13, 8:12] = True
        accepted, rejected, _, rejected_count, field_rejected = (
            filter_red_ink_components(
                ink,
                field_max_fraction=0.40,
                component_max_fraction=0.30,
                component_max_border_sides=1,
            )
        )

        self.assertTrue(np.array_equal(accepted, ink))
        self.assertFalse(np.any(rejected))
        self.assertEqual(0, rejected_count)
        self.assertFalse(field_rejected)

    def test_trial_writes_sauvola_contact_sheet(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source_database = root / "source.sqlite"
            sample_database = root / "sample.sqlite"
            output_directory = root / "sauvola"
            create_source_database(source_database)
            build_sample_database(
                source_database=source_database,
                output_database=sample_database,
                seed=42,
                jp_samples_per_label=2,
            )

            summary = run_sauvola_trial(
                sample_database=sample_database,
                output_directory=output_directory,
                parameters=SauvolaParameters(window_size=5),
                diagnostic_count=4,
            )

            self.assertEqual(7, summary["crop_count"])
            for name in (
                "metrics.csv",
                "summary.json",
                "contact_sheet.png",
                "manual_highest_black_ratio.png",
                "red_labels_lowest_red_ratio.png",
                "non_red_labels_highest_red_ratio.png",
                "red_surface_rejections.png",
            ):
                path = output_directory / name
                self.assertTrue(path.is_file(), name)
                self.assertGreater(path.stat().st_size, 0, name)


class LabThresholdTests(unittest.TestCase):
    def test_white_black_red_and_green_are_classified_as_expected(self) -> None:
        rgb = np.array(
            [[[255, 255, 255], [0, 0, 0], [255, 0, 0], [0, 128, 0]]],
            dtype=np.uint8,
        )
        lab = rgb_to_lab(rgb)
        white, black, red = classify_lab(lab, Thresholds())

        self.assertEqual([True, False, False, False], white[0].tolist())
        self.assertEqual([False, True, False, True], black[0].tolist())
        self.assertEqual([False, False, True, False], red[0].tolist())

    def test_threshold_image_emits_exact_three_colors(self) -> None:
        image = Image.new("RGB", (4, 1))
        image.putdata(
            [
                (255, 255, 255),
                (0, 0, 0),
                (255, 0, 0),
                (0, 128, 0),
            ]
        )
        result, metrics = threshold_image(image, Thresholds())

        self.assertEqual(
            [(255, 255, 255), (0, 0, 0), (255, 0, 0), (0, 0, 0)],
            list(result.get_flattened_data()),
        )
        self.assertEqual(1, metrics["white_pixels"])
        self.assertEqual(2, metrics["black_pixels"])
        self.assertEqual(1, metrics["red_pixels"])

    def test_trial_writes_metrics_summary_and_contact_sheets(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source_database = root / "source.sqlite"
            sample_database = root / "sample.sqlite"
            output_directory = root / "trial"
            create_source_database(source_database)
            build_sample_database(
                source_database=source_database,
                output_database=sample_database,
                seed=42,
                jp_samples_per_label=2,
            )

            summary = run_trial(
                sample_database=sample_database,
                output_directory=output_directory,
                thresholds=Thresholds(),
                diagnostic_count=4,
            )

            self.assertEqual(7, summary["crop_count"])
            for name in (
                "metrics.csv",
                "summary.json",
                "contact_sheet.png",
                "highest_red_ratio_non_red.png",
                "lowest_black_ratio_non_white.png",
                "manual_highest_black_ratio.png",
            ):
                path = output_directory / name
                self.assertTrue(path.is_file(), name)
                self.assertGreater(path.stat().st_size, 0, name)


def create_source_database(path: Path) -> None:
    png = encoded_test_png()
    with closing(sqlite3.connect(path)) as connection:
        connection.executescript(SOURCE_SCHEMA)
        rows: list[tuple[object, ...]] = []
        for label in ("1m", "1p"):
            for index in range(10):
                rows.append(
                    source_row(
                        crop_id=f"jp-train-{label}-{index}",
                        source="jp",
                        partition="train",
                        label=label,
                        png=png,
                    )
                )
        for label in ("1m", "1p"):
            rows.append(
                source_row(
                    crop_id=f"jp-valid-{label}",
                    source="jp",
                    partition="valid",
                    label=label,
                    png=png,
                )
            )
        for index, label in enumerate(("1m", "green", "red")):
            rows.append(
                source_row(
                    crop_id=f"manual-{index}",
                    source="manual",
                    partition="capture",
                    label=label,
                    png=png,
                    brightness="dark" if index == 0 else "normal",
                )
            )
        connection.executemany(
            """
            INSERT INTO tile_crop(
                crop_id,
                source,
                source_partition,
                tile_label,
                image_width,
                image_height,
                image_png,
                source_image_path,
                source_image_id,
                capture_id,
                layout_id,
                region,
                brightness,
                shadow
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
        connection.execute(
            "INSERT INTO dataset_metadata(key, value) VALUES (?, ?)",
            ("source.jp.train.crop_count", "20"),
        )
        connection.commit()


def source_row(
    *,
    crop_id: str,
    source: str,
    partition: str,
    label: str,
    png: bytes,
    brightness: str | None = None,
) -> tuple[object, ...]:
    return (
        crop_id,
        source,
        partition,
        label,
        4,
        1,
        png,
        f"images/{crop_id}.png",
        crop_id,
        crop_id if source == "manual" else None,
        "layout-1" if source == "manual" else None,
        "completed_hand" if source == "manual" else None,
        brightness,
        "none" if source == "manual" else None,
    )


def encoded_test_png() -> bytes:
    image = Image.new("RGB", (4, 1))
    image.putdata(
        [
            (255, 255, 255),
            (0, 0, 0),
            (255, 0, 0),
            (0, 128, 0),
        ]
    )
    output = io.BytesIO()
    image.save(output, format="PNG")
    return output.getvalue()


def selected_crop_ids(path: Path) -> list[str]:
    with closing(sqlite3.connect(path)) as connection:
        return [
            str(row[0])
            for row in connection.execute(
                "SELECT crop_id FROM tile_crop ORDER BY crop_id"
            )
        ]


if __name__ == "__main__":
    unittest.main()
