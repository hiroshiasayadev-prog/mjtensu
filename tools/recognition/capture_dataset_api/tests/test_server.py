from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from tools.recognition.capture_dataset_api.server import (
    CaptureApi,
    RequestError,
    UploadedPart,
    expected_front_groups,
    jpeg_dimensions,
    png_dimensions,
    validate_annotation_payload,
    validate_manifest,
)


class ImageHeaderTest(unittest.TestCase):
    def test_png_dimensions(self) -> None:
        png = (
            b"\x89PNG\r\n\x1a\n"
            + (13).to_bytes(4, "big")
            + b"IHDR"
            + (320).to_bytes(4, "big")
            + (320).to_bytes(4, "big")
            + b"\x08\x06\x00\x00\x00"
        )
        self.assertEqual((320, 320), png_dimensions(png))

    def test_jpeg_dimensions(self) -> None:
        jpeg = (
            b"\xff\xd8"
            + b"\xff\xc0"
            + (17).to_bytes(2, "big")
            + b"\x08"
            + (1080).to_bytes(2, "big")
            + (1920).to_bytes(2, "big")
            + b"\x03"
            + b"\x01\x11\x00\x02\x11\x00\x03\x11\x00"
            + b"\xff\xd9"
        )
        self.assertEqual((1920, 1080), jpeg_dimensions(jpeg))


class AnnotationValidationTest(unittest.TestCase):
    def test_complete_annotation_uses_only_front_facing_closed_kan_tiles(self) -> None:
        capture = annotation_capture()
        payload = complete_annotation_payload()
        status, document = validate_annotation_payload(payload, capture)
        self.assertEqual("complete", status)
        self.assertEqual(2, len(document["boxes"]["melds"]))
        groups = expected_front_groups(capture["task"])
        self.assertEqual([2], [len(group) for group in groups["melds"]])

    def test_complete_annotation_rejects_back_tile_boxes_for_closed_kan(self) -> None:
        capture = annotation_capture()
        payload = complete_annotation_payload()
        payload["document"]["boxes"]["melds"].extend(
            [rotated_box("meld-extra-1", 20, 50), rotated_box("meld-extra-2", 80, 50)]
        )
        with self.assertRaises(RequestError):
            validate_annotation_payload(payload, capture)

    def test_arbitrary_rotation_is_accepted(self) -> None:
        capture = annotation_capture()
        payload = complete_annotation_payload()
        payload["document"]["boxes"]["completed_hand"][0]["angleDeg"] = 12.75
        validate_annotation_payload(payload, capture)


class RawCatalogCaptureTest(unittest.TestCase):
    def test_raw_catalog_uses_original_as_annotation_asset(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            api = CaptureApi(Path(temporary_directory))
            task = api.next_task("tile-catalog-warm-4-v2")
            self.assertIsNotNone(task)
            assert task is not None
            manifest = valid_raw_catalog_manifest()
            manifest["taskId"] = task["id"]
            parts = {
                "manifest": UploadedPart(
                    name="manifest",
                    filename="manifest.json",
                    content_type="application/json",
                    data=json.dumps(manifest).encode("utf-8"),
                ),
                "original": UploadedPart(
                    name="original",
                    filename="original.jpg",
                    content_type="image/jpeg",
                    data=jpeg_bytes(1920, 1080),
                ),
                "composite": UploadedPart(
                    name="composite",
                    filename="composite.png",
                    content_type="image/png",
                    data=png_bytes(320, 320),
                ),
            }
            result = api.save_capture(parts)
            detail = api.annotation_capture(result["captureId"])
            self.assertIsNotNone(detail)
            assert detail is not None
            self.assertEqual(detail["original"]["path"], detail["regionPaths"]["melds"])
            self.assertIsNone(detail["regionPaths"]["completed_hand"])
            self.assertIsNone(detail["regionPaths"]["dora_indicators"])
            self.assertEqual([], detail["detections"])


class ManifestValidationTest(unittest.TestCase):
    def test_capture_manifest_with_preview_geometry(self) -> None:
        manifest = valid_manifest()
        validate_manifest(manifest)

    def test_tile_catalog_layout_version_is_accepted(self) -> None:
        manifest = valid_manifest()
        manifest["layoutVersion"] = "TILE-CATALOG-CAPTURE-v2"
        validate_manifest(manifest)

    def test_raw_tile_catalog_allows_disabled_empty_regions(self) -> None:
        manifest = valid_raw_catalog_manifest()
        validate_manifest(manifest)

    def test_preview_detection_requires_original_detection(self) -> None:
        manifest = valid_manifest()
        manifest["detections"] = [
            {
                "detectionIndex": 0,
                "region": "completed_hand",
                "confidence": 0.9,
                "composite": {"x": 10, "y": 10, "width": 20, "height": 30},
                "original": None,
                "preview": None,
            }
        ]
        with self.assertRaises(RequestError):
            validate_manifest(manifest)


def png_bytes(width: int, height: int) -> bytes:
    return (
        b"\x89PNG\r\n\x1a\n"
        + (13).to_bytes(4, "big")
        + b"IHDR"
        + width.to_bytes(4, "big")
        + height.to_bytes(4, "big")
        + b"\x08\x06\x00\x00\x00"
    )


def jpeg_bytes(width: int, height: int) -> bytes:
    return (
        b"\xff\xd8"
        + b"\xff\xc0"
        + (17).to_bytes(2, "big")
        + b"\x08"
        + height.to_bytes(2, "big")
        + width.to_bytes(2, "big")
        + b"\x03"
        + b"\x01\x11\x00\x02\x11\x00\x03\x11\x00"
        + b"\xff\xd9"
    )


def annotation_capture() -> dict[str, object]:
    region = {
        "enabled": True,
        "pixel": {"x": 0, "y": 0, "width": 100, "height": 100},
    }
    return {
        "captureId": "cap-annotation",
        "manifest": {
            "regionRects": {
                "completed_hand": region,
                "dora_indicators": region,
                "melds": region,
            }
        },
        "task": {
            "hand": [
                {"ordinal": 0, "tile": "1m", "face": "front", "rotation": 0},
                {"ordinal": 1, "tile": "2m", "face": "front", "rotation": 0},
            ],
            "dora": {
                "visible": [
                    {"ordinal": 0, "tile": "3p", "face": "front", "rotation": 0}
                ],
                "ura": [
                    {"ordinal": 0, "tile": "4p", "face": "front", "rotation": 0}
                ],
            },
            "melds": [
                {
                    "ordinal": 0,
                    "kind": "closed-kan",
                    "tiles": [
                        {"ordinal": 0, "tile": "east", "face": "back", "rotation": 0},
                        {"ordinal": 1, "tile": "east", "face": "front", "rotation": 0},
                        {"ordinal": 2, "tile": "east", "face": "front", "rotation": 0},
                        {"ordinal": 3, "tile": "east", "face": "back", "rotation": 0},
                    ],
                }
            ],
        },
    }


def complete_annotation_payload() -> dict[str, object]:
    return {
        "status": "complete",
        "document": {
            "schemaVersion": 1,
            "captureId": "cap-annotation",
            "boxes": {
                "completed_hand": [
                    rotated_box("hand-1", 30, 50),
                    rotated_box("hand-2", 70, 50),
                ],
                "dora_indicators": [
                    rotated_box("dora-1", 50, 25),
                    rotated_box("dora-2", 50, 75),
                ],
                "melds": [
                    rotated_box("meld-1", 35, 50),
                    rotated_box("meld-2", 65, 50),
                ],
            },
        },
    }


def rotated_box(box_id: str, center_x: float, center_y: float) -> dict[str, object]:
    return {
        "id": box_id,
        "centerX": center_x,
        "centerY": center_y,
        "width": 12,
        "height": 20,
        "angleDeg": -7.5,
    }


def valid_raw_catalog_manifest() -> dict[str, object]:
    manifest = valid_manifest()
    manifest["campaignId"] = "tile-catalog-warm-4-v2"
    manifest["taskId"] = "tile-catalog-warm-4-v2-normal-front"
    manifest["layoutVersion"] = "TILE-CATALOG-CAPTURE-v2"
    manifest["provider"] = "webgl-visual-only"
    manifest["preview"] = {
        "width": 1920,
        "height": 1080,
        "devicePixelRatio": 1,
        "videoElement": {"x": 0, "y": 0, "width": 1920, "height": 1080},
        "sourceToDisplayScale": 1,
        "sourceDisplayOffsetX": 0,
        "sourceDisplayOffsetY": 0,
    }
    zero = {
        "enabled": False,
        "pixel": {"x": 0, "y": 0, "width": 0, "height": 0},
        "normalized": {"x": 0, "y": 0, "width": 0, "height": 0},
        "display": {"x": 0, "y": 0, "width": 0, "height": 0},
    }
    manifest["regionRects"] = {
        "completed_hand": zero,
        "dora_indicators": zero,
        "melds": {
            "enabled": True,
            "pixel": {"x": 0, "y": 0, "width": 1920, "height": 1080},
            "normalized": {"x": 0, "y": 0, "width": 1, "height": 1},
            "display": {"x": 0, "y": 0, "width": 1920, "height": 1080},
        },
    }
    manifest["detections"] = []
    manifest["catalog"] = {
        "schemaVersion": 2,
        "variantId": "normal-front",
        "rows": [],
        "smartphoneDetector": "visual-only",
        "annotationDetector": "pc-after-upload",
    }
    return manifest


def valid_manifest() -> dict[str, object]:
    return {
        "uploadClientId": "00000000-0000-4000-8000-000000000001",
        "taskId": "task-1",
        "campaignId": "initial-120",
        "capturedAt": "2026-08-05T08:00:00.000Z",
        "original": {"width": 1920, "height": 1080},
        "preview": {
            "width": 844,
            "height": 390,
            "devicePixelRatio": 3,
            "videoElement": {"x": 0, "y": 0, "width": 844, "height": 390},
            "sourceToDisplayScale": 0.4395833333333333,
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
        "regionRects": {
            "completed_hand": {
                "enabled": True,
                "pixel": {
                    "x": 55.7345971563981,
                    "y": 524.0758293838862,
                    "width": 1198.862559241706,
                    "height": 282.08530805687207,
                },
                "normalized": {
                    "x": 0.029028436018957344,
                    "y": 0.48525539757767244,
                    "width": 0.6244075829383886,
                    "height": 0.2611901000526593,
                },
                "display": {"x": 24.5, "y": 188, "width": 527, "height": 124},
            },
            "dora_indicators": {
                "enabled": True,
                "pixel": {
                    "x": 55.7345971563981,
                    "y": 219.24170616113744,
                    "width": 1198.862559241706,
                    "height": 282.08530805687207,
                },
                "normalized": {
                    "x": 0.029028436018957344,
                    "y": 0.20300157977883096,
                    "width": 0.6244075829383886,
                    "height": 0.2611901000526593,
                },
                "display": {"x": 24.5, "y": 54, "width": 527, "height": 124},
            },
            "melds": {
                "enabled": False,
                "pixel": {
                    "x": 1277.345971563981,
                    "y": 219.24170616113744,
                    "width": 586.9194312796209,
                    "height": 586.9194312796209,
                },
                "normalized": {
                    "x": 0.6652843601895735,
                    "y": 0.20300157977883096,
                    "width": 0.3056872037914692,
                    "height": 0.5434439178515008,
                },
                "display": {"x": 561.5, "y": 54, "width": 258, "height": 258},
            },
        },
        "detections": [],
    }


if __name__ == "__main__":
    unittest.main()
