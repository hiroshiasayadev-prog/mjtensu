from __future__ import annotations

import argparse
import json
import math
import os
import re
import tempfile
import uuid
from dataclasses import dataclass
from datetime import datetime
from email.parser import BytesParser
from email.policy import default
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, unquote, urlparse

from .campaign import CAMPAIGN_ID, generate_campaign
from .database import CaptureDatabase
from .tile_catalog_campaign import (
    CAMPAIGN_ID as TILE_CATALOG_CAMPAIGN_ID,
    LAYOUT_VERSION as TILE_CATALOG_LAYOUT_VERSION,
    generate_tile_catalog_campaign,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_STORAGE_ROOT = REPOSITORY_ROOT / ".local" / "recognition" / "capture_dataset"
EXPECTED_LAYOUT_VERSIONS = {
    "PRODUCT-ADR-RECOGNITION-002-v1",
    TILE_CATALOG_LAYOUT_VERSION,
}
MAX_REQUEST_BYTES = 80 * 1024 * 1024
MAX_ANNOTATION_BYTES = 2 * 1024 * 1024
CAMPAIGN_PATH = re.compile(r"^/api/campaigns/([^/]+)/(overview|next-task)$")
UNDO_LAST_CAPTURE_PATH = re.compile(r"^/api/campaigns/([^/]+)/last-capture$")
ANNOTATION_CAPTURE_PATH = re.compile(r"^/api/annotations/captures/([^/]+)$")


@dataclass(frozen=True)
class UploadedPart:
    name: str
    filename: str | None
    content_type: str
    data: bytes


class CaptureApi:
    def __init__(self, storage_root: Path) -> None:
        self.storage_root = storage_root.resolve()
        self.storage_root.mkdir(parents=True, exist_ok=True)
        self.originals = self.storage_root / "originals"
        self.composites = self.storage_root / "composites"
        self.hand_regions = self.storage_root / "regions" / "hand"
        self.dora_regions = self.storage_root / "regions" / "dora"
        self.meld_regions = self.storage_root / "regions" / "meld"
        for directory in (
            self.originals,
            self.composites,
            self.hand_regions,
            self.dora_regions,
            self.meld_regions,
        ):
            directory.mkdir(parents=True, exist_ok=True)
        self.database = CaptureDatabase(self.storage_root / "dataset.sqlite")
        self.campaign = generate_campaign()
        self.campaigns = {
            self.campaign["id"]: self.campaign,
            TILE_CATALOG_CAMPAIGN_ID: generate_tile_catalog_campaign(),
        }
        for campaign in self.campaigns.values():
            self.database.seed_campaign(campaign)
            write_atomic(
                self.storage_root / f"campaign-{campaign['id']}.json",
                (json.dumps(campaign, ensure_ascii=False, indent=2) + "\n").encode("utf-8"),
            )

    def overview(self, campaign_id: str) -> dict[str, Any] | None:
        return self.database.campaign_overview(campaign_id)

    def next_task(self, campaign_id: str) -> dict[str, Any] | None:
        return self.database.next_task(campaign_id)

    def undo_last_capture(self, campaign_id: str) -> dict[str, Any]:
        if self.database.campaign_overview(campaign_id) is None:
            raise RequestError(HTTPStatus.NOT_FOUND, f"Unknown campaign: {campaign_id}")
        undone = self.database.undo_last_capture(campaign_id)
        if undone is None:
            raise RequestError(HTTPStatus.NOT_FOUND, "No saved capture to undo")

        removed_paths: list[str] = []
        for relative_path in undone["paths"]:
            candidate = (self.storage_root / relative_path).resolve()
            try:
                candidate.relative_to(self.storage_root)
            except ValueError as error:
                raise RuntimeError(f"Stored capture path escapes storage root: {relative_path}") from error
            candidate.unlink(missing_ok=True)
            removed_paths.append(relative_path)

        return {
            "captureId": undone["captureId"],
            "taskId": undone["taskId"],
            "removedPaths": removed_paths,
        }

    def annotation_campaigns(self) -> list[dict[str, Any]]:
        return self.database.annotation_campaigns()

    def annotation_capture_list(self, campaign_id: str) -> list[dict[str, Any]]:
        return self.database.annotation_capture_list(campaign_id)

    def annotation_capture(self, capture_id: str) -> dict[str, Any]:
        capture = self.database.annotation_capture(capture_id)
        if capture is None:
            raise RequestError(HTTPStatus.NOT_FOUND, f"Unknown capture: {capture_id}")
        return capture

    def annotation_asset(self, relative_path: str) -> tuple[bytes, str]:
        candidate = (self.storage_root / relative_path).resolve()
        try:
            candidate.relative_to(self.storage_root)
        except ValueError as error:
            raise RequestError(HTTPStatus.BAD_REQUEST, "Asset path escapes storage root") from error
        if not candidate.is_file():
            raise RequestError(HTTPStatus.NOT_FOUND, f"Asset not found: {relative_path}")
        content_types = {
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".png": "image/png",
        }
        content_type = content_types.get(candidate.suffix.lower())
        if content_type is None:
            raise RequestError(HTTPStatus.UNSUPPORTED_MEDIA_TYPE, "Unsupported annotation asset type")
        return candidate.read_bytes(), content_type

    def save_annotation(self, capture_id: str, payload: Any) -> dict[str, Any]:
        capture = self.annotation_capture(capture_id)
        status, document = validate_annotation_payload(payload, capture)
        self.database.save_annotation(capture_id, status, 1, document)
        return {"captureId": capture_id, "status": status}

    def save_capture(self, parts: dict[str, UploadedPart]) -> dict[str, Any]:
        manifest_part = parts.get("manifest")
        original_part = parts.get("original")
        composite_part = parts.get("composite")
        if manifest_part is None or original_part is None or composite_part is None:
            raise RequestError(
                HTTPStatus.BAD_REQUEST,
                "multipart fields manifest, original, and composite are required",
            )
        try:
            manifest = json.loads(manifest_part.data.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise RequestError(HTTPStatus.BAD_REQUEST, f"Invalid manifest JSON: {error}") from error
        validate_manifest(manifest)

        task = self.database.task(manifest["taskId"])
        if task is None:
            raise RequestError(HTTPStatus.BAD_REQUEST, f"Unknown task: {manifest['taskId']}")
        if task["campaignId"] != manifest["campaignId"]:
            raise RequestError(HTTPStatus.BAD_REQUEST, "Task campaign does not match manifest campaign")
        expected_enabled_regions = {
            "completed_hand": len(task["hand"]) > 0,
            "dora_indicators": len(task["dora"]["visible"]) + len(task["dora"]["ura"]) > 0,
            "melds": len(task["melds"]) > 0,
        }
        actual_enabled_regions = {
            key: bool(manifest["regionRects"][key]["enabled"])
            for key in expected_enabled_regions
        }
        if actual_enabled_regions != expected_enabled_regions:
            raise RequestError(
                HTTPStatus.BAD_REQUEST,
                "Enabled capture regions do not match the task: "
                f"expected={expected_enabled_regions}, actual={actual_enabled_regions}",
            )

        existing = self.database.existing_capture(manifest["uploadClientId"])
        if existing is not None:
            if existing["taskId"] != manifest["taskId"]:
                raise RequestError(
                    HTTPStatus.CONFLICT,
                    "uploadClientId is already associated with a different task",
                )
            return {
                "captureId": existing["captureId"],
                "taskCompleted": True,
                "nextTaskId": self.database.next_task_id(manifest["campaignId"]),
                "idempotentReplay": True,
            }
        existing_task_capture = self.database.capture_for_task(manifest["taskId"])
        if existing_task_capture is not None:
            raise RequestError(
                HTTPStatus.CONFLICT,
                f"Task already has capture {existing_task_capture['captureId']}",
            )

        validate_image_part(original_part, "original", "image/jpeg")
        validate_image_part(composite_part, "composite", "image/png")
        original_dimensions = jpeg_dimensions(original_part.data)
        manifest_dimensions = (
            int(manifest["original"]["width"]),
            int(manifest["original"]["height"]),
        )
        if original_dimensions != manifest_dimensions:
            raise RequestError(
                HTTPStatus.BAD_REQUEST,
                f"Original JPEG dimensions {original_dimensions} do not match manifest {manifest_dimensions}",
            )
        composite_dimensions = png_dimensions(composite_part.data)
        if composite_dimensions != (320, 320):
            raise RequestError(
                HTTPStatus.BAD_REQUEST,
                f"Composite PNG must be 320 x 320, received {composite_dimensions}",
            )
        crop_contract = {
            "completed_hand": "hand_crop",
            "dora_indicators": "dora_crop",
            "melds": "meld_crop",
        }
        raw_catalog_capture = (
            str(manifest["campaignId"]).startswith("tile-catalog")
            and isinstance(manifest.get("catalog"), dict)
            and manifest["catalog"].get("schemaVersion") == 2
        )
        if raw_catalog_capture:
            catalog_rect = manifest["regionRects"]["melds"]["pixel"]
            expected_catalog_rect = {
                "x": 0.0,
                "y": 0.0,
                "width": float(manifest_dimensions[0]),
                "height": float(manifest_dimensions[1]),
            }
            validate_rect_close(
                catalog_rect,
                expected_catalog_rect,
                "regionRects.melds.pixel",
                tolerance=0.001,
            )
        for region_key, part_name in crop_contract.items():
            optional_part = parts.get(part_name)
            enabled = manifest["regionRects"][region_key]["enabled"]
            original_is_region_asset = raw_catalog_capture and region_key == "melds"
            if enabled and optional_part is None and not original_is_region_asset:
                raise RequestError(
                    HTTPStatus.BAD_REQUEST,
                    f"Enabled region {region_key} requires multipart field {part_name}",
                )
            if not enabled and optional_part is not None:
                raise RequestError(
                    HTTPStatus.BAD_REQUEST,
                    f"Disabled region {region_key} must not include multipart field {part_name}",
                )
            if optional_part is not None:
                validate_image_part(optional_part, part_name, "image/png")
                actual_crop_dimensions = png_dimensions(optional_part.data)
                pixel_rect = manifest["regionRects"][region_key]["pixel"]
                expected_crop_dimensions = (
                    max(1, math.floor(float(pixel_rect["width"]) + 0.5)),
                    max(1, math.floor(float(pixel_rect["height"]) + 0.5)),
                )
                if actual_crop_dimensions != expected_crop_dimensions:
                    raise RequestError(
                        HTTPStatus.BAD_REQUEST,
                        f"{part_name} dimensions {actual_crop_dimensions} do not match "
                        f"region crop {expected_crop_dimensions}",
                    )

        capture_id = f"cap_{uuid.uuid4().hex}"
        date_prefix = safe_date_prefix(manifest["capturedAt"])
        destinations = {
            "original": self.originals / date_prefix / f"{capture_id}.jpg",
            "composite": self.composites / date_prefix / f"{capture_id}.png",
            "hand_crop": self.hand_regions / date_prefix / f"{capture_id}.png",
            "dora_crop": self.dora_regions / date_prefix / f"{capture_id}.png",
            "meld_crop": self.meld_regions / date_prefix / f"{capture_id}.png",
        }
        source_parts = {
            "original": original_part,
            "composite": composite_part,
            "hand_crop": parts.get("hand_crop"),
            "dora_crop": parts.get("dora_crop"),
            "meld_crop": parts.get("meld_crop"),
        }

        written: list[Path] = []
        relative_paths: dict[str, str | None] = {}
        try:
            for key, destination in destinations.items():
                part = source_parts[key]
                if part is None:
                    relative_paths[key] = None
                    continue
                destination.parent.mkdir(parents=True, exist_ok=True)
                write_atomic(destination, part.data)
                written.append(destination)
                relative_paths[key] = destination.relative_to(self.storage_root).as_posix()
            if raw_catalog_capture:
                relative_paths["meld_crop"] = relative_paths["original"]
            self.database.insert_capture(capture_id, manifest, relative_paths)
        except Exception:
            for path in written:
                path.unlink(missing_ok=True)
            raise

        return {
            "captureId": capture_id,
            "taskCompleted": True,
            "nextTaskId": self.database.next_task_id(manifest["campaignId"]),
            "idempotentReplay": False,
        }


class RequestError(Exception):
    def __init__(self, status: HTTPStatus, message: str) -> None:
        super().__init__(message)
        self.status = status
        self.message = message


def validate_annotation_payload(
    payload: Any,
    capture: dict[str, Any],
) -> tuple[str, dict[str, Any]]:
    if not isinstance(payload, dict):
        raise RequestError(HTTPStatus.BAD_REQUEST, "Annotation payload must be an object")
    status = payload.get("status")
    if status not in {"draft", "complete"}:
        raise RequestError(HTTPStatus.BAD_REQUEST, "Annotation status must be draft or complete")
    document = payload.get("document")
    if not isinstance(document, dict):
        raise RequestError(HTTPStatus.BAD_REQUEST, "Annotation document must be an object")
    if document.get("schemaVersion") != 1:
        raise RequestError(HTTPStatus.BAD_REQUEST, "Annotation schemaVersion must be 1")
    if document.get("captureId") != capture["captureId"]:
        raise RequestError(HTTPStatus.BAD_REQUEST, "Annotation captureId does not match the URL")

    boxes_by_region = document.get("boxes")
    expected_regions = {"completed_hand", "dora_indicators", "melds"}
    if not isinstance(boxes_by_region, dict) or set(boxes_by_region) != expected_regions:
        raise RequestError(
            HTTPStatus.BAD_REQUEST,
            f"Annotation boxes must contain exactly {sorted(expected_regions)}",
        )

    manifest_regions = capture["manifest"].get("regionRects")
    if not isinstance(manifest_regions, dict):
        raise RequestError(HTTPStatus.INTERNAL_SERVER_ERROR, "Capture region geometry is missing")

    seen_box_ids: set[str] = set()
    normalized_boxes: dict[str, list[dict[str, Any]]] = {}
    for region_key in sorted(expected_regions):
        raw_boxes = boxes_by_region.get(region_key)
        if not isinstance(raw_boxes, list):
            raise RequestError(HTTPStatus.BAD_REQUEST, f"boxes.{region_key} must be an array")
        region_manifest = manifest_regions.get(region_key)
        if not isinstance(region_manifest, dict) or not isinstance(region_manifest.get("pixel"), dict):
            raise RequestError(
                HTTPStatus.INTERNAL_SERVER_ERROR,
                f"Capture geometry for {region_key} is missing",
            )
        region_width = max(1, math.floor(float(region_manifest["pixel"]["width"]) + 0.5))
        region_height = max(1, math.floor(float(region_manifest["pixel"]["height"]) + 0.5))
        normalized_boxes[region_key] = []
        for box_index, raw_box in enumerate(raw_boxes):
            label = f"boxes.{region_key}[{box_index}]"
            box = validate_rotated_box(raw_box, label, seen_box_ids)
            if status == "complete" and not rotated_box_inside(
                box,
                region_width,
                region_height,
            ):
                raise RequestError(
                    HTTPStatus.BAD_REQUEST,
                    f"{label} extends outside its region crop",
                )
            normalized_boxes[region_key].append(box)

    normalized_document = dict(document)
    normalized_document["boxes"] = normalized_boxes
    if status == "complete":
        validate_complete_annotation(normalized_document, capture["task"])
    return str(status), normalized_document


def validate_rotated_box(
    raw_box: Any,
    label: str,
    seen_box_ids: set[str],
) -> dict[str, Any]:
    if not isinstance(raw_box, dict):
        raise RequestError(HTTPStatus.BAD_REQUEST, f"{label} must be an object")
    box_id = raw_box.get("id")
    if not isinstance(box_id, str) or not box_id:
        raise RequestError(HTTPStatus.BAD_REQUEST, f"{label}.id must be a non-empty string")
    if box_id in seen_box_ids:
        raise RequestError(HTTPStatus.BAD_REQUEST, f"Duplicate annotation box id: {box_id}")
    seen_box_ids.add(box_id)
    values: dict[str, float] = {}
    for key in ("centerX", "centerY", "width", "height", "angleDeg"):
        value = raw_box.get(key)
        if not is_finite_number(value):
            raise RequestError(HTTPStatus.BAD_REQUEST, f"{label}.{key} must be finite numeric")
        values[key] = float(value)
    if values["width"] <= 1 or values["height"] <= 1:
        raise RequestError(HTTPStatus.BAD_REQUEST, f"{label} width and height must exceed one pixel")
    if abs(values["angleDeg"]) > 3600:
        raise RequestError(HTTPStatus.BAD_REQUEST, f"{label}.angleDeg is unreasonable")
    return {"id": box_id, **values}


def rotated_box_inside(box: dict[str, Any], region_width: int, region_height: int) -> bool:
    radians = math.radians(float(box["angleDeg"]))
    cosine = math.cos(radians)
    sine = math.sin(radians)
    half_width = float(box["width"]) / 2
    half_height = float(box["height"]) / 2
    for local_x, local_y in (
        (-half_width, -half_height),
        (half_width, -half_height),
        (half_width, half_height),
        (-half_width, half_height),
    ):
        x = float(box["centerX"]) + local_x * cosine - local_y * sine
        y = float(box["centerY"]) + local_x * sine + local_y * cosine
        if x < -0.001 or y < -0.001 or x > region_width + 0.001 or y > region_height + 0.001:
            return False
    return True


def validate_complete_annotation(document: dict[str, Any], task: dict[str, Any]) -> None:
    expected_groups = expected_front_groups(task)
    boxes_by_region = document["boxes"]
    for region_key, groups in expected_groups.items():
        boxes = boxes_by_region[region_key]
        expected_counts = [len(group) for group in groups]
        if not expected_counts:
            if boxes:
                raise RequestError(
                    HTTPStatus.BAD_REQUEST,
                    f"{region_key} expects no visible tiles but has {len(boxes)} boxes",
                )
            continue
        if len(boxes) != sum(expected_counts):
            raise RequestError(
                HTTPStatus.BAD_REQUEST,
                f"{region_key} box count {len(boxes)} does not match expected {sum(expected_counts)}",
            )
        # Group identity is derived deterministically from geometry: boxes are ordered
        # top-to-bottom and partitioned by these known visible group sizes. Once the
        # total count matches, each group therefore has its expected cardinality.


def expected_front_groups(task: dict[str, Any]) -> dict[str, list[list[dict[str, Any]]]]:
    hand = [[slot for slot in task["hand"] if slot.get("face") == "front"]]
    dora = [
        [slot for slot in task["dora"]["visible"] if slot.get("face") == "front"],
        [slot for slot in task["dora"]["ura"] if slot.get("face") == "front"],
    ]
    melds = [
        [slot for slot in meld["tiles"] if slot.get("face") == "front"]
        for meld in task["melds"]
    ]
    return {
        "completed_hand": [group for group in hand if group],
        "dora_indicators": [group for group in dora if group],
        "melds": [group for group in melds if group],
    }



class CaptureRequestHandler(BaseHTTPRequestHandler):
    api: CaptureApi
    server_version = "MjtensuCaptureApi/0.1"

    def do_OPTIONS(self) -> None:  # noqa: N802
        self.send_response(HTTPStatus.NO_CONTENT)
        self._cors_headers()
        self.send_header("Access-Control-Allow-Methods", "GET, POST, PUT, DELETE, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Max-Age", "600")
        self.end_headers()

    def do_GET(self) -> None:  # noqa: N802
        try:
            parsed = urlparse(self.path)
            path = parsed.path
            query = parse_qs(parsed.query)
            if path == "/api/health":
                self._json(
                    HTTPStatus.OK,
                    {
                        "status": "ok",
                        "campaignId": CAMPAIGN_ID,
                        "campaignIds": list(self.api.campaigns),
                        "storageRoot": str(self.api.storage_root),
                    },
                )
                return
            if path == "/api/annotation-campaigns":
                self._json(HTTPStatus.OK, {"campaigns": self.api.annotation_campaigns()})
                return
            if path == "/api/annotations/captures":
                campaign_values = query.get("campaignId", [])
                if len(campaign_values) != 1 or not campaign_values[0]:
                    raise RequestError(HTTPStatus.BAD_REQUEST, "campaignId query parameter is required")
                captures = self.api.annotation_capture_list(campaign_values[0])
                self._json(HTTPStatus.OK, {"captures": captures})
                return
            annotation_match = ANNOTATION_CAPTURE_PATH.fullmatch(path)
            if annotation_match is not None:
                capture_id = unquote(annotation_match.group(1))
                self._json(HTTPStatus.OK, self.api.annotation_capture(capture_id))
                return
            if path == "/api/annotation-asset":
                path_values = query.get("path", [])
                if len(path_values) != 1 or not path_values[0]:
                    raise RequestError(HTTPStatus.BAD_REQUEST, "path query parameter is required")
                data, content_type = self.api.annotation_asset(path_values[0])
                self._binary(HTTPStatus.OK, data, content_type)
                return
            match = CAMPAIGN_PATH.fullmatch(path)
            if match is None:
                raise RequestError(HTTPStatus.NOT_FOUND, "Not found")
            campaign_id = unquote(match.group(1))
            action = match.group(2)
            if action == "overview":
                overview = self.api.overview(campaign_id)
                if overview is None:
                    raise RequestError(HTTPStatus.NOT_FOUND, f"Unknown campaign: {campaign_id}")
                self._json(HTTPStatus.OK, overview)
                return
            task = self.api.next_task(campaign_id)
            if task is None:
                raise RequestError(HTTPStatus.NOT_FOUND, "No pending task")
            self._json(HTTPStatus.OK, task)
        except RequestError as error:
            self._json(error.status, {"error": error.message})
        except Exception as error:  # pragma: no cover - last-resort server boundary
            self.log_error("GET failed: %s", error)
            self._json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": str(error)})

    def do_PUT(self) -> None:  # noqa: N802
        try:
            path = urlparse(self.path).path
            match = ANNOTATION_CAPTURE_PATH.fullmatch(path)
            if match is None:
                raise RequestError(HTTPStatus.NOT_FOUND, "Not found")
            content_length = parse_content_length(self.headers.get("Content-Length"))
            if content_length > MAX_ANNOTATION_BYTES:
                raise RequestError(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, "Annotation payload is too large")
            content_type = self.headers.get("Content-Type", "").split(";", 1)[0].strip().lower()
            if content_type != "application/json":
                raise RequestError(HTTPStatus.UNSUPPORTED_MEDIA_TYPE, "Expected application/json")
            body = self.rfile.read(content_length)
            if len(body) != content_length:
                raise RequestError(HTTPStatus.BAD_REQUEST, "Request body ended early")
            try:
                payload = json.loads(body.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                raise RequestError(HTTPStatus.BAD_REQUEST, f"Invalid annotation JSON: {error}") from error
            capture_id = unquote(match.group(1))
            self._json(HTTPStatus.OK, self.api.save_annotation(capture_id, payload))
        except RequestError as error:
            self._json(error.status, {"error": error.message})
        except Exception as error:  # pragma: no cover - last-resort server boundary
            self.log_error("PUT failed: %s", error)
            self._json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": str(error)})

    def do_DELETE(self) -> None:  # noqa: N802
        try:
            path = urlparse(self.path).path
            match = UNDO_LAST_CAPTURE_PATH.fullmatch(path)
            if match is None:
                raise RequestError(HTTPStatus.NOT_FOUND, "Not found")
            campaign_id = unquote(match.group(1))
            response = self.api.undo_last_capture(campaign_id)
            self._json(HTTPStatus.OK, response)
        except RequestError as error:
            self._json(error.status, {"error": error.message})
        except Exception as error:  # pragma: no cover - last-resort server boundary
            self.log_error("DELETE failed: %s", error)
            self._json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": str(error)})

    def do_POST(self) -> None:  # noqa: N802
        try:
            path = urlparse(self.path).path
            if path != "/api/captures":
                raise RequestError(HTTPStatus.NOT_FOUND, "Not found")
            content_length = parse_content_length(self.headers.get("Content-Length"))
            if content_length > MAX_REQUEST_BYTES:
                raise RequestError(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, "Capture upload is too large")
            content_type = self.headers.get("Content-Type", "")
            if not content_type.lower().startswith("multipart/form-data"):
                raise RequestError(HTTPStatus.UNSUPPORTED_MEDIA_TYPE, "Expected multipart/form-data")
            body = self.rfile.read(content_length)
            if len(body) != content_length:
                raise RequestError(HTTPStatus.BAD_REQUEST, "Request body ended early")
            parts = parse_multipart(content_type, body)
            response = self.api.save_capture(parts)
            self._json(HTTPStatus.OK, response)
        except RequestError as error:
            self._json(error.status, {"error": error.message})
        except Exception as error:  # pragma: no cover - last-resort server boundary
            self.log_error("POST failed: %s", error)
            self._json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": str(error)})

    def log_message(self, format: str, *args: object) -> None:
        print(f"[{self.log_date_time_string()}] {self.client_address[0]} {format % args}")

    def _json(self, status: HTTPStatus, payload: dict[str, Any]) -> None:
        data = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        self.send_response(status)
        self._cors_headers()
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def _binary(self, status: HTTPStatus, data: bytes, content_type: str) -> None:
        self.send_response(status)
        self._cors_headers()
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def _cors_headers(self) -> None:
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Cross-Origin-Resource-Policy", "cross-origin")


def parse_multipart(content_type: str, body: bytes) -> dict[str, UploadedPart]:
    message = BytesParser(policy=default).parsebytes(
        b"Content-Type: "
        + content_type.encode("ascii", errors="strict")
        + b"\r\nMIME-Version: 1.0\r\n\r\n"
        + body
    )
    if not message.is_multipart():
        raise RequestError(HTTPStatus.BAD_REQUEST, "Malformed multipart request")
    parts: dict[str, UploadedPart] = {}
    for part in message.iter_parts():
        name = part.get_param("name", header="content-disposition")
        if not isinstance(name, str) or not name:
            continue
        if name in parts:
            raise RequestError(HTTPStatus.BAD_REQUEST, f"Duplicate multipart field: {name}")
        payload = part.get_payload(decode=True)
        if payload is None:
            payload = b""
        parts[name] = UploadedPart(
            name=name,
            filename=part.get_filename(),
            content_type=part.get_content_type(),
            data=payload,
        )
    return parts


def validate_image_part(part: UploadedPart, label: str, expected_content_type: str) -> None:
    if part.content_type != expected_content_type:
        raise RequestError(
            HTTPStatus.UNSUPPORTED_MEDIA_TYPE,
            f"{label} must be {expected_content_type}, received {part.content_type}",
        )
    if not part.data:
        raise RequestError(HTTPStatus.BAD_REQUEST, f"{label} is empty")


def png_dimensions(data: bytes) -> tuple[int, int]:
    signature = b"\x89PNG\r\n\x1a\n"
    if len(data) < 24 or not data.startswith(signature) or data[12:16] != b"IHDR":
        raise RequestError(HTTPStatus.BAD_REQUEST, "Invalid PNG header")
    width = int.from_bytes(data[16:20], "big")
    height = int.from_bytes(data[20:24], "big")
    if width <= 0 or height <= 0:
        raise RequestError(HTTPStatus.BAD_REQUEST, "PNG dimensions are invalid")
    return width, height


def jpeg_dimensions(data: bytes) -> tuple[int, int]:
    if len(data) < 4 or data[:2] != b"\xff\xd8":
        raise RequestError(HTTPStatus.BAD_REQUEST, "Invalid JPEG header")
    start_of_frame_markers = {
        0xC0, 0xC1, 0xC2, 0xC3,
        0xC5, 0xC6, 0xC7,
        0xC9, 0xCA, 0xCB,
        0xCD, 0xCE, 0xCF,
    }
    no_length_markers = {0x01, 0xD8, 0xD9, *range(0xD0, 0xD8)}
    offset = 2
    while offset < len(data):
        while offset < len(data) and data[offset] == 0xFF:
            offset += 1
        if offset >= len(data):
            break
        marker = data[offset]
        offset += 1
        if marker in no_length_markers:
            continue
        if marker == 0xDA:
            break
        if offset + 2 > len(data):
            break
        segment_length = int.from_bytes(data[offset:offset + 2], "big")
        if segment_length < 2 or offset + segment_length > len(data):
            raise RequestError(HTTPStatus.BAD_REQUEST, "Malformed JPEG segment")
        if marker in start_of_frame_markers:
            if segment_length < 7:
                raise RequestError(HTTPStatus.BAD_REQUEST, "Malformed JPEG SOF segment")
            height = int.from_bytes(data[offset + 3:offset + 5], "big")
            width = int.from_bytes(data[offset + 5:offset + 7], "big")
            if width <= 0 or height <= 0:
                raise RequestError(HTTPStatus.BAD_REQUEST, "JPEG dimensions are invalid")
            return width, height
        offset += segment_length
    raise RequestError(HTTPStatus.BAD_REQUEST, "JPEG dimensions were not found")


def validate_manifest(manifest: Any) -> None:
    if not isinstance(manifest, dict):
        raise RequestError(HTTPStatus.BAD_REQUEST, "Manifest must be an object")
    required_strings = (
        "uploadClientId",
        "taskId",
        "campaignId",
        "capturedAt",
        "layoutVersion",
        "provider",
    )
    for key in required_strings:
        if not isinstance(manifest.get(key), str) or not manifest[key]:
            raise RequestError(HTTPStatus.BAD_REQUEST, f"Manifest field {key} must be a non-empty string")
    try:
        uuid.UUID(manifest["uploadClientId"])
    except ValueError as error:
        raise RequestError(HTTPStatus.BAD_REQUEST, "uploadClientId must be a UUID") from error
    try:
        datetime.fromisoformat(manifest["capturedAt"].replace("Z", "+00:00"))
    except ValueError as error:
        raise RequestError(HTTPStatus.BAD_REQUEST, "capturedAt must be an ISO timestamp") from error
    if manifest["layoutVersion"] not in EXPECTED_LAYOUT_VERSIONS:
        raise RequestError(
            HTTPStatus.BAD_REQUEST,
            f"Unsupported layout version: {manifest['layoutVersion']}",
        )
    original = manifest.get("original")
    if not isinstance(original, dict):
        raise RequestError(HTTPStatus.BAD_REQUEST, "Manifest original must be an object")
    for dimension in ("width", "height"):
        value = original.get(dimension)
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise RequestError(HTTPStatus.BAD_REQUEST, f"original.{dimension} must be positive")
    preview = manifest.get("preview")
    if not isinstance(preview, dict):
        raise RequestError(HTTPStatus.BAD_REQUEST, "Manifest preview must be an object")
    for dimension in ("width", "height"):
        value = preview.get(dimension)
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise RequestError(HTTPStatus.BAD_REQUEST, f"preview.{dimension} must be positive")
    device_pixel_ratio = preview.get("devicePixelRatio")
    if not is_finite_number(device_pixel_ratio) or float(device_pixel_ratio) <= 0:
        raise RequestError(HTTPStatus.BAD_REQUEST, "preview.devicePixelRatio must be positive")
    video_element = preview.get("videoElement")
    validate_rect(video_element, "preview.videoElement")
    assert isinstance(video_element, dict)
    validate_viewport_rect(
        video_element,
        "preview.videoElement",
        int(preview["width"]),
        int(preview["height"]),
    )
    source_to_display_scale = preview.get("sourceToDisplayScale")
    if not is_finite_number(source_to_display_scale) or float(source_to_display_scale) <= 0:
        raise RequestError(
            HTTPStatus.BAD_REQUEST,
            "preview.sourceToDisplayScale must be positive",
        )
    for key in ("sourceDisplayOffsetX", "sourceDisplayOffsetY"):
        if not is_finite_number(preview.get(key)):
            raise RequestError(HTTPStatus.BAD_REQUEST, f"preview.{key} must be numeric")

    model = manifest.get("model")
    if not isinstance(model, dict):
        raise RequestError(HTTPStatus.BAD_REQUEST, "Manifest model must be an object")
    for key in ("name", "sha256"):
        if not isinstance(model.get(key), str) or not model[key]:
            raise RequestError(HTTPStatus.BAD_REQUEST, f"model.{key} must be a non-empty string")
    if not re.fullmatch(r"[0-9a-f]{64}", model["sha256"]):
        raise RequestError(HTTPStatus.BAD_REQUEST, "model.sha256 is invalid")
    for key in ("confidenceThreshold", "nmsIouThreshold"):
        value = manifest.get(key)
        if not is_finite_number(value) or not 0 <= float(value) <= 1:
            raise RequestError(HTTPStatus.BAD_REQUEST, f"{key} must be between 0 and 1")
    for key in ("camera", "telemetry", "regionRects"):
        if not isinstance(manifest.get(key), dict):
            raise RequestError(HTTPStatus.BAD_REQUEST, f"{key} must be an object")
    region_rects = manifest["regionRects"]
    expected_regions = {"completed_hand", "dora_indicators", "melds"}
    if set(region_rects) != expected_regions:
        raise RequestError(
            HTTPStatus.BAD_REQUEST,
            f"regionRects must contain exactly {sorted(expected_regions)}",
        )
    for region_key in expected_regions:
        region = region_rects[region_key]
        if not isinstance(region, dict) or not isinstance(region.get("enabled"), bool):
            raise RequestError(
                HTTPStatus.BAD_REQUEST,
                f"regionRects.{region_key}.enabled must be boolean",
            )
        pixel_rect = region.get("pixel")
        validate_rect(pixel_rect, f"regionRects.{region_key}.pixel")
        assert isinstance(pixel_rect, dict)
        validate_source_region_rect(
            pixel_rect,
            f"regionRects.{region_key}.pixel",
            int(original["width"]),
            int(original["height"]),
            bool(region["enabled"]),
        )
        normalized_rect = region.get("normalized")
        validate_normalized_rect(
            normalized_rect,
            f"regionRects.{region_key}.normalized",
        )
        assert isinstance(normalized_rect, dict)
        expected_normalized = {
            "x": float(pixel_rect["x"]) / int(original["width"]),
            "y": float(pixel_rect["y"]) / int(original["height"]),
            "width": float(pixel_rect["width"]) / int(original["width"]),
            "height": float(pixel_rect["height"]) / int(original["height"]),
        }
        validate_rect_close(
            normalized_rect,
            expected_normalized,
            f"regionRects.{region_key}.normalized",
            tolerance=0.000001,
        )
        display_rect = region.get("display")
        validate_rect(display_rect, f"regionRects.{region_key}.display")
        assert isinstance(display_rect, dict)
        validate_viewport_rect(
            display_rect,
            f"regionRects.{region_key}.display",
            int(preview["width"]),
            int(preview["height"]),
            allow_empty=not bool(region["enabled"]),
        )
        expected_display = {
            "x": (
                float(video_element["x"])
                + float(preview["sourceDisplayOffsetX"])
                + float(pixel_rect["x"]) * float(preview["sourceToDisplayScale"])
            ),
            "y": (
                float(video_element["y"])
                + float(preview["sourceDisplayOffsetY"])
                + float(pixel_rect["y"]) * float(preview["sourceToDisplayScale"])
            ),
            "width": float(pixel_rect["width"]) * float(preview["sourceToDisplayScale"]),
            "height": float(pixel_rect["height"]) * float(preview["sourceToDisplayScale"]),
        }
        if bool(region["enabled"]):
            validate_rect_close(
                display_rect,
                expected_display,
                f"regionRects.{region_key}.display",
                tolerance=0.05,
            )
    detections = manifest.get("detections")
    if not isinstance(detections, list):
        raise RequestError(HTTPStatus.BAD_REQUEST, "detections must be an array")
    if len(detections) > 200:
        raise RequestError(HTTPStatus.BAD_REQUEST, "detections exceeds NanoDet maxDets=200")
    seen_indices: set[int] = set()
    for detection in detections:
        validate_detection(
            detection,
            seen_indices,
            int(original["width"]),
            int(original["height"]),
            int(preview["width"]),
            int(preview["height"]),
        )


def validate_detection(
    detection: Any,
    seen_indices: set[int],
    original_width: int,
    original_height: int,
    preview_width: int,
    preview_height: int,
) -> None:
    if not isinstance(detection, dict):
        raise RequestError(HTTPStatus.BAD_REQUEST, "Each detection must be an object")
    index = detection.get("detectionIndex")
    if not isinstance(index, int) or isinstance(index, bool) or index < 0 or index in seen_indices:
        raise RequestError(HTTPStatus.BAD_REQUEST, "Detection index must be unique and non-negative")
    seen_indices.add(index)
    if detection.get("region") not in {
        "completed_hand",
        "dora_indicators",
        "melds",
        "invalid",
    }:
        raise RequestError(HTTPStatus.BAD_REQUEST, "Detection region is invalid")
    confidence = detection.get("confidence")
    if not is_finite_number(confidence) or not 0 <= float(confidence) <= 1:
        raise RequestError(HTTPStatus.BAD_REQUEST, "Detection confidence is invalid")
    composite = detection.get("composite")
    validate_rect(composite, "detection.composite")
    assert isinstance(composite, dict)
    if (
        float(composite["x"]) < 0
        or float(composite["y"]) < 0
        or float(composite["x"]) + float(composite["width"]) > 320.000001
        or float(composite["y"]) + float(composite["height"]) > 320.000001
    ):
        raise RequestError(HTTPStatus.BAD_REQUEST, "detection.composite exceeds 320 x 320 input")
    original = detection.get("original")
    if original is not None:
        validate_rect(original, "detection.original")
        assert isinstance(original, dict)
        validate_source_region_rect(
            original,
            "detection.original",
            original_width,
            original_height,
            True,
        )
    preview = detection.get("preview")
    if preview is not None:
        validate_rect(preview, "detection.preview")
        assert isinstance(preview, dict)
        validate_viewport_rect(
            preview,
            "detection.preview",
            preview_width,
            preview_height,
        )
    if (original is None) != (preview is None):
        raise RequestError(
            HTTPStatus.BAD_REQUEST,
            "detection.original and detection.preview must both be null or both be rectangles",
        )
    region = detection["region"]
    if region == "invalid" and original is not None:
        raise RequestError(
            HTTPStatus.BAD_REQUEST,
            "Invalid-region detections must not have original or preview coordinates",
        )
    if region != "invalid" and original is None:
        raise RequestError(
            HTTPStatus.BAD_REQUEST,
            "Semantic-region detections require original and preview coordinates",
        )


def is_finite_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def validate_rect(rect: Any, label: str) -> None:
    if not isinstance(rect, dict):
        raise RequestError(HTTPStatus.BAD_REQUEST, f"{label} must be an object")
    for key in ("x", "y", "width", "height"):
        value = rect.get(key)
        if not is_finite_number(value):
            raise RequestError(HTTPStatus.BAD_REQUEST, f"{label}.{key} must be finite numeric")
        if key in {"width", "height"} and float(value) < 0:
            raise RequestError(HTTPStatus.BAD_REQUEST, f"{label}.{key} must not be negative")


def validate_rect_close(
    actual: dict[str, Any],
    expected: dict[str, float],
    label: str,
    tolerance: float,
) -> None:
    for key in ("x", "y", "width", "height"):
        if abs(float(actual[key]) - expected[key]) > tolerance:
            raise RequestError(
                HTTPStatus.BAD_REQUEST,
                f"{label}.{key} does not match its source geometry: "
                f"actual={actual[key]}, expected={expected[key]}",
            )


def validate_viewport_rect(
    rect: dict[str, Any],
    label: str,
    viewport_width: int,
    viewport_height: int,
    *,
    allow_empty: bool = False,
) -> None:
    x = float(rect["x"])
    y = float(rect["y"])
    width = float(rect["width"])
    height = float(rect["height"])
    if x < -0.000001 or y < -0.000001:
        raise RequestError(HTTPStatus.BAD_REQUEST, f"{label} starts outside the viewport")
    if allow_empty:
        if width < 0 or height < 0:
            raise RequestError(HTTPStatus.BAD_REQUEST, f"{label} must not have negative size")
    elif width <= 0 or height <= 0:
        raise RequestError(HTTPStatus.BAD_REQUEST, f"{label} must have positive size")
    if x + width > viewport_width + 0.000001 or y + height > viewport_height + 0.000001:
        raise RequestError(HTTPStatus.BAD_REQUEST, f"{label} exceeds the viewport")


def validate_source_region_rect(
    rect: dict[str, Any],
    label: str,
    source_width: int,
    source_height: int,
    enabled: bool,
) -> None:
    x = float(rect["x"])
    y = float(rect["y"])
    width = float(rect["width"])
    height = float(rect["height"])
    if x < 0 or y < 0:
        raise RequestError(HTTPStatus.BAD_REQUEST, f"{label} starts outside the source frame")
    if enabled and (width <= 0 or height <= 0):
        raise RequestError(HTTPStatus.BAD_REQUEST, f"{label} must have positive size when enabled")
    if x + width > source_width + 0.000001 or y + height > source_height + 0.000001:
        raise RequestError(HTTPStatus.BAD_REQUEST, f"{label} exceeds the source frame")


def validate_normalized_rect(rect: Any, label: str) -> None:
    validate_rect(rect, label)
    assert isinstance(rect, dict)
    for key in ("x", "y", "width", "height"):
        value = float(rect[key])
        if not 0 <= value <= 1:
            raise RequestError(
                HTTPStatus.BAD_REQUEST,
                f"{label}.{key} must be between 0 and 1",
            )
    if float(rect["x"]) + float(rect["width"]) > 1.000001:
        raise RequestError(HTTPStatus.BAD_REQUEST, f"{label} exceeds normalized width")
    if float(rect["y"]) + float(rect["height"]) > 1.000001:
        raise RequestError(HTTPStatus.BAD_REQUEST, f"{label} exceeds normalized height")


def parse_content_length(value: str | None) -> int:
    if value is None:
        raise RequestError(HTTPStatus.LENGTH_REQUIRED, "Content-Length is required")
    try:
        length = int(value)
    except ValueError as error:
        raise RequestError(HTTPStatus.BAD_REQUEST, "Invalid Content-Length") from error
    if length < 0:
        raise RequestError(HTTPStatus.BAD_REQUEST, "Invalid Content-Length")
    return length


def write_atomic(destination: Path, data: bytes) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
        delete=False,
    ) as temporary:
        temporary.write(data)
        temporary.flush()
        os.fsync(temporary.fileno())
        temporary_path = Path(temporary.name)
    try:
        os.replace(temporary_path, destination)
    finally:
        temporary_path.unlink(missing_ok=True)


def safe_date_prefix(captured_at: str) -> str:
    match = re.match(r"^(\d{4})-(\d{2})-(\d{2})", captured_at)
    return "unknown-date" if match is None else "/".join(match.groups())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Mjtensu capture dataset local API")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=8787, type=int)
    parser.add_argument("--storage-root", type=Path, default=DEFAULT_STORAGE_ROOT)
    return parser


def main() -> int:
    arguments = build_parser().parse_args()
    api = CaptureApi(arguments.storage_root)
    handler = type("ConfiguredCaptureRequestHandler", (CaptureRequestHandler,), {"api": api})
    server = ThreadingHTTPServer((arguments.host, arguments.port), handler)
    print(f"Capture API: http://{arguments.host}:{arguments.port}")
    print(f"Storage: {api.storage_root}")
    for campaign in api.campaigns.values():
        print(
            f"Campaign: {campaign['id']} "
            f"({len(campaign['layouts'])} layouts, {len(campaign['tasks'])} tasks, "
            f"sha256={campaign['definitionSha256']})"
        )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("Stopping capture API")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
