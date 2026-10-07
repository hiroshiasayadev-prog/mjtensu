from __future__ import annotations

import argparse
import csv
import hashlib
import json
import mimetypes
import os
import tempfile
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

JP_LABELS = (
    "1m", "2m", "3m", "4m", "5m", "red5m", "6m", "7m", "8m", "9m",
    "1p", "2p", "3p", "4p", "5p", "red5p", "6p", "7p", "8p", "9p",
    "1s", "2s", "3s", "4s", "5s", "red5s", "6s", "7s", "8s", "9s",
    "east", "south", "west", "north", "white", "green", "red",
)
REVIEW_STATUSES = {"high_confidence_mismatch", "ambiguous_mismatch"}
DECISIONS = {"keep_annotation", "change", "remove_annotation", "hold"}

JP_NUMERIC_TILE_LABELS = (
    "1m", "2m", "3m", "4m", "5m", "red5m", "6m", "7m", "8m", "9m",
    "1p", "2p", "3p", "4p", "5p", "red5p", "6p", "7p", "8p", "9p",
    "1s", "2s", "3s", "4s", "5s", "red5s", "6s", "7s", "8s", "9s",
    "east", "south", "west", "north", "white", "green", "red",
)
JP_ALIASES = {"5mr": "red5m", "5pr": "red5p", "5sr": "red5s"}


@dataclass(frozen=True)
class CategoryEntry:
    category_id: int
    raw_name: str
    semantic_label: str | None
    family: str


def _normalize_jp_category(raw_name: str) -> str | None:
    name = raw_name.strip()
    if name == "mahjong-tiles":
        return None
    if name.isdecimal():
        index = int(name)
        if 0 <= index < len(JP_NUMERIC_TILE_LABELS):
            return JP_NUMERIC_TILE_LABELS[index]
        raise ValueError(f"Unsupported JP numeric category: {raw_name!r}")
    name = JP_ALIASES.get(name, name)
    if name in JP_LABELS:
        return name
    raise ValueError(f"Unsupported JP category: {raw_name!r}")


def _build_category_map(payload: dict[str, Any]) -> dict[int, CategoryEntry]:
    raw_categories = payload.get("categories")
    if not isinstance(raw_categories, list):
        raise ValueError("COCO categories must be a list")
    by_id: dict[int, CategoryEntry] = {}
    for raw in raw_categories:
        category_id = int(raw["id"])
        raw_name = str(raw["name"])
        semantic = _normalize_jp_category(raw_name)
        entry = CategoryEntry(
            category_id=category_id,
            raw_name=raw_name,
            semantic_label=semantic,
            family="numeric" if raw_name.strip().isdecimal() else "named",
        )
        previous = by_id.get(category_id)
        if previous is not None and previous.semantic_label != semantic:
            raise ValueError(f"Conflicting duplicate category id {category_id}")
        by_id.setdefault(category_id, entry)
    return by_id


def _choose_category_id(
    entries: dict[int, CategoryEntry], *, approved_label: str, original_category_id: int
) -> int:
    candidates = [entry for entry in entries.values() if entry.semantic_label == approved_label]
    if not candidates:
        raise ValueError(f"No source category maps to approved label {approved_label!r}")
    original = entries.get(original_category_id)
    if original is not None:
        same_family = [entry for entry in candidates if entry.family == original.family]
        if same_family:
            candidates = same_family
    return min(entry.category_id for entry in candidates)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, path)
    except Exception:
        try:
            os.unlink(temp_name)
        except FileNotFoundError:
            pass
        raise


@dataclass(frozen=True)
class Candidate:
    key: str
    record: dict[str, Any]

    def public(self) -> dict[str, Any]:
        keys = (
            "dataset_id", "split", "source_annotation_sha256", "annotation_id",
            "image_id", "image_file", "category_id", "raw_category_name",
            "annotated_label", "annotated_base_label", "bbox", "crop_pixel_box",
            "crop_width", "crop_height", "crop_clipped",
            "base_predicted_label", "base_confidence", "base_margin",
            "base_second_label", "base_second_confidence",
            "red_five_predicted", "red_five_confidence", "red_five_margin",
            "predicted_label", "confidence", "margin", "review_status",
            "consensus_tier", "secondary_review_status", "secondary_predicted_label",
            "secondary_confidence", "secondary_margin",
        )
        return {"key": self.key, **{key: self.record.get(key) for key in keys}}


class ReviewStore:
    def __init__(
        self,
        *,
        audit_jsonl: Path,
        secondary_audit_jsonl: Path | None,
        dataset_root: Path,
        state_path: Path,
        export_root: Path,
    ) -> None:
        self.audit_jsonl = audit_jsonl.resolve()
        self.secondary_audit_jsonl = None if secondary_audit_jsonl is None else secondary_audit_jsonl.resolve()
        self.dataset_root = dataset_root.resolve()
        self.state_path = state_path.resolve()
        self.export_root = export_root.resolve()
        self.secondary_by_key = self._load_secondary()
        self.candidates = self._load_candidates()
        self.by_key = {candidate.key: candidate for candidate in self.candidates}
        self.audit_sha256 = _sha256(self.audit_jsonl)
        self.state = self._load_state()

    def _load_secondary(self) -> dict[str, dict[str, Any]]:
        if self.secondary_audit_jsonl is None or not self.secondary_audit_jsonl.is_file():
            return {}
        rows: dict[str, dict[str, Any]] = {}
        with self.secondary_audit_jsonl.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                record = json.loads(line)
                if record.get("dataset_id") != "coco_mahjong_jp_v2":
                    continue
                if record.get("review_status") not in REVIEW_STATUSES:
                    continue
                key = f"{record['split']}:{int(record['annotation_id'])}"
                rows[key] = record
        return rows

    def _consensus_tier(self, key: str, record: dict[str, Any]) -> str:
        secondary = self.secondary_by_key.get(key)
        if secondary is None:
            return "plain_only"
        same_prediction = record.get("predicted_label") == secondary.get("predicted_label")
        both_high = (
            record.get("review_status") == "high_confidence_mismatch"
            and secondary.get("review_status") == "high_confidence_mismatch"
        )
        if same_prediction and both_high:
            return "dual_high_same"
        if same_prediction:
            return "dual_same_other"
        return "dual_disagree"

    def _load_candidates(self) -> list[Candidate]:
        candidates: list[Candidate] = []
        seen: set[str] = set()
        with self.audit_jsonl.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                record = json.loads(line)
                if record.get("dataset_id") != "coco_mahjong_jp_v2":
                    continue
                if record.get("review_status") not in REVIEW_STATUSES:
                    continue
                split = str(record["split"])
                annotation_id = int(record["annotation_id"])
                key = f"{split}:{annotation_id}"
                if key in seen:
                    raise ValueError(f"Duplicate review candidate: {key}")
                seen.add(key)
                secondary = self.secondary_by_key.get(key)
                record = dict(record)
                record["consensus_tier"] = self._consensus_tier(key, record)
                record["secondary_review_status"] = None if secondary is None else secondary.get("review_status")
                record["secondary_predicted_label"] = None if secondary is None else secondary.get("predicted_label")
                record["secondary_confidence"] = None if secondary is None else secondary.get("confidence")
                record["secondary_margin"] = None if secondary is None else secondary.get("margin")
                candidates.append(Candidate(key=key, record=record))
        tier_priority = {"dual_high_same": 0, "dual_same_other": 1, "dual_disagree": 2, "plain_only": 3}
        priority = {"high_confidence_mismatch": 0, "ambiguous_mismatch": 1}
        candidates.sort(
            key=lambda item: (
                tier_priority[item.record["consensus_tier"]],
                priority[item.record["review_status"]],
                -float(item.record.get("confidence") or 0.0),
                -float(item.record.get("margin") or 0.0),
                str(item.record["split"]),
                int(item.record["annotation_id"]),
            )
        )
        return candidates

    def _load_state(self) -> dict[str, Any]:
        if not self.state_path.is_file():
            return {
                "schema_version": 1,
                "audit_jsonl": str(self.audit_jsonl),
                "audit_sha256": self.audit_sha256,
                "reviews": {},
                "exports": [],
            }
        state = json.loads(self.state_path.read_text(encoding="utf-8"))
        if state.get("schema_version") != 1:
            raise ValueError("Unsupported review state schema")
        if state.get("audit_sha256") != self.audit_sha256:
            raise ValueError(
                "Review state belongs to a different audit result; move it aside before continuing"
            )
        reviews = state.get("reviews")
        if not isinstance(reviews, dict):
            raise ValueError("Invalid review state: reviews must be an object")
        return state

    @property
    def reviews(self) -> dict[str, dict[str, Any]]:
        return self.state["reviews"]

    def bootstrap(self) -> dict[str, Any]:
        counts: dict[str, int] = {}
        for candidate in self.candidates:
            status = candidate.record["review_status"]
            counts[status] = counts.get(status, 0) + 1
        consensus_counts: dict[str, int] = {}
        for candidate in self.candidates:
            tier = candidate.record["consensus_tier"]
            consensus_counts[tier] = consensus_counts.get(tier, 0) + 1
        decision_counts = {"unreviewed": 0, "keep_annotation": 0, "change": 0, "remove_annotation": 0, "hold": 0}
        for candidate in self.candidates:
            review = self.reviews.get(candidate.key)
            if review is None:
                decision_counts["unreviewed"] += 1
            else:
                decision_counts[review["decision"]] += 1
        return {
            "candidates": [candidate.public() for candidate in self.candidates],
            "reviews": self.reviews,
            "labels": list(JP_LABELS),
            "counts": counts,
            "consensus_counts": consensus_counts,
            "decision_counts": decision_counts,
            "audit_sha256": self.audit_sha256,
            "state_path": str(self.state_path),
            "export_root": str(self.export_root),
            "exports": self.state.get("exports", []),
        }

    def save_review(self, key: str, payload: dict[str, Any]) -> dict[str, Any]:
        candidate = self.by_key.get(key)
        if candidate is None:
            raise KeyError(key)
        decision = str(payload.get("decision", ""))
        if decision not in DECISIONS:
            raise ValueError(f"Unsupported decision: {decision!r}")
        approved_label = payload.get("approved_label")
        if approved_label is not None:
            approved_label = str(approved_label)
        if decision == "change":
            if approved_label not in JP_LABELS:
                raise ValueError("change requires approved_label from the JP tile label set")
        else:
            approved_label = None
        note = str(payload.get("note", "")).strip()
        review = {
            "decision": decision,
            "approved_label": approved_label,
            "note": note,
            "annotated_label": candidate.record["annotated_label"],
            "predicted_label": candidate.record["predicted_label"],
            "source_annotation_sha256": candidate.record["source_annotation_sha256"],
        }
        self.reviews[key] = review
        _atomic_write_json(self.state_path, self.state)
        return review

    def delete_review(self, key: str) -> bool:
        if key not in self.by_key:
            raise KeyError(key)
        removed = self.reviews.pop(key, None) is not None
        if removed:
            _atomic_write_json(self.state_path, self.state)
        return removed

    def image_path(self, key: str) -> Path:
        candidate = self.by_key.get(key)
        if candidate is None:
            raise KeyError(key)
        split = str(candidate.record["split"])
        file_name = str(candidate.record["image_file"])
        candidate_path = (self.dataset_root / split / file_name).resolve()
        allowed_root = (self.dataset_root / split).resolve()
        try:
            candidate_path.relative_to(allowed_root)
        except ValueError as error:
            raise ValueError("Image path escapes dataset split") from error
        if not candidate_path.is_file():
            raise FileNotFoundError(candidate_path)
        return candidate_path

    def export_corrected(self) -> dict[str, Any]:
        self.export_root.mkdir(parents=True, exist_ok=True)
        result: dict[str, Any] = {"splits": {}, "total_changes": 0}
        for split in ("train", "valid", "test"):
            source_path = self.dataset_root / split / "_annotations.coco.json"
            if not source_path.is_file():
                continue
            payload = json.loads(source_path.read_text(encoding="utf-8"))
            source_sha256 = _sha256(source_path)
            category_entries = _build_category_map(payload)
            by_id = {int(item["id"]): item for item in payload["annotations"]}
            changes: list[dict[str, Any]] = []
            removals: list[dict[str, Any]] = []
            remove_ids: set[int] = set()
            manifest_rows: list[dict[str, Any]] = []
            for key, review in sorted(self.reviews.items()):
                candidate = self.by_key[key]
                if candidate.record["split"] != split or review["decision"] not in {"change", "remove_annotation"}:
                    continue
                if candidate.record["source_annotation_sha256"] != source_sha256:
                    raise ValueError(
                        f"Source annotation changed since audit for split {split}; refusing export"
                    )
                annotation_id = int(candidate.record["annotation_id"])
                annotation = by_id.get(annotation_id)
                if annotation is None:
                    raise ValueError(f"Annotation {annotation_id} missing from split {split}")
                approved_label = review.get("approved_label")
                manifest_rows.append({
                    "approve": "yes",
                    "decision": review["decision"],
                    "approved_label": approved_label or "__REMOVE__",
                    "review_note": review.get("note", ""),
                    "dataset_id": "coco_mahjong_jp_v2",
                    "split": split,
                    "source_annotation_sha256": source_sha256,
                    "annotation_id": annotation_id,
                    "annotated_label": candidate.record["annotated_label"],
                    "proposed_label": candidate.record["predicted_label"],
                    "confidence": candidate.record["confidence"],
                    "margin": candidate.record["margin"],
                    "image_file": candidate.record["image_file"],
                    "bbox": json.dumps(candidate.record["bbox"], separators=(",", ":")),
                })
                if review["decision"] == "remove_annotation":
                    remove_ids.add(annotation_id)
                    removals.append({
                        "annotation_id": annotation_id,
                        "old_category_id": int(annotation["category_id"]),
                        "annotated_label": candidate.record["annotated_label"],
                        "note": review.get("note", ""),
                    })
                    continue
                approved_label = str(approved_label)
                old_category_id = int(annotation["category_id"])
                new_category_id = _choose_category_id(
                    category_entries,
                    approved_label=approved_label,
                    original_category_id=old_category_id,
                )
                if new_category_id == old_category_id:
                    continue
                annotation["category_id"] = new_category_id
                changes.append({
                    "annotation_id": annotation_id,
                    "old_category_id": old_category_id,
                    "new_category_id": new_category_id,
                    "annotated_label": candidate.record["annotated_label"],
                    "approved_label": approved_label,
                    "note": review.get("note", ""),
                })

            if remove_ids:
                payload["annotations"] = [
                    annotation for annotation in payload["annotations"]
                    if int(annotation["id"]) not in remove_ids
                ]

            output_json = self.export_root / f"{split}_annotations.corrected.coco.json"
            _atomic_write_json(output_json, payload)
            manifest_path = self.export_root / f"{split}_approved_corrections.csv"
            columns = (
                "approve", "decision", "approved_label", "review_note", "dataset_id", "split",
                "source_annotation_sha256", "annotation_id", "annotated_label",
                "proposed_label", "confidence", "margin", "image_file", "bbox",
            )
            with manifest_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=columns)
                writer.writeheader()
                writer.writerows(manifest_rows)
            split_result = {
                "source": str(source_path),
                "source_sha256": source_sha256,
                "corrected_json": str(output_json),
                "corrected_sha256": _sha256(output_json),
                "manifest": str(manifest_path),
                "reviewed_changes": len(manifest_rows),
                "applied_changes": len(changes),
                "removed_annotations": len(removals),
                "changes": changes,
                "removals": removals,
            }
            result["splits"][split] = split_result
            result["total_changes"] += len(changes)
            result.setdefault("total_removals", 0)
            result["total_removals"] += len(removals)

        report_path = self.export_root / "export_report.json"
        _atomic_write_json(report_path, result)
        result["report"] = str(report_path)
        self.state.setdefault("exports", []).append({
            "report": str(report_path),
            "total_changes": result["total_changes"],
        })
        _atomic_write_json(self.state_path, self.state)
        return result


class ReviewHandler(BaseHTTPRequestHandler):
    server_version = "MjtensuCocoReview/1"

    @property
    def store(self) -> ReviewStore:
        return self.server.store  # type: ignore[attr-defined]

    @property
    def static_root(self) -> Path:
        return self.server.static_root  # type: ignore[attr-defined]

    def log_message(self, format: str, *args: Any) -> None:
        print(f"[review-web] {self.address_string()} {format % args}", flush=True)

    def _json(self, status: HTTPStatus, payload: Any) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0"))
        if length <= 0 or length > 64 * 1024:
            raise ValueError("Invalid request body size")
        payload = json.loads(self.rfile.read(length).decode("utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("JSON body must be an object")
        return payload

    def _serve_file(self, path: Path, content_type: str | None = None) -> None:
        body = path.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type or mimetypes.guess_type(path.name)[0] or "application/octet-stream")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "private, max-age=3600")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        try:
            if parsed.path == "/api/bootstrap":
                self._json(HTTPStatus.OK, self.store.bootstrap())
                return
            if parsed.path == "/api/image":
                key = parse_qs(parsed.query).get("key", [""])[0]
                self._serve_file(self.store.image_path(key))
                return
            if parsed.path in {"/", "/index.html"}:
                self._serve_file(self.static_root / "index.html", "text/html; charset=utf-8")
                return
            self._json(HTTPStatus.NOT_FOUND, {"error": "not found"})
        except KeyError:
            self._json(HTTPStatus.NOT_FOUND, {"error": "unknown candidate"})
        except FileNotFoundError as error:
            self._json(HTTPStatus.NOT_FOUND, {"error": str(error)})
        except Exception as error:
            self._json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": str(error)})

    def do_POST(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        try:
            if parsed.path == "/api/review":
                payload = self._read_json()
                key = str(payload.pop("key", ""))
                review = self.store.save_review(key, payload)
                self._json(HTTPStatus.OK, {"key": key, "review": review})
                return
            if parsed.path == "/api/export":
                self._json(HTTPStatus.OK, self.store.export_corrected())
                return
            self._json(HTTPStatus.NOT_FOUND, {"error": "not found"})
        except KeyError:
            self._json(HTTPStatus.NOT_FOUND, {"error": "unknown candidate"})
        except ValueError as error:
            self._json(HTTPStatus.BAD_REQUEST, {"error": str(error)})
        except Exception as error:
            self._json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": str(error)})

    def do_DELETE(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path != "/api/review":
            self._json(HTTPStatus.NOT_FOUND, {"error": "not found"})
            return
        key = parse_qs(parsed.query).get("key", [""])[0]
        try:
            removed = self.store.delete_review(key)
            self._json(HTTPStatus.OK, {"key": key, "removed": removed})
        except KeyError:
            self._json(HTTPStatus.NOT_FOUND, {"error": "unknown candidate"})


def build_parser() -> argparse.ArgumentParser:
    repository_root = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser(description="Review and correct JP-v2 COCO annotation mismatches")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18082)
    parser.add_argument(
        "--audit-jsonl",
        type=Path,
        default=repository_root / ".local/recognition/current_plain_audit/full/flagged_samples.jsonl",
    )
    parser.add_argument(
        "--secondary-audit-jsonl", type=Path, default=None,
        help="Optional second classifier audit JSONL used to rank consensus candidates.",
    )
    parser.add_argument(
        "--dataset-root", type=Path, default=repository_root / "data/coco_mahjong_jp_v2"
    )
    parser.add_argument(
        "--state-path",
        type=Path,
        default=repository_root / ".local/recognition/coco_annotation_review/reviews.json",
    )
    parser.add_argument(
        "--export-root",
        type=Path,
        default=repository_root / ".local/recognition/coco_annotation_review/corrected",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    static_root = Path(__file__).resolve().parent
    store = ReviewStore(
        audit_jsonl=args.audit_jsonl,
        secondary_audit_jsonl=args.secondary_audit_jsonl,
        dataset_root=args.dataset_root,
        state_path=args.state_path,
        export_root=args.export_root,
    )
    server = ThreadingHTTPServer((args.host, args.port), ReviewHandler)
    server.store = store  # type: ignore[attr-defined]
    server.static_root = static_root  # type: ignore[attr-defined]
    print(
        f"[review-web] http://{args.host}:{args.port} candidates={len(store.candidates)} "
        f"state={store.state_path}",
        flush=True,
    )
    server.serve_forever()


if __name__ == "__main__":
    main()
