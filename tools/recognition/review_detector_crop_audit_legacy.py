from __future__ import annotations

import argparse
import io
import json
import math
import mimetypes
import sqlite3
import threading
import webbrowser
from contextlib import closing
from dataclasses import dataclass
from datetime import UTC, datetime
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Sequence
from urllib.parse import parse_qs, quote, urlparse

from PIL import Image

try:
    from .build_tile_classifier_dataset import BASE_LABELS
except ImportError:  # direct script execution
    from build_tile_classifier_dataset import BASE_LABELS


INVALID_REASONS = (
    "background",
    "partial_tile",
    "multi_tile",
    "other",
)
REVIEW_SCHEMA = """
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS review (
    candidate_id    TEXT PRIMARY KEY,
    decision        TEXT NOT NULL CHECK (decision IN ('valid', 'invalid')),
    label           TEXT,
    invalid_reason  TEXT,
    note            TEXT NOT NULL DEFAULT '',
    reviewed_at     TEXT NOT NULL,
    CHECK (
        (decision = 'valid' AND label IS NOT NULL AND invalid_reason IS NULL)
        OR
        (decision = 'invalid' AND label IS NULL AND invalid_reason IS NOT NULL)
    )
);

CREATE INDEX IF NOT EXISTS idx_review_decision
ON review(decision, invalid_reason, reviewed_at);

CREATE TABLE IF NOT EXISTS review_metadata (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS duplicate_review (
    suppression_id TEXT PRIMARY KEY,
    reviewed_at    TEXT NOT NULL
);
"""


@dataclass(frozen=True)
class Candidate:
    index: int
    values: dict[str, Any]

    @property
    def candidate_id(self) -> str:
        return str(self.values["candidate_id"])


@dataclass(frozen=True)
class DuplicateCandidate:
    index: int
    values: dict[str, Any]

    @property
    def suppression_id(self) -> str:
        return str(self.values["suppression_id"])


class ReviewStore:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with closing(self.connect()) as connection:
            connection.executescript(REVIEW_SCHEMA)

    def connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode = WAL")
        connection.execute("PRAGMA synchronous = NORMAL")
        return connection

    def bind_dataset(self, *, detector_run_key: str, source_dataset: str) -> None:
        with closing(self.connect()) as connection:
            existing = connection.execute(
                "SELECT value FROM review_metadata WHERE key='detector_run_key'"
            ).fetchone()
            review_count = int(connection.execute("SELECT COUNT(*) FROM review").fetchone()[0])
            if existing is not None and str(existing[0]) != detector_run_key:
                raise ValueError(
                    "Review sidecar belongs to another detector run: "
                    f"stored={existing[0]}, requested={detector_run_key}"
                )
            if existing is None and review_count > 0:
                raise ValueError(
                    "Existing review sidecar has decisions but no detector_run_key binding; "
                    "refusing to attach it to a candidate dataset implicitly"
                )
            connection.executemany(
                """
                INSERT INTO review_metadata(key, value) VALUES (?, ?)
                ON CONFLICT(key) DO UPDATE SET value = excluded.value
                """,
                [
                    ("detector_run_key", detector_run_key),
                    ("source_dataset", source_dataset),
                ],
            )
            connection.commit()

    def get(self, candidate_id: str) -> dict[str, Any] | None:
        with closing(self.connect()) as connection:
            row = connection.execute(
                """
                SELECT candidate_id, decision, label, invalid_reason, note, reviewed_at
                FROM review WHERE candidate_id = ?
                """,
                (candidate_id,),
            ).fetchone()
        return None if row is None else dict(row)

    def all(self) -> dict[str, dict[str, Any]]:
        with closing(self.connect()) as connection:
            rows = connection.execute(
                "SELECT candidate_id, decision, label, invalid_reason, note, reviewed_at FROM review"
            ).fetchall()
        return {str(row["candidate_id"]): dict(row) for row in rows}

    def save(
        self,
        candidate_id: str,
        *,
        decision: str,
        label: str | None,
        invalid_reason: str | None,
        note: str,
    ) -> dict[str, Any]:
        if decision == "valid":
            if label not in BASE_LABELS:
                raise ValueError("valid review requires one of the 34 base tile labels")
            invalid_reason = None
        elif decision == "invalid":
            if invalid_reason not in INVALID_REASONS:
                raise ValueError(f"invalid review requires one of {INVALID_REASONS}")
            label = None
        else:
            raise ValueError("decision must be valid or invalid")
        reviewed_at = datetime.now(UTC).isoformat()
        with closing(self.connect()) as connection:
            connection.execute(
                """
                INSERT INTO review(candidate_id, decision, label, invalid_reason, note, reviewed_at)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(candidate_id) DO UPDATE SET
                    decision = excluded.decision,
                    label = excluded.label,
                    invalid_reason = excluded.invalid_reason,
                    note = excluded.note,
                    reviewed_at = excluded.reviewed_at
                """,
                (candidate_id, decision, label, invalid_reason, note.strip(), reviewed_at),
            )
            connection.commit()
        result = self.get(candidate_id)
        assert result is not None
        return result

    def delete(self, candidate_id: str) -> bool:
        with closing(self.connect()) as connection:
            deleted = connection.execute(
                "DELETE FROM review WHERE candidate_id = ?", (candidate_id,)
            ).rowcount
            connection.commit()
        return bool(deleted)

    def duplicate_reviews(self) -> set[str]:
        with closing(self.connect()) as connection:
            rows = connection.execute("SELECT suppression_id FROM duplicate_review").fetchall()
        return {str(row[0]) for row in rows}

    def confirm_duplicate(self, suppression_id: str) -> None:
        reviewed_at = datetime.now(UTC).isoformat()
        with closing(self.connect()) as connection:
            connection.execute(
                """
                INSERT INTO duplicate_review(suppression_id, reviewed_at) VALUES (?, ?)
                ON CONFLICT(suppression_id) DO UPDATE SET reviewed_at = excluded.reviewed_at
                """,
                (suppression_id, reviewed_at),
            )
            connection.commit()


class PredictionStore:
    def __init__(self, path: Path | None, model_key: str | None) -> None:
        self.path = path
        self.model_key = model_key
        self.detector_run_key: str | None = None
        self.predictions: dict[str, dict[str, Any]] = {}
        if path is not None and path.is_file():
            self._load()

    def _load(self) -> None:
        assert self.path is not None
        with closing(
            sqlite3.connect(f"{self.path.as_uri()}?mode=ro", uri=True, timeout=30)
        ) as connection:
            connection.row_factory = sqlite3.Row
            table = connection.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name='classifier_prediction'"
            ).fetchone()
            if table is None:
                raise ValueError(f"Classifier audit DB has no classifier_prediction table: {self.path}")
            model_key = self.model_key
            metadata_table = connection.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name='audit_metadata'"
            ).fetchone()
            if metadata_table is not None:
                run_row = connection.execute(
                    "SELECT value FROM audit_metadata WHERE key='source_detector_run_key'"
                ).fetchone()
                if run_row is not None:
                    self.detector_run_key = str(run_row[0])
            if model_key is None and metadata_table is not None:
                row = connection.execute(
                    "SELECT value FROM audit_metadata WHERE key='latest_model_key'"
                ).fetchone()
                if row is not None:
                    model_key = str(row[0])
            if model_key is None:
                row = connection.execute(
                    "SELECT model_key FROM classifier_prediction ORDER BY predicted_at DESC LIMIT 1"
                ).fetchone()
                model_key = None if row is None else str(row[0])
            self.model_key = model_key
            if model_key is None:
                return
            rows = connection.execute(
                """
                SELECT candidate_id, model_key, predicted_label, confidence,
                       invalid_probability, predicted_at
                FROM classifier_prediction
                WHERE model_key = ?
                """,
                (model_key,),
            ).fetchall()
        self.predictions = {str(row["candidate_id"]): dict(row) for row in rows}

    def get(self, candidate_id: str) -> dict[str, Any] | None:
        return self.predictions.get(candidate_id)


class ReviewApplication:
    def __init__(
        self,
        *,
        repository_root: Path,
        dataset_database: Path,
        review_store: ReviewStore,
        prediction_store: PredictionStore,
    ) -> None:
        self.repository_root = repository_root
        self.dataset_database = dataset_database
        self.review_store = review_store
        self.prediction_store = prediction_store
        self.candidates = self._load_candidates()
        self.by_id = {candidate.candidate_id: candidate for candidate in self.candidates}
        self.duplicates = self._load_duplicates()
        self.duplicate_by_id = {
            candidate.suppression_id: candidate for candidate in self.duplicates
        }

    def _load_candidates(self) -> tuple[Candidate, ...]:
        with closing(
            sqlite3.connect(
                f"{self.dataset_database.as_uri()}?mode=ro", uri=True, timeout=60
            )
        ) as connection:
            connection.row_factory = sqlite3.Row
            rows = connection.execute(
                """
                SELECT
                    candidate.candidate_id, candidate.capture_id, candidate.campaign_id,
                    candidate.layout_id, candidate.layout_ordinal,
                    candidate.brightness, candidate.shadow, candidate.region,
                    candidate.source_region_path, candidate.source_composite_path,
                    candidate.detection_index, candidate.detection_confidence,
                    candidate.bbox_x, candidate.bbox_y, candidate.bbox_width,
                    candidate.bbox_height, candidate.crop_width, candidate.crop_height,
                    candidate.suggested_state, candidate.suggested_label,
                    candidate.best_gt_id, candidate.best_gt_label, candidate.best_iou,
                    candidate.best_gt_coverage, candidate.best_detection_coverage,
                    candidate.substantial_gt_count, candidate.gt_json
                FROM candidate AS candidate
                JOIN postprocess_decision AS postprocess
                  ON postprocess.candidate_id = candidate.candidate_id
                WHERE postprocess.status = 'keep'
                ORDER BY
                    CASE suggested_state
                        WHEN 'background' THEN 0
                        WHEN 'multi_gt' THEN 1
                        WHEN 'partial' THEN 2
                        ELSE 3
                    END,
                    candidate.detection_confidence DESC,
                    candidate.candidate_id
                """
            ).fetchall()
        return tuple(Candidate(index=index, values=dict(row)) for index, row in enumerate(rows))

    def _load_duplicates(self) -> tuple[DuplicateCandidate, ...]:
        """Load one audit item per final duplicate group, keyed by the surviving winner."""
        with closing(
            sqlite3.connect(
                f"{self.dataset_database.as_uri()}?mode=ro", uri=True, timeout=60
            )
        ) as connection:
            connection.row_factory = sqlite3.Row
            table = connection.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name='postprocess_decision'"
            ).fetchone()
            if table is None:
                raise ValueError("Detector dataset has no postprocess_decision table; rebuild it")
            rows = connection.execute(
                """
                SELECT
                    removed.detector_run_key,
                    removed.capture_id,
                    removed.region,
                    removed.source_region_path,
                    winner.candidate_id AS winner_candidate_id,
                    winner.detection_index AS winner_detection_index,
                    winner.detection_confidence AS winner_confidence,
                    winner.bbox_x AS winner_bbox_x,
                    winner.bbox_y AS winner_bbox_y,
                    winner.bbox_width AS winner_bbox_width,
                    winner.bbox_height AS winner_bbox_height,
                    removed.candidate_id AS removed_candidate_id,
                    removed.detection_index AS removed_detection_index,
                    removed.detection_confidence AS removed_confidence,
                    removed.bbox_x AS removed_bbox_x,
                    removed.bbox_y AS removed_bbox_y,
                    removed.bbox_width AS removed_bbox_width,
                    removed.bbox_height AS removed_bbox_height,
                    postprocess.overlap_ratio
                FROM postprocess_decision AS postprocess
                JOIN candidate AS removed ON removed.candidate_id = postprocess.candidate_id
                JOIN candidate AS winner ON winner.candidate_id = postprocess.winner_candidate_id
                WHERE postprocess.status = 'remove' AND postprocess.reason = 'duplicate'
                ORDER BY winner.candidate_id, removed.detection_confidence DESC,
                         removed.candidate_id
                """
            ).fetchall()

        groups: dict[str, dict[str, Any]] = {}
        for row in rows:
            winner_candidate_id = str(row["winner_candidate_id"])
            group = groups.get(winner_candidate_id)
            if group is None:
                group = {
                    "suppression_id": (
                        f"duplicate-group:{row['detector_run_key']}:{winner_candidate_id}"
                    ),
                    "detector_run_key": str(row["detector_run_key"]),
                    "capture_id": str(row["capture_id"]),
                    "region": str(row["region"]),
                    "source_region_path": str(row["source_region_path"]),
                    "winner_candidate_id": winner_candidate_id,
                    "winner_detection_index": int(row["winner_detection_index"]),
                    "winner_confidence": float(row["winner_confidence"]),
                    "winner_bbox_x": float(row["winner_bbox_x"]),
                    "winner_bbox_y": float(row["winner_bbox_y"]),
                    "winner_bbox_width": float(row["winner_bbox_width"]),
                    "winner_bbox_height": float(row["winner_bbox_height"]),
                    "removed": [],
                }
                groups[winner_candidate_id] = group
            group["removed"].append(
                {
                    "candidate_id": str(row["removed_candidate_id"]),
                    "detection_index": int(row["removed_detection_index"]),
                    "confidence": float(row["removed_confidence"]),
                    "bbox_x": float(row["removed_bbox_x"]),
                    "bbox_y": float(row["removed_bbox_y"]),
                    "bbox_width": float(row["removed_bbox_width"]),
                    "bbox_height": float(row["removed_bbox_height"]),
                    "overlap_ratio": float(row["overlap_ratio"]),
                }
            )

        values = list(groups.values())
        for group in values:
            group["removed_count"] = len(group["removed"])
            group["max_overlap_ratio"] = max(
                item["overlap_ratio"] for item in group["removed"]
            )
        values.sort(
            key=lambda group: (
                -float(group["max_overlap_ratio"]),
                -float(group["winner_confidence"]),
                str(group["suppression_id"]),
            )
        )
        return tuple(
            DuplicateCandidate(index=index, values=value)
            for index, value in enumerate(values)
        )

    def summary(self) -> dict[str, Any]:
        reviews = self.review_store.all()
        relevant = {key: value for key, value in reviews.items() if key in self.by_id}
        invalid_counts = {reason: 0 for reason in INVALID_REASONS}
        valid_count = 0
        invalid_count = 0
        for review in relevant.values():
            if review["decision"] == "valid":
                valid_count += 1
            else:
                invalid_count += 1
                reason = str(review["invalid_reason"])
                if reason in invalid_counts:
                    invalid_counts[reason] += 1
        duplicate_reviews = self.review_store.duplicate_reviews()
        relevant_duplicate_reviews = {
            value for value in duplicate_reviews if value in self.duplicate_by_id
        }
        return {
            "candidate_count": len(self.candidates),
            "reviewed_count": len(relevant),
            "unreviewed_count": len(self.candidates) - len(relevant),
            "valid_count": valid_count,
            "invalid_count": invalid_count,
            "invalid_reason_counts": invalid_counts,
            "classifier_model_key": self.prediction_store.model_key,
            "classifier_prediction_count": len(self.prediction_store.predictions),
            "duplicate_count": len(self.duplicates),
            "duplicate_reviewed_count": len(relevant_duplicate_reviews),
            "duplicate_unreviewed_count": len(self.duplicates) - len(relevant_duplicate_reviews),
            "labels": list(BASE_LABELS),
            "invalid_reasons": list(INVALID_REASONS),
        }

    def candidate_payload(self, global_index: int) -> dict[str, Any]:
        if global_index < 0 or global_index >= len(self.candidates):
            raise IndexError(global_index)
        candidate = self.candidates[global_index]
        values = candidate.values
        return {
            "global_index": global_index,
            **values,
            "gt": json.loads(str(values["gt_json"])),
            "crop_url": f"/api/crop?candidate_id={quote(candidate.candidate_id, safe='')}",
            "region_url": f"/api/region?candidate_id={quote(candidate.candidate_id, safe='')}",
            "review": self.review_store.get(candidate.candidate_id),
            "prediction": self.prediction_store.get(candidate.candidate_id),
        }

    def filtered_indices(
        self,
        *,
        review_filter: str,
        state_filter: str,
        classifier_filter: str,
        confidence_below: float,
    ) -> list[int]:
        reviews = self.review_store.all()
        result: list[int] = []
        for candidate in self.candidates:
            values = candidate.values
            review = reviews.get(candidate.candidate_id)
            prediction = self.prediction_store.get(candidate.candidate_id)
            if review_filter == "unreviewed" and review is not None:
                continue
            if review_filter == "reviewed" and review is None:
                continue
            if review_filter == "valid" and (review is None or review["decision"] != "valid"):
                continue
            if review_filter == "invalid" and (review is None or review["decision"] != "invalid"):
                continue
            if state_filter != "all" and values["suggested_state"] != state_filter:
                continue
            if not classifier_matches_filter(
                classifier_filter,
                candidate=values,
                review=review,
                prediction=prediction,
                confidence_below=confidence_below,
            ):
                continue
            result.append(candidate.index)
        return result

    def duplicate_indices(self, *, review_filter: str) -> list[int]:
        reviewed = self.review_store.duplicate_reviews()
        result: list[int] = []
        for candidate in self.duplicates:
            is_reviewed = candidate.suppression_id in reviewed
            if review_filter == "unreviewed" and is_reviewed:
                continue
            if review_filter == "reviewed" and not is_reviewed:
                continue
            result.append(candidate.index)
        return result

    def duplicate_payload(self, global_index: int) -> dict[str, Any]:
        if global_index < 0 or global_index >= len(self.duplicates):
            raise IndexError(global_index)
        candidate = self.duplicates[global_index]
        values = candidate.values
        suppression_id = quote(candidate.suppression_id, safe="")
        removed = [
            {
                **item,
                "crop_url": (
                    f"/api/duplicate/crop?suppression_id={suppression_id}"
                    f"&which=removed&index={index}"
                ),
            }
            for index, item in enumerate(values["removed"])
        ]
        return {
            "global_index": global_index,
            **values,
            "removed": removed,
            "reviewed": candidate.suppression_id in self.review_store.duplicate_reviews(),
            "region_url": f"/api/duplicate/region?suppression_id={suppression_id}",
            "winner_crop_url": (
                f"/api/duplicate/crop?suppression_id={suppression_id}&which=winner"
            ),
        }

    def duplicate_region_asset(self, suppression_id: str) -> tuple[bytes, str]:
        candidate = self.duplicate_by_id.get(suppression_id)
        if candidate is None:
            raise KeyError(suppression_id)
        value = str(candidate.values["source_region_path"])
        path = Path(value)
        if not path.is_absolute():
            path = self.repository_root / path
        path = path.resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        return path.read_bytes(), content_type

    def duplicate_crop(
        self,
        suppression_id: str,
        which: str,
        removed_index: int = 0,
    ) -> bytes:
        candidate = self.duplicate_by_id.get(suppression_id)
        if candidate is None:
            raise KeyError(suppression_id)
        values = candidate.values
        if which == "winner":
            candidate_id = str(values["winner_candidate_id"])
        elif which == "removed":
            removed = values["removed"]
            if removed_index < 0 or removed_index >= len(removed):
                raise IndexError(removed_index)
            candidate_id = str(removed[removed_index]["candidate_id"])
        else:
            raise ValueError("which must be winner or removed")
        with closing(
            sqlite3.connect(
                f"{self.dataset_database.as_uri()}?mode=ro", uri=True, timeout=30
            )
        ) as connection:
            row = connection.execute(
                "SELECT image_png FROM candidate WHERE candidate_id = ?",
                (candidate_id,),
            ).fetchone()
        if row is None:
            raise KeyError(candidate_id)
        return bytes(row[0])

    def load_crop(self, candidate_id: str) -> bytes:
        if candidate_id not in self.by_id:
            raise KeyError(candidate_id)
        with closing(
            sqlite3.connect(
                f"{self.dataset_database.as_uri()}?mode=ro", uri=True, timeout=30
            )
        ) as connection:
            row = connection.execute(
                "SELECT image_png FROM candidate WHERE candidate_id = ?", (candidate_id,)
            ).fetchone()
        if row is None:
            raise KeyError(candidate_id)
        return bytes(row[0])

    def region_asset(self, candidate_id: str) -> tuple[bytes, str]:
        candidate = self.by_id.get(candidate_id)
        if candidate is None:
            raise KeyError(candidate_id)
        value = str(candidate.values["source_region_path"])
        path = Path(value)
        if not path.is_absolute():
            path = self.repository_root / path
        path = path.resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        return path.read_bytes(), content_type


def classifier_matches_filter(
    filter_name: str,
    *,
    candidate: dict[str, Any],
    review: dict[str, Any] | None,
    prediction: dict[str, Any] | None,
    confidence_below: float,
) -> bool:
    if filter_name == "all":
        return True
    if filter_name == "no_prediction":
        return prediction is None
    if prediction is None:
        return False
    predicted = str(prediction["predicted_label"])
    confidence = float(prediction["confidence"])
    if filter_name == "uncertain":
        return confidence < confidence_below
    if filter_name == "predict_invalid":
        return predicted == "invalid"
    if filter_name == "review_disagreement":
        if review is None:
            return False
        expected = "invalid" if review["decision"] == "invalid" else str(review["label"])
        return predicted != expected
    if filter_name == "strong_gt_disagreement":
        return (
            review is None
            and candidate["suggested_state"] == "single_gt"
            and candidate["suggested_label"] is not None
            and predicted != str(candidate["suggested_label"])
        )
    if filter_name == "suspected_invalid_predicted_tile":
        return (
            review is None
            and candidate["suggested_state"] != "single_gt"
            and predicted != "invalid"
        )
    raise ValueError(f"Unsupported classifier filter: {filter_name}")


def parse_args() -> argparse.Namespace:
    repository_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(
        description=(
            "Browser review UI for NanoDet-derived classifier crops. It overlays the detector "
            "bbox, corrected human GT, and optional gray35 classifier prediction on the source "
            "region image. Human decisions remain in a separate SQLite sidecar."
        )
    )
    parser.add_argument("--repository-root", type=Path, default=repository_root)
    parser.add_argument(
        "--database",
        type=Path,
        help="Defaults to .local/recognition/detector_crop_dataset/dataset.sqlite.",
    )
    parser.add_argument(
        "--review-database",
        type=Path,
        help=(
            "Optional explicit review sidecar. By default the detector run key is read from "
            "dataset.sqlite and reviews.<detector_run_key>.sqlite is used."
        ),
    )
    parser.add_argument(
        "--classifier-audit-database",
        type=Path,
        help="Optional predictions sidecar produced by audit_detector_crop_classifier.py.",
    )
    parser.add_argument("--model-key", help="Optional classifier model key to display.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument("--open-browser", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repository_root = args.repository_root.resolve()
    root = repository_root / ".local" / "recognition" / "detector_crop_dataset"
    database = (args.database or root / "dataset.sqlite").resolve()
    if not database.is_file():
        raise FileNotFoundError(database)
    detector_run_key = load_detector_run_key(database)
    review_database = (
        args.review_database.resolve()
        if args.review_database is not None
        else root / f"reviews.{detector_run_key}.sqlite"
    )
    prediction_database = (
        None
        if args.classifier_audit_database is None
        else args.classifier_audit_database.resolve()
    )
    if prediction_database is None:
        default_prediction = root / "classifier_audit.sqlite"
        if default_prediction.is_file():
            prediction_database = default_prediction.resolve()
    if not 1 <= int(args.port) <= 65535:
        raise ValueError("--port must be in [1,65535]")

    reviews = ReviewStore(review_database)
    reviews.bind_dataset(detector_run_key=detector_run_key, source_dataset=str(database))
    predictions = PredictionStore(prediction_database, args.model_key)
    if (
        predictions.detector_run_key is not None
        and predictions.detector_run_key != detector_run_key
    ):
        raise ValueError(
            "Classifier audit sidecar belongs to another detector run: "
            f"stored={predictions.detector_run_key}, requested={detector_run_key}"
        )
    application = ReviewApplication(
        repository_root=repository_root,
        dataset_database=database,
        review_store=reviews,
        prediction_store=predictions,
    )
    server = ThreadingHTTPServer((str(args.host), int(args.port)), make_handler(application))
    url = f"http://{args.host}:{args.port}/"
    print(f"[detector-crop-review] candidates={len(application.candidates):,}")
    print(f"[detector-crop-review] reviews={review_database}")
    print(f"[detector-crop-review] classifier_model={predictions.model_key or 'none'}")
    print(f"[detector-crop-review] open: {url}")
    if args.open_browser:
        threading.Timer(0.4, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[detector-crop-review] stopped")
    finally:
        server.server_close()


def load_detector_run_key(database: Path) -> str:
    with closing(
        sqlite3.connect(f"{database.as_uri()}?mode=ro", uri=True, timeout=30)
    ) as connection:
        row = connection.execute(
            "SELECT value FROM dataset_metadata WHERE key='detector_run_key'"
        ).fetchone()
    if row is None:
        raise ValueError(f"Candidate DB has no detector_run_key metadata: {database}")
    return str(row[0])


def make_handler(application: ReviewApplication) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        server_version = "MjtensuDetectorCropReview/1.0"

        def do_GET(self) -> None:  # noqa: N802
            try:
                parsed = urlparse(self.path)
                if parsed.path == "/":
                    self.send_bytes(PAGE_HTML.encode("utf-8"), "text/html; charset=utf-8")
                    return
                if parsed.path == "/duplicates":
                    self.send_bytes(DUPLICATE_PAGE_HTML.encode("utf-8"), "text/html; charset=utf-8")
                    return
                if parsed.path == "/api/summary":
                    self.send_json(application.summary())
                    return
                if parsed.path == "/api/indices":
                    query = parse_qs(parsed.query)
                    review_filter = one(query, "review", "unreviewed")
                    state_filter = one(query, "state", "all")
                    classifier_filter = one(query, "classifier", "all")
                    confidence_below = float(one(query, "confidence_below", "0.80"))
                    if review_filter not in ("all", "unreviewed", "reviewed", "valid", "invalid"):
                        raise ValueError("unsupported review filter")
                    if state_filter not in ("all", "single_gt", "multi_gt", "partial", "background"):
                        raise ValueError("unsupported state filter")
                    if classifier_filter not in (
                        "all", "no_prediction", "uncertain", "predict_invalid",
                        "review_disagreement", "strong_gt_disagreement",
                        "suspected_invalid_predicted_tile",
                    ):
                        raise ValueError("unsupported classifier filter")
                    indices = application.filtered_indices(
                        review_filter=review_filter,
                        state_filter=state_filter,
                        classifier_filter=classifier_filter,
                        confidence_below=confidence_below,
                    )
                    self.send_json({"indices": indices, "count": len(indices)})
                    return
                if parsed.path == "/api/candidate":
                    query = parse_qs(parsed.query)
                    index = int(one(query, "index"))
                    self.send_json(application.candidate_payload(index))
                    return
                if parsed.path == "/api/duplicate/indices":
                    query = parse_qs(parsed.query)
                    review_filter = one(query, "review", "unreviewed")
                    if review_filter not in ("all", "unreviewed", "reviewed"):
                        raise ValueError("unsupported duplicate review filter")
                    indices = application.duplicate_indices(review_filter=review_filter)
                    self.send_json({"indices": indices, "count": len(indices)})
                    return
                if parsed.path == "/api/duplicate":
                    query = parse_qs(parsed.query)
                    index = int(one(query, "index"))
                    self.send_json(application.duplicate_payload(index))
                    return
                if parsed.path == "/api/duplicate/region":
                    query = parse_qs(parsed.query)
                    suppression_id = one(query, "suppression_id")
                    content, content_type = application.duplicate_region_asset(suppression_id)
                    self.send_bytes(content, content_type)
                    return
                if parsed.path == "/api/duplicate/crop":
                    query = parse_qs(parsed.query)
                    suppression_id = one(query, "suppression_id")
                    which = one(query, "which")
                    removed_index = int(one(query, "index", "0"))
                    self.send_bytes(
                        application.duplicate_crop(suppression_id, which, removed_index),
                        "image/png",
                    )
                    return
                if parsed.path == "/api/crop":
                    query = parse_qs(parsed.query)
                    candidate_id = one(query, "candidate_id")
                    self.send_bytes(application.load_crop(candidate_id), "image/png")
                    return
                if parsed.path == "/api/region":
                    query = parse_qs(parsed.query)
                    candidate_id = one(query, "candidate_id")
                    content, content_type = application.region_asset(candidate_id)
                    self.send_bytes(content, content_type)
                    return
                self.send_error(HTTPStatus.NOT_FOUND)
            except (ValueError, IndexError, KeyError, FileNotFoundError) as error:
                self.send_json({"error": str(error)}, status=HTTPStatus.BAD_REQUEST)
            except Exception as error:  # pragma: no cover
                self.send_json(
                    {"error": f"{type(error).__name__}: {error}"},
                    status=HTTPStatus.INTERNAL_SERVER_ERROR,
                )

        def do_POST(self) -> None:  # noqa: N802
            try:
                parsed = urlparse(self.path)
                payload = self.read_json()
                if parsed.path == "/api/review":
                    candidate_id = str(payload.get("candidate_id", ""))
                    if candidate_id not in application.by_id:
                        raise KeyError(candidate_id)
                    review = application.review_store.save(
                        candidate_id,
                        decision=str(payload.get("decision", "")),
                        label=None if payload.get("label") in (None, "") else str(payload["label"]),
                        invalid_reason=(
                            None
                            if payload.get("invalid_reason") in (None, "")
                            else str(payload["invalid_reason"])
                        ),
                        note=str(payload.get("note", "")),
                    )
                    self.send_json({"review": review, "summary": application.summary()})
                    return
                if parsed.path == "/api/review/delete":
                    candidate_id = str(payload.get("candidate_id", ""))
                    if candidate_id not in application.by_id:
                        raise KeyError(candidate_id)
                    deleted = application.review_store.delete(candidate_id)
                    self.send_json({"deleted": deleted, "summary": application.summary()})
                    return
                if parsed.path == "/api/duplicate/review":
                    suppression_id = str(payload.get("suppression_id", ""))
                    if suppression_id not in application.duplicate_by_id:
                        raise KeyError(suppression_id)
                    application.review_store.confirm_duplicate(suppression_id)
                    self.send_json({"confirmed": True, "summary": application.summary()})
                    return
                self.send_error(HTTPStatus.NOT_FOUND)
            except (ValueError, KeyError) as error:
                self.send_json({"error": str(error)}, status=HTTPStatus.BAD_REQUEST)
            except Exception as error:  # pragma: no cover
                self.send_json(
                    {"error": f"{type(error).__name__}: {error}"},
                    status=HTTPStatus.INTERNAL_SERVER_ERROR,
                )

        def log_message(self, format: str, *args: Any) -> None:
            return

        def read_json(self) -> dict[str, Any]:
            length = int(self.headers.get("Content-Length", "0"))
            value = json.loads(self.rfile.read(length).decode("utf-8"))
            if not isinstance(value, dict):
                raise ValueError("Expected JSON object")
            return value

        def send_json(self, payload: Any, *, status: int = HTTPStatus.OK) -> None:
            raw = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(raw)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(raw)

        def send_bytes(self, content: bytes, content_type: str) -> None:
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(content)))
            self.send_header("Cache-Control", "private, max-age=3600")
            self.end_headers()
            self.wfile.write(content)

    return Handler


def one(query: dict[str, list[str]], key: str, default: str | None = None) -> str:
    values = query.get(key)
    if not values:
        if default is None:
            raise ValueError(f"Missing query parameter: {key}")
        return default
    return values[0]


PAGE_HTML = r"""<!doctype html>
<html lang="ja">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Detector crop audit</title>
<style>
:root { color-scheme: light dark; font-family: system-ui,-apple-system,"Segoe UI",sans-serif; }
* { box-sizing: border-box; }
body { margin:0; background:Canvas; color:CanvasText; }
header { position:sticky; top:0; z-index:5; display:flex; flex-wrap:wrap; gap:9px; align-items:center; padding:10px 12px; border-bottom:1px solid color-mix(in srgb,CanvasText 22%,transparent); background:Canvas; }
header .grow { flex:1; }
.saved-banner { font-weight:800; color:#28a745; }
.hint-box { border-left:4px solid #888; padding:8px 10px; margin:8px 0; background:color-mix(in srgb,CanvasText 6%,transparent); }
.human-truth { border:2px solid #28a745; }
select,button,input,textarea { font:inherit; }
select,button,input { min-height:36px; }
button { cursor:pointer; }
main { max-width:1600px; margin:0 auto; padding:14px; display:grid; grid-template-columns:minmax(520px,1.35fr) minmax(390px,.8fr); gap:16px; }
.viewer { display:grid; gap:10px; min-width:0; }
.canvas-wrap { position:relative; min-height:520px; border:1px solid color-mix(in srgb,CanvasText 20%,transparent); background:#111; display:grid; place-items:center; overflow:auto; }
canvas { max-width:100%; max-height:72vh; }
.crop-row { display:flex; gap:12px; align-items:center; border:1px solid color-mix(in srgb,CanvasText 20%,transparent); padding:10px; }
.crop-row img { width:150px; height:150px; object-fit:contain; background:#111; }
.hero { font-size:26px; font-weight:800; }
.badge { display:inline-block; border:1px solid currentColor; border-radius:999px; padding:2px 8px; font-size:12px; margin-right:5px; }
.panel { border:1px solid color-mix(in srgb,CanvasText 20%,transparent); padding:11px; margin-bottom:10px; }
.grid { display:grid; grid-template-columns:1fr 1fr; gap:7px; }
.grid button { min-height:48px; text-align:left; }
.grid .wide { grid-column:1/-1; }
label.control { display:grid; gap:4px; margin:8px 0; }
textarea { width:100%; min-height:56px; }
pre { white-space:pre-wrap; overflow-wrap:anywhere; font:12px/1.4 ui-monospace,Consolas,monospace; }
.error { color:#d33; font-weight:700; }
.status { opacity:.8; font-size:13px; }
@media(max-width:980px){main{grid-template-columns:1fr}.canvas-wrap{min-height:360px}}
</style>
</head>
<body>
<header>
<strong>Detector crop human review</strong><a href="/duplicates">重複除去audit</a><span id="progress" class="status"></span><span id="last-saved" class="saved-banner"></span><span class="grow"></span>
<label>表示 <select id="review-filter"><option value="unreviewed">未判定だけ</option><option value="all">全件</option><option value="reviewed">判定済み</option><option value="valid">human valid</option><option value="invalid">human invalid</option></select></label>
<label>geometry hint <select id="state-filter"><option value="all">all</option><option value="background">background</option><option value="multi_gt">multi</option><option value="partial">partial</option><option value="single_gt">single</option></select></label>
<label id="classifier-control">gray35 <select id="classifier-filter"><option value="all">all</option><option value="review_disagreement">人力と不一致</option><option value="strong_gt_disagreement">strong GTと不一致</option><option value="suspected_invalid_predicted_tile">怪しいcropを牌判定</option><option value="predict_invalid">invalid判定</option><option value="uncertain">低confidence</option><option value="no_prediction">予測なし</option></select></label>
<label id="confidence-control">conf&lt; <input id="confidence-below" type="number" min="0" max="1" step="0.05" value="0.80" style="width:72px"></label>
<button id="undo-last" disabled>直前の判定を取消</button><button id="prev">←</button><button id="next">→</button>
</header>
<main>
<section class="viewer">
  <div class="canvas-wrap"><canvas id="region-canvas"></canvas></div>
  <div class="crop-row"><img id="crop" alt="NanoDet crop"><div><div id="hero" class="hero"></div><div id="badges"></div><div id="geometry-explain" class="hint-box"></div><div id="prediction"></div></div></div>
</section>
<section>
  <div id="error" class="error"></div>
  <div class="panel human-truth">
    <strong>人力確定 — ここだけがgray35の教師ラベルになる</strong>
    <div class="hint-box">crop単体を見て「1枚の牌として分類させてよいか」を決める。未判定表示では、保存するとその候補は一覧から外れて次へ自動移動する。</div>
    <label class="control">validの場合の牌種<select id="label"></select></label>
    <div class="grid">
      <button class="wide" data-valid="1"><b>V</b> — valid / 選択ラベル</button>
      <button data-invalid="background"><b>B</b> — 背景</button>
      <button data-invalid="partial_tile"><b>P</b> — 牌が欠けすぎ</button>
      <button data-invalid="multi_tile"><b>M</b> — 複数牌</button>
      <button class="wide" data-invalid="other"><b>O</b> — その他invalid</button>
    </div>
    <label class="control">note<textarea id="note"></textarea></label>
    <button id="clear">この候補の判定を解除</button> <strong id="current-review"></strong>
  </div>
  <div class="panel"><strong>metadata</strong><pre id="metadata"></pre></div>
  <div class="status">キー: V/B/P/M/O = 人力確定、←→ = 移動。緑=既存human GT、赤=今回のNanoDet bbox。confidence比較で自動除去される重複bboxはこのreview対象外で、別の重複除去auditにのみ出る。geometry hintはGTとの重なりから機械的に出した参考値で、教師ラベルではない。</div>
</section>
</main>
<script>
const state={indices:[],position:0,candidate:null,summary:null,labels:[],lastSavedCandidateId:null,lastSavedText:''};
const $=id=>document.getElementById(id);
async function api(url,opts){const r=await fetch(url,opts);const d=await r.json();if(!r.ok)throw new Error(d.error||r.statusText);return d;}
function esc(v){return String(v).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));}
async function refreshSummary(){state.summary=await api('/api/summary');state.labels=state.summary.labels;$('label').innerHTML=state.labels.map(x=>`<option value="${esc(x)}">${esc(x)}</option>`).join('');const hasPred=state.summary.classifier_prediction_count>0;$('classifier-filter').disabled=!hasPred;$('confidence-below').disabled=!hasPred;$('classifier-control').title=hasPred?'gray35予測で絞り込み':'gray35未学習。初回human reviewでは使用しない';$('confidence-control').title=$('classifier-control').title;if(!hasPred)$('classifier-filter').value='all';renderProgress();}
function renderProgress(){const s=state.summary;if(!s)return;$('progress').textContent=`human review ${s.reviewed_count}/${s.candidate_count} · valid ${s.valid_count} · invalid ${s.invalid_count}`;$('last-saved').textContent=state.lastSavedText;$('undo-last').disabled=!state.lastSavedCandidateId;}
async function refreshIndices(){const q=new URLSearchParams({review:$('review-filter').value,state:$('state-filter').value,classifier:$('classifier-filter').value,confidence_below:$('confidence-below').value});const d=await api('/api/indices?'+q);state.indices=d.indices;state.position=Math.min(state.position,Math.max(0,state.indices.length-1));if(!state.indices.length){state.candidate=null;$('hero').textContent='該当なし';clearCanvas();return;}await loadCurrent();}
async function loadCurrent(){const index=state.indices[state.position];state.candidate=await api('/api/candidate?index='+index);renderCandidate();}
function clearCanvas(){const c=$('region-canvas');const x=c.getContext('2d');x.clearRect(0,0,c.width,c.height);}
function rotatedCorners(g){const a=g.angleDeg*Math.PI/180,co=Math.cos(a),si=Math.sin(a),hw=g.width/2,hh=g.height/2;return [[-hw,-hh],[hw,-hh],[hw,hh],[-hw,hh]].map(([x,y])=>[g.centerX+co*x-si*y,g.centerY+si*x+co*y]);}
async function drawOverlay(c){const img=new Image();img.src=c.region_url+'&t='+Date.now();await img.decode();const canvas=$('region-canvas'),ctx=canvas.getContext('2d');canvas.width=img.naturalWidth;canvas.height=img.naturalHeight;ctx.drawImage(img,0,0);ctx.lineWidth=Math.max(2,img.naturalWidth/500*2);ctx.font=`${Math.max(14,img.naturalWidth/70)}px system-ui`;for(const g of c.gt){const pts=rotatedCorners(g);ctx.beginPath();ctx.moveTo(...pts[0]);for(let i=1;i<pts.length;i++)ctx.lineTo(...pts[i]);ctx.closePath();ctx.strokeStyle='#28df72';ctx.stroke();ctx.fillStyle='#28df72';ctx.fillText(g.label,pts[0][0],Math.max(14,pts[0][1]-3));}ctx.strokeStyle='#ff3b30';ctx.lineWidth=Math.max(3,img.naturalWidth/500*3);ctx.strokeRect(c.bbox_x,c.bbox_y,c.bbox_width,c.bbox_height);const p=c.prediction;if(p){const text=`gray35 ${p.predicted_label} ${(p.confidence*100).toFixed(1)}%`;ctx.fillStyle='#ffd60a';ctx.strokeStyle='#000';ctx.lineWidth=4;const tx=c.bbox_x,ty=Math.max(18,c.bbox_y-5);ctx.strokeText(text,tx,ty);ctx.fillText(text,tx,ty);}}
function renderCandidate(){const c=state.candidate;if(!c)return;$('error').textContent='';$('crop').src=c.crop_url+'&t='+Date.now();const pred=c.prediction;$('hero').textContent=`geometry hint: ${c.suggested_state}${c.suggested_label?' → '+c.suggested_label:''}`;$('badges').innerHTML=`<span class="badge">NanoDet ${(c.detection_confidence*100).toFixed(1)}%</span><span class="badge">IoU ${c.best_iou.toFixed(3)}</span><span class="badge">GT cover ${c.best_gt_coverage.toFixed(3)}</span><span class="badge">crop purity ${c.best_detection_coverage.toFixed(3)}</span>`;$('geometry-explain').textContent=geometryExplanation(c);$('prediction').textContent=pred?`gray35参考予測: ${pred.predicted_label} conf=${Number(pred.confidence).toFixed(4)} invalid=${Number(pred.invalid_probability).toFixed(4)}`:'gray35はまだ未学習/未推論。初回human reviewではここは使わない。';const defaultLabel=c.review?.label||c.suggested_label||(pred&&pred.predicted_label!=='invalid'?pred.predicted_label:'1m');if(state.labels.includes(defaultLabel))$('label').value=defaultLabel;$('note').value=c.review?.note||'';$('current-review').textContent=c.review?(c.review.decision==='valid'?`保存済み: VALID → ${c.review.label}`:`保存済み: INVALID / ${c.review.invalid_reason}`):'この候補はまだ人力未判定';$('metadata').textContent=[`filter position: ${state.position+1}/${state.indices.length}`,`candidate: ${c.candidate_id}`,`capture: ${c.capture_id}`,`campaign/layout: ${c.campaign_id}/${c.layout_id} (${c.layout_ordinal+1})`,`condition: ${c.brightness}/${c.shadow}`,`region: ${c.region}`,`detection index/conf: ${c.detection_index} / ${c.detection_confidence}`,`geometry hint: ${c.suggested_state} / ${c.suggested_label||''}`,`best GT: ${c.best_gt_label||''} id=${c.best_gt_id||''}`,`substantial GT count: ${c.substantial_gt_count}`].join('\n');drawOverlay(c).catch(e=>$('error').textContent=String(e));}
function geometryExplanation(c){if(c.suggested_state==='single_gt')return `参考判定: NanoDet cropが主に1個のhuman GTを含む。GT cover=${c.best_gt_coverage.toFixed(3)}, crop purity=${c.best_detection_coverage.toFixed(3)}。これは教師ラベルではない。`;if(c.suggested_state==='multi_gt')return `参考判定: NanoDet bboxが30%以上含むhuman GTが${c.substantial_gt_count}個あるためmulti候補。実cropを見て人間が確定する。`;if(c.suggested_state==='background')return `参考判定: 最も重なるhuman GTでもGT cover=${c.best_gt_coverage.toFixed(3)} (<0.10) のためbackground候補。`;return `参考判定: human GTとの重なりがsingle判定基準を満たさないためpartial候補。実cropを見て人間が確定する。`;}
async function save(decision,invalidReason=null){const c=state.candidate;if(!c)return;try{const d=await api('/api/review',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({candidate_id:c.candidate_id,decision,label:decision==='valid'?$('label').value:null,invalid_reason:invalidReason,note:$('note').value})});state.summary=d.summary;state.lastSavedCandidateId=c.candidate_id;state.lastSavedText=decision==='valid'?`✓ 保存: ${c.candidate_id} = VALID/${$('label').value}`:`✓ 保存: ${c.candidate_id} = INVALID/${invalidReason}`;renderProgress();await refreshIndices();}catch(e){$('error').textContent=String(e);}}
async function clearReview(){const c=state.candidate;if(!c)return;const d=await api('/api/review/delete',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({candidate_id:c.candidate_id})});state.summary=d.summary;state.lastSavedText=`判定解除: ${c.candidate_id}`;renderProgress();await refreshIndices();}
async function undoLast(){const candidateId=state.lastSavedCandidateId;if(!candidateId)return;try{const d=await api('/api/review/delete',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({candidate_id:candidateId})});state.summary=d.summary;state.lastSavedText=`直前の判定を取消: ${candidateId}`;state.lastSavedCandidateId=null;renderProgress();await refreshIndices();}catch(e){$('error').textContent=String(e);}}
function move(delta){if(!state.indices.length)return;state.position=Math.max(0,Math.min(state.indices.length-1,state.position+delta));loadCurrent();}
document.querySelector('[data-valid]').onclick=()=>save('valid');for(const b of document.querySelectorAll('[data-invalid]'))b.onclick=()=>save('invalid',b.dataset.invalid);$('clear').onclick=clearReview;$('undo-last').onclick=undoLast;$('prev').onclick=()=>move(-1);$('next').onclick=()=>move(1);for(const id of ['review-filter','state-filter','classifier-filter','confidence-below'])$(id).onchange=()=>{state.position=0;refreshIndices();};document.addEventListener('keydown',e=>{if(e.target.matches('textarea,select,input'))return;const k=e.key.toLowerCase();if(k==='v')save('valid');else if(k==='b')save('invalid','background');else if(k==='p')save('invalid','partial_tile');else if(k==='m')save('invalid','multi_tile');else if(k==='o')save('invalid','other');else if(e.key==='ArrowLeft')move(-1);else if(e.key==='ArrowRight')move(1);});
(async()=>{try{await refreshSummary();await refreshIndices();}catch(e){$('error').textContent=String(e);}})();
</script>
</body>
</html>
"""


DUPLICATE_PAGE_HTML = r"""<!doctype html>
<html lang="ja">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>NanoDet duplicate suppression audit</title>
<style>
:root { color-scheme:light dark; font-family:system-ui,-apple-system,"Segoe UI",sans-serif; }
* { box-sizing:border-box; }
body { margin:0; background:Canvas; color:CanvasText; }
header { position:sticky; top:0; z-index:5; display:flex; gap:10px; align-items:center; flex-wrap:wrap; padding:10px 12px; background:Canvas; border-bottom:1px solid color-mix(in srgb,CanvasText 22%,transparent); }
header .grow { flex:1; }
button,select { font:inherit; min-height:38px; }
button { cursor:pointer; }
main { max-width:1500px; margin:0 auto; padding:14px; display:grid; gap:14px; }
.canvas-wrap { min-height:480px; display:grid; place-items:center; overflow:auto; background:#111; border:1px solid color-mix(in srgb,CanvasText 20%,transparent); }
canvas { max-width:100%; max-height:70vh; }
.cards { display:grid; grid-template-columns:repeat(auto-fit,minmax(340px,1fr)); gap:14px; }
.card { border:2px solid color-mix(in srgb,CanvasText 25%,transparent); padding:12px; display:grid; grid-template-columns:180px 1fr; gap:12px; align-items:center; }
.card.winner { border-color:#28df72; }
.card.removed { border-color:#ff453a; }
.card img { width:180px; height:180px; object-fit:contain; background:#111; }
.hero { font-size:26px; font-weight:800; }
.metric { font:15px/1.5 ui-monospace,Consolas,monospace; }
.explain { padding:10px 12px; border-left:4px solid #888; background:color-mix(in srgb,CanvasText 6%,transparent); }
.confirm { min-height:54px; font-size:18px; font-weight:800; }
.error { color:#d33; font-weight:700; }
@media(max-width:850px){.cards{grid-template-columns:1fr}.card{grid-template-columns:130px 1fr}.card img{width:130px;height:130px}.canvas-wrap{min-height:320px}}
</style>
</head>
<body>
<header>
<strong>NanoDet duplicate suppression audit</strong><a href="/">crop教師レビューへ戻る</a><span id="progress"></span><span class="grow"></span>
<label>表示 <select id="review-filter"><option value="unreviewed">未確認だけ</option><option value="all">全件</option><option value="reviewed">確認済み</option></select></label>
<button id="prev">←</button><button id="next">→</button>
</header>
<main>
<div class="explain">同一capture+region内で overlap≥0.80 でつながるbboxを1つのdupクラスタとして扱う。クラスタごとにNanoDet detector confidence最大の1bboxをKEEPし、残りをREMOVEする。1 shot内にdupクラスタが2つあればwinnerも2つ。この画面はクラスタ単位で1回だけ確認する。</div>
<div id="error" class="error"></div>
<div class="canvas-wrap"><canvas id="region-canvas"></canvas></div>
<div class="cards">
  <div class="card winner"><img id="winner-crop"><div><div class="hero">KEEP / WINNER</div><div id="winner-info" class="metric"></div></div></div>
  <div id="removed-cards" style="display:contents"></div>
</div>
<div id="pair-info" class="metric"></div>
<button id="confirm" class="confirm">確認して次へ →</button>
</main>
<script>
const state={indices:[],position:0,item:null,summary:null};
const $=id=>document.getElementById(id);
async function api(url,opts){const r=await fetch(url,opts);const d=await r.json();if(!r.ok)throw new Error(d.error||r.statusText);return d;}
async function refreshSummary(){state.summary=await api('/api/summary');renderProgress();}
function renderProgress(){const s=state.summary;if(!s)return;$('progress').textContent=`確認 ${s.duplicate_reviewed_count}/${s.duplicate_count}`;}
async function refreshIndices(){const q=new URLSearchParams({review:$('review-filter').value});const d=await api('/api/duplicate/indices?'+q);state.indices=d.indices;state.position=Math.min(state.position,Math.max(0,state.indices.length-1));if(!state.indices.length){state.item=null;$('pair-info').textContent='該当なし';clearCanvas();return;}await loadCurrent();}
async function loadCurrent(){state.item=await api('/api/duplicate?index='+state.indices[state.position]);render();}
function clearCanvas(){const c=$('region-canvas'),x=c.getContext('2d');x.clearRect(0,0,c.width,c.height);}
async function drawOverlay(c){const img=new Image();img.src=c.region_url+'&t='+Date.now();await img.decode();const canvas=$('region-canvas'),ctx=canvas.getContext('2d');canvas.width=img.naturalWidth;canvas.height=img.naturalHeight;ctx.drawImage(img,0,0);const lw=Math.max(3,img.naturalWidth/500*3);ctx.lineWidth=lw;ctx.font=`${Math.max(16,img.naturalWidth/65)}px system-ui`;ctx.strokeStyle='#28df72';ctx.strokeRect(c.winner_bbox_x,c.winner_bbox_y,c.winner_bbox_width,c.winner_bbox_height);ctx.fillStyle='#28df72';ctx.fillText(`KEEP ${(c.winner_confidence*100).toFixed(1)}%`,c.winner_bbox_x,Math.max(18,c.winner_bbox_y-5));for(const r of c.removed){ctx.strokeStyle='#ff453a';ctx.strokeRect(r.bbox_x,r.bbox_y,r.bbox_width,r.bbox_height);ctx.fillStyle='#ff453a';ctx.fillText(`REMOVE ${(r.confidence*100).toFixed(1)}%`,r.bbox_x,Math.max(18,r.bbox_y+r.bbox_height+20));}}
function render(){const c=state.item;if(!c)return;$('error').textContent='';$('winner-crop').src=c.winner_crop_url+'&t='+Date.now();$('winner-info').textContent=`detector confidence = ${c.winner_confidence.toFixed(6)}\ndetection index = ${c.winner_detection_index}`;$('removed-cards').innerHTML=c.removed.map((r,i)=>`<div class="card removed"><img src="${r.crop_url}&t=${Date.now()}"><div><div class="hero">REMOVE ${i+1}</div><div class="metric">detector confidence = ${r.confidence.toFixed(6)}\ndetection index = ${r.detection_index}\ncluster overlap = ${r.overlap_ratio.toFixed(6)}</div></div></div>`).join('');$('pair-info').textContent=`${state.position+1}/${state.indices.length} · group size=${c.removed_count+1} · winner + removed ${c.removed_count} · capture=${c.capture_id} · region=${c.region} · ${c.reviewed?'確認済み':'未確認'}`;drawOverlay(c).catch(e=>$('error').textContent=String(e));}
async function confirmCurrent(){const c=state.item;if(!c)return;try{const d=await api('/api/duplicate/review',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({suppression_id:c.suppression_id})});state.summary=d.summary;renderProgress();await refreshIndices();}catch(e){$('error').textContent=String(e);}}
function move(delta){if(!state.indices.length)return;state.position=Math.max(0,Math.min(state.indices.length-1,state.position+delta));loadCurrent();}
$('confirm').onclick=confirmCurrent;$('prev').onclick=()=>move(-1);$('next').onclick=()=>move(1);$('review-filter').onchange=()=>{state.position=0;refreshIndices();};document.addEventListener('keydown',e=>{if(e.key==='Enter'||e.key===' '){e.preventDefault();confirmCurrent();}else if(e.key==='ArrowLeft')move(-1);else if(e.key==='ArrowRight')move(1);});
(async()=>{try{await refreshSummary();await refreshIndices();}catch(e){$('error').textContent=String(e);}})();
</script>
</body>
</html>
"""


if __name__ == "__main__":
    main()
