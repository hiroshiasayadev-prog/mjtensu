from __future__ import annotations

import argparse
import json
import mimetypes
import sqlite3
import threading
import webbrowser
from contextlib import closing
from dataclasses import dataclass
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, quote, urlparse

try:
    from .build_tile_classifier_dataset import BASE_LABELS
    from .detector_duplicate_groups import (
        DEFAULT_DUPLICATE_OVERLAP_THRESHOLD,
        DuplicateCluster,
        DuplicatePlan,
        load_duplicate_plan,
    )
except ImportError:  # direct script execution
    from build_tile_classifier_dataset import BASE_LABELS
    from detector_duplicate_groups import (
        DEFAULT_DUPLICATE_OVERLAP_THRESHOLD,
        DuplicateCluster,
        DuplicatePlan,
        load_duplicate_plan,
    )


INVALID_REASONS = ("background", "partial_tile", "multi_tile", "other")

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
class CandidateRecord:
    index: int
    values: dict[str, Any]

    @property
    def candidate_id(self) -> str:
        return str(self.values["candidate_id"])


class ReviewStore:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with closing(self.connect()) as connection:
            connection.executescript(REVIEW_SCHEMA)
            connection.commit()

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
            if existing is None and review_count:
                raise ValueError(
                    "Existing review sidecar has decisions but no detector_run_key binding"
                )
            connection.executemany(
                """
                INSERT INTO review_metadata(key, value) VALUES (?, ?)
                ON CONFLICT(key) DO UPDATE SET value=excluded.value
                """,
                [
                    ("detector_run_key", detector_run_key),
                    ("source_dataset", source_dataset),
                ],
            )
            connection.commit()

    def all_reviews(self) -> dict[str, dict[str, Any]]:
        with closing(self.connect()) as connection:
            rows = connection.execute(
                "SELECT candidate_id, decision, label, invalid_reason, note, reviewed_at FROM review"
            ).fetchall()
        return {str(row["candidate_id"]): dict(row) for row in rows}

    def get_review(self, candidate_id: str) -> dict[str, Any] | None:
        return self.all_reviews().get(candidate_id)

    def save_review(
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
                raise ValueError("valid review requires one of the 34 base labels")
            invalid_reason = None
        elif decision == "invalid":
            if invalid_reason not in INVALID_REASONS:
                raise ValueError(f"invalid review requires one of {INVALID_REASONS}")
            label = None
        else:
            raise ValueError("decision must be valid or invalid")
        reviewed_at = datetime.now(timezone.utc).isoformat()
        with closing(self.connect()) as connection:
            connection.execute(
                """
                INSERT INTO review(candidate_id, decision, label, invalid_reason, note, reviewed_at)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(candidate_id) DO UPDATE SET
                    decision=excluded.decision,
                    label=excluded.label,
                    invalid_reason=excluded.invalid_reason,
                    note=excluded.note,
                    reviewed_at=excluded.reviewed_at
                """,
                (candidate_id, decision, label, invalid_reason, note.strip(), reviewed_at),
            )
            connection.commit()
        result = self.get_review(candidate_id)
        assert result is not None
        return result

    def delete_review(self, candidate_id: str) -> bool:
        with closing(self.connect()) as connection:
            deleted = connection.execute(
                "DELETE FROM review WHERE candidate_id=?", (candidate_id,)
            ).rowcount
            connection.commit()
        return bool(deleted)

    # Compatibility with existing tests/scripts that used the v1 store API.
    def get(self, candidate_id: str) -> dict[str, Any] | None:
        return self.get_review(candidate_id)

    def all(self) -> dict[str, dict[str, Any]]:
        return self.all_reviews()

    def save(
        self,
        candidate_id: str,
        *,
        decision: str,
        label: str | None,
        invalid_reason: str | None,
        note: str,
    ) -> dict[str, Any]:
        return self.save_review(
            candidate_id,
            decision=decision,
            label=label,
            invalid_reason=invalid_reason,
            note=note,
        )

    def delete(self, candidate_id: str) -> bool:
        return self.delete_review(candidate_id)

    def duplicate_reviews(self) -> set[str]:
        with closing(self.connect()) as connection:
            rows = connection.execute("SELECT suppression_id FROM duplicate_review").fetchall()
        return {str(row[0]) for row in rows}

    def confirm_duplicate(self, cluster_id: str) -> None:
        with closing(self.connect()) as connection:
            connection.execute(
                """
                INSERT INTO duplicate_review(suppression_id, reviewed_at) VALUES (?, ?)
                ON CONFLICT(suppression_id) DO UPDATE SET reviewed_at=excluded.reviewed_at
                """,
                (cluster_id, datetime.now(timezone.utc).isoformat()),
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
        with closing(sqlite3.connect(f"{self.path.as_uri()}?mode=ro", uri=True, timeout=30)) as connection:
            connection.row_factory = sqlite3.Row
            table = connection.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name='classifier_prediction'"
            ).fetchone()
            if table is None:
                raise ValueError(f"No classifier_prediction table: {self.path}")
            meta = connection.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name='audit_metadata'"
            ).fetchone()
            if meta is not None:
                row = connection.execute(
                    "SELECT value FROM audit_metadata WHERE key='source_detector_run_key'"
                ).fetchone()
                if row is not None:
                    self.detector_run_key = str(row[0])
                if self.model_key is None:
                    row = connection.execute(
                        "SELECT value FROM audit_metadata WHERE key='latest_model_key'"
                    ).fetchone()
                    if row is not None:
                        self.model_key = str(row[0])
            if self.model_key is None:
                row = connection.execute(
                    "SELECT model_key FROM classifier_prediction ORDER BY predicted_at DESC LIMIT 1"
                ).fetchone()
                self.model_key = None if row is None else str(row[0])
            if self.model_key is None:
                return
            rows = connection.execute(
                """
                SELECT candidate_id, predicted_label, confidence, invalid_probability
                FROM classifier_prediction WHERE model_key=?
                """,
                (self.model_key,),
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
        duplicate_plan: DuplicatePlan,
    ) -> None:
        self.repository_root = repository_root
        self.dataset_database = dataset_database
        self.review_store = review_store
        self.prediction_store = prediction_store
        self.duplicate_plan = duplicate_plan
        self.raw_by_id = load_candidate_rows(dataset_database)

        winner_ids = duplicate_plan.winner_candidate_ids
        missing = sorted(winner_ids - set(self.raw_by_id))
        if missing:
            raise ValueError(f"Duplicate plan references missing candidates: {missing[:5]}")

        winner_rows = [self.raw_by_id[candidate_id] for candidate_id in winner_ids]
        winner_rows.sort(key=candidate_sort_key)
        self.candidates = tuple(
            CandidateRecord(index=index, values=row)
            for index, row in enumerate(winner_rows)
        )
        self.by_id = {candidate.candidate_id: candidate for candidate in self.candidates}
        self.clusters = duplicate_plan.clusters
        self.cluster_by_id = {cluster.cluster_id: cluster for cluster in self.clusters}

    def summary(self) -> dict[str, Any]:
        reviews = self.review_store.all_reviews()
        relevant = {key: value for key, value in reviews.items() if key in self.by_id}
        valid_count = sum(value["decision"] == "valid" for value in relevant.values())
        invalid_count = sum(value["decision"] == "invalid" for value in relevant.values())
        reviewed_clusters = self.review_store.duplicate_reviews() & set(self.cluster_by_id)
        return {
            "raw_candidate_count": len(self.raw_by_id),
            "candidate_count": len(self.candidates),
            "loser_count": len(self.duplicate_plan.loser_candidate_ids),
            "duplicate_cluster_count": len(self.clusters),
            "reviewed_count": len(relevant),
            "valid_count": valid_count,
            "invalid_count": invalid_count,
            "duplicate_reviewed_count": len(reviewed_clusters),
            "classifier_prediction_count": len(self.prediction_store.predictions),
            "classifier_model_key": self.prediction_store.model_key,
            "duplicate_overlap_threshold": self.duplicate_plan.threshold,
            "labels": list(BASE_LABELS),
        }

    def filtered_indices(
        self,
        *,
        review_filter: str,
        state_filter: str,
        classifier_filter: str,
        confidence_below: float,
    ) -> list[int]:
        reviews = self.review_store.all_reviews()
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
            if state_filter != "all" and str(values["suggested_state"]) != state_filter:
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

    def candidate_payload(self, index: int) -> dict[str, Any]:
        candidate = self.candidates[index]
        values = candidate.values
        # image_png is a BLOB and is served separately by /api/crop. Never put it in JSON.
        payload = {key: value for key, value in values.items() if key != "image_png"}
        return {
            **payload,
            "index": index,
            "gt": json.loads(str(values["gt_json"])),
            "review": self.review_store.get_review(candidate.candidate_id),
            "prediction": self.prediction_store.get(candidate.candidate_id),
            "crop_url": f"/api/crop?candidate_id={quote(candidate.candidate_id, safe='')}",
            "region_url": f"/api/region?candidate_id={quote(candidate.candidate_id, safe='')}",
        }

    def cluster_indices(self, review_filter: str) -> list[int]:
        reviewed = self.review_store.duplicate_reviews()
        result: list[int] = []
        for index, cluster in enumerate(self.clusters):
            is_reviewed = cluster.cluster_id in reviewed
            if review_filter == "unreviewed" and is_reviewed:
                continue
            if review_filter == "reviewed" and not is_reviewed:
                continue
            result.append(index)
        return result

    def cluster_payload(self, index: int) -> dict[str, Any]:
        cluster = self.clusters[index]
        winner = self.raw_by_id[cluster.winner.candidate_id]
        losers = []
        for member in cluster.losers:
            row = self.raw_by_id[member.candidate.candidate_id]
            losers.append(
                {
                    "candidate_id": member.candidate.candidate_id,
                    "detection_index": member.candidate.detection_index,
                    "confidence": member.candidate.confidence,
                    "bbox_x": member.candidate.bbox_x,
                    "bbox_y": member.candidate.bbox_y,
                    "bbox_width": member.candidate.bbox_width,
                    "bbox_height": member.candidate.bbox_height,
                    "max_overlap_to_cluster": member.max_overlap_to_cluster,
                    "crop_url": f"/api/crop?candidate_id={quote(member.candidate.candidate_id, safe='')}",
                }
            )
        return {
            "index": index,
            "cluster_id": cluster.cluster_id,
            "capture_id": cluster.capture_id,
            "region": cluster.region,
            "winner_candidate_id": cluster.winner.candidate_id,
            "winner_detection_index": cluster.winner.detection_index,
            "winner_confidence": cluster.winner.confidence,
            "winner_bbox_x": cluster.winner.bbox_x,
            "winner_bbox_y": cluster.winner.bbox_y,
            "winner_bbox_width": cluster.winner.bbox_width,
            "winner_bbox_height": cluster.winner.bbox_height,
            "winner_crop_url": f"/api/crop?candidate_id={quote(cluster.winner.candidate_id, safe='')}",
            "region_url": f"/api/region?candidate_id={quote(cluster.winner.candidate_id, safe='')}",
            "losers": losers,
            "reviewed": cluster.cluster_id in self.review_store.duplicate_reviews(),
            "source_region_path": winner["source_region_path"],
        }

    def crop_bytes(self, candidate_id: str) -> bytes:
        row = self.raw_by_id.get(candidate_id)
        if row is None:
            raise KeyError(candidate_id)
        return bytes(row["image_png"])

    def region_asset(self, candidate_id: str) -> tuple[bytes, str]:
        row = self.raw_by_id.get(candidate_id)
        if row is None:
            raise KeyError(candidate_id)
        path = Path(str(row["source_region_path"]))
        if not path.is_absolute():
            path = self.repository_root / path
        path = path.resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        return path.read_bytes(), mimetypes.guess_type(path.name)[0] or "application/octet-stream"


def load_candidate_rows(database: Path) -> dict[str, dict[str, Any]]:
    with closing(sqlite3.connect(f"{database.as_uri()}?mode=ro", uri=True, timeout=60)) as connection:
        connection.row_factory = sqlite3.Row
        rows = connection.execute(
            """
            SELECT candidate_id, capture_id, campaign_id, layout_id, layout_ordinal,
                   brightness, shadow, region, source_region_path, source_composite_path,
                   detection_index, detection_confidence,
                   bbox_x, bbox_y, bbox_width, bbox_height,
                   crop_width, crop_height, image_png,
                   suggested_state, suggested_label,
                   best_gt_id, best_gt_label, best_iou, best_gt_coverage,
                   best_detection_coverage, substantial_gt_count, gt_json
            FROM candidate
            ORDER BY capture_id, region, detection_index, candidate_id
            """
        ).fetchall()
    return {str(row["candidate_id"]): dict(row) for row in rows}


def candidate_sort_key(values: dict[str, Any]) -> tuple[Any, ...]:
    state_order = {"background": 0, "multi_gt": 1, "partial": 2, "single_gt": 3}
    return (
        state_order.get(str(values["suggested_state"]), 4),
        -float(values["detection_confidence"]),
        str(values["candidate_id"]),
    )


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


def load_metadata(database: Path) -> dict[str, str]:
    with closing(sqlite3.connect(f"{database.as_uri()}?mode=ro", uri=True, timeout=30)) as connection:
        rows = connection.execute("SELECT key, value FROM dataset_metadata").fetchall()
    return {str(key): str(value) for key, value in rows}


def parse_args() -> argparse.Namespace:
    repository_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description="Review detector crops using freshly computed duplicate clusters.")
    parser.add_argument("--repository-root", type=Path, default=repository_root)
    parser.add_argument("--database", type=Path)
    parser.add_argument("--review-database", type=Path)
    parser.add_argument("--classifier-audit-database", type=Path)
    parser.add_argument("--model-key")
    parser.add_argument("--duplicate-overlap-threshold", type=float)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument("--open-browser", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repository_root = args.repository_root.resolve()
    default_root = repository_root / ".local" / "recognition" / "detector_crop_dataset"
    database = (args.database or default_root / "dataset.sqlite").resolve()
    root = database.parent if args.database is not None else default_root
    if not database.is_file():
        raise FileNotFoundError(database)
    metadata = load_metadata(database)
    detector_run_key = metadata.get("detector_run_key")
    if not detector_run_key:
        raise ValueError("Detector dataset has no detector_run_key")
    threshold = (
        float(args.duplicate_overlap_threshold)
        if args.duplicate_overlap_threshold is not None
        else float(metadata.get("duplicate_overlap_threshold", DEFAULT_DUPLICATE_OVERLAP_THRESHOLD))
    )
    duplicate_plan = load_duplicate_plan(database, threshold=threshold)

    review_database = (
        args.review_database.resolve()
        if args.review_database is not None
        else root / f"reviews.{detector_run_key}.sqlite"
    )
    prediction_database = (
        args.classifier_audit_database.resolve()
        if args.classifier_audit_database is not None
        else (root / "classifier_audit.sqlite" if (root / "classifier_audit.sqlite").is_file() else None)
    )
    reviews = ReviewStore(review_database)
    reviews.bind_dataset(detector_run_key=detector_run_key, source_dataset=str(database))
    predictions = PredictionStore(prediction_database, args.model_key)
    if predictions.detector_run_key and predictions.detector_run_key != detector_run_key:
        raise ValueError(
            "Classifier audit sidecar belongs to another detector run: "
            f"stored={predictions.detector_run_key}, requested={detector_run_key}"
        )

    application = ReviewApplication(
        repository_root=repository_root,
        dataset_database=database,
        review_store=reviews,
        prediction_store=predictions,
        duplicate_plan=duplicate_plan,
    )
    server = ThreadingHTTPServer((args.host, args.port), make_handler(application))
    url = f"http://{args.host}:{args.port}/"
    print(
        "[detector-crop-review] "
        f"raw={len(application.raw_by_id):,} "
        f"review_candidates={len(application.candidates):,} "
        f"dup_clusters={len(application.clusters):,} "
        f"losers={len(duplicate_plan.loser_candidate_ids):,}"
    )
    print(f"[detector-crop-review] duplicate_threshold={threshold:.6g}")
    print(f"[detector-crop-review] reviews={review_database}")
    print(f"[detector-crop-review] open: {url}")
    if args.open_browser:
        threading.Timer(0.4, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[detector-crop-review] stopped")
    finally:
        server.server_close()


def make_handler(application: ReviewApplication) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        server_version = "MjtensuDetectorCropReview/2.0"

        def do_GET(self) -> None:  # noqa: N802
            try:
                parsed = urlparse(self.path)
                if parsed.path == "/":
                    return self.send_bytes(PAGE_HTML.encode("utf-8"), "text/html; charset=utf-8")
                if parsed.path == "/duplicates":
                    return self.send_bytes(DUPLICATE_PAGE_HTML.encode("utf-8"), "text/html; charset=utf-8")
                if parsed.path == "/api/summary":
                    return self.send_json(application.summary())
                if parsed.path == "/api/indices":
                    query = parse_qs(parsed.query)
                    review_filter = one(query, "review", "unreviewed")
                    state_filter = one(query, "state", "all")
                    classifier_filter = one(query, "classifier", "all")
                    confidence_below = float(one(query, "confidence_below", "0.80"))
                    indices = application.filtered_indices(
                        review_filter=review_filter,
                        state_filter=state_filter,
                        classifier_filter=classifier_filter,
                        confidence_below=confidence_below,
                    )
                    return self.send_json({"indices": indices})
                if parsed.path == "/api/candidate":
                    return self.send_json(application.candidate_payload(int(one(parse_qs(parsed.query), "index"))))
                if parsed.path == "/api/cluster/indices":
                    query = parse_qs(parsed.query)
                    return self.send_json({"indices": application.cluster_indices(one(query, "review", "unreviewed"))})
                if parsed.path == "/api/cluster":
                    return self.send_json(application.cluster_payload(int(one(parse_qs(parsed.query), "index"))))
                if parsed.path == "/api/crop":
                    candidate_id = one(parse_qs(parsed.query), "candidate_id")
                    return self.send_bytes(application.crop_bytes(candidate_id), "image/png")
                if parsed.path == "/api/region":
                    candidate_id = one(parse_qs(parsed.query), "candidate_id")
                    content, content_type = application.region_asset(candidate_id)
                    return self.send_bytes(content, content_type)
                self.send_error(HTTPStatus.NOT_FOUND)
            except (ValueError, KeyError, IndexError, FileNotFoundError) as error:
                self.send_json({"error": str(error)}, status=HTTPStatus.BAD_REQUEST)

        def do_POST(self) -> None:  # noqa: N802
            try:
                parsed = urlparse(self.path)
                payload = self.read_json()
                if parsed.path == "/api/review":
                    candidate_id = str(payload.get("candidate_id", ""))
                    if candidate_id not in application.by_id:
                        raise KeyError(candidate_id)
                    review = application.review_store.save_review(
                        candidate_id,
                        decision=str(payload.get("decision", "")),
                        label=None if not payload.get("label") else str(payload["label"]),
                        invalid_reason=None if not payload.get("invalid_reason") else str(payload["invalid_reason"]),
                        note=str(payload.get("note", "")),
                    )
                    return self.send_json({"review": review, "summary": application.summary()})
                if parsed.path == "/api/review/delete":
                    candidate_id = str(payload.get("candidate_id", ""))
                    return self.send_json(
                        {
                            "deleted": application.review_store.delete_review(candidate_id),
                            "summary": application.summary(),
                        }
                    )
                if parsed.path == "/api/cluster/confirm":
                    cluster_id = str(payload.get("cluster_id", ""))
                    if cluster_id not in application.cluster_by_id:
                        raise KeyError(cluster_id)
                    application.review_store.confirm_duplicate(cluster_id)
                    return self.send_json({"confirmed": True, "summary": application.summary()})
                self.send_error(HTTPStatus.NOT_FOUND)
            except (ValueError, KeyError, IndexError) as error:
                self.send_json({"error": str(error)}, status=HTTPStatus.BAD_REQUEST)

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
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(content)

    return Handler


def one(query: dict[str, list[str]], key: str, default: str | None = None) -> str:
    values = query.get(key)
    if values:
        return values[0]
    if default is not None:
        return default
    raise ValueError(f"Missing query parameter: {key}")


PAGE_HTML = r"""<!doctype html>
<html lang="ja"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Detector crop review</title><style>
:root{color-scheme:dark;font-family:system-ui,sans-serif}*{box-sizing:border-box}body{margin:0;background:#111;color:#eee}header{position:sticky;top:0;z-index:5;display:flex;gap:8px;align-items:center;flex-wrap:wrap;padding:10px;border-bottom:1px solid #444;background:#111}header .grow{flex:1}main{max-width:1500px;margin:auto;padding:14px;display:grid;grid-template-columns:minmax(550px,1.35fr) minmax(380px,.8fr);gap:14px}.viewer,.panel{border:1px solid #444;padding:10px}.canvas{min-height:480px;display:grid;place-items:center;background:#080808}.canvas canvas{max-width:100%;max-height:70vh}.crop{display:flex;gap:12px;align-items:center;margin-top:10px}.crop img{width:150px;height:150px;object-fit:contain;background:#000}.grid{display:grid;grid-template-columns:1fr 1fr;gap:7px}.grid button{min-height:46px}.wide{grid-column:1/-1}button,select,input,textarea{font:inherit}textarea{width:100%;min-height:55px}.meta{white-space:pre-wrap;font:12px ui-monospace,monospace}.error{color:#ff6961;font-weight:700}@media(max-width:900px){main{grid-template-columns:1fr}}
</style></head><body>
<header><b>Detector crop human review</b><a href="/duplicates">重複除去audit</a><span id="progress"></span><span class="grow"></span>
<label>表示 <select id="review"><option value="unreviewed">未判定だけ</option><option value="all">全件</option><option value="reviewed">判定済み</option><option value="valid">valid</option><option value="invalid">invalid</option></select></label>
<label>geometry <select id="geometry"><option value="all">all</option><option value="background">background</option><option value="multi_gt">multi</option><option value="partial">partial</option><option value="single_gt">single</option></select></label>
<label>gray35 <select id="classifier"><option value="all">all</option><option value="review_disagreement">human不一致</option><option value="strong_gt_disagreement">GT不一致</option><option value="suspected_invalid_predicted_tile">怪しいcropを牌判定</option><option value="predict_invalid">invalid予測</option><option value="uncertain">低confidence</option><option value="no_prediction">予測なし</option></select></label>
<label>conf&lt;<input id="conf" type="number" value="0.80" min="0" max="1" step="0.05" style="width:70px"></label><button id="prev">←</button><button id="next">→</button></header>
<main><section class="viewer"><div class="canvas"><canvas id="canvas"></canvas></div><div class="crop"><img id="crop"><div><h2 id="hero"></h2><div id="stats"></div><div id="prediction"></div></div></div></section>
<section><div id="error" class="error"></div><div class="panel"><b>人力確定</b><p>このcrop単体をgray35へ入力してよいかだけ判定する。重複loserはこの画面には来ない。</p><label>valid牌種 <select id="label"></select></label><div class="grid"><button class="wide" data-valid>V — valid</button><button data-invalid="background">B — 背景</button><button data-invalid="partial_tile">P — 欠けすぎ</button><button data-invalid="multi_tile">M — 複数牌</button><button class="wide" data-invalid="other">O — その他invalid</button></div><p><textarea id="note" placeholder="note"></textarea></p><button id="clear">判定解除</button> <span id="review-state"></span></div><div class="panel meta" id="meta"></div></section></main>
<script>
const S={indices:[],pos:0,item:null,summary:null};const $=id=>document.getElementById(id);async function api(u,o){const r=await fetch(u,o);const d=await r.json();if(!r.ok)throw Error(d.error||r.statusText);return d}function esc(x){return String(x).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]))}
async function summary(){S.summary=await api('/api/summary');$('label').innerHTML=S.summary.labels.map(x=>`<option>${esc(x)}</option>`).join('');$('progress').textContent=`human ${S.summary.reviewed_count}/${S.summary.candidate_count} · raw ${S.summary.raw_candidate_count} · losers ${S.summary.loser_count} · dup groups ${S.summary.duplicate_cluster_count}`;if(!S.summary.classifier_prediction_count){$('classifier').value='all';$('classifier').disabled=true;$('conf').disabled=true}}
async function indices(){const q=new URLSearchParams({review:$('review').value,state:$('geometry').value,classifier:$('classifier').value,confidence_below:$('conf').value});S.indices=(await api('/api/indices?'+q)).indices;S.pos=Math.min(S.pos,Math.max(0,S.indices.length-1));if(!S.indices.length){S.item=null;$('hero').textContent='該当なし';return}await load()}
async function load(){S.item=await api('/api/candidate?index='+S.indices[S.pos]);render()}
function corners(g){const a=g.angleDeg*Math.PI/180,c=Math.cos(a),s=Math.sin(a),w=g.width/2,h=g.height/2;return [[-w,-h],[w,-h],[w,h],[-w,h]].map(([x,y])=>[g.centerX+c*x-s*y,g.centerY+s*x+c*y])}
async function overlay(c){const img=new Image();img.src=c.region_url+'&t='+Date.now();await img.decode();const v=$('canvas'),x=v.getContext('2d');v.width=img.naturalWidth;v.height=img.naturalHeight;x.drawImage(img,0,0);x.lineWidth=3;x.font='16px system-ui';for(const g of c.gt){const p=corners(g);x.strokeStyle='#35e87d';x.fillStyle='#35e87d';x.beginPath();x.moveTo(...p[0]);for(let i=1;i<p.length;i++)x.lineTo(...p[i]);x.closePath();x.stroke();x.fillText(g.label,p[0][0],Math.max(16,p[0][1]-3))}x.strokeStyle='#ff453a';x.strokeRect(c.bbox_x,c.bbox_y,c.bbox_width,c.bbox_height)}
function render(){const c=S.item;$('crop').src=c.crop_url+'&t='+Date.now();$('hero').textContent=`${c.suggested_state}${c.suggested_label?' → '+c.suggested_label:''}`;$('stats').textContent=`NanoDet ${(c.detection_confidence*100).toFixed(1)}% · GT cover ${c.best_gt_coverage.toFixed(3)} · purity ${c.best_detection_coverage.toFixed(3)}`;$('prediction').textContent=c.prediction?`gray35 ${c.prediction.predicted_label} ${(c.prediction.confidence*100).toFixed(1)}%`:'gray35 predictionなし';$('note').value=c.review?.note||'';const d=c.review?.label||c.suggested_label||(c.prediction&&c.prediction.predicted_label!=='invalid'?c.prediction.predicted_label:'1m');if([...$('label').options].some(o=>o.value===d))$('label').value=d;$('review-state').textContent=c.review?(c.review.decision==='valid'?`保存済 VALID/${c.review.label}`:`保存済 INVALID/${c.review.invalid_reason}`):'未判定';$('meta').textContent=`${S.pos+1}/${S.indices.length}\ncandidate=${c.candidate_id}\ncapture=${c.capture_id}\nregion=${c.region}\ndetection index/conf=${c.detection_index} / ${c.detection_confidence}\ngeometry=${c.suggested_state}\nbest GT=${c.best_gt_label||''}`;overlay(c).catch(e=>$('error').textContent=e)}
async function save(decision,reason=null){const c=S.item;if(!c)return;await api('/api/review',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({candidate_id:c.candidate_id,decision,label:decision==='valid'?$('label').value:null,invalid_reason:reason,note:$('note').value})});await summary();await indices()}
async function clear(){if(!S.item)return;await api('/api/review/delete',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({candidate_id:S.item.candidate_id})});await summary();await indices()}function move(d){if(!S.indices.length)return;S.pos=Math.max(0,Math.min(S.indices.length-1,S.pos+d));load()}
document.querySelector('[data-valid]').onclick=()=>save('valid');document.querySelectorAll('[data-invalid]').forEach(b=>b.onclick=()=>save('invalid',b.dataset.invalid));$('clear').onclick=clear;$('prev').onclick=()=>move(-1);$('next').onclick=()=>move(1);['review','geometry','classifier','conf'].forEach(id=>$(id).onchange=()=>{S.pos=0;indices()});document.addEventListener('keydown',e=>{if(e.target.matches('textarea,select,input'))return;const k=e.key.toLowerCase();if(k==='v')save('valid');else if(k==='b')save('invalid','background');else if(k==='p')save('invalid','partial_tile');else if(k==='m')save('invalid','multi_tile');else if(k==='o')save('invalid','other');else if(e.key==='ArrowLeft')move(-1);else if(e.key==='ArrowRight')move(1)});(async()=>{try{await summary();await indices()}catch(e){$('error').textContent=e}})();
</script></body></html>"""


DUPLICATE_PAGE_HTML = r"""<!doctype html>
<html lang="ja"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Duplicate audit</title><style>
:root{color-scheme:dark;font-family:system-ui,sans-serif}*{box-sizing:border-box}body{margin:0;background:#111;color:#eee}header{position:sticky;top:0;z-index:5;display:flex;gap:8px;align-items:center;padding:10px;background:#111;border-bottom:1px solid #444}header .grow{flex:1}main{max-width:1500px;margin:auto;padding:14px}.canvas{min-height:450px;display:grid;place-items:center;background:#080808;border:1px solid #444}.canvas canvas{max-width:100%;max-height:68vh}.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:10px;margin-top:10px}.card{border:2px solid #555;padding:10px;display:flex;gap:10px;align-items:center}.winner{border-color:#35e87d}.loser{border-color:#ff453a}.card img{width:130px;height:130px;object-fit:contain;background:#000}.confirm{width:100%;min-height:52px;margin-top:10px;font-size:18px;font-weight:700}.error{color:#ff6961}
</style></head><body><header><b>重複除去audit</b><a href="/">通常reviewへ</a><span id="progress"></span><span class="grow"></span><label>表示 <select id="filter"><option value="unreviewed">未確認</option><option value="all">全件</option><option value="reviewed">確認済み</option></select></label><button id="prev">←</button><button id="next">→</button></header><main><p>1 dup cluster = 1 review。緑がdetector confidence最大の最終winner、赤が同クラスタのloser。1 shotに複数clusterがあれば別々に出る。</p><div id="error" class="error"></div><div class="canvas"><canvas id="canvas"></canvas></div><div class="cards"><div class="card winner"><img id="winner"><div><h2>KEEP / WINNER</h2><pre id="winner-info"></pre></div></div><div id="losers" style="display:contents"></div></div><pre id="meta"></pre><button id="confirm" class="confirm">確認して次へ →</button></main><script>
const S={indices:[],pos:0,item:null,summary:null};const $=id=>document.getElementById(id);async function api(u,o){const r=await fetch(u,o);const d=await r.json();if(!r.ok)throw Error(d.error||r.statusText);return d}async function summary(){S.summary=await api('/api/summary');$('progress').textContent=`確認 ${S.summary.duplicate_reviewed_count}/${S.summary.duplicate_cluster_count}`}
async function indices(){S.indices=(await api('/api/cluster/indices?review='+$('filter').value)).indices;S.pos=Math.min(S.pos,Math.max(0,S.indices.length-1));if(!S.indices.length){$('meta').textContent='該当なし';return}await load()}async function load(){S.item=await api('/api/cluster?index='+S.indices[S.pos]);render()}
async function overlay(c){const img=new Image();img.src=c.region_url+'&t='+Date.now();await img.decode();const v=$('canvas'),x=v.getContext('2d');v.width=img.naturalWidth;v.height=img.naturalHeight;x.drawImage(img,0,0);x.lineWidth=3;x.font='16px system-ui';x.strokeStyle='#35e87d';x.fillStyle='#35e87d';x.strokeRect(c.winner_bbox_x,c.winner_bbox_y,c.winner_bbox_width,c.winner_bbox_height);x.fillText(`KEEP ${(c.winner_confidence*100).toFixed(1)}%`,c.winner_bbox_x,Math.max(18,c.winner_bbox_y-5));for(const r of c.losers){x.strokeStyle='#ff453a';x.fillStyle='#ff453a';x.strokeRect(r.bbox_x,r.bbox_y,r.bbox_width,r.bbox_height);x.fillText(`REMOVE ${(r.confidence*100).toFixed(1)}%`,r.bbox_x,Math.max(18,r.bbox_y+r.bbox_height+18))}}
function render(){const c=S.item;$('winner').src=c.winner_crop_url+'&t='+Date.now();$('winner-info').textContent=`conf=${c.winner_confidence.toFixed(6)}\nindex=${c.winner_detection_index}`;$('losers').innerHTML=c.losers.map((r,i)=>`<div class="card loser"><img src="${r.crop_url}&t=${Date.now()}"><div><h2>REMOVE ${i+1}</h2><pre>conf=${r.confidence.toFixed(6)}\nindex=${r.detection_index}\nmax overlap=${r.max_overlap_to_cluster.toFixed(6)}</pre></div></div>`).join('');$('meta').textContent=`${S.pos+1}/${S.indices.length}\ncluster=${c.cluster_id}\ncapture=${c.capture_id}\nregion=${c.region}\nsize=${c.losers.length+1}\n${c.reviewed?'確認済み':'未確認'}`;overlay(c).catch(e=>$('error').textContent=e)}
async function confirm(){if(!S.item)return;await api('/api/cluster/confirm',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({cluster_id:S.item.cluster_id})});await summary();await indices()}function move(d){if(!S.indices.length)return;S.pos=Math.max(0,Math.min(S.indices.length-1,S.pos+d));load()}$('confirm').onclick=confirm;$('prev').onclick=()=>move(-1);$('next').onclick=()=>move(1);$('filter').onchange=()=>{S.pos=0;indices()};document.addEventListener('keydown',e=>{if(e.key==='Enter'||e.key===' '){e.preventDefault();confirm()}else if(e.key==='ArrowLeft')move(-1);else if(e.key==='ArrowRight')move(1)});(async()=>{try{await summary();await indices()}catch(e){$('error').textContent=e}})();
</script></body></html>"""


if __name__ == "__main__":
    main()
