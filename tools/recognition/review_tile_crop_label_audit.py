from __future__ import annotations

import argparse
import csv
import json
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

try:
    from .build_tile_classifier_dataset import BASE_LABELS
except ImportError:  # direct script execution: python tools/recognition/review_tile_crop_label_audit.py
    from build_tile_classifier_dataset import BASE_LABELS


DECISIONS = (
    "label_error",
    "false_detection",
    "unusable_crop",
    "background",
)
SOURCE_LABELS = tuple(BASE_LABELS) + ("red5m", "red5p", "red5s")

REVIEW_SCHEMA = """
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS review (
    crop_id             TEXT PRIMARY KEY,
    decision            TEXT NOT NULL CHECK (
        decision IN ('label_error', 'false_detection', 'unusable_crop', 'background')
    ),
    corrected_label     TEXT,
    note                TEXT NOT NULL DEFAULT '',
    source              TEXT NOT NULL,
    source_partition    TEXT NOT NULL,
    audit_tier          INTEGER NOT NULL,
    audit_rank          INTEGER NOT NULL,
    expected_label      TEXT NOT NULL,
    consensus_prediction TEXT NOT NULL,
    reviewed_at         TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_review_decision
ON review(decision, reviewed_at);

CREATE TABLE IF NOT EXISTS review_metadata (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""


@dataclass(frozen=True)
class Candidate:
    rank: int
    values: dict[str, str]

    @property
    def crop_id(self) -> str:
        return self.values["crop_id"]

    @property
    def tier(self) -> int:
        return int(self.values["tier"])

    @property
    def source(self) -> str:
        return self.values["source"]


def parse_args() -> argparse.Namespace:
    repository_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(
        description=(
            "Serve a local browser UI for manually reviewing tile-crop audit candidates. "
            "Review decisions are stored in a sidecar SQLite database; the persistent crop "
            "dataset is never mutated by this tool."
        )
    )
    parser.add_argument(
        "--audit-dir",
        type=Path,
        required=True,
        help="Directory produced by audit_tile_crop_dataset_labels.py.",
    )
    parser.add_argument(
        "--database",
        type=Path,
        default=repository_root
        / ".local"
        / "recognition"
        / "tile_crop_dataset"
        / "dataset.sqlite",
        help="Persistent tile crop database used to serve crop images.",
    )
    parser.add_argument(
        "--review-database",
        type=Path,
        default=repository_root
        / ".local"
        / "recognition"
        / "tile_crop_dataset"
        / "quality_audit.sqlite",
        help="Sidecar SQLite database receiving review decisions.",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument(
        "--open-browser",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    audit_dir = args.audit_dir.resolve()
    candidate_csv = audit_dir / "candidates.csv"
    database = args.database.resolve()
    review_database = args.review_database.resolve()

    if not candidate_csv.is_file():
        raise FileNotFoundError(candidate_csv)
    if not database.is_file():
        raise FileNotFoundError(database)
    if args.port < 1 or args.port > 65535:
        raise ValueError("--port must be in [1, 65535]")

    candidates = load_candidates(candidate_csv)
    if not candidates:
        raise ValueError(f"No candidates found in {candidate_csv}")

    review_store = ReviewStore(review_database)
    review_store.record_metadata(
        {
            "source_crop_database": str(database),
            "last_audit_directory": str(audit_dir),
            "last_candidates_csv": str(candidate_csv),
        }
    )
    application = ReviewApplication(
        candidates=candidates,
        source_database=database,
        review_store=review_store,
    )

    handler_class = make_handler(application)
    server = ThreadingHTTPServer((str(args.host), int(args.port)), handler_class)
    url = f"http://{args.host}:{args.port}/"
    print(f"[review-ui] candidates={len(candidates):,}")
    print(f"[review-ui] review database: {review_database}")
    print(f"[review-ui] open: {url}")
    if bool(args.open_browser):
        threading.Timer(0.4, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[review-ui] stopped")
    finally:
        server.server_close()


def load_candidates(path: Path) -> list[Candidate]:
    with path.open("r", encoding="utf-8-sig", newline="") as input_file:
        reader = csv.DictReader(input_file)
        required = {
            "tier",
            "crop_id",
            "source",
            "source_partition",
            "expected_label",
            "consensus_prediction",
            "consensus_count",
            "consensus_confidence",
        }
        missing = sorted(required - set(reader.fieldnames or ()))
        if missing:
            raise ValueError(f"Candidates CSV is missing columns: {missing}")
        result: list[Candidate] = []
        for rank, row in enumerate(reader, start=1):
            values = {str(key): "" if value is None else str(value) for key, value in row.items()}
            if not values.get("crop_id"):
                raise ValueError(f"Candidate row {rank} has no crop_id")
            result.append(Candidate(rank=rank, values=values))
    return result


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

    def record_metadata(self, values: dict[str, str]) -> None:
        with closing(self.connect()) as connection:
            connection.executemany(
                """
                INSERT INTO review_metadata(key, value) VALUES (?, ?)
                ON CONFLICT(key) DO UPDATE SET value = excluded.value
                """,
                list(values.items()),
            )
            connection.commit()

    def get(self, crop_id: str) -> dict[str, Any] | None:
        with closing(self.connect()) as connection:
            row = connection.execute(
                """
                SELECT crop_id, decision, corrected_label, note, source, source_partition,
                       audit_tier, audit_rank, expected_label, consensus_prediction, reviewed_at
                FROM review
                WHERE crop_id = ?
                """,
                (crop_id,),
            ).fetchone()
        return None if row is None else dict(row)

    def all_by_crop_id(self) -> dict[str, dict[str, Any]]:
        with closing(self.connect()) as connection:
            rows = connection.execute(
                """
                SELECT crop_id, decision, corrected_label, note, source, source_partition,
                       audit_tier, audit_rank, expected_label, consensus_prediction, reviewed_at
                FROM review
                """
            ).fetchall()
        return {str(row["crop_id"]): dict(row) for row in rows}

    def save(
        self,
        candidate: Candidate,
        *,
        decision: str,
        corrected_label: str | None,
        note: str,
    ) -> dict[str, Any]:
        if decision not in DECISIONS:
            raise ValueError(f"Unsupported decision: {decision}")
        if decision == "label_error":
            if corrected_label not in SOURCE_LABELS:
                raise ValueError(
                    "label_error requires corrected_label to be one of the supported tile labels"
                )
        else:
            corrected_label = None
        reviewed_at = datetime.now(UTC).isoformat()
        with closing(self.connect()) as connection:
            connection.execute(
                """
                INSERT INTO review(
                    crop_id, decision, corrected_label, note,
                    source, source_partition, audit_tier, audit_rank,
                    expected_label, consensus_prediction, reviewed_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(crop_id) DO UPDATE SET
                    decision = excluded.decision,
                    corrected_label = excluded.corrected_label,
                    note = excluded.note,
                    source = excluded.source,
                    source_partition = excluded.source_partition,
                    audit_tier = excluded.audit_tier,
                    audit_rank = excluded.audit_rank,
                    expected_label = excluded.expected_label,
                    consensus_prediction = excluded.consensus_prediction,
                    reviewed_at = excluded.reviewed_at
                """,
                (
                    candidate.crop_id,
                    decision,
                    corrected_label,
                    note.strip(),
                    candidate.values["source"],
                    candidate.values["source_partition"],
                    candidate.tier,
                    candidate.rank,
                    candidate.values["expected_label"],
                    candidate.values["consensus_prediction"],
                    reviewed_at,
                ),
            )
            connection.commit()
        result = self.get(candidate.crop_id)
        assert result is not None
        return result

    def delete(self, crop_id: str) -> bool:
        with closing(self.connect()) as connection:
            deleted = connection.execute(
                "DELETE FROM review WHERE crop_id = ?",
                (crop_id,),
            ).rowcount
            connection.commit()
        return bool(deleted)


class ReviewApplication:
    def __init__(
        self,
        *,
        candidates: Sequence[Candidate],
        source_database: Path,
        review_store: ReviewStore,
    ) -> None:
        self.candidates = tuple(candidates)
        self.source_database = source_database
        self.review_store = review_store
        self.by_crop_id = {candidate.crop_id: candidate for candidate in candidates}
        if len(self.by_crop_id) != len(self.candidates):
            raise ValueError("Candidate crop_id values are not unique")

    def summary(self) -> dict[str, Any]:
        reviews = self.review_store.all_by_crop_id()
        relevant_reviews = {
            crop_id: review
            for crop_id, review in reviews.items()
            if crop_id in self.by_crop_id
        }
        counts = {decision: 0 for decision in DECISIONS}
        for review in relevant_reviews.values():
            decision = str(review["decision"])
            if decision in counts:
                counts[decision] += 1
        reviewed = len(relevant_reviews)
        return {
            "candidate_count": len(self.candidates),
            "reviewed_count": reviewed,
            "unreviewed_count": len(self.candidates) - reviewed,
            "decision_counts": counts,
            "labels": list(SOURCE_LABELS),
        }

    def candidate_payload(self, index: int) -> dict[str, Any]:
        if index < 0 or index >= len(self.candidates):
            raise IndexError(index)
        candidate = self.candidates[index]
        values = candidate.values
        angle_results = []
        for key in sorted(
            (key for key in values if key.endswith("_prediction") and key[0].isdigit()),
            key=angle_sort_key,
        ):
            angle = key[: -len("_prediction")]
            angle_results.append(
                {
                    "angle": angle,
                    "prediction": values.get(f"{angle}_prediction", ""),
                    "confidence": float_or_none(values.get(f"{angle}_confidence", "")),
                    "expected_confidence": float_or_none(
                        values.get(f"{angle}_expected_confidence", "")
                    ),
                }
            )
        payload: dict[str, Any] = {
            "index": index,
            "rank": candidate.rank,
            "crop_id": candidate.crop_id,
            "tier": candidate.tier,
            "source": values["source"],
            "source_partition": values["source_partition"],
            "original_label": values.get("original_label", ""),
            "expected_label": values.get("expected_label", ""),
            "consensus_prediction": values.get("consensus_prediction", ""),
            "consensus_count": int(values.get("consensus_count", "0") or 0),
            "consensus_confidence": float(values.get("consensus_confidence", "0") or 0),
            "expected_mean_confidence": float(
                values.get("expected_mean_confidence", "0") or 0
            ),
            "zero_prediction": values.get("zero_prediction", ""),
            "zero_confidence": float(values.get("zero_confidence", "0") or 0),
            "source_image_path": values.get("source_image_path", ""),
            "source_image_id": values.get("source_image_id", ""),
            "source_annotation_id": values.get("source_annotation_id", ""),
            "capture_id": values.get("capture_id", ""),
            "layout_id": values.get("layout_id", ""),
            "layout_ordinal": values.get("layout_ordinal", ""),
            "region": values.get("region", ""),
            "group_name": values.get("group_name", ""),
            "group_ordinal": values.get("group_ordinal", ""),
            "tile_ordinal": values.get("tile_ordinal", ""),
            "brightness": values.get("brightness", ""),
            "shadow": values.get("shadow", ""),
            "annotation_angle_deg": values.get("annotation_angle_deg", ""),
            "expected_rotation_deg": values.get("expected_rotation_deg", ""),
            "angles": angle_results,
            "image_url": f"/api/image?crop_id={quote(candidate.crop_id, safe='')}",
            "review": self.review_store.get(candidate.crop_id),
        }
        return payload

    def filtered_indices(
        self,
        *,
        review_filter: str,
        tier: int | None,
        source: str | None,
        corrected_label: str | None,
    ) -> list[int]:
        reviews = self.review_store.all_by_crop_id()
        result: list[int] = []
        for index, candidate in enumerate(self.candidates):
            if tier is not None and candidate.tier != tier:
                continue
            if source is not None and candidate.source != source:
                continue
            review = reviews.get(candidate.crop_id)
            if corrected_label is not None:
                if review is None or review.get("corrected_label") != corrected_label:
                    continue
            if review_filter == "unreviewed" and review is not None:
                continue
            if review_filter == "reviewed" and review is None:
                continue
            if review_filter in DECISIONS:
                if review is None or review["decision"] != review_filter:
                    continue
            result.append(index)
        return result

    def load_image_png(self, crop_id: str) -> bytes:
        if crop_id not in self.by_crop_id:
            raise KeyError(crop_id)
        with closing(sqlite3.connect(self.source_database, timeout=30)) as connection:
            row = connection.execute(
                "SELECT image_png FROM tile_crop WHERE crop_id = ?",
                (crop_id,),
            ).fetchone()
        if row is None:
            raise KeyError(crop_id)
        return bytes(row[0])


def make_handler(application: ReviewApplication) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        server_version = "MjtensuTileCropReview/1.0"

        def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
            try:
                parsed = urlparse(self.path)
                if parsed.path == "/":
                    self.send_html(PAGE_HTML)
                    return
                if parsed.path == "/api/summary":
                    self.send_json(application.summary())
                    return
                if parsed.path == "/api/candidate":
                    query = parse_qs(parsed.query)
                    index = int(single_query(query, "index", "0"))
                    self.send_json(application.candidate_payload(index))
                    return
                if parsed.path == "/api/indices":
                    query = parse_qs(parsed.query)
                    review_filter = single_query(query, "review", "unreviewed")
                    tier_text = single_query(query, "tier", "")
                    source_text = single_query(query, "source", "")
                    corrected_label_text = single_query(query, "corrected_label", "")
                    if review_filter not in ("all", "unreviewed", "reviewed", *DECISIONS):
                        raise ValueError(f"Unsupported review filter: {review_filter}")
                    tier = None if not tier_text else int(tier_text)
                    if tier not in (None, 1, 2, 3):
                        raise ValueError("tier must be empty, 1, 2, or 3")
                    source = None if not source_text else source_text
                    if source not in (None, "jp", "manual"):
                        raise ValueError("source must be empty, jp, or manual")
                    corrected_label = None if not corrected_label_text else corrected_label_text
                    if corrected_label not in (None, *SOURCE_LABELS):
                        raise ValueError("corrected_label must be empty or a supported tile label")
                    indices = application.filtered_indices(
                        review_filter=review_filter,
                        tier=tier,
                        source=source,
                        corrected_label=corrected_label,
                    )
                    self.send_json({"indices": indices, "count": len(indices)})
                    return
                if parsed.path == "/api/image":
                    query = parse_qs(parsed.query)
                    crop_id = single_query(query, "crop_id")
                    self.send_png(application.load_image_png(crop_id))
                    return
                self.send_error(HTTPStatus.NOT_FOUND)
            except (ValueError, IndexError, KeyError) as error:
                self.send_json({"error": str(error)}, status=HTTPStatus.BAD_REQUEST)
            except Exception as error:  # pragma: no cover - defensive server boundary
                self.send_json(
                    {"error": f"{type(error).__name__}: {error}"},
                    status=HTTPStatus.INTERNAL_SERVER_ERROR,
                )

        def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
            try:
                parsed = urlparse(self.path)
                payload = self.read_json_body()
                if parsed.path == "/api/review":
                    crop_id = str(payload.get("crop_id", ""))
                    candidate = application.by_crop_id.get(crop_id)
                    if candidate is None:
                        raise KeyError(crop_id)
                    decision = str(payload.get("decision", ""))
                    corrected_label_raw = payload.get("corrected_label")
                    corrected_label = (
                        None
                        if corrected_label_raw in (None, "")
                        else str(corrected_label_raw)
                    )
                    note = str(payload.get("note", ""))
                    review = application.review_store.save(
                        candidate,
                        decision=decision,
                        corrected_label=corrected_label,
                        note=note,
                    )
                    self.send_json(
                        {
                            "review": review,
                            "summary": application.summary(),
                        }
                    )
                    return
                if parsed.path == "/api/review/delete":
                    crop_id = str(payload.get("crop_id", ""))
                    if crop_id not in application.by_crop_id:
                        raise KeyError(crop_id)
                    deleted = application.review_store.delete(crop_id)
                    self.send_json(
                        {
                            "deleted": deleted,
                            "summary": application.summary(),
                        }
                    )
                    return
                self.send_error(HTTPStatus.NOT_FOUND)
            except (ValueError, KeyError) as error:
                self.send_json({"error": str(error)}, status=HTTPStatus.BAD_REQUEST)
            except Exception as error:  # pragma: no cover - defensive server boundary
                self.send_json(
                    {"error": f"{type(error).__name__}: {error}"},
                    status=HTTPStatus.INTERNAL_SERVER_ERROR,
                )

        def log_message(self, format: str, *args: Any) -> None:
            return

        def read_json_body(self) -> dict[str, Any]:
            length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(length)
            value = json.loads(raw.decode("utf-8"))
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

        def send_html(self, content: str) -> None:
            raw = content.encode("utf-8")
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(raw)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(raw)

        def send_png(self, content: bytes) -> None:
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "image/png")
            self.send_header("Content-Length", str(len(content)))
            self.send_header("Cache-Control", "private, max-age=3600")
            self.end_headers()
            self.wfile.write(content)

    return Handler


def single_query(query: dict[str, list[str]], key: str, default: str | None = None) -> str:
    values = query.get(key)
    if not values:
        if default is None:
            raise ValueError(f"Missing query parameter: {key}")
        return default
    return values[0]


def float_or_none(value: str | None) -> float | None:
    if value in (None, ""):
        return None
    return float(value)


def angle_sort_key(field_name: str) -> tuple[float, str]:
    angle = field_name[: -len("_prediction")]
    numeric = angle.removesuffix("deg")
    try:
        return float(numeric), angle
    except ValueError:
        return float("inf"), angle


PAGE_HTML = r"""<!doctype html>
<html lang="ja">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Tile crop quality review</title>
<style>
:root { color-scheme: light dark; font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }
* { box-sizing: border-box; }
body { margin: 0; background: Canvas; color: CanvasText; }
header { position: sticky; top: 0; z-index: 5; display: flex; flex-wrap: wrap; gap: 10px; align-items: center; padding: 10px 14px; border-bottom: 1px solid color-mix(in srgb, CanvasText 25%, transparent); background: Canvas; }
header .grow { flex: 1; }
select, button, textarea { font: inherit; }
select, button { min-height: 38px; }
button { cursor: pointer; padding: 7px 12px; }
button.primary { font-weight: 700; }
main { display: grid; grid-template-columns: minmax(360px, 46vw) 1fr; gap: 18px; padding: 18px; max-width: 1500px; margin: 0 auto; }
.viewer { min-width: 0; }
.crop-frame { display: grid; place-items: center; min-height: 520px; padding: 14px; border: 1px solid color-mix(in srgb, CanvasText 20%, transparent); background: color-mix(in srgb, Canvas 92%, CanvasText 8%); }
.crop-frame img { width: min(100%, 600px); height: min(68vh, 600px); object-fit: contain; image-rendering: auto; }
.hero { display: flex; flex-wrap: wrap; gap: 10px; align-items: baseline; margin: 0 0 12px; }
.hero .pair { font-size: 30px; font-weight: 800; }
.badge { display: inline-block; padding: 3px 8px; border: 1px solid currentColor; border-radius: 999px; font-size: 12px; }
.panel { border: 1px solid color-mix(in srgb, CanvasText 20%, transparent); padding: 12px; margin-bottom: 12px; }
.angles { display: grid; grid-template-columns: repeat(auto-fit, minmax(145px, 1fr)); gap: 8px; }
.angle { padding: 8px; border: 1px solid color-mix(in srgb, CanvasText 18%, transparent); font-family: ui-monospace, SFMono-Regular, Consolas, monospace; font-size: 12px; }
.decision-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; }
.decision-grid button { min-height: 52px; text-align: left; }
.decision-grid .label-error { grid-column: 1 / -1; }
label.control { display: grid; gap: 5px; margin: 9px 0; }
textarea { width: 100%; min-height: 62px; resize: vertical; }
.metadata { margin: 0; white-space: pre-wrap; overflow-wrap: anywhere; font: 12px/1.45 ui-monospace, SFMono-Regular, Consolas, monospace; }
.reviewed { border-left: 6px solid currentColor; }
.status-line { font-size: 13px; opacity: .85; }
.shortcuts { font-size: 12px; opacity: .72; }
.error { color: #d33; font-weight: 700; }
@media (max-width: 900px) { main { grid-template-columns: 1fr; } .crop-frame { min-height: 360px; } }
</style>
</head>
<body>
<header>
  <strong>Tile crop quality review</strong>
  <span id="progress" class="status-line"></span>
  <span class="grow"></span>
  <label>表示
    <select id="review-filter">
      <option value="unreviewed">未判定</option>
      <option value="all">全件</option>
      <option value="reviewed">判定済み</option>
      <option value="label_error">ラベルミス</option>
      <option value="false_detection">モデル誤分類</option>
      <option value="unusable_crop">切れ/不適</option>
      <option value="background">背景</option>
    </select>
  </label>
  <label>Tier
    <select id="tier-filter">
      <option value="">all</option>
      <option value="1">1</option><option value="2">2</option><option value="3">3</option>
    </select>
  </label>
  <label>source
    <select id="source-filter">
      <option value="">all</option><option value="jp">jp</option><option value="manual">manual</option>
    </select>
  </label>
  <label>修正ラベル
    <select id="corrected-label-filter">
      <option value="">all</option>
      <option value="5m">5m</option><option value="5p">5p</option><option value="5s">5s</option>
      <option value="red5m">red5m</option><option value="red5p">red5p</option><option value="red5s">red5s</option>
    </select>
  </label>
  <button id="prev">←</button><button id="next">→</button>
</header>
<main>
  <section class="viewer">
    <div class="crop-frame"><img id="crop-image" alt="tile crop"></div>
  </section>
  <section>
    <div id="error" class="error"></div>
    <div class="hero">
      <span id="pair" class="pair"></span>
      <span id="tier" class="badge"></span>
      <span id="consensus" class="badge"></span>
    </div>
    <div class="panel">
      <strong>回転consensus</strong>
      <div id="angles" class="angles"></div>
    </div>
    <div id="decision-panel" class="panel">
      <strong>人手判定</strong>
      <label class="control">ラベルミス時の正解
        <select id="corrected-label"></select>
      </label>
      <div class="decision-grid">
        <button class="label-error primary" data-decision="label_error">1 — ラベルミス → 選択した正解ラベル</button>
        <button data-decision="false_detection">2 — モデルの単純な誤分類</button>
        <button data-decision="unusable_crop">3 — 切れ/複数牌/判別不能で学習不適</button>
        <button data-decision="background">4 — 背景</button>
      </div>
      <label class="control">メモ（任意）<textarea id="note"></textarea></label>
      <div style="display:flex; gap:8px; align-items:center; flex-wrap:wrap">
        <button id="clear-review">判定を解除</button>
        <span id="current-review" class="status-line"></span>
      </div>
      <div class="shortcuts">キー: 1=ラベルミス / 2=モデル誤分類 / 3=不適 / 4=背景 / ←→=移動</div>
    </div>
    <div class="panel">
      <strong>metadata</strong>
      <pre id="metadata" class="metadata"></pre>
    </div>
  </section>
</main>
<script>
const state = { indices: [], position: 0, candidate: null, summary: null, labels: [] };
const $ = id => document.getElementById(id);

async function api(url, options) {
  const response = await fetch(url, options);
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || response.statusText);
  return data;
}

async function refreshSummary() {
  state.summary = await api('/api/summary');
  state.labels = state.summary.labels;
  const select = $('corrected-label');
  const old = select.value;
  select.innerHTML = state.labels.map(label => `<option value="${escapeHtml(label)}">${escapeHtml(label)}</option>`).join('');
  if (state.labels.includes(old)) select.value = old;
  renderProgress();
}

function renderProgress() {
  const s = state.summary;
  if (!s) return;
  $('progress').textContent = `判定 ${s.reviewed_count}/${s.candidate_count} / 未判定 ${s.unreviewed_count}`;
}

async function refreshIndices({keepCropId = null} = {}) {
  const params = new URLSearchParams({
    review: $('review-filter').value,
    tier: $('tier-filter').value,
    source: $('source-filter').value,
    corrected_label: $('corrected-label-filter').value,
  });
  const data = await api('/api/indices?' + params.toString());
  state.indices = data.indices;
  if (!state.indices.length) {
    state.candidate = null;
    $('pair').textContent = '該当候補なし';
    $('crop-image').removeAttribute('src');
    $('angles').innerHTML = '';
    $('metadata').textContent = '';
    $('current-review').textContent = '';
    return;
  }
  let position = Math.min(state.position, state.indices.length - 1);
  if (keepCropId) {
    for (let p = 0; p < state.indices.length; p++) {
      const candidate = await api('/api/candidate?index=' + state.indices[p]);
      if (candidate.crop_id === keepCropId) { position = p; break; }
      if (p >= 50) break;
    }
  }
  state.position = Math.max(0, position);
  await loadCurrent();
}

async function loadCurrent() {
  if (!state.indices.length) return;
  const globalIndex = state.indices[state.position];
  state.candidate = await api('/api/candidate?index=' + globalIndex);
  renderCandidate();
}

function renderCandidate() {
  const c = state.candidate;
  if (!c) return;
  $('error').textContent = '';
  $('crop-image').src = c.image_url + '&t=' + Date.now();
  $('pair').textContent = `${c.expected_label} → ${c.consensus_prediction}`;
  $('tier').textContent = `Tier ${c.tier}`;
  $('consensus').textContent = `${c.consensus_count}/${c.angles.length}  conf=${c.consensus_confidence.toFixed(4)}`;
  $('angles').innerHTML = c.angles.map(a =>
    `<div class="angle"><b>${escapeHtml(a.angle)}</b><br>${escapeHtml(a.prediction)} conf=${fmt(a.confidence)}<br>expected=${fmt(a.expected_confidence)}</div>`
  ).join('');
  $('metadata').textContent = [
    `rank: ${c.rank}  (${state.position + 1}/${state.indices.length} in filter)`,
    `crop: ${c.crop_id}`,
    `source: ${c.source}/${c.source_partition}`,
    `original label: ${c.original_label}`,
    `expected base: ${c.expected_label}`,
    `source image: ${c.source_image_path}`,
    `source image id: ${c.source_image_id}`,
    `source ann: ${c.source_annotation_id}`,
    `capture: ${c.capture_id}`,
    `layout: ${c.layout_id} (${c.layout_ordinal})`,
    `region: ${c.region}`,
    `slot: ${c.group_name}/${c.group_ordinal}/${c.tile_ordinal}`,
    `condition: ${c.brightness}/${c.shadow}`,
    `annotation angle: ${c.annotation_angle_deg}`,
    `expected rotation: ${c.expected_rotation_deg}`,
  ].join('\n');

  const defaultLabel = state.labels.includes(c.consensus_prediction) ? c.consensus_prediction : c.expected_label;
  $('corrected-label').value = c.review?.corrected_label || defaultLabel;
  $('note').value = c.review?.note || '';
  $('decision-panel').classList.toggle('reviewed', !!c.review);
  $('current-review').textContent = c.review
    ? `現在: ${decisionLabel(c.review.decision)}${c.review.corrected_label ? ' → ' + c.review.corrected_label : ''}`
    : '未判定';
}

async function saveDecision(decision) {
  const c = state.candidate;
  if (!c) return;
  try {
    const corrected = decision === 'label_error' ? $('corrected-label').value : null;
    const data = await api('/api/review', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({crop_id: c.crop_id, decision, corrected_label: corrected, note: $('note').value}),
    });
    state.summary = data.summary;
    renderProgress();
    const oldPosition = state.position;
    await refreshIndices();
    if ($('review-filter').value !== 'unreviewed' && state.indices.length) {
      state.position = Math.min(oldPosition + 1, state.indices.length - 1);
      await loadCurrent();
    }
  } catch (error) {
    $('error').textContent = String(error);
  }
}

async function clearReview() {
  if (!state.candidate) return;
  try {
    const data = await api('/api/review/delete', {
      method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({crop_id: state.candidate.crop_id}),
    });
    state.summary = data.summary;
    renderProgress();
    await refreshIndices();
  } catch (error) { $('error').textContent = String(error); }
}

function move(delta) {
  if (!state.indices.length) return;
  state.position = Math.max(0, Math.min(state.indices.length - 1, state.position + delta));
  loadCurrent();
}

function decisionLabel(value) {
  return ({
    label_error: 'ラベルミス', false_detection: 'モデル誤分類',
    unusable_crop: '切れ/不適', background: '背景'
  })[value] || value;
}
function fmt(v) { return v == null ? '' : Number(v).toFixed(4); }
function escapeHtml(value) {
  return String(value).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
}

for (const button of document.querySelectorAll('[data-decision]')) {
  button.addEventListener('click', () => saveDecision(button.dataset.decision));
}
$('clear-review').addEventListener('click', clearReview);
$('prev').addEventListener('click', () => move(-1));
$('next').addEventListener('click', () => move(1));
for (const id of ['review-filter', 'tier-filter', 'source-filter', 'corrected-label-filter']) {
  $(id).addEventListener('change', () => { state.position = 0; refreshIndices(); });
}
document.addEventListener('keydown', event => {
  if (event.target.matches('textarea, select, input')) return;
  if (event.key === '1') saveDecision('label_error');
  else if (event.key === '2') saveDecision('false_detection');
  else if (event.key === '3') saveDecision('unusable_crop');
  else if (event.key === '4') saveDecision('background');
  else if (event.key === 'ArrowLeft') move(-1);
  else if (event.key === 'ArrowRight') move(1);
});

(async () => {
  try {
    await refreshSummary();
    await refreshIndices();
  } catch (error) { $('error').textContent = String(error); }
})();
</script>
</body>
</html>
"""


if __name__ == "__main__":
    main()
