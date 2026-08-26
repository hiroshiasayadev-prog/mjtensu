from __future__ import annotations

import argparse
import html
import json
import sqlite3
from collections import defaultdict
from pathlib import Path
from typing import Any


MODEL_ORDER = ("rgb", "cr", "ycr")


def parse_args() -> argparse.Namespace:
    repository_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(
        description=(
            "Extract the union of RGB/Cr/Y+Cr red-five classifier errors into one "
            "deduplicated image gallery for manual audit."
        )
    )
    parser.add_argument("--repository-root", type=Path, default=repository_root)
    parser.add_argument(
        "--errors-jsonl",
        type=Path,
        help=(
            "Defaults to <repository-root>/.local/recognition/red_five_runs/"
            "c8_rgb_cr_ycr_seed42/all_samples_errors.jsonl."
        ),
    )
    parser.add_argument(
        "--source-database",
        type=Path,
        help=(
            "Optional source DB. If omitted, red_five_all.sqlite is preferred when "
            "present, otherwise tile_crop_dataset/dataset.sqlite is used."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help=(
            "Defaults to <repository-root>/.local/recognition/red_five_runs/"
            "c8_rgb_cr_ycr_seed42/model_error_audit."
        ),
    )
    parser.add_argument(
        "--angle",
        type=float,
        default=0.0,
        help="Only extract errors from this evaluation angle. Defaults to 0 degrees.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.repository_root.resolve()
    run_root = (
        root
        / ".local"
        / "recognition"
        / "red_five_runs"
        / "c8_rgb_cr_ycr_seed42"
    )
    errors_jsonl = (
        args.errors_jsonl.resolve()
        if args.errors_jsonl is not None
        else run_root / "all_samples_errors.jsonl"
    )
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else run_root / "model_error_audit"
    )
    source_database = resolve_source_database(root, args.source_database)

    if not errors_jsonl.is_file():
        raise FileNotFoundError(errors_jsonl)
    if not source_database.is_file():
        raise FileNotFoundError(source_database)

    output_dir.mkdir(parents=True, exist_ok=True)
    image_dir = output_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)

    errors = load_errors(errors_jsonl, angle=float(args.angle))
    by_crop: defaultdict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for error in errors:
        mode = str(error["input_mode"])
        crop_id = str(error["crop_id"])
        by_crop[crop_id][mode] = error

    print(
        f"[red-five-error-audit] error rows={len(errors)} unique crops={len(by_crop)} "
        f"angle={float(args.angle):g}"
    )

    source_kind = detect_source_kind(source_database)
    source_rows = fetch_source_rows(
        source_database,
        source_kind=source_kind,
        crop_ids=sorted(by_crop),
    )

    missing = sorted(set(by_crop) - set(source_rows))
    if missing:
        raise RuntimeError(f"Failed to find {len(missing)} crops in source DB: {missing[:10]}")

    manifest: list[dict[str, Any]] = []
    for index, crop_id in enumerate(sorted(by_crop), 1):
        model_errors = by_crop[crop_id]
        representative = next(iter(model_errors.values()))
        source = source_rows[crop_id]

        target_is_red = int(representative["target_is_red"])
        target_name = "red" if target_is_red else "normal"
        error_modes = [mode for mode in MODEL_ORDER if mode in model_errors]
        filename = safe_filename(
            f"{index:02d}_{crop_id}_{source['source_label']}_target-{target_name}_"
            f"errors-{'-'.join(error_modes)}.png"
        )
        (image_dir / filename).write_bytes(source["image_png"])

        model_payload: dict[str, Any] = {}
        for mode in MODEL_ORDER:
            error = model_errors.get(mode)
            if error is None:
                model_payload[mode] = {
                    "status": "correct",
                    "prediction_is_red": target_is_red,
                    "red_probability": None,
                }
            else:
                model_payload[mode] = {
                    "status": "error",
                    "prediction_is_red": int(error["prediction_is_red"]),
                    "red_probability": float(error["red_probability"]),
                }

        manifest.append(
            {
                "index": index,
                "crop_id": crop_id,
                "image": f"images/{filename}",
                "source": str(representative["source"]),
                "source_partition": str(representative["source_partition"]),
                "source_label": str(representative["source_label"]),
                "target_is_red": target_is_red,
                "target": target_name,
                "experiment_membership": str(representative["experiment_membership"]),
                "source_image_path": str(representative["source_image_path"]),
                "source_image_id": representative.get("source_image_id"),
                "source_annotation_id": str(representative["source_annotation_id"]),
                "capture_id": representative.get("capture_id"),
                "brightness": str(representative.get("brightness", "")),
                "shadow": str(representative.get("shadow", "")),
                "error_models": error_modes,
                "models": model_payload,
            }
        )

    summary = build_summary(manifest)
    manifest_payload = {
        "status": "completed",
        "angle_deg": float(args.angle),
        "errors_jsonl": str(errors_jsonl),
        "source_database": str(source_database),
        "source_kind": source_kind,
        "unique_error_crops": len(manifest),
        "summary": summary,
        "samples": manifest,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest_payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (output_dir / "index.html").write_text(
        render_html(manifest_payload),
        encoding="utf-8",
    )

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"HTML: {output_dir / 'index.html'}")
    print(f"manifest: {output_dir / 'manifest.json'}")


def resolve_source_database(root: Path, requested: Path | None) -> Path:
    if requested is not None:
        return requested.resolve()
    red_five_all = (
        root / ".local" / "recognition" / "red_five_datasets" / "red_five_all.sqlite"
    )
    if red_five_all.is_file():
        return red_five_all
    return root / ".local" / "recognition" / "tile_crop_dataset" / "dataset.sqlite"


def load_errors(path: Path, *, angle: float) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if abs(float(row["angle_deg"]) - angle) > 1.0e-9:
            continue
        mode = str(row["input_mode"])
        if mode not in MODEL_ORDER:
            continue
        result.append(row)
    return result


def detect_source_kind(database: Path) -> str:
    with sqlite3.connect(f"{database.as_uri()}?mode=ro", uri=True, timeout=60) as conn:
        tables = {
            str(row[0])
            for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }
        if "sample" in tables:
            columns = {str(row[1]) for row in conn.execute("PRAGMA table_info(sample)")}
            if {"crop_id", "source_label", "image_png"}.issubset(columns):
                return "red_five_all"
        if "tile_crop" in tables:
            columns = {str(row[1]) for row in conn.execute("PRAGMA table_info(tile_crop)")}
            if {"crop_id", "tile_label", "image_png"}.issubset(columns):
                return "tile_crop_dataset"
    raise ValueError(f"Unsupported source DB schema: {database}")


def fetch_source_rows(
    database: Path,
    *,
    source_kind: str,
    crop_ids: list[str],
) -> dict[str, dict[str, Any]]:
    if not crop_ids:
        return {}
    result: dict[str, dict[str, Any]] = {}
    with sqlite3.connect(f"{database.as_uri()}?mode=ro", uri=True, timeout=60) as conn:
        conn.row_factory = sqlite3.Row
        for start in range(0, len(crop_ids), 500):
            chunk = crop_ids[start : start + 500]
            placeholders = ",".join("?" for _ in chunk)
            if source_kind == "red_five_all":
                query = f"""
                    SELECT crop_id, source_label, image_png
                    FROM sample
                    WHERE crop_id IN ({placeholders})
                """
            elif source_kind == "tile_crop_dataset":
                query = f"""
                    SELECT crop_id, tile_label AS source_label, image_png
                    FROM tile_crop
                    WHERE crop_id IN ({placeholders})
                """
            else:
                raise ValueError(source_kind)
            for row in conn.execute(query, chunk):
                result[str(row["crop_id"])] = {
                    "source_label": str(row["source_label"]),
                    "image_png": bytes(row["image_png"]),
                }
    return result


def build_summary(manifest: list[dict[str, Any]]) -> dict[str, Any]:
    per_model = {mode: 0 for mode in MODEL_ORDER}
    patterns: defaultdict[str, int] = defaultdict(int)
    labels: defaultdict[str, int] = defaultdict(int)
    memberships: defaultdict[str, int] = defaultdict(int)

    for sample in manifest:
        for mode in sample["error_models"]:
            per_model[mode] += 1
        pattern = "+".join(sample["error_models"])
        patterns[pattern] += 1
        labels[str(sample["source_label"])] += 1
        memberships[str(sample["experiment_membership"])] += 1

    return {
        "unique_error_crops": len(manifest),
        "error_rows_by_model": per_model,
        "error_model_patterns": dict(sorted(patterns.items())),
        "unique_crops_by_source_label": dict(sorted(labels.items())),
        "unique_crops_by_membership": dict(sorted(memberships.items())),
    }


def render_html(payload: dict[str, Any]) -> str:
    cards = []
    for sample in payload["samples"]:
        model_cells = []
        for mode in MODEL_ORDER:
            model = sample["models"][mode]
            if model["status"] == "correct":
                text = "correct"
                css = "ok"
            else:
                pred = "red" if model["prediction_is_red"] else "normal"
                probability = model["red_probability"]
                text = f"ERROR → {pred}<br>p(red)={probability:.6f}"
                css = "bad"
            model_cells.append(
                f'<div class="model {css}"><b>{html.escape(mode.upper())}</b><br>{text}</div>'
            )

        cards.append(
            f"""
            <article class="card">
              <img src="{html.escape(sample['image'])}" alt="{html.escape(sample['crop_id'])}">
              <div class="info">
                <h2>#{sample['index']} {html.escape(sample['source_label'])}</h2>
                <div><b>crop:</b> <code>{html.escape(sample['crop_id'])}</code></div>
                <div><b>target:</b> {html.escape(sample['target'])}</div>
                <div><b>errors:</b> {html.escape(', '.join(sample['error_models']))}</div>
                <div><b>membership:</b> {html.escape(sample['experiment_membership'])}</div>
                <div><b>source:</b> {html.escape(sample['source'])}/{html.escape(sample['source_partition'])}</div>
                <div><b>brightness/shadow:</b> {html.escape(sample['brightness'])} / {html.escape(sample['shadow'])}</div>
                <div><b>source image:</b> <code>{html.escape(sample['source_image_path'])}</code></div>
                <div class="models">{''.join(model_cells)}</div>
              </div>
            </article>
            """
        )

    summary_json = html.escape(json.dumps(payload["summary"], ensure_ascii=False, indent=2))
    return f"""<!doctype html>
<html lang="ja">
<head>
<meta charset="utf-8">
<title>Red-five model error audit</title>
<style>
body {{ font-family: sans-serif; margin: 20px; background: #eee; }}
pre {{ background: white; padding: 12px; border-radius: 8px; overflow-x: auto; }}
.grid {{ display: grid; gap: 16px; }}
.card {{ display: grid; grid-template-columns: 260px 1fr; gap: 16px; background: white; padding: 14px; border-radius: 10px; }}
.card img {{ width: 256px; height: 256px; object-fit: contain; background: #888; }}
h2 {{ margin: 0 0 8px; }}
code {{ overflow-wrap: anywhere; }}
.models {{ display: grid; grid-template-columns: repeat(3, minmax(130px, 1fr)); gap: 8px; margin-top: 12px; }}
.model {{ padding: 10px; border-radius: 6px; text-align: center; }}
.ok {{ background: #dff5e1; }}
.bad {{ background: #ffdada; }}
@media (max-width: 700px) {{
  .card {{ grid-template-columns: 1fr; }}
  .models {{ grid-template-columns: 1fr; }}
}}
</style>
</head>
<body>
<h1>Red-five model error audit</h1>
<p>angle={payload['angle_deg']:g}°, unique crops={payload['unique_error_crops']}</p>
<pre>{summary_json}</pre>
<div class="grid">
{''.join(cards)}
</div>
</body>
</html>
"""


def safe_filename(value: str) -> str:
    allowed = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_.")
    return "".join(character if character in allowed else "_" for character in value)


if __name__ == "__main__":
    main()
