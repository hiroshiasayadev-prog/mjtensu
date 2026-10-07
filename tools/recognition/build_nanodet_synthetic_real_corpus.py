from __future__ import annotations

import argparse
import hashlib
import json
import random
import sqlite3
import tarfile
from pathlib import Path

import build_nanodet_capture_finetune_dataset as capture

ONE_CLASS = [{"id": 1, "name": "mahjong_tile", "supercategory": "mahjong_tile"}]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")


def rewrite_coco(source: Path, output: Path) -> dict[str, int]:
    data = load_json(source)
    for image in data["images"]:
        image["file_name"] = Path(image["file_name"]).name
    for ann in data["annotations"]:
        ann["category_id"] = 1
    data["categories"] = ONE_CLASS
    write_json(output, data)
    return {"images": len(data["images"]), "annotations": len(data["annotations"])}


def complete_capture_rows(database: Path, campaign_id: str) -> tuple[list[dict], list[dict]]:
    con = sqlite3.connect(database)
    con.row_factory = sqlite3.Row
    try:
        layout_rows = [dict(row) for row in con.execute(
            """
            SELECT ct.layout_id, ct.layout_ordinal,
                   COUNT(*) AS task_count,
                   COUNT(c.id) AS capture_count,
                   SUM(CASE WHEN ca.status = 'complete' THEN 1 ELSE 0 END) AS complete_count
            FROM capture_task ct
            LEFT JOIN capture c ON c.task_id = ct.id
            LEFT JOIN capture_annotation ca ON ca.capture_id = c.id
            WHERE ct.campaign_id = ?
            GROUP BY ct.layout_id, ct.layout_ordinal
            ORDER BY ct.layout_ordinal
            """,
            (campaign_id,),
        )]
        complete_layouts = {
            str(row["layout_id"])
            for row in layout_rows
            if int(row["task_count"]) > 0
            and int(row["task_count"]) == int(row["capture_count"] or 0)
            and int(row["task_count"]) == int(row["complete_count"] or 0)
        }
        rows = [dict(row) for row in con.execute(
            """
            SELECT c.id AS capture_id, c.composite_path, c.manifest_json,
                   ct.layout_id, ct.layout_ordinal, ct.environment_ordinal,
                   ct.brightness, ct.shadow, ca.annotation_json
            FROM capture c
            JOIN capture_task ct ON ct.id = c.task_id
            JOIN capture_annotation ca ON ca.capture_id = c.id
            WHERE ct.campaign_id = ? AND ca.status = 'complete'
            ORDER BY ct.layout_ordinal, ct.environment_ordinal
            """,
            (campaign_id,),
        ) if str(row["layout_id"]) in complete_layouts]
        return layout_rows, rows
    finally:
        con.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", type=Path, required=True)
    ap.add_argument("--synthetic-root", type=Path, required=True)
    ap.add_argument("--capture-root", type=Path, required=True)
    ap.add_argument("--campaign-id", default="initial-120")
    ap.add_argument("--real-train-fraction", type=float, default=0.8)
    ap.add_argument("--real-split-seed", type=int, default=42)
    ap.add_argument("--reserved-val-layout", action="append", default=[])
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    repo = args.repo.resolve()
    synthetic = args.synthetic_root.resolve()
    capture_root = args.capture_root.resolve()
    out = args.output.resolve()
    stage = out.parent / f"{out.stem}.stage"

    if not 0 < args.real_train_fraction < 1:
        raise ValueError("real-train-fraction must be strictly between zero and one")
    if stage.exists():
        import shutil
        shutil.rmtree(stage)
    (stage / "annotations").mkdir(parents=True)

    synthetic_train = synthetic / "annotations" / "instances_train.json"
    synthetic_val = synthetic / "annotations" / "instances_val.json"
    database = capture_root / "dataset.sqlite"
    layout_path = repo / "tools" / "recognition" / "capture_layout.v1.json"
    for path in (synthetic_train, synthetic_val, database, layout_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    counts = {
        "synthetic_train": rewrite_coco(
            synthetic_train, stage / "annotations" / "synthetic_train.coco.json"
        ),
        "synthetic_val": rewrite_coco(
            synthetic_val, stage / "annotations" / "synthetic_val.coco.json"
        ),
    }

    layout_rows, real_rows = complete_capture_rows(database, args.campaign_id)
    layout_ids = sorted(
        {str(row["layout_id"]) for row in real_rows},
        key=lambda layout_id: min(
            int(row["layout_ordinal"]) for row in real_rows if str(row["layout_id"]) == layout_id
        ),
    )
    reserved_val = frozenset(str(x) for x in args.reserved_val_layout)
    unknown_reserved = reserved_val - frozenset(layout_ids)
    if unknown_reserved:
        raise ValueError(f"reserved validation layouts are not complete/eligible: {sorted(unknown_reserved)}")
    remaining_layouts = [layout_id for layout_id in layout_ids if layout_id not in reserved_val]
    shuffled = list(remaining_layouts)
    random.Random(int(args.real_split_seed)).shuffle(shuffled)
    target_train_count = round(len(layout_ids) * float(args.real_train_fraction))
    target_train_count = max(1, min(target_train_count, len(shuffled)))
    train_layouts = frozenset(shuffled[:target_train_count])
    val_layouts = reserved_val | frozenset(shuffled[target_train_count:])
    layout = capture.load_json(layout_path)
    capture.validate_capture_layout(layout)

    real_train = capture.captures_to_coco(
        real_rows,
        included_layout_ids=train_layouts,
        repository_root=repo,
        storage_root=capture_root,
        layout=layout,
        description="Complete-layout real capture training images",
        check_images=True,
    )
    real_val = capture.captures_to_coco(
        real_rows,
        included_layout_ids=val_layouts,
        repository_root=repo,
        storage_root=capture_root,
        layout=layout,
        description="Layout-disjoint real capture validation images",
        check_images=True,
    )
    for payload in (real_train, real_val):
        payload["categories"] = ONE_CLASS
        for image in payload["images"]:
            image["file_name"] = Path(image["file_name"]).name
        for ann in payload["annotations"]:
            ann["category_id"] = 1
    write_json(stage / "annotations" / "real_train.coco.json", real_train)
    write_json(stage / "annotations" / "real_val.coco.json", real_val)
    counts["real_train"] = capture.coco_counts(real_train)
    counts["real_val"] = capture.coco_counts(real_val)

    incomplete_layouts = [
        {
            "layout_id": str(row["layout_id"]),
            "layout_ordinal": int(row["layout_ordinal"]),
            "task_count": int(row["task_count"]),
            "capture_count": int(row["capture_count"] or 0),
            "complete_count": int(row["complete_count"] or 0),
        }
        for row in layout_rows
        if str(row["layout_id"]) not in set(layout_ids)
    ]
    provenance = {
        "schema": "mjtensu.nanodet-synthetic-real-corpus/v1",
        "synthetic_root": str(synthetic),
        "synthetic_package_manifest_sha256": sha256(synthetic / "package_manifest.json"),
        "synthetic_validation_report_sha256": sha256(synthetic / "qa" / "validation_report.json"),
        "campaign_id": args.campaign_id,
        "real_split_unit": "layout_id",
        "real_split_seed": int(args.real_split_seed),
        "real_train_fraction": float(args.real_train_fraction),
        "reserved_validation_layouts": sorted(reserved_val),
        "eligible_complete_layouts": layout_ids,
        "real_train_layouts": sorted(train_layouts),
        "real_val_layouts": sorted(val_layouts),
        "excluded_incomplete_layouts": incomplete_layouts,
        "counts": counts,
        "jp_v2": {
            "storage": "reused from sealed nanodet/mahjong-training-baseline-v1 archive",
            "included_in_this_archive": False,
        },
    }
    write_json(stage / "provenance.json", provenance)

    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        out.unlink()
    with tarfile.open(out, "w", format=tarfile.PAX_FORMAT) as tf:
        for path in sorted((stage / "annotations").iterdir()):
            tf.add(path, arcname=f"annotations/{path.name}", recursive=False)
        tf.add(stage / "provenance.json", arcname="provenance.json", recursive=False)

        synthetic_images = synthetic / "images"
        for path in sorted(synthetic_images.iterdir()):
            if path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}:
                tf.add(path, arcname=f"images/synthetic/{path.name}", recursive=False)

        seen: set[str] = set()
        for row in real_rows:
            rel = capture.safe_relative_path(str(row["composite_path"]))
            src = capture_root / rel
            if not src.is_file():
                raise FileNotFoundError(src)
            name = src.name
            if name in seen:
                raise ValueError(f"duplicate real capture basename: {name}")
            seen.add(name)
            tf.add(src, arcname=f"images/real_capture/{name}", recursive=False)

        tf.add(
            synthetic / "package_manifest.json",
            arcname="source/synthetic_package_manifest.json",
            recursive=False,
        )
        tf.add(
            synthetic / "qa" / "validation_report.json",
            arcname="source/synthetic_validation_report.json",
            recursive=False,
        )

    result = {
        "archive": str(out),
        "bytes": out.stat().st_size,
        "sha256": sha256(out),
        "counts": counts,
        "real_complete_layouts": len(layout_ids),
        "real_train_layouts": len(train_layouts),
        "real_val_layouts": len(val_layouts),
        "real_images_archived": len(seen),
        "excluded_incomplete_layouts": incomplete_layouts,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
