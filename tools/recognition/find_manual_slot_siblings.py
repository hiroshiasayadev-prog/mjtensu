from __future__ import annotations

import argparse
import base64
import csv
import html
import json
import sqlite3
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


SLOT_COLUMNS = (
    "layout_id",
    "region",
    "group_name",
    "group_ordinal",
    "tile_ordinal",
)

ROW_COLUMNS = (
    "crop_id",
    "tile_label",
    "capture_id",
    "layout_id",
    "layout_ordinal",
    "region",
    "group_name",
    "group_ordinal",
    "tile_ordinal",
    "brightness",
    "shadow",
    "source_annotation_id",
    "source_image_path",
    "annotation_angle_deg",
    "expected_rotation_deg",
)


def parse_args() -> argparse.Namespace:
    repository_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(
        description=(
            "Given one or more suspicious manual crop IDs, resolve their logical manual "
            "layout slots and export every crop from the same slot across capture conditions."
        )
    )
    parser.add_argument(
        "--database",
        type=Path,
        default=repository_root
        / ".local"
        / "recognition"
        / "tile_crop_dataset"
        / "dataset.sqlite",
    )
    parser.add_argument(
        "--crop-id",
        action="append",
        default=[],
        help=(
            "Suspicious crop ID. May be repeated. The leading 'manual:' is optional, "
            "e.g. cap_xxx:melds:uuid."
        ),
    )
    parser.add_argument(
        "--crop-id-file",
        type=Path,
        help=(
            "Optional UTF-8 text file containing one crop ID per line. Blank lines and "
            "lines beginning with # are ignored."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=repository_root
        / ".local"
        / "recognition"
        / "manual_slot_siblings",
    )
    return parser.parse_args()


def normalize_crop_id(value: str) -> str:
    value = value.strip()
    if not value:
        raise ValueError("Empty crop ID")
    return value if value.startswith("manual:") else f"manual:{value}"


def load_requested_crop_ids(args: argparse.Namespace) -> list[str]:
    values = list(args.crop_id)
    if args.crop_id_file is not None:
        for line in args.crop_id_file.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            values.append(stripped)

    normalized = list(dict.fromkeys(normalize_crop_id(value) for value in values))
    if not normalized:
        raise ValueError("Provide at least one --crop-id or --crop-id-file")
    return normalized


def row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
    return {key: row[key] for key in row.keys()}


def slot_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return tuple(row[column] for column in SLOT_COLUMNS)


def slot_key_dict(row: dict[str, Any]) -> dict[str, Any]:
    return {column: row[column] for column in SLOT_COLUMNS}


def fetch_seed_rows(
    connection: sqlite3.Connection,
    crop_ids: Iterable[str],
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    missing: list[str] = []
    for crop_id in crop_ids:
        row = connection.execute(
            """
            SELECT *
            FROM tile_crop
            WHERE crop_id = ? AND source = 'manual'
            """,
            (crop_id,),
        ).fetchone()
        if row is None:
            missing.append(crop_id)
            continue
        item = row_to_dict(row)
        missing_slot_columns = [column for column in SLOT_COLUMNS if item[column] is None]
        if missing_slot_columns:
            raise ValueError(
                f"Crop {crop_id} has null slot identity columns: "
                f"{', '.join(missing_slot_columns)}"
            )
        result.append(item)

    if missing:
        raise ValueError("Crop ID(s) not found in manual dataset: " + ", ".join(missing))
    return result


def fetch_slot_rows(
    connection: sqlite3.Connection,
    seed: dict[str, Any],
) -> list[dict[str, Any]]:
    rows = connection.execute(
        """
        SELECT *
        FROM tile_crop
        WHERE source = 'manual'
          AND layout_id = ?
          AND region = ?
          AND group_name = ?
          AND group_ordinal = ?
          AND tile_ordinal = ?
        ORDER BY
            brightness,
            shadow,
            capture_id,
            source_annotation_id
        """,
        slot_key(seed),
    ).fetchall()
    return [row_to_dict(row) for row in rows]


def condition_name(row: dict[str, Any]) -> str:
    return f"{row.get('brightness') or ''}/{row.get('shadow') or ''}"


def image_data_uri(image_png: bytes) -> str:
    encoded = base64.b64encode(bytes(image_png)).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def export_csv(path: Path, groups: list[dict[str, Any]]) -> None:
    fieldnames = [
        "slot_index",
        "seed_crop_ids",
        "condition",
        *ROW_COLUMNS,
    ]
    with path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        for slot_index, group in enumerate(groups, start=1):
            seed_ids = " | ".join(group["seed_crop_ids"])
            for row in group["rows"]:
                payload = {
                    "slot_index": slot_index,
                    "seed_crop_ids": seed_ids,
                    "condition": condition_name(row),
                }
                payload.update({column: row.get(column) for column in ROW_COLUMNS})
                writer.writerow(payload)


def export_json(path: Path, groups: list[dict[str, Any]]) -> None:
    serializable: list[dict[str, Any]] = []
    for slot_index, group in enumerate(groups, start=1):
        serializable.append(
            {
                "slot_index": slot_index,
                "slot": group["slot"],
                "seed_crop_ids": group["seed_crop_ids"],
                "rows": [
                    {
                        "condition": condition_name(row),
                        **{column: row.get(column) for column in ROW_COLUMNS},
                    }
                    for row in group["rows"]
                ],
            }
        )
    path.write_text(
        json.dumps(serializable, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def export_html(path: Path, groups: list[dict[str, Any]]) -> None:
    sections: list[str] = []
    for slot_index, group in enumerate(groups, start=1):
        slot = group["slot"]
        cards: list[str] = []
        for row in group["rows"]:
            cards.append(
                "<article class='card'>"
                f"<img src='{image_data_uri(row['image_png'])}' alt='tile crop'>"
                f"<div class='label'>label: <b>{html.escape(str(row['tile_label']))}</b></div>"
                f"<div>condition: {html.escape(condition_name(row))}</div>"
                f"<div>capture: {html.escape(str(row['capture_id']))}</div>"
                f"<div>crop: {html.escape(str(row['crop_id']))}</div>"
                f"<div>source ann: {html.escape(str(row['source_annotation_id']))}</div>"
                "</article>"
            )

        slot_text = " / ".join(
            f"{column}={slot[column]}" for column in SLOT_COLUMNS
        )
        seed_text = "<br>".join(html.escape(value) for value in group["seed_crop_ids"])
        sections.append(
            f"<section><h2>slot {slot_index}</h2>"
            f"<p><b>{html.escape(slot_text)}</b></p>"
            f"<p>seed crop(s):<br>{seed_text}</p>"
            f"<p>siblings found: {len(group['rows'])}</p>"
            f"<div class='grid'>{''.join(cards)}</div></section>"
        )

    document = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Manual slot siblings</title>
<style>
body {{ font-family: system-ui, sans-serif; margin: 20px; background: #f4f4f4; }}
section {{ margin: 0 0 36px 0; padding: 16px; background: white; border: 1px solid #ccc; }}
.grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(230px, 1fr)); gap: 12px; }}
.card {{ border: 1px solid #bbb; padding: 10px; font-size: 12px; overflow-wrap: anywhere; }}
.card img {{ display: block; width: 160px; height: 160px; object-fit: contain; margin: 0 auto 8px auto; background: #ddd; image-rendering: auto; }}
.label {{ font-size: 16px; margin-bottom: 4px; }}
</style>
</head>
<body>
<h1>Manual logical-slot siblings</h1>
<p>Rows are grouped by layout_id + region + group_name + group_ordinal + tile_ordinal.</p>
{''.join(sections)}
</body>
</html>
"""
    path.write_text(document, encoding="utf-8")


def print_groups(groups: list[dict[str, Any]]) -> None:
    for slot_index, group in enumerate(groups, start=1):
        slot = group["slot"]
        print(f"\n=== slot {slot_index} ===")
        print(" ".join(f"{column}={slot[column]}" for column in SLOT_COLUMNS))
        print("seed(s):")
        for seed_crop_id in group["seed_crop_ids"]:
            print(f"  {seed_crop_id}")
        print(f"siblings={len(group['rows'])}")
        for row in group["rows"]:
            print(
                f"  {condition_name(row):24s} "
                f"label={str(row['tile_label']):8s} "
                f"capture={row['capture_id']} "
                f"crop={row['crop_id']}"
            )


def main() -> None:
    args = parse_args()
    database = args.database.resolve()
    if not database.is_file():
        raise FileNotFoundError(database)

    crop_ids = load_requested_crop_ids(args)
    with sqlite3.connect(database) as connection:
        connection.row_factory = sqlite3.Row
        seeds = fetch_seed_rows(connection, crop_ids)

        seeds_by_slot: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
        for seed in seeds:
            seeds_by_slot[slot_key(seed)].append(seed)

        groups: list[dict[str, Any]] = []
        for _, slot_seeds in seeds_by_slot.items():
            representative = slot_seeds[0]
            siblings = fetch_slot_rows(connection, representative)
            groups.append(
                {
                    "slot": slot_key_dict(representative),
                    "seed_crop_ids": [str(seed["crop_id"]) for seed in slot_seeds],
                    "rows": siblings,
                }
            )

    groups.sort(key=lambda group: tuple(str(group["slot"][column]) for column in SLOT_COLUMNS))

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    export_csv(output_dir / "siblings.csv", groups)
    export_json(output_dir / "siblings.json", groups)
    export_html(output_dir / "siblings.html", groups)
    print_groups(groups)
    print(f"\nCSV:  {output_dir / 'siblings.csv'}")
    print(f"JSON: {output_dir / 'siblings.json'}")
    print(f"HTML: {output_dir / 'siblings.html'}")


if __name__ == "__main__":
    main()
