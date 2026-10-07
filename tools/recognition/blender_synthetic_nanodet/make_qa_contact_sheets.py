from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("root", type=Path)
    p.add_argument("--per-sheet", type=int, default=24)
    return p.parse_args()

def spread(records: list[dict], count: int) -> list[dict]:
    if len(records) <= count:
        return records
    if count == 1:
        return [records[len(records)//2]]
    return [records[round(i*(len(records)-1)/(count-1))] for i in range(count)]

def draw_sample(root: Path, record: dict, size: int = 240) -> Image.Image:
    image = Image.open(root / record["image"]["file_name"]).convert("RGB")
    draw = ImageDraw.Draw(image)
    for ann in record["annotations"]:
        x,y,w,h = ann["bbox"]
        draw.rectangle((x,y,x+w,y+h), outline=(255,0,0), width=1)
    lighting = record["scene"]["lighting"]
    label = (
        f"{record['index']} {record['scene']['placement_regime']} "
        f"m{record['image']['meld_group_count']} "
        f"{lighting['temperature']}/{lighting['shadow_style']}/{lighting['brightness']}"
    )
    draw.rectangle((0,0,min(319,6*len(label)+4),13), fill=(0,0,0))
    draw.text((2,1), label, fill=(255,255,0))
    image.thumbnail((size,size))
    return image

def write_sheet(root: Path, name: str, records: list[dict], count: int) -> Path:
    selected = spread(records, count)
    cols = 6
    cell = 240
    rows = max(1, (len(selected)+cols-1)//cols)
    sheet = Image.new("RGB", (cols*cell, rows*cell), (24,24,24))
    for i, record in enumerate(selected):
        image = draw_sample(root, record, cell)
        sheet.paste(image, ((i%cols)*cell, (i//cols)*cell))
    out_dir = root / "qa" / "contact_sheets"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.png"
    sheet.save(path, optimize=True)
    return path

def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    records = [json.loads(p.read_text()) for p in sorted((root / "records").glob("synthetic_*.json"))]
    categories = {
        "normal": [
            r for r in records
            if r["scene"]["placement_regime"] in {"neat", "ordinary"}
            and r["scene"]["lighting"]["shadow_style"] in {"soft", "hard"}
            and r["scene"]["lighting"]["brightness"] != "dim"
        ],
        "messy": [r for r in records if r["scene"]["placement_regime"] == "messy"],
        "empty_meld": [r for r in records if int(r["image"]["meld_group_count"]) == 0],
        "white_dragon": [r for r in records if bool(r["image"].get("white_dragon_meld"))],
        "shadow_glare": [
            r for r in records
            if r["scene"]["lighting"]["shadow_style"] in {"partial", "glare"}
        ],
        "multi_meld": [r for r in records if int(r["image"]["meld_group_count"]) >= 3],
    }
    outputs = {}
    for name, matches in categories.items():
        if not matches:
            raise RuntimeError(f"no matches for {name}")
        path = write_sheet(root, name, matches, args.per_sheet)
        outputs[name] = {"matches": len(matches), "path": str(path)}
    summary_path = root / "qa" / "contact_sheets.json"
    summary_path.write_text(json.dumps(outputs, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(outputs, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
