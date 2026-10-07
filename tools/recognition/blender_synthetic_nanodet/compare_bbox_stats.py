from __future__ import annotations
import json
import math
import sqlite3
import sys
from pathlib import Path

REGIONS = ("completed_hand", "dora_indicators", "melds")

def quantile(values: list[float], p: float) -> float:
    values = sorted(values)
    return values[min(len(values) - 1, round((len(values) - 1) * p))]

def summarize(name: str, values: dict[str, list[tuple[float, float]]]) -> None:
    print(name)
    for region in REGIONS:
        pairs = values[region]
        if not pairs:
            print(region, "n=0")
            continue
        widths = [w for w, _ in pairs]
        heights = [h for _, h in pairs]
        areas = [w * h for w, h in pairs]
        print(region, "n", len(pairs),
              "w", tuple(round(quantile(widths, p), 1) for p in (0.1, 0.5, 0.9)),
              "h", tuple(round(quantile(heights, p), 1) for p in (0.1, 0.5, 0.9)),
              "area", tuple(round(quantile(areas, p)) for p in (0.1, 0.5, 0.9)))

def real_stats(repo: Path) -> dict[str, list[tuple[float, float]]]:
    layout = json.loads((repo / "tools/recognition/capture_layout.v1.json").read_text())
    con = sqlite3.connect(repo / ".local/recognition/capture_dataset/dataset.sqlite")
    con.row_factory = sqlite3.Row
    rows = con.execute("""SELECT c.manifest_json, a.annotation_json FROM capture c
        JOIN capture_task t ON t.id=c.task_id JOIN capture_annotation a ON a.capture_id=c.id
        WHERE t.campaign_id='initial-120' AND a.status='complete'""").fetchall()
    values = {region: [] for region in REGIONS}
    for row in rows:
        manifest = json.loads(row["manifest_json"])
        annotation = json.loads(row["annotation_json"])
        for region in REGIONS:
            pixel = manifest["regionRects"][region]["pixel"]
            crop_w = max(1, math.floor(float(pixel["width"]) + 0.5))
            crop_h = max(1, math.floor(float(pixel["height"]) + 0.5))
            dest = layout["regions"][region]["destination"]
            sx, sy = dest["width"] / crop_w, dest["height"] / crop_h
            for box in annotation["boxes"][region]:
                cx, cy = float(box["centerX"]), float(box["centerY"])
                hw, hh = float(box["width"]) / 2, float(box["height"]) / 2
                angle = math.radians(float(box["angleDeg"]))
                cosine, sine = math.cos(angle), math.sin(angle)
                pts = [(dest["x"] + (cx + lx*cosine - ly*sine)*sx,
                        dest["y"] + (cy + lx*sine + ly*cosine)*sy)
                       for lx, ly in ((-hw,-hh),(hw,-hh),(hw,hh),(-hw,hh))]
                values[region].append((max(x for x,_ in pts)-min(x for x,_ in pts),
                                       max(y for _,y in pts)-min(y for _,y in pts)))
    return values

def synthetic_stats(root: Path) -> dict[str, list[tuple[float, float]]]:
    coco = json.loads((root / "annotations/instances_all.json").read_text())
    return {region: [(a["bbox"][2], a["bbox"][3]) for a in coco["annotations"]
                     if a.get("region") == region] for region in REGIONS}

def main() -> None:
    repo = Path(__file__).resolve().parents[3]
    synthetic_root = Path(sys.argv[1]).resolve()
    summarize("real", real_stats(repo))
    summarize("synthetic", synthetic_stats(synthetic_root))

if __name__ == "__main__":
    main()
