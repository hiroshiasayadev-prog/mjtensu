from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

REGIONS = {
    "completed_hand": (7, 0, 306, 72),
    "dora_indicators": (7, 74, 306, 72),
    "melds": (74, 148, 172, 172),
}
ALL_TILES = tuple(
    [f"{n}{s}" for s in "mps" for n in range(1, 10)]
    + ["east", "south", "west", "north", "white", "green", "red"]
    + ["red5m", "red5p", "red5s"]
)
W, H = 0.027, 0.038

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("root", type=Path)
    p.add_argument("--minimum-images", type=int, default=1)
    p.add_argument("--near-duplicate-distance", type=int, default=4)
    return p.parse_args()

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def clip_ratio(box: list[float], crop: list[int]) -> float:
    bx, by, bw, bh = box
    cx, cy, cw, ch = crop
    l, t = max(bx, cx), max(by, cy)
    r, b = min(bx + bw, cx + cw), min(by + bh, cy + ch)
    if r <= l or b <= t:
        return 0.0
    return (r - l) * (b - t) / (bw * bh)

def corners(x: float, y: float, yaw_deg: float) -> list[tuple[float, float]]:
    yaw = math.radians(yaw_deg)
    c, s = math.cos(yaw), math.sin(yaw)
    return [(x + dx*c - dy*s, y + dx*s + dy*c)
            for dx, dy in ((-W/2,-H/2),(W/2,-H/2),(W/2,H/2),(-W/2,H/2))]

def interval(poly: list[tuple[float,float]], axis: tuple[float,float]) -> tuple[float,float]:
    vals = [x*axis[0] + y*axis[1] for x,y in poly]
    return min(vals), max(vals)

def overlap(a: list[tuple[float,float]], b: list[tuple[float,float]]) -> bool:
    for poly in (a, b):
        for i in range(4):
            x1,y1 = poly[i]
            x2,y2 = poly[(i+1) % 4]
            axis = (-(y2-y1), x2-x1)
            n = math.hypot(*axis)
            axis = (axis[0]/n, axis[1]/n)
            amin,amax = interval(a,axis)
            bmin,bmax = interval(b,axis)
            if amax <= bmin or bmax <= amin:
                return False
    return True

def region_dhash768(path: Path) -> int:
    """Hash only detector-content regions so fixed black padding cannot dominate."""
    specs = (
        ((7, 0, 306, 72), (33, 8)),
        ((7, 74, 306, 72), (33, 8)),
        ((74, 148, 172, 172), (17, 16)),
    )
    value = 0
    with Image.open(path) as im:
        gray = im.convert("L")
        for (x, y, w, h), (rw, rh) in specs:
            arr = np.asarray(
                gray.crop((x, y, x + w, y + h)).resize((rw, rh), Image.Resampling.BILINEAR),
                dtype=np.int16,
            )
            bits = (arr[:, 1:] > arr[:, :-1]).reshape(-1)
            for bit in bits:
                value = (value << 1) | int(bit)
    return value

def near_duplicate_pairs(hashes: list[int], max_distance: int, bit_count: int = 768) -> int:
    bit_count = int(bit_count)
    chunks = max_distance + 1
    widths = [bit_count // chunks] * chunks
    for i in range(bit_count % chunks):
        widths[i] += 1
    shifts = []
    remaining = bit_count
    for width in widths:
        remaining -= width
        shifts.append((remaining, (1 << width) - 1))
    buckets: defaultdict[tuple[int,int], list[int]] = defaultdict(list)
    count = 0
    for i, value in enumerate(hashes):
        candidates: set[int] = set()
        for ci, (shift, mask) in enumerate(shifts):
            candidates.update(buckets[(ci, (value >> shift) & mask)])
        count += sum((value ^ hashes[j]).bit_count() <= max_distance for j in candidates)
        for ci, (shift, mask) in enumerate(shifts):
            buckets[(ci, (value >> shift) & mask)].append(i)
    return count

def percentile(values: list[float], p: float) -> float | None:
    if not values:
        return None
    vals = sorted(values)
    return vals[min(len(vals)-1, round((len(vals)-1)*p))]

def validate_padding(image_path: Path, outside_mask: np.ndarray) -> bool:
    with Image.open(image_path) as im:
        arr = np.asarray(im.convert("RGB"))
    return bool(np.all(arr[outside_mask] == 0))

def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    records = [json.loads(p.read_text()) for p in sorted((root / "records").glob("synthetic_*.json"))]
    coco = json.loads((root / "annotations/instances_all.json").read_text())
    errors: list[str] = []
    if len(records) < args.minimum_images:
        errors.append(f"image count {len(records)} < minimum {args.minimum_images}")
    if len(coco["images"]) != len(records):
        errors.append("COCO image count != record count")
    record_annotations = sum(len(r["annotations"]) for r in records)
    if len(coco["annotations"]) != record_annotations:
        errors.append("COCO annotation count != record annotation count")

    outside = np.ones((320, 320), dtype=bool)
    for x,y,w,h in REGIONS.values():
        outside[y:y+h, x:x+w] = False
    invalid_boxes = 0
    oob_boxes = 0
    padding_failures = 0
    foreign_crop_overlaps = 0
    min_retained = 1.0
    physical_overlap_pairs = 0
    identities: Counter[str] = Counter()
    meld_counts: Counter[int] = Counter()
    meld_types: Counter[str] = Counter()
    meld_type_images: Counter[str] = Counter()
    white_meld_images = 0
    white_meld_verified = 0
    placement_regimes: Counter[str] = Counter()

    lighting_temperature: Counter[str] = Counter()
    lighting_shadow: Counter[str] = Counter()
    lighting_brightness: Counter[str] = Counter()
    lenses: list[float] = []
    rolls: list[float] = []
    camera_x: list[float] = []
    camera_y: list[float] = []
    camera_z: list[float] = []
    camera_tilt: list[float] = []
    yaws: list[float] = []
    image_hashes: list[str] = []
    perceptual_hashes: list[int] = []
    crop_geometries: Counter[tuple] = Counter()

    for r in records:
        image_path = root / r["image"]["file_name"]
        if not image_path.is_file():
            errors.append(f"missing image {image_path}")
            continue
        if not validate_padding(image_path, outside):
            padding_failures += 1
        digest = sha256(image_path)
        image_hashes.append(digest)
        perceptual_hashes.append(region_dhash768(image_path))

        meld_count = int(r["image"]["meld_group_count"])
        meld_counts[meld_count] += 1
        placement_regimes[r["scene"]["placement_regime"]] += 1
        if r["image"].get("white_dragon_meld"):
            white_meld_images += 1
            if any(a["region"] == "melds" and a["tile_identity"] == "white" for a in r["annotations"]):
                white_meld_verified += 1

        kinds_in_image = set()
        for meld in r["scene"]["hand"]["melds"]:
            kind = str(meld["kind"])
            meld_types[kind] += 1
            kinds_in_image.add(kind)
        for kind in kinds_in_image:
            meld_type_images[kind] += 1

        lighting = r["scene"]["lighting"]
        lighting_temperature[str(lighting["temperature"])] += 1
        lighting_shadow[str(lighting["shadow_style"])] += 1
        lighting_brightness[str(lighting["brightness"])] += 1
        camera = r["scene"]["camera"]
        lenses.append(float(camera["lens_mm"]))
        rolls.append(float(camera["roll_deg"]))
        camera_x.append(float(camera["location"][0]))
        camera_y.append(float(camera["location"][1]))
        camera_z.append(float(camera["location"][2]))
        if "tilt_deg" in camera:
            camera_tilt.append(float(camera["tilt_deg"]))

        crops = r["scene"]["crops"]
        crop_key = tuple(
            (region, tuple(round(float(v), 4) for v in crops[region]))
            for region in ("completed_hand", "dora_indicators", "melds")
        )
        crop_geometries[crop_key] += 1
        tiles = r["scene"]["tiles"]
        polys = []
        for tile in tiles:
            identities[str(tile["tile_identity"])] += 1
            yaws.append(float(tile["transform"]["yaw_deg"]))
            polys.append(corners(float(tile["transform"]["x"]), float(tile["transform"]["y"]),
                                 float(tile["transform"]["yaw_deg"])))
            full_box = [float(v) for v in tile["bbox_full"]]
            own_crop = crops[tile["region"]]
            retained = clip_ratio(full_box, own_crop)
            min_retained = min(min_retained, retained)
            for region, crop in crops.items():
                if region != tile["region"] and clip_ratio(full_box, crop) > 0.0:
                    foreign_crop_overlaps += 1

        for i in range(len(polys)):
            for j in range(i + 1, len(polys)):
                if overlap(polys[i], polys[j]):
                    physical_overlap_pairs += 1

        for a in r["annotations"]:
            x,y,w,h = map(float, a["bbox"])
            if not all(math.isfinite(v) for v in (x,y,w,h)) or w <= 0 or h <= 0:
                invalid_boxes += 1
            if x < -1e-6 or y < -1e-6 or x+w > 320.0001 or y+h > 320.0001:
                oob_boxes += 1

    duplicate_exact_pairs = len(image_hashes) - len(set(image_hashes))
    near_pairs = near_duplicate_pairs(perceptual_hashes, args.near_duplicate_distance)
    missing_identities = sorted(set(ALL_TILES) - set(identities))

    required_meld_types = {"chi", "pon", "open-kan", "closed-kan"}
    if invalid_boxes:
        errors.append(f"invalid boxes: {invalid_boxes}")
    if oob_boxes:
        errors.append(f"OOB boxes: {oob_boxes}")
    if padding_failures:
        errors.append(f"non-black padding images: {padding_failures}")
    if foreign_crop_overlaps:
        errors.append(f"foreign crop overlaps: {foreign_crop_overlaps}")
    if physical_overlap_pairs:
        errors.append(f"physical overlap pairs: {physical_overlap_pairs}")
    if duplicate_exact_pairs:
        errors.append(f"exact duplicate images: {duplicate_exact_pairs}")
    if missing_identities:
        errors.append(f"missing tile identities: {missing_identities}")
    if set(meld_counts) != {0,1,2,3,4}:
        errors.append(f"missing meld-count bins: {sorted({0,1,2,3,4} - set(meld_counts))}")
    if not required_meld_types.issubset(meld_types):
        errors.append(f"missing meld types: {sorted(required_meld_types - set(meld_types))}")
    if white_meld_images == 0 or white_meld_verified != white_meld_images:
        errors.append("white-dragon meld coverage/verification failed")
    if min_retained < 0.985 - 1e-6:
        errors.append(f"minimum retained ratio too small: {min_retained:.6f}")
    if len(crop_geometries) != 1:
        errors.append(f"capture guide geometry varied across records: {len(crop_geometries)} unique geometries")

    report: dict[str, Any] = {
        "schema": "mjtensu.blender-synthetic-nanodet-validation/v1",
        "status": "PASS" if not errors else "FAIL",
        "root": str(root),
        "counts": {
            "accepted_images": len(records),
            "coco_images": len(coco["images"]),
            "coco_annotations": len(coco["annotations"]),
            "record_annotations": record_annotations,
        },
        "bbox": {
            "invalid": invalid_boxes,
            "out_of_bounds": oob_boxes,
            "minimum_retained_area_ratio": min_retained,
            "foreign_crop_overlaps": foreign_crop_overlaps,
        },
        "composite": {"non_black_padding_images": padding_failures},
        "capture_guides": {
            "unique_source_geometries": len(crop_geometries),
            "geometry_counts": [
                {"regions": {region: list(rect) for region, rect in geometry}, "count": count}
                for geometry, count in crop_geometries.items()
            ],
        },
        "physical": {"interpenetrating_tile_pairs": physical_overlap_pairs},
        "meld_group_distribution": {str(k): meld_counts[k] for k in range(5)},
        "meld_presence": {
            "empty": meld_counts[0],
            "present": len(records) - meld_counts[0],
        },
        "white_dragon_meld": {
            "images": white_meld_images,
            "verified_images": white_meld_verified,
        },
        "meld_type_groups": dict(sorted(meld_types.items())),
        "meld_type_images": dict(sorted(meld_type_images.items())),
        "lighting": {
            "temperature": dict(sorted(lighting_temperature.items())),
            "shadow_style": dict(sorted(lighting_shadow.items())),
            "brightness": dict(sorted(lighting_brightness.items())),
        },
        "placement_regimes": dict(sorted(placement_regimes.items())),
        "camera": {
            "lens_mm": {"min": min(lenses), "median": percentile(lenses,0.5), "max": max(lenses)},
            "roll_deg": {"min": min(rolls), "median": percentile(rolls,0.5), "max": max(rolls)},
            "x": {"min": min(camera_x), "max": max(camera_x)},
            "y": {"min": min(camera_y), "max": max(camera_y)},
            "z": {"min": min(camera_z), "max": max(camera_z)},
            "tilt_deg": (
                {"min": min(camera_tilt), "median": percentile(camera_tilt,0.5), "max": max(camera_tilt)}
                if camera_tilt else None
            ),
        },
        "pose": {
            "yaw_deg": {"min": min(yaws), "p10": percentile(yaws,0.1),
                        "median": percentile(yaws,0.5), "p90": percentile(yaws,0.9), "max": max(yaws)},
        },
        "tile_identity_counts": {tile: identities[tile] for tile in ALL_TILES},
        "missing_tile_identities": missing_identities,
        "duplicates": {
            "exact_duplicate_images": duplicate_exact_pairs,
            "region_dhash768_distance_threshold": args.near_duplicate_distance,
            "near_duplicate_pairs": near_pairs,
        },
        "errors": errors,
    }
    qa_dir = root / "qa"
    qa_dir.mkdir(parents=True, exist_ok=True)
    path = qa_dir / "validation_report.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if errors:
        raise SystemExit(1)

if __name__ == "__main__":
    main()
