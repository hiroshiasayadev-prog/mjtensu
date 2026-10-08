from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

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
TILE_W, TILE_H = 0.027, 0.038
INNER_SIZE = 0.785


def tile_capacity(tile: str) -> int:
    if tile in {"5m", "5p", "5s"}:
        return 3
    if tile in {"red5m", "red5p", "red5s"}:
        return 1
    return 4


def tile_rank(tile: str) -> int | None:
    if tile.startswith("red5"):
        return 5
    if len(tile) == 2 and tile[0].isdigit() and tile[1] in "mps":
        return int(tile[0])
    return None



def is_standard_winning_shape(concealed: list[str], meld_count: int) -> bool:
    """Independent normal-hand check: 4 total groups + one pair, treating red5 as 5."""
    if len(concealed) != 14 - 3 * meld_count:
        return False
    normalized = [t[3:] if t.startswith("red5") else t for t in concealed]
    tally = Counter(normalized)

    def groups_left(groups: int) -> bool:
        if groups == 0:
            return not any(tally.values())
        first = next((t for t in sorted(tally) if tally[t] > 0), None)
        if first is None:
            return False
        if tally[first] >= 3:
            tally[first] -= 3
            if groups_left(groups - 1):
                tally[first] += 3
                return True
            tally[first] += 3
        if len(first) == 2 and first[0].isdigit() and first[1] in "mps" and int(first[0]) <= 7:
            follow = [f"{int(first[0]) + d}{first[1]}" for d in (1, 2)]
            if all(tally[x] > 0 for x in follow):
                for x in (first, *follow):
                    tally[x] -= 1
                if groups_left(groups - 1):
                    for x in (first, *follow):
                        tally[x] += 1
                    return True
                for x in (first, *follow):
                    tally[x] += 1
        return False

    for tile in list(tally):
        if tally[tile] >= 2:
            tally[tile] -= 2
            if groups_left(4 - meld_count):
                return True
            tally[tile] += 2
    return False


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--contact-count", type=int, default=20)
    ap.add_argument("--near-hamming", type=int, default=6)
    ap.add_argument("--min-images", type=int, default=0)
    return ap.parse_args()
def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    xs = sorted(values)
    pos = (len(xs) - 1) * q
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return xs[lo]
    return xs[lo] * (hi - pos) + xs[hi] * (pos - lo)


def summary(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"min": None, "p10": None, "median": None, "p90": None, "max": None}
    return {
        "min": round(min(values), 4),
        "p10": round(percentile(values, 0.10) or 0.0, 4),
        "median": round(statistics.median(values), 4),
        "p90": round(percentile(values, 0.90) or 0.0, 4),
        "max": round(max(values), 4),
    }


def ahash256(im: Image.Image) -> int:
    a = np.asarray(im.convert("L").resize((16, 16), Image.Resampling.BILINEAR), dtype=np.uint8)
    threshold = int(np.median(a))
    value = 0
    for bit in (a >= threshold).reshape(-1):
        value = (value << 1) | int(bit)
    return value


class BKNode:
    def __init__(self, value: int, index: int):
        self.value = value
        self.index = index
        self.children: dict[int, "BKNode"] = {}


def hamming(a: int, b: int) -> int:
    return (a ^ b).bit_count()


def tile_corners(x: float, y: float, yaw_deg: float) -> list[tuple[float, float]]:
    yaw = math.radians(yaw_deg)
    c, s = math.cos(yaw), math.sin(yaw)
    out = []
    for dx, dy in (
        (-TILE_W / 2, -TILE_H / 2),
        (TILE_W / 2, -TILE_H / 2),
        (TILE_W / 2, TILE_H / 2),
        (-TILE_W / 2, TILE_H / 2),
    ):
        out.append((x + dx * c - dy * s, y + dx * s + dy * c))
    return out


def polygons_overlap(a: list[tuple[float, float]], b: list[tuple[float, float]]) -> bool:
    for poly in (a, b):
        for i in range(4):
            x1, y1 = poly[i]
            x2, y2 = poly[(i + 1) % 4]
            ax, ay = -(y2 - y1), x2 - x1
            n = math.hypot(ax, ay)
            ax, ay = ax / n, ay / n
            pa = [x * ax + y * ay for x, y in a]
            pb = [x * ax + y * ay for x, y in b]
            if max(pa) <= min(pb) + 1e-7 or max(pb) <= min(pa) + 1e-7:
                return False
    return True


def valid_chi(tiles: list[str]) -> bool:
    if len(tiles) != 3:
        return False
    suits = [t[-1] for t in tiles]
    ranks = [tile_rank(t) for t in tiles]
    return len(set(suits)) == 1 and suits[0] in "mps" and None not in ranks and sorted(ranks) == list(range(min(ranks), min(ranks) + 3))


def bk_insert(root: BKNode, value: int, index: int) -> None:
    node = root
    while True:
        d = hamming(value, node.value)
        child = node.children.get(d)
        if child is None:
            node.children[d] = BKNode(value, index)
            return
        node = child


def bk_find(root: BKNode, value: int, radius: int, limit: int = 3) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    stack = [root]
    while stack and len(out) < limit:
        node = stack.pop()
        d = hamming(value, node.value)
        if d <= radius:
            out.append((node.index, d))
        lo, hi = d - radius, d + radius
        for edge, child in node.children.items():
            if lo <= edge <= hi:
                stack.append(child)
    return out


def choose_even(records: list[dict[str, Any]], count: int) -> list[dict[str, Any]]:
    if len(records) <= count:
        return records
    if count <= 1:
        return [records[0]]
    positions = [round(i * (len(records) - 1) / (count - 1)) for i in range(count)]
    return [records[i] for i in positions]


def draw_contact(root: Path, qa_dir: Path, name: str, records: list[dict[str, Any]], count: int) -> str | None:
    chosen = choose_even(records, count)
    if not chosen:
        return None
    thumb = 256
    cols = 5
    rows = math.ceil(len(chosen) / cols)
    sheet = Image.new("RGB", (cols * thumb, rows * thumb), (24, 24, 24))
    for slot, record in enumerate(chosen):
        image_path = root / record["image"]["file_name"]
        im = Image.open(image_path).convert("RGB")
        draw = ImageDraw.Draw(im)
        for ann in record["annotations"]:
            x, y, w, h = ann["bbox"]
            draw.rectangle((x, y, x + w, y + h), outline=(255, 0, 0), width=1)
        draw.rectangle((0, 0, 94, 15), fill=(0, 0, 0))
        draw.text((2, 2), f"idx={record['index']}", fill=(255, 255, 0))
        im.thumbnail((thumb, thumb))
        sheet.paste(im, ((slot % cols) * thumb, (slot // cols) * thumb))
    path = qa_dir / f"contact_{name}.png"
    sheet.save(path, optimize=True)
    return str(path.relative_to(root))


def write_split_coco(root: Path, coco: dict[str, Any]) -> dict[str, int]:
    val_ids = {im["id"] for im in coco["images"] if im["id"] % 20 == 0}
    counts: dict[str, int] = {}
    for split, keep_val in (("train", False), ("val", True)):
        images = [im for im in coco["images"] if (im["id"] in val_ids) == keep_val]
        ids = {im["id"] for im in images}
        annotations = [a for a in coco["annotations"] if a["image_id"] in ids]
        payload = {**coco, "images": images, "annotations": annotations}
        path = root / "annotations" / f"instances_{split}.json"
        path.write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n", encoding="utf-8")
        counts[f"{split}_images"] = len(images)
        counts[f"{split}_annotations"] = len(annotations)
    return counts
def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    qa_dir = root / "qa"
    qa_dir.mkdir(parents=True, exist_ok=True)

    record_paths = sorted((root / "records").glob("synthetic_*.json"))
    records = [json.loads(p.read_text(encoding="utf-8")) for p in record_paths]
    coco_path = root / "annotations" / "instances_all.json"
    coco = json.loads(coco_path.read_text(encoding="utf-8"))
    image_files = sorted((root / "images").glob("synthetic_*.png"))

    problems: list[str] = []
    if len(records) != len(image_files):
        problems.append(f"record/image count mismatch: {len(records)} != {len(image_files)}")
    if len(coco["images"]) != len(records):
        problems.append(f"COCO image count mismatch: {len(coco['images'])} != {len(records)}")
    if args.min_images and len(records) < args.min_images:
        problems.append(f"accepted image count below minimum: {len(records)} < {args.min_images}")

    ann_total = sum(len(r["annotations"]) for r in records)
    if len(coco["annotations"]) != ann_total:
        problems.append(f"COCO annotation count mismatch: {len(coco['annotations'])} != {ann_total}")

    image_ids = [im["id"] for im in coco["images"]]
    ann_ids = [ann["id"] for ann in coco["annotations"]]
    if len(set(image_ids)) != len(image_ids):
        problems.append("duplicate COCO image ids")
    if len(set(ann_ids)) != len(ann_ids):
        problems.append("duplicate COCO annotation ids")
    bbox_invalid = 0
    bbox_region_oob = 0
    bbox_sizes: dict[str, list[tuple[float, float]]] = defaultdict(list)
    identities = Counter()
    faces = Counter()
    meld_kinds = Counter()
    meld_groups = Counter()
    regimes = Counter()
    lighting = Counter()
    temperature = Counter()
    camera_lens: list[float] = []
    camera_roll: list[float] = []
    camera_x: list[float] = []
    camera_y: list[float] = []
    camera_z: list[float] = []
    tile_yaws: list[float] = []
    called_sideways = 0
    white_meld_actual = 0
    legal_inventory_violations = 0
    meld_structure_violations = 0
    force_white_failures = 0
    physical_overlap_pairs = 0
    outside_table_tiles = 0
    record_annotation_mismatches = 0

    for record in records:
        image = record["image"]
        meld_groups[image["meld_group_count"]] += 1
        regimes[image["placement_regime"]] += 1
        scene = record["scene"]
        lt = scene["lighting"]
        lighting[(lt["temperature"], lt["shadow_style"], lt["brightness"])] += 1
        temperature[lt["temperature"]] += 1
        cam = scene["camera"]
        camera_lens.append(cam["lens_mm"])
        camera_roll.append(cam["roll_deg"])
        camera_x.append(cam["location"][0])
        camera_y.append(cam["location"][1])
        camera_z.append(cam["location"][2])
        actual_white = False
        visible_inventory = Counter(scene["hand"]["concealed"] + scene["hand"]["dora"])
        for meld in scene["hand"]["melds"]:
            kind = meld["kind"]
            meld_kinds[kind] += 1
            slots = meld["tiles"]
            tile_ids = [slot["tile"] for slot in slots]
            visible_inventory.update(tile_ids)
            rotations = [int(slot.get("rotation", 0)) for slot in slots]
            faces_here = [slot.get("face", "front") for slot in slots]
            called_sideways += sum(r == 90 for r in rotations)
            actual_white = actual_white or "white" in tile_ids

            valid = True
            if kind == "chi":
                valid = valid_chi(tile_ids) and len(slots) == 3
            elif kind == "pon":
                valid = len(slots) == 3 and len(set(tile_ids)) == 1
            elif kind in {"open-kan", "closed-kan"}:
                valid = len(slots) == 4 and len(set(tile_ids)) == 1
            else:
                valid = False
            if kind == "closed-kan":
                valid = valid and rotations.count(90) == 0 and faces_here == ["back", "front", "front", "back"]
            else:
                valid = valid and rotations.count(90) == 1 and all(face == "front" for face in faces_here)
            if not valid:
                meld_structure_violations += 1

        white_meld_actual += int(actual_white)
        if image.get("white_dragon_meld") and not actual_white:
            force_white_failures += 1
        recorded_inventory = Counter(scene["hand"]["inventory"])
        if visible_inventory != recorded_inventory or any(
            count > tile_capacity(tile) for tile, count in visible_inventory.items()
        ):
            legal_inventory_violations += 1

        hand = scene["hand"]
        if hand["completed_hand_empty"]:
            if hand["concealed"] or hand.get("winning_shape_complete", False):
                problems.append(f"image {image['id']}: empty hand marked winning")
        elif not (hand.get("winning_shape_complete") and is_standard_winning_shape(
            hand["concealed"], image["meld_group_count"]
        )):
            problems.append(f"image {image['id']}: nonempty completed hand is not a winning hand")
        contact = scene.get("black_frame_contact")
        if image["meld_group_count"]:
            if not contact or any(abs(contact[key]) > 0.001 for key in ("right_clearance_mm", "bottom_clearance_mm")):
                problems.append(f"image {image['id']}: meld bodies are not touching the black frame")
        elif contact is not None:
            problems.append(f"image {image['id']}: unexpected black-frame contact")
        if len(scene["tiles"]) != len(record["annotations"]):
            record_annotation_mismatches += 1
        footprints = []
        for tile in scene["tiles"]:
            tile_yaws.append(float(tile["transform"]["yaw_deg"]))
            tf = tile["transform"]
            corners = tile_corners(float(tf["x"]), float(tf["y"]), float(tf["yaw_deg"]))
            if tile["region"] != "melds" and any(
                abs(x) > INNER_SIZE / 2 + 1e-7 or abs(y) > INNER_SIZE / 2 + 1e-7
                for x, y in corners
            ):
                outside_table_tiles += 1
            for old in footprints:
                if polygons_overlap(corners, old):
                    physical_overlap_pairs += 1
            footprints.append(corners)

        for ann in record["annotations"]:
            x, y, w, h = map(float, ann["bbox"])
            identities[ann["tile_identity"]] += 1
            faces[ann["face"]] += 1
            bbox_sizes[ann["region"]].append((w, h))
            if w <= 0 or h <= 0 or x < -1e-5 or y < -1e-5 or x + w > 320.0001 or y + h > 320.0001:
                bbox_invalid += 1
            rx, ry, rw, rh = REGIONS[ann["region"]]
            if x < rx - 1e-4 or y < ry - 1e-4 or x + w > rx + rw + 1e-4 or y + h > ry + rh + 1e-4:
                bbox_region_oob += 1

    missing_identities = sorted(set(ALL_TILES) - set(identities))
    if bbox_invalid:
        problems.append(f"invalid/OOB boxes: {bbox_invalid}")
    if bbox_region_oob:
        problems.append(f"boxes outside assigned destination: {bbox_region_oob}")
    if missing_identities:
        problems.append(f"missing tile identities: {missing_identities}")
    if legal_inventory_violations:
        problems.append(f"illegal inventory records: {legal_inventory_violations}")
    if meld_structure_violations:
        problems.append(f"invalid meld structures: {meld_structure_violations}")
    if force_white_failures:
        problems.append(f"forced white-dragon meld failures: {force_white_failures}")
    if physical_overlap_pairs:
        problems.append(f"physical tile overlap pairs: {physical_overlap_pairs}")
    if outside_table_tiles:
        problems.append(f"tiles outside inner table: {outside_table_tiles}")
    if record_annotation_mismatches:
        problems.append(f"scene/annotation count mismatches: {record_annotation_mismatches}")
    profile_names = {r["scene"]["lighting"].get("profile") for r in records}
    diffuse = "diffuse-room-v1" in profile_names
    if diffuse and profile_names != {"diffuse-room-v1"}:
        problems.append(f"mixed lighting provenance profiles: {profile_names}")
    if diffuse and any(r["scene"]["lighting"].get("shadow_style") != "soft"
                       or r["scene"]["lighting"].get("off_frame_blocker") is not None
                       for r in records):
        problems.append("diffuse lighting contains hard shadows or an artificial blocker")
    if args.min_images >= 100:
        if any(meld_groups[i] == 0 for i in range(5)):
            problems.append("one or more meld-count buckets 0..4 are missing")
        missing_meld_kinds = sorted({"chi", "pon", "open-kan", "closed-kan"} - set(meld_kinds))
        if missing_meld_kinds:
            problems.append(f"missing meld kinds: {missing_meld_kinds}")
        if white_meld_actual == 0:
            problems.append("white-dragon meld coverage is missing")
        if called_sideways == 0:
            problems.append("sideways called-tile coverage is missing")
        if faces["back"] == 0:
            problems.append("closed-kan back-face coverage is missing")
        if diffuse:
            expected = {("warm", "soft", "normal"), ("neutral", "soft", "normal"),
                        ("cool", "soft", "normal"), ("cool", "soft", "dim"),
                        ("warm", "soft", "bright")}
            if set(lighting) != expected:
                problems.append(f"diffuse lighting profile coverage mismatch: {set(lighting) ^ expected}")
        elif len(lighting) < 8:
            problems.append(f"legacy lighting profile coverage incomplete: {len(lighting)}/8")
    allowed = np.zeros((320, 320), dtype=bool)
    for x, y, w, h in REGIONS.values():
        allowed[y:y+h, x:x+w] = True
    padding_mask = ~allowed
    padding_failures: list[int] = []
    percept_root: BKNode | None = None
    near_pairs: list[dict[str, int]] = []
    percept_exact = Counter()

    for pos, record in enumerate(records):
        image_path = root / record["image"]["file_name"]
        im = Image.open(image_path).convert("RGB")
        arr = np.asarray(im)
        if np.any(arr[padding_mask] != 0):
            padding_failures.append(record["index"])
        ph = ahash256(im)
        percept_exact[ph] += 1
        if percept_root is None:
            percept_root = BKNode(ph, record["index"])
        else:
            hits = bk_find(percept_root, ph, args.near_hamming, limit=2)
            for other, dist in hits:
                if len(near_pairs) < 50:
                    near_pairs.append({"a": other, "b": record["index"], "hamming": dist})
            bk_insert(percept_root, ph, record["index"])

    if padding_failures:
        problems.append(f"non-black padding images: {len(padding_failures)}")

    exact_sha = Counter()
    manifest_path = root / "manifest.jsonl"
    if manifest_path.is_file():
        for line in manifest_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                exact_sha[json.loads(line)["sha256"]] += 1
    exact_duplicate_groups = sum(1 for n in exact_sha.values() if n > 1)
    percept_duplicate_groups = sum(1 for n in percept_exact.values() if n > 1)
    if exact_duplicate_groups:
        problems.append(f"exact duplicate SHA-256 groups: {exact_duplicate_groups}")
    split_counts = write_split_coco(root, coco)

    categories = {
        "normal": [r for r in records if r["image"]["placement_regime"] == "ordinary"
                   and r["scene"]["lighting"]["shadow_style"] == "soft"],
        "messy": [r for r in records if r["image"]["placement_regime"] == "messy"],
        "empty_meld": [r for r in records if r["image"]["meld_group_count"] == 0],
        "white_dragon": [r for r in records if any(
            slot["tile"] == "white"
            for meld in r["scene"]["hand"]["melds"]
            for slot in meld["tiles"]
        )],
        "shadow_glare": [r for r in records if r["scene"]["lighting"]["shadow_style"] in {"hard", "partial", "glare"}],
        "multi_meld": [r for r in records if r["image"]["meld_group_count"] >= 3],
    }
    contact_sheets = {
        name: draw_contact(root, qa_dir, name, group, args.contact_count)
        for name, group in categories.items()
    }

    bbox_summary = {}
    for region, values in bbox_sizes.items():
        bbox_summary[region] = {
            "count": len(values),
            "width": summary([v[0] for v in values]),
            "height": summary([v[1] for v in values]),
        }

    report = {
        "schema": "mjtensu.blender-synthetic-nanodet-qa/v2",
        "root": str(root),
        "accepted_images": len(records),
        "image_files": len(image_files),
        "coco_images": len(coco["images"]),
        "coco_annotations": len(coco["annotations"]),
        "invalid_or_oob_boxes": bbox_invalid,
        "region_oob_boxes": bbox_region_oob,
        "non_black_padding_images": len(padding_failures),
        "padding_failure_examples": padding_failures[:20],
        "legal_inventory_violations": legal_inventory_violations,
        "meld_structure_violations": meld_structure_violations,
        "forced_white_dragon_failures": force_white_failures,
        "physical_overlap_pairs": physical_overlap_pairs,
        "outside_table_tiles": outside_table_tiles,
        "scene_annotation_count_mismatches": record_annotation_mismatches,
        "meld_group_distribution": {str(k): meld_groups[k] for k in range(5)},
        "empty_meld_images": meld_groups[0],
        "present_meld_images": len(records) - meld_groups[0],
        "white_dragon_meld_images": white_meld_actual,
        "meld_kind_groups": dict(sorted(meld_kinds.items())),
        "lighting_profiles": {"|".join(k): v for k, v in sorted(lighting.items())},
        "temperature": dict(sorted(temperature.items())),
        "placement_regimes": dict(sorted(regimes.items())),
        "camera": {
            "lens_mm": summary(camera_lens),
            "roll_deg": summary(camera_roll),
            "x": summary(camera_x),
            "y": summary(camera_y),
            "z": summary(camera_z),
        },
        "tile_yaw_deg": summary(tile_yaws),
        "called_sideways_tiles": called_sideways,
        "tile_identity_counts": dict(sorted(identities.items())),
        "missing_tile_identities": missing_identities,
        "face_counts": dict(sorted(faces.items())),
        "bbox_summary": bbox_summary,
        "exact_duplicate_sha256_groups": exact_duplicate_groups,
        "exact_perceptual_hash_groups": percept_duplicate_groups,
        "near_identical_pair_examples": near_pairs,
        "split_counts": split_counts,
        "contact_sheets": contact_sheets,
        "problems": problems,
        "status": "PASS" if not problems else "FAIL",
    }
    (qa_dir / "qa_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    lines = [
        "# Blender Synthetic NanoDet QA",
        "",
        f"- status: **{report['status']}**",
        f"- accepted images: {len(records)}",
        f"- COCO annotations: {len(coco['annotations'])}",
        f"- invalid/OOB boxes: {bbox_invalid}",
        f"- destination-region OOB boxes: {bbox_region_oob}",
        f"- non-black padding images: {len(padding_failures)}",
        f"- legal inventory violations: {legal_inventory_violations}",
        f"- meld structure violations: {meld_structure_violations}",
        f"- forced white-dragon failures: {force_white_failures}",
        f"- physical overlap pairs: {physical_overlap_pairs}",
        f"- outside-table tiles: {outside_table_tiles}",
        f"- scene/annotation count mismatches: {record_annotation_mismatches}",
        f"- empty/present meld: {meld_groups[0]} / {len(records)-meld_groups[0]}",
        f"- meld groups 0/1/2/3/4: {[meld_groups[i] for i in range(5)]}",
        f"- white-dragon meld images: {white_meld_actual}",
        f"- meld kinds: {dict(sorted(meld_kinds.items()))}",
        f"- missing tile identities: {missing_identities}",
        f"- exact SHA-256 duplicate groups: {exact_duplicate_groups}",
        f"- exact perceptual-hash groups: {percept_duplicate_groups}",
        f"- near-identical examples (hamming <= {args.near_hamming}): {len(near_pairs)} stored",
        "",
        "## Contact sheets",
    ]
    for name, path in contact_sheets.items():
        lines.append(f"- {name}: {path}")
    if problems:
        lines += ["", "## Problems"] + [f"- {p}" for p in problems]
    (qa_dir / "QA_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
