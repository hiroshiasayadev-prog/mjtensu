from __future__ import annotations

import argparse
import hashlib
import json
import tarfile
from pathlib import Path


ONE_CLASS = [{"id": 1, "name": "mahjong_tile", "supercategory": "mahjong_tile"}]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def collapse_coco(source: Path, output: Path) -> dict[str, int]:
    data = json.loads(source.read_text(encoding="utf-8"))
    images = data["images"]
    annotations = data["annotations"]
    for image in images:
        image["file_name"] = Path(image["file_name"]).name
    for ann in annotations:
        ann["category_id"] = 1
    data["categories"] = ONE_CLASS
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(data, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    return {"images": len(images), "annotations": len(annotations)}


def rewrite_composite(source: Path, output: Path) -> dict[str, int]:
    data = json.loads(source.read_text(encoding="utf-8"))
    for image in data["images"]:
        image["file_name"] = Path(image["file_name"]).name
    data["categories"] = ONE_CLASS
    for ann in data["annotations"]:
        ann["category_id"] = 1
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(data, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    return {"images": len(data["images"]), "annotations": len(data["annotations"])}


def add_tree(tf: tarfile.TarFile, source: Path, arc_prefix: str) -> int:
    count = 0
    for path in sorted(source.iterdir()):
        if not path.is_file() or path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp"}:
            continue
        tf.add(path, arcname=f"{arc_prefix}/{path.name}", recursive=False)
        count += 1
    return count


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", type=Path, required=True)
    ap.add_argument("--corrected-root", type=Path, required=True)
    ap.add_argument("--pretrained", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    repo = args.repo.resolve()
    out = args.output.resolve()
    stage = out.parent / (out.stem + ".stage")
    if stage.exists():
        import shutil
        shutil.rmtree(stage)
    (stage / "annotations").mkdir(parents=True)

    corrected = args.corrected_root.resolve()
    jp = repo / "data" / "coco_mahjong_jp_v2"
    comp_ann = repo / ".local" / "recognition" / "nanodet_composite_augmented_dataset" / "annotations"
    comp_img = repo / ".local" / "recognition" / "composite_capture_test_dataset" / "images"

    source_corrected = {
        "train": corrected / "train_annotations.corrected.coco.json",
        "valid": corrected / "valid_annotations.corrected.coco.json",
        "test": corrected / "test_annotations.corrected.coco.json",
    }
    counts: dict[str, dict[str, int]] = {}
    for split, source in source_corrected.items():
        counts[f"jp_{split}"] = collapse_coco(
            source, stage / "annotations" / f"jp_{split}.coco.json"
        )
    counts["composite_train"] = rewrite_composite(
        comp_ann / "instances_composite_train.json",
        stage / "annotations" / "composite_train.coco.json",
    )
    counts["composite_val"] = rewrite_composite(
        comp_ann / "instances_composite_val.json",
        stage / "annotations" / "composite_val.coco.json",
    )

    provenance = {
        "schema": "mjtensu.nanodet-training-baseline-corpus/v1",
        "policy": {
            "base": "human-reviewed corrected coco_mahjong_jp_v2 only",
            "legacy_coco_mahjong": "excluded because semantic audit found many high-confidence disagreements and no reviewed corrected source",
            "target_domain": "existing fixed-layout 320x320 three-region composite train/val",
            "detector_label_space": "all source tile categories collapsed to mahjong_tile without changing bbox geometry",
        },
        "counts": counts,
        "source_sha256": {
            f"jp_{split}_corrected": sha256(path) for split, path in source_corrected.items()
        },
        "pretrained_sha256": sha256(args.pretrained.resolve()),
    }
    export_report = corrected / "export_report.json"
    provenance["correction_export_report_sha256"] = sha256(export_report)
    (stage / "provenance.json").write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        out.unlink()
    with tarfile.open(out, "w", format=tarfile.PAX_FORMAT) as tf:
        for path in sorted((stage / "annotations").iterdir()):
            tf.add(path, arcname=f"annotations/{path.name}", recursive=False)
        tf.add(stage / "provenance.json", arcname="provenance.json", recursive=False)
        tf.add(export_report, arcname="correction_export_report.json", recursive=False)
        tf.add(args.pretrained.resolve(), arcname="pretrained/nanodet-plus-m_320.pth", recursive=False)

        image_counts = {
            "jp_train": add_tree(tf, jp / "train", "images/jp_train"),
            "jp_valid": add_tree(tf, jp / "valid", "images/jp_valid"),
            "jp_test": add_tree(tf, jp / "test", "images/jp_test"),
            "composite": add_tree(tf, comp_img, "images/composite"),
        }

    expected = {
        "jp_train": counts["jp_train"]["images"],
        "jp_valid": counts["jp_valid"]["images"],
        "jp_test": counts["jp_test"]["images"],
    }
    for key, exp in expected.items():
        if image_counts[key] != exp:
            raise RuntimeError(f"{key} image count mismatch: {image_counts[key]} != {exp}")
    composite_expected = counts["composite_train"]["images"] + counts["composite_val"]["images"]
    if image_counts["composite"] < composite_expected:
        raise RuntimeError(
            f"composite image directory too small: {image_counts['composite']} < {composite_expected}"
        )

    print(json.dumps({
        "archive": str(out),
        "bytes": out.stat().st_size,
        "sha256": sha256(out),
        "counts": counts,
        "image_files": image_counts,
    }, indent=2))


if __name__ == "__main__":
    main()
