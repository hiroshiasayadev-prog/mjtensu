from __future__ import annotations

import argparse
from pathlib import Path


EXPECTED_COMMIT = "d3fb34fa91d6020f273d6d063bf324dcd97bac12"
MAX_DETECTIONS = 200


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Patch NanoDet v1.0.0 so NanoDetPlus post-processing and COCOeval "
            "retain up to 200 detections per image."
        )
    )
    parser.add_argument(
        "nanodet_root",
        type=Path,
        help="Path to the NanoDet repository root containing nanodet/ and tools/.",
    )
    return parser.parse_args()


def replace_exactly_once(path: Path, old: str, new: str) -> str:
    text = path.read_text(encoding="utf-8")

    if new in text:
        return "already_applied"

    count = text.count(old)
    if count != 1:
        raise RuntimeError(
            f"Expected exactly one unpatched occurrence in {path}, found {count}. "
            "Refusing to modify an unexpected NanoDet source revision."
        )

    path.write_text(text.replace(old, new, 1), encoding="utf-8", newline="\n")
    return "patched"


def main() -> None:
    args = parse_args()
    root = args.nanodet_root.resolve()

    head_path = root / "nanodet" / "model" / "head" / "nanodet_plus_head.py"
    evaluator_path = root / "nanodet" / "evaluator" / "coco_detection.py"

    for path in (head_path, evaluator_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    head_result = replace_exactly_once(
        head_path,
        "                max_num=100,\n",
        f"                max_num={MAX_DETECTIONS},\n",
    )

    evaluator_old = (
        "        coco_eval = COCOeval(\n"
        "            copy.deepcopy(self.coco_api), copy.deepcopy(coco_dets), \"bbox\"\n"
        "        )\n"
        "        coco_eval.evaluate()\n"
    )
    evaluator_new = (
        "        coco_eval = COCOeval(\n"
        "            copy.deepcopy(self.coco_api), copy.deepcopy(coco_dets), \"bbox\"\n"
        "        )\n"
        f"        coco_eval.params.maxDets = [1, 10, {MAX_DETECTIONS}]\n"
        "        coco_eval.evaluate()\n"
    )
    evaluator_result = replace_exactly_once(
        evaluator_path,
        evaluator_old,
        evaluator_new,
    )

    print(
        "NanoDet dense-detection patch complete:\n"
        f"  expected_commit: {EXPECTED_COMMIT}\n"
        f"  max_detections: {MAX_DETECTIONS}\n"
        f"  nanodet_plus_head: {head_result}\n"
        f"  coco_evaluator: {evaluator_result}"
    )


if __name__ == "__main__":
    main()
