from __future__ import annotations

import argparse
import importlib
import py_compile
from pathlib import Path


MAX_DETECTIONS = 200


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Patch the active pycocotools COCOeval summary so its overall bbox AP "
            "uses params.maxDets[2] instead of the hard-coded default of 100."
        )
    )
    parser.add_argument(
        "--expected-max-detections",
        type=int,
        default=MAX_DETECTIONS,
        help="Expected third COCOeval maxDets value used by the NanoDet evaluator.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    module = importlib.import_module("pycocotools.cocoeval")
    module_path = Path(module.__file__).resolve()

    old = "            stats[0] = _summarize(1)\n"
    new = (
        "            stats[0] = _summarize(\n"
        "                1, maxDets=self.params.maxDets[2]\n"
        "            )\n"
    )

    text = module_path.read_text(encoding="utf-8")
    if new in text:
        result = "already_applied"
    else:
        count = text.count(old)
        if count != 1:
            raise RuntimeError(
                f"Expected exactly one unpatched COCOeval summary occurrence in "
                f"{module_path}, found {count}. Refusing to modify an unexpected "
                "pycocotools implementation."
            )
        module_path.write_text(
            text.replace(old, new, 1), encoding="utf-8", newline="\n"
        )
        result = "patched"

    py_compile.compile(str(module_path), doraise=True)

    print(
        "pycocotools custom-maxDets summary patch complete:\n"
        f"  module: {module_path}\n"
        f"  expected_max_detections: {args.expected_max_detections}\n"
        f"  summary: {result}"
    )


if __name__ == "__main__":
    main()
