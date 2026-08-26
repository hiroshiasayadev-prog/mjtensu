from __future__ import annotations

import argparse
import contextlib
import io
import time
from pathlib import Path
from typing import Any

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from nanodet.evaluator.parallel_coco_eval import ParallelCOCOeval


MAX_DETS = [1, 10, 200]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare upstream serial COCO evaluation with NanoDet's parallel "
            "evaluateImg implementation on the same annotations and results."
        )
    )
    parser.add_argument("--annotations", required=True, type=Path)
    parser.add_argument("--results", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="Optional smoke-test limit. Omit for the required full validation.",
    )
    return parser.parse_args()


def _run_evaluation(
    evaluator: COCOeval,
    *,
    image_ids: list[int] | None,
) -> tuple[float, float]:
    evaluator.params.maxDets = list(MAX_DETS)
    if image_ids is not None:
        evaluator.params.imgIds = image_ids

    start = time.perf_counter()
    evaluator.evaluate()
    evaluate_seconds = time.perf_counter() - start

    start = time.perf_counter()
    evaluator.accumulate()
    with contextlib.redirect_stdout(io.StringIO()):
        evaluator.summarize()
    accumulate_seconds = time.perf_counter() - start
    return evaluate_seconds, accumulate_seconds


def _assert_array_equal(name: str, actual: Any, expected: Any) -> None:
    actual_array = np.asarray(actual)
    expected_array = np.asarray(expected)
    if actual_array.shape != expected_array.shape:
        raise AssertionError(
            f"{name}: shape mismatch {actual_array.shape} != {expected_array.shape}"
        )
    if not np.array_equal(actual_array, expected_array, equal_nan=True):
        difference = np.nanmax(np.abs(actual_array - expected_array))
        raise AssertionError(f"{name}: value mismatch; max_abs_diff={difference}")


def _assert_eval_img_equal(
    index: int,
    parallel_item: dict[str, Any] | None,
    serial_item: dict[str, Any] | None,
) -> None:
    if parallel_item is None or serial_item is None:
        if parallel_item is not serial_item:
            raise AssertionError(f"evalImgs[{index}]: None mismatch")
        return

    if parallel_item.keys() != serial_item.keys():
        raise AssertionError(f"evalImgs[{index}]: key mismatch")

    array_keys = {
        "dtIds",
        "gtIds",
        "dtMatches",
        "gtMatches",
        "dtScores",
        "gtIgnore",
        "dtIgnore",
    }
    for key in serial_item:
        if key in array_keys:
            _assert_array_equal(
                f"evalImgs[{index}].{key}",
                parallel_item[key],
                serial_item[key],
            )
        elif parallel_item[key] != serial_item[key]:
            raise AssertionError(
                f"evalImgs[{index}].{key}: "
                f"{parallel_item[key]!r} != {serial_item[key]!r}"
            )


def main() -> None:
    args = parse_args()
    annotations = args.annotations.resolve()
    results = args.results.resolve()
    if not annotations.is_file():
        raise FileNotFoundError(annotations)
    if not results.is_file():
        raise FileNotFoundError(results)
    if args.workers < 1:
        raise ValueError("--workers must be at least 1")

    coco_gt = COCO(str(annotations))
    coco_dt = coco_gt.loadRes(str(results))

    image_ids = sorted(coco_gt.getImgIds())
    if args.max_images is not None:
        image_ids = image_ids[: args.max_images]

    serial = COCOeval(coco_gt, coco_dt, "bbox")
    parallel = ParallelCOCOeval(
        coco_gt,
        coco_dt,
        "bbox",
        workers=args.workers,
    )

    serial_evaluate_seconds, serial_accumulate_seconds = _run_evaluation(
        serial,
        image_ids=image_ids,
    )
    parallel_evaluate_seconds, parallel_accumulate_seconds = _run_evaluation(
        parallel,
        image_ids=image_ids,
    )

    if len(parallel.evalImgs) != len(serial.evalImgs):
        raise AssertionError(
            "evalImgs length mismatch: "
            f"{len(parallel.evalImgs)} != {len(serial.evalImgs)}"
        )
    for index, (parallel_item, serial_item) in enumerate(
        zip(parallel.evalImgs, serial.evalImgs)
    ):
        _assert_eval_img_equal(index, parallel_item, serial_item)

    for key in ("precision", "recall", "scores"):
        _assert_array_equal(
            f"eval.{key}",
            parallel.eval[key],
            serial.eval[key],
        )
    _assert_array_equal("stats", parallel.stats, serial.stats)

    print(f"equivalent_eval_imgs: {len(serial.evalImgs)}")
    print("equivalent_accumulated_arrays: precision, recall, scores")
    print("equivalent_stats: 12")
    print(f"serial_evaluate_seconds:   {serial_evaluate_seconds:.3f}")
    print(f"parallel_evaluate_seconds: {parallel_evaluate_seconds:.3f}")
    print(
        "evaluate_speedup:          "
        f"{serial_evaluate_seconds / parallel_evaluate_seconds:.2f}x"
    )
    print(f"serial_accumulate_seconds:   {serial_accumulate_seconds:.3f}")
    print(f"parallel_accumulate_seconds: {parallel_accumulate_seconds:.3f}")


if __name__ == "__main__":
    main()
