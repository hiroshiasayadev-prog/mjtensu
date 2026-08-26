from __future__ import annotations

import argparse
import py_compile
from pathlib import Path


PARALLEL_MODULE = '''from __future__ import annotations

import copy
import multiprocessing as mp
import os
import time
from typing import Any

import numpy as np
from pycocotools.cocoeval import COCOeval


_WORKER_EVALUATOR: COCOeval | None = None


def _initialize_worker(evaluator: COCOeval) -> None:
    global _WORKER_EVALUATOR
    _WORKER_EVALUATOR = evaluator


def _evaluate_image_task(task: tuple[Any, Any, Any, int]):
    if _WORKER_EVALUATOR is None:
        raise RuntimeError("parallel COCO evaluator worker was not initialized")
    img_id, cat_id, area_rng, max_det = task
    return _WORKER_EVALUATOR.evaluateImg(img_id, cat_id, area_rng, max_det)


class ParallelCOCOeval(COCOeval):
    """COCOeval variant that parallelizes independent evaluateImg calls.

    The parent process performs preparation and IoU computation exactly as the
    upstream implementation does. Only the category/area/image evaluateImg
    calls are distributed. ``Pool.map`` preserves the task order expected by
    ``COCOeval.accumulate``.
    """

    def __init__(
        self,
        *args,
        workers: int | None = None,
        chunksize: int | None = None,
        start_method: str = "spawn",
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        configured_workers = int(
            os.environ.get("NANODET_COCO_EVAL_WORKERS", workers or 10)
        )
        self.parallel_workers = max(1, configured_workers)
        configured_chunksize = int(
            os.environ.get("NANODET_COCO_EVAL_CHUNKSIZE", chunksize or 0)
        )
        self.parallel_chunksize = max(0, configured_chunksize)
        self.parallel_start_method = os.environ.get(
            "NANODET_COCO_EVAL_START_METHOD", start_method
        )

    def evaluate(self):
        tic = time.time()
        print("Running per image evaluation...")
        p = self.params
        if p.useSegm is not None:
            p.iouType = "segm" if p.useSegm == 1 else "bbox"
            print(
                "useSegm (deprecated) is not None. Running {} evaluation".format(
                    p.iouType
                )
            )
        print("Evaluate annotation type *{}*".format(p.iouType))
        p.imgIds = list(np.unique(p.imgIds))
        if p.useCats:
            p.catIds = list(np.unique(p.catIds))
        p.maxDets = sorted(p.maxDets)
        self.params = p

        self._prepare()
        cat_ids = p.catIds if p.useCats else [-1]

        if p.iouType in ("segm", "bbox"):
            compute_iou = self.computeIoU
        elif p.iouType == "keypoints":
            compute_iou = self.computeOks
        else:
            raise ValueError("unsupported iouType: {}".format(p.iouType))

        self.ious = {
            (img_id, cat_id): compute_iou(img_id, cat_id)
            for img_id in p.imgIds
            for cat_id in cat_ids
        }

        max_det = p.maxDets[-1]
        tasks = [
            (img_id, cat_id, area_rng, max_det)
            for cat_id in cat_ids
            for area_rng in p.areaRng
            for img_id in p.imgIds
        ]

        worker_count = min(self.parallel_workers, len(tasks))
        if worker_count <= 1:
            self.evalImgs = [
                self.evaluateImg(img_id, cat_id, area_rng, max_det)
                for img_id, cat_id, area_rng, max_det in tasks
            ]
        else:
            chunksize = self.parallel_chunksize
            if chunksize <= 0:
                chunksize = max(1, len(tasks) // (worker_count * 8))
            print(
                "Parallel evaluateImg: workers={}, tasks={}, chunksize={}, "
                "start_method={}".format(
                    worker_count,
                    len(tasks),
                    chunksize,
                    self.parallel_start_method,
                )
            )
            context = mp.get_context(self.parallel_start_method)
            with context.Pool(
                processes=worker_count,
                initializer=_initialize_worker,
                initargs=(self,),
            ) as pool:
                self.evalImgs = pool.map(
                    _evaluate_image_task,
                    tasks,
                    chunksize=chunksize,
                )

        self._paramsEval = copy.deepcopy(self.params)
        toc = time.time()
        print("DONE (t={:0.2f}s).".format(toc - tic))
'''


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Add spawn-based parallel COCO evaluateImg execution to the "
            "pinned NanoDet v1.0.0 evaluator."
        )
    )
    parser.add_argument(
        "nanodet_root",
        type=Path,
        help="NanoDet repository root containing nanodet/evaluator.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.nanodet_root.resolve()
    evaluator_dir = root / "nanodet" / "evaluator"
    evaluator_path = evaluator_dir / "coco_detection.py"
    parallel_path = evaluator_dir / "parallel_coco_eval.py"

    if not evaluator_path.is_file():
        raise FileNotFoundError(evaluator_path)

    if parallel_path.exists():
        existing = parallel_path.read_text(encoding="utf-8")
        if existing != PARALLEL_MODULE:
            raise RuntimeError(
                f"Refusing to overwrite unexpected existing file: {parallel_path}"
            )
        module_result = "already_present"
    else:
        parallel_path.write_text(PARALLEL_MODULE, encoding="utf-8", newline="\n")
        module_result = "created"

    text = evaluator_path.read_text(encoding="utf-8")
    old_import = "from pycocotools.cocoeval import COCOeval\n"
    new_import = "from .parallel_coco_eval import ParallelCOCOeval\n"
    old_constructor = "        coco_eval = COCOeval(\n"
    new_constructor = "        coco_eval = ParallelCOCOeval(\n"

    if new_import in text and new_constructor in text:
        evaluator_result = "already_patched"
    elif text.count(old_import) == 1 and text.count(old_constructor) == 1:
        text = text.replace(old_import, new_import, 1)
        text = text.replace(old_constructor, new_constructor, 1)
        evaluator_path.write_text(text, encoding="utf-8", newline="\n")
        evaluator_result = "patched"
    else:
        raise RuntimeError(
            "Unexpected NanoDet COCO evaluator source layout; refusing to patch."
        )

    py_compile.compile(str(parallel_path), doraise=True)
    py_compile.compile(str(evaluator_path), doraise=True)

    print("NanoDet parallel COCO evaluation patch complete:")
    print(f"  parallel module: {module_result}")
    print(f"  evaluator source: {evaluator_result}")
    print("  default workers: 10")
    print("  multiprocessing start method: spawn")
    print("  task order: preserved by Pool.map")


if __name__ == "__main__":
    main()
