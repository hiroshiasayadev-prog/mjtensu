from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import shutil
import sqlite3
import statistics
import subprocess
import sys
import tempfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Sequence

import numpy as np
from PIL import Image, ImageDraw

if __package__:
    from .build_nanodet_capture_finetune_dataset import (
        REGION_KEYS,
        captures_to_coco,
        empty_coco,
        load_json,
    )
    from .build_nanodet_region_rotation_augmented_dataset import annotation_polygon
    from .nanodet.evaluate_composite_onnx import (
        Detection,
        decode_output,
        preprocess_image,
    )
else:  # direct script execution; avoid colliding with the installed NanoDet package.
    _repository_root_for_import = Path(__file__).resolve().parents[2]
    if str(_repository_root_for_import) not in sys.path:
        sys.path.insert(0, str(_repository_root_for_import))
    from tools.recognition.build_nanodet_capture_finetune_dataset import (  # type: ignore[no-redef]
        REGION_KEYS,
        captures_to_coco,
        empty_coco,
        load_json,
    )
    from tools.recognition.build_nanodet_region_rotation_augmented_dataset import (  # type: ignore[no-redef]
        annotation_polygon,
    )
    from tools.recognition.nanodet.evaluate_composite_onnx import (  # type: ignore[no-redef]
        Detection,
        decode_output,
        preprocess_image,
    )


INVESTIGATION_ID = "PRODUCT-INV-RECOGNITION-010"
SEED = 42
OPERATING_THRESHOLD = 0.30
CANDIDATE_THRESHOLD = 0.001
NMS_IOU_THRESHOLD = 0.60
MAX_DETECTIONS = 200
MATCH_IOU_THRESHOLD = 0.50
A1_MAX_ROTATION_DEG = 12.0
A1_MAX_SHRINK_FRACTION = 0.10
A1_REAL_VARIANTS_PER_SOURCE = 4
A1_COMPOSITE_VARIANTS_PER_SOURCE = 1
R1_EXTRA_TRAIN_FRACTION = 0.75


@dataclass(frozen=True)
class Paths:
    repository_root: Path
    experiment_root: Path
    capture_root: Path
    capture_database: Path
    baseline_dataset_root: Path
    composite_dataset_root: Path
    nanodet_root: Path
    nanodet_python: Path
    baseline_finetune_config: Path
    joint_retrain_config: Path
    baseline_checkpoint: Path
    baseline_run: Path
    baseline_onnx: Path
    region_augment_script: Path


@dataclass(frozen=True)
class Condition:
    key: str
    source_mix: str
    augmentation: str
    dataset_path: Path
    run_directory: Path
    config_path: Path
    onnx_path: Path
    training_mode: str = "finetune"


@dataclass(frozen=True)
class BoxMetrics:
    iou: float
    gt_coverage: float
    crop_purity: float
    center_error_x: float
    center_error_y: float
    width_ratio: float
    height_ratio: float


@dataclass(frozen=True)
class MatchRecord:
    image_id: int
    file_name: str
    region: str
    gt_annotation_id: int
    prediction_index: int
    score: float
    prediction_bbox: tuple[float, float, float, float]
    gt_bbox: tuple[float, float, float, float]
    metrics: BoxMetrics


@dataclass(frozen=True)
class PredictionRecord:
    image_id: int
    prediction_index: int
    region: str
    score: float
    bbox: tuple[float, float, float, float]


@dataclass(frozen=True)
class EvalResult:
    report: dict[str, Any]
    matches: tuple[MatchRecord, ...]
    predictions: tuple[PredictionRecord, ...]


def parse_args() -> argparse.Namespace:
    repository_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(
        description=(
            "Run INV-010 end to end on the training host: inventory real annotations, "
            "build R1/A1 datasets, train D1-D3 from the accepted composite checkpoint, "
            "export ONNX, evaluate detector crop quality, and compare all candidates with D0."
        )
    )
    parser.add_argument("--repository-root", type=Path, default=repository_root)
    parser.add_argument("--experiment-root", type=Path)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--operating-threshold", type=float, default=OPERATING_THRESHOLD)
    parser.add_argument("--candidate-threshold", type=float, default=CANDIDATE_THRESHOLD)
    parser.add_argument("--nms-iou-threshold", type=float, default=NMS_IOU_THRESHOLD)
    parser.add_argument("--max-detections", type=int, default=MAX_DETECTIONS)
    parser.add_argument("--match-iou-threshold", type=float, default=MATCH_IOU_THRESHOLD)
    parser.add_argument("--a1-max-rotation-deg", type=float, default=A1_MAX_ROTATION_DEG)
    parser.add_argument(
        "--a1-max-shrink-fraction", type=float, default=A1_MAX_SHRINK_FRACTION
    )
    parser.add_argument(
        "--a1-real-variants-per-source", type=int, default=A1_REAL_VARIANTS_PER_SOURCE
    )
    parser.add_argument(
        "--extra-train-fraction", type=float, default=R1_EXTRA_TRAIN_FRACTION
    )
    parser.add_argument(
        "--conditions",
        nargs="+",
        choices=("D1", "D2", "D3"),
        default=("D1", "D2", "D3"),
        help="Fine-tune candidates to train. D0 is always reused/evaluated.",
    )
    parser.add_argument(
        "--run-d4",
        action="store_true",
        help=(
            "After D1-D3 comparison, run the conditional 40-epoch joint retrain from the "
            "official NanoDet checkpoint using the best candidate dataset selected by crop quality."
        ),
    )
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Inventory and build datasets/configs, but do not train/export/evaluate.",
    )
    parser.add_argument(
        "--skip-preparation",
        action="store_true",
        help="Reuse already prepared datasets/configs under experiment-root.",
    )
    parser.add_argument(
        "--skip-training",
        action="store_true",
        help="Reuse existing D1-D3 run directories and only export/evaluate.",
    )
    parser.add_argument(
        "--overwrite-runs",
        action="store_true",
        help=(
            "Delete and retrain D1-D3 even when a reusable model_best checkpoint already exists. "
            "By default completed conditions are reused so an interrupted overnight run can resume."
        ),
    )
    parser.add_argument(
        "--skip-export",
        action="store_true",
        help="Reuse existing ONNX files under each run directory.",
    )
    parser.add_argument(
        "--overwrite-evaluation",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--worst-case-count", type=int, default=16)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    validate_args(args)
    paths = resolve_paths(args)
    validate_required_paths(paths)
    paths.experiment_root.mkdir(parents=True, exist_ok=True)

    if not args.skip_preparation:
        prepared = prepare_experiment(paths, args)
    else:
        prepared_path = paths.experiment_root / "preparation.json"
        if not prepared_path.is_file():
            raise FileNotFoundError(
                f"--skip-preparation requires an existing preparation manifest: {prepared_path}"
            )
        prepared = load_json(prepared_path)

    conditions = conditions_from_preparation(paths, prepared)
    selected_keys = tuple(str(key) for key in args.conditions)
    print_preparation_summary(prepared)

    if args.prepare_only:
        print(
            json.dumps(
                {
                    "status": "prepared",
                    "experiment_root": str(paths.experiment_root),
                    "conditions": {key: str(conditions[key].dataset_path) for key in selected_keys},
                    "preparation": str(paths.experiment_root / "preparation.json"),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0

    failures: dict[str, dict[str, str]] = {}
    usable_keys: list[str] = []
    if not args.skip_training:
        for key in selected_keys:
            try:
                train_condition(
                    paths,
                    conditions[key],
                    seed=int(args.seed),
                    overwrite=bool(args.overwrite_runs),
                )
                usable_keys.append(key)
            except Exception as error:
                failures[key] = {
                    "stage": "training",
                    "error_type": type(error).__name__,
                    "error": str(error),
                }
                print(
                    f"[condition-failed] {key} training: {type(error).__name__}: {error}",
                    file=sys.stderr,
                    flush=True,
                )
    else:
        usable_keys = list(selected_keys)

    if not args.skip_export:
        ensure_baseline_onnx(paths)
        exported_keys: list[str] = []
        for key in usable_keys:
            try:
                export_condition_onnx(paths, conditions[key])
                exported_keys.append(key)
            except Exception as error:
                failures[key] = {
                    "stage": "export",
                    "error_type": type(error).__name__,
                    "error": str(error),
                }
                print(
                    f"[condition-failed] {key} export: {type(error).__name__}: {error}",
                    file=sys.stderr,
                    flush=True,
                )
        usable_keys = exported_keys
    else:
        if not paths.baseline_onnx.is_file():
            raise FileNotFoundError(paths.baseline_onnx)
        reusable_keys: list[str] = []
        for key in usable_keys:
            if conditions[key].onnx_path.is_file():
                reusable_keys.append(key)
            else:
                failures[key] = {
                    "stage": "export",
                    "error_type": "FileNotFoundError",
                    "error": str(conditions[key].onnx_path),
                }
        usable_keys = reusable_keys

    atomic_write_json(paths.experiment_root / "condition_failures.json", failures)
    evaluation_sets = resolve_evaluation_sets(paths, prepared)
    model_paths: dict[str, Path] = {"D0": paths.baseline_onnx}
    model_paths.update({key: conditions[key].onnx_path for key in usable_keys})

    reports = evaluate_models(
        paths,
        model_paths=model_paths,
        evaluation_sets=evaluation_sets,
        candidate_threshold=float(args.candidate_threshold),
        operating_threshold=float(args.operating_threshold),
        nms_iou_threshold=float(args.nms_iou_threshold),
        max_detections=int(args.max_detections),
        match_iou_threshold=float(args.match_iou_threshold),
        worst_case_count=int(args.worst_case_count),
        overwrite=bool(args.overwrite_evaluation),
    )
    comparison = write_comparison(paths, reports, prepared)

    if args.run_d4:
        if not usable_keys:
            raise RuntimeError("--run-d4 requires at least one successful D1-D3 candidate")
        best_key = select_best_candidate(comparison, usable_keys)
        d4 = build_d4_condition(paths, conditions[best_key], prepared, seed=int(args.seed))
        if not args.skip_training:
            train_condition(
                paths,
                d4,
                seed=int(args.seed),
                overwrite=bool(args.overwrite_runs),
            )
        if not args.skip_export:
            export_condition_onnx(paths, d4)
        elif not d4.onnx_path.is_file():
            raise FileNotFoundError(d4.onnx_path)
        model_paths["D4"] = d4.onnx_path
        reports = evaluate_models(
            paths,
            model_paths={"D4": d4.onnx_path},
            evaluation_sets=evaluation_sets,
            candidate_threshold=float(args.candidate_threshold),
            operating_threshold=float(args.operating_threshold),
            nms_iou_threshold=float(args.nms_iou_threshold),
            max_detections=int(args.max_detections),
            match_iou_threshold=float(args.match_iou_threshold),
            worst_case_count=int(args.worst_case_count),
            overwrite=bool(args.overwrite_evaluation),
            existing_reports=reports,
        )
        comparison = write_comparison(paths, reports, prepared)
        comparison["d4"] = {
            "selected_source_condition": best_key,
            "dataset": str(d4.dataset_path),
            "config": str(d4.config_path),
            "onnx": str(d4.onnx_path),
        }
        atomic_write_json(paths.experiment_root / "comparison.json", comparison)

    print(
        json.dumps(
            {
                "status": "completed",
                "experiment_root": str(paths.experiment_root),
                "comparison": str(paths.experiment_root / "comparison.json"),
                "comparison_csv": str(paths.experiment_root / "comparison.csv"),
                "models": {key: str(path) for key, path in model_paths.items()},
                "condition_failures": failures,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


def validate_args(args: argparse.Namespace) -> None:
    if not 0.0 <= float(args.operating_threshold) <= 1.0:
        raise ValueError("--operating-threshold must be within [0,1]")
    if not 0.0 <= float(args.candidate_threshold) <= float(args.operating_threshold):
        raise ValueError("--candidate-threshold must be within [0, operating-threshold]")
    if not 0.0 <= float(args.nms_iou_threshold) <= 1.0:
        raise ValueError("--nms-iou-threshold must be within [0,1]")
    if not 0.0 <= float(args.match_iou_threshold) <= 1.0:
        raise ValueError("--match-iou-threshold must be within [0,1]")
    if int(args.max_detections) < 1:
        raise ValueError("--max-detections must be positive")
    if not 0.0 <= float(args.a1_max_rotation_deg) <= 45.0:
        raise ValueError("--a1-max-rotation-deg must be within [0,45]")
    if not 0.0 <= float(args.a1_max_shrink_fraction) < 1.0:
        raise ValueError("--a1-max-shrink-fraction must be within [0,1)")
    if int(args.a1_real_variants_per_source) < 1:
        raise ValueError("--a1-real-variants-per-source must be positive")
    if not 0.0 < float(args.extra_train_fraction) < 1.0:
        raise ValueError("--extra-train-fraction must be strictly between 0 and 1")
    if int(args.worst_case_count) < 1:
        raise ValueError("--worst-case-count must be positive")


def resolve_paths(args: argparse.Namespace) -> Paths:
    repository_root = args.repository_root.resolve()
    experiment_root = (
        args.experiment_root.resolve()
        if args.experiment_root is not None
        else repository_root / ".local" / "recognition" / "nanodet_localization_retrain"
    )
    baseline_run = (
        repository_root
        / ".local"
        / "recognition"
        / "nanodet_runs"
        / "E1_plus_m_320_real_capture_ft10_l10_seed42"
    )
    baseline_onnx = (
        baseline_run / "model_best" / "nanodet-plus-m-320-real-capture-ft10-l10.onnx"
    )
    baseline_checkpoint = (
        repository_root
        / ".local"
        / "recognition"
        / "nanodet_runs"
        / "E1_plus_m_320_composite_augmented_amp40_seed42"
        / "model_best"
        / "nanodet_model_best.pth"
    )
    nanodet_root = repository_root / "nanodet" / "nanodet"
    nanodet_python = nanodet_root / ".venv" / "bin" / "python"
    if os.name == "nt":
        windows_python = nanodet_root / ".venv" / "Scripts" / "python.exe"
        if windows_python.is_file():
            nanodet_python = windows_python
    return Paths(
        repository_root=repository_root,
        experiment_root=experiment_root,
        capture_root=repository_root / ".local" / "recognition" / "capture_dataset",
        capture_database=repository_root
        / ".local"
        / "recognition"
        / "capture_dataset"
        / "dataset.sqlite",
        baseline_dataset_root=repository_root
        / ".local"
        / "recognition"
        / "nanodet_capture_finetune_dataset",
        composite_dataset_root=repository_root
        / ".local"
        / "recognition"
        / "nanodet_composite_augmented_dataset",
        nanodet_root=nanodet_root,
        nanodet_python=nanodet_python,
        baseline_finetune_config=repository_root
        / "tools"
        / "recognition"
        / "nanodet"
        / "configs"
        / "e1_nanodet_plus_m_320_real_capture_ft10_l10.yml",
        joint_retrain_config=repository_root
        / "tools"
        / "recognition"
        / "nanodet"
        / "configs"
        / "e1_nanodet_plus_m_320_composite_augmented_amp40.yml",
        baseline_checkpoint=baseline_checkpoint,
        baseline_run=baseline_run,
        baseline_onnx=baseline_onnx,
        region_augment_script=repository_root
        / "tools"
        / "recognition"
        / "build_nanodet_region_rotation_augmented_dataset.py",
    )


def validate_required_paths(paths: Paths) -> None:
    required = (
        paths.capture_database,
        paths.baseline_dataset_root / "annotations" / "instances_train.json",
        paths.baseline_dataset_root / "annotations" / "instances_real_train.json",
        paths.baseline_dataset_root / "annotations" / "instances_real_val.json",
        paths.baseline_dataset_root / "provenance.json",
        paths.composite_dataset_root / "annotations" / "instances_train.json",
        paths.composite_dataset_root / "annotations" / "instances_val.json",
        paths.composite_dataset_root / "annotations" / "instances_composite_train.json",
        paths.composite_dataset_root / "annotations" / "instances_composite_val.json",
        paths.baseline_finetune_config,
        paths.joint_retrain_config,
        paths.baseline_checkpoint,
        paths.region_augment_script,
        paths.nanodet_root / "tools" / "train.py",
        paths.nanodet_root / "tools" / "export_onnx.py",
    )
    for path in required:
        if not path.exists():
            raise FileNotFoundError(path)
    if not paths.nanodet_python.is_file():
        raise FileNotFoundError(
            f"NanoDet virtualenv Python not found. Run on the training host: {paths.nanodet_python}"
        )


def prepare_experiment(paths: Paths, args: argparse.Namespace) -> dict[str, Any]:
    datasets_root = paths.experiment_root / "datasets"
    configs_root = paths.experiment_root / "configs"
    datasets_root.mkdir(parents=True, exist_ok=True)
    configs_root.mkdir(parents=True, exist_ok=True)

    baseline_provenance = load_json(paths.baseline_dataset_root / "provenance.json")
    baseline_campaign = str(baseline_provenance["campaign_id"])
    baseline_train_layout_ids = frozenset(str(value) for value in baseline_provenance["train_layout_ids"])
    baseline_val_layout_ids = frozenset(str(value) for value in baseline_provenance["val_layout_ids"])

    inventory = inventory_complete_capture_layouts(
        paths.capture_database,
        campaign_id=baseline_campaign,
    )
    complete_layout_ids = [
        str(item["layout_id"]) for item in inventory["layouts"] if bool(item["complete"])
    ]
    complete_layout_set = frozenset(complete_layout_ids)
    known = baseline_train_layout_ids | baseline_val_layout_ids
    # D0's historical train/validation partitions are frozen COCO artifacts. Their
    # current capture-DB annotation status may legitimately change later (for example,
    # a layout can be reopened for correction). Do not make INV010 depend on being able
    # to reconstruct those frozen partitions from the mutable DB. R1 starts from the
    # frozen D0 real-train COCO and uses the DB only for newly completed layouts.
    inventory["frozen_layouts_not_complete_now"] = sorted(known - complete_layout_set)
    extras = [layout_id for layout_id in complete_layout_ids if layout_id not in known]
    extra_train, extra_holdout = split_extra_layouts(
        extras,
        train_fraction=float(args.extra_train_fraction),
        seed=int(args.seed),
    )
    r1_train_layout_ids = frozenset((*baseline_train_layout_ids, *extra_train))

    baseline_real_unique = load_coco(
        paths.baseline_dataset_root / "annotations" / "instances_real_train.json"
    )
    completed_captures = load_complete_campaign_captures(
        paths.capture_database,
        campaign_id=baseline_campaign,
        required_layout_ids=frozenset(complete_layout_ids),
    )
    inventory["complete_capture_detail"] = summarize_completed_captures(completed_captures)
    layout = load_json(
        paths.repository_root / "tools" / "recognition" / "capture_layout.v1.json"
    )
    extra_train_real: dict[str, Any] | None = None
    if extra_train:
        extra_train_real = captures_to_coco(
            completed_captures,
            included_layout_ids=frozenset(extra_train),
            repository_root=paths.repository_root,
            storage_root=paths.capture_root,
            layout=layout,
            description=f"{INVESTIGATION_ID} newly completed real training captures",
            check_images=True,
        )
    r1_real_unique = combine_frozen_real_train_with_new_layouts(
        baseline_real_unique,
        extra_train_real,
    )
    r1_real_unique_path = datasets_root / "r1_real_unique.json"
    atomic_write_json(r1_real_unique_path, r1_real_unique)

    final_holdout_path: Path | None = None
    final_holdout_count = {"images": 0, "annotations": 0}
    if extra_holdout:
        holdout_payload = captures_to_coco(
            completed_captures,
            included_layout_ids=frozenset(extra_holdout),
            repository_root=paths.repository_root,
            storage_root=paths.capture_root,
            layout=layout,
            description=f"{INVESTIGATION_ID} layout-disjoint final real holdout",
            check_images=True,
        )
        final_holdout_path = datasets_root / "real_final_holdout.json"
        atomic_write_json(final_holdout_path, holdout_payload)
        final_holdout_count = coco_counts(holdout_payload)

    baseline_train = load_coco(
        paths.baseline_dataset_root / "annotations" / "instances_train.json"
    )
    composite_train = load_coco(
        paths.composite_dataset_root / "annotations" / "instances_composite_train.json"
    )
    composite_merged_train = load_coco(
        paths.composite_dataset_root / "annotations" / "instances_train.json"
    )
    composite_merged_val = load_coco(
        paths.composite_dataset_root / "annotations" / "instances_val.json"
    )
    base_val = filter_payload_by_image(
        composite_merged_val,
        lambda image: image.get("dataset_origin") == "base_val",
        description=f"{INVESTIGATION_ID} frozen base regression validation",
    )
    if not base_val["images"] or not base_val["annotations"]:
        raise ValueError("Could not extract dataset_origin=base_val from composite merged validation")
    base_val_path = datasets_root / "base_val.json"
    atomic_write_json(base_val_path, base_val)

    baseline_base_replay = filter_payload_by_image(
        baseline_train,
        lambda image: image.get("fine_tune_source") == "base_replay",
        description="D0 exact base replay subset",
    )
    base_pool = filter_payload_by_image(
        composite_merged_train,
        lambda image: image.get("dataset_origin") == "base_train",
        description="Base replay candidate pool",
    )

    composite_annotations = len(composite_train["annotations"])
    r1_unique_annotations = len(r1_real_unique["annotations"])
    if r1_unique_annotations < 1:
        raise ValueError("R1 unique real dataset is empty")
    r1_real_repeat = max(1, round(composite_annotations / r1_unique_annotations))
    r1_real_a0 = repeat_each_image(r1_real_unique, r1_real_repeat, source_name="real_capture")
    r1_target_base_annotations = round(
        (len(r1_real_a0["annotations"]) + composite_annotations) / 2
    )
    r1_base_replay = sample_images_near_annotation_target(
        base_pool,
        target_annotations=r1_target_base_annotations,
        seed=int(args.seed),
    )

    a1_root = datasets_root / "a1"
    a1_baseline_real_root = a1_root / "baseline_real"
    a1_r1_real_root = a1_root / "r1_real"
    a1_composite_root = a1_root / "composite"
    run_region_augmentation(
        paths,
        annotations_path=paths.baseline_dataset_root
        / "annotations"
        / "instances_real_train.json",
        output_directory=a1_baseline_real_root,
        copies_per_image=int(args.a1_real_variants_per_source),
        max_rotation_deg=float(args.a1_max_rotation_deg),
        max_shrink_fraction=float(args.a1_max_shrink_fraction),
        seed=int(args.seed),
    )
    run_region_augmentation(
        paths,
        annotations_path=r1_real_unique_path,
        output_directory=a1_r1_real_root,
        copies_per_image=int(args.a1_real_variants_per_source),
        max_rotation_deg=float(args.a1_max_rotation_deg),
        max_shrink_fraction=float(args.a1_max_shrink_fraction),
        seed=int(args.seed),
    )
    run_region_augmentation(
        paths,
        annotations_path=paths.composite_dataset_root
        / "annotations"
        / "instances_composite_train.json",
        output_directory=a1_composite_root,
        copies_per_image=A1_COMPOSITE_VARIANTS_PER_SOURCE,
        max_rotation_deg=float(args.a1_max_rotation_deg),
        max_shrink_fraction=float(args.a1_max_shrink_fraction),
        seed=int(args.seed),
    )

    baseline_real_a1 = load_coco(a1_baseline_real_root / "annotations" / "instances_train.json")
    r1_real_a1 = load_coco(a1_r1_real_root / "annotations" / "instances_train.json")
    composite_a1 = load_coco(a1_composite_root / "annotations" / "instances_train.json")

    baseline_real_repeat = int(baseline_provenance["real_repeat"])
    d2_real = build_source_variant_exposures(
        baseline_real_unique,
        baseline_real_a1,
        exposures_per_source=baseline_real_repeat,
        seed=int(args.seed),
    )
    d3_real = build_source_variant_exposures(
        r1_real_unique,
        r1_real_a1,
        exposures_per_source=r1_real_repeat,
        seed=int(args.seed),
    )

    d1 = merge_named_payloads(
        (
            ("real_capture", r1_real_a0),
            ("composite_replay", composite_train),
            ("base_replay", r1_base_replay),
        ),
        description=f"{INVESTIGATION_ID} D1 R1 balanced A0",
    )
    d2 = merge_named_payloads(
        (
            ("real_capture_a1", d2_real),
            ("composite_replay_a1", composite_a1),
            ("base_replay", baseline_base_replay),
        ),
        description=f"{INVESTIGATION_ID} D2 R0 current source exposure A1 fixed-layout geometry",
    )
    d3 = merge_named_payloads(
        (
            ("real_capture_a1", d3_real),
            ("composite_replay_a1", composite_a1),
            ("base_replay", r1_base_replay),
        ),
        description=f"{INVESTIGATION_ID} D3 R1 balanced A1 fixed-layout geometry",
    )

    dataset_paths = {
        "D1": datasets_root / "D1_R1_A0" / "instances_train.json",
        "D2": datasets_root / "D2_R0_A1" / "instances_train.json",
        "D3": datasets_root / "D3_R1_A1" / "instances_train.json",
    }
    payloads = {"D1": d1, "D2": d2, "D3": d3}
    for key, path in dataset_paths.items():
        atomic_write_json(path, payloads[key])
        atomic_write_json(
            path.parent / "provenance.json",
            {
                "investigation": INVESTIGATION_ID,
                "condition": key,
                "counts": coco_counts(payloads[key]),
                "sources": source_mix_stats(payloads[key]),
                "sha256": sha256_file(path),
            },
        )

    config_paths: dict[str, Path] = {}
    for key in ("D1", "D2", "D3"):
        run_directory = condition_run_directory(paths, key, seed=int(args.seed))
        config_path = configs_root / f"inv010_{key.lower()}_seed{int(args.seed)}.yml"
        write_candidate_config(
            template_path=paths.baseline_finetune_config,
            output_path=config_path,
            save_dir=run_directory,
            train_annotations=dataset_paths[key],
            val_annotations=paths.baseline_dataset_root
            / "annotations"
            / "instances_real_val.json",
            repository_root=paths.repository_root,
            starting_checkpoint=paths.baseline_checkpoint,
            training_mode="finetune",
        )
        config_paths[key] = config_path

    preparation = {
        "status": "prepared",
        "investigation": INVESTIGATION_ID,
        "seed": int(args.seed),
        "inventory": inventory,
        "baseline": {
            "campaign_id": baseline_campaign,
            "train_layout_ids": sorted(baseline_train_layout_ids),
            "val_layout_ids": sorted(baseline_val_layout_ids),
            "real_repeat": baseline_real_repeat,
            "dataset_counts": coco_counts(baseline_train),
            "base_replay_counts": coco_counts(baseline_base_replay),
        },
        "r1": {
            "extra_complete_layout_ids": extras,
            "extra_train_layout_ids": sorted(extra_train),
            "extra_holdout_layout_ids": sorted(extra_holdout),
            "train_layout_ids": sorted(r1_train_layout_ids),
            "unique_real_counts": coco_counts(r1_real_unique),
            "real_repeat": r1_real_repeat,
            "real_exposure_counts": coco_counts(r1_real_a0),
            "base_target_annotations": r1_target_base_annotations,
            "base_replay_counts": coco_counts(r1_base_replay),
            "composite_counts": coco_counts(composite_train),
        },
        "a1": {
            "max_rotation_deg": float(args.a1_max_rotation_deg),
            "max_shrink_fraction": float(args.a1_max_shrink_fraction),
            "real_variants_per_source": int(args.a1_real_variants_per_source),
            "composite_variants_per_source": A1_COMPOSITE_VARIANTS_PER_SOURCE,
            "baseline_real_augmented": str(a1_baseline_real_root),
            "r1_real_augmented": str(a1_r1_real_root),
            "composite_augmented": str(a1_composite_root),
            "nanodet_training_pipeline": "kept identical to D0/A0 so D2 isolates the static fixed-layout A1 dataset transform",
        },
        "datasets": {
            key: {
                "path": str(dataset_paths[key]),
                "counts": coco_counts(payloads[key]),
                "sources": source_mix_stats(payloads[key]),
            }
            for key in ("D1", "D2", "D3")
        },
        "configs": {key: str(path) for key, path in config_paths.items()},
        "evaluation": {
            "real_val": str(
                paths.baseline_dataset_root / "annotations" / "instances_real_val.json"
            ),
            "real_final_holdout": None if final_holdout_path is None else str(final_holdout_path),
            "real_final_holdout_counts": final_holdout_count,
            "base_val": str(base_val_path),
            "composite_val": str(
                paths.composite_dataset_root
                / "annotations"
                / "instances_composite_val.json"
            ),
        },
    }
    atomic_write_json(paths.experiment_root / "preparation.json", preparation)
    return preparation


def print_preparation_summary(prepared: dict[str, Any]) -> None:
    summary = {
        "phase0": {
            "complete_layout_count": prepared["inventory"]["complete_layout_count"],
            "frozen_layouts_not_complete_now": prepared["inventory"].get(
                "frozen_layouts_not_complete_now", []
            ),
            "extra_train_layout_ids": prepared["r1"]["extra_train_layout_ids"],
            "extra_holdout_layout_ids": prepared["r1"]["extra_holdout_layout_ids"],
            "r1_unique_real_counts": prepared["r1"]["unique_real_counts"],
            "r1_real_repeat": prepared["r1"]["real_repeat"],
            "r1_base_replay_counts": prepared["r1"]["base_replay_counts"],
        },
        "conditions": {
            key: prepared["datasets"][key]["sources"] for key in ("D1", "D2", "D3")
        },
        "final_holdout": prepared["evaluation"].get("real_final_holdout"),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def inventory_complete_capture_layouts(database: Path, *, campaign_id: str) -> dict[str, Any]:
    with sqlite3.connect(database) as connection:
        connection.row_factory = sqlite3.Row
        rows = connection.execute(
            """
            SELECT
                capture_task.layout_id,
                MIN(capture_task.layout_ordinal) AS layout_ordinal,
                COUNT(*) AS task_count,
                SUM(CASE WHEN capture.id IS NOT NULL THEN 1 ELSE 0 END) AS capture_count,
                SUM(CASE WHEN capture_annotation.status = 'complete' THEN 1 ELSE 0 END) AS complete_count
            FROM capture_task
            LEFT JOIN capture ON capture.task_id = capture_task.id
            LEFT JOIN capture_annotation ON capture_annotation.capture_id = capture.id
            WHERE capture_task.campaign_id = ?
            GROUP BY capture_task.layout_id
            ORDER BY MIN(capture_task.layout_ordinal)
            """,
            (campaign_id,),
        ).fetchall()
    layouts = []
    for row in rows:
        task_count = int(row["task_count"])
        capture_count = int(row["capture_count"] or 0)
        complete_count = int(row["complete_count"] or 0)
        layouts.append(
            {
                "layout_id": str(row["layout_id"]),
                "layout_ordinal": int(row["layout_ordinal"]),
                "task_count": task_count,
                "capture_count": capture_count,
                "complete_count": complete_count,
                "complete": task_count > 0
                and task_count == capture_count
                and task_count == complete_count,
            }
        )
    return {
        "campaign_id": campaign_id,
        "layout_count": len(layouts),
        "complete_layout_count": sum(bool(item["complete"]) for item in layouts),
        "layouts": layouts,
    }


def summarize_completed_captures(captures: Sequence[dict[str, Any]]) -> dict[str, Any]:
    region_counts = {key: 0 for key in REGION_KEYS}
    environment_counts: defaultdict[str, int] = defaultdict(int)
    absolute_angles: list[float] = []
    per_layout: defaultdict[str, dict[str, Any]] = defaultdict(
        lambda: {"captures": 0, "annotations": 0, "regions": {key: 0 for key in REGION_KEYS}}
    )
    for capture in captures:
        environment_key = f"{capture['brightness']}|{capture['shadow']}"
        environment_counts[environment_key] += 1
        layout_id = str(capture["layout_id"])
        per_layout[layout_id]["captures"] += 1
        annotation = json.loads(str(capture["annotation_json"]))
        boxes = annotation.get("boxes")
        if not isinstance(boxes, dict):
            raise ValueError(f"Capture {capture['capture_id']} annotation has no boxes object")
        for region in REGION_KEYS:
            values = boxes.get(region)
            if not isinstance(values, list):
                raise ValueError(
                    f"Capture {capture['capture_id']} boxes.{region} is not an array"
                )
            count = len(values)
            region_counts[region] += count
            per_layout[layout_id]["regions"][region] += count
            per_layout[layout_id]["annotations"] += count
            for box in values:
                absolute_angles.append(abs(float(box.get("angleDeg", 0.0))))
    return {
        "captures": len(captures),
        "layouts": len(per_layout),
        "annotations": sum(region_counts.values()),
        "region_annotations": region_counts,
        "environment_capture_counts": dict(sorted(environment_counts.items())),
        "absolute_box_angle_deg": distribution_summary(absolute_angles),
        "boxes_abs_angle_ge_10deg": sum(value >= 10.0 for value in absolute_angles),
        "boxes_abs_angle_ge_20deg": sum(value >= 20.0 for value in absolute_angles),
        "per_layout": dict(sorted(per_layout.items())),
    }


def combine_frozen_real_train_with_new_layouts(
    frozen_baseline_real: dict[str, Any],
    newly_completed_real: dict[str, Any] | None,
) -> dict[str, Any]:
    sources: list[tuple[str, dict[str, Any]]] = [
        ("frozen_d0_real_train", frozen_baseline_real)
    ]
    if newly_completed_real is not None:
        sources.append(("newly_completed_real", newly_completed_real))
    return merge_named_payloads(
        sources,
        description=(
            f"{INVESTIGATION_ID} R1 unique real training captures: "
            "frozen D0 train plus newly completed layout-disjoint additions"
        ),
    )


def split_extra_layouts(
    layout_ids: Sequence[str], *, train_fraction: float, seed: int
) -> tuple[frozenset[str], frozenset[str]]:
    values = list(dict.fromkeys(str(value) for value in layout_ids))
    if not values:
        return frozenset(), frozenset()
    shuffled = list(values)
    random.Random(seed).shuffle(shuffled)
    if len(shuffled) == 1:
        # A single new layout is more valuable as a true never-trained holdout.
        return frozenset(), frozenset(shuffled)
    train_count = round(len(shuffled) * train_fraction)
    train_count = max(1, min(len(shuffled) - 1, train_count))
    return frozenset(shuffled[:train_count]), frozenset(shuffled[train_count:])


def load_complete_campaign_captures(
    database: Path,
    *,
    campaign_id: str,
    required_layout_ids: frozenset[str],
) -> list[dict[str, Any]]:
    if not required_layout_ids:
        raise ValueError("No complete layouts available")
    with sqlite3.connect(database) as connection:
        connection.row_factory = sqlite3.Row
        rows = connection.execute(
            """
            SELECT
                capture.id AS capture_id,
                capture.composite_path,
                capture.manifest_json,
                capture_task.layout_id,
                capture_task.layout_ordinal,
                capture_task.environment_ordinal,
                capture_task.brightness,
                capture_task.shadow,
                capture_annotation.annotation_json
            FROM capture
            JOIN capture_task ON capture_task.id = capture.task_id
            JOIN capture_annotation ON capture_annotation.capture_id = capture.id
            WHERE capture_task.campaign_id = ? AND capture_annotation.status = 'complete'
            ORDER BY capture_task.layout_ordinal, capture_task.environment_ordinal
            """,
            (campaign_id,),
        ).fetchall()
    selected = [dict(row) for row in rows if str(row["layout_id"]) in required_layout_ids]
    selected_layout_ids = {str(row["layout_id"]) for row in selected}
    missing = required_layout_ids - selected_layout_ids
    if missing:
        raise ValueError(f"Complete-layout inventory and capture rows disagree: missing={sorted(missing)}")
    return selected


def run_region_augmentation(
    paths: Paths,
    *,
    annotations_path: Path,
    output_directory: Path,
    copies_per_image: int,
    max_rotation_deg: float,
    max_shrink_fraction: float,
    seed: int,
) -> None:
    command = [
        str(paths.nanodet_python),
        str(paths.region_augment_script),
        "--repository-root",
        str(paths.repository_root),
        "--annotations",
        str(annotations_path),
        "--output-directory",
        str(output_directory),
        "--copies-per-image",
        str(copies_per_image),
        "--max-rotation-deg",
        repr(max_rotation_deg),
        "--max-shrink-fraction",
        repr(max_shrink_fraction),
        "--seed",
        str(seed),
        "--preflight-count",
        "12",
        "--overwrite",
    ]
    run_command(command, cwd=paths.repository_root, label=f"A1 augmentation: {output_directory.name}")


def load_coco(path: Path) -> dict[str, Any]:
    payload = load_json(path)
    for key in ("images", "annotations", "categories"):
        if not isinstance(payload.get(key), list):
            raise ValueError(f"Invalid COCO {key}: {path}")
    return payload


def coco_counts(payload: dict[str, Any]) -> dict[str, int]:
    return {"images": len(payload["images"]), "annotations": len(payload["annotations"])}


def filter_payload_by_image(
    payload: dict[str, Any], predicate: Any, *, description: str
) -> dict[str, Any]:
    selected_images = [dict(image) for image in payload["images"] if predicate(image)]
    selected_ids = {int(image["id"]) for image in selected_images}
    selected_annotations = [
        dict(annotation)
        for annotation in payload["annotations"]
        if int(annotation["image_id"]) in selected_ids
    ]
    result = empty_coco(description)
    result["images"] = selected_images
    result["annotations"] = selected_annotations
    return result


def repeat_each_image(
    payload: dict[str, Any], repeat: int, *, source_name: str
) -> dict[str, Any]:
    if repeat < 1:
        raise ValueError("repeat must be positive")
    annotations_by_image = annotations_by_image_id(payload)
    result = empty_coco(f"{payload['info'].get('description', '')} repeated {repeat}x")
    next_image_id = 1
    next_annotation_id = 1
    for image in payload["images"]:
        source_image_id = int(image["id"])
        for repeat_index in range(repeat):
            clone = {key: value for key, value in image.items() if key != "id"}
            clone.update(
                {
                    "id": next_image_id,
                    "inv010_source": source_name,
                    "inv010_source_image_id": source_image_id,
                    "inv010_repeat_index": repeat_index,
                }
            )
            result["images"].append(clone)
            for annotation in annotations_by_image[source_image_id]:
                copied = {
                    key: value
                    for key, value in annotation.items()
                    if key not in {"id", "image_id", "category_id"}
                }
                copied.update(
                    {
                        "id": next_annotation_id,
                        "image_id": next_image_id,
                        "category_id": 1,
                    }
                )
                result["annotations"].append(copied)
                next_annotation_id += 1
            next_image_id += 1
    return result


def build_source_variant_exposures(
    originals: dict[str, Any],
    augmented: dict[str, Any],
    *,
    exposures_per_source: int,
    seed: int,
) -> dict[str, Any]:
    """Keep source exposure fixed while replacing repeated originals with geometric variants.

    Every original source image gets exactly ``exposures_per_source`` entries. Its version
    pool contains the untouched original followed by all A1 copies whose ``source_image_id``
    points to it. Selection cycles through a deterministic source-specific rotation of that
    pool, so A1 changes geometric diversity without silently changing image/target exposure.
    """
    if exposures_per_source < 1:
        raise ValueError("exposures_per_source must be positive")
    original_annotations = annotations_by_image_id(originals)
    augmented_annotations = annotations_by_image_id(augmented)
    augmented_by_source: defaultdict[int, list[dict[str, Any]]] = defaultdict(list)
    for image in augmented["images"]:
        if "source_image_id" not in image:
            raise ValueError("A1 image is missing source_image_id")
        augmented_by_source[int(image["source_image_id"])].append(image)
    for values in augmented_by_source.values():
        values.sort(key=lambda image: (int(image.get("region_rotation_copy_index", 0)), int(image["id"])))

    result = empty_coco("INV010 fixed source exposure with A1 variants")
    next_image_id = 1
    next_annotation_id = 1
    for original in originals["images"]:
        source_id = int(original["id"])
        variants: list[tuple[str, dict[str, Any], list[dict[str, Any]]]] = [
            ("original", original, original_annotations[source_id])
        ]
        variants.extend(
            ("a1", image, augmented_annotations[int(image["id"])])
            for image in augmented_by_source.get(source_id, [])
        )
        if len(variants) < 2:
            raise ValueError(f"Source image {source_id} has no A1 variants")
        offset = deterministic_index(seed, source_id, modulo=len(variants))
        for exposure_index in range(exposures_per_source):
            variant_kind, variant_image, variant_annotations = variants[
                (offset + exposure_index) % len(variants)
            ]
            copied_image = {key: value for key, value in variant_image.items() if key != "id"}
            copied_image.update(
                {
                    "id": next_image_id,
                    "inv010_source": "real_capture_a1",
                    "inv010_source_image_id": source_id,
                    "inv010_exposure_index": exposure_index,
                    "inv010_variant_kind": variant_kind,
                }
            )
            result["images"].append(copied_image)
            for annotation in variant_annotations:
                copied_annotation = {
                    key: value
                    for key, value in annotation.items()
                    if key not in {"id", "image_id", "category_id"}
                }
                copied_annotation.update(
                    {
                        "id": next_annotation_id,
                        "image_id": next_image_id,
                        "category_id": 1,
                    }
                )
                result["annotations"].append(copied_annotation)
                next_annotation_id += 1
            next_image_id += 1
    return result


def deterministic_index(seed: int, source_id: int, *, modulo: int) -> int:
    if modulo < 1:
        raise ValueError("modulo must be positive")
    digest = hashlib.sha256(f"{seed}:{source_id}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % modulo


def annotations_by_image_id(payload: dict[str, Any]) -> dict[int, list[dict[str, Any]]]:
    result: defaultdict[int, list[dict[str, Any]]] = defaultdict(list)
    for annotation in payload["annotations"]:
        result[int(annotation["image_id"])].append(annotation)
    return dict(result)


def sample_images_near_annotation_target(
    payload: dict[str, Any], *, target_annotations: int, seed: int
) -> dict[str, Any]:
    if target_annotations < 1:
        raise ValueError("target_annotations must be positive")
    annotations_by_image = annotations_by_image_id(payload)
    candidates = list(payload["images"])
    random.Random(seed).shuffle(candidates)
    selected: list[dict[str, Any]] = []
    total = 0
    for image in candidates:
        count = len(annotations_by_image.get(int(image["id"]), []))
        before_error = abs(target_annotations - total)
        after_error = abs(target_annotations - (total + count))
        if selected and total >= target_annotations:
            break
        if selected and after_error > before_error and total >= target_annotations * 0.90:
            break
        selected.append(image)
        total += count
    if not selected:
        raise ValueError("Base replay sampler selected no images")
    selected_ids = {int(image["id"]) for image in selected}
    result = empty_coco("INV010 base replay sampled near annotation target")
    result["images"] = [dict(image) for image in selected]
    result["annotations"] = [
        dict(annotation)
        for annotation in payload["annotations"]
        if int(annotation["image_id"]) in selected_ids
    ]
    return result


def merge_named_payloads(
    sources: Iterable[tuple[str, dict[str, Any]]], *, description: str
) -> dict[str, Any]:
    result = empty_coco(description)
    next_image_id = 1
    next_annotation_id = 1
    for source_name, payload in sources:
        by_image = annotations_by_image_id(payload)
        for source_image in payload["images"]:
            old_id = int(source_image["id"])
            image = {key: value for key, value in source_image.items() if key != "id"}
            image.update({"id": next_image_id, "fine_tune_source": source_name})
            result["images"].append(image)
            for source_annotation in by_image.get(old_id, []):
                annotation = {
                    key: value
                    for key, value in source_annotation.items()
                    if key not in {"id", "image_id", "category_id"}
                }
                annotation.update(
                    {
                        "id": next_annotation_id,
                        "image_id": next_image_id,
                        "category_id": 1,
                    }
                )
                result["annotations"].append(annotation)
                next_annotation_id += 1
            next_image_id += 1
    return result


def source_mix_stats(payload: dict[str, Any]) -> dict[str, Any]:
    annotations_by_image = annotations_by_image_id(payload)
    stats: defaultdict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "images": 0,
            "annotations": 0,
            "file_names": set(),
            "source_bases": set(),
        }
    )
    for image in payload["images"]:
        source = str(image.get("fine_tune_source", image.get("inv010_source", "unknown")))
        item = stats[source]
        item["images"] += 1
        item["annotations"] += len(annotations_by_image.get(int(image["id"]), []))
        item["file_names"].add(str(image.get("file_name", "")))
        source_base = image.get("inv010_source_image_id", image.get("source_image_id", image.get("file_name")))
        item["source_bases"].add(str(source_base))
    total_images = max(1, len(payload["images"]))
    total_annotations = max(1, len(payload["annotations"]))
    return {
        source: {
            "images": int(item["images"]),
            "annotations": int(item["annotations"]),
            "image_entry_share": item["images"] / total_images,
            "annotation_target_share": item["annotations"] / total_annotations,
            "unique_file_names": len(item["file_names"]),
            "unique_source_bases": len(item["source_bases"]),
        }
        for source, item in sorted(stats.items())
    }


def write_candidate_config(
    *,
    template_path: Path,
    output_path: Path,
    save_dir: Path,
    train_annotations: Path,
    val_annotations: Path,
    repository_root: Path,
    starting_checkpoint: Path,
    training_mode: str,
) -> None:
    try:
        import yaml
    except ImportError as error:
        raise RuntimeError("PyYAML is required in the NanoDet environment") from error
    with template_path.open("r", encoding="utf-8") as source:
        config = yaml.safe_load(source)
    if not isinstance(config, dict):
        raise ValueError(f"Invalid NanoDet config: {template_path}")
    config["save_dir"] = str(save_dir)
    config["data"]["train"]["img_path"] = str(repository_root)
    config["data"]["train"]["ann_path"] = str(train_annotations)
    config["data"]["val"]["img_path"] = str(repository_root)
    config["data"]["val"]["ann_path"] = str(val_annotations)
    config["schedule"]["load_model"] = str(starting_checkpoint)
    if training_mode == "finetune":
        # D1-D3 intentionally keep every NanoDet pipeline/schedule setting identical
        # to D0. A1 is precomputed inside fixed semantic regions, so D2 isolates it.
        pass
    elif training_mode == "joint":
        pass
    else:
        raise ValueError(f"Unknown training mode: {training_mode}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="\n") as output:
        yaml.safe_dump(config, output, sort_keys=False, allow_unicode=True)


def condition_run_directory(paths: Paths, key: str, *, seed: int) -> Path:
    return (
        paths.repository_root
        / ".local"
        / "recognition"
        / "nanodet_runs"
        / f"INV010_{key}_seed{seed}"
    )


def conditions_from_preparation(paths: Paths, prepared: dict[str, Any]) -> dict[str, Condition]:
    result: dict[str, Condition] = {}
    for key in ("D1", "D2", "D3"):
        dataset_path = Path(str(prepared["datasets"][key]["path"])).resolve()
        config_path = Path(str(prepared["configs"][key])).resolve()
        run_directory = condition_run_directory(paths, key, seed=int(prepared["seed"]))
        result[key] = Condition(
            key=key,
            source_mix="R1" if key in {"D1", "D3"} else "R0",
            augmentation="A1" if key in {"D2", "D3"} else "A0",
            dataset_path=dataset_path,
            run_directory=run_directory,
            config_path=config_path,
            onnx_path=run_directory / "model_best" / f"inv010-{key.lower()}.onnx",
        )
    return result


def train_condition(
    paths: Paths,
    condition: Condition,
    *,
    seed: int,
    overwrite: bool = False,
) -> None:
    completion_marker = condition.run_directory / "inv010_training_completed.json"
    if condition.run_directory.exists() and not overwrite and completion_marker.is_file():
        marker = load_json(completion_marker)
        expected_config_sha = sha256_file(condition.config_path)
        if (
            marker.get("status") == "completed"
            and int(marker.get("seed", -1)) == seed
            and marker.get("config_sha256") == expected_config_sha
        ):
            try:
                checkpoint = resolve_best_checkpoint(condition.run_directory)
            except (FileNotFoundError, RuntimeError):
                checkpoint = None
            if checkpoint is not None:
                print(
                    f"[resume] reuse completed {condition.key} checkpoint: {checkpoint}",
                    flush=True,
                )
                return
    if condition.run_directory.exists():
        print(f"[train] removing incomplete/overwritten run: {condition.run_directory}")
        shutil.rmtree(condition.run_directory)
    command = [
        str(paths.nanodet_python),
        "tools/train.py",
        str(condition.config_path),
        "--seed",
        str(seed),
    ]
    run_command(command, cwd=paths.nanodet_root, label=f"train {condition.key}")
    checkpoint = resolve_best_checkpoint(condition.run_directory)
    atomic_write_json(
        completion_marker,
        {
            "status": "completed",
            "condition": condition.key,
            "seed": seed,
            "config": str(condition.config_path),
            "config_sha256": sha256_file(condition.config_path),
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": sha256_file(checkpoint),
        },
    )


def resolve_best_checkpoint(run_directory: Path) -> Path:
    model_best = run_directory / "model_best"
    exact = (
        model_best / "model_best.ckpt",
        model_best / "nanodet_model_best.pth",
        model_best / "model_best.pth",
    )
    for path in exact:
        if path.is_file():
            return path
    candidates = sorted(
        path
        for path in model_best.rglob("*")
        if path.is_file() and path.suffix.lower() in {".ckpt", ".pth", ".pt"}
    )
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        raise FileNotFoundError(f"No best checkpoint under {model_best}")
    raise RuntimeError(f"Multiple best-checkpoint candidates under {model_best}: {candidates}")


def export_condition_onnx(paths: Paths, condition: Condition) -> None:
    checkpoint = resolve_best_checkpoint(condition.run_directory)
    condition.onnx_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        str(paths.nanodet_python),
        "tools/export_onnx.py",
        "--cfg_path",
        str(condition.config_path),
        "--model_path",
        str(checkpoint),
        "--out_path",
        str(condition.onnx_path),
        "--input_shape",
        "320,320",
    ]
    run_command(command, cwd=paths.nanodet_root, label=f"export {condition.key}")
    validate_onnx(condition.onnx_path)


def ensure_baseline_onnx(paths: Paths) -> None:
    if paths.baseline_onnx.is_file():
        validate_onnx(paths.baseline_onnx)
        return
    checkpoint = resolve_best_checkpoint(paths.baseline_run)
    paths.baseline_onnx.parent.mkdir(parents=True, exist_ok=True)
    command = [
        str(paths.nanodet_python),
        "tools/export_onnx.py",
        "--cfg_path",
        str(paths.baseline_finetune_config),
        "--model_path",
        str(checkpoint),
        "--out_path",
        str(paths.baseline_onnx),
        "--input_shape",
        "320,320",
    ]
    run_command(command, cwd=paths.nanodet_root, label="export D0 baseline")
    validate_onnx(paths.baseline_onnx)


def validate_onnx(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)
    try:
        import onnx
    except ImportError as error:
        raise RuntimeError("onnx is required for INV010 export validation") from error
    model = onnx.load(str(path))
    onnx.checker.check_model(model)


def resolve_evaluation_sets(paths: Paths, prepared: dict[str, Any]) -> dict[str, Path]:
    sets = {
        "real_val": Path(str(prepared["evaluation"]["real_val"])).resolve(),
        "base_val": Path(str(prepared["evaluation"]["base_val"])).resolve(),
        "composite_val": Path(str(prepared["evaluation"]["composite_val"])).resolve(),
    }
    final_holdout = prepared["evaluation"].get("real_final_holdout")
    if final_holdout:
        sets["real_final_holdout"] = Path(str(final_holdout)).resolve()
    for key, path in sets.items():
        if not path.is_file():
            raise FileNotFoundError(f"Evaluation set {key}: {path}")
    return sets


def evaluate_models(
    paths: Paths,
    *,
    model_paths: dict[str, Path],
    evaluation_sets: dict[str, Path],
    candidate_threshold: float,
    operating_threshold: float,
    nms_iou_threshold: float,
    max_detections: int,
    match_iou_threshold: float,
    worst_case_count: int,
    overwrite: bool,
    existing_reports: dict[str, dict[str, Any]] | None = None,
) -> dict[str, dict[str, Any]]:
    reports = {} if existing_reports is None else dict(existing_reports)
    for condition_key, model_path in model_paths.items():
        condition_reports: dict[str, Any] = {}
        for dataset_key, annotations_path in evaluation_sets.items():
            output_directory = (
                paths.experiment_root / "evaluation" / condition_key / dataset_key
            )
            report_path = output_directory / "report.json"
            if report_path.is_file() and not overwrite:
                condition_reports[dataset_key] = load_json(report_path)
                continue
            if output_directory.exists():
                shutil.rmtree(output_directory)
            result = evaluate_model_on_coco(
                model_path=model_path,
                annotations_path=annotations_path,
                repository_root=paths.repository_root,
                output_directory=output_directory,
                use_fixed_regions=(dataset_key != "base_val"),
                candidate_threshold=candidate_threshold,
                operating_threshold=operating_threshold,
                nms_iou_threshold=nms_iou_threshold,
                max_detections=max_detections,
                match_iou_threshold=match_iou_threshold,
                worst_case_count=worst_case_count,
            )
            condition_reports[dataset_key] = result.report
        reports[condition_key] = condition_reports
    return reports


def evaluate_model_on_coco(
    *,
    model_path: Path,
    annotations_path: Path,
    repository_root: Path,
    output_directory: Path,
    use_fixed_regions: bool,
    candidate_threshold: float,
    operating_threshold: float,
    nms_iou_threshold: float,
    max_detections: int,
    match_iou_threshold: float,
    worst_case_count: int,
) -> EvalResult:
    try:
        import onnxruntime as ort
    except ImportError as error:
        raise RuntimeError("onnxruntime is required for INV010 evaluation") from error

    coco = load_coco(annotations_path)
    images = {int(image["id"]): image for image in coco["images"]}
    gt_by_image: defaultdict[int, list[dict[str, Any]]] = defaultdict(list)
    for annotation in coco["annotations"]:
        if int(annotation.get("iscrowd", 0)) == 0:
            gt_by_image[int(annotation["image_id"])].append(annotation)
    layout = load_json(repository_root / "tools" / "recognition" / "capture_layout.v1.json")
    region_destinations = {
        key: layout["regions"][key]["destination"] for key in REGION_KEYS
    }

    session = ort.InferenceSession(
        str(model_path),
        providers=(
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if "CUDAExecutionProvider" in ort.get_available_providers()
            else ["CPUExecutionProvider"]
        ),
    )
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    output_directory.mkdir(parents=True, exist_ok=True)
    all_predictions: list[PredictionRecord] = []
    all_matches: list[MatchRecord] = []
    false_negatives: list[dict[str, Any]] = []
    false_positives: list[dict[str, Any]] = []
    coco_predictions: list[dict[str, Any]] = []

    for ordinal, (image_id, image_record) in enumerate(sorted(images.items()), start=1):
        image_path = resolve_repository_image(repository_root, str(image_record["file_name"]))
        tensor, source = preprocess_image(image_path)
        source.close()
        raw = session.run([output_name], {input_name: tensor})[0]
        candidates = decode_output(
            raw,
            confidence_threshold=candidate_threshold,
            nms_iou_threshold=nms_iou_threshold,
            max_detections=max_detections,
        )
        for detection in candidates:
            coco_predictions.append(
                {
                    "image_id": image_id,
                    "category_id": 1,
                    "bbox": detection.box.to_coco(),
                    "score": float(detection.score),
                }
            )
        operating = [item for item in candidates if float(item.score) >= operating_threshold]
        predictions: list[PredictionRecord] = []
        for prediction_index, detection in enumerate(operating):
            bbox = tuple(float(value) for value in detection.box.to_coco())
            center_x = bbox[0] + bbox[2] / 2.0
            center_y = bbox[1] + bbox[3] / 2.0
            region = (
                region_for_point(center_x, center_y, region_destinations)
                if use_fixed_regions
                else "all"
            )
            record = PredictionRecord(
                image_id=image_id,
                prediction_index=prediction_index,
                region=region,
                score=float(detection.score),
                bbox=bbox,
            )
            predictions.append(record)
            all_predictions.append(record)

        matches, image_false_negatives, image_false_positives = match_image(
            image_id=image_id,
            file_name=str(image_record["file_name"]),
            ground_truths=gt_by_image.get(image_id, []),
            predictions=predictions,
            region_destinations=region_destinations,
            use_fixed_regions=use_fixed_regions,
            match_iou_threshold=match_iou_threshold,
        )
        all_matches.extend(matches)
        false_negatives.extend(image_false_negatives)
        false_positives.extend(image_false_positives)
        if ordinal % 20 == 0 or ordinal == len(images):
            print(
                f"[eval] {model_path.name} {annotations_path.name} "
                f"{ordinal}/{len(images)} matches={len(all_matches)} "
                f"fn={len(false_negatives)} fp={len(false_positives)}"
            )

    predictions_path = output_directory / "predictions.json"
    atomic_write_json(predictions_path, coco_predictions)
    official_metrics, official_error = official_coco_metrics(
        annotations_path,
        predictions_path,
        max_detections=max_detections,
    )

    overall = summarize_crop_quality(
        all_matches,
        false_negative_count=len(false_negatives),
        false_positive_count=len(false_positives),
        ground_truth_count=len(coco["annotations"]),
    )
    by_region = {}
    region_summary_keys = (*REGION_KEYS, "outside") if use_fixed_regions else ("all",)
    for region in region_summary_keys:
        region_matches = [match for match in all_matches if match.region == region]
        region_fn = sum(item["region"] == region for item in false_negatives)
        region_fp = sum(item["region"] == region for item in false_positives)
        region_gt = sum(
            (
                annotation_region(annotation, region_destinations)
                if use_fixed_regions
                else "all"
            )
            == region
            for annotation in coco["annotations"]
        )
        if region_gt or region_matches or region_fn or region_fp:
            by_region[region] = summarize_crop_quality(
                region_matches,
                false_negative_count=region_fn,
                false_positive_count=region_fp,
                ground_truth_count=region_gt,
            )

    worst_directory = output_directory / "worst_crops"
    render_worst_cases(
        repository_root=repository_root,
        images=images,
        ground_truths=gt_by_image,
        matches=all_matches,
        false_negatives=false_negatives,
        false_positives=false_positives,
        output_directory=worst_directory,
        limit=worst_case_count,
    )

    report = {
        "status": "completed",
        "model": str(model_path),
        "model_sha256": sha256_file(model_path),
        "annotations": str(annotations_path),
        "images": len(coco["images"]),
        "ground_truths": len(coco["annotations"]),
        "postprocess": {
            "candidate_threshold": candidate_threshold,
            "operating_threshold": operating_threshold,
            "nms_iou_threshold": nms_iou_threshold,
            "max_detections": max_detections,
            "match_iou_threshold": match_iou_threshold,
            "fixed_layout_region_matching": use_fixed_regions,
        },
        "official_coco": official_metrics,
        "official_coco_error": official_error,
        "crop_quality": {
            "overall": overall,
            "by_region": by_region,
        },
        "artifacts": {
            "predictions": str(predictions_path),
            "worst_crops": str(worst_directory),
        },
        "runtime": {
            "onnxruntime_version": ort.__version__,
            "providers": session.get_providers(),
        },
    }
    atomic_write_json(output_directory / "report.json", report)
    return EvalResult(report=report, matches=tuple(all_matches), predictions=tuple(all_predictions))


def match_image(
    *,
    image_id: int,
    file_name: str,
    ground_truths: Sequence[dict[str, Any]],
    predictions: Sequence[PredictionRecord],
    region_destinations: dict[str, dict[str, Any]],
    use_fixed_regions: bool,
    match_iou_threshold: float,
) -> tuple[list[MatchRecord], list[dict[str, Any]], list[dict[str, Any]]]:
    gt_region = {
        int(annotation["id"]): (
            annotation_region(annotation, region_destinations)
            if use_fixed_regions
            else "all"
        )
        for annotation in ground_truths
    }
    matched_gt_ids: set[int] = set()
    matches: list[MatchRecord] = []
    false_positives: list[dict[str, Any]] = []
    for prediction in sorted(predictions, key=lambda item: item.score, reverse=True):
        candidates = [
            annotation
            for annotation in ground_truths
            if int(annotation["id"]) not in matched_gt_ids
            and gt_region[int(annotation["id"])] == prediction.region
        ]
        scored = [
            (annotation, prediction_gt_metrics(prediction.bbox, annotation))
            for annotation in candidates
        ]
        if scored:
            annotation, metrics = max(scored, key=lambda item: item[1].iou)
        else:
            annotation, metrics = None, None
        if annotation is None or metrics is None or metrics.iou < match_iou_threshold:
            false_positives.append(
                {
                    "image_id": image_id,
                    "file_name": file_name,
                    "region": prediction.region,
                    "prediction_index": prediction.prediction_index,
                    "score": prediction.score,
                    "bbox": list(prediction.bbox),
                }
            )
            continue
        annotation_id = int(annotation["id"])
        matched_gt_ids.add(annotation_id)
        matches.append(
            MatchRecord(
                image_id=image_id,
                file_name=file_name,
                region=prediction.region,
                gt_annotation_id=annotation_id,
                prediction_index=prediction.prediction_index,
                score=prediction.score,
                prediction_bbox=prediction.bbox,
                gt_bbox=tuple(float(value) for value in annotation["bbox"]),
                metrics=metrics,
            )
        )
    false_negatives = [
        {
            "image_id": image_id,
            "file_name": file_name,
            "region": gt_region[int(annotation["id"])],
            "gt_annotation_id": int(annotation["id"]),
            "bbox": [float(value) for value in annotation["bbox"]],
        }
        for annotation in ground_truths
        if int(annotation["id"]) not in matched_gt_ids
    ]
    return matches, false_negatives, false_positives


def prediction_gt_metrics(
    prediction_bbox: Sequence[float], annotation: dict[str, Any]
) -> BoxMetrics:
    x, y, width, height = (float(value) for value in prediction_bbox)
    polygon = tuple((float(px), float(py)) for px, py in annotation_polygon(annotation))
    gt_area = polygon_area(polygon)
    clipped = clip_polygon_to_rect(polygon, x, y, x + width, y + height)
    intersection = polygon_area(clipped)
    pred_area = max(0.0, width) * max(0.0, height)
    union = pred_area + gt_area - intersection
    iou = 0.0 if union <= 0.0 else intersection / union
    gt_coverage = 0.0 if gt_area <= 0.0 else intersection / gt_area
    crop_purity = 0.0 if pred_area <= 0.0 else intersection / pred_area

    gx, gy, gw, gh = (float(value) for value in annotation["bbox"])
    pred_cx = x + width / 2.0
    pred_cy = y + height / 2.0
    gt_cx = gx + gw / 2.0
    gt_cy = gy + gh / 2.0
    return BoxMetrics(
        iou=iou,
        gt_coverage=gt_coverage,
        crop_purity=crop_purity,
        center_error_x=abs(pred_cx - gt_cx) / max(gw, 1.0e-9),
        center_error_y=abs(pred_cy - gt_cy) / max(gh, 1.0e-9),
        width_ratio=width / max(gw, 1.0e-9),
        height_ratio=height / max(gh, 1.0e-9),
    )


def annotation_region(
    annotation: dict[str, Any], region_destinations: dict[str, dict[str, Any]]
) -> str:
    explicit = annotation.get("region")
    if explicit in REGION_KEYS:
        return str(explicit)
    x, y, width, height = (float(value) for value in annotation["bbox"])
    return region_for_point(x + width / 2.0, y + height / 2.0, region_destinations)


def region_for_point(
    x: float, y: float, region_destinations: dict[str, dict[str, Any]]
) -> str:
    for key in REGION_KEYS:
        destination = region_destinations[key]
        left = float(destination["x"])
        top = float(destination["y"])
        right = left + float(destination["width"])
        bottom = top + float(destination["height"])
        if left <= x <= right and top <= y <= bottom:
            return key
    return "outside"


def summarize_crop_quality(
    matches: Sequence[MatchRecord],
    *,
    false_negative_count: int,
    false_positive_count: int,
    ground_truth_count: int,
) -> dict[str, Any]:
    tp = len(matches)
    precision = tp / (tp + false_positive_count) if tp + false_positive_count else 0.0
    recall = tp / ground_truth_count if ground_truth_count else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "true_positives": tp,
        "false_positives": false_positive_count,
        "false_negatives": false_negative_count,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "iou": distribution_summary([match.metrics.iou for match in matches]),
        "gt_coverage": distribution_summary([match.metrics.gt_coverage for match in matches]),
        "crop_purity": distribution_summary([match.metrics.crop_purity for match in matches]),
        "center_error_x": distribution_summary([match.metrics.center_error_x for match in matches]),
        "center_error_y": distribution_summary([match.metrics.center_error_y for match in matches]),
        "width_ratio": distribution_summary([match.metrics.width_ratio for match in matches]),
        "height_ratio": distribution_summary([match.metrics.height_ratio for match in matches]),
    }


def distribution_summary(values: Sequence[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "p10": None, "median": None, "p90": None, "mean": None}
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": len(values),
        "p10": float(np.percentile(array, 10.0)),
        "median": float(np.median(array)),
        "p90": float(np.percentile(array, 90.0)),
        "mean": float(np.mean(array)),
    }


def polygon_area(polygon: Sequence[tuple[float, float]]) -> float:
    if len(polygon) < 3:
        return 0.0
    return abs(
        sum(
            polygon[index][0] * polygon[(index + 1) % len(polygon)][1]
            - polygon[(index + 1) % len(polygon)][0] * polygon[index][1]
            for index in range(len(polygon))
        )
    ) / 2.0


def clip_polygon_to_rect(
    polygon: Sequence[tuple[float, float]],
    left: float,
    top: float,
    right: float,
    bottom: float,
) -> list[tuple[float, float]]:
    result = list(polygon)
    result = clip_polygon(result, lambda p: p[0] >= left, lambda a, b: intersect_vertical(a, b, left))
    result = clip_polygon(result, lambda p: p[0] <= right, lambda a, b: intersect_vertical(a, b, right))
    result = clip_polygon(result, lambda p: p[1] >= top, lambda a, b: intersect_horizontal(a, b, top))
    result = clip_polygon(result, lambda p: p[1] <= bottom, lambda a, b: intersect_horizontal(a, b, bottom))
    return result


def clip_polygon(
    polygon: Sequence[tuple[float, float]], inside: Any, intersection: Any
) -> list[tuple[float, float]]:
    if not polygon:
        return []
    output: list[tuple[float, float]] = []
    previous = polygon[-1]
    previous_inside = bool(inside(previous))
    for current in polygon:
        current_inside = bool(inside(current))
        if current_inside:
            if not previous_inside:
                output.append(intersection(previous, current))
            output.append(current)
        elif previous_inside:
            output.append(intersection(previous, current))
        previous = current
        previous_inside = current_inside
    return output


def intersect_vertical(
    left: tuple[float, float], right: tuple[float, float], x: float
) -> tuple[float, float]:
    delta = right[0] - left[0]
    if abs(delta) < 1.0e-12:
        return (x, left[1])
    ratio = (x - left[0]) / delta
    return (x, left[1] + ratio * (right[1] - left[1]))


def intersect_horizontal(
    left: tuple[float, float], right: tuple[float, float], y: float
) -> tuple[float, float]:
    delta = right[1] - left[1]
    if abs(delta) < 1.0e-12:
        return (left[0], y)
    ratio = (y - left[1]) / delta
    return (left[0] + ratio * (right[0] - left[0]), y)


def official_coco_metrics(
    annotations_path: Path, predictions_path: Path, *, max_detections: int
) -> tuple[dict[str, float] | None, str | None]:
    try:
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval
    except ImportError:
        return None, "pycocotools unavailable"
    try:
        ground_truth = COCO(str(annotations_path))
        detections = ground_truth.loadRes(str(predictions_path))
        evaluator = COCOeval(ground_truth, detections, "bbox")
        evaluator.params.maxDets = [1, 10, max_detections]
        evaluator.evaluate()
        evaluator.accumulate()
        evaluator.summarize()
        stats = evaluator.stats
        return (
            {
                "AP": float(stats[0]),
                "AP50": float(stats[1]),
                "AP75": float(stats[2]),
                f"AR_max_{max_detections}": float(stats[8]),
            },
            None,
        )
    except Exception as error:
        return None, f"{type(error).__name__}: {error}"


def render_worst_cases(
    *,
    repository_root: Path,
    images: dict[int, dict[str, Any]],
    ground_truths: dict[int, list[dict[str, Any]]],
    matches: Sequence[MatchRecord],
    false_negatives: Sequence[dict[str, Any]],
    false_positives: Sequence[dict[str, Any]],
    output_directory: Path,
    limit: int,
) -> None:
    output_directory.mkdir(parents=True, exist_ok=True)
    ranked_matches = sorted(
        matches,
        key=lambda match: min(
            match.metrics.gt_coverage,
            match.metrics.crop_purity,
            match.metrics.iou,
        ),
    )
    cases: list[tuple[str, Any]] = []
    cases.extend(("fn", item) for item in false_negatives)
    cases.extend(("fp", item) for item in false_positives)
    cases.extend(("match", item) for item in ranked_matches)
    cases = cases[:limit]
    if not cases:
        return

    rows: list[Image.Image] = []
    for kind, item in cases:
        image_id = int(item.image_id if isinstance(item, MatchRecord) else item["image_id"])
        image_record = images[image_id]
        image_path = resolve_repository_image(repository_root, str(image_record["file_name"]))
        with Image.open(image_path) as opened:
            source = opened.convert("RGB")
        overlay = source.copy()
        draw = ImageDraw.Draw(overlay)
        for annotation in ground_truths.get(image_id, []):
            x, y, width, height = (float(value) for value in annotation["bbox"])
            draw.rectangle((x, y, x + width, y + height), outline=(0, 255, 0), width=2)

        crop_canvas = Image.new("RGB", (320, 320), (0, 0, 0))
        if kind == "match":
            assert isinstance(item, MatchRecord)
            x, y, width, height = item.prediction_bbox
            draw.rectangle((x, y, x + width, y + height), outline=(255, 0, 0), width=2)
            crop = crop_bbox(source, item.prediction_bbox)
            label = (
                f"match {item.region} iou={item.metrics.iou:.3f} "
                f"cov={item.metrics.gt_coverage:.3f} purity={item.metrics.crop_purity:.3f}"
            )
        elif kind == "fp":
            x, y, width, height = (float(value) for value in item["bbox"])
            draw.rectangle((x, y, x + width, y + height), outline=(255, 0, 0), width=2)
            crop = crop_bbox(source, (x, y, width, height))
            label = f"FP {item['region']} score={float(item['score']):.3f}"
        else:
            crop = None
            label = f"FN {item['region']} gt={item['gt_annotation_id']}"
        if crop is not None:
            fitted = fit_image(crop, (300, 280))
            crop_canvas.paste(
                fitted,
                ((320 - fitted.width) // 2, (300 - fitted.height) // 2),
            )
            crop.close()
            fitted.close()
        row = Image.new("RGB", (640, 348), (24, 24, 24))
        row.paste(overlay, (0, 0))
        row.paste(crop_canvas, (320, 0))
        row_draw = ImageDraw.Draw(row)
        row_draw.text((4, 324), label, fill=(255, 255, 255))
        rows.append(row)
        source.close()
        overlay.close()
        crop_canvas.close()

    sheet = Image.new("RGB", (640, 348 * len(rows)), (24, 24, 24))
    for index, row in enumerate(rows):
        sheet.paste(row, (0, index * 348))
        row.close()
    sheet.save(output_directory / "contact_sheet.jpg", format="JPEG", quality=92)
    sheet.close()


def crop_bbox(image: Image.Image, bbox: Sequence[float]) -> Image.Image | None:
    x, y, width, height = (float(value) for value in bbox)
    left = max(0, math.floor(x))
    top = max(0, math.floor(y))
    right = min(image.width, math.ceil(x + width))
    bottom = min(image.height, math.ceil(y + height))
    if right <= left or bottom <= top:
        return None
    return image.crop((left, top, right, bottom)).convert("RGB")


def fit_image(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    copy = image.copy()
    copy.thumbnail(size, Image.Resampling.BILINEAR)
    return copy


def resolve_repository_image(repository_root: Path, file_name: str) -> Path:
    pure = PurePosixPath(file_name.replace("\\", "/"))
    if pure.is_absolute() or ".." in pure.parts:
        raise ValueError(f"Unsafe repository-relative image path: {file_name}")
    path = repository_root.joinpath(*pure.parts)
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def write_comparison(
    paths: Paths, reports: dict[str, dict[str, Any]], prepared: dict[str, Any]
) -> dict[str, Any]:
    dataset_order = ["real_final_holdout", "real_val", "base_val", "composite_val"]
    rows: list[dict[str, Any]] = []
    for condition_key in sorted(reports):
        for dataset_key in dataset_order:
            if dataset_key not in reports[condition_key]:
                continue
            report = reports[condition_key][dataset_key]
            overall = report["crop_quality"]["overall"]
            official = report.get("official_coco") or {}
            row = {
                "condition": condition_key,
                "dataset": dataset_key,
                "AP": official.get("AP"),
                "AP50": official.get("AP50"),
                "AP75": official.get("AP75"),
                "precision": overall["precision"],
                "recall": overall["recall"],
                "f1": overall["f1"],
                "fp": overall["false_positives"],
                "fn": overall["false_negatives"],
                "iou_p10": overall["iou"]["p10"],
                "iou_median": overall["iou"]["median"],
                "gt_coverage_p10": overall["gt_coverage"]["p10"],
                "gt_coverage_median": overall["gt_coverage"]["median"],
                "crop_purity_p10": overall["crop_purity"]["p10"],
                "crop_purity_median": overall["crop_purity"]["median"],
                "center_error_x_p90": overall["center_error_x"]["p90"],
                "center_error_y_p90": overall["center_error_y"]["p90"],
                "width_ratio_median": overall["width_ratio"]["median"],
                "height_ratio_median": overall["height_ratio"]["median"],
            }
            by_region = report["crop_quality"].get("by_region", {})
            for region in REGION_KEYS:
                region_report = by_region.get(region)
                prefix = region.replace("_", "-")
                row[f"{prefix}_recall"] = (
                    None if region_report is None else region_report["recall"]
                )
                row[f"{prefix}_gt_coverage_p10"] = (
                    None
                    if region_report is None
                    else region_report["gt_coverage"]["p10"]
                )
                row[f"{prefix}_crop_purity_p10"] = (
                    None
                    if region_report is None
                    else region_report["crop_purity"]["p10"]
                )
            rows.append(row)

    baseline_by_dataset = {
        row["dataset"]: row for row in rows if row["condition"] == "D0"
    }
    deltas = []
    for row in rows:
        if row["condition"] == "D0" or row["dataset"] not in baseline_by_dataset:
            continue
        baseline = baseline_by_dataset[row["dataset"]]
        delta = {"condition": row["condition"], "dataset": row["dataset"]}
        for key in (
            "AP",
            "AP50",
            "AP75",
            "precision",
            "recall",
            "f1",
            "iou_p10",
            "gt_coverage_p10",
            "crop_purity_p10",
        ):
            if row.get(key) is not None and baseline.get(key) is not None:
                delta[key] = float(row[key]) - float(baseline[key])
        delta["fp"] = int(row["fp"]) - int(baseline["fp"])
        delta["fn"] = int(row["fn"]) - int(baseline["fn"])
        deltas.append(delta)

    csv_path = paths.experiment_root / "comparison.csv"
    if rows:
        with csv_path.open("w", encoding="utf-8", newline="") as output:
            writer = csv.DictWriter(output, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    comparison = {
        "status": "completed",
        "investigation": INVESTIGATION_ID,
        "preparation": str(paths.experiment_root / "preparation.json"),
        "has_layout_disjoint_final_holdout": bool(
            prepared["evaluation"].get("real_final_holdout")
        ),
        "rows": rows,
        "deltas_vs_D0": deltas,
        "reports": reports,
        "artifacts": {"csv": str(csv_path)},
    }
    atomic_write_json(paths.experiment_root / "comparison.json", comparison)
    return comparison


def select_best_candidate(comparison: dict[str, Any], candidates: Sequence[str]) -> str:
    preferred_dataset = (
        "real_final_holdout"
        if comparison.get("has_layout_disjoint_final_holdout")
        else "real_val"
    )
    rows = [
        row
        for row in comparison["rows"]
        if row["condition"] in set(candidates) and row["dataset"] == preferred_dataset
    ]
    if not rows:
        raise ValueError(f"No candidate rows available for {preferred_dataset}")

    def key(row: dict[str, Any]) -> tuple[float, float, float, float]:
        coverage = float(row.get("gt_coverage_p10") or 0.0)
        purity = float(row.get("crop_purity_p10") or 0.0)
        recall = float(row.get("recall") or 0.0)
        ap = float(row.get("AP") or 0.0)
        return (min(coverage, purity), coverage + purity, recall, ap)

    return str(max(rows, key=key)["condition"])


def build_d4_condition(
    paths: Paths, source_condition: Condition, prepared: dict[str, Any], *, seed: int
) -> Condition:
    run_directory = condition_run_directory(paths, "D4", seed=seed)
    config_path = paths.experiment_root / "configs" / f"inv010_d4_seed{seed}.yml"
    try:
        import yaml
    except ImportError as error:
        raise RuntimeError("PyYAML is required in the NanoDet environment") from error
    with paths.joint_retrain_config.open("r", encoding="utf-8") as source:
        config = yaml.safe_load(source)
    with paths.baseline_finetune_config.open("r", encoding="utf-8") as source:
        finetune_config = yaml.safe_load(source)
    # D4 changes initialization/schedule, not the augmentation contract. Preserve the
    # same NanoDet train pipeline used by D0-D3; A1 remains the precomputed fixed-layout
    # region transform encoded in source_condition.dataset_path.
    config["data"]["train"]["pipeline"] = finetune_config["data"]["train"]["pipeline"]
    config["save_dir"] = str(run_directory)
    config["data"]["train"]["img_path"] = str(paths.repository_root)
    config["data"]["train"]["ann_path"] = str(source_condition.dataset_path)
    config["data"]["val"]["img_path"] = str(paths.repository_root)
    config["data"]["val"]["ann_path"] = str(
        Path(prepared["evaluation"]["real_val"]).resolve()
    )
    # The template already supplies the official pretrained checkpoint and 40-epoch schedule.
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with config_path.open("w", encoding="utf-8", newline="\n") as output:
        yaml.safe_dump(config, output, sort_keys=False, allow_unicode=True)
    return Condition(
        key="D4",
        source_mix=source_condition.source_mix,
        augmentation=source_condition.augmentation,
        dataset_path=source_condition.dataset_path,
        run_directory=run_directory,
        config_path=config_path,
        onnx_path=run_directory / "model_best" / "inv010-d4.onnx",
        training_mode="joint",
    )


def run_command(command: Sequence[str], *, cwd: Path, label: str) -> None:
    print(json.dumps({"step": label, "cwd": str(cwd), "command": list(command)}, ensure_ascii=False))
    subprocess.run(list(command), cwd=cwd, check=True)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent, text=True
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as output:
            json.dump(payload, output, ensure_ascii=False, indent=2)
            output.write("\n")
        os.replace(name, path)
    except Exception:
        try:
            os.unlink(name)
        except FileNotFoundError:
            pass
        raise


if __name__ == "__main__":
    raise SystemExit(main())
