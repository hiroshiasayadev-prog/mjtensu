from __future__ import annotations

"""Render Plain-heavy vs f8-r1-heavy internal feature maps for 5m/6m/7m.

The goal is diagnostic rather than attribution: inspect where fine-grained manzu
separability appears or disappears inside the two trained architectures under the same
input condition. The default comparison renders both a front-facing input and the
INV-013 corrected recrop case where f8-r1-heavy retained a 6m -> 5m/7m confusion.

Outputs:
- one self-contained HTML report with source crops, predictions, all-channel contact
  sheets and detailed top 6m-vs-(5m,7m) channels;
- one JSON summary with stage shapes, prediction counts and ranked channel statistics.

The report uses class-mean activation maps. Channel ranking is based on the absolute
standardized difference between the mean pooled 6m activation and the average pooled
5m/7m activation. Channel numbers are meaningful only within one model/stage; do not
compare Plain channel N to f8-r1 channel N as if they represented the same feature.
"""

import argparse
import base64
from dataclasses import asdict, dataclass
import hashlib
import html
import io
import json
import math
from pathlib import Path
import sqlite3
from typing import Any, Iterable, Sequence

import numpy as np
import torch
from torch import nn

from PIL import Image, ImageDraw

try:
    from perspective_classifier_augmentation import (
        PERSPECTIVE_EVALUATION_CASES,
        GeometryCase,
        apply_evaluation_case,
    )
    from run_perspective_classifier_experiment import load_condition_checkpoint
    from run_perspective_classifier_mixture_experiment import preletterbox_content_extent
    from run_rotation_classifier_experiment import load_cache
except ModuleNotFoundError:  # package-style import used by tests
    from tools.recognition.perspective_classifier_augmentation import (
        PERSPECTIVE_EVALUATION_CASES,
        GeometryCase,
        apply_evaluation_case,
    )
    from tools.recognition.run_perspective_classifier_experiment import (
        load_condition_checkpoint,
    )
    from tools.recognition.run_perspective_classifier_mixture_experiment import (
        preletterbox_content_extent,
    )
    from tools.recognition.run_rotation_classifier_experiment import load_cache


DEFAULT_LABELS = ("5m", "6m", "7m")
DEFAULT_CASES = ("front-facing", "recrop-yaw-left-0p10-angle20")
DEFAULT_SPLIT = "manual_val"
DEFAULT_TOP_CHANNELS = 16
DEFAULT_SOURCE_EXAMPLES = 8
DEFAULT_CONTACT_TILE = 28
DEFAULT_CONTACT_COLUMNS = 12


@dataclass(frozen=True)
class SelectedSamples:
    split: str
    labels: tuple[str, ...]
    images_u8: np.ndarray
    class_indices: np.ndarray
    sample_ids: tuple[str, ...]
    label_positions: dict[str, np.ndarray]
    content_extent_x: np.ndarray
    content_extent_y: np.ndarray

    @property
    def count(self) -> int:
        return int(self.images_u8.shape[0])


@dataclass(frozen=True)
class StageSpec:
    name: str
    module: nn.Module
    description: str


@dataclass
class CaseModelResult:
    architecture: str
    case_name: str
    transformed_u8: np.ndarray
    logits: np.ndarray
    predictions: np.ndarray
    activations: dict[str, np.ndarray]
    stage_descriptions: dict[str, str]


def parse_args() -> argparse.Namespace:
    repository_root = Path(__file__).resolve().parents[2]
    mixture_root = (
        repository_root
        / ".local"
        / "recognition"
        / "perspective_classifier_mixture_experiment"
    )
    default_output = (
        repository_root
        / ".local"
        / "recognition"
        / "classifier_activation_comparison"
        / "plain-vs-f8-heavy.html"
    )
    parser = argparse.ArgumentParser(
        description=(
            "Visualize 5m/6m/7m internal activations for Plain-heavy and f8-r1-heavy."
        )
    )
    parser.add_argument(
        "--database",
        type=Path,
        default=(
            repository_root
            / ".local"
            / "recognition"
            / "tile_classifier_datasets"
            / "gray35_jp500_seed42_v3_jp189.sqlite"
        ),
    )
    parser.add_argument(
        "--plain-checkpoint",
        type=Path,
        default=mixture_root / "plain-mix-heavy" / "training" / "best.pt",
    )
    parser.add_argument(
        "--f8-checkpoint",
        type=Path,
        default=mixture_root / "f8-r1-mix-heavy" / "training" / "best.pt",
    )
    parser.add_argument("--output", type=Path, default=default_output)
    parser.add_argument("--summary-output", type=Path)
    parser.add_argument("--split", default=DEFAULT_SPLIT)
    parser.add_argument("--labels", nargs="+", default=list(DEFAULT_LABELS))
    parser.add_argument(
        "--cases",
        nargs="+",
        default=list(DEFAULT_CASES),
        choices=[case.name for case in PERSPECTIVE_EVALUATION_CASES],
    )
    parser.add_argument(
        "--samples-per-label",
        type=int,
        default=0,
        help="0 uses every selected split sample for each label; positive values subsample deterministically.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--top-channels", type=int, default=DEFAULT_TOP_CHANNELS)
    parser.add_argument("--source-examples", type=int, default=DEFAULT_SOURCE_EXAMPLES)
    parser.add_argument("--contact-tile-size", type=int, default=DEFAULT_CONTACT_TILE)
    parser.add_argument("--contact-columns", type=int, default=DEFAULT_CONTACT_COLUMNS)
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    validate_args(args)
    database = args.database.resolve()
    plain_checkpoint = args.plain_checkpoint.resolve()
    f8_checkpoint = args.f8_checkpoint.resolve()
    output = args.output.resolve()
    summary_output = (
        args.summary_output.resolve()
        if args.summary_output is not None
        else output.with_suffix(".json")
    )

    if not plain_checkpoint.is_file():
        raise FileNotFoundError(plain_checkpoint)
    if not f8_checkpoint.is_file():
        raise FileNotFoundError(f8_checkpoint)

    device = resolve_device(str(args.device))
    cache = load_cache(database, cache_device="cpu")
    labels = tuple(str(label) for label in args.labels)
    samples = select_samples(
        database,
        cache=cache,
        split_name=str(args.split),
        labels=labels,
        samples_per_label=int(args.samples_per_label),
        seed=int(args.seed),
    )
    cases = resolve_cases(tuple(str(name) for name in args.cases))

    models = {
        "plain": load_condition_checkpoint(
            plain_checkpoint,
            architecture="plain",
            class_count=len(cache.class_labels),
            image_size=cache.image_size,
            device=device,
        ),
        "f8-r1": load_condition_checkpoint(
            f8_checkpoint,
            architecture="f8-r1",
            class_count=len(cache.class_labels),
            image_size=cache.image_size,
            device=device,
        ),
    }
    stage_specs = {
        "plain": plain_stage_specs(models["plain"]),
        "f8-r1": f8_stage_specs(models["f8-r1"], image_size=cache.image_size, device=device),
    }

    class_label_array = tuple(cache.class_labels)
    results: dict[tuple[str, str], CaseModelResult] = {}
    for case in cases:
        transformed = transform_samples_for_case(samples, case=case, device=device)
        for architecture, model in models.items():
            result = capture_case_model(
                model,
                architecture=architecture,
                case=case,
                transformed=transformed,
                stages=stage_specs[architecture],
                mean=cache.mean,
                std=cache.std,
                device=device,
            )
            results[(case.name, architecture)] = result

    report, summary = build_report(
        results,
        samples=samples,
        cases=cases,
        class_labels=class_label_array,
        checkpoints={
            "plain": str(plain_checkpoint),
            "f8-r1": str(f8_checkpoint),
        },
        database=str(database),
        mean=cache.mean,
        std=cache.std,
        top_channels=int(args.top_channels),
        source_examples=int(args.source_examples),
        contact_tile_size=int(args.contact_tile_size),
        contact_columns=int(args.contact_columns),
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(report, encoding="utf-8")
    summary_output.parent.mkdir(parents=True, exist_ok=True)
    summary_output.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"device={device}")
    print(f"samples={samples.count} labels={labels} split={samples.split}")
    print(f"html={output}")
    print(f"summary={summary_output}")


def validate_args(args: argparse.Namespace) -> None:
    if args.samples_per_label < 0:
        raise ValueError("--samples-per-label must be >= 0")
    if args.top_channels < 1:
        raise ValueError("--top-channels must be positive")
    if args.source_examples < 1:
        raise ValueError("--source-examples must be positive")
    if args.contact_tile_size < 12:
        raise ValueError("--contact-tile-size must be >= 12")
    if args.contact_columns < 1:
        raise ValueError("--contact-columns must be positive")
    labels = tuple(str(label) for label in args.labels)
    if labels != DEFAULT_LABELS:
        raise ValueError(
            "This diagnostic currently ranks 6m against 5m/7m; use exactly --labels 5m 6m 7m"
        )


def resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if value == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but CUDA is unavailable")
    return torch.device(value)


def resolve_cases(names: Sequence[str]) -> tuple[GeometryCase, ...]:
    by_name = {case.name: case for case in PERSPECTIVE_EVALUATION_CASES}
    return tuple(by_name[name] for name in names)


def select_samples(
    database: Path,
    *,
    cache: Any,
    split_name: str,
    labels: tuple[str, ...],
    samples_per_label: int,
    seed: int,
) -> SelectedSamples:
    if split_name not in cache.splits:
        raise ValueError(f"Unknown split {split_name!r}; available={tuple(cache.splits)}")
    label_to_index = {label: index for index, label in enumerate(cache.class_labels)}
    missing_labels = [label for label in labels if label not in label_to_index]
    if missing_labels:
        raise ValueError(f"Unknown labels: {missing_labels}")

    split = cache.splits[split_name]
    split_labels = split.labels.detach().cpu().numpy()
    selected_indices: list[int] = []
    label_positions: dict[str, np.ndarray] = {}
    position_cursor = 0
    for label in labels:
        class_index = label_to_index[label]
        candidates = np.flatnonzero(split_labels == class_index).astype(np.int64)
        if candidates.size == 0:
            raise ValueError(f"No samples for split={split_name!r}, label={label!r}")
        if samples_per_label > 0 and candidates.size > samples_per_label:
            ranked = sorted(
                candidates.tolist(),
                key=lambda index: stable_rank(split.sample_ids[index], seed=seed),
            )[:samples_per_label]
            candidates = np.asarray(ranked, dtype=np.int64)
        else:
            candidates = np.asarray(
                sorted(candidates.tolist(), key=lambda index: split.sample_ids[index]),
                dtype=np.int64,
            )
        selected_indices.extend(int(index) for index in candidates)
        label_positions[label] = np.arange(
            position_cursor,
            position_cursor + len(candidates),
            dtype=np.int64,
        )
        position_cursor += len(candidates)

    index_tensor = torch.tensor(selected_indices, dtype=torch.long)
    images_u8 = split.images_u8.index_select(0, index_tensor).detach().cpu().numpy().copy()
    class_indices = split.labels.index_select(0, index_tensor).detach().cpu().numpy().copy()
    sample_ids = tuple(split.sample_ids[index] for index in selected_indices)
    dimensions = load_original_dimensions(database, sample_ids)
    extents = np.asarray(
        [
            preletterbox_content_extent(
                dimensions[sample_id][0],
                dimensions[sample_id][1],
                image_size=int(cache.image_size),
            )
            for sample_id in sample_ids
        ],
        dtype=np.float32,
    )
    return SelectedSamples(
        split=split_name,
        labels=labels,
        images_u8=images_u8,
        class_indices=class_indices,
        sample_ids=sample_ids,
        label_positions=label_positions,
        content_extent_x=extents[:, 0],
        content_extent_y=extents[:, 1],
    )


def stable_rank(sample_id: str, *, seed: int) -> bytes:
    return hashlib.sha256(f"{seed}\0activation-gallery\0{sample_id}".encode("utf-8")).digest()


def load_original_dimensions(
    database: Path,
    sample_ids: Sequence[str],
) -> dict[str, tuple[int, int]]:
    connection = sqlite3.connect(f"{database.resolve().as_uri()}?mode=ro", uri=True, timeout=60)
    connection.row_factory = sqlite3.Row
    try:
        placeholders = ",".join("?" for _ in sample_ids)
        rows = connection.execute(
            f"""
            SELECT sample_id, original_width, original_height
            FROM sample
            WHERE sample_id IN ({placeholders})
            """,
            tuple(sample_ids),
        ).fetchall()
    finally:
        connection.close()
    result = {
        str(row["sample_id"]): (int(row["original_width"]), int(row["original_height"]))
        for row in rows
    }
    missing = [sample_id for sample_id in sample_ids if sample_id not in result]
    if missing:
        raise ValueError(f"Missing original dimensions for {missing[:5]}")
    return result


def plain_stage_specs(model: nn.Module) -> tuple[StageSpec, ...]:
    features = getattr(model, "features", None)
    if not isinstance(features, nn.Sequential):
        raise TypeError("Plain model has no Sequential features")
    pools = [
        (index, module)
        for index, module in enumerate(features)
        if isinstance(module, nn.MaxPool2d)
    ]
    if len(pools) != 3:
        raise ValueError(f"Expected three Plain MaxPool stages; found {len(pools)}")
    result = [
        StageSpec(
            name=f"pool{ordinal}",
            module=module,
            description=f"Plain MaxPool{ordinal} output after Conv/BN/SiLU",
        )
        for ordinal, (_, module) in enumerate(pools, start=1)
    ]
    result.append(
        StageSpec(
            name="conv4",
            module=features[-1],
            description="Plain final Conv4/BN/SiLU output before GlobalAvgPool",
        )
    )
    return tuple(result)


def f8_stage_specs(
    model: nn.Module,
    *,
    image_size: int,
    device: torch.device,
) -> tuple[StageSpec, ...]:
    features = getattr(model, "features", None)
    if not isinstance(features, nn.Sequential):
        raise TypeError("f8-r1 model has no Sequential features")
    shapes: list[tuple[int, int, int, int]] = []
    value = torch.zeros((1, 1, image_size, image_size), dtype=torch.float32, device=device)
    with torch.inference_mode():
        for module in features:
            value = module(value)
            if value.ndim != 4:
                raise ValueError("f8-r1 top-level feature module returned a non-NCHW tensor")
            shapes.append(tuple(int(v) for v in value.shape))

    spatial = [shape[-1] for shape in shapes]
    index32 = first_index(spatial, 32)
    index16 = first_index(spatial, 16)
    index8 = first_index(spatial, 8)
    final_index = len(features) - 1
    late_index = final_index - 1
    if spatial[final_index] != 8 or spatial[late_index] != 8:
        raise ValueError(f"Unexpected f8-r1 final spatial schedule: {spatial}")

    selections = (
        ("stem32", index32, "f8-r1 stem output at 32x32"),
        ("stage16", index16, "f8-r1 first 16x16 block output"),
        ("stage8-early", index8, "f8-r1 first 8x8 block output"),
        ("stage8-late", late_index, "f8-r1 terminal 8x8 x 96 block output"),
        ("stage8-final", final_index, "f8-r1 final 1x1-expanded 8x8 x 576 feature map before GAP"),
    )
    seen: set[int] = set()
    result: list[StageSpec] = []
    for name, index, description in selections:
        if index in seen:
            continue
        seen.add(index)
        shape = shapes[index]
        result.append(
            StageSpec(
                name=name,
                module=features[index],
                description=f"{description}; output={shape[1]}x{shape[2]}x{shape[3]}",
            )
        )
    return tuple(result)


def first_index(values: Sequence[int], target: int) -> int:
    try:
        return list(values).index(target)
    except ValueError as error:
        raise ValueError(f"Spatial resolution {target} missing from schedule {list(values)}") from error


def transform_samples_for_case(
    samples: SelectedSamples,
    *,
    case: GeometryCase,
    device: torch.device,
) -> torch.Tensor:
    images = torch.from_numpy(samples.images_u8).to(device=device, dtype=torch.float32)
    images = images.unsqueeze(1).mul_(1.0 / 255.0)
    extent_x = torch.from_numpy(samples.content_extent_x).to(device=device, dtype=torch.float32)
    extent_y = torch.from_numpy(samples.content_extent_y).to(device=device, dtype=torch.float32)
    with torch.inference_mode():
        return apply_evaluation_case(
            images,
            case,
            content_extent_x=extent_x,
            content_extent_y=extent_y,
        )


def capture_case_model(
    model: nn.Module,
    *,
    architecture: str,
    case: GeometryCase,
    transformed: torch.Tensor,
    stages: Sequence[StageSpec],
    mean: float,
    std: float,
    device: torch.device,
) -> CaseModelResult:
    captured: dict[str, torch.Tensor] = {}
    handles: list[Any] = []

    def make_hook(name: str):
        def hook(_module: nn.Module, _inputs: tuple[Any, ...], output: Any) -> None:
            if not isinstance(output, torch.Tensor) or output.ndim != 4:
                raise TypeError(f"Stage {name} did not produce an NCHW tensor")
            captured[name] = output.detach().cpu()

        return hook

    for stage in stages:
        handles.append(stage.module.register_forward_hook(make_hook(stage.name)))
    try:
        normalized = transformed.sub(mean).div(std)
        with torch.inference_mode():
            logits = model(normalized)
    finally:
        for handle in handles:
            handle.remove()

    missing = [stage.name for stage in stages if stage.name not in captured]
    if missing:
        raise RuntimeError(f"Missing captured stages for {architecture}: {missing}")
    transformed_u8 = (
        transformed.detach()
        .clamp(0.0, 1.0)
        .mul(255.0)
        .round()
        .to(dtype=torch.uint8)
        .squeeze(1)
        .cpu()
        .numpy()
    )
    return CaseModelResult(
        architecture=architecture,
        case_name=case.name,
        transformed_u8=transformed_u8,
        logits=logits.detach().cpu().numpy().astype(np.float32),
        predictions=logits.argmax(dim=1).detach().cpu().numpy().astype(np.int64),
        activations={name: value.numpy().astype(np.float32) for name, value in captured.items()},
        stage_descriptions={stage.name: stage.description for stage in stages},
    )


def build_report(
    results: dict[tuple[str, str], CaseModelResult],
    *,
    samples: SelectedSamples,
    cases: Sequence[GeometryCase],
    class_labels: Sequence[str],
    checkpoints: dict[str, str],
    database: str,
    mean: float,
    std: float,
    top_channels: int,
    source_examples: int,
    contact_tile_size: int,
    contact_columns: int,
) -> tuple[str, dict[str, Any]]:
    class_labels = tuple(str(label) for label in class_labels)
    label_to_index = {label: index for index, label in enumerate(class_labels)}
    summary: dict[str, Any] = {
        "database": database,
        "split": samples.split,
        "labels": list(samples.labels),
        "sample_counts": {
            label: int(len(samples.label_positions[label])) for label in samples.labels
        },
        "normalization": {"mean": mean, "std": std},
        "checkpoints": checkpoints,
        "cases": {},
    }

    body: list[str] = []
    body.append("<h1>Plain-heavy vs f8-r1-heavy: 5m / 6m / 7m internal activations</h1>")
    body.append(
        "<p class='lead'>Each stage is shown using class-mean activation maps. "
        "The detailed channel table is ranked by how strongly 6m differs from the "
        "average of 5m and 7m after spatially pooling that channel. Channel IDs are "
        "model-local; ch42 in Plain has no correspondence to ch42 in f8-r1.</p>"
    )
    body.append("<div class='meta'>")
    body.append(f"<div><b>DB</b>: {html.escape(database)}</div>")
    body.append(f"<div><b>split</b>: {html.escape(samples.split)}</div>")
    body.append(
        "<div><b>samples</b>: "
        + ", ".join(
            f"{label}={len(samples.label_positions[label])}" for label in samples.labels
        )
        + "</div>"
    )
    body.append(f"<div><b>normalization</b>: mean={mean:.12f}, std={std:.12f}</div>")
    body.append("</div>")

    for case in cases:
        body.append(f"<section class='case'><h2>{html.escape(case.name)}</h2>")
        body.append(f"<pre>{html.escape(json.dumps(asdict(case), indent=2))}</pre>")
        summary["cases"][case.name] = {}

        # The transformed model input is architecture-independent, so use Plain's copy.
        source_result = results[(case.name, "plain")]
        body.append("<h3>Model inputs</h3><div class='source-grid'>")
        for label in samples.labels:
            positions = samples.label_positions[label]
            chosen = positions[: min(source_examples, len(positions))]
            sheet = source_example_sheet(
                source_result.transformed_u8[chosen],
                sample_ids=[samples.sample_ids[int(index)] for index in chosen],
                label=label,
            )
            body.append(
                f"<div><h4>{html.escape(label)}</h4><img class='source-sheet' src='{png_data_uri(sheet)}'></div>"
            )
        body.append("</div>")

        for architecture in ("plain", "f8-r1"):
            result = results[(case.name, architecture)]
            prediction_summary = summarize_predictions(
                result.predictions,
                samples=samples,
                class_labels=class_labels,
            )
            stage_summary: dict[str, Any] = {}
            summary["cases"][case.name][architecture] = {
                "predictions": prediction_summary,
                "stages": stage_summary,
            }

            body.append(
                f"<div class='model'><h3>{html.escape(architecture)}</h3>"
                f"<div class='checkpoint'>{html.escape(checkpoints[architecture])}</div>"
            )
            body.append(prediction_table_html(prediction_summary, samples.labels))

            focus_logits = {
                label: label_to_index[label] for label in samples.labels
            }
            body.append(
                logits_table_html(
                    result.logits,
                    samples=samples,
                    focus_logits=focus_logits,
                )
            )

            for stage_name, activation in result.activations.items():
                if activation.ndim != 4:
                    raise ValueError(f"Unexpected activation rank for {stage_name}: {activation.shape}")
                stage_stats = compute_stage_statistics(
                    activation,
                    samples=samples,
                    top_channels=top_channels,
                )
                stage_summary[stage_name] = {
                    "description": result.stage_descriptions[stage_name],
                    "shape": list(activation.shape),
                    "top_channels": stage_stats["top_channels"],
                }
                body.append(
                    f"<article class='stage'><h4>{html.escape(stage_name)} — "
                    f"{activation.shape[1]} channels, {activation.shape[2]}x{activation.shape[3]}</h4>"
                    f"<p>{html.escape(result.stage_descriptions[stage_name])}</p>"
                )
                body.append("<h5>Top 6m-vs-(5m,7m) channels</h5>")
                body.append(
                    top_channel_table_html(
                        activation,
                        samples=samples,
                        stats=stage_stats,
                    )
                )
                body.append("<details><summary>Show all channels as contact sheets</summary>")
                sheets = all_channel_contact_sheets(
                    activation,
                    samples=samples,
                    tile_size=contact_tile_size,
                    columns=contact_columns,
                )
                body.append("<div class='contact-grid'>")
                for label in samples.labels:
                    body.append(
                        f"<div><h5>{html.escape(label)}</h5>"
                        f"<img class='contact-sheet' src='{png_data_uri(sheets[label])}'></div>"
                    )
                body.append("</div></details></article>")
            body.append("</div>")
        body.append("</section>")

    document = """<!doctype html>
<html><head><meta charset="utf-8"><title>Classifier activation comparison</title>
<style>
body{font-family:system-ui,-apple-system,sans-serif;margin:24px;background:#111;color:#eee;line-height:1.45}
h1,h2,h3,h4,h5{color:#fff}.lead{max-width:1100px}.meta,.model,.stage,.case{border:1px solid #444;border-radius:10px;padding:14px;margin:14px 0;background:#181818}
.case{background:#141414}.model{background:#1c1c1c}.stage{background:#202020}.checkpoint{font-family:monospace;font-size:12px;word-break:break-all;color:#bbb}
pre{overflow:auto;background:#0c0c0c;padding:10px;border-radius:6px}table{border-collapse:collapse;margin:10px 0;max-width:100%}th,td{border:1px solid #555;padding:5px 8px;text-align:right}th:first-child,td:first-child{text-align:left}
.source-grid,.contact-grid{display:flex;gap:16px;flex-wrap:wrap;align-items:flex-start}.source-sheet{max-width:620px;height:auto;background:#000}.contact-sheet{max-width:min(100%,900px);height:auto;background:#000;image-rendering:pixelated}
.channel-map{width:96px;height:96px;image-rendering:pixelated;background:#000}.channel-table td.map{text-align:center}.positive{color:#8f8}.negative{color:#f99}summary{cursor:pointer;font-weight:700;margin:8px 0}.small{font-size:12px;color:#bbb}
</style></head><body>""" + "\n".join(body) + "</body></html>\n"
    return document, summary


def summarize_predictions(
    predictions: np.ndarray,
    *,
    samples: SelectedSamples,
    class_labels: Sequence[str],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for label in samples.labels:
        positions = samples.label_positions[label]
        observed = predictions[positions]
        counts: dict[str, int] = {}
        for class_index, count in zip(*np.unique(observed, return_counts=True)):
            counts[str(class_labels[int(class_index)])] = int(count)
        expected_index = int(samples.class_indices[int(positions[0])])
        correct = int(np.count_nonzero(observed == expected_index))
        result[label] = {
            "count": int(len(positions)),
            "correct": correct,
            "accuracy": correct / max(len(positions), 1),
            "prediction_counts": counts,
        }
    return result


def prediction_table_html(summary: dict[str, Any], labels: Sequence[str]) -> str:
    rows = ["<h4>Predictions</h4><table><tr><th>true</th><th>accuracy</th><th>prediction counts</th></tr>"]
    for label in labels:
        row = summary[label]
        counts = ", ".join(
            f"{name}:{count}" for name, count in sorted(row["prediction_counts"].items())
        )
        rows.append(
            f"<tr><td>{html.escape(label)}</td><td>{row['accuracy']:.3f}</td>"
            f"<td>{html.escape(counts)}</td></tr>"
        )
    rows.append("</table>")
    return "".join(rows)


def logits_table_html(
    logits: np.ndarray,
    *,
    samples: SelectedSamples,
    focus_logits: dict[str, int],
) -> str:
    rows = [
        "<h4>Mean logits for 5m / 6m / 7m</h4><table><tr><th>true</th>"
        + "".join(f"<th>{html.escape(label)}</th>" for label in samples.labels)
        + "</tr>"
    ]
    for true_label in samples.labels:
        positions = samples.label_positions[true_label]
        mean_logits = logits[positions].mean(axis=0)
        rows.append(f"<tr><td>{html.escape(true_label)}</td>")
        for predicted_label in samples.labels:
            rows.append(f"<td>{float(mean_logits[focus_logits[predicted_label]]):.4f}</td>")
        rows.append("</tr>")
    rows.append("</table>")
    return "".join(rows)


def compute_stage_statistics(
    activation: np.ndarray,
    *,
    samples: SelectedSamples,
    top_channels: int,
) -> dict[str, Any]:
    # Per-sample scalar activation keeps the channel ranking independent of map size.
    pooled = activation.mean(axis=(2, 3), dtype=np.float64)
    means: dict[str, np.ndarray] = {}
    variances: dict[str, np.ndarray] = {}
    for label in samples.labels:
        values = pooled[samples.label_positions[label]]
        means[label] = values.mean(axis=0)
        variances[label] = values.var(axis=0)
    neighbor_mean = (means["5m"] + means["7m"]) * 0.5
    signed = means["6m"] - neighbor_mean
    pooled_std = np.sqrt(
        (variances["5m"] + variances["6m"] + variances["7m"]) / 3.0
        + 1.0e-12
    )
    effect = np.abs(signed) / np.maximum(pooled_std, 1.0e-6)
    # A nearly constant channel can otherwise receive a huge effect score from tiny
    # floating-point differences. Require a material absolute class separation too.
    absolute_range = np.maximum.reduce(
        [means["5m"], means["6m"], means["7m"]]
    ) - np.minimum.reduce([means["5m"], means["6m"], means["7m"]])
    scale = np.maximum(
        np.mean(np.abs(pooled), axis=0),
        1.0e-4,
    )
    normalized_range = absolute_range / scale
    ranking_score = effect * np.minimum(normalized_range, 10.0)
    order = np.argsort(-ranking_score)
    limit = min(int(top_channels), int(activation.shape[1]))
    top: list[dict[str, Any]] = []
    for channel in order[:limit]:
        channel = int(channel)
        top.append(
            {
                "channel": channel,
                "ranking_score": float(ranking_score[channel]),
                "effect_size": float(effect[channel]),
                "signed_6m_vs_neighbors": float(signed[channel]),
                "mean_5m": float(means["5m"][channel]),
                "mean_6m": float(means["6m"][channel]),
                "mean_7m": float(means["7m"][channel]),
            }
        )
    return {
        "top_channels": top,
        "class_means": means,
    }


def top_channel_table_html(
    activation: np.ndarray,
    *,
    samples: SelectedSamples,
    stats: dict[str, Any],
) -> str:
    class_mean_maps = {
        label: activation[samples.label_positions[label]].mean(axis=0)
        for label in samples.labels
    }
    rows = [
        "<table class='channel-table'><tr><th>channel</th><th>score</th>"
        "<th>6m-neighbor</th>"
        + "".join(f"<th>{html.escape(label)} map</th>" for label in samples.labels)
        + "</tr>"
    ]
    for row in stats["top_channels"]:
        channel = int(row["channel"])
        maps = [class_mean_maps[label][channel] for label in samples.labels]
        low = min(float(np.min(value)) for value in maps)
        high = max(float(np.max(value)) for value in maps)
        signed = float(row["signed_6m_vs_neighbors"])
        sign_class = "positive" if signed >= 0.0 else "negative"
        rows.append(
            f"<tr><td>ch{channel}</td><td>{row['ranking_score']:.3f}</td>"
            f"<td class='{sign_class}'>{signed:+.5f}</td>"
        )
        for value in maps:
            rendered = render_heatmap(value, low=low, high=high, output_size=96)
            rows.append(
                f"<td class='map'><img class='channel-map' src='{png_data_uri(rendered)}'></td>"
            )
        rows.append("</tr>")
    rows.append("</table>")
    rows.append(
        "<div class='small'>Positive 6m-neighbor means this channel is more active for 6m than the 5m/7m average; negative means less active. Maps in one row share the same value scale.</div>"
    )
    return "".join(rows)


def all_channel_contact_sheets(
    activation: np.ndarray,
    *,
    samples: SelectedSamples,
    tile_size: int,
    columns: int,
) -> dict[str, Image.Image]:
    class_mean_maps = {
        label: activation[samples.label_positions[label]].mean(axis=0)
        for label in samples.labels
    }
    channels = int(activation.shape[1])
    rows = int(math.ceil(channels / columns))
    label_height = 11
    sheets = {
        label: Image.new(
            "RGB",
            (columns * tile_size, rows * (tile_size + label_height)),
            (12, 12, 12),
        )
        for label in samples.labels
    }
    draws = {label: ImageDraw.Draw(image) for label, image in sheets.items()}
    for channel in range(channels):
        maps = [class_mean_maps[label][channel] for label in samples.labels]
        low = min(float(np.min(value)) for value in maps)
        high = max(float(np.max(value)) for value in maps)
        col = channel % columns
        row = channel // columns
        x = col * tile_size
        y = row * (tile_size + label_height)
        for label, value in zip(samples.labels, maps):
            heatmap = render_heatmap(
                value,
                low=low,
                high=high,
                output_size=tile_size,
            )
            sheets[label].paste(heatmap, (x, y))
            draws[label].text((x + 1, y + tile_size), str(channel), fill=(210, 210, 210))
    return sheets


def source_example_sheet(
    images_u8: np.ndarray,
    *,
    sample_ids: Sequence[str],
    label: str,
) -> Image.Image:
    tile = 96
    caption = 22
    columns = min(4, max(1, len(images_u8)))
    rows = int(math.ceil(len(images_u8) / columns))
    sheet = Image.new("RGB", (columns * tile, rows * (tile + caption)), (16, 16, 16))
    draw = ImageDraw.Draw(sheet)
    for index, (image_u8, sample_id) in enumerate(zip(images_u8, sample_ids)):
        row = index // columns
        col = index % columns
        x = col * tile
        y = row * (tile + caption)
        image = Image.fromarray(image_u8.astype(np.uint8), mode="L").resize(
            (tile, tile), resample=Image.Resampling.NEAREST
        ).convert("RGB")
        sheet.paste(image, (x, y))
        text = sample_id if len(sample_id) <= 15 else sample_id[:12] + "..."
        draw.text((x + 2, y + tile + 2), text, fill=(220, 220, 220))
    return sheet


def render_heatmap(
    value: np.ndarray,
    *,
    low: float,
    high: float,
    output_size: int,
) -> Image.Image:
    array = np.asarray(value, dtype=np.float32)
    if not np.isfinite(array).all():
        raise ValueError("Activation map contains NaN/inf")
    if high - low < 1.0e-12:
        normalized = np.zeros_like(array, dtype=np.float32)
    else:
        normalized = np.clip((array - low) / (high - low), 0.0, 1.0)
    rgb = simple_inferno(normalized)
    image = Image.fromarray(rgb, mode="RGB")
    return image.resize((output_size, output_size), resample=Image.Resampling.NEAREST)


def simple_inferno(normalized: np.ndarray) -> np.ndarray:
    """Small dependency-free dark->purple->orange->yellow activation colormap."""
    anchors = np.asarray(
        [
            [0.0, 0.0, 4.0],
            [45.0, 10.0, 80.0],
            [120.0, 28.0, 109.0],
            [200.0, 62.0, 70.0],
            [245.0, 140.0, 35.0],
            [252.0, 255.0, 164.0],
        ],
        dtype=np.float32,
    )
    scaled = np.clip(normalized, 0.0, 1.0) * (len(anchors) - 1)
    lower = np.floor(scaled).astype(np.int64)
    upper = np.minimum(lower + 1, len(anchors) - 1)
    fraction = (scaled - lower)[..., None]
    rgb = anchors[lower] * (1.0 - fraction) + anchors[upper] * fraction
    return np.clip(rgb, 0.0, 255.0).round().astype(np.uint8)


def png_data_uri(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG", optimize=True)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


if __name__ == "__main__":
    main()
