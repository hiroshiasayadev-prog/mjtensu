from __future__ import annotations

"""Human-readable diagnostic evaluation for 5m/6m/7m classifier confusion.

The goal is not another aggregate accuracy score.  This tool runs the exact ONNX
artifact used by the product and produces a report that lets a human follow:

1. what the model predicts on real frozen validation crops;
2. how confidence changes under the existing INV-013 view-geometry cases;
3. which image regions causally support the true-class logit margin (occlusion);
4. where 5m/6m/7m become more or less separable inside the ONNX feature stack.

The nine exemplar crops are selected deterministically as baseline-hard,
5m/6m/7m-neighbor-fragile under the tested view/crop perturbations, and view-stable
inside each class. They are explanatory examples; all numeric summary metrics are
computed over every matching manual_val sample.
"""

import argparse
import copy
import html
import json
import math
import sqlite3
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import onnx
import onnxruntime as ort
import torch
from PIL import Image, ImageDraw
from onnx import shape_inference

try:
    from perspective_classifier_augmentation import (
        PERSPECTIVE_EVALUATION_CASES,
        apply_evaluation_case,
    )
except ModuleNotFoundError:
    from tools.recognition.perspective_classifier_augmentation import (
        PERSPECTIVE_EVALUATION_CASES,
        apply_evaluation_case,
    )

TARGET_LABELS = ("5m", "6m", "7m")
DEFAULT_OCCLUSION_PATCH = 8
TAP_CANDIDATES = (
    ("stem-32x32", "/features/features.0/features.0.2/HardSwish_output_0"),
    ("early-16x16", "/features/features.1/block/block.2/block.2.0/Conv_output_0"),
    ("early-8x8", "/features/features.3/Add_output_0"),
    ("mid-8x8", "/features/features.6/Add_output_0"),
    ("late-8x8", "/features/features.8/Add_output_0"),
    ("prepool-8x8", "/features/features.10/features.10.2/HardSwish_output_0"),
    ("pooled-576", "/Flatten_output_0"),
    ("penultimate-1024", "/classifier/classifier.1/HardSwish_output_0"),
)


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--database",
        type=Path,
        default=root / ".local/recognition/tile_classifier_datasets/gray35_jp500_seed42_v3_jp189.sqlite",
    )
    parser.add_argument(
        "--onnx",
        type=Path,
        default=root / "vendor/recognition-models/mobile-tile-f8-r1.onnx",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=root / ".local/recognition/manzu_diagnostic_eval/mobile-tile-f8-r1",
    )
    parser.add_argument("--occlusion-patch", type=int, default=DEFAULT_OCCLUSION_PATCH)
    return parser.parse_args()


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def entropy(probabilities: np.ndarray) -> np.ndarray:
    clipped = np.clip(probabilities, 1.0e-12, 1.0)
    return -(clipped * np.log(clipped)).sum(axis=1)


def true_margin(logits: np.ndarray, targets: np.ndarray) -> np.ndarray:
    rows = np.arange(logits.shape[0])
    true_values = logits[rows, targets]
    masked = logits.copy()
    masked[rows, targets] = -np.inf
    return true_values - masked.max(axis=1)


def preletterbox_content_extent(width: int, height: int, image_size: int = 64) -> tuple[float, float]:
    scale = min(image_size / width, image_size / height)
    resized_width = max(1, min(image_size, int(math.floor(width * scale + 0.5))))
    resized_height = max(1, min(image_size, int(math.floor(height * scale + 0.5))))
    return resized_width / image_size, resized_height / image_size


def load_metadata(connection: sqlite3.Connection) -> tuple[list[str], float, float]:
    metadata = dict(connection.execute("SELECT key, value FROM experiment_metadata"))
    labels = [str(value) for value in json.loads(metadata["base_labels"])]
    pixel_sum = 0.0
    pixel_sq_sum = 0.0
    pixel_count = 0
    for (payload,) in connection.execute("SELECT image_gray_u8 FROM sample WHERE split='train'"):
        values = np.frombuffer(payload, dtype=np.uint8).astype(np.float64) / 255.0
        pixel_sum += float(values.sum())
        pixel_sq_sum += float(np.square(values).sum())
        pixel_count += int(values.size)
    mean = pixel_sum / pixel_count
    variance = max(pixel_sq_sum / pixel_count - mean * mean, 0.0)
    return labels, mean, max(math.sqrt(variance), 1.0 / 255.0)


def load_target_rows(connection: sqlite3.Connection, labels: Sequence[str]) -> list[dict[str, Any]]:
    placeholders = ",".join("?" for _ in labels)
    rows = connection.execute(
        f"""
        SELECT sample_id, base_label, class_index, image_gray_u8, original_width,
               original_height, source, capture_id, layout_id, region, source_image_path
        FROM sample
        WHERE split='manual_val' AND base_label IN ({placeholders})
        ORDER BY base_label, sample_id
        """,
        tuple(labels),
    ).fetchall()
    result: list[dict[str, Any]] = []
    for row in rows:
        result.append(
            {
                "sample_id": str(row[0]),
                "base_label": str(row[1]),
                "class_index": int(row[2]),
                "image": np.frombuffer(row[3], dtype=np.uint8).copy().reshape(64, 64),
                "original_width": int(row[4]),
                "original_height": int(row[5]),
                "source": str(row[6]),
                "capture_id": None if row[7] is None else str(row[7]),
                "layout_id": None if row[8] is None else str(row[8]),
                "region": None if row[9] is None else str(row[9]),
                "source_image_path": str(row[10]),
            }
        )
    return result


def normalize(images_01: np.ndarray, mean: float, std: float) -> np.ndarray:
    return ((images_01.astype(np.float32) - np.float32(mean)) / np.float32(std)).astype(np.float32)


def infer(session: ort.InferenceSession, images_01: np.ndarray, mean: float, std: float) -> np.ndarray:
    inputs = normalize(images_01, mean, std)[:, None, :, :]
    return np.asarray(session.run([session.get_outputs()[0].name], {session.get_inputs()[0].name: inputs})[0])


def topk(logits: np.ndarray, labels: Sequence[str], count: int = 5) -> list[list[dict[str, Any]]]:
    probabilities = softmax(logits)
    output: list[list[dict[str, Any]]] = []
    for row_logits, row_probabilities in zip(logits, probabilities, strict=True):
        order = np.argsort(row_logits)[::-1][:count]
        output.append(
            [
                {"label": labels[index], "logit": float(row_logits[index]), "probability": float(row_probabilities[index])}
                for index in order
            ]
        )
    return output


def transform_case(rows: Sequence[dict[str, Any]], case: Any) -> np.ndarray:
    images = np.stack([row["image"] for row in rows]).astype(np.float32)[:, None] / 255.0
    extent = [preletterbox_content_extent(row["original_width"], row["original_height"]) for row in rows]
    extent_x = torch.tensor([item[0] for item in extent], dtype=torch.float32)
    extent_y = torch.tensor([item[1] for item in extent], dtype=torch.float32)
    with torch.inference_mode():
        transformed = apply_evaluation_case(
            torch.from_numpy(images),
            case,
            content_extent_x=extent_x,
            content_extent_y=extent_y,
        )
    return transformed[:, 0].numpy().astype(np.float32)


def shift_image(image: np.ndarray, dx: int, dy: int) -> np.ndarray:
    result = np.empty_like(image)
    y_source = np.clip(np.arange(64) - dy, 0, 63)
    x_source = np.clip(np.arange(64) - dx, 0, 63)
    result[:] = image[np.ix_(y_source, x_source)]
    return result


def summarize_condition(
    logits: np.ndarray,
    rows: Sequence[dict[str, Any]],
    labels: Sequence[str],
) -> dict[str, Any]:
    targets = np.asarray([row["class_index"] for row in rows], dtype=np.int64)
    predictions = logits.argmax(axis=1)
    probabilities = softmax(logits)
    margins = true_margin(logits, targets)
    entropies = entropy(probabilities)
    confusion = {true: {pred: 0 for pred in TARGET_LABELS + ("other",)} for true in TARGET_LABELS}
    for row, prediction in zip(rows, predictions, strict=True):
        predicted_label = labels[int(prediction)]
        bucket = predicted_label if predicted_label in TARGET_LABELS else "other"
        confusion[row["base_label"]][bucket] += 1
    six_rows = np.asarray([row["base_label"] == "6m" for row in rows], dtype=bool)
    six_mix = 0
    if six_rows.any():
        predicted_labels = np.asarray([labels[int(value)] for value in predictions], dtype=object)
        six_mix = int(np.count_nonzero(six_rows & np.isin(predicted_labels, ["5m", "7m"])))
    return {
        "count": len(rows),
        "accuracy": float(np.mean(predictions == targets)),
        "mean_true_probability": float(np.mean(probabilities[np.arange(len(rows)), targets])),
        "mean_true_margin": float(np.mean(margins)),
        "mean_entropy_nats": float(np.mean(entropies)),
        "six_m_to_5m_or_7m_count": six_mix,
        "six_m_to_5m_or_7m_rate": float(six_mix / max(1, int(six_rows.sum()))),
        "confusion_5m_6m_7m_other": confusion,
    }


def select_representatives(
    rows: Sequence[dict[str, Any]],
    baseline_logits: np.ndarray,
    labels: Sequence[str],
    condition_logits: dict[str, np.ndarray],
) -> list[dict[str, Any]]:
    targets = np.asarray([row["class_index"] for row in rows], dtype=np.int64)
    baseline_margins = true_margin(baseline_logits, targets)
    all_margins = {name: true_margin(logits, targets) for name, logits in condition_logits.items()}
    selected: list[dict[str, Any]] = []
    for label in TARGET_LABELS:
        indices = [index for index, row in enumerate(rows) if row["base_label"] == label]
        hard = min(indices, key=lambda index: float(baseline_margins[index]))
        neighbor_events: list[tuple[float, int, str, str]] = []
        for case_name, logits in condition_logits.items():
            predictions = logits.argmax(axis=1)
            for index in indices:
                predicted = labels[int(predictions[index])]
                if predicted in TARGET_LABELS and predicted != label:
                    neighbor_events.append((float(all_margins[case_name][index]), index, case_name, predicted))
        neighbor_events.sort(key=lambda item: item[0])
        fragile_event = next((item for item in neighbor_events if item[1] != hard), None)
        fragile = fragile_event[1] if fragile_event is not None else hard
        stable = max(
            (index for index in indices if index not in {hard, fragile}),
            key=lambda index: min(float(margins[index]) for margins in all_margins.values()),
            default=max(indices, key=lambda index: float(baseline_margins[index])),
        )
        picks = [
            ("baseline-hard", hard, None),
            ("neighbor-fragile", fragile, fragile_event),
            ("view-stable", stable, None),
        ]
        seen: set[int] = set()
        for difficulty, index, event in picks:
            if index in seen:
                alternatives = [candidate for candidate in indices if candidate not in seen]
                index = alternatives[len(alternatives) // 2]
                event = None
            seen.add(index)
            item: dict[str, Any] = {
                "difficulty": difficulty,
                "row_index": index,
                "margin": float(baseline_margins[index]),
            }
            if event is not None:
                item["neighbor_failure"] = {"case": event[2], "prediction": event[3], "margin": event[0]}
            selected.append(item)
    return selected


def instrument_onnx(source: Path, destination: Path) -> list[tuple[str, str]]:
    inferred = shape_inference.infer_shapes(onnx.load(str(source)))
    value_infos = {value.name: value for value in inferred.graph.value_info}
    model = copy.deepcopy(inferred)
    present = {output.name for output in model.graph.output}
    taps: list[tuple[str, str]] = []
    for display_name, tensor_name in TAP_CANDIDATES:
        info = value_infos.get(tensor_name)
        if info is None:
            continue
        taps.append((display_name, tensor_name))
        if tensor_name not in present:
            model.graph.output.append(copy.deepcopy(info))
            present.add(tensor_name)
    onnx.checker.check_model(model)
    onnx.save(model, str(destination))
    return taps


def vectorize_feature(value: np.ndarray) -> np.ndarray:
    if value.ndim == 4:
        return value.mean(axis=(2, 3))
    if value.ndim == 2:
        return value
    return value.reshape(value.shape[0], -1)


def leave_one_out_centroid(feature: np.ndarray, rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    norms = np.linalg.norm(feature, axis=1, keepdims=True)
    normalized = feature / np.maximum(norms, 1.0e-12)
    labels = np.asarray([row["base_label"] for row in rows], dtype=object)
    predictions: list[str] = []
    margins: list[float] = []
    for index in range(len(rows)):
        similarities: dict[str, float] = {}
        for label in TARGET_LABELS:
            mask = labels == label
            if labels[index] == label:
                mask[index] = False
            members = normalized[mask]
            centroid = members.mean(axis=0)
            centroid /= max(float(np.linalg.norm(centroid)), 1.0e-12)
            similarities[label] = float(np.dot(normalized[index], centroid))
        ranked = sorted(similarities.items(), key=lambda item: item[1], reverse=True)
        predictions.append(ranked[0][0])
        true_similarity = similarities[str(labels[index])]
        competing = max(value for label, value in similarities.items() if label != labels[index])
        margins.append(true_similarity - competing)
    accuracy = float(np.mean(np.asarray(predictions, dtype=object) == labels))
    centroids: dict[str, np.ndarray] = {}
    for label in TARGET_LABELS:
        centroid = normalized[labels == label].mean(axis=0)
        centroids[label] = centroid / max(float(np.linalg.norm(centroid)), 1.0e-12)
    return {
        "loo_nearest_centroid_accuracy": accuracy,
        "mean_true_centroid_margin": float(np.mean(margins)),
        "centroid_cosine": {
            "5m_vs_6m": float(np.dot(centroids["5m"], centroids["6m"])),
            "6m_vs_7m": float(np.dot(centroids["6m"], centroids["7m"])),
            "5m_vs_7m": float(np.dot(centroids["5m"], centroids["7m"])),
        },
        "predictions": predictions,
        "margins": margins,
    }


def build_representation_report(
    debug_session: ort.InferenceSession,
    taps: Sequence[tuple[str, str]],
    rows: Sequence[dict[str, Any]],
    images_01: np.ndarray,
    mean: float,
    std: float,
) -> dict[str, Any]:
    input_name = debug_session.get_inputs()[0].name
    output_names = [tensor_name for _display_name, tensor_name in taps]
    values = debug_session.run(output_names, {input_name: normalize(images_01, mean, std)[:, None]})
    result: dict[str, Any] = {}
    for (display_name, _tensor_name), value in zip(taps, values, strict=True):
        feature = vectorize_feature(np.asarray(value, dtype=np.float32))
        result[display_name] = leave_one_out_centroid(feature, rows)
    return result


def occlusion_map(
    session: ort.InferenceSession,
    image_01: np.ndarray,
    target_index: int,
    mean: float,
    std: float,
    patch: int,
) -> np.ndarray:
    baseline = infer(session, image_01[None], mean, std)
    baseline_margin = float(true_margin(baseline, np.asarray([target_index]))[0])
    variants: list[np.ndarray] = []
    cells: list[tuple[int, int]] = []
    for y in range(0, 64, patch):
        for x in range(0, 64, patch):
            variant = image_01.copy()
            variant[y : y + patch, x : x + patch] = mean
            variants.append(variant)
            cells.append((y, x))
    logits = infer(session, np.stack(variants), mean, std)
    margins = true_margin(logits, np.full(len(variants), target_index, dtype=np.int64))
    heat = np.zeros((64, 64), dtype=np.float32)
    for (y, x), observed in zip(cells, margins, strict=True):
        heat[y : y + patch, x : x + patch] = baseline_margin - float(observed)
    return heat


def save_occlusion_overlay(image_u8: np.ndarray, heat: np.ndarray, path: Path) -> None:
    base = np.repeat(image_u8[:, :, None], 3, axis=2).astype(np.float32)
    scale = max(float(np.max(np.abs(heat))), 1.0e-6)
    strength = np.clip(np.abs(heat) / scale, 0.0, 1.0)[:, :, None]
    positive = np.zeros_like(base); positive[:, :, 0] = 255.0
    negative = np.zeros_like(base); negative[:, :, 2] = 255.0
    color = np.where((heat[:, :, None] >= 0), positive, negative)
    mixed = base * (1.0 - 0.55 * strength) + color * (0.55 * strength)
    image = Image.fromarray(np.clip(mixed, 0, 255).astype(np.uint8), mode="RGB").resize((256, 256), Image.Resampling.NEAREST)
    image.save(path)


def save_source_image(image_u8: np.ndarray, path: Path) -> None:
    Image.fromarray(image_u8, mode="L").resize((256, 256), Image.Resampling.NEAREST).save(path)


def make_contact_sheet(exemplars: Sequence[dict[str, Any]], output_root: Path) -> Path:
    cell_w, cell_h = 540, 320
    sheet = Image.new("RGB", (cell_w * 3, cell_h * 3), "white")
    draw = ImageDraw.Draw(sheet)
    for index, item in enumerate(exemplars):
        row, column = divmod(index, 3)
        x0, y0 = column * cell_w, row * cell_h
        source = Image.open(output_root / item["source_png"]).convert("RGB")
        occ = Image.open(output_root / item["occlusion_png"]).convert("RGB")
        sheet.paste(source.resize((220, 220)), (x0 + 10, y0 + 45))
        sheet.paste(occ.resize((220, 220)), (x0 + 245, y0 + 45))
        top = item["baseline_topk"][0]
        draw.text((x0 + 10, y0 + 10), f"{item['true_label']} {item['difficulty']} | pred={top['label']} p={top['probability']:.3f}", fill="black")
        draw.text((x0 + 245, y0 + 275), "red: hiding hurts true margin; blue: hiding helps", fill="black")
    path = output_root / "manzu-exemplars-contact-sheet.png"
    sheet.save(path)
    return path


def render_html(payload: dict[str, Any], path: Path) -> None:
    labels = payload["labels"]
    baseline = payload["conditions"]["front-facing"]
    rows = []
    for name, result in payload["conditions"].items():
        rows.append(
            f"<tr><td>{html.escape(name)}</td><td>{result['accuracy']:.3f}</td>"
            f"<td>{result['mean_true_margin']:.3f}</td><td>{result['mean_entropy_nats']:.3f}</td>"
            f"<td>{result['six_m_to_5m_or_7m_rate']:.3f}</td></tr>"
        )
    layer_rows = []
    for name, result in payload["representation"].items():
        cos = result["centroid_cosine"]
        layer_rows.append(
            f"<tr><td>{html.escape(name)}</td><td>{result['loo_nearest_centroid_accuracy']:.3f}</td>"
            f"<td>{result['mean_true_centroid_margin']:.3f}</td><td>{cos['5m_vs_6m']:.3f}</td>"
            f"<td>{cos['6m_vs_7m']:.3f}</td><td>{cos['5m_vs_7m']:.3f}</td></tr>"
        )
    cards = []
    for item in payload["exemplars"]:
        perturb_rows = "".join(
            f"<tr><td>{html.escape(row['case'])}</td><td>{row['prediction']}</td><td>{row['true_probability']:.3f}</td><td>{row['true_margin']:.3f}</td></tr>"
            for row in item["perturbations"]
        )
        trajectory = "".join(
            f"<tr><td>{html.escape(layer)}</td><td>{value['nearest_centroid']}</td><td>{value['true_centroid_margin']:.3f}</td></tr>"
            for layer, value in item["representation_trajectory"].items()
        )
        top = ", ".join(f"{row['label']} {row['probability']:.2f}" for row in item["baseline_topk"][:3])
        cards.append(
            f"<section class='card'><h3>{item['true_label']} / {item['difficulty']}</h3>"
            f"<p><code>{html.escape(item['sample_id'])}</code><br>baseline: {html.escape(top)}</p>"
            f"<div class='images'><figure><img src='{item['source_png']}'><figcaption>actual manual_val crop</figcaption></figure>"
            f"<figure><img src='{item['occlusion_png']}'><figcaption>occlusion sensitivity</figcaption></figure></div>"
            f"<h4>Geometry / crop perturbation</h4><table><tr><th>case</th><th>top1</th><th>P(true)</th><th>true margin</th></tr>{perturb_rows}</table>"
            f"<h4>Inside representation</h4><table><tr><th>tap</th><th>nearest 5/6/7 centroid</th><th>true-centroid margin</th></tr>{trajectory}</table></section>"
        )
    document = f"""<!doctype html><meta charset='utf-8'><title>5m/6m/7m diagnostic</title>
<style>body{{font:15px system-ui;margin:30px;max-width:1500px}}table{{border-collapse:collapse}}td,th{{border:1px solid #bbb;padding:5px 8px;text-align:right}}td:first-child,th:first-child{{text-align:left}}.cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(520px,1fr));gap:18px}}.card{{border:1px solid #aaa;border-radius:8px;padding:14px}}.images{{display:flex;gap:12px}}figure{{margin:0}}img{{width:256px;image-rendering:pixelated}}code{{font-size:12px}}.note{{background:#f3f3f3;padding:12px;border-radius:6px}}</style>
<h1>5m / 6m / 7m diagnostic — production ONNX</h1>
<div class='note'><b>How to read this:</b> accuracy is the final answer. True margin is logit(true) minus the strongest competing logit; below 0 means the model prefers another class. Entropy rises when probability is spread across alternatives. Occlusion is causal: red means hiding that patch reduces the true-class margin, blue means hiding it improves the true-class margin. Intermediate centroid analysis is diagnostic, not proof of causality: it asks whether 5m/6m/7m are geometrically separable in each internal feature representation.</div>
<h2>1. All {baseline['count']} manual_val 5m/6m/7m samples</h2>
<table><tr><th>view condition</th><th>accuracy</th><th>mean true margin</th><th>entropy</th><th>6m→5m/7m</th></tr>{''.join(rows)}</table>
<h2>2. Where do 5m/6m/7m separate internally?</h2>
<p>Nearest-centroid accuracy is leave-one-out over every 5m/6m/7m sample. Lower centroid cosine means the class clusters are farther apart.</p>
<table><tr><th>tap</th><th>LOO centroid acc</th><th>true-centroid margin</th><th>cos 5-6</th><th>cos 6-7</th><th>cos 5-7</th></tr>{''.join(layer_rows)}</table>
<h2>3. Nine concrete crops</h2><p>For each class: one baseline-hard crop, one crop that actually flips toward another 5m/6m/7m class under the tested geometry when available, and one crop stable across the tested views. This makes the examples diagnostic rather than cherry-picked.</p>
<div class='cards'>{''.join(cards)}</div>
<p>Model: <code>{html.escape(payload['onnx'])}</code><br>Labels: {html.escape(', '.join(labels))}</p>"""
    path.write_text(document, encoding="utf-8")


def main() -> int:
    args = parse_args()
    database = args.database.resolve()
    onnx_path = args.onnx.resolve()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    exemplar_dir = output_root / "exemplars"
    exemplar_dir.mkdir(exist_ok=True)
    for stale in exemplar_dir.glob("*.png"):
        stale.unlink()

    connection = sqlite3.connect(database)
    labels, mean, std = load_metadata(connection)
    rows = load_target_rows(connection, TARGET_LABELS)
    connection.close()
    if {row["base_label"] for row in rows} != set(TARGET_LABELS):
        raise RuntimeError("manual_val does not contain all 5m/6m/7m labels")

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    front_images = np.stack([row["image"] for row in rows]).astype(np.float32) / 255.0
    front_logits = infer(session, front_images, mean, std)
    condition_logits: dict[str, np.ndarray] = {"front-facing": front_logits}
    condition_results: dict[str, Any] = {"front-facing": summarize_condition(front_logits, rows, labels)}
    transformed_cache: dict[str, np.ndarray] = {"front-facing": front_images}
    for case in PERSPECTIVE_EVALUATION_CASES:
        if case.name == "front-facing":
            continue
        images = transform_case(rows, case)
        transformed_cache[case.name] = images
        logits = infer(session, images, mean, std)
        condition_logits[case.name] = logits
        condition_results[case.name] = summarize_condition(logits, rows, labels)
    for name, dx, dy in (("shift-x-minus-2", -2, 0), ("shift-x-plus-2", 2, 0), ("shift-y-minus-2", 0, -2), ("shift-y-plus-2", 0, 2)):
        images = np.stack([shift_image(row["image"], dx, dy) for row in rows]).astype(np.float32) / 255.0
        transformed_cache[name] = images
        logits = infer(session, images, mean, std)
        condition_logits[name] = logits
        condition_results[name] = summarize_condition(logits, rows, labels)

    instrumented_path = output_root / "instrumented-model.onnx"
    taps = instrument_onnx(onnx_path, instrumented_path)
    debug_session = ort.InferenceSession(str(instrumented_path), providers=["CPUExecutionProvider"])
    representation = build_representation_report(debug_session, taps, rows, front_images, mean, std)

    representatives = select_representatives(rows, front_logits, labels, condition_logits)
    baseline_topk = topk(front_logits, labels)
    targets = np.asarray([row["class_index"] for row in rows], dtype=np.int64)
    exemplar_payload: list[dict[str, Any]] = []
    for pick in representatives:
        index = int(pick["row_index"])
        row = rows[index]
        safe = f"{row['base_label']}-{pick['difficulty']}-{row['sample_id'].replace(':','_').replace('/','_')}"
        source_rel = Path("exemplars") / f"{safe}.png"
        occ_rel = Path("exemplars") / f"{safe}-occlusion.png"
        save_source_image(row["image"], output_root / source_rel)
        heat = occlusion_map(session, front_images[index], int(targets[index]), mean, std, int(args.occlusion_patch))
        save_occlusion_overlay(row["image"], heat, output_root / occ_rel)

        perturbations: list[dict[str, Any]] = []
        for case_name, images in transformed_cache.items():
            logits = infer(session, images[index : index + 1], mean, std)
            probs = softmax(logits)[0]
            prediction = int(logits.argmax(axis=1)[0])
            margin = float(true_margin(logits, np.asarray([targets[index]]))[0])
            perturbations.append(
                {
                    "case": case_name,
                    "prediction": labels[prediction],
                    "true_probability": float(probs[targets[index]]),
                    "true_margin": margin,
                }
            )

        trajectory: dict[str, Any] = {}
        for layer, layer_result in representation.items():
            trajectory[layer] = {
                "nearest_centroid": layer_result["predictions"][index],
                "true_centroid_margin": float(layer_result["margins"][index]),
            }
        exemplar_payload.append(
            {
                "sample_id": row["sample_id"],
                "true_label": row["base_label"],
                "difficulty": pick["difficulty"],
                "baseline_true_margin": pick["margin"],
                "neighbor_failure": pick.get("neighbor_failure"),
                "baseline_topk": baseline_topk[index],
                "source": {key: row[key] for key in ("source", "capture_id", "layout_id", "region", "source_image_path")},
                "source_png": source_rel.as_posix(),
                "occlusion_png": occ_rel.as_posix(),
                "occlusion_positive_peak": float(np.max(heat)),
                "occlusion_negative_peak": float(np.min(heat)),
                "perturbations": perturbations,
                "representation_trajectory": trajectory,
            }
        )

    payload = {
        "schema": "mjtensu.recognition/manzu-diagnostic-eval/v1",
        "database": str(database),
        "onnx": str(onnx_path),
        "labels": labels,
        "target_labels": list(TARGET_LABELS),
        "normalization": {"mean": mean, "std": std},
        "selection": "per-class baseline-hard, neighbor-fragile under tested view/crop perturbations, and view-stable exemplar",
        "conditions": condition_results,
        "representation": representation,
        "exemplars": exemplar_payload,
        "notes": {
            "occlusion": "Positive/red means replacing the patch with the training-set mean reduces the true-class logit margin; this is a causal input intervention.",
            "representation": "Nearest-centroid analysis measures 5m/6m/7m separation at internal ONNX tensors. It is diagnostic geometry, not causal attribution.",
        },
    }
    report_json = output_root / "report.json"
    report_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    render_html(payload, output_root / "report.html")
    contact_sheet = make_contact_sheet(exemplar_payload, output_root)
    instrumented_path.unlink(missing_ok=True)
    print(json.dumps({"report": str(report_json), "html": str(output_root / 'report.html'), "contact_sheet": str(contact_sheet), "target_count": len(rows)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
