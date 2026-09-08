import numpy as np
import torch

from mldb.src.evaluation.interface import EvaluationResult
from mldb.src.model.loading import load_model
from tools.recognition.train_tile_shape_classifier import (
    angle_key,
    configure_cuda,
    evaluate_all,
    load_training_cache,
)


def evaluate(context):
    parameters = context.parameters
    configure_cuda(tf32=bool(parameters["tf32"]))
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for tile-shape evaluation")
    device = torch.device("cuda")
    cache = load_training_cache(
        context.corpus.artifact_path,
        device=device,
        cache_device=str(parameters["cache_device"]),
        cache_vram_fraction=float(parameters["cache_vram_fraction"]),
    )
    model = load_model(context.model).to(device)
    angles = tuple(float(value) for value in parameters["eval_angles"])
    validation = evaluate_all(
        model,
        cache,
        device=device,
        batch_size=int(parameters["batch_size"]),
        angles=angles,
        amp=bool(parameters["amp"]),
    )
    manual = validation["manual_val"]["angles"]
    jp = validation["jp_val"]["angles"]
    manual_values = [float(manual[angle_key(angle)]["accuracy"]) for angle in angles]
    jp_values = [float(jp[angle_key(angle)]["accuracy"]) for angle in angles]
    metrics = {
        "manual_accuracy_0deg": float(manual[angle_key(0.0)]["accuracy"]),
        "jp_accuracy_0deg": float(jp[angle_key(0.0)]["accuracy"]),
        "manual_angle_mean": float(np.mean(manual_values)),
        "jp_angle_mean": float(np.mean(jp_values)),
    }
    return EvaluationResult(metrics=metrics, artifacts={}, unavailable_outputs=())
