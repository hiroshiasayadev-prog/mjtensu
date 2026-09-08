from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sqlite3
import sys
from pathlib import Path
from textwrap import dedent

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from mldb.src.common.ids import (
    ArchitectureId,
    CorpusId,
    EvaluationProtocolId,
    StudyId,
    TaskId,
    TrainProtocolId,
)
from mldb.src.repository._local_filesystem import LocalFilesystem
from mldb.src.repository.layout import RepositoryLayout
from mldb.src.runtime.resolution import (
    resolve_architecture,
    resolve_corpus,
    resolve_evaluation_protocol,
    resolve_study,
    resolve_task,
    resolve_train_protocol,
)

TASK_ID = TaskId("tile-shape-classification-35-v1")
CORPUS_ID = CorpusId("gray35-jp500-seed42-v3-jp189-v1")
ARCHITECTURE_ID = ArchitectureId("tile-plain-gray35-v1")
TRAIN_PROTOCOL_ID = TrainProtocolId("tile-shape-train-gpu-v1")
EVALUATION_PROTOCOL_ID = EvaluationProtocolId("tile-shape-eval-angle-v1")
STUDY_ID = StudyId("tile-plain-smoke-v1")
LABELS = (
    "1m", "2m", "3m", "4m", "5m", "6m", "7m", "8m", "9m",
    "1p", "2p", "3p", "4p", "5p", "6p", "7p", "8p", "9p",
    "1s", "2s", "3s", "4s", "5s", "6s", "7s", "8s", "9s",
    "east", "south", "west", "north", "white", "green", "red", "invalid",
)

DEFAULT_SOURCE = Path(
    ".local/recognition/tile_classifier_datasets/gray35_jp500_seed42_v3_jp189.sqlite"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(dedent(value).lstrip(), encoding="utf-8", newline="\n")

def _ensure_corpus(source: Path, target: Path) -> tuple[str, int, dict[str, int]]:
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        print(f"materializing MLDB corpus once: {source} -> {target}")
        shutil.copy2(source, target)
        connection = sqlite3.connect(target)
        try:
            columns = {row[1] for row in connection.execute("PRAGMA table_info(sample)")}
            if "target" not in columns:
                connection.execute("ALTER TABLE sample ADD COLUMN target TEXT")
                connection.execute("UPDATE sample SET target = base_label")
                connection.commit()
        finally:
            connection.close()
    connection = sqlite3.connect(target)
    try:
        splits = {
            str(split): int(count)
            for split, count in connection.execute(
                "SELECT split, COUNT(*) FROM sample GROUP BY split"
            )
        }
    finally:
        connection.close()
    return _sha256(target), target.stat().st_size, splits


def _architecture_source() -> str:
    return '''
import torch
from torch import nn


class PlainTileShapeClassifier(nn.Module):
    def __init__(self, class_count: int = 35) -> None:
        super().__init__()
        channels = (32, 64, 128, 192)
        layers = []
        in_channels = 1
        for index, out_channels in enumerate(channels):
            layers.extend([
                nn.Conv2d(in_channels, out_channels, 5 if index == 0 else 3,
                          padding=2 if index == 0 else 1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.SiLU(inplace=True),
            ])
            if index < len(channels) - 1:
                layers.append(nn.MaxPool2d(2, 2))
            in_channels = out_channels
        self.features = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(nn.Flatten(), nn.Linear(192, 256),
                                        nn.SiLU(inplace=True), nn.Linear(256, class_count))

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.pool(self.features(images)))


def build() -> nn.Module:
    return PlainTileShapeClassifier(35)
'''

def _train_protocol_source() -> str:
    return '''
import torch

from mldb.src.runtime.executable_loader import load_architecture_build
from tools.recognition.train_tile_shape_classifier import (
    configure_cuda,
    load_training_cache,
    seed_everything,
    train_one_epoch,
)


def train(context):
    parameters = context.parameters
    seed_everything(int(context.seed))
    configure_cuda(tf32=bool(parameters["tf32"]))
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for tile-shape training")
    device = torch.device("cuda")
    cache = load_training_cache(
        context.corpus.artifact_path,
        device=device,
        cache_device=str(parameters["cache_device"]),
        cache_vram_fraction=float(parameters["cache_vram_fraction"]),
    )
    model = load_architecture_build(context.architecture)().to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(parameters["learning_rate"]),
        weight_decay=float(parameters["weight_decay"]),
    )
    epochs = int(parameters["epochs"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=float(parameters["learning_rate"]) * 0.05
    )
    scaler = torch.cuda.amp.GradScaler(enabled=bool(parameters["amp"]))
    for epoch in range(1, epochs + 1):
        train_one_epoch(
            model,
            cache.splits["train"],
            optimizer=optimizer,
            scaler=scaler,
            batch_size=int(parameters["batch_size"]),
            device=device,
            mean=cache.mean,
            std=cache.std,
            rotation_augment_deg=float(parameters["rotation_augment_deg"]),
            perspective_augment=float(parameters["perspective_augment"]),
            shear_augment=float(parameters["shear_augment"]),
            stretch_augment=float(parameters["stretch_augment"]),
            projective_augment_probability=float(parameters["projective_augment_probability"]),
            amp=bool(parameters["amp"]),
            epoch=epoch,
            seed=int(context.seed),
        )
        scheduler.step()
    return model
'''


def _evaluation_protocol_source() -> str:
    return '''
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
'''


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create the first real tile-shape MLDB definitions.")
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--source-database", type=Path, default=DEFAULT_SOURCE)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    source = args.source_database
    if not source.is_absolute():
        source = repo_root / source
    source = source.resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    try:
        source_identity = source.relative_to(repo_root).as_posix()
    except ValueError:
        source_identity = str(source)

    layout = RepositoryLayout(repo_root)
    filesystem = LocalFilesystem()
    corpus_path = layout.corpus_artifact_path(CORPUS_ID)
    corpus_sha, corpus_bytes, splits = _ensure_corpus(source, corpus_path)
    _write_json(
        layout.task_metadata_path(TASK_ID),
        {
            "schema": "mjtensu.mldb/task/v1",
            "id": str(TASK_ID),
            "name": "Mahjong tile shape classification (35 classes)",
            "problem_type": "multiclass-classification",
            "description": "Grayscale 64x64 tile-shape classification including invalid crops.",
            "input": {"semantic_unit": "single-tile-image"},
            "target": {"type": "categorical", "labels": list(LABELS)},
            "semantics": {"red_five_policy": "red fives are folded into base fives"},
            "scope": {"includes": ["mahjong tiles", "invalid crops"], "excludes": []},
        },
    )
    _write_text(
        layout.corpus_builder_path(CORPUS_ID),
        '''
def build():
    raise RuntimeError("This Corpus is materialized from the existing classifier dataset by tools/mldb/bootstrap_tile_shape.py")
''',
    )
    _write_json(
        layout.corpus_metadata_path(CORPUS_ID),
        {
            "schema": "mjtensu.mldb/corpus/v1",
            "id": str(CORPUS_ID),
            "task": str(TASK_ID),
            "artifact": {"format": "sqlite", "sha256": corpus_sha, "bytes": corpus_bytes},
            "data": {"schema": "mjtensu.mldb/image-classification-corpus/v1", "table": "sample"},
            "representation": {"kind": "image", "dtype": "uint8", "payload_column": "image_gray_u8", "shape": [64, 64]},
            "builder": {"parameters": {"source": source_identity}},
            "splits": splits,
            "description": "gray35 jp500 seed42 v3 jp189 classifier dataset",
            "origin": {"source_database": source_identity},
        },
    )
    architecture_path = layout.architecture_implementation_path(ARCHITECTURE_ID)
    _write_text(architecture_path, _architecture_source())
    architecture_sha = _sha256(architecture_path)
    _write_json(
        layout.architecture_metadata_path(ARCHITECTURE_ID),
        {
            "schema": "mjtensu.mldb/architecture/v1",
            "id": str(ARCHITECTURE_ID),
            "status": "sealed",
            "task": str(TASK_ID),
            "name": "Plain gray35 classifier",
            "family": "plain-cnn",
            "description": "Existing four-stage grayscale CNN baseline.",
            "implementation": {"framework": "pytorch", "entrypoint": "build", "sha256": architecture_sha},
            "interface": {"input": {"kind": "image", "shape": [1, 64, 64]}, "output": {"kind": "classification-logits"}},
            "structure": {"summary": "Conv32-64-128-192 with max-pooling and 256-unit head", "traits": ["grayscale", "baseline"]},
        },
    )

    train_path = layout.train_protocol_implementation_path(TRAIN_PROTOCOL_ID)
    _write_text(train_path, _train_protocol_source())
    train_sha = _sha256(train_path)
    _write_json(
        layout.train_protocol_metadata_path(TRAIN_PROTOCOL_ID),
        {
            "schema": "mjtensu.mldb/train-protocol/v1",
            "id": str(TRAIN_PROTOCOL_ID),
            "status": "sealed",
            "task": str(TASK_ID),
            "name": "Tile shape GPU training",
            "description": "Reuse the existing cached SQLite tile-classifier training loop.",
            "implementation": {"entrypoint": "train", "sha256": train_sha},
            "parameters": {
                "epochs": {"default": 50},
                "batch_size": {"default": 1024},
                "learning_rate": {"default": 0.001},
                "weight_decay": {"default": 0.0001},
                "rotation_augment_deg": {"default": 0.0},
                "perspective_augment": {"default": 0.0},
                "shear_augment": {"default": 0.0},
                "stretch_augment": {"default": 0.0},
                "projective_augment_probability": {"default": 0.0},
                "amp": {"default": True},
                "tf32": {"default": True},
                "cache_device": {"default": "auto"},
                "cache_vram_fraction": {"default": 0.5},
            },
        },
    )
    evaluation_path = layout.evaluation_protocol_implementation_path(EVALUATION_PROTOCOL_ID)
    _write_text(evaluation_path, _evaluation_protocol_source())
    evaluation_sha = _sha256(evaluation_path)
    _write_json(
        layout.evaluation_protocol_metadata_path(EVALUATION_PROTOCOL_ID),
        {
            "schema": "mjtensu.mldb/evaluation-protocol/v1",
            "id": str(EVALUATION_PROTOCOL_ID),
            "status": "sealed",
            "task": str(TASK_ID),
            "name": "Tile shape angle evaluation",
            "description": "Evaluate manual/jp validation accuracy across configured angles.",
            "implementation": {"entrypoint": "evaluate", "sha256": evaluation_sha},
            "parameters": {
                "batch_size": {"default": 1024},
                "eval_angles": {"default": [0.0, 15.0, 30.0, 45.0]},
                "amp": {"default": True},
                "tf32": {"default": True},
                "cache_device": {"default": "auto"},
                "cache_vram_fraction": {"default": 0.5},
            },
            "outputs": {
                "metrics": {
                    "manual_accuracy_0deg": {"type": "number"},
                    "jp_accuracy_0deg": {"type": "number"},
                    "manual_angle_mean": {"type": "number"},
                    "jp_angle_mean": {"type": "number"},
                },
                "artifacts": {},
            },
        },
    )
    _write_json(
        layout.study_metadata_path(STUDY_ID),
        {
            "schema": "mjtensu.mldb/study/v1",
            "id": str(STUDY_ID),
            "status": "sealed",
            "name": "Plain tile classifier smoke",
            "description": "One-epoch end-to-end smoke Study on the real gray35 corpus.",
            "model": {
                "train": {
                    "corpus": str(CORPUS_ID),
                    "protocol": str(TRAIN_PROTOCOL_ID),
                    "architectures": [str(ARCHITECTURE_ID)],
                    "parameters": {
                        "epochs": {"values": [1]},
                        "batch_size": {"values": [256]},
                    },
                    "seeds": [42],
                }
            },
            "evaluations": [
                {
                    "stage": "angle-holdout",
                    "corpus": str(CORPUS_ID),
                    "protocol": str(EVALUATION_PROTOCOL_ID),
                    "parameters": {"batch_size": 1024},
                }
            ],
        },
    )

    resolved = (
        resolve_task(TASK_ID, layout, filesystem),
        resolve_corpus(CORPUS_ID, layout, filesystem),
        resolve_architecture(ARCHITECTURE_ID, layout, filesystem),
        resolve_train_protocol(TRAIN_PROTOCOL_ID, layout, filesystem),
        resolve_evaluation_protocol(EVALUATION_PROTOCOL_ID, layout, filesystem),
        resolve_study(STUDY_ID, layout, filesystem),
    )
    print("ready:", ", ".join(str(item.metadata.id) for item in resolved))


if __name__ == "__main__":
    main()
