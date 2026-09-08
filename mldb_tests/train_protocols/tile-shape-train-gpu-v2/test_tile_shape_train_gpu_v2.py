from pathlib import Path
import sqlite3
from types import SimpleNamespace

import torch

from mldb.src.common.ids import ArchitectureId, TrainProtocolId
from mldb.src.repository._local_filesystem import LocalFilesystem
from mldb.src.repository.layout import RepositoryLayout
from mldb.src.runtime.executable_loader import load_train_entrypoint
from mldb.src.runtime.resolution import resolve_architecture, resolve_train_protocol
from mldb.src.training.weights import accept_trained_state


REPO_ROOT = Path(__file__).resolve().parents[3]


def _database(path: Path) -> None:
    with sqlite3.connect(path) as connection:
        connection.execute(
            "CREATE TABLE sample (sample_id TEXT PRIMARY KEY, split TEXT, image_gray_u8 BLOB, class_index INTEGER)"
        )
        for index in range(4):
            payload = bytes([index * 40] * (64 * 64))
            connection.execute(
                "INSERT INTO sample VALUES (?, 'train', ?, ?)",
                (f"s{index}", payload, index % 2),
            )
        connection.commit()


def test_train_returns_architecture_compatible_module(tmp_path: Path) -> None:
    database = tmp_path / "fixture.sqlite"
    _database(database)
    layout = RepositoryLayout(REPO_ROOT)
    filesystem = LocalFilesystem()
    architecture = resolve_architecture(ArchitectureId("tile-plain-gray35-v2"), layout, filesystem)
    protocol = resolve_train_protocol(TrainProtocolId("tile-shape-train-gpu-v2"), layout, filesystem)
    train = load_train_entrypoint(protocol)
    context = SimpleNamespace(
        corpus=SimpleNamespace(artifact_path=database),
        architecture=architecture,
        seed=7,
        parameters={
            "epochs": 1,
            "batch_size": 2,
            "learning_rate": 0.001,
            "weight_decay": 0.0,
            "rotation_augment_deg": 5.0,
            "amp": False,
            "tf32": False,
        },
        work_dir=tmp_path,
    )
    trained = train(context)
    assert isinstance(trained, torch.nn.Module)
    fresh_build = __import__("mldb.src.runtime.executable_loader", fromlist=["load_architecture_build"]).load_architecture_build(architecture)
    accepted = accept_trained_state(trained, fresh_build)
    assert accepted
