from pathlib import Path
import json
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
    annotations = json.dumps([
        {"label": "mahjong_tile", "obb": [160.0, 160.0, 28.0, 40.0, 0.0]}
    ])
    payload = bytes(3 * 320 * 320)
    with sqlite3.connect(path) as connection:
        connection.execute(
            "CREATE TABLE sample (sample_id TEXT PRIMARY KEY, split TEXT, annotations_json TEXT, image_rgb_u8 BLOB)"
        )
        connection.execute(
            "INSERT INTO sample VALUES ('s0', 'train', ?, ?)",
            (annotations, payload),
        )
        connection.commit()


def test_train_returns_selected_architecture_module(tmp_path: Path) -> None:
    database = tmp_path / "fixture.sqlite"
    _database(database)
    layout = RepositoryLayout(REPO_ROOT)
    filesystem = LocalFilesystem()
    architecture = resolve_architecture(
        ArchitectureId("rotated-fcos-nano-s05-f64-v1"), layout, filesystem
    )
    protocol = resolve_train_protocol(
        TrainProtocolId("rotated-fcos-standard-v1"), layout, filesystem
    )
    train = load_train_entrypoint(protocol)
    context = SimpleNamespace(
        corpus=SimpleNamespace(artifact_path=database),
        architecture=architecture,
        seed=7,
        parameters={
            "epochs": 1,
            "batch_size": 1,
            "learning_rate": 0.0003,
            "weight_decay": 0.0,
            "warmup_epochs": 0.0,
            "center_radius": 1.5,
            "gradient_clip": 10.0,
            "amp": False,
            "tf32": False,
        },
        work_dir=tmp_path,
    )
    trained = train(context)
    assert isinstance(trained, torch.nn.Module)
    fresh_build = __import__(
        "mldb.src.runtime.executable_loader", fromlist=["load_architecture_build"]
    ).load_architecture_build(architecture)
    assert accept_trained_state(trained, fresh_build)
