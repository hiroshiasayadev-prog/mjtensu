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
        for index in range(6):
            payload = bytes([(index * 35) % 256] * (64 * 64))
            connection.execute(
                "INSERT INTO sample VALUES (?, 'train', ?, ?)",
                (f"s{index}", payload, index % 2),
            )
        connection.commit()


def _context(database: Path, architecture, *, cache_fraction: float = 0.5):
    return SimpleNamespace(
        corpus=SimpleNamespace(artifact_path=database),
        architecture=architecture,
        seed=7,
        parameters={
            "epochs": 1,
            "batch_size": 2,
            "learning_rate": 0.001,
            "weight_decay": 0.0,
            "rotation_augment_deg": 5.0,
            "perspective_augment": 0.05,
            "shear_augment": 0.05,
            "stretch_augment": 0.05,
            "projective_augment_probability": 1.0,
            "amp": False,
            "tf32": False,
            "cache_device": "cpu",
            "cache_vram_fraction": cache_fraction,
        },
        work_dir=database.parent,
    )


def test_train_supports_full_augmentation_and_cpu_cache(tmp_path: Path) -> None:
    database = tmp_path / "fixture.sqlite"
    _database(database)
    layout = RepositoryLayout(REPO_ROOT)
    filesystem = LocalFilesystem()
    architecture = resolve_architecture(
        ArchitectureId("tile-plain-gray35-v2"), layout, filesystem
    )
    protocol = resolve_train_protocol(
        TrainProtocolId("tile-shape-train-gpu-v3"), layout, filesystem
    )
    train = load_train_entrypoint(protocol)
    trained = train(_context(database, architecture))
    assert isinstance(trained, torch.nn.Module)
    fresh_build = __import__(
        "mldb.src.runtime.executable_loader",
        fromlist=["load_architecture_build"],
    ).load_architecture_build(architecture)
    assert accept_trained_state(trained, fresh_build)


def test_train_rejects_invalid_cache_fraction(tmp_path: Path) -> None:
    database = tmp_path / "fixture.sqlite"
    _database(database)
    layout = RepositoryLayout(REPO_ROOT)
    filesystem = LocalFilesystem()
    architecture = resolve_architecture(
        ArchitectureId("tile-plain-gray35-v2"), layout, filesystem
    )
    protocol = resolve_train_protocol(
        TrainProtocolId("tile-shape-train-gpu-v3"), layout, filesystem
    )
    train = load_train_entrypoint(protocol)
    try:
        train(_context(database, architecture, cache_fraction=0.9))
    except ValueError as error:
        assert "cache_vram_fraction" in str(error)
    else:
        raise AssertionError("invalid cache_vram_fraction was accepted")
