from pathlib import Path
import sqlite3
from types import SimpleNamespace

from mldb.src.common.ids import ArchitectureId, EvaluationProtocolId
from mldb.src.repository._local_filesystem import LocalFilesystem
from mldb.src.repository.layout import RepositoryLayout
from mldb.src.runtime.executable_loader import load_architecture_build, load_evaluation_entrypoint
from mldb.src.runtime.resolution import resolve_architecture, resolve_evaluation_protocol


REPO_ROOT = Path(__file__).resolve().parents[3]


def _database(path: Path) -> None:
    with sqlite3.connect(path) as connection:
        connection.execute(
            "CREATE TABLE sample (sample_id TEXT PRIMARY KEY, split TEXT, image_gray_u8 BLOB, class_index INTEGER)"
        )
        for split in ("train", "manual_val", "jp_val"):
            for index in range(2):
                connection.execute(
                    "INSERT INTO sample VALUES (?, ?, ?, ?)",
                    (f"{split}-{index}", split, bytes([index * 100] * (64 * 64)), index),
                )
        connection.commit()


def test_evaluate_returns_declared_metrics(tmp_path: Path) -> None:
    database = tmp_path / "fixture.sqlite"
    _database(database)
    layout = RepositoryLayout(REPO_ROOT)
    filesystem = LocalFilesystem()
    architecture = resolve_architecture(ArchitectureId("tile-plain-gray35-v2"), layout, filesystem)
    model = load_architecture_build(architecture)()
    protocol = resolve_evaluation_protocol(EvaluationProtocolId("tile-shape-eval-angle-v2"), layout, filesystem)
    evaluate = load_evaluation_entrypoint(protocol)
    evaluate.__globals__["load_model"] = lambda _handle: model
    context = SimpleNamespace(
        corpus=SimpleNamespace(artifact_path=database),
        model=object(),
        parameters={"batch_size": 2, "eval_angles": [0.0, 15.0], "amp": False, "tf32": False},
        work_dir=tmp_path,
    )
    result = evaluate(context)
    assert set(result.metrics) == {
        "manual_accuracy_0deg",
        "jp_accuracy_0deg",
        "manual_angle_mean",
        "jp_angle_mean",
    }
    assert result.artifacts == {}
    assert result.unavailable_outputs == ()
