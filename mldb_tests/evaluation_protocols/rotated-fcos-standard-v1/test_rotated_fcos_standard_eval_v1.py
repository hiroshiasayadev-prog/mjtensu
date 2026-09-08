from pathlib import Path
import json
import sqlite3
from types import SimpleNamespace

from mldb.src.common.ids import ArchitectureId, EvaluationProtocolId
from mldb.src.repository._local_filesystem import LocalFilesystem
from mldb.src.repository.layout import RepositoryLayout
from mldb.src.runtime.executable_loader import load_architecture_build, load_evaluation_entrypoint
from mldb.src.runtime.resolution import resolve_architecture, resolve_evaluation_protocol


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
            "INSERT INTO sample VALUES ('v0', 'val', ?, ?)",
            (annotations, payload),
        )
        connection.commit()


def test_evaluate_returns_declared_detector_metrics(tmp_path: Path) -> None:
    database = tmp_path / "fixture.sqlite"
    _database(database)
    layout = RepositoryLayout(REPO_ROOT)
    filesystem = LocalFilesystem()
    architecture = resolve_architecture(
        ArchitectureId("rotated-fcos-nano-s05-f64-v1"), layout, filesystem
    )
    model = load_architecture_build(architecture)()
    protocol = resolve_evaluation_protocol(
        EvaluationProtocolId("rotated-fcos-standard-v1"), layout, filesystem
    )
    evaluate = load_evaluation_entrypoint(protocol)
    evaluate.__globals__["load_model"] = lambda _handle: model
    context = SimpleNamespace(
        corpus=SimpleNamespace(artifact_path=database),
        model=object(),
        parameters={
            "batch_size": 1,
            "score_threshold": 0.2,
            "nms_iou_threshold": 0.45,
            "match_iou_threshold": 0.5,
            "max_detections": 64,
            "amp": False,
            "tf32": False,
        },
        work_dir=tmp_path,
    )
    result = evaluate(context)
    assert set(result.metrics) == {
        "precision", "recall", "f1", "rotated_iou_mean",
        "angle_error_deg_mean", "tp", "fp", "fn",
    }
    assert result.artifacts == {}
    assert result.unavailable_outputs == ()
