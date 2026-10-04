from pathlib import Path

from mldb_v2.src.catalog.architecture_build import _load_architecture_build

ROOT = Path(__file__).resolve().parents[4] / "mldb_data"


def test_architecture_build_contract() -> None:
    build = _load_architecture_build(ROOT, "tile-classifier/tile-plain-gray35-half-stage1-24-v1")
    model = build()
    assert callable(build)
    assert tuple(model(__import__("torch").zeros(2, 1, 64, 64)).shape) == (2, 35)
