from pathlib import Path

from mldb_v2.src.catalog.architecture_build import _load_architecture_build

ROOT = Path(__file__).resolve().parents[4] / "mldb_data"


def test_architecture_build_contract() -> None:
    build = _load_architecture_build(ROOT, "rotated-fcos/rotated-fcos-s05-rgb-baseline-v1")
    assert callable(build)
