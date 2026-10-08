from pathlib import Path
import importlib.util

from mldb_v2.src.training.train_interface import _load_train_callable

ROOT = Path(__file__).resolve().parents[4]
DATA = ROOT / "mldb_data"
PROTOCOL = DATA / "nanodet" / "train_protocols" / "nanodet-aabb-synthetic-real-mix-gpu-v2.py"


def _module():
    spec = importlib.util.spec_from_file_location("synthetic_real_mix_protocol", PROTOCOL)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_train_callable_contract() -> None:
    assert callable(_load_train_callable(DATA, "nanodet/nanodet-aabb-synthetic-real-mix-gpu-v2"))


def test_fixed_source_mix_sampler_preserves_budget_and_source_counts() -> None:
    module = _module()
    sampler = module._FixedSourceMixSampler(
        synthetic_size=10, real_size=2, jp_size=20, source_budget=10,
        real_repeat=3, jp_fraction=0.5, seed=42,
    )
    indices = list(iter(sampler))
    assert len(indices) == len(sampler) == 16
    assert sum(0 <= x < 10 for x in indices) == 5
    assert sum(10 <= x < 12 for x in indices) == 6
    assert sum(x >= 12 for x in indices) == 5
    assert indices.count(10) == 3
    assert indices.count(11) == 3
