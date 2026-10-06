from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

from mldb_v2.src.training.train_interface import _load_train_callable

ROOT = Path(__file__).resolve().parents[4] / "mldb_data"
IMPLEMENTATION = ROOT / "nanodet" / "train_protocols" / "nanodet-aabb-deployment-finetune-gpu-v1.py"


def _module():
    spec = importlib.util.spec_from_file_location("nanodet_deployment_finetune_v1", IMPLEMENTATION)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_train_callable_contract() -> None:
    assert callable(_load_train_callable(ROOT, "nanodet/nanodet-aabb-deployment-finetune-gpu-v1"))


def test_sampler_exposes_all_deployment_and_exact_fresh_jp_subset() -> None:
    module = _module()
    sampler = module._DeploymentReplaySampler(
        deployment_size=6,
        jp_size=20,
        jp_replay_images=5,
        seed=42,
    )
    epoch0 = list(iter(sampler))
    epoch1 = list(iter(sampler))
    assert len(epoch0) == len(epoch1) == 11
    assert {i for i in epoch0 if i < 6} == set(range(6))
    assert {i for i in epoch1 if i < 6} == set(range(6))
    jp0 = {i - 6 for i in epoch0 if i >= 6}
    jp1 = {i - 6 for i in epoch1 if i >= 6}
    assert len(jp0) == len(jp1) == 5
    assert jp0 != jp1
    assert all(0 <= i < 20 for i in jp0 | jp1)
