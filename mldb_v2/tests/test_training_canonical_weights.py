from __future__ import annotations

from io import BytesIO
import json
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from mldb_v2.src.catalog.architecture_build import _load_architecture_build
from mldb_v2.src.training.canonical_weights import (
    _build_fresh_architecture_module,
    _canonicalize_state_dict,
    _canonicalize_trained_module_state,
    _load_canonical_state_dict_bytes,
    _load_state_into_fresh_architecture,
    _serialize_canonical_state_dict,
)


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _make_root(tmp_path: Path) -> Path:
    root = tmp_path / "mldb_data"
    _write_json(
        root / "demo" / "namespace.yaml",
        {
            "schema": "mjtensu.mldb-v2/namespace/v1",
            "id": "demo",
            "name": "Demo",
            "description": "",
        },
    )
    return root


def _write_architecture(root: Path, local_id: str, source: str) -> str:
    entity_id = f"demo/{local_id}"
    directory = root / "demo" / "architectures"
    _write_json(
        directory / f"{local_id}.yaml",
        {
            "schema": "mjtensu.mldb-v2/architecture/v1",
            "id": entity_id,
            "status": "draft",
            "task": "demo/task-v1",
            "name": local_id,
            "family": "toy",
            "description": "",
            "implementation": {"framework": "pytorch", "entrypoint": "build"},
            "interface": {"input": {"kind": "tensor"}, "output": {"kind": "tensor"}},
            "structure": {"summary": "toy"},
        },
    )
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"{local_id}.py").write_text(source, encoding="utf-8")
    return entity_id


def test_plain_state_dict_is_detached_cpu_and_serialized_directly() -> None:
    source = torch.arange(6.0, requires_grad=True).reshape(2, 3)
    canonical = _canonicalize_state_dict({"weight": source})
    tensor = canonical["weight"]
    assert tensor.device.type == "cpu"
    assert not tensor.requires_grad
    assert tensor.grad_fn is None

    data = _serialize_canonical_state_dict({"weight": source})
    loaded = torch.load(BytesIO(data), map_location="cpu", weights_only=True)
    assert type(loaded) is dict
    assert set(loaded) == {"weight"}
    assert isinstance(loaded["weight"], torch.Tensor)
    assert "state_dict" not in loaded


def test_loader_uses_cpu_and_weights_only(monkeypatch: pytest.MonkeyPatch) -> None:
    data = _serialize_canonical_state_dict({"weight": torch.ones(2)})
    original = torch.load
    seen: dict[str, object] = {}

    def spy(*args, **kwargs):
        seen.update(kwargs)
        return original(*args, **kwargs)

    monkeypatch.setattr(torch, "load", spy)
    state = _load_canonical_state_dict_bytes(data)
    assert set(state) == {"weight"}
    assert seen["map_location"] == "cpu"
    assert seen["weights_only"] is True


@pytest.mark.parametrize(
    "value",
    [
        {"state_dict": {"weight": torch.ones(1)}},
        {"optimizer": {"lr": 0.1}},
        {"scheduler": object()},
        {"scaler": {}},
        {"config": {"epochs": 1}},
        {"epoch": 3},
    ],
)
def test_wrapped_or_checkpoint_payloads_are_rejected(value: object) -> None:
    with pytest.raises(ValueError):
        _serialize_canonical_state_dict(value)


def test_state_dict_rejects_non_string_keys_and_non_tensor_values() -> None:
    with pytest.raises(ValueError, match="keys"):
        _canonicalize_state_dict({1: torch.ones(1)})
    with pytest.raises(ValueError, match="values"):
        _canonicalize_state_dict({"weight": [1.0]})


def test_strict_loading_rejects_missing_and_unexpected_keys(tmp_path: Path) -> None:
    root = _make_root(tmp_path)
    architecture_id = _write_architecture(
        root,
        "linear-v1",
        "import torch.nn as nn\ndef build():\n    return nn.Linear(3, 2)\n",
    )
    exact = nn.Linear(3, 2).state_dict()
    loaded = _load_state_into_fresh_architecture(root, architecture_id, exact)
    assert isinstance(loaded, nn.Linear)

    missing = dict(exact)
    missing.pop("bias")
    with pytest.raises(ValueError, match="strictly compatible"):
        _load_state_into_fresh_architecture(root, architecture_id, missing)

    unexpected = dict(exact)
    unexpected["extra"] = torch.ones(1)
    with pytest.raises(ValueError, match="strictly compatible"):
        _load_state_into_fresh_architecture(root, architecture_id, unexpected)


def test_two_structurally_different_architectures_use_same_runtime(tmp_path: Path) -> None:
    root = _make_root(tmp_path)
    linear_id = _write_architecture(
        root,
        "linear-v1",
        "import torch.nn as nn\ndef build():\n    return nn.Linear(3, 2)\n",
    )
    stack_id = _write_architecture(
        root,
        "stack-v1",
        "import torch.nn as nn\ndef build():\n    return nn.Sequential(nn.Linear(3, 4), nn.ReLU(), nn.Linear(4, 2))\n",
    )

    first = _build_fresh_architecture_module(root, linear_id)
    second = _build_fresh_architecture_module(root, stack_id)
    assert isinstance(first, nn.Linear)
    assert isinstance(second, nn.Sequential)

    fresh_a = _build_fresh_architecture_module(root, linear_id)
    fresh_b = _build_fresh_architecture_module(root, linear_id)
    assert fresh_a is not fresh_b


def test_trained_module_must_strictly_match_fresh_architecture(tmp_path: Path) -> None:
    root = _make_root(tmp_path)
    architecture_id = _write_architecture(
        root, "linear-v1", "import torch.nn as nn\ndef build():\n    return nn.Linear(3, 2)\n"
    )
    valid = nn.Linear(3, 2)
    state = _canonicalize_trained_module_state(root, architecture_id, valid)
    assert set(state) == {"weight", "bias"}

    with pytest.raises(ValueError, match="strictly compatible"):
        _canonicalize_trained_module_state(root, architecture_id, nn.Linear(4, 2))
    with pytest.raises(ValueError, match="torch.nn.Module"):
        _canonicalize_trained_module_state(root, architecture_id, object())


def test_runtime_reuses_w002_architecture_loader_without_importlib_duplication(tmp_path: Path) -> None:
    root = _make_root(tmp_path)
    architecture_id = _write_architecture(
        root, "linear-v1", "import torch.nn as nn\ndef build():\n    return nn.Linear(3, 2)\n"
    )
    assert callable(_load_architecture_build(root, architecture_id))

    source_path = Path(__file__).resolve().parents[1] / "src" / "training" / "canonical_weights.py"
    source = source_path.read_text(encoding="utf-8")
    assert "_load_architecture_build" in source
    assert "importlib" not in source
