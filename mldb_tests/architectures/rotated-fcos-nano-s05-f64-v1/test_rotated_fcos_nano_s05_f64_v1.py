from pathlib import Path

import torch

from mldb.src.common.ids import ArchitectureId
from mldb.src.repository._local_filesystem import LocalFilesystem
from mldb.src.repository.layout import RepositoryLayout
from mldb.src.runtime.executable_loader import load_architecture_build
from mldb.src.runtime.resolution import resolve_architecture


REPO_ROOT = Path(__file__).resolve().parents[3]


def test_build_matches_declared_rotated_fcos_interface() -> None:
    handle = resolve_architecture(
        ArchitectureId("rotated-fcos-nano-s05-f64-v1"),
        RepositoryLayout(REPO_ROOT),
        LocalFilesystem(),
    )
    model = load_architecture_build(handle)()
    outputs = model(torch.zeros(1, 3, 320, 320))
    assert isinstance(model, torch.nn.Module)
    assert isinstance(outputs, tuple)
    assert [tuple(item.shape) for item in outputs] == [
        (1, 8, 40, 40),
        (1, 8, 20, 20),
        (1, 8, 10, 10),
    ]
