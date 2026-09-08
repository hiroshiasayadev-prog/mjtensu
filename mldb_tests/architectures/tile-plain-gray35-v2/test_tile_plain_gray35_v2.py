from pathlib import Path

import torch

from mldb.src.common.ids import ArchitectureId
from mldb.src.repository._local_filesystem import LocalFilesystem
from mldb.src.repository.layout import RepositoryLayout
from mldb.src.runtime.executable_loader import load_architecture_build
from mldb.src.runtime.resolution import resolve_architecture


REPO_ROOT = Path(__file__).resolve().parents[3]


def test_build_matches_declared_classifier_interface() -> None:
    handle = resolve_architecture(
        ArchitectureId("tile-plain-gray35-v2"),
        RepositoryLayout(REPO_ROOT),
        LocalFilesystem(),
    )
    model = load_architecture_build(handle)()
    output = model(torch.zeros(2, 1, 64, 64))
    assert isinstance(model, torch.nn.Module)
    assert output.shape == (2, 35)
