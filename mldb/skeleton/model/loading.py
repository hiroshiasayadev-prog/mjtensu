"""Public Python signatures for MLDB Model loading.

This skeleton fixes the runtime shape of one prepared Model and the stable operation
that reconstructs its learned ``torch.nn.Module`` through immutable Model lineage.
It intentionally does not parse YAML, scan repositories, resolve IDs, calculate or
compare hashes, import Architecture implementations, reimplement canonical PyTorch
state loading, evaluate Models, or define repository/service abstractions.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from ..runtime.catalog_handles import ArchitectureHandle
from ..training.run import TrainingRun
from .identity import Model


@dataclass(frozen=True, slots=True)
class ModelHandle:
    """Validated Model lineage plus the resources required to load learned state.

    For a resolver-produced handle, ``metadata_path`` is the canonical Model YAML and
    ``training_run_metadata_path`` is the authoritative canonical ``run.yaml``. For a
    Worker-materialized handle both provenance paths are ``None`` because Controller
    supplies the already validated Model and completed Training Run metadata by value;
    Worker does not fabricate repository metadata files.

    ``weights_path`` is always the concrete execution path for the learned-weight bytes
    described by ``training_run.result.weights``. It is the canonical Training Run
    ``artifacts/weights.pt`` for resolver output, or the Worker-local materialization of
    the exact Controller-selected bytes after SHA-256 and byte-count verification. The
    local path is not canonical provenance, and the frozen ``CanonicalWeightsArtifact``
    metadata remains authoritative for the expected learned bytes in both cases.

    ``architecture`` is the same ``ArchitectureHandle`` selected by
    ``training_run.architecture``. It may likewise be resolver-produced or Worker-
    materialized under that handle's contract. Constructing this handle must not import
    or execute Architecture Python code or load weights.

    Applicable learned-weight and implementation integrity requirements remain mandatory
    before runtime consumption; Worker materialization does not weaken the common loader
    checks or create a second source of truth.
    """

    metadata: Model
    metadata_path: Path | None
    training_run: TrainingRun
    training_run_metadata_path: Path | None
    weights_path: Path
    architecture: ArchitectureHandle


def load_model(model: ModelHandle) -> torch.nn.Module:
    """Load the learned module represented by one runtime ``ModelHandle``.

    This is the stable Model-feature boundary used by downstream code, including an
    Evaluation Protocol that receives only ``EvaluationContext.model``. No caller
    supplies a Training Run, Architecture build callable, repository object, or
    weights path separately.

    The implementation must use ``model.architecture`` as the handoff to the common
    executable-loader responsibility. That loader verifies any required sealed
    Architecture implementation integrity before importing the selected Python file
    and exposing its declared no-argument ``ArchitectureBuild`` callable. Merely
    resolving or constructing ``ModelHandle`` must not import executable user code.

    Before learned bytes are consumed, their recorded Training Run integrity must be
    satisfied against the actual bytes at ``model.weights_path`` regardless of whether
    that path is the resolver-produced canonical artifact or a verified Worker-local
    execution copy. Hash calculation and filesystem verification remain runtime
    integrity responsibilities rather than a new Model-domain hash contract.

    Once the ``ArchitectureBuild`` callable is available, the implementation must
    delegate ``model.weights_path`` plus that callable to the frozen
    ``training.weights.load_canonical_weights()`` operation. Restricted CPU loading,
    plain ``Mapping[str, Tensor]`` validation, fresh Architecture construction, and
    strict ``load_state_dict`` semantics are therefore reused rather than redefined
    here.

    The returned value is the fresh learned ``torch.nn.Module``. This operation does
    not cache modules, move them to an execution device, select eval/train mode,
    perform model-family-specific preprocessing or inference, or own Evaluation
    behavior.
    """

    ...
