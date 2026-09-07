"""Canonical persistence boundary for deterministic MLDB Model records.

A completed Training Run has exactly one deterministic Model identity, but the existing
frozen Model domain skeleton intentionally owns no filesystem mutation. This module adds
only the narrow idempotent ensure operation required by Training result acceptance and
later reconciliation.

It does not introduce a Model repository/service, learned-byte storage, Model revision,
promotion state, generic YAML codec, transaction coordinator, or Training Run lifecycle
mutation. Canonical Model placement remains owned by :class:`RepositoryLayout`, and the
record shape remains exactly the frozen three-field :class:`Model` contract.
"""

from __future__ import annotations

from ..repository.layout import RepositoryLayout
from ..repository.ports import FilesystemPort
from ..training.run import TrainingRun
from .identity import Model, model_for_completed_training_run


def ensure_model_for_completed_training_run(
    training_run: TrainingRun,
    layout: RepositoryLayout,
    filesystem: FilesystemPort,
) -> Model:
    """Ensure the one exact deterministic Model for a canonical completed Training Run.

    ``training_run`` must be the canonical completed Run value whose result already
    contains valid canonical-weight metadata. The implementation must construct the
    expected record only through ``model_for_completed_training_run(training_run)``; it
    must not reimplement the Training-Run-to-Model ID mapping or add any Model fields.

    Before mutation, the supplied Run must agree with the canonical Training Run record
    for the same ID. A stale, non-canonical, non-completed, or otherwise invalid Run is
    an operation failure. This check is intended to reuse the frozen typed Training Run
    read/validation semantics rather than adding a second Training Run parser here.

    The canonical destination is only
    ``layout.model_metadata_path(expected_model.id)``. Persistence is the canonical
    Model YAML containing exactly ``schema``, ``id``, and ``training_run``; no learned
    bytes, Architecture metadata, result hash, quality status, or sidecar file is written.

    Idempotency is required:

    - if the canonical Model file is absent, create the exact expected Model record;
    - if it already exists and parses/validates as exactly the same Model identity and
      lineage, return the expected Model without rewriting it;
    - if the existing canonical file is malformed, contains undeclared fields, or
      disagrees with the expected deterministic Model, reject the inconsistency rather
      than overwrite or repurpose it;
    - if another canonical Model record already claims the same Training Run under a
      different identity, reject the duplicate-lineage inconsistency rather than create
      a second Model.

    Existing Model parsing and YAML serialization may remain implementation-private to
    this function. A public ModelRepository, model listing service, or generic YAML codec
    is neither required nor introduced. Successful return means the deterministic Model
    YAML exists and is exact; it says nothing about Queue satisfaction or downstream
    Evaluation scheduling.

    The operation is deliberately independently callable after Training Run completion.
    Therefore an interruption after the Run reached ``COMPLETED`` but before Model YAML
    creation can be repaired by replay or later reconciliation without reopening or
    rewriting the terminal Training Run.
    """

    ...
