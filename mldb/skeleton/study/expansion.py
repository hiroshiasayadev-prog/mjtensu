"""Public Python signature for deterministic MLDB Study plan materialization.

This skeleton fixes the boundary that turns one preflight-ready sealed :class:`Study`
plus the protocol metadata required for public-parameter resolution into one ordered,
fully resolved :class:`StudyPlan`.

It owns training-grid expansion, canonical training-axis ordering, existing-Model
ordering, public-parameter default resolution, evaluation-stage replication/order,
and Study-local trial numbering. It intentionally does not resolve repository assets,
verify executable or artifact integrity, establish Task compatibility, validate Model
lineage, allocate Study Runs or child Runs, serialize/hash ``plan.jsonl``, or perform
Queue/Worker scheduling.
"""

from __future__ import annotations

from collections.abc import Mapping

from ..common.ids import EvaluationProtocolId
from ..common.parameters import resolve_public_parameters
from ..evaluation.protocol import EvaluationProtocol
from ..training.protocol import TrainProtocol
from .definition import (
    Study,
    StudyExistingModelSource,
    StudyStatus,
    StudyTrainingModelSource,
)
from .plan import StudyPlan


def materialize_study_plan(
    study: Study,
    train_protocol: TrainProtocol | None,
    evaluation_protocols: Mapping[EvaluationProtocolId, EvaluationProtocol],
) -> StudyPlan:
    """Materialize one preflight-ready sealed Study into its ordered resolved plan.

    ``study`` must be ``StudyStatus.SEALED`` and must already satisfy Study metadata
    validation plus execution preflight. A draft Study is not a provisional-plan
    input and must be rejected by this operation. Preflight is responsible for
    referenced-asset existence, required sealed/integrity checks, common Task
    agreement, Study-supplied parameter-key publication, existing-Model validity and
    completed lineage, and the other cross-asset conditions required before Study Run
    allocation. This operation does not repeat those repository/runtime checks.

    For :class:`StudyTrainingModelSource`, ``train_protocol`` is required and must be
    the exact Train Protocol metadata referenced by ``study.model.protocol``. For
    :class:`StudyExistingModelSource`, no Training Protocol participates in
    materialization and ``train_protocol`` must be ``None``. The function does not
    accept a Training Protocol handle, repository, executable loader, or Training Run.

    ``evaluation_protocols`` supplies the reusable metadata needed to resolve every
    declared Study evaluation stage. Each referenced ``EvaluationProtocolId`` must be
    present, and the selected value's ``id`` must agree with both its mapping key and
    the stage reference. The same Evaluation Protocol may serve multiple stages.
    Unreferenced mapping entries are not materialization dependencies and must not
    affect the resulting plan. Protocol byte integrity and Task compatibility remain
    preflight responsibilities.

    Training-source materialization enumerates exactly one row per Cartesian-product
    coordinate. Axis nesting from outermost to innermost is:

    1. Architecture in authored declaration order;
    2. training parameter keys in locale-independent ascending lexicographic order;
    3. each sorted key's values in that axis's authored declaration order;
    4. training seed in authored declaration order.

    Mapping insertion order under ``study.model.parameters`` is therefore never
    semantic. Architecture, parameter-value, and seed sequence order is semantic.
    Seed is always the innermost axis and remains separate from protocol parameters.
    An empty training-parameter mapping contributes no parameter axis, so expansion is
    simply Architecture x seed.

    For every concrete training grid coordinate, the implementation forms caller
    overrides only from Study-declared parameter axes and applies the frozen
    :func:`resolve_public_parameters` operation to ``train_protocol.parameters``.
    The resulting ``StudyPlanTraining.parameters`` is the complete resolved public
    mapping, including defaults for every published Train Protocol parameter omitted
    from the Study. ``seed`` is never inserted into that mapping.

    Existing-Model materialization creates exactly one row for each authored Model ID,
    preserving ``study.model.models`` order exactly. Each such row stores that Model
    ID in ``StudyPlanRow.model`` with ``StudyPlanRow.training`` set to ``None``. It
    creates no training coordinate, Training Run, or new Model.

    Every materialized row receives every Study evaluation stage in authored
    ``study.evaluations`` order. Evaluation parameters are fixed per stage and are not
    Cartesian-product axes. For each stage, the implementation applies
    :func:`resolve_public_parameters` to the referenced Evaluation Protocol's public
    declarations and that stage's fixed overrides, storing the complete resolved
    mapping in ``StudyPlanEvaluation.parameters``.

    After the canonical row order is established, trial strings are assigned
    sequentially as ``trial-0001``, ``trial-0002``, and so on, starting at one and
    without gaps. Evaluation stages never affect trial numbering. The same sealed
    Study plus the same referenced immutable protocol metadata therefore yields the
    same ordered plan values independently of mapping/hash iteration accidents,
    filesystem traversal, locale, Queue order, Worker scheduling, or process timing.

    Parameter-resolution failure, missing or disagreeing required protocol metadata,
    a non-sealed Study, or any state that cannot satisfy the frozen ``StudyPlan``
    contract is an operation failure and returns no plan. This skeleton intentionally
    fixes neither a feature-specific exception hierarchy nor a result/report wrapper.
    Successful materialization returns only ``StudyPlan``; callers may separately use
    the frozen ``validate_study_plan()`` operation where useful.

    Exact JSONL bytes, UTF-8/LF policy, object-key ordering, plan path, SHA-256, byte
    size, Study Run allocation/lifecycle, child Run identity/creation, retries, and
    Queue/Worker behavior are deliberately outside this operation.
    """

    ...
