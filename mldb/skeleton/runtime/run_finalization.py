"""Controller-side unsuccessful terminalization of canonical child Runs.

This module fixes the narrow boundary that turns one already allocated canonical
``RUNNING`` Training or Evaluation Run into ``FAILED`` or ``CANCELLED`` after Worker
execution or Controller-side result handling did not produce an accepted successful
terminal outcome.

Every operation begins by freshly reading the canonical Run through the frozen typed
persistence boundary and requires exact equality with the supplied ``RUNNING`` value
before constructing a terminal record. Terminal Runs are never reopened, normalized,
or rewritten. The terminal value is committed only through the corresponding frozen
``persist_*_run_transition()`` operation.

This boundary owns no Worker acknowledgement, lease validation, Queue close/retry/stop
policy, success-result acceptance, Model ensure, Study finalization, or generic Run
terminalization framework. In particular it deliberately imports no Queue port.
"""

from __future__ import annotations

from ..evaluation.run import (
    EvaluationRun,
    EvaluationRunExecution,
    EvaluationRunFailure,
    EvaluationRunStatus,
)
from ..repository.layout import RepositoryLayout
from ..repository.ports import FilesystemPort
from ..training.run import (
    TrainingRun,
    TrainingRunExecution,
    TrainingRunFailure,
    TrainingRunStatus,
)
from .run_persistence import (
    persist_evaluation_run_transition,
    persist_training_run_transition,
    read_evaluation_run,
    read_training_run,
)


def fail_training_run(
    run: TrainingRun,
    finished_at: object,
    layout: RepositoryLayout,
    filesystem: FilesystemPort,
    *,
    failure: TrainingRunFailure | None = None,
) -> TrainingRun:
    """Terminalize one exact canonical ``RUNNING`` Training Run as ``FAILED``.

    ``run`` is the caller's canonical-running view of the concrete attempt. Before any
    terminal value is constructed, the implementation must call
    :func:`read_training_run` for ``run.id`` and require both that the fresh canonical
    value has :class:`TrainingRunStatus.RUNNING` and that it equals ``run`` exactly.
    Stale disagreement is an operation failure. A fresh terminal value is never
    reopened or rewritten, even when the supplied value was previously running.

    The failed value is constructed from that freshly read canonical Run. It preserves
    exactly ``schema``, ``id``, ``corpus``, ``architecture``, ``train_protocol``, the
    complete resolved ``parameters`` mapping, ``execution.seed``,
    ``execution.started_at``, ``study``, ``environment``, and ``work``. Its execution
    value is a new :class:`TrainingRunExecution` carrying the same seed/start facts plus
    the required non-``None`` supplied ``finished_at``; ``status`` is exactly
    :class:`TrainingRunStatus.FAILED`; and ``result`` is ``None``. No successful weight
    metadata is synthesized from Worker state, candidate bytes, or a partially executed
    success-acceptance path.

    ``failure`` is optional because the frozen Training Run contract records concise
    failure text only when safely available. When supplied, the exact existing
    :class:`TrainingRunFailure` value is stored; this boundary does not define another
    failure DTO, derive text from exception classes, or invent missing failure facts.
    When omitted, terminal ``failure`` remains ``None``.

    The constructed value is committed only through
    :func:`persist_training_run_transition`; successful return is the exact failed domain
    value whose canonical transition was committed. This function chooses no Queue
    retry disposition and does not close the owning attempt.

    A caller catching an exception from ``accept_training_success()`` must first read
    the fresh canonical Run and may route the failure here only if that value is still
    ``RUNNING``. If success acceptance already committed ``COMPLETED``, this function
    must not be called. In particular, failure of deterministic Model ensure after the
    Training Run reached ``COMPLETED`` is a Model-repair/reconciliation problem, never
    permission for ``COMPLETED -> FAILED``.
    """

    ...


def cancel_training_run(
    run: TrainingRun,
    finished_at: object,
    layout: RepositoryLayout,
    filesystem: FilesystemPort,
) -> TrainingRun:
    """Terminalize one exact canonical ``RUNNING`` Training Run as ``CANCELLED``.

    The implementation freshly loads ``run.id`` through :func:`read_training_run` and
    requires exact equality with the supplied value plus current
    :class:`TrainingRunStatus.RUNNING`. Any stale disagreement or terminal canonical
    state is an operation failure; cancellation never reopens or rewrites history.

    The terminal value preserves exactly the fresh canonical ``schema``, ``id``,
    ``corpus``, ``architecture``, ``train_protocol``, complete ``parameters``, seed,
    ``started_at``, ``study``, ``environment``, and ``work`` facts. It sets
    ``status=CANCELLED``, supplies the required non-``None`` ``finished_at`` through a new
    :class:`TrainingRunExecution`, and carries ``result=None`` and ``failure=None``.
    Cancellation is not represented by fabricated failure type/message metadata.

    Persistence occurs only through :func:`persist_training_run_transition`; successful
    return is the exact cancelled Run committed there. A cancelled child Run does not by
    itself choose whether the logical Queue job becomes ``retry_wait``, ``cancelled``,
    or ``failed``; that later disposition remains orchestration policy.
    """

    ...


def fail_evaluation_run(
    run: EvaluationRun,
    finished_at: object,
    layout: RepositoryLayout,
    filesystem: FilesystemPort,
    *,
    failure: EvaluationRunFailure | None = None,
) -> EvaluationRun:
    """Terminalize one exact canonical ``RUNNING`` Evaluation Run as ``FAILED``.

    Before construction, the implementation must freshly call
    :func:`read_evaluation_run` for ``run.id`` and require the returned canonical value
    to equal ``run`` exactly and still have :class:`EvaluationRunStatus.RUNNING`.
    Stale disagreement or any already-terminal state is rejected rather than normalized
    or rewritten.

    The failed value preserves exactly the fresh canonical ``schema``, ``id``,
    ``model``, ``corpus``, ``evaluation_protocol``, complete resolved ``parameters``,
    ``execution.started_at``, ``study``, and ``environment`` facts. A new
    :class:`EvaluationRunExecution` adds the required non-``None`` supplied
    ``finished_at`` and ``status`` becomes exactly
    :class:`EvaluationRunStatus.FAILED`. ``result`` is ``None``,
    ``unavailable_outputs`` and ``validation_issues`` are empty, and no accepted
    metric/artifact metadata or partial-result fact is synthesized by this failure
    boundary.

    ``failure`` reuses the frozen :class:`EvaluationRunFailure` contract. The exact
    supplied value is stored when concise failure information is safely available; when
    omitted, ``failure`` remains ``None``. This function neither invents failure text nor
    introduces an acceptance-error or Worker-error hierarchy.

    The terminal value is committed only through
    :func:`persist_evaluation_run_transition`; successful return is that exact committed
    failed Run. Queue attempt closure and retry/stop disposition remain outside this
    module.

    A caller catching an exception from ``accept_evaluation_success()`` must freshly
    re-read the canonical Evaluation Run before selecting this path. It may call this
    boundary only while that Run is still ``RUNNING``. If result acceptance already
    established any terminal Evaluation state, this boundary must not rewrite it to
    ``FAILED``; the later outcome/reconciliation layer handles the already-terminal
    fact instead.
    """

    ...


def cancel_evaluation_run(
    run: EvaluationRun,
    finished_at: object,
    layout: RepositoryLayout,
    filesystem: FilesystemPort,
) -> EvaluationRun:
    """Terminalize one exact canonical ``RUNNING`` Evaluation Run as ``CANCELLED``.

    The implementation first reads the fresh canonical value through
    :func:`read_evaluation_run` and requires exact equality with ``run`` plus current
    :class:`EvaluationRunStatus.RUNNING`. It rejects stale supplied state and every
    already-terminal canonical Run rather than reopening or rewriting it.

    The cancelled value preserves exactly ``schema``, ``id``, ``model``, ``corpus``,
    ``evaluation_protocol``, complete ``parameters``, ``started_at``, ``study``, and
    ``environment`` from the fresh canonical value. A new
    :class:`EvaluationRunExecution` adds the required non-``None`` supplied
    ``finished_at``;
    ``status`` is ``CANCELLED``; ``result`` is ``None``; incompleteness collections are
    empty; and ``failure`` is ``None``. Cooperative cancellation does not fabricate
    failure metadata or pretend that unsuccessful work yielded accepted formal results.

    The value is committed only through :func:`persist_evaluation_run_transition` and
    returned after that canonical transition succeeds. The meaning of the cancelled
    child attempt for its logical Queue job remains outside this boundary: retry,
    operational cancellation, or final failure is chosen by later orchestration policy.
    """

    ...
