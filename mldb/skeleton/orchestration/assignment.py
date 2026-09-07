"""Controller-side projection of committed Queue attempts into Worker assignments.

This boundary starts only after concrete launch preflight succeeded, the canonical child
Run was allocated and durably established as ``running``, and Queue attempt activation
committed. It checks agreement among those three already-authoritative inputs and
projects the frozen Worker API assignment values plus replay-stable immutable asset
descriptors.

It deliberately does not select Queue jobs, look up acquire tokens, run preflight,
allocate or persist Runs, activate attempts, implement retry/heartbeat policy, execute
Worker code, upload candidates, accept outcomes, or own a Worker loop. A committed
assignment must never be exposed before Queue attempt activation succeeds.
"""

from __future__ import annotations

from ..evaluation.preflight import EvaluationPreflight
from ..evaluation.run import EvaluationRun, EvaluationRunStatus
from ..training.preflight import TrainingPreflight
from ..training.run import TrainingRun, TrainingRunStatus
from .asset_source import ImmutableAssetSource
from .queue import QueueAttempt
from .worker_api import EvaluationAssignment, TrainingAssignment


def project_training_assignment(
    preflight: TrainingPreflight,
    run: TrainingRun,
    attempt: QueueAttempt,
    assets: ImmutableAssetSource,
) -> TrainingAssignment:
    """Project one committed Training attempt into the frozen Worker assignment.

    ``preflight`` is the successful concrete :class:`TrainingPreflight` from which the
    canonical child ``run`` was allocated. ``run`` is that already-persisted canonical
    Training Run and must still have :class:`TrainingRunStatus.RUNNING`. ``attempt`` is
    the Queue record returned only after successful ``activate_attempt`` commit and must
    still be open (``finished_at is None``). This function neither creates nor mutates
    any of those values.

    Projection requires exact execution-identity agreement before exposing an
    assignment:

    - ``attempt.run_id == run.id``;
    - ``run.corpus == preflight.corpus.metadata.id``;
    - ``run.architecture == preflight.architecture.metadata.id``;
    - ``run.train_protocol == preflight.protocol.metadata.id``;
    - ``run.execution.seed == preflight.seed``;
    - ``run.parameters == preflight.parameters``;
    - ``run.status is TrainingRunStatus.RUNNING``;
    - ``attempt.finished_at is None``.

    Any disagreement is an operation failure. It is not repaired, coerced, or treated
    as permission to re-run preflight, allocate another Run, or activate another Queue
    attempt.

    Successful projection copies Queue authority exactly from the committed attempt:
    ``attempt_id``, ``run_id``, ``lease_token``, and ``lease_until``. ``kind`` is exactly
    ``"training"``. Metadata is copied by value from the successful preflight handles:
    ``task = preflight.task.metadata``, ``corpus = preflight.corpus.metadata``,
    ``architecture = preflight.architecture.metadata``, and
    ``protocol = preflight.protocol.metadata``. ``seed`` and the complete resolved
    ``parameters`` mapping are copied unchanged from preflight after the Run-agreement
    checks above.

    The three immutable descriptors are derived through ``assets.describe()`` from the
    exact Controller-selected execution paths already carried by the preflight handles.
    Their assignment-local keys and integrity authorities are fixed as follows:

    - ``corpus_artifact``: key ``"corpus_artifact"``,
      ``preflight.corpus.artifact_path``,
      ``preflight.corpus.metadata.artifact.sha256`` and
      ``preflight.corpus.metadata.artifact.bytes``;
    - ``architecture_implementation``: key ``"architecture_implementation"``,
      ``preflight.architecture.implementation_path``, the sealed
      ``preflight.architecture.metadata.implementation.sha256``, and ``bytes=None``
      because current Architecture metadata records no implementation byte count;
    - ``train_protocol_implementation``: key
      ``"train_protocol_implementation"``, ``preflight.protocol.implementation_path``,
      the sealed ``preflight.protocol.metadata.implementation.sha256``, and
      ``bytes=None`` because current Train Protocol metadata records no implementation
      byte count.

    Successful Training preflight already established sealed executable state and the
    required non-null implementation hashes; this projection does not invent another
    integrity source or recalculate those hashes. ``assets.describe()`` must preserve
    the supplied integrity facts exactly and provide the opaque replay-stable retrieval
    identity required by the Worker API.

    Re-running this function for the same still-authorized committed attempt, canonical
    Run, and reconstructed equivalent successful preflight must therefore reproduce the
    same logical assignment and the same descriptor identities. No Queue row stores the
    Worker API response and the ephemeral original preflight object itself need not be
    persisted; a later Controller may reconstruct the same successful preflight from
    canonical Run-selected immutable inputs and project again.
    """

    ...


def project_evaluation_assignment(
    preflight: EvaluationPreflight,
    run: EvaluationRun,
    attempt: QueueAttempt,
    assets: ImmutableAssetSource,
) -> EvaluationAssignment:
    """Project one committed Evaluation attempt into the frozen Worker assignment.

    ``preflight`` is the successful concrete :class:`EvaluationPreflight` from which the
    canonical child ``run`` was allocated. ``run`` is that already-persisted canonical
    Evaluation Run and must still have :class:`EvaluationRunStatus.RUNNING`.
    ``attempt`` is the committed Queue activation for that Run and must still be open.
    No input is created, persisted, or mutated by this operation.

    Projection requires exact execution-identity agreement before Worker exposure:

    - ``attempt.run_id == run.id``;
    - ``run.model == preflight.model.metadata.id``;
    - ``run.corpus == preflight.corpus.metadata.id``;
    - ``run.evaluation_protocol == preflight.protocol.metadata.id``;
    - ``run.parameters == preflight.parameters``;
    - ``run.status is EvaluationRunStatus.RUNNING``;
    - ``attempt.finished_at is None``.

    It also preserves the frozen Evaluation assignment lineage guarantees already
    established by successful Model resolution/preflight:

    - ``preflight.model.training_run.id == preflight.model.metadata.training_run``;
    - ``preflight.model.training_run.status is TrainingRunStatus.COMPLETED``;
    - ``preflight.model.training_run.result is not None``;
    - ``preflight.model.training_run.architecture ==
      preflight.model.architecture.metadata.id``.

    Any disagreement is an operation failure. This boundary does not repair Model
    lineage, choose another Model, rerun preflight, allocate a replacement Evaluation
    Run, or activate another Queue attempt.

    Successful projection copies ``attempt_id``, ``run_id``, ``lease_token``, and
    ``lease_until`` exactly from the committed Queue attempt and fixes ``kind`` to
    ``"evaluation"``. Metadata is copied by value from the successful preflight:
    ``task = preflight.task.metadata``, ``corpus = preflight.corpus.metadata``,
    ``model = preflight.model.metadata``,
    ``training_run = preflight.model.training_run``,
    ``model_architecture = preflight.model.architecture.metadata``, and
    ``protocol = preflight.protocol.metadata``. The complete resolved ``parameters``
    mapping is copied unchanged after Run agreement succeeds.

    The four immutable descriptors are derived through ``assets.describe()`` from the
    exact Controller-selected preflight paths and frozen integrity authorities:

    - ``corpus_artifact``: key ``"corpus_artifact"``,
      ``preflight.corpus.artifact_path``,
      ``preflight.corpus.metadata.artifact.sha256`` and
      ``preflight.corpus.metadata.artifact.bytes``;
    - ``model_architecture_implementation``: key
      ``"model_architecture_implementation"``,
      ``preflight.model.architecture.implementation_path``, the sealed lineage
      Architecture ``implementation.sha256``, and ``bytes=None`` because current
      Architecture metadata records no implementation byte count;
    - ``model_weights``: key ``"model_weights"``, ``preflight.model.weights_path``,
      ``preflight.model.training_run.result.weights.sha256`` and
      ``preflight.model.training_run.result.weights.bytes``;
    - ``evaluation_protocol_implementation``: key
      ``"evaluation_protocol_implementation"``,
      ``preflight.protocol.implementation_path``, the sealed Evaluation Protocol
      ``implementation.sha256``, and ``bytes=None`` because current Evaluation Protocol
      metadata records no implementation byte count.

    Successful Evaluation preflight already verified the exact learned-weight bytes and
    the selected sealed Architecture/Evaluation Protocol implementation bytes against
    those authoritative identities. Projection therefore does not hash again, inspect
    learned state, or create a second artifact descriptor type.

    Re-running this function for the same still-authorized committed attempt, canonical
    Evaluation Run, and reconstructed equivalent successful preflight must reproduce the
    same logical assignment and replay-stable immutable descriptors. The original
    acquire response and original ephemeral preflight object need not be stored in Queue
    state; Controller replay may reconstruct authoritative input handles from the
    canonical Run-selected identities and execute this projection again.
    """

    ...
