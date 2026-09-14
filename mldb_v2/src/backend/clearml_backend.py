"""Concrete ClearML BackendPort composition and registry activation."""

from __future__ import annotations

from typing import cast

from mldb_v2.src.backend._clearml_admission import ClearMLAdmissionService
from mldb_v2.src.backend._clearml_cancellation import (
    ClearMLCancellationService,
    ClearMLLogRecord,
    ClearMLLogService,
)
from mldb_v2.src.backend._clearml_observation import (
    ClearMLObservationService,
    _stage_key_from_input,
)
from mldb_v2.src.backend._clearml_sdk import ClearMLSDKAdapter
from mldb_v2.src.backend._config import BackendConfig
from mldb_v2.src.backend._registry import BackendRegistry
from mldb_v2.src.backend.candidate_outcome import (
    BackendObservation,
    StageKey,
    TerminalCandidate,
)
from mldb_v2.src.backend.stage_input import StageInput
from mldb_v2.src.common.ids import StudyResultId


class ClearMLBackendError(RuntimeError):
    """Bounded composition/activation failure for the concrete ClearML backend."""


class ClearMLBackend:
    """One concrete BackendPort composed from the completed ClearML capabilities."""

    def __init__(
        self,
        *,
        admission: ClearMLAdmissionService,
        observation: ClearMLObservationService,
        cancellation: ClearMLCancellationService,
        logs: ClearMLLogService | None = None,
    ) -> None:
        self._admission = admission
        self._observation = observation
        self._cancellation = cancellation
        self._logs = logs

    def admit(self, *, stage_input: StageInput) -> BackendObservation:
        """Idempotently admit exact work, then return its exact backend observation."""
        self._admission.admit(stage_input=stage_input)
        stage_key = _stage_key_from_input(stage_input)
        observed = self._observation.observe(stage_key=stage_key)
        if observed is None:
            raise ClearMLBackendError(
                "ClearML admission succeeded but exact owned work was not observable"
            )
        return observed

    def observe(self, *, stage_key: StageKey) -> BackendObservation | None:
        return self._observation.observe(stage_key=stage_key)

    def collect(self, *, stage_key: StageKey) -> TerminalCandidate | None:
        return self._observation.collect(stage_key=stage_key)

    def cancel_study(self, *, study_result: StudyResultId) -> None:
        self._cancellation.cancel_study(study_result=study_result)

    def read_task_logs(
        self, *, study_result: StudyResultId, task_id: str
    ) -> ClearMLLogRecord | None:
        """Optional observational capability; intentionally outside BackendPort."""
        if self._logs is None:
            return None
        return self._logs.read_task_logs(study_result=study_result, task_id=task_id)


def _option_string(
    config: BackendConfig, name: str, *, required: bool = False
) -> str | None:
    value = config.options.get(name)
    if value is None:
        if required:
            raise ClearMLBackendError(f"ClearML backend option is required: {name}")
        return None
    if type(value) is not str or not value or value.strip() != value:
        raise ClearMLBackendError(
            f"ClearML backend option {name!r} must be a non-empty trimmed string"
        )
    return value


def clearml_backend_factory(config: BackendConfig) -> ClearMLBackend:
    """Resolve operational config to one SDK-neutral/genuine ClearMLBackend."""
    if not isinstance(config, BackendConfig) or config.backend_type != "clearml":
        raise ClearMLBackendError("ClearML factory requires BackendConfig('clearml', ...)")

    client = config.options.get("client")
    if client is None:
        client = ClearMLSDKAdapter.from_backend_config(config)
    queue = _option_string(config, "queue")
    recovery_raw = config.options.get("recovery_search_attempts", 3)
    if type(recovery_raw) is not int or recovery_raw < 1:
        raise ClearMLBackendError(
            "ClearML recovery_search_attempts must be a positive integer"
        )

    admission = ClearMLAdmissionService(
        client=cast(object, client),  # runtime structural Protocol
        queue=queue,
        recovery_search_attempts=recovery_raw,
    )
    observation = ClearMLObservationService(client=cast(object, client))
    cancellation = ClearMLCancellationService(client=cast(object, client))
    logs = ClearMLLogService(client=cast(object, client))
    return ClearMLBackend(
        admission=admission,
        observation=observation,
        cancellation=cancellation,
        logs=logs,
    )


def register_clearml_backend(registry: BackendRegistry) -> None:
    """Explicitly register generic backend type ``clearml`` without import side effects."""
    if not isinstance(registry, BackendRegistry):
        raise TypeError("registry must be a BackendRegistry")
    registry.register("clearml", clearml_backend_factory)
