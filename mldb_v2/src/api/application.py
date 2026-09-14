"""Concrete transport-independent MLDB v2 Application composition."""

from __future__ import annotations

import copy
from collections.abc import Callable, Collection, Iterable, Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import TypeVar, cast

from mldb_v2.src.backend._config import BackendConfig
from mldb_v2.src.backend._registry import BackendRegistry
from mldb_v2.src.backend.backend_port import BackendPort
from mldb_v2.src.backend.candidate_outcome import BackendObservation, StageKey, TerminalCandidate
from mldb_v2.src.backend.stage_input import StageInput
from mldb_v2.src.common.ids import (
    EntityKind,
    ExecutionKey,
    NamespaceId,
    StudyId,
    StudyPlanId,
    StudyResultId,
    _validate_typed_reference,
)
from mldb_v2.src.repository.canonical_writes import CanonicalRepositoryWriter
from mldb_v2.src.repository.listing import CanonicalListing
from mldb_v2.src.repository.mutation_coordination import StudyResultMutationCoordinator
from mldb_v2.src.repository.resolution import (
    CanonicalDocument,
    CanonicalEntityId,
    CanonicalRepositoryResolver,
)
from mldb_v2.src.results.study_result import StudyResult, StudyResultStatus, _validate_study_result
from mldb_v2.src.storage.object_bytes import _ObjectByteAccess
from mldb_v2.src.study.plan import StudyPlan
from mldb_v2.src.study.study_driver import advance_study as _advance_study
from mldb_v2.src.verification.definition_lifecycle import DefinitionLifecycleStatus

from ._authoring import AuthoringPlanningService
from ._errors import (
    _ApplicationBoundaryError,
    _application_error,
    _raise_application_error,
)
from ._execution import _ExecutionCompositionShell, _StudyResultOnlyValidator
from ._query import (
    ReadOnlyQueryService,
    _InvalidQueryRequest,
    _QueryBackendFailure,
    _UnsupportedQueryCapability,
)
from .application_interface import (
    AdvanceStudyResponse,
    CancelStudyResponse,
    DefinitionReport,
    DefinitionScope,
    SealResultItem,
)
from .query_interface import (
    BackendLogChunk,
    BackendLogRequest,
    DiagnosisCheck,
    EntityResource,
    StudyObservation,
    StudyResultView,
)

_T = TypeVar("_T")
_TERMINAL_STATUSES = {"completed", "completed_with_failures", "failed", "cancelled"}


class _BackendUnavailable(RuntimeError):
    """Bounded backend activation/call failure."""


class _ConfiguredBackends:
    def __init__(
        self,
        *,
        registry: BackendRegistry,
        configs: Mapping[str, BackendConfig] | None,
    ) -> None:
        if not isinstance(registry, BackendRegistry):
            raise TypeError("backend_registry must be a BackendRegistry")
        self._registry = registry
        self._configs = dict(configs or {})
        self._instances: dict[str, BackendPort] = {}
        for name, config in self._configs.items():
            if type(name) is not str or not name:
                raise ValueError("backend config names must be non-empty strings")
            if not isinstance(config, BackendConfig):
                raise TypeError("backend configs must contain BackendConfig values")

    def resolve(self, name: str) -> BackendPort:
        if type(name) is not str or not name:
            raise ValueError("backend name must be a non-empty string")
        existing = self._instances.get(name)
        if existing is not None:
            return existing
        config = self._configs.get(name)
        if config is None:
            config = BackendConfig(backend_type=name)
        try:
            backend = self._registry.resolve(config)
        except Exception as error:
            raise _BackendUnavailable("configured backend could not be resolved") from error
        self._instances[name] = backend
        return backend


def _validated_result(
    resolver: CanonicalRepositoryResolver,
    study_result: StudyResultId | str,
) -> StudyResult:
    validated_id = StudyResultId(_validate_typed_reference(study_result))
    document = resolver.resolve(kind=EntityKind.STUDY_RESULT, entity_id=validated_id)
    result = _validate_study_result(document)
    if result["id"] != validated_id:
        raise ValueError("resolved StudyResult identity mismatch")
    return result


class _QueryBackendRouter:
    """Resolve the exact StudyResult backend only when T007-02 needs one."""

    def __init__(
        self,
        *,
        mldb_data_root: Path,
        resolver: CanonicalRepositoryResolver,
        backends: _ConfiguredBackends,
    ) -> None:
        self._root = mldb_data_root
        self._resolver = resolver
        self._backends = backends

    def _backend(self, study_result: StudyResultId | str) -> BackendPort:
        result = _validated_result(self._resolver, study_result)
        return self._backends.resolve(result["backend"])

    def observe(self, *, stage_key: StageKey) -> BackendObservation | None:
        backend = self._backend(stage_key["study_result"])
        try:
            return backend.observe(stage_key=stage_key)
        except Exception as error:
            raise _BackendUnavailable("backend observation is unavailable") from error

    def admit(self, *, stage_input: StageInput) -> BackendObservation:
        raise AssertionError("query router must not admit work")

    def collect(self, *, stage_key: StageKey) -> TerminalCandidate | None:
        raise AssertionError("query router must not collect work")

    def cancel_study(self, *, study_result: StudyResultId) -> None:
        raise AssertionError("query router must not cancel work")

    def read_backend_logs(self, *, request: BackendLogRequest) -> Iterable[BackendLogChunk]:
        backend = self._backend(request["study_result"])
        service = ReadOnlyQueryService(mldb_data_root=self._root, backend=backend)
        try:
            return service.read_backend_logs(request=request)
        except _UnsupportedQueryCapability:
            raise
        except Exception as error:
            raise _BackendUnavailable("backend log access is unavailable") from error


class _CancellationRequestAdapter:
    """Application-owned cancellation request; W006 owns later progression."""
    def __init__(
        self,
        *,
        repository_root: Path,
        resolver: CanonicalRepositoryResolver,
        backends: _ConfiguredBackends,
    ) -> None:
        self._resolver = resolver
        self._backends = backends
        self._coordinator = StudyResultMutationCoordinator(repository_root)
        self._writer = CanonicalRepositoryWriter(
            repository_root,
            record_validator=_StudyResultOnlyValidator(),
        )

    def __call__(self, *, study_result: StudyResultId) -> Mapping[str, object]:
        result_id = StudyResultId(_validate_typed_reference(study_result))
        backend_name: str | None = None
        outcome: str

        with self._coordinator.acquire(study_result_id=result_id):
            current = _validated_result(self._resolver, result_id)
            status = current["status"]
            if status in _TERMINAL_STATUSES:
                return {
                    "study_result": result_id,
                    "outcome": "already_terminal",
                    "status": status,
                }
            backend_name = current["backend"]
            if status == "cancelling":
                outcome = "already_cancelling"
            elif status == "submitted":
                replacement = copy.deepcopy(current)
                replacement["status"] = "cancelling"
                stored = self._writer.replace_nonterminal_study_result(
                    entity_id=result_id,
                    replacement=replacement,
                )
                status = cast(StudyResultStatus, stored["status"])
                outcome = "accepted"
            else:
                raise ValueError("StudyResult status cannot accept cancellation")

        backend = self._backends.resolve(backend_name)
        try:
            backend.cancel_study(study_result=result_id)
        except Exception as error:
            raise _BackendUnavailable("backend cancellation is unavailable") from error

        latest = _validated_result(self._resolver, result_id)
        return {
            "study_result": result_id,
            "outcome": outcome,
            "status": latest["status"],
        }


class Application:
    """Concrete MLDB v2 Application composed over completed lower-layer seams."""

    def __init__(
        self,
        *,
        repository_root: str | Path,
        backend_registry: BackendRegistry,
        object_bytes: _ObjectByteAccess,
        backend_configs: Mapping[str, BackendConfig] | None = None,
        mldb_tests_root: str | Path | None = None,
        diagnostic_probes: Sequence[Callable[[], DiagnosisCheck]] = (),
        wait: Callable[[], None] = lambda: None,
    ) -> None:
        if not isinstance(object_bytes, _ObjectByteAccess):
            raise TypeError("object_bytes must be _ObjectByteAccess")
        self._repository_root = Path(repository_root)
        self._mldb_data_root = self._repository_root / "mldb_data"
        self._resolver = CanonicalRepositoryResolver(self._mldb_data_root)
        self._backends = _ConfiguredBackends(
            registry=backend_registry,
            configs=backend_configs,
        )
        self._object_bytes = object_bytes
        tests_root = (
            Path(mldb_tests_root)
            if mldb_tests_root is not None
            else self._repository_root / "mldb_v2" / "tests"
        )
        self._authoring = AuthoringPlanningService(
            repository_root=self._repository_root,
            mldb_data_root=self._mldb_data_root,
            mldb_tests_root=tests_root,
            object_access=self._object_bytes,
        )
        router = _QueryBackendRouter(
            mldb_data_root=self._mldb_data_root,
            resolver=self._resolver,
            backends=self._backends,
        )
        self._query = ReadOnlyQueryService(
            mldb_data_root=self._mldb_data_root,
            backend=cast(BackendPort, router),
            diagnostic_probes=diagnostic_probes,
        )
        cancellation = _CancellationRequestAdapter(
            repository_root=self._repository_root,
            resolver=self._resolver,
            backends=self._backends,
        )
        self._execution = _ExecutionCompositionShell(
            repository_root=self._repository_root,
            plan_study=self._authoring.plan_study,
            advance_one_pass=self._advance_one_pass,
            request_cancel=cancellation,
            wait=wait,
        )

    def _advance_one_pass(self, *, study_result: StudyResultId) -> AdvanceStudyResponse:
        current = _validated_result(self._resolver, study_result)
        backend = self._backends.resolve(current["backend"])
        return cast(
            AdvanceStudyResponse,
            _advance_study(
                repository_root=self._repository_root,
                study_result_id=study_result,
                backend=backend,
                object_bytes=self._object_bytes,
            ),
        )

    def _public(self, operation: Callable[[], _T]) -> _T:
        try:
            return operation()
        except _ApplicationBoundaryError:
            raise
        except _InvalidQueryRequest as error:
            raise _ApplicationBoundaryError(
                _application_error("invalid_request", "application request is invalid")
            ) from error
        except _UnsupportedQueryCapability as error:
            raise _ApplicationBoundaryError(
                _application_error("unsupported_capability", "requested capability is unsupported")
            ) from error
        except (_QueryBackendFailure, _BackendUnavailable) as error:
            raise _ApplicationBoundaryError(
                _application_error("backend_unavailable", "required backend operation is unavailable")
            ) from error
        except Exception as error:
            _raise_application_error(error)

    def validate_scope(
        self,
        *,
        scope: DefinitionScope | None = None,
    ) -> DefinitionReport:
        return self._public(lambda: self._authoring.validate_scope(scope=scope))

    def verify_scope(
        self,
        *,
        scope: DefinitionScope | None = None,
    ) -> DefinitionReport:
        return self._public(lambda: self._authoring.verify_scope(scope=scope))

    def seal_scope(
        self,
        *,
        scope: DefinitionScope,
        bulk: bool = False,
    ) -> list[SealResultItem]:
        return self._public(lambda: self._authoring.seal_scope(scope=scope, bulk=bulk))

    def plan_study(self, *, study: StudyId) -> StudyPlan:
        return self._public(lambda: self._authoring.plan_study(study=study))

    def start_study(
        self,
        *,
        plan: StudyPlanId,
        backend: str,
        execution_key: ExecutionKey,
    ) -> StudyResult:
        return self._public(
            lambda: self._execution.start_study(
                plan=plan,
                backend=backend,
                execution_key=execution_key,
            )
        )

    def advance_study(
        self,
        *,
        study_result: StudyResultId,
    ) -> AdvanceStudyResponse:
        return self._public(lambda: self._execution.advance_study(study_result=study_result))

    def run_study(
        self,
        *,
        study: StudyId,
        backend: str,
    ) -> StudyResult:
        return self._public(lambda: self._execution.run_study(study=study, backend=backend))

    def resume_study(
        self,
        *,
        study_result: StudyResultId,
    ) -> StudyResult:
        return self._public(lambda: self._execution.resume_study(study_result=study_result))

    def rerun_study(
        self,
        *,
        source: StudyResultId,
        backend: str | None = None,
    ) -> StudyResult:
        return self._public(
            lambda: self._execution.rerun_study(source=source, backend=backend)
        )

    def cancel_study(
        self,
        *,
        study_result: StudyResultId,
    ) -> CancelStudyResponse:
        return self._public(lambda: self._execution.cancel_study(study_result=study_result))

    def list_entities(
        self,
        *,
        resource: EntityResource,
        namespace: NamespaceId | None = None,
        status: DefinitionLifecycleStatus | None = None,
    ) -> CanonicalListing:
        return self._public(
            lambda: self._query.list_entities(
                resource=resource,
                namespace=namespace,
                status=status,
            )
        )

    def get_entity(
        self,
        *,
        kind: EntityKind,
        entity_id: CanonicalEntityId,
    ) -> CanonicalDocument:
        return self._public(lambda: self._query.get_entity(kind=kind, entity_id=entity_id))

    def list_study_results(
        self,
        *,
        namespace: NamespaceId | None = None,
        study: StudyId | None = None,
        statuses: Collection[StudyResultStatus] | None = None,
        created_at_from: datetime | None = None,
        created_at_to: datetime | None = None,
        limit: int | None = None,
    ) -> CanonicalListing:
        return self._public(
            lambda: self._query.list_study_results(
                namespace=namespace,
                study=study,
                statuses=statuses,
                created_at_from=created_at_from,
                created_at_to=created_at_to,
                limit=limit,
            )
        )

    def get_study_result(
        self,
        *,
        study_result: StudyResultId,
    ) -> StudyResultView:
        return self._public(lambda: self._query.get_study_result(study_result=study_result))

    def observe_study(
        self,
        *,
        study_result: StudyResultId,
    ) -> StudyObservation:
        return self._public(lambda: self._query.observe_study(study_result=study_result))

    def read_backend_logs(
        self,
        *,
        request: BackendLogRequest,
    ) -> Iterable[BackendLogChunk]:
        return self._public(lambda: self._query.read_backend_logs(request=request))

    def diagnose(self) -> list[DiagnosisCheck]:
        return self._public(self._query.diagnose)
