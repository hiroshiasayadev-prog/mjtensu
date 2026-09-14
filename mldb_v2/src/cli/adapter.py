"""Thin normalized-request dispatch from the CLI to the public Application API."""

from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone
from typing import cast

from mldb_v2.src.api.application_interface import ApplicationInterface, DefinitionScope
from mldb_v2.src.api.query_interface import BackendLogRequest
from mldb_v2.src.cli.requests import CliCommandRequest
from mldb_v2.src.cli.types import CliCommandName, CliDefinitionKind, CliResource, CommonSelectors
from mldb_v2.src.common.ids import DefinitionKind, EntityKind
from mldb_v2.src.results.study_result import StudyResultStatus
from mldb_v2.src.verification.definition_lifecycle import DefinitionLifecycleStatus


_RESOURCE_KINDS = {
    CliResource.NAMESPACES: EntityKind.NAMESPACE,
    CliResource.TASKS: EntityKind.TASK,
    CliResource.CORPORA: EntityKind.CORPUS,
    CliResource.ARCHITECTURES: EntityKind.ARCHITECTURE,
    CliResource.TRAIN_PROTOCOLS: EntityKind.TRAIN_PROTOCOL,
    CliResource.EVALUATION_PROTOCOLS: EntityKind.EVALUATION_PROTOCOL,
    CliResource.STUDIES: EntityKind.STUDY,
    CliResource.PLANS: EntityKind.STUDY_PLAN,
    CliResource.RUNS: EntityKind.STUDY_RESULT,
    CliResource.TRAINING_RESULTS: EntityKind.TRAINING_RESULT,
    CliResource.MODELS: EntityKind.MODEL,
    CliResource.EVALUATION_RESULTS: EntityKind.EVALUATION_RESULT,
}

_DEFINITION_KINDS = {
    CliDefinitionKind.TASK: DefinitionKind.TASK,
    CliDefinitionKind.CORPUS: DefinitionKind.CORPUS,
    CliDefinitionKind.ARCHITECTURE: DefinitionKind.ARCHITECTURE,
    CliDefinitionKind.TRAIN_PROTOCOL: DefinitionKind.TRAIN_PROTOCOL,
    CliDefinitionKind.EVALUATION_PROTOCOL: DefinitionKind.EVALUATION_PROTOCOL,
    CliDefinitionKind.STUDY: DefinitionKind.STUDY,
}
_DEFINITION_RESOURCES = frozenset(
    {
        CliResource.DEFINITIONS,
        CliResource.TASKS,
        CliResource.CORPORA,
        CliResource.ARCHITECTURES,
        CliResource.TRAIN_PROTOCOLS,
        CliResource.EVALUATION_PROTOCOLS,
        CliResource.STUDIES,
    }
)
_ACTIVE_STATUSES: tuple[StudyResultStatus, ...] = ("submitted", "cancelling")
_TERMINAL_STATUSES = frozenset(
    {"completed", "completed_with_failures", "failed", "cancelled"}
)
_DURATION_RE = re.compile(r"([1-9][0-9]*)([smhdw])\Z")
_DURATION_SECONDS = {"s": 1, "m": 60, "h": 3600, "d": 86400, "w": 604800}

def _definition_kind(value: CliDefinitionKind) -> DefinitionKind:
    try:
        return _DEFINITION_KINDS[value]
    except KeyError as error:
        raise ValueError(f"unsupported definition kind: {value!r}") from error


def _since_to_datetime(value: str) -> datetime:
    match = _DURATION_RE.fullmatch(value)
    if match is not None:
        amount = int(match.group(1))
        seconds = amount * _DURATION_SECONDS[match.group(2)]
        return datetime.now(timezone.utc) - timedelta(seconds=seconds)
    if not value.endswith("Z"):
        raise ValueError("--since must be an RFC3339 UTC time or integer duration like 30m/2h/7d")
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as error:
        raise ValueError("--since must be a valid RFC3339 UTC time") from error
    if parsed.utcoffset() != timezone.utc.utcoffset(parsed):
        raise ValueError("--since must use UTC")
    return parsed


def _study_listing_kwargs(
    selectors: CommonSelectors | None,
    *,
    active_only: bool,
) -> dict[str, object]:
    selectors = selectors or {}
    status = selectors.get("status")
    if active_only and status in _TERMINAL_STATUSES:
        raise ValueError("terminal --status requires ps --all or get runs")
    statuses = (
        (cast(StudyResultStatus, str(status)),)
        if status is not None
        else (_ACTIVE_STATUSES if active_only else None)
    )
    since = selectors.get("since")
    return {
        "namespace": selectors.get("namespace"),
        "study": selectors.get("study"),
        "statuses": statuses,
        "created_at_from": None if since is None else _since_to_datetime(str(since)),
        "created_at_to": None,
        "limit": selectors.get("limit"),
    }


def _entity_listing(
    application: ApplicationInterface,
    *,
    resource: CliResource,
    selectors: CommonSelectors | None,
) -> object:
    selectors = selectors or {}
    unsupported = set(selectors) - {"namespace", "status"}
    if unsupported:
        raise ValueError(f"unsupported selector(s) for {resource.value}: {sorted(unsupported)!r}")

    status = selectors.get("status")
    if status is not None and resource not in _DEFINITION_RESOURCES:
        raise ValueError(f"--status is unsupported for {resource.value}")
    lifecycle_status = (
        None if status is None else cast(DefinitionLifecycleStatus, str(status))
    )
    api_resource: EntityKind | str
    if resource is CliResource.DEFINITIONS:
        api_resource = "definitions"
    else:
        try:
            api_resource = _RESOURCE_KINDS[resource]
        except KeyError as error:
            raise ValueError(f"unsupported resource: {resource.value}") from error
    return application.list_entities(
        resource=cast(object, api_resource),
        namespace=selectors.get("namespace"),
        status=lifecycle_status,
    )


def _exact_entity(
    application: ApplicationInterface,
    *,
    resource: CliResource,
    entity_id: object,
) -> object:
    if resource is CliResource.DEFINITIONS:
        raise ValueError("aggregate definitions view has no exact-ID operation")
    try:
        kind = _RESOURCE_KINDS[resource]
    except KeyError as error:
        raise ValueError(f"unsupported resource: {resource.value}") from error
    return application.get_entity(kind=kind, entity_id=cast(object, entity_id))


def _definition_scope(request: dict[str, object]) -> DefinitionScope | None:
    scope: DefinitionScope = {}
    kind = request.get("kind")
    typed_id = request.get("typed_id")
    namespace = request.get("namespace")
    if typed_id is not None and kind is None:
        raise ValueError("exact definition target requires an explicit kind")
    if kind is not None:
        scope["kind"] = _definition_kind(cast(CliDefinitionKind, kind))
    if typed_id is not None:
        scope["id"] = cast(object, typed_id)
        if namespace is not None and str(typed_id).split("/", 1)[0] != namespace:
            raise ValueError("typed ID namespace conflicts with namespace selector")
    if namespace is not None:
        scope["namespace"] = cast(object, namespace)
    return scope or None


class CliApplicationAdapter:
    """Dispatch normalized CLI requests through ``ApplicationInterface`` only."""

    def dispatch(
        self,
        *,
        application: ApplicationInterface,
        request: CliCommandRequest,
    ) -> object:
        command = request["command"]

        if command is CliCommandName.PS:
            all_results = request["all"]
            if type(all_results) is not bool:
                raise ValueError("ps all must be boolean")
            kwargs = _study_listing_kwargs(
                request.get("selectors"), active_only=not all_results
            )
            return application.list_study_results(**kwargs)

        if command is CliCommandName.GET:
            resource = request["resource"]
            typed_id = request.get("typed_id")
            selectors = request.get("selectors")
            if typed_id is not None:
                if selectors is not None:
                    raise ValueError("selectors are not valid with an exact get target")
                return _exact_entity(application, resource=resource, entity_id=typed_id)
            if resource is CliResource.RUNS:
                kwargs = _study_listing_kwargs(selectors, active_only=False)
                return application.list_study_results(**kwargs)
            return _entity_listing(application, resource=resource, selectors=selectors)

        if command is CliCommandName.DESCRIBE:
            return _exact_entity(
                application,
                resource=request["resource"],
                entity_id=request["typed_id"],
            )

        if command is CliCommandName.STATUS:
            study_result = request.get("study_result")
            if study_result is None:
                return application.list_study_results(
                    **_study_listing_kwargs(None, active_only=True)
                )
            return application.observe_study(study_result=study_result)

        if command in {CliCommandName.VALIDATE, CliCommandName.VERIFY}:
            raw_request = cast(dict[str, object], request)
            if raw_request.get("fail_fast") is True:
                raise ValueError("--fail-fast has no ApplicationInterface delegation seam")
            scope = _definition_scope(raw_request)
            if command is CliCommandName.VALIDATE:
                return application.validate_scope(scope=scope)
            return application.verify_scope(scope=scope)

        if command is CliCommandName.SEAL:
            raw_request = cast(dict[str, object], request)
            typed_id = raw_request.get("typed_id")
            if typed_id is not None:
                if raw_request.get("all") is True or raw_request.get("namespace") is not None:
                    raise ValueError("exact seal cannot be combined with bulk selectors")
                scope = _definition_scope(raw_request)
                if scope is None:
                    raise ValueError("exact seal requires a target")
                return application.seal_scope(scope=scope, bulk=False)
            if raw_request.get("all") is not True or raw_request.get("namespace") is None:
                raise ValueError("bulk seal requires explicit all and namespace")
            scope = _definition_scope(raw_request)
            if scope is None:
                raise ValueError("bulk seal requires a scope")
            return application.seal_scope(scope=scope, bulk=True)

        if command is CliCommandName.PLAN:
            return application.plan_study(study=request["study"])

        if command is CliCommandName.RUN:
            backend = request.get("backend")
            if backend is None:
                raise ValueError(
                    "run omitted --backend, but ApplicationInterface.run_study has no default-backend seam"
                )
            return application.run_study(study=request["study"], backend=backend)

        if command is CliCommandName.RESUME:
            return application.resume_study(study_result=request["study_result"])

        if command is CliCommandName.RERUN:
            return application.rerun_study(
                source=request["study_result"],
                backend=request.get("backend"),
            )

        if command is CliCommandName.CANCEL:
            return application.cancel_study(study_result=request["study_result"])

        if command is CliCommandName.ADVANCE:
            return application.advance_study(study_result=request["study_result"])

        if command is CliCommandName.WATCH:
            study_result = request.get("study_result")
            selectors = request.get("selectors")
            if study_result is not None:
                if selectors is not None:
                    raise ValueError("selectors are not valid with an exact watch target")
                return application.observe_study(study_result=study_result)
            return application.list_study_results(
                **_study_listing_kwargs(selectors, active_only=True)
            )

        if command is CliCommandName.LOGS:
            log_request: BackendLogRequest = {
                "study_result": request["study_result"],
                "trial": request.get("trial"),
                "coordinate": request.get("stage"),
                "failed_only": request["failed_only"],
                "follow": request["follow"],
            }
            return application.read_backend_logs(request=log_request)

        if command is CliCommandName.DOCTOR:
            return application.diagnose()

        raise ValueError(f"unsupported CLI command: {command!r}")
