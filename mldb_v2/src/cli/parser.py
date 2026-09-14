"""Argument parsing and normalization for the MLDB v2 CLI."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from typing import cast, get_args

from mldb_v2.src.cli.requests import CliCommandRequest
from mldb_v2.src.cli.types import (
    CliCommandName,
    CliDefinitionKind,
    CliPresentation,
    CliResource,
    CliSinceSelector,
    CliStatusSelector,
    CommonSelectors,
    OutputFormat,
)
from mldb_v2.src.common.ids import (
    EvaluationCoordinateId,
    NamespaceId,
    StudyId,
    StudyResultId,
    TrialId,
    TypedEntityId,
    _validate_evaluation_coordinate_id,
    _validate_namespace_id,
    _validate_trial_id,
    _validate_typed_reference,
)
from mldb_v2.src.results.study_result import StudyResultStatus
from mldb_v2.src.verification.definition_lifecycle import (
    DefinitionLifecycleStatus,
    SealableDefinitionId,
)


class _ArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        raise ValueError(message)


def _positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be a positive integer") from error
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _namespace_id(value: str) -> NamespaceId:
    try:
        return _validate_namespace_id(value)
    except ValueError as error:
        raise ValueError(str(error)) from error


def _typed_reference(value: str) -> str:
    try:
        return _validate_typed_reference(value)
    except ValueError as error:
        raise ValueError(str(error)) from error


def _trial_id(value: str) -> TrialId:
    try:
        return _validate_trial_id(value)
    except ValueError as error:
        raise ValueError(str(error)) from error


def _coordinate_id(value: str) -> EvaluationCoordinateId:
    try:
        return _validate_evaluation_coordinate_id(value)
    except ValueError as error:
        raise ValueError(str(error)) from error


def _add_output(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("-o", "--output", type=OutputFormat, choices=tuple(OutputFormat))
    parser.add_argument("--json", action="store_true")


def _add_common_selectors(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--namespace", type=_namespace_id)
    parser.add_argument("--status")
    parser.add_argument("--study", type=_typed_reference)
    parser.add_argument("--since")
    parser.add_argument("--limit", type=_positive_int)


def _presentation(args: argparse.Namespace) -> CliPresentation | None:
    output = getattr(args, "output", None)
    json_alias = getattr(args, "json", False)
    if json_alias:
        if output is not None and output is not OutputFormat.JSON:
            raise ValueError("--json conflicts with non-json -o/--output")
        output = OutputFormat.JSON
    if output is None:
        return None
    return {"format": output}


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
_STUDY_RESULT_RESOURCES = frozenset({CliResource.RUNS})
_NAMESPACE_ONLY_RESOURCES = frozenset(
    {
        CliResource.NAMESPACES,
        CliResource.PLANS,
        CliResource.TRAINING_RESULTS,
        CliResource.MODELS,
        CliResource.EVALUATION_RESULTS,
    }
)
_LIFECYCLE_STATUSES = frozenset(get_args(DefinitionLifecycleStatus))
_STUDY_RESULT_STATUSES = frozenset(get_args(StudyResultStatus))

def _common_selectors(
    args: argparse.Namespace,
    *,
    allowed: frozenset[str],
    statuses: frozenset[str] | None = None,
) -> CommonSelectors | None:
    raw = {
        "namespace": getattr(args, "namespace", None),
        "status": getattr(args, "status", None),
        "study": getattr(args, "study", None),
        "since": getattr(args, "since", None),
        "limit": getattr(args, "limit", None),
    }
    unsupported = sorted(key for key, value in raw.items() if value is not None and key not in allowed)
    if unsupported:
        joined = ", ".join(f"--{name}" for name in unsupported)
        raise ValueError(f"unsupported selector(s) for requested resource: {joined}")
    if raw["status"] is not None and statuses is not None and raw["status"] not in statuses:
        raise ValueError(f"unsupported status selector: {raw['status']!r}")

    selectors: CommonSelectors = {}
    if raw["namespace"] is not None:
        selectors["namespace"] = raw["namespace"]
    if raw["status"] is not None:
        selectors["status"] = CliStatusSelector(raw["status"])
    if raw["study"] is not None:
        selectors["study"] = StudyId(raw["study"])
    if raw["since"] is not None:
        selectors["since"] = CliSinceSelector(raw["since"])
    if raw["limit"] is not None:
        selectors["limit"] = raw["limit"]
    return selectors or None

def _selectors_for_get(args: argparse.Namespace, resource: CliResource) -> CommonSelectors | None:
    if resource in _DEFINITION_RESOURCES:
        allowed = frozenset({"namespace", "status"})
        statuses = _LIFECYCLE_STATUSES
    elif resource in _STUDY_RESULT_RESOURCES:
        allowed = frozenset({"namespace", "status", "study", "since", "limit"})
        statuses = _STUDY_RESULT_STATUSES
    elif resource in _NAMESPACE_ONLY_RESOURCES:
        allowed = frozenset({"namespace"})
        statuses = None
    else:
        raise AssertionError(f"unhandled CLI resource: {resource}")
    return _common_selectors(args, allowed=allowed, statuses=statuses)


def _exact_get_id(resource: CliResource, value: str) -> NamespaceId | TypedEntityId:
    if resource is CliResource.NAMESPACES:
        return _namespace_id(value)
    if resource is CliResource.DEFINITIONS:
        raise ValueError("aggregate definitions view cannot resolve an exact ID without an explicit kind")
    return cast(TypedEntityId, _typed_reference(value))


def _definition_id(value: str) -> SealableDefinitionId:
    return cast(SealableDefinitionId, _typed_reference(value))


def _study_id(value: str) -> StudyId:
    return StudyId(_typed_reference(value))


def _study_result_id(value: str) -> StudyResultId:
    return StudyResultId(_typed_reference(value))


def _scope_namespace_matches(typed_id: str, namespace: NamespaceId | None) -> None:
    if namespace is not None and typed_id.split("/", 1)[0] != namespace:
        raise ValueError("typed ID namespace conflicts with --namespace")


def _attach_presentation(request: dict[str, object], args: argparse.Namespace) -> None:
    presentation = _presentation(args)
    if presentation is not None:
        request["presentation"] = presentation

def _build_parser() -> _ArgumentParser:
    parser = _ArgumentParser(prog="mldb")
    subparsers = parser.add_subparsers(dest="command", required=True)

    ps = subparsers.add_parser(CliCommandName.PS.value)
    ps.add_argument("--all", action="store_true")
    _add_common_selectors(ps)
    _add_output(ps)

    get = subparsers.add_parser(CliCommandName.GET.value)
    get.add_argument("resource", type=CliResource, choices=tuple(CliResource))
    get.add_argument("typed_id", nargs="?")
    _add_common_selectors(get)
    _add_output(get)

    describe = subparsers.add_parser(CliCommandName.DESCRIBE.value)
    describe.add_argument("resource", type=CliResource, choices=tuple(CliResource))
    describe.add_argument("typed_id")
    _add_output(describe)

    status = subparsers.add_parser(CliCommandName.STATUS.value)
    status.add_argument("study_result", nargs="?")
    _add_output(status)

    for name in (CliCommandName.VALIDATE, CliCommandName.VERIFY):
        check = subparsers.add_parser(name.value)
        check.add_argument("kind", nargs="?", type=CliDefinitionKind, choices=tuple(CliDefinitionKind))
        check.add_argument("typed_id", nargs="?")
        check.add_argument("--namespace", type=_namespace_id)
        check.add_argument("--fail-fast", action="store_true")
        _add_output(check)
    seal = subparsers.add_parser(CliCommandName.SEAL.value)
    seal.add_argument("kind", nargs="?", type=CliDefinitionKind, choices=tuple(CliDefinitionKind))
    seal.add_argument("typed_id", nargs="?")
    seal.add_argument("--namespace", type=_namespace_id)
    seal.add_argument("--all", action="store_true")

    plan = subparsers.add_parser(CliCommandName.PLAN.value)
    plan.add_argument("study")

    run = subparsers.add_parser(CliCommandName.RUN.value)
    run.add_argument("study")
    run.add_argument("--backend")

    resume = subparsers.add_parser(CliCommandName.RESUME.value)
    resume.add_argument("study_result")

    rerun = subparsers.add_parser(CliCommandName.RERUN.value)
    rerun.add_argument("study_result")
    rerun.add_argument("--backend")

    cancel = subparsers.add_parser(CliCommandName.CANCEL.value)
    cancel.add_argument("study_result")

    advance = subparsers.add_parser(CliCommandName.ADVANCE.value)
    advance.add_argument("study_result")

    watch = subparsers.add_parser(CliCommandName.WATCH.value)
    watch.add_argument("study_result", nargs="?")
    _add_common_selectors(watch)
    _add_output(watch)
    logs = subparsers.add_parser(CliCommandName.LOGS.value)
    logs.add_argument("study_result")
    logs.add_argument("--trial", type=_trial_id)
    logs.add_argument("--stage", type=_coordinate_id)
    logs.add_argument("--failed", action="store_true")
    logs.add_argument("-f", "--follow", action="store_true")
    _add_output(logs)

    doctor = subparsers.add_parser(CliCommandName.DOCTOR.value)
    _add_output(doctor)

    return parser


def parse_cli_request(argv: Sequence[str]) -> CliCommandRequest:
    """Parse ``argv`` (excluding the program name) into one normalized frozen request shape."""

    args = _build_parser().parse_args(list(argv))
    command = CliCommandName(args.command)

    if command is CliCommandName.PS:
        request: dict[str, object] = {"command": command, "all": args.all}
        selectors = _common_selectors(
            args,
            allowed=frozenset({"namespace", "status", "study", "since", "limit"}),
            statuses=_STUDY_RESULT_STATUSES,
        )
        if selectors is not None:
            request["selectors"] = selectors
        _attach_presentation(request, args)
        return cast(CliCommandRequest, request)

    if command is CliCommandName.GET:
        resource = cast(CliResource, args.resource)
        request = {"command": command, "resource": resource}
        selectors = _selectors_for_get(args, resource)
        if args.typed_id is not None:
            if selectors is not None:
                raise ValueError("selectors are not valid with an exact get target")
            request["typed_id"] = _exact_get_id(resource, args.typed_id)
        elif selectors is not None:
            request["selectors"] = selectors
        _attach_presentation(request, args)
        return cast(CliCommandRequest, request)

    if command is CliCommandName.DESCRIBE:
        resource = cast(CliResource, args.resource)
        request = {
            "command": command,
            "resource": resource,
            "typed_id": _exact_get_id(resource, args.typed_id),
        }
        _attach_presentation(request, args)
        return cast(CliCommandRequest, request)

    if command is CliCommandName.STATUS:
        request = {"command": command}
        if args.study_result is not None:
            request["study_result"] = _study_result_id(args.study_result)
        _attach_presentation(request, args)
        return cast(CliCommandRequest, request)

    if command in {CliCommandName.VALIDATE, CliCommandName.VERIFY}:
        request = {"command": command, "fail_fast": args.fail_fast}
        if args.kind is not None:
            request["kind"] = args.kind
        if args.typed_id is not None:
            typed_id = _definition_id(args.typed_id)
            _scope_namespace_matches(str(typed_id), args.namespace)
            request["typed_id"] = typed_id
        if args.namespace is not None:
            request["namespace"] = args.namespace
        _attach_presentation(request, args)
        return cast(CliCommandRequest, request)
    if command is CliCommandName.SEAL:
        if args.typed_id is not None:
            if args.kind is None:
                raise ValueError("exact seal requires an explicit definition kind")
            if args.all or args.namespace is not None:
                raise ValueError("exact seal cannot be combined with --all or --namespace")
            return cast(
                CliCommandRequest,
                {"command": command, "kind": args.kind, "typed_id": _definition_id(args.typed_id)},
            )
        if not args.all:
            raise ValueError("seal without an exact ID requires explicit --all")
        if args.namespace is None:
            raise ValueError("bulk seal requires --namespace")
        request = {"command": command, "namespace": args.namespace, "all": True}
        if args.kind is not None:
            request["kind"] = args.kind
        return cast(CliCommandRequest, request)

    if command is CliCommandName.PLAN:
        return cast(CliCommandRequest, {"command": command, "study": _study_id(args.study)})

    if command is CliCommandName.RUN:
        request = {"command": command, "study": _study_id(args.study)}
        if args.backend is not None:
            request["backend"] = args.backend
        return cast(CliCommandRequest, request)

    if command is CliCommandName.RESUME:
        return cast(
            CliCommandRequest,
            {"command": command, "study_result": _study_result_id(args.study_result)},
        )

    if command is CliCommandName.RERUN:
        request = {"command": command, "study_result": _study_result_id(args.study_result)}
        if args.backend is not None:
            request["backend"] = args.backend
        return cast(CliCommandRequest, request)
    if command is CliCommandName.CANCEL:
        return cast(
            CliCommandRequest,
            {"command": command, "study_result": _study_result_id(args.study_result)},
        )

    if command is CliCommandName.ADVANCE:
        return cast(
            CliCommandRequest,
            {"command": command, "study_result": _study_result_id(args.study_result)},
        )

    if command is CliCommandName.WATCH:
        request = {"command": command}
        selectors = _common_selectors(
            args,
            allowed=frozenset({"namespace", "status", "study", "since", "limit"}),
            statuses=_STUDY_RESULT_STATUSES,
        )
        if args.study_result is not None:
            if selectors is not None:
                raise ValueError("selectors are not valid with an exact watch target")
            request["study_result"] = _study_result_id(args.study_result)
        elif selectors is not None:
            request["selectors"] = selectors
        _attach_presentation(request, args)
        return cast(CliCommandRequest, request)

    if command is CliCommandName.LOGS:
        request = {
            "command": command,
            "study_result": _study_result_id(args.study_result),
            "failed_only": args.failed,
            "follow": args.follow,
        }
        if args.trial is not None:
            request["trial"] = args.trial
        if args.stage is not None:
            request["stage"] = args.stage
        _attach_presentation(request, args)
        return cast(CliCommandRequest, request)
    if command is CliCommandName.DOCTOR:
        request = {"command": command}
        _attach_presentation(request, args)
        return cast(CliCommandRequest, request)

    raise AssertionError(f"unhandled CLI command: {command}")
