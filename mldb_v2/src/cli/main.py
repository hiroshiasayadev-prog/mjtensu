"""Runtime composition for the MLDB v2 command-line adapter."""

from __future__ import annotations

import os
import sys
from collections.abc import Callable, Mapping, Sequence
from typing import TextIO, cast

from mldb_v2.src.api import ApplicationInterface, compose_application
from mldb_v2.src.cli.adapter import CliApplicationAdapter
from mldb_v2.src.cli.exit_status import (
    EXIT_SUCCESS,
    exit_status_for_application_error,
    exit_status_for_definition_report,
)
from mldb_v2.src.cli.parser import parse_cli_request
from mldb_v2.src.cli.rendering import RenderedOutput, render_application_error, render_success
from mldb_v2.src.cli.requests import CliCommandRequest
from mldb_v2.src.cli.types import CliCommandName, OutputFormat

ApplicationFactory = Callable[[], ApplicationInterface]
_INTERRUPT_EXIT = 130


def _output_format(request: CliCommandRequest) -> str:
    presentation = request.get("presentation")
    if presentation is None:
        return OutputFormat.TABLE.value
    selected = presentation.get("format")
    if isinstance(selected, OutputFormat):
        return selected.value
    if type(selected) is str:
        return selected
    raise ValueError("CLI presentation format is invalid")


def _production_application() -> tuple[ApplicationInterface, str | None]:
    repository_root = os.environ.get("MLDB_REPO_ROOT")
    if not repository_root:
        raise ValueError("MLDB_REPO_ROOT is not configured")
    composition = compose_application(repository_root=repository_root)
    return composition.application, composition.default_backend


def _resolve_runtime_request(
    request: CliCommandRequest,
    *,
    default_backend: str | None,
) -> CliCommandRequest:
    if request["command"] is CliCommandName.RUN and request.get("backend") is None:
        if default_backend is None:
            raise ValueError(
                "run requires --backend when no default backend is configured"
            )
        copied = dict(request)
        copied["backend"] = default_backend
        return cast(CliCommandRequest, copied)
    return request


def _dispatch_request(request: CliCommandRequest) -> CliCommandRequest:
    """Apply CLI-only control semantics without changing Application behavior."""

    if request["command"] in {CliCommandName.VALIDATE, CliCommandName.VERIFY}:
        if request.get("fail_fast") is True:
            copied = dict(request)
            copied["fail_fast"] = False
            return cast(CliCommandRequest, copied)
    return request


def _application_error_from_exception(error: BaseException) -> dict[str, str]:
    public = getattr(error, "error", None)
    if isinstance(public, Mapping):
        code = public.get("code")
        message = public.get("message")
        if type(code) is str and type(message) is str:
            return {"code": code, "message": message}
    if isinstance(error, ValueError):
        return {"code": "invalid_request", "message": str(error)}
    return {"code": "internal_failure", "message": "unexpected application failure"}


def _emit(rendered: RenderedOutput, *, stdout: TextIO, stderr: TextIO) -> None:
    if rendered.stdout:
        stdout.write(rendered.stdout)
    if rendered.stderr:
        stderr.write(rendered.stderr)


def _result_exit_status(request: CliCommandRequest, result: object) -> int:
    if request["command"] in {CliCommandName.VALIDATE, CliCommandName.VERIFY}:
        if not isinstance(result, Mapping):
            raise ValueError("definition report must be a mapping")
        return exit_status_for_definition_report(cast(Mapping[str, object], result))
    return EXIT_SUCCESS


def main(
    argv: Sequence[str] | None = None,
    *,
    application_factory: ApplicationFactory | None = None,
    default_backend: str | None = None,
    stdout: TextIO | None = None,
    stderr: TextIO | None = None,
) -> int:
    """Run one CLI command using an injected Application construction boundary."""

    args = list(sys.argv[1:] if argv is None else argv)
    out = sys.stdout if stdout is None else stdout
    err = sys.stderr if stderr is None else stderr

    try:
        request = parse_cli_request(args)
        if application_factory is None:
            application, runtime_default_backend = _production_application()
        else:
            application = application_factory()
            runtime_default_backend = default_backend
        runtime_request = _resolve_runtime_request(
            request,
            default_backend=runtime_default_backend,
        )
        dispatch_request = _dispatch_request(runtime_request)
        result = CliApplicationAdapter().dispatch(
            application=application,
            request=dispatch_request,
        )
        rendered = render_success(result, _output_format(request))
        _emit(rendered, stdout=out, stderr=err)
        return _result_exit_status(request, result)
    except KeyboardInterrupt:
        err.write("interrupted\n")
        return _INTERRUPT_EXIT
    except Exception as error:
        public_error = _application_error_from_exception(error)
        rendered = render_application_error(public_error)
        _emit(rendered, stdout=out, stderr=err)
        return exit_status_for_application_error(public_error)
