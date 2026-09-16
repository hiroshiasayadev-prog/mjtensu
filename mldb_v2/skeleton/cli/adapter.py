"""Transport-independent delegation boundary for the MLDB v2 CLI."""

from typing import Protocol

from mldb_v2.skeleton.api.application_interface import ApplicationInterface
from mldb_v2.skeleton.cli.requests import CliCommandRequest


class CliApplicationAdapter(Protocol):
    """Dispatch normalized CLI requests through the public Application API only.

    Unsupported selector/resource combinations are rejected, never silently ignored. ``ps``, ID-less
    ``get``/``status``, and ID-less ``watch`` remain discovery operations without filesystem lookup.
    ``watch`` is strictly query/observation-only and never progresses or writes canonical state.

    ``run`` and ``resume`` delegate to foreground Study-driver operations; ``advance`` is the explicit
    one-pass progression command. Repository, BackendPort, ClearML, YAML/filesystem, polling,
    terminal rendering, and parser construction are outside this boundary.

    Dispatch returns the Application API value for the selected operation; CLI does not redefine
    canonical response shapes.
    """

    def dispatch(
        self,
        *,
        application: ApplicationInterface,
        request: CliCommandRequest,
    ) -> object: ...
