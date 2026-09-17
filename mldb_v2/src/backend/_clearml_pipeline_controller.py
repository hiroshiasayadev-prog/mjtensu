"""ClearML Pipeline controller Task marker used by the W011 UI projection.

The controller Task is created/recovered as the Study-level operational container.
MLDB semantic reconciliation releases logical stages; the ClearML backend then
creates/binds/enqueues the child Task under this Pipeline. The controller marker
itself is not a supported direct execution entrypoint.
"""

from __future__ import annotations


class ClearMLPipelineControllerError(RuntimeError):
    """Raised when the Pipeline marker is executed as if it were a child worker."""


def main() -> None:
    raise ClearMLPipelineControllerError(
        "MLDB ClearML Pipeline controller is a Study-level UI/ownership marker and is not directly executable"
    )


if __name__ == "__main__":
    main()
