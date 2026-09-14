"""Exit-code policy for MLDB v2 CLI presentation."""
from __future__ import annotations

from collections.abc import Mapping, Sequence

EXIT_SUCCESS = 0
EXIT_FAILURE = 1


def exit_status_for_application_error(error: Mapping[str, object]) -> int:
    """Every ApplicationError is a command failure."""

    if type(error.get("code")) is not str or type(error.get("message")) is not str:
        raise ValueError("ApplicationError requires string code and message")
    return EXIT_FAILURE


def definition_report_has_failures(report: Mapping[str, object]) -> bool:
    """Return whether a validate/verify report contains any selected failure."""

    repository_issues = report.get("repository_issues")
    if not isinstance(repository_issues, Sequence) or isinstance(repository_issues, (str, bytes)):
        raise ValueError("DefinitionReport repository_issues must be a sequence")
    if repository_issues:
        return True

    items = report.get("items")
    if not isinstance(items, Sequence) or isinstance(items, (str, bytes)):
        raise ValueError("DefinitionReport items must be a sequence")
    for item in items:
        if not isinstance(item, Mapping) or item.get("valid") is not True:
            return True
    return False


def exit_status_for_definition_report(report: Mapping[str, object]) -> int:
    """Validate/verify succeeds only when the complete selected report is clean."""

    return EXIT_FAILURE if definition_report_has_failures(report) else EXIT_SUCCESS
