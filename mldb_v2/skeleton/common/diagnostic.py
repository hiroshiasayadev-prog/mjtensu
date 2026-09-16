"""Shared MLDB v2 persisted diagnostic shape."""

from typing import TypedDict


class Diagnostic(TypedDict):
    code: str
    message: str
