"""Shared MLDB v2 persisted diagnostic runtime value and validation."""

import re as _re
from typing import TypedDict


class Diagnostic(TypedDict):
    code: str
    message: str


_DIAGNOSTIC_CODE_RE = _re.compile(r"[a-z][a-z0-9]*(?:_[a-z0-9]+)*", _re.ASCII)
_MAX_DIAGNOSTIC_MESSAGE_BYTES = 4096


def _validate_diagnostic(value: object) -> Diagnostic | None:
    if value is None:
        return None
    if type(value) is not dict or set(value) != {"code", "message"}:
        raise ValueError("diagnostic must contain exactly code and message")

    code = value["code"]
    message = value["message"]
    if type(code) is not str or _DIAGNOSTIC_CODE_RE.fullmatch(code) is None:
        raise ValueError("invalid diagnostic code")
    if type(message) is not str:
        raise ValueError("diagnostic message must be a string")
    if len(message.encode("utf-8")) > _MAX_DIAGNOSTIC_MESSAGE_BYTES:
        raise ValueError("diagnostic message exceeds 4096 UTF-8 bytes")
    return value
