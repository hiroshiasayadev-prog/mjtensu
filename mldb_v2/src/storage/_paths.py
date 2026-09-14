"""Cross-platform lexical validation for canonical relative object paths."""

from __future__ import annotations

import re as _re


_WINDOWS_DRIVE_RE = _re.compile(r"[A-Za-z]:")


def _validate_safe_relative_path(value: object, *, label: str = "path") -> str:
    if type(value) is not str or not value:
        raise ValueError(f"{label} must be a non-empty string")
    if "\\" in value:
        raise ValueError(f"{label} must use '/' separators")
    if value.startswith("/") or _WINDOWS_DRIVE_RE.match(value):
        raise ValueError(f"{label} must be relative")
    if "\x00" in value:
        raise ValueError(f"{label} contains a NUL byte")

    parts = value.split("/")
    if any(part in {"", ".", ".."} for part in parts):
        raise ValueError(f"{label} contains an unsafe path segment")
    if any(":" in part for part in parts):
        raise ValueError(f"{label} is not portable across repository platforms")
    return value
