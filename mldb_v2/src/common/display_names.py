"""User-facing display-name safety shared across MLDB domains."""

from __future__ import annotations

_FORBIDDEN_DISPLAY_SUBSTRINGS = ("trial-",)


def validate_user_facing_display_name(value: object, *, field: str = "display name") -> str:
    if type(value) is not str or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    lowered = value.lower()
    forbidden = next(
        (token for token in _FORBIDDEN_DISPLAY_SUBSTRINGS if token in lowered),
        None,
    )
    if forbidden is not None:
        raise ValueError(
            f"{field} must not expose internal identifier token {forbidden!r}"
        )
    return value
