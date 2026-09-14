"""Presentation-only rendering helpers for the MLDB v2 CLI."""
from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any


class YamlSerializerUnavailable(RuntimeError):
    """Raised when YAML output is requested without a repository-supported serializer."""


@dataclass(frozen=True)
class RenderedOutput:
    stdout: str = ""
    stderr: str = ""


def render_value(value: object, output_format: str) -> str:
    """Render an application value without changing its semantic fields."""

    if output_format == "json":
        return _render_json(value)
    if output_format == "yaml":
        return _render_yaml(value)
    if output_format == "table":
        return _render_table(value, wide=False)
    if output_format == "wide":
        return _render_table(value, wide=True)
    raise ValueError(f"unsupported output format: {output_format}")


def render_success(value: object, output_format: str) -> RenderedOutput:
    """Return success output on stdout only."""

    return RenderedOutput(stdout=render_value(value, output_format))


def render_application_error(error: Mapping[str, object]) -> RenderedOutput:
    """Surface the stable application error code/message on stderr only."""

    code = error.get("code")
    message = error.get("message")
    if type(code) is not str or type(message) is not str:
        raise ValueError("ApplicationError requires string code and message")
    return RenderedOutput(stderr=f"{code}: {message}\n")


def _render_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        indent=2,
        allow_nan=False,
    ) + "\n"


def _render_yaml(value: object) -> str:
    try:
        import yaml  # type: ignore[import-not-found]
    except ModuleNotFoundError as error:
        raise YamlSerializerUnavailable(
            "YAML output requires an available repository/runtime YAML serializer"
        ) from error
    dumped = yaml.safe_dump(
        value,
        allow_unicode=True,
        default_flow_style=False,
        sort_keys=True,
    )
    return dumped if dumped.endswith("\n") else dumped + "\n"


def _render_table(value: object, *, wide: bool) -> str:
    if isinstance(value, Mapping):
        rows = [(str(key), _cell(value[key], wide=wide)) for key in sorted(value, key=str)]
        return _ascii_table(("field", "value"), rows)

    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        items = list(value)
        if not items:
            return ""
        if all(isinstance(item, Mapping) for item in items):
            return _mapping_sequence_table(items, wide=wide)
        rows = [(str(index), _cell(item, wide=wide)) for index, item in enumerate(items)]
        return _ascii_table(("index", "value"), rows)

    return _ascii_table(("value",), [(_cell(value, wide=wide),)])


def _mapping_sequence_table(items: list[object], *, wide: bool) -> str:
    mappings = [item for item in items if isinstance(item, Mapping)]
    columns = sorted({str(key) for item in mappings for key in item})
    rows: list[tuple[str, ...]] = []
    for item in mappings:
        row = tuple(_cell(item.get(column), wide=wide) for column in columns)
        rows.append(row)
    return _ascii_table(tuple(columns), rows)


def _cell(value: object, *, wide: bool) -> str:
    if value is None:
        text = ""
    elif isinstance(value, (Mapping, list, tuple)):
        text = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    else:
        text = str(value)
    if wide or len(text) <= 48:
        return text
    return text[:45] + "..."


def _ascii_table(headers: tuple[str, ...], rows: list[tuple[str, ...]]) -> str:
    widths = [len(header) for header in headers]
    for row in rows:
        if len(row) != len(headers):
            raise ValueError("table row width does not match headers")
        for index, cell in enumerate(row):
            widths[index] = max(widths[index], len(cell))

    def format_row(row: tuple[str, ...]) -> str:
        return " | ".join(cell.ljust(widths[index]) for index, cell in enumerate(row)).rstrip()

    header = format_row(headers)
    separator = "-+-".join("-" * width for width in widths)
    body = [format_row(row) for row in rows]
    return "\n".join([header, separator, *body]) + "\n"
