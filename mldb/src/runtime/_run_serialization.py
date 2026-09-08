from __future__ import annotations

import json
from datetime import date, datetime
from enum import Enum
from typing import Any


def dump_yaml_document(value: dict[str, Any]) -> str:
    return _dump_mapping(value, 0)


def load_yaml_document(text: str) -> dict[str, Any]:
    stripped = text.lstrip()
    if stripped.startswith("{"):
        value = json.loads(text)
    else:
        lines = [line.rstrip() for line in text.splitlines() if line.strip() and not line.lstrip().startswith("#")]
        value, index = _parse_block(lines, 0, 0)
        if index != len(lines):
            raise ValueError("unexpected trailing YAML content")
    if not isinstance(value, dict):
        raise ValueError("run metadata must be a mapping")
    return value


def encode_scalar(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, dict):
        return {str(k): encode_scalar(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [encode_scalar(v) for v in value]
    raise TypeError(f"unsupported persisted value: {type(value).__name__}")


def decode_timestamp(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return value


def _dump_mapping(mapping: dict[str, Any], indent: int) -> str:
    out: list[str] = []
    prefix = " " * indent
    for key, raw in mapping.items():
        value = encode_scalar(raw)
        if isinstance(value, dict):
            if not value:
                out.append(f"{prefix}{key}: {{}}\n")
            elif _requires_flow_container(value):
                out.append(f"{prefix}{key}: {_flow_text(value)}\n")
            else:
                out.append(f"{prefix}{key}:\n")
                out.append(_dump_mapping(value, indent + 2))
        elif isinstance(value, list):
            if not value:
                out.append(f"{prefix}{key}: []\n")
            elif _requires_flow_container(value):
                out.append(f"{prefix}{key}: {_flow_text(value)}\n")
            else:
                out.append(f"{prefix}{key}:\n")
                out.append(_dump_sequence(value, indent + 2))
        else:
            out.append(f"{prefix}{key}: {_scalar_text(value)}\n")
    return "".join(out)


def _dump_sequence(values: list[Any], indent: int) -> str:
    out: list[str] = []
    prefix = " " * indent
    for raw in values:
        value = encode_scalar(raw)
        if isinstance(value, dict):
            if not value:
                out.append(f"{prefix}- {{}}\n")
                continue
            first, *rest = value.items()
            key, item = first
            if isinstance(item, (dict, list)):
                out.append(f"{prefix}- {key}:\n")
                out.append(_dump_nested(item, indent + 4))
            else:
                out.append(f"{prefix}- {key}: {_scalar_text(item)}\n")
            if rest:
                out.append(_dump_mapping(dict(rest), indent + 2))
        else:
            out.append(f"{prefix}- {_scalar_text(value)}\n")
    return "".join(out)


def _dump_nested(value: Any, indent: int) -> str:
    return _dump_mapping(value, indent) if isinstance(value, dict) else _dump_sequence(value, indent)


def _requires_flow_container(value: dict[str, Any] | list[Any]) -> bool:
    if isinstance(value, dict):
        return any(
            not _is_plain_mapping_key(key) or isinstance(item, (dict, list))
            for key, item in value.items()
        )
    return any(isinstance(item, (dict, list)) for item in value)


def _is_plain_mapping_key(key: object) -> bool:
    return (
        isinstance(key, str)
        and bool(key)
        and all(
            character.isascii()
            and (character.isalnum() or character in "_-.")
            for character in key
        )
    )


def _flow_text(value: dict[str, Any] | list[Any]) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    )


def _scalar_text(value: Any) -> str:
    if value is None:
        return "null"
    if value is True:
        return "true"
    if value is False:
        return "false"
    if isinstance(value, (int, float)):
        return json.dumps(value, allow_nan=False)
    return json.dumps(value, ensure_ascii=False)


def _parse_block(lines: list[str], index: int, indent: int) -> tuple[Any, int]:
    if index >= len(lines):
        return {}, index
    current = lines[index]
    stripped = current.lstrip(" ")
    actual = len(current) - len(stripped)
    if actual != indent:
        raise ValueError("invalid YAML indentation")
    if stripped.startswith("- "):
        return _parse_sequence(lines, index, indent)
    return _parse_mapping(lines, index, indent)


def _parse_mapping(lines: list[str], index: int, indent: int) -> tuple[dict[str, Any], int]:
    result: dict[str, Any] = {}
    while index < len(lines):
        line = lines[index]
        stripped = line.lstrip(" ")
        actual = len(line) - len(stripped)
        if actual < indent:
            break
        if actual != indent or stripped.startswith("- "):
            break
        key, sep, tail = stripped.partition(":")
        if not sep or not key:
            raise ValueError("invalid YAML mapping entry")
        tail = tail.strip()
        index += 1
        if tail:
            result[key] = _parse_scalar(tail)
        elif index < len(lines):
            next_line = lines[index]
            next_indent = len(next_line) - len(next_line.lstrip(" "))
            if next_indent <= indent:
                result[key] = None
            else:
                result[key], index = _parse_block(lines, index, indent + 2)
        else:
            result[key] = None
    return result, index


def _parse_sequence(lines: list[str], index: int, indent: int) -> tuple[list[Any], int]:
    result: list[Any] = []
    while index < len(lines):
        line = lines[index]
        stripped = line.lstrip(" ")
        actual = len(line) - len(stripped)
        if actual < indent:
            break
        if actual != indent or not stripped.startswith("- "):
            break
        tail = stripped[2:].strip()
        index += 1
        if not tail:
            value, index = _parse_block(lines, index, indent + 2)
            result.append(value)
            continue
        if ":" in tail and not tail.startswith(('"', "'")):
            key, _, scalar = tail.partition(":")
            item: dict[str, Any] = {key: _parse_scalar(scalar.strip()) if scalar.strip() else None}
            if index < len(lines):
                next_indent = len(lines[index]) - len(lines[index].lstrip(" "))
                if next_indent > indent:
                    rest, index = _parse_mapping(lines, index, indent + 2)
                    item.update(rest)
            result.append(item)
        else:
            result.append(_parse_scalar(tail))
    return result, index


def _parse_scalar(text: str) -> Any:
    if text == "{}": return {}
    if text == "[]": return []
    if text in {"null", "~"}: return None
    if text == "true": return True
    if text == "false": return False
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text
