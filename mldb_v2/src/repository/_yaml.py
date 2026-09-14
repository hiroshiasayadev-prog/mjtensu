"""Small implementation-private YAML loader for MLDB v2 canonical repository documents.

Supports the YAML subset used by MLDB records: mappings, sequences, flow JSON-like
values, plain/quoted scalars, and comments. Unsupported YAML node syntax is rejected
rather than reinterpreted. It is not a public codec abstraction or a full YAML parser.
"""
from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass


class _YamlError(ValueError):
    pass


def _json_object_from_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise _YamlError(f"duplicate mapping key: {key!r}")
        result[key] = value
    return result


def _json_float(token: str) -> float:
    value = float(token)
    if not math.isfinite(value):
        raise _YamlError(f"non-finite float is unsupported: {token!r}")
    return value


def _reject_json_constant(token: str) -> object:
    raise _YamlError(f"non-finite float is unsupported: {token!r}")


@dataclass(frozen=True)
class _Line:
    indent: int
    text: str


def _lstrip_ascii_space(text: str) -> str:
    return text.lstrip(" ")


def _rstrip_ascii_space(text: str) -> str:
    return text.rstrip(" ")


def _strip_ascii_space(text: str) -> str:
    return text.strip(" ")


def _is_separation_space(char: str) -> bool:
    return char == " "


def _split_yaml_lines(text: str) -> list[str]:
    normalized = text.replace("\r\n", "\n")
    if "\r" in normalized:
        raise _YamlError("bare carriage return is unsupported in canonical YAML")
    if any(char in normalized for char in ("\x85", "\u2028", "\u2029")):
        raise _YamlError("Unicode line separators are unsupported in canonical YAML")
    return normalized.split("\n")


def _load_yaml(text: str) -> object:
    if text.startswith("\ufeff"):
        if text.startswith("\ufeff\ufeff"):
            raise _YamlError("multiple document-leading UTF-8 BOM markers are unsupported")
        text = text[1:]

    stripped = text.lstrip(" \r\n")
    if not stripped:
        raise _YamlError("empty YAML document")
    if "\t" in text:
        raise _YamlError("literal tabs are unsupported in canonical YAML")
    try:
        return json.loads(
            text,
            object_pairs_hook=_json_object_from_pairs,
            parse_float=_json_float,
            parse_constant=_reject_json_constant,
        )
    except json.JSONDecodeError:
        pass
    lines = _prepare_lines(text)
    if not lines:
        raise _YamlError("empty YAML document")
    value, index = _parse_block(lines, 0, lines[0].indent)
    if index != len(lines):
        raise _YamlError(f"unexpected content near line {index + 1}")
    return value


def _prepare_lines(text: str) -> list[_Line]:
    result: list[_Line] = []
    raw_lines = _split_yaml_lines(text)
    i = 0
    while i < len(raw_lines):
        raw = raw_lines[i]
        if not _strip_ascii_space(raw) or _lstrip_ascii_space(raw).startswith("#"):
            i += 1
            continue
        indent = len(raw) - len(raw.lstrip(" "))
        body = _rstrip_ascii_space(_strip_comment(raw[indent:]))
        if not body:
            i += 1
            continue
        _reject_unsupported_block_scalar_node(body)
        result.append(_Line(indent, body))
        i += 1
    return result


_BLOCK_SCALAR_HEADER = re.compile(r"^[|>](?:[+-][1-9]?|[1-9][+-]?)?$")


def _is_block_scalar_header(text: str) -> bool:
    return _BLOCK_SCALAR_HEADER.fullmatch(_strip_ascii_space(text)) is not None


def _reject_unsupported_block_scalar_node(text: str) -> None:
    candidate = _strip_ascii_space(text)
    if candidate.startswith("-"):
        candidate = _strip_ascii_space(candidate[1:])
        if _is_block_scalar_header(candidate):
            raise _YamlError(f"unsupported YAML block scalar: {candidate!r}")

    separator_index = _mapping_separator_index(candidate)
    if separator_index is None:
        return
    value = _strip_ascii_space(candidate[separator_index + 1 :])
    if _is_block_scalar_header(value):
        raise _YamlError(f"unsupported YAML block scalar: {value!r}")


def _strip_comment(text: str) -> str:
    quote: str | None = None
    escaped = False
    for index, char in enumerate(text):
        if escaped:
            escaped = False
            continue
        if char == "\\" and quote == '"':
            escaped = True
            continue
        if char in {'"', "'"}:
            if quote is None:
                quote = char
            elif quote == char:
                quote = None
        elif char == "#" and quote is None and (index == 0 or _is_separation_space(text[index - 1])):
            return text[:index]
    return text


def _mapping_separator_index(text: str) -> int | None:
    quote: str | None = None
    escaped = False
    for index, char in enumerate(text):
        if escaped:
            escaped = False
            continue
        if char == "\\" and quote == '"':
            escaped = True
            continue
        if char in {'"', "'"}:
            if quote is None:
                quote = char
            elif quote == char:
                quote = None
            continue
        if char == ":" and quote is None and (
            index + 1 == len(text) or _is_separation_space(text[index + 1])
        ):
            return index
    return None


def _reject_unsupported_node_indicator(text: str) -> None:
    if not text:
        return

    indicator = text[0]
    if indicator in "!&*":
        kind = {"!": "tag", "&": "anchor", "*": "alias"}[indicator]
        raise _YamlError(f"YAML {kind} syntax is unsupported: {text!r}")
    if indicator in "@`":
        raise _YamlError(f"YAML reserved indicator is unsupported: {text!r}")
    if indicator in ",[]{}|>%":
        raise _YamlError(f"YAML node indicator is unsupported: {text!r}")
    if indicator in "?:-" and (len(text) == 1 or _is_separation_space(text[1])):
        raise _YamlError(f"YAML node indicator is unsupported: {text!r}")


def _parse_mapping_key(text: str) -> str:
    key = _strip_ascii_space(text)
    if not key:
        raise _YamlError("empty mapping key")
    if key[0] in "'\"":
        value, end = _parse_quoted(key, 0)
        if _strip_ascii_space(key[end:]):
            raise _YamlError(f"unexpected content after quoted mapping key: {key!r}")
        return value
    value = _parse_plain_scalar(key)
    if type(value) is not str:
        raise _YamlError(f"mapping keys must be strings: {key!r}")
    return value


def _parse_block(lines: list[_Line], index: int, indent: int) -> tuple[object, int]:
    if lines[index].indent != indent:
        raise _YamlError("invalid indentation")
    if lines[index].text.startswith("- ") or lines[index].text == "-":
        return _parse_sequence(lines, index, indent)
    return _parse_mapping(lines, index, indent)


def _parse_mapping(lines: list[_Line], index: int, indent: int) -> tuple[dict[str, object], int]:
    result: dict[str, object] = {}
    while index < len(lines) and lines[index].indent == indent and not lines[index].text.startswith("-"):
        line_text = lines[index].text
        separator_index = _mapping_separator_index(line_text)
        if separator_index is None:
            raise _YamlError(f"invalid mapping entry: {line_text!r}")
        key = _parse_mapping_key(line_text[:separator_index])
        if key in result:
            raise _YamlError(f"duplicate mapping key: {key!r}")
        rest = _strip_ascii_space(line_text[separator_index + 1:])
        index += 1
        if rest:
            result[key] = _parse_scalar(rest)
        elif index < len(lines) and lines[index].indent > indent:
            result[key], index = _parse_block(lines, index, lines[index].indent)
        else:
            result[key] = None
    return result, index


def _parse_sequence(lines: list[_Line], index: int, indent: int) -> tuple[list[object], int]:
    result: list[object] = []
    while index < len(lines) and lines[index].indent == indent and lines[index].text.startswith("-"):
        rest = _strip_ascii_space(lines[index].text[1:])
        index += 1
        if not rest:
            if index >= len(lines) or lines[index].indent <= indent:
                result.append(None)
            else:
                value, index = _parse_block(lines, index, lines[index].indent)
                result.append(value)
            continue
        separator_index = (
            None if rest.startswith(("[", "{")) else _mapping_separator_index(rest)
        )
        if separator_index is not None:
            key = _parse_mapping_key(rest[:separator_index])
            tail = _strip_ascii_space(rest[separator_index + 1:])
            item: dict[str, object] = {}
            if tail:
                item[key] = _parse_scalar(tail)
            else:
                item[key] = None
                if index < len(lines) and lines[index].indent > indent + 2:
                    item[key], index = _parse_block(lines, index, lines[index].indent)
            if index < len(lines) and lines[index].indent > indent:
                continuation_indent = lines[index].indent
                if continuation_indent != indent + 2:
                    raise _YamlError("invalid sequence mapping indentation")
                extra, index = _parse_mapping(lines, index, continuation_indent)
                duplicate = next((name for name in item if name in extra), None)
                if duplicate is not None:
                    raise _YamlError(f"duplicate mapping key: {duplicate!r}")
                item.update(extra)
            result.append(item)
        else:
            result.append(_parse_scalar(rest))
    return result, index


def _parse_scalar(text: str) -> object:
    text = _strip_ascii_space(text)
    if not text:
        return ""
    if text[0] in "[{":
        parser = _FlowParser(text)
        value = parser.parse_value()
        parser.require_end()
        return value
    if text[0] in "'\"":
        value, end = _parse_quoted(text, 0)
        if _strip_ascii_space(text[end:]):
            raise _YamlError(f"unexpected content after quoted scalar: {text!r}")
        return value
    return _parse_plain_scalar(text)


def _parse_plain_scalar(text: str) -> object:
    _reject_unsupported_node_indicator(text)
    lowered = text.lower()
    if lowered in {"null", "~"}:
        return None
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if lowered in {".inf", "+.inf", "-.inf", ".nan"}:
        raise _YamlError(f"non-finite float is unsupported: {text!r}")
    if re.fullmatch(r"[-+]?0[xX][A-Za-z0-9_]+", text) or re.fullmatch(r"[-+]?0[oO][A-Za-z0-9_]+", text) or re.fullmatch(r"[-+]?0[bB][A-Za-z0-9_]+", text):
        raise _YamlError(f"unsupported alternate numeric form: {text!r}")
    if "_" in text and re.match(r"[-+]?(?:[0-9]|\.[0-9])", text):
        raise _YamlError(f"unsupported alternate numeric form: {text!r}")
    if re.fullmatch(r"[-+]?[0-9]+", text):
        return int(text)
    if re.fullmatch(
        r"[-+]?(?:(?:[0-9]+\.[0-9]*|[0-9]*\.[0-9]+)(?:[eE][-+]?[0-9]+)?|[0-9]+[eE][-+]?[0-9]+)",
        text,
    ):
        value = float(text)
        if not math.isfinite(value):
            raise _YamlError(f"non-finite float is unsupported: {text!r}")
        return value
    return text


def _parse_quoted(text: str, start: int) -> tuple[str, int]:
    quote = text[start]
    if quote == '"':
        index = start + 1
        escaped = False
        while index < len(text):
            char = text[index]
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                token = text[start:index + 1]
                try:
                    return json.loads(token), index + 1
                except json.JSONDecodeError as error:
                    raise _YamlError(f"invalid double-quoted scalar: {token!r}") from error
            index += 1
        raise _YamlError("unterminated double-quoted scalar")

    index = start + 1
    parts: list[str] = []
    while index < len(text):
        char = text[index]
        if char == "'":
            if index + 1 < len(text) and text[index + 1] == "'":
                parts.append("'")
                index += 2
                continue
            return "".join(parts), index + 1
        parts.append(char)
        index += 1
    raise _YamlError("unterminated single-quoted scalar")


class _FlowParser:
    def __init__(self, text: str) -> None:
        self._text = text
        self._index = 0

    def parse_value(self) -> object:
        self._skip_space()
        if self._index >= len(self._text):
            raise _YamlError("missing flow value")
        char = self._text[self._index]
        if char == "[":
            return self._parse_sequence()
        if char == "{":
            return self._parse_mapping()
        if char in "'\"":
            value, self._index = _parse_quoted(self._text, self._index)
            return value
        return self._parse_plain()

    def require_end(self) -> None:
        self._skip_space()
        if self._index != len(self._text):
            raise _YamlError(f"unexpected flow content: {self._text[self._index:]!r}")

    def _parse_sequence(self) -> list[object]:
        self._index += 1
        result: list[object] = []
        self._skip_space()
        if self._consume("]"):
            return result
        while True:
            result.append(self.parse_value())
            self._skip_space()
            if self._consume("]"):
                return result
            self._require(",")
            if self._consume("]"):
                return result

    def _parse_mapping(self) -> dict[str, object]:
        self._index += 1
        result: dict[str, object] = {}
        self._skip_space()
        if self._consume("}"):
            return result
        while True:
            key = self._parse_key()
            self._skip_space()
            self._require(":")
            value = self.parse_value()
            if key in result:
                raise _YamlError(f"duplicate flow mapping key: {key!r}")
            result[key] = value
            self._skip_space()
            if self._consume("}"):
                return result
            self._require(",")
            if self._consume("}"):
                return result

    def _parse_key(self) -> str:
        self._skip_space()
        if self._index >= len(self._text):
            raise _YamlError("missing flow mapping key")
        if self._text[self._index] in "'\"":
            value, self._index = _parse_quoted(self._text, self._index)
            return value
        start = self._index
        while self._index < len(self._text) and self._text[self._index] not in ":,{}[]":
            self._index += 1
        key = _strip_ascii_space(self._text[start:self._index])
        if not key:
            raise _YamlError("empty flow mapping key")
        return _parse_mapping_key(key)

    def _parse_plain(self) -> object:
        start = self._index
        while self._index < len(self._text) and self._text[self._index] not in ",]}":
            self._index += 1
        token = _strip_ascii_space(self._text[start:self._index])
        if not token:
            raise _YamlError("empty flow scalar")
        if any(char in token for char in "[{"):
            raise _YamlError(f"invalid plain flow scalar: {token!r}")
        if _mapping_separator_index(token) is not None:
            raise _YamlError(f"unsupported implicit flow mapping or extra colon: {token!r}")
        return _parse_plain_scalar(token)

    def _skip_space(self) -> None:
        while self._index < len(self._text) and _is_separation_space(self._text[self._index]):
            self._index += 1

    def _consume(self, expected: str) -> bool:
        if self._index < len(self._text) and self._text[self._index] == expected:
            self._index += 1
            return True
        return False

    def _require(self, expected: str) -> None:
        self._skip_space()
        if not self._consume(expected):
            found = self._text[self._index:self._index + 1] or "end of input"
            raise _YamlError(f"expected {expected!r}, found {found!r}")
        self._skip_space()

