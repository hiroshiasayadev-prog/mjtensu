"""Small implementation-private YAML loader for frozen MLDB metadata.

Supports the YAML subset used by MLDB records: mappings, sequences, flow JSON-like
values, plain/quoted scalars, comments, and folded/literal block strings. It is not a
public codec abstraction.
"""
from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass


class _YamlError(ValueError):
    pass


@dataclass(frozen=True)
class _Line:
    indent: int
    text: str


def _load_yaml(text: str) -> object:
    stripped = text.lstrip()
    if not stripped:
        raise _YamlError("empty YAML document")
    try:
        return json.loads(text)
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
    raw_lines = text.splitlines()
    i = 0
    while i < len(raw_lines):
        raw = raw_lines[i].expandtabs(2)
        if not raw.strip() or raw.lstrip().startswith("#"):
            i += 1
            continue
        indent = len(raw) - len(raw.lstrip(" "))
        body = _strip_comment(raw[indent:]).rstrip()
        if not body:
            i += 1
            continue
        if body.endswith(": >") or body.endswith(": |"):
            marker = body[-1]
            key = body[:-3].rstrip()
            block_indent = None
            parts: list[str] = []
            i += 1
            while i < len(raw_lines):
                candidate = raw_lines[i].expandtabs(2)
                if not candidate.strip():
                    parts.append("")
                    i += 1
                    continue
                candidate_indent = len(candidate) - len(candidate.lstrip(" "))
                if candidate_indent <= indent:
                    break
                if block_indent is None:
                    block_indent = candidate_indent
                parts.append(candidate[block_indent:])
                i += 1
            value = "\n".join(parts) if marker == "|" else " ".join(part.strip() for part in parts)
            result.append(_Line(indent, f"{key}: {json.dumps(value)}"))
            continue
        result.append(_Line(indent, body))
        i += 1
    return result


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
        elif char == "#" and quote is None and (index == 0 or text[index - 1].isspace()):
            return text[:index]
    return text


def _parse_block(lines: list[_Line], index: int, indent: int) -> tuple[object, int]:
    if lines[index].indent != indent:
        raise _YamlError("invalid indentation")
    if lines[index].text.startswith("- ") or lines[index].text == "-":
        return _parse_sequence(lines, index, indent)
    return _parse_mapping(lines, index, indent)


def _parse_mapping(lines: list[_Line], index: int, indent: int) -> tuple[dict[str, object], int]:
    result: dict[str, object] = {}
    while index < len(lines) and lines[index].indent == indent and not lines[index].text.startswith("-"):
        key, separator, rest = lines[index].text.partition(":")
        if not separator or not key.strip():
            raise _YamlError(f"invalid mapping entry: {lines[index].text!r}")
        key = key.strip()
        if key in result:
            raise _YamlError(f"duplicate mapping key: {key!r}")
        rest = rest.strip()
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
        rest = lines[index].text[1:].strip()
        index += 1
        if not rest:
            if index >= len(lines) or lines[index].indent <= indent:
                result.append(None)
            else:
                value, index = _parse_block(lines, index, lines[index].indent)
                result.append(value)
            continue
        if ":" in rest and not rest.startswith(("[", "{", "'", '"')):
            key, _, tail = rest.partition(":")
            item: dict[str, object] = {key.strip(): _parse_scalar(tail.strip()) if tail.strip() else None}
            if index < len(lines) and lines[index].indent > indent:
                extra, index = _parse_mapping(lines, index, lines[index].indent)
                item.update(extra)
            result.append(item)
        else:
            result.append(_parse_scalar(rest))
    return result, index


def _parse_scalar(text: str) -> object:
    text = text.strip()
    if not text:
        return ""
    if text[0] in "[{":
        parser = _FlowParser(text)
        value = parser.parse_value()
        parser.require_end()
        return value
    if text[0] in "'\"":
        value, end = _parse_quoted(text, 0)
        if text[end:].strip():
            raise _YamlError(f"unexpected content after quoted scalar: {text!r}")
        return value
    return _parse_plain_scalar(text)


def _parse_plain_scalar(text: str) -> object:
    lowered = text.lower()
    if lowered in {"null", "~"}:
        return None
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if lowered in {".inf", "+.inf", "-.inf", ".nan"}:
        raise _YamlError(f"non-finite float is unsupported: {text!r}")
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
        key = self._text[start:self._index].strip()
        if not key:
            raise _YamlError("empty flow mapping key")
        return key

    def _parse_plain(self) -> object:
        start = self._index
        while self._index < len(self._text) and self._text[self._index] not in ",]}":
            self._index += 1
        token = self._text[start:self._index].strip()
        if not token:
            raise _YamlError("empty flow scalar")
        if any(char in token for char in "[{"):
            raise _YamlError(f"invalid plain flow scalar: {token!r}")
        return _parse_plain_scalar(token)

    def _skip_space(self) -> None:
        while self._index < len(self._text) and self._text[self._index].isspace():
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
