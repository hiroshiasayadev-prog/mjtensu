from __future__ import annotations

import json
from pathlib import Path

import pytest

from mldb_v2.src.common.ids import EntityKind
from mldb_v2.src.repository._yaml import _YamlError, _load_yaml
from mldb_v2.src.repository.resolution import CanonicalRepositoryResolver


def _write_namespace(root: Path, namespace: str, *, recorded_id: str | None = None) -> None:
    path = root / namespace / "namespace.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "schema": "mjtensu.mldb-v2/namespace/v1",
                "id": recorded_id or namespace,
                "name": namespace,
                "description": "test namespace",
            }
        ),
        encoding="utf-8",
    )


def _write_entity(
    root: Path,
    namespace: str,
    domain: str,
    local_id: str,
    schema: str,
    *,
    recorded_id: str | None = None,
) -> Path:
    path = root / namespace / domain / f"{local_id}.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"schema": schema, "id": recorded_id or f"{namespace}/{local_id}"}),
        encoding="utf-8",
    )
    return path


@pytest.mark.parametrize(
    "source",
    [
        "value: !!str 1\n",
        "value: !custom 1\n",
        "value: &base ordinary\n",
        "value: *base\n",
    ],
    ids=["tagged", "custom-tag", "anchor", "alias"],
)
def test_yaml_unsupported_node_syntax_is_rejected(source: str) -> None:
    with pytest.raises(_YamlError):
        _load_yaml(source)


@pytest.mark.parametrize(
    "source",
    [
        "value: |\n  abc\n",
        "value: >\n  abc\n",
        "value: |-\n  abc\n",
        "value: |+\n  abc\n",
        "value: >-\n  abc\n",
        "value: >+\n  abc\n",
        "value: |2\n  abc\n",
        "value: >2\n  abc\n",
        "value: |2-\n  abc\n",
        "value: >+2\n  abc\n",
        "values:\n  - |\n    abc\n",
        "values:\n  - >\n    abc\n",
        "values:\n  - value: |2-\n      abc\n",
    ],
)
def test_yaml_block_scalar_nodes_are_rejected(source: str) -> None:
    with pytest.raises(_YamlError, match="unsupported YAML block scalar"):
        _load_yaml(source)


def test_yaml_block_scalar_characters_in_ordinary_strings_are_preserved() -> None:
    assert _load_yaml(
        'value: a|b\ncomparison: a > b\nquoted_pipe: "|"\nquoted_gt: ">"\n'
    ) == {
        "value": "a|b",
        "comparison": "a > b",
        "quoted_pipe": "|",
        "quoted_gt": ">",
    }


@pytest.mark.parametrize(
    "source",
    [
        "value: @reserved\n",
        "value: `reserved\n",
        "value: |reserved\n",
        "value: >reserved\n",
        "value: %reserved\n",
        "value: ,reserved\n",
        "value: - reserved\n",
        "value: ? reserved\n",
        "value: : reserved\n",
    ],
)
def test_yaml_unsupported_plain_node_indicators_are_rejected(source: str) -> None:
    with pytest.raises(_YamlError):
        _load_yaml(source)


def test_yaml_reserved_indicator_characters_are_valid_when_not_node_leading() -> None:
    assert _load_yaml(
        'quoted_at: "@reserved"\nquoted_tick: "`reserved"\nplain_at: abc@def\nplain_tick: abc`def\nnegative: -3\ndash_plain: -reserved\nquestion_plain: ?reserved\ncolon_plain: :reserved\nflow: [1, {"key": "value"}]\n'
    ) == {
        "quoted_at": "@reserved",
        "quoted_tick": "`reserved",
        "plain_at": "abc@def",
        "plain_tick": "abc`def",
        "negative": -3,
        "dash_plain": "-reserved",
        "question_plain": "?reserved",
        "colon_plain": ":reserved",
        "flow": [1, {"key": "value"}],
    }


def test_yaml_sequence_inline_mapping_duplicate_key_is_rejected() -> None:
    with pytest.raises(_YamlError, match="duplicate mapping key"):
        _load_yaml("values:\n  - key: first\n    key: second\n")


def test_yaml_sequence_inline_mapping_duplicate_decoded_quoted_key_is_rejected() -> None:
    with pytest.raises(_YamlError, match="duplicate mapping key"):
        _load_yaml('values:\n  - "key": first\n    key: second\n')


def test_yaml_sequence_inline_mapping_nonduplicate_keys_parse() -> None:
    assert _load_yaml(
        'values:\n  - "key": first\n    other: second\n'
    ) == {"values": [{"key": "first", "other": "second"}]}


def test_yaml_nested_duplicate_mapping_key_is_rejected() -> None:
    with pytest.raises(_YamlError, match="duplicate mapping key"):
        _load_yaml("outer:\n  inner:\n    key: first\n    key: second\n")


def test_yaml_plain_scalar_colons_are_not_mapping_separators() -> None:
    assert _load_yaml(
        """values:
  - s3://bucket/path/to/object
  - https://example.invalid/path
  - abc:def
plain: abc:def
"""
    ) == {
        "values": [
            "s3://bucket/path/to/object",
            "https://example.invalid/path",
            "abc:def",
        ],
        "plain": "abc:def",
    }


def test_yaml_ordinary_inline_mapping_still_parses() -> None:
    assert _load_yaml(
        """values:
  - uri: s3://bucket/path/to/object
    label: abc:def
"""
    ) == {
        "values": [
            {"uri": "s3://bucket/path/to/object", "label": "abc:def"},
        ]
    }


def test_yaml_json_compatible_document_still_parses() -> None:
    source = json.dumps(
        {
            "value": "!!str 1",
            "uri": "s3://bucket/path/to/object",
            "nested": [{"label": "abc:def"}],
        }
    )
    assert _load_yaml(source) == json.loads(source)


def test_yaml_malformed_document_remains_rejected() -> None:
    with pytest.raises(_YamlError):
        _load_yaml("value: [1, 2\n")



def test_yaml_sequence_inline_mapping_nested_value_is_preserved() -> None:
    assert _load_yaml("values:\n  - key:\n      nested: value\n") == {
        "values": [{"key": {"nested": "value"}}]
    }


def test_yaml_sequence_inline_mapping_sibling_continuation_remains_supported() -> None:
    assert _load_yaml("values:\n  - key: first\n    other: second\n") == {
        "values": [{"key": "first", "other": "second"}]
    }


@pytest.mark.parametrize("source", ["values: [key: value]\n", "value: {a: b: c}\n"])
def test_yaml_unsupported_flow_colon_forms_fail_closed(source: str) -> None:
    with pytest.raises(_YamlError):
        _load_yaml(source)


def test_yaml_flow_colon_plain_scalars_remain_supported() -> None:
    assert _load_yaml("values: [s3://bucket/path, abc:def]\n") == {
        "values": ["s3://bucket/path", "abc:def"]
    }


@pytest.mark.parametrize(
    "source",
    ["value: 0x10\n", "value: 0o10\n", "value: 0b10\n", "value: 1_000\n", "value: 1_000.5\n"],
)
def test_yaml_unsupported_alternate_numeric_forms_fail_closed(source: str) -> None:
    with pytest.raises(_YamlError, match="numeric"):
        _load_yaml(source)


def test_yaml_nonnumeric_underscore_strings_remain_strings() -> None:
    assert _load_yaml("a: version_1\nb: abc_123\n") == {
        "a": "version_1",
        "b": "abc_123",
    }


def test_yaml_supported_decimal_numeric_forms_remain_numeric() -> None:
    assert _load_yaml("a: 1\nb: -1\nc: 1.0\nd: .5\ne: 1e3\n") == {
        "a": 1,
        "b": -1,
        "c": 1.0,
        "d": 0.5,
        "e": 1000.0,
    }


def test_yaml_literal_tab_is_rejected_but_escaped_tab_is_supported() -> None:
    with pytest.raises(_YamlError, match="literal tabs"):
        _load_yaml("outer:\n\tkey: value\n")
    assert _load_yaml('value: "\\t"\n') == {"value": "\t"}


@pytest.mark.parametrize(
    "source",
    [
        "1: value\n",
        "true: value\n",
        "null: value\n",
        "value: {1: x}\n",
        "value: {true: x}\n",
        "value: {null: x}\n",
    ],
)
def test_yaml_non_string_mapping_keys_are_rejected(source: str) -> None:
    with pytest.raises(_YamlError, match="mapping keys must be strings"):
        _load_yaml(source)


def test_yaml_quoted_string_like_mapping_keys_are_supported() -> None:
    assert _load_yaml('"1": a\n"true": b\n"null": c\nvalue: {"1": x, "true": y}\n') == {
        "1": "a",
        "true": "b",
        "null": "c",
        "value": {"1": "x", "true": "y"},
    }


@pytest.mark.parametrize(
    "source",
    [
        '{"a":1,"a":2}',
        '{"outer":{"a":1,"a":2}}',
    ],
)
def test_yaml_json_fast_path_rejects_duplicate_keys(source: str) -> None:
    with pytest.raises(_YamlError, match="duplicate mapping key"):
        _load_yaml(source)


@pytest.mark.parametrize(
    "source",
    [
        '{"value":NaN}',
        '{"value":Infinity}',
        '{"value":-Infinity}',
        '{"value":1e10000}',
    ],
)
def test_yaml_json_fast_path_rejects_non_finite_numbers(source: str) -> None:
    with pytest.raises(_YamlError, match="non-finite"):
        _load_yaml(source)


def test_yaml_strict_json_fast_path_still_parses() -> None:
    assert _load_yaml('{"a":1,"nested":{"b":2.5},"items":[true,null,"x"]}') == {
        "a": 1,
        "nested": {"b": 2.5},
        "items": [True, None, "x"],
    }


def test_current_mldb_data_canonical_yaml_examples_parse() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    roots = [
        repo_root / "mldb_data" / "tile-classifier",
        repo_root / "mldb_data" / "rotated-fcos",
    ]
    for root in roots:
        paths = sorted(root.rglob("*.yaml"))
        assert paths, f"no canonical YAML examples found under {root}"
        for path in paths:
            assert type(_load_yaml(path.read_text(encoding="utf-8"))) is dict, path


def test_namespace_success_supports_non_json_yaml(tmp_path: Path) -> None:
    root = tmp_path / "mldb_data"
    path = root / "alpha" / "namespace.yaml"
    path.parent.mkdir(parents=True)
    path.write_text(
        """schema: mjtensu.mldb-v2/namespace/v1
id: alpha
name: Alpha
description: actual YAML, not JSON
""",
        encoding="utf-8",
    )

    document = CanonicalRepositoryResolver(root).resolve(
        kind=EntityKind.NAMESPACE, entity_id="alpha"
    )
    assert document["id"] == "alpha"
    assert document["description"] == "actual YAML, not JSON"


def test_typed_entity_success(tmp_path: Path) -> None:
    root = tmp_path / "mldb_data"
    _write_namespace(root, "alpha")
    _write_entity(root, "alpha", "tasks", "task-v1", "mjtensu.mldb-v2/task/v1")

    document = CanonicalRepositoryResolver(root).resolve(
        kind=EntityKind.TASK, entity_id="alpha/task-v1"
    )
    assert document["id"] == "alpha/task-v1"


def test_missing_exact_file_is_not_found(tmp_path: Path) -> None:
    root = tmp_path / "mldb_data"
    _write_namespace(root, "alpha")
    with pytest.raises(FileNotFoundError):
        CanonicalRepositoryResolver(root).resolve(
            kind=EntityKind.TASK, entity_id="alpha/missing-v1"
        )


def test_missing_namespace_is_not_found(tmp_path: Path) -> None:
    root = tmp_path / "mldb_data"
    root.mkdir()
    with pytest.raises(FileNotFoundError):
        CanonicalRepositoryResolver(root).resolve(
            kind=EntityKind.TASK, entity_id="missing/task-v1"
        )


def test_malformed_typed_id_is_rejected_before_lookup(tmp_path: Path) -> None:
    root = tmp_path / "mldb_data"
    root.mkdir()
    with pytest.raises(ValueError):
        CanonicalRepositoryResolver(root).resolve(
            kind=EntityKind.TASK, entity_id="alpha/task/extra"
        )


def test_namespace_id_mismatch_is_rejected(tmp_path: Path) -> None:
    root = tmp_path / "mldb_data"
    _write_namespace(root, "alpha", recorded_id="beta")
    with pytest.raises(ValueError, match="does not match directory"):
        CanonicalRepositoryResolver(root).resolve(
            kind=EntityKind.TASK, entity_id="alpha/task-v1"
        )


def test_document_id_mismatch_is_rejected(tmp_path: Path) -> None:
    root = tmp_path / "mldb_data"
    _write_namespace(root, "alpha")
    _write_entity(
        root,
        "alpha",
        "tasks",
        "task-v1",
        "mjtensu.mldb-v2/task/v1",
        recorded_id="alpha/other-v1",
    )
    with pytest.raises(ValueError, match="document id does not match"):
        CanonicalRepositoryResolver(root).resolve(
            kind=EntityKind.TASK, entity_id="alpha/task-v1"
        )


def test_schema_kind_mismatch_is_rejected(tmp_path: Path) -> None:
    root = tmp_path / "mldb_data"
    _write_namespace(root, "alpha")
    _write_entity(root, "alpha", "tasks", "task-v1", "mjtensu.mldb-v2/corpus/v1")
    with pytest.raises(ValueError, match="expected schema"):
        CanonicalRepositoryResolver(root).resolve(
            kind=EntityKind.TASK, entity_id="alpha/task-v1"
        )


def test_cross_namespace_exact_resolution(tmp_path: Path) -> None:
    root = tmp_path / "mldb_data"
    _write_namespace(root, "alpha")
    _write_namespace(root, "beta")
    _write_entity(root, "beta", "tasks", "task-v1", "mjtensu.mldb-v2/task/v1")

    document = CanonicalRepositoryResolver(root).resolve(
        kind=EntityKind.TASK, entity_id="beta/task-v1"
    )
    assert document["id"] == "beta/task-v1"


def test_legacy_flat_layout_is_never_a_fallback(tmp_path: Path) -> None:
    root = tmp_path / "mldb_data"
    legacy = root / "tasks" / "task-v1.yaml"
    legacy.parent.mkdir(parents=True)
    legacy.write_text(
        json.dumps({"schema": "mjtensu.mldb-v2/task/v1", "id": "tasks/task-v1"}),
        encoding="utf-8",
    )

    with pytest.raises(FileNotFoundError):
        CanonicalRepositoryResolver(root).resolve(
            kind=EntityKind.TASK, entity_id="tasks/task-v1"
        )


def test_yaml_single_document_leading_bom_is_supported_for_block_yaml() -> None:
    assert _load_yaml(
        "\ufeffschema: mjtensu.mldb-v2/namespace/v1\nid: demo\n"
    ) == {
        "schema": "mjtensu.mldb-v2/namespace/v1",
        "id": "demo",
    }


def test_yaml_single_document_leading_bom_is_supported_for_strict_json() -> None:
    assert _load_yaml(
        '\ufeff{"schema":"mjtensu.mldb-v2/namespace/v1","id":"demo"}'
    ) == {
        "schema": "mjtensu.mldb-v2/namespace/v1",
        "id": "demo",
    }


@pytest.mark.parametrize(
    "source",
    [
        "\ufeff\ufeffschema: mjtensu.mldb-v2/namespace/v1\nid: demo\n",
        '\ufeff\ufeff{"schema":"mjtensu.mldb-v2/namespace/v1","id":"demo"}',
    ],
)
def test_yaml_double_document_leading_bom_fails_closed(source: str) -> None:
    with pytest.raises(_YamlError, match="BOM"):
        _load_yaml(source)


def test_yaml_bom_after_ascii_leading_spaces_is_not_consumed_as_document_bom() -> None:
    assert _load_yaml("  \ufeffschema: value\n") == {"\ufeffschema": "value"}


def test_yaml_mid_key_and_mid_scalar_bom_content_is_preserved() -> None:
    assert _load_yaml("sche\ufeffma: value\nplain: a\ufeffb\n") == {
        "sche\ufeffma": "value",
        "plain": "a\ufeffb",
    }


@pytest.mark.parametrize("quote", ['"', "'"])
def test_yaml_quoted_bom_content_is_preserved(quote: str) -> None:
    assert _load_yaml(f"value: {quote}\ufeff{quote}\n") == {"value": "\ufeff"}


@pytest.mark.parametrize("space", ["\u00a0", "\u2003"])
def test_yaml_non_ascii_whitespace_plain_scalar_content_is_preserved(space: str) -> None:
    assert _load_yaml(f"value: foo{space}\n") == {"value": f"foo{space}"}
    assert _load_yaml(f"value: a{space}b\n") == {"value": f"a{space}b"}


def test_yaml_leading_nbsp_mapping_key_is_preserved() -> None:
    assert _load_yaml("\u00a0key: value\n") == {"\u00a0key": "value"}


@pytest.mark.parametrize("quote", ['"', "'"])
@pytest.mark.parametrize("space", ["\u00a0", "\u2003"])
def test_yaml_quoted_non_ascii_whitespace_is_preserved(quote: str, space: str) -> None:
    assert _load_yaml(f"value: {quote}foo{space}{quote}\n") == {"value": f"foo{space}"}


def test_yaml_nbsp_before_hash_is_not_comment_separation() -> None:
    assert _load_yaml("value: abc\u00a0#tail\n") == {"value": "abc\u00a0#tail"}


def test_yaml_nbsp_after_colon_is_not_mapping_separation() -> None:
    with pytest.raises(_YamlError):
        _load_yaml("key:\u00a0value\n")


def test_yaml_ascii_space_mapping_and_comment_behavior_regression() -> None:
    assert _load_yaml("key: value # comment\n") == {"key": "value"}


def test_yaml_lf_and_crlf_documents_parse_identically() -> None:
    expected = {"first": "one", "second": "two"}
    assert _load_yaml("first: one\nsecond: two\n") == expected
    assert _load_yaml("first: one\r\nsecond: two\r\n") == expected


@pytest.mark.parametrize("separator", ["\x85", "\u2028", "\u2029"])
def test_yaml_unicode_line_separators_fail_closed(separator: str) -> None:
    with pytest.raises(_YamlError, match="line separators"):
        _load_yaml(f"value: one{separator}other: two\n")
