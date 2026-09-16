import copy
import json

import pytest

from mldb_v2.src.common.ids import EntityKind, StudyPlanId, StudyResultId
from mldb_v2.src.repository._yaml import _YamlError, _load_yaml
from mldb_v2.src.repository.canonical_writes import CanonicalRepositoryWriter


def test_yaml_tagged_scalar_is_rejected_instead_of_reinterpreted() -> None:
    with pytest.raises(_YamlError):
        _load_yaml("value: !!str 1\n")


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("base: &base value\ncopy: *base\n", {"base": "value", "copy": "value"}),
        ("values:\n  - s3://bucket/path/to/object\n", {"values": ["s3://bucket/path/to/object"]}),
    ],
)
def test_yaml_syntax_is_either_rejected_or_parsed_without_semantic_reinterpretation(
    source: str,
    expected: object,
) -> None:
    try:
        parsed = _load_yaml(source)
    except _YamlError:
        return
    assert parsed == expected


def _repo(tmp_path):
    namespace = tmp_path / "mldb_data" / "demo"
    namespace.mkdir(parents=True)
    (namespace / "namespace.yaml").write_text(
        json.dumps({"schema": "mjtensu.mldb-v2/namespace/v1", "id": "demo"}),
        encoding="utf-8",
    )
    return tmp_path


class FakeImmutableValidationError(ValueError):
    pass


class AcceptingRecordValidator:
    def validate(self, *, kind, entity_id, document) -> None:
        pass


class RejectingRecordValidator:
    def validate(self, *, kind, entity_id, document) -> None:
        raise FakeImmutableValidationError("verification fake rejected immutable record")


def _writer(repo, *, record_validator=None):
    if record_validator is None:
        record_validator = AcceptingRecordValidator()
    return CanonicalRepositoryWriter(repo, record_validator=record_validator)


@pytest.mark.parametrize(
    "document",
    [
        {"schema": "mjtensu.mldb-v2/study-plan/v1", "id": "demo/example-plan"},
        {"schema": "mjtensu.mldb-v2/model/v1", "id": "demo/example-plan"},
    ],
)
def test_immutable_writer_rejects_incomplete_or_wrong_kind_documents(tmp_path, document) -> None:
    repo = _repo(tmp_path)
    writer = _writer(repo, record_validator=RejectingRecordValidator())
    path = repo / "mldb_data" / "demo" / "study_plans" / "example-plan.yaml"
    with pytest.raises(FakeImmutableValidationError, match="verification fake rejected"):
        writer.create_immutable(
            kind=EntityKind.STUDY_PLAN,
            entity_id=StudyPlanId("demo/example-plan"),
            document=document,
        )
    assert not path.exists()


def _study_result():
    key = "123e4567e89b42d3a456426614174000"
    return {
        "schema": "mjtensu.mldb-v2/study-result/v1",
        "id": f"demo/run-{key}",
        "execution_key": key,
        "plan": "demo/example-plan",
        "study": "demo/example-study",
        "source_commit": "b" * 40,
        "backend": "clearml",
        "created_at": "2026-09-09T09:00:00Z",
        "status": "submitted",
        "diagnostic": None,
        "trials": [
            {
                "trial": "trial-0001",
                "training": {"disposition": "pending", "result": None, "reason": None},
                "evaluations": [
                    {
                        "coordinate": "eval-0001",
                        "stage": "holdout",
                        "disposition": "pending",
                        "result": None,
                        "reason": None,
                    }
                ],
            }
        ],
    }


def _terminalize_slots(document):
    replacement = copy.deepcopy(document)
    key = replacement["execution_key"]
    trial = replacement["trials"][0]
    trial["training"] = {
        "disposition": "completed",
        "result": f"demo/run-{key}-trial-0001-train",
        "reason": None,
    }
    trial["evaluations"][0].update(
        disposition="completed",
        result=f"demo/run-{key}-trial-0001-eval-0001",
        reason=None,
    )
    return replacement


def test_submitted_study_result_cannot_remain_nonterminal_after_all_stages_close(tmp_path) -> None:
    repo = _repo(tmp_path)
    writer = _writer(repo)
    initial = _study_result()
    entity_id = StudyResultId(initial["id"])
    writer.create_study_result(entity_id=entity_id, document=initial)

    invalid = _terminalize_slots(initial)
    invalid["status"] = "submitted"
    with pytest.raises(ValueError):
        writer.replace_nonterminal_study_result(entity_id=entity_id, replacement=invalid)


def test_cancelling_study_result_must_close_cancelled_when_all_stages_are_terminal(tmp_path) -> None:
    repo = _repo(tmp_path)
    writer = _writer(repo)
    initial = _study_result()
    entity_id = StudyResultId(initial["id"])
    writer.create_study_result(entity_id=entity_id, document=initial)

    cancelling = copy.deepcopy(initial)
    cancelling["status"] = "cancelling"
    writer.replace_nonterminal_study_result(entity_id=entity_id, replacement=cancelling)

    invalid = _terminalize_slots(cancelling)
    invalid["status"] = "cancelling"
    with pytest.raises(ValueError):
        writer.replace_nonterminal_study_result(entity_id=entity_id, replacement=invalid)


def test_yaml_block_scalars_are_rejected_or_preserve_yaml_clip_semantics() -> None:
    cases = [
        ("value: |\n  a\n  b\n", {"value": "a\nb\n"}),
        ("value: >\n  a\n  b\n", {"value": "a b\n"}),
    ]
    for source, expected in cases:
        try:
            parsed = _load_yaml(source)
        except _YamlError:
            continue
        assert parsed == expected


def test_failed_global_failure_closure_cannot_contain_study_cancelled_skip(tmp_path) -> None:
    repo = _repo(tmp_path)
    writer = _writer(repo)
    initial = _study_result()
    entity_id = StudyResultId(initial["id"])
    writer.create_study_result(entity_id=entity_id, document=initial)

    invalid = copy.deepcopy(initial)
    invalid["trials"][0]["training"] = {
        "disposition": "skipped", "result": None, "reason": "global_failure"
    }
    invalid["trials"][0]["evaluations"][0].update(
        disposition="skipped", result=None, reason="study_cancelled"
    )
    invalid["status"] = "failed"
    invalid["diagnostic"] = {"code": "global_failure", "message": "cannot progress"}
    with pytest.raises(ValueError):
        writer.replace_nonterminal_study_result(entity_id=entity_id, replacement=invalid)


def test_yaml_residual_unsupported_syntax_is_not_silently_reinterpreted() -> None:
    silently_accepted: list[str] = []
    for source in ["value: @reserved\n", "value: `reserved\n"]:
        try:
            _load_yaml(source)
        except _YamlError:
            pass
        else:
            silently_accepted.append(source.strip())

    duplicate = "values:\n  - key: first\n    key: second\n"
    try:
        _load_yaml(duplicate)
    except _YamlError:
        pass
    else:
        silently_accepted.append("duplicate key in block sequence mapping")

    assert silently_accepted == []


def test_submitted_study_result_requires_pending_work_when_plan_topology_is_empty(tmp_path) -> None:
    repo = _repo(tmp_path)
    writer = _writer(repo)
    initial = _study_result()
    initial["trials"] = []
    entity_id = StudyResultId(initial["id"])
    with pytest.raises(ValueError, match="pending"):
        writer.create_study_result(entity_id=entity_id, document=initial)


def test_study_result_created_at_rejects_non_rfc3339_iso8601_forms(tmp_path) -> None:
    repo = _repo(tmp_path)
    writer = _writer(repo)
    initial = _study_result()
    initial["created_at"] = "2026-W37-3T09:00:00Z"
    entity_id = StudyResultId(initial["id"])
    with pytest.raises(ValueError, match="RFC3339"):
        writer.create_study_result(entity_id=entity_id, document=initial)


def test_study_result_identity_namespace_must_match_source_study_namespace(tmp_path) -> None:
    repo = _repo(tmp_path)
    writer = _writer(repo)
    initial = _study_result()
    initial["study"] = "other/example-study"
    entity_id = StudyResultId(initial["id"])
    with pytest.raises(ValueError, match="namespace"):
        writer.create_study_result(entity_id=entity_id, document=initial)


def test_canonical_writer_rejects_namespace_metadata_id_mismatch(tmp_path) -> None:
    namespace = tmp_path / "mldb_data" / "demo"
    namespace.mkdir(parents=True)
    (namespace / "namespace.yaml").write_text(
        json.dumps({"schema": "mjtensu.mldb-v2/namespace/v1", "id": "other"}),
        encoding="utf-8",
    )
    writer = _writer(tmp_path)
    document = {
        "schema": "mjtensu.mldb-v2/study-plan/v1",
        "id": "demo/example-plan",
    }
    with pytest.raises(ValueError, match="namespace"):
        writer.create_immutable(
            kind=EntityKind.STUDY_PLAN,
            entity_id=StudyPlanId("demo/example-plan"),
            document=document,
        )


def test_study_result_rejects_ids_beyond_four_digit_trial_eval_grammar(tmp_path) -> None:
    from mldb_v2.src.repository.canonical_writes import _validate_study_result_record

    accepted: list[str] = []
    initial = _study_result()
    trial_case = copy.deepcopy(initial)
    trial_case["status"] = "completed"
    trial_case["trials"] = [
        {"trial": f"trial-{index:04d}", "training": None, "evaluations": []}
        for index in range(1, 10001)
    ]
    try:
        _validate_study_result_record(trial_case, entity_id=StudyResultId(trial_case["id"]))
    except ValueError:
        pass
    else:
        accepted.append("trial-10000")

    eval_case = copy.deepcopy(initial)
    eval_case["status"] = "completed"
    eval_case["trials"][0]["training"] = None
    eval_case["trials"][0]["evaluations"] = [
        {
            "coordinate": f"eval-{index:04d}",
            "stage": "holdout",
            "disposition": "completed",
            "result": f"demo/run-{eval_case['execution_key']}-trial-0001-eval-{index:04d}",
            "reason": None,
        }
        for index in range(1, 10001)
    ]
    try:
        _validate_study_result_record(eval_case, entity_id=StudyResultId(eval_case["id"]))
    except ValueError:
        pass
    else:
        accepted.append("eval-10000")

    assert accepted == []


def test_artifact_ref_rejects_forbidden_credential_and_cache_metadata() -> None:
    import hashlib

    from mldb_v2.src.storage.artifact_reference import _validate_artifact_ref

    base = {
        "uri": "s3://bucket/object.bin",
        "bytes": 1,
        "sha256": hashlib.sha256(b"x").hexdigest(),
    }
    accepted: list[str] = []
    for key, value in {
        "password": "secret",
        "access_key": "secret",
        "presigned_url": "https://example.invalid/?signature=secret",
        "cache_path": "C:/runtime/cache/object.bin",
    }.items():
        candidate = dict(base)
        candidate[key] = value
        try:
            _validate_artifact_ref(candidate)
        except ValueError:
            pass
        else:
            accepted.append(key)
    assert accepted == []


def test_corpus_manifest_rejects_duplicate_json_members() -> None:
    from mldb_v2.src.storage.corpus_manifest import _parse_corpus_manifest

    digest = "0" * 64
    cases = [
        f'{{"path":"a","path":"b","bytes":1,"sha256":"{digest}"}}'.encode(),
        f'{{"path":"a","bytes":1,"bytes":2,"sha256":"{digest}"}}'.encode(),
        f'{{"path":"a","bytes":1,"sha256":"{digest}","sha256":"{"1" * 64}"}}'.encode(),
    ]
    accepted: list[bytes] = []
    for row in cases:
        try:
            _parse_corpus_manifest(row + b"\n")
        except ValueError:
            pass
        else:
            accepted.append(row)
    assert accepted == []


def test_yaml_unicode_nonbreaking_space_is_not_silently_stripped() -> None:
    # YAML separation whitespace is ASCII space/tab; NBSP is scalar content.
    # The supported subset must preserve it or fail closed, never normalize it away.
    source = "value: foo\u00a0\n"
    try:
        parsed = _load_yaml(source)
    except _YamlError:
        return
    assert parsed == {"value": "foo\u00a0"}


def test_listing_reports_unexpected_files_in_canonical_namespace_tree(tmp_path) -> None:
    repo = _repo(tmp_path)
    namespace = repo / "mldb_data" / "demo"
    tasks = namespace / "tasks"
    tasks.mkdir()
    (tasks / "garbage.txt").write_text("not canonical\n", encoding="utf-8")
    (namespace / "unexpected.txt").write_text("not canonical\n", encoding="utf-8")

    from mldb_v2.src.repository.listing import CanonicalRepositoryListing

    listing = CanonicalRepositoryListing(repo / "mldb_data").list_entities()
    messages = [issue["message"] for issue in listing["issues"]]
    assert any("garbage.txt" in message for message in messages)
    assert any("unexpected.txt" in message for message in messages)


def test_yaml_leading_bom_is_supported_as_document_bom_or_fails_closed() -> None:
    source = "\ufeffschema: mjtensu.mldb-v2/namespace/v1\nid: demo\n"
    try:
        parsed = _load_yaml(source)
    except _YamlError:
        return
    assert parsed == {
        "schema": "mjtensu.mldb-v2/namespace/v1",
        "id": "demo",
    }


def test_artifact_ref_rejects_backend_local_absolute_path_under_generic_nested_key() -> None:
    import hashlib

    from mldb_v2.src.storage.artifact_reference import _validate_artifact_ref

    base = {
        "uri": "s3://bucket/object.bin",
        "bytes": 1,
        "sha256": hashlib.sha256(b"x").hexdigest(),
    }
    accepted: list[str] = []
    for path in ["C:/runtime/object.bin", "/srv/runtime/object.bin"]:
        candidate = dict(base)
        candidate["metadata"] = {"backend": {"path": path}}
        try:
            _validate_artifact_ref(candidate)
        except ValueError:
            pass
        else:
            accepted.append(path)
    assert accepted == []
