"""Audited recovery must never silently discard a genuine terminal result."""
import copy
import hashlib
import json
from pathlib import Path

import pytest

from mldb_v2.src.common.ids import StudyResultId, _canonical_json_bytes
from mldb_v2.tests.test_repository_canonical_writes import _repo, _writer, _study_result


def _failed_with_condition(tmp_path: Path):
    root = _repo(tmp_path)
    writer = _writer(root)
    initial = _study_result()
    initial["trials"][0]["condition"] = {
        "label": "NanoDet M320 | jp_fraction=0.25",
        "parameters": {"jp_fraction": 0.25},
    }
    study_id = StudyResultId(initial["id"])
    writer.create_study_result(entity_id=study_id, document=initial)
    failed = copy.deepcopy(initial)
    failed["status"] = "failed"
    failed["diagnostic"] = {
        "code": "study_progression_failed",
        "message": "Study progression cannot continue from canonical inputs.",
    }
    failed["trials"][0]["training"].update(
        disposition="skipped", result=None, reason="global_failure"
    )
    failed["trials"][0]["evaluations"][0].update(
        disposition="skipped", result=None, reason="global_failure"
    )
    writer.replace_nonterminal_study_result(entity_id=study_id, replacement=failed)
    digest = hashlib.sha256(_canonical_json_bytes(failed)).hexdigest()
    return root, writer, study_id, failed, digest


def test_recovery_preserves_failed_snapshot_and_all_conditions(tmp_path: Path) -> None:
    root, writer, study_id, failed, digest = _failed_with_condition(tmp_path)
    evidence = {"training_task_ids": {"trial-0001": "clearml-owner-1"}}
    with pytest.raises(ValueError, match="fingerprint"):
        writer.recover_premature_global_failure(
            entity_id=study_id, expected_sha256="f"*64, backend_evidence=evidence
        )
    reopened = writer.recover_premature_global_failure(
        entity_id=study_id, expected_sha256=digest, backend_evidence=evidence
    )
    assert reopened["status"] == "submitted"
    assert reopened["diagnostic"] is None
    assert reopened["trials"][0]["condition"] == failed["trials"][0]["condition"]
    assert reopened["trials"][0]["training"]["disposition"] == "pending"
    assert reopened["trials"][0]["evaluations"][0]["disposition"] == "pending"
    audit = root / ".local/mldb_v2/recovery_audit" / f"{study_id.split('/', 1)[1]}-{digest}.json"
    assert json.loads(audit.read_text())["prior_record"] == failed
    with pytest.raises(ValueError, match="fingerprint"):
        writer.recover_premature_global_failure(
            entity_id=study_id, expected_sha256=digest, backend_evidence=evidence
        )


def test_recovery_rejects_unverified_training_tasks(tmp_path: Path) -> None:
    _root, writer, study_id, _failed, digest = _failed_with_condition(tmp_path)
    with pytest.raises(ValueError, match="evidence"):
        writer.recover_premature_global_failure(
            entity_id=study_id, expected_sha256=digest,
            backend_evidence={"training_task_ids": {}},
        )
