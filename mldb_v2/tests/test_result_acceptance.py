from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
import torch

from mldb_v2.src.common.ids import EntityKind, ModelId, TrainingResultId
from mldb_v2.src.repository.canonical_writes import CanonicalRepositoryWriter
from mldb_v2.src.storage.object_bytes import _ObjectByteAccess
from mldb_v2.src.training.canonical_weights import _serialize_canonical_state_dict
from mldb_v2.src.verification.result_acceptance import (
    AcceptedResultRecordValidator,
    ResultAcceptor,
)
import mldb_v2.tests.test_evaluation_result_acceptance as eval_fx
import mldb_v2.tests.test_training_result_acceptance as train_fx


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def test_result_acceptor_delegates_training_to_actual_runtime(tmp_path: Path) -> None:
    plan = train_fx._plan()
    study_result = train_fx._study_result(plan)
    stage_input = train_fx._stage_input(plan)
    root = train_fx._install_architecture(tmp_path)
    data = _serialize_canonical_state_dict(torch.nn.Linear(3, 2).state_dict())
    candidate = train_fx._completed_candidate(stage_input, train_fx._weight_ref(data))
    transport = train_fx.DictTransport({candidate["result"]["weights"]["uri"]: data})
    acceptor = ResultAcceptor(
        mldb_data_root=root,
        object_bytes=_ObjectByteAccess(transport),
    )

    accepted = acceptor.accept_training(
        request={
            "study_result": study_result,
            "plan": plan,
            "stage_input": stage_input,
            "candidate": candidate,
        }
    )

    assert accepted["training_result"]["status"] == "completed"
    assert accepted["model"] is not None
    assert accepted["model"]["training_result"] == accepted["training_result"]["id"]


def test_result_acceptor_delegates_evaluation_to_actual_runtime(tmp_path: Path) -> None:
    fx = eval_fx._fixture(tmp_path)
    acceptor = ResultAcceptor(
        mldb_data_root=fx["root"],
        object_bytes=fx["object_bytes"],
    )
    accepted = acceptor.accept_evaluation(request=fx["request"])
    assert accepted["evaluation_result"]["status"] == "completed"
    assert accepted["evaluation_result"]["id"].endswith("trial-0001-eval-0001")


def test_accepted_record_validator_and_writer_validate_replay(tmp_path: Path) -> None:
    plan = train_fx._plan()
    root = train_fx._install_architecture(tmp_path)
    _write_json(
        root / "demo" / "study_plans" / f"{plan['id'].split('/', 1)[1]}.yaml",
        plan,
    )
    study_result = train_fx._study_result(plan)
    stage_input = train_fx._stage_input(plan)
    data = _serialize_canonical_state_dict(torch.nn.Linear(3, 2).state_dict())
    candidate = train_fx._completed_candidate(stage_input, train_fx._weight_ref(data))
    transport = train_fx.DictTransport({candidate["result"]["weights"]["uri"]: data})
    accepted = ResultAcceptor(
        mldb_data_root=root,
        object_bytes=_ObjectByteAccess(transport),
    ).accept_training(
        request={
            "study_result": study_result,
            "plan": plan,
            "stage_input": stage_input,
            "candidate": candidate,
        }
    )

    validator = AcceptedResultRecordValidator(mldb_data_root=root)
    writer = CanonicalRepositoryWriter(tmp_path, record_validator=validator)
    training = accepted["training_result"]
    model = accepted["model"]
    assert model is not None
    writer.create_immutable(
        kind=EntityKind.TRAINING_RESULT,
        entity_id=TrainingResultId(training["id"]),
        document=training,
    )
    writer.create_immutable(
        kind=EntityKind.MODEL,
        entity_id=ModelId(model["id"]),
        document=model,
    )

    training_path = (
        root
        / "demo"
        / "training_results"
        / f"{training['id'].split('/', 1)[1]}.yaml"
    )
    invalid_existing = copy.deepcopy(training)
    invalid_existing["result"]["model"] = "demo/wrong-model"
    _write_json(training_path, invalid_existing)

    with pytest.raises(ValueError):
        writer.create_immutable(
            kind=EntityKind.TRAINING_RESULT,
            entity_id=TrainingResultId(training["id"]),
            document=training,
        )


def test_evaluation_persistence_validator_reuses_acceptance_contract_without_object_io(
    tmp_path: Path,
) -> None:
    fx = eval_fx._fixture(tmp_path)
    plan = fx["plan"]
    _write_json(
        fx["root"] / "demo" / "study_plans" / f"{plan['id'].split('/', 1)[1]}.yaml",
        plan,
    )
    accepted = ResultAcceptor(
        mldb_data_root=fx["root"],
        object_bytes=fx["object_bytes"],
    ).accept_evaluation(request=fx["request"])["evaluation_result"]

    validator = AcceptedResultRecordValidator(mldb_data_root=fx["root"])
    validator.validate(
        kind=EntityKind.EVALUATION_RESULT,
        entity_id=accepted["id"],
        document=accepted,
    )

    invalid = copy.deepcopy(accepted)
    invalid["result"]["metrics"]["extra"] = 1
    with pytest.raises(ValueError, match="undeclared"):
        validator.validate(
            kind=EntityKind.EVALUATION_RESULT,
            entity_id=accepted["id"],
            document=invalid,
        )

    # Persistence-time validation must not fetch artifact/object bytes again.
    assert fx["transport"].publish_calls == 0
