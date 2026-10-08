"""Regression: visible Study conditions must survive training and evaluation acceptance."""
import copy
from pathlib import Path

import torch

import mldb_v2.tests.test_evaluation_result_acceptance as eval_fx
import mldb_v2.tests.test_training_result_acceptance as train_fx
from mldb_v2.src.storage.object_bytes import _ObjectByteAccess
from mldb_v2.src.training.canonical_weights import _serialize_canonical_state_dict
from mldb_v2.src.verification.result_acceptance import ResultAcceptor


def _with_condition(study_result: dict) -> dict:
    payload = copy.deepcopy(study_result)
    for trial in payload["trials"]:
        trial["condition"] = {
            "label": "NanoDet Plus M320 | jp_fraction=0.25",
            "parameters": {"jp_fraction": 0.25, "seed": 42},
        }
    return payload


def test_training_accepts_visible_study_condition(tmp_path: Path) -> None:
    plan = train_fx._plan()
    result = _with_condition(train_fx._study_result(plan))
    stage_input = train_fx._stage_input(plan)
    root = train_fx._install_architecture(tmp_path)
    weights = _serialize_canonical_state_dict(torch.nn.Linear(3, 2).state_dict())
    candidate = train_fx._completed_candidate(stage_input, train_fx._weight_ref(weights))
    transport = train_fx.DictTransport({candidate["result"]["weights"]["uri"]: weights})
    output = ResultAcceptor(
        mldb_data_root=root,
        object_bytes=_ObjectByteAccess(transport),
    ).accept_training(request={
        "study_result": result, "plan": plan,
        "stage_input": stage_input, "candidate": candidate,
    })
    assert output["training_result"]["status"] == "completed"


def test_evaluation_accepts_visible_study_condition(tmp_path: Path) -> None:
    fx = eval_fx._fixture(tmp_path)
    request = dict(fx["request"])
    request["study_result"] = _with_condition(request["study_result"])
    accepted = ResultAcceptor(
        mldb_data_root=fx["root"],
        object_bytes=fx["object_bytes"],
    ).accept_evaluation(request=request)
    assert accepted["evaluation_result"]["status"] == "completed"
