from __future__ import annotations

import inspect
from pathlib import Path
from typing import cast, get_type_hints

import pytest

from mldb_v2.src.backend._config import BackendConfig, BackendConfigurationError
from mldb_v2.src.backend._registry import (
    BackendRegistrationError,
    BackendRegistry,
    UnknownBackendError,
)
from mldb_v2.src.backend.backend_port import BackendPort
from mldb_v2.src.backend.candidate_outcome import (
    BackendObservation,
    StageKey,
    TerminalCandidate,
    TrainingStageKey,
)
from mldb_v2.src.backend.stage_input import StageInput, TrainingStageInput
from mldb_v2.src.common.ids import (
    ArchitectureId,
    CorpusId,
    StudyPlanId,
    StudyResultId,
    TaskId,
    TrainProtocolId,
    TrialId,
)


def _training_stage_input() -> TrainingStageInput:
    return {
        "schema": "mjtensu.mldb-v2/stage-input/v1",
        "study_result": StudyResultId("demo/run-abcd"),
        "plan": StudyPlanId("demo/study-plan-0123456789abcdef"),
        "plan_sha256": "1" * 64,
        "trial": TrialId("trial-0001"),
        "kind": "training",
        "coordinate": None,
        "source_commit": "2" * 40,
        "pins": [],
        "stage": {
            "task": TaskId("demo/task"),
            "corpus": CorpusId("demo/corpus"),
            "architecture": ArchitectureId("demo/architecture"),
            "train_protocol": TrainProtocolId("demo/train-protocol"),
            "parameters": {"batch_size": 8},
            "seed": 42,
        },
        "runtime_model": None,
    }


def _stage_key(stage_input: TrainingStageInput) -> TrainingStageKey:
    return {
        "study_result": stage_input["study_result"],
        "plan": stage_input["plan"],
        "trial": stage_input["trial"],
        "kind": "training",
        "coordinate": None,
        "source_commit": stage_input["source_commit"],
    }


class RecordingBackend:
    def __init__(self, config: BackendConfig) -> None:
        self.config = config
        self.admitted: StageInput | None = None
        self.cancelled: StudyResultId | None = None
        self.observation: BackendObservation | None = None
        self.candidate: TerminalCandidate | None = None

    def admit(self, *, stage_input: StageInput) -> BackendObservation:
        self.admitted = stage_input
        key = cast(StageKey, _stage_key(cast(TrainingStageInput, stage_input)))
        self.observation = {
            "state": "active",
            "stage_key": key,
            "backend": self.config.backend_type,
            "execution_ids": ["opaque-execution-id"],
        }
        return self.observation

    def observe(self, *, stage_key: StageKey) -> BackendObservation | None:
        if self.observation is not None and self.observation["stage_key"] == stage_key:
            return self.observation
        return self.candidate

    def collect(self, *, stage_key: StageKey) -> TerminalCandidate | None:
        if self.candidate is None or self.candidate["stage_key"] != stage_key:
            return None
        return self.candidate

    def cancel_study(self, *, study_result: StudyResultId) -> None:
        self.cancelled = study_result


def test_backend_port_surface_matches_frozen_runtime_contract() -> None:
    admit = inspect.signature(BackendPort.admit)
    observe = inspect.signature(BackendPort.observe)
    collect = inspect.signature(BackendPort.collect)
    cancel = inspect.signature(BackendPort.cancel_study)

    assert admit.parameters["stage_input"].kind is inspect.Parameter.KEYWORD_ONLY
    assert observe.parameters["stage_key"].kind is inspect.Parameter.KEYWORD_ONLY
    assert collect.parameters["stage_key"].kind is inspect.Parameter.KEYWORD_ONLY
    assert cancel.parameters["study_result"].kind is inspect.Parameter.KEYWORD_ONLY

    assert get_type_hints(BackendPort.admit) == {
        "stage_input": StageInput,
        "return": BackendObservation,
    }
    assert get_type_hints(BackendPort.observe) == {
        "stage_key": StageKey,
        "return": BackendObservation | None,
    }
    assert get_type_hints(BackendPort.collect) == {
        "stage_key": StageKey,
        "return": TerminalCandidate | None,
    }
    assert get_type_hints(BackendPort.cancel_study) == {
        "study_result": StudyResultId,
        "return": type(None),
    }


def test_registry_resolves_configured_backend_and_preserves_runtime_values() -> None:
    registry = BackendRegistry()
    created: list[RecordingBackend] = []

    def factory(config: BackendConfig) -> BackendPort:
        backend = RecordingBackend(config)
        created.append(backend)
        return backend

    registry.register("fake", factory)
    sdk_object = object()
    config = BackendConfig(
        backend_type="fake",
        options={
            "endpoint": "https://backend.invalid",
            "credentials": {"token": "top-secret"},
            "queue": "gpu-a",
            "sdk": sdk_object,
        },
    )
    backend = registry.resolve(config)
    assert backend is created[0]
    assert created[0].config is config
    assert config.options["sdk"] is sdk_object
    assert "top-secret" not in repr(config)

    stage_input = _training_stage_input()
    original = dict(stage_input)
    observation = backend.admit(stage_input=stage_input)
    stage_key = _stage_key(stage_input)

    assert created[0].admitted is stage_input
    assert stage_input == original
    assert set(stage_input) == {
        "schema",
        "study_result",
        "plan",
        "plan_sha256",
        "trial",
        "kind",
        "coordinate",
        "source_commit",
        "pins",
        "stage",
        "runtime_model",
    }
    assert "endpoint" not in stage_input
    assert "credentials" not in stage_input
    assert "queue" not in stage_input
    assert observation["stage_key"] == stage_key
    assert backend.observe(stage_key=stage_key) == observation

    candidate = cast(
        TerminalCandidate,
        {
            "state": "terminal",
            "stage_key": stage_key,
            "attempts": [
                {
                    "backend": "fake",
                    "execution_id": "opaque-execution-id",
                    "status": "completed",
                    "started_at": None,
                    "ended_at": None,
                    "diagnostic": None,
                }
            ],
            "status": "completed",
            "diagnostic": None,
            "result": {
                "weights": {
                    "uri": "s3://bucket/weights.bin",
                    "bytes": 1,
                    "sha256": "3" * 64,
                    "format": "pytorch-state-dict/v1",
                }
            },
        },
    )
    created[0].candidate = candidate
    assert backend.collect(stage_key=stage_key) is candidate
    assert "top-secret" not in repr(candidate)

    backend.cancel_study(study_result=stage_input["study_result"])
    assert created[0].cancelled == stage_input["study_result"]


def test_backend_config_is_top_level_immutable_and_allows_operational_objects() -> None:
    sdk_object = object()
    source = {"sdk": sdk_object}
    config = BackendConfig("fake", source)
    source["later"] = object()
    assert dict(config.options) == {"sdk": sdk_object}
    with pytest.raises(TypeError):
        config.options["other"] = object()  # type: ignore[index]


@pytest.mark.parametrize("backend_type", ["", " fake", "fake ", "   "])
def test_invalid_backend_configuration_rejects_invalid_backend_type(
    backend_type: str,
) -> None:
    with pytest.raises(BackendConfigurationError):
        BackendConfig(backend_type)


@pytest.mark.parametrize(
    "options",
    [
        [],
        {1: "value"},
        {"": "value"},
        {" bad": "value"},
        {"bad ": "value"},
    ],
)
def test_invalid_backend_configuration_rejects_invalid_options(options: object) -> None:
    with pytest.raises(BackendConfigurationError):
        BackendConfig("fake", cast(object, options))  # type: ignore[arg-type]


def test_registry_rejects_invalid_duplicate_and_non_callable_registration() -> None:
    registry = BackendRegistry()
    factory = lambda config: RecordingBackend(config)
    registry.register("fake", factory)

    with pytest.raises(BackendRegistrationError, match="already registered"):
        registry.register("fake", factory)
    with pytest.raises(BackendRegistrationError):
        registry.register(" fake", factory)
    with pytest.raises(BackendRegistrationError, match="callable"):
        registry.register("other", cast(object, 123))  # type: ignore[arg-type]


def test_unknown_backend_is_bounded_resolution_error() -> None:
    registry = BackendRegistry()
    with pytest.raises(UnknownBackendError, match="unknown backend type: missing"):
        registry.resolve(BackendConfig("missing"))


def test_factory_must_return_backend_port_compatible_object() -> None:
    registry = BackendRegistry()
    registry.register("broken", lambda config: cast(BackendPort, object()))
    with pytest.raises(TypeError, match="missing BackendPort methods"):
        registry.resolve(BackendConfig("broken"))


def test_backend_runtime_modules_have_no_forbidden_runtime_dependencies() -> None:
    root = Path(__file__).parents[1]
    for relative in (
        "src/backend/backend_port.py",
        "src/backend/_config.py",
        "src/backend/_registry.py",
    ):
        source = (root / relative).read_text(encoding="utf-8")
        assert "mldb_v2.skeleton" not in source
        assert "clearml" not in source.lower()
        assert "execution_readiness" not in source
        assert "result_acceptance" not in source
        assert "scheduler" not in source.lower()
