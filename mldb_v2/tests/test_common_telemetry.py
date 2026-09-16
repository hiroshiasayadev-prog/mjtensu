from __future__ import annotations

import ast
import dataclasses
import inspect
from pathlib import Path
from typing import get_type_hints

import pytest

import mldb_v2.skeleton.common.telemetry as skeleton_telemetry_module
import mldb_v2.src.common.telemetry as telemetry_module
from mldb_v2.skeleton.common.telemetry import TelemetryReporter as SkeletonTelemetryReporter
from mldb_v2.skeleton.evaluation.evaluate_interface import (
    EvaluationContext as SkeletonEvaluationContext,
)
from mldb_v2.skeleton.training.train_interface import TrainContext as SkeletonTrainContext
from mldb_v2.src.common.telemetry import (
    TelemetryReporter,
    _AcceptedScalarEvent,
    _RecordingTelemetryReporter,
)
from mldb_v2.src.evaluation.evaluate_interface import EvaluationContext
from mldb_v2.src.training.train_interface import TrainContext


def _field_names(cls: type) -> tuple[str, ...]:
    return tuple(field.name for field in dataclasses.fields(cls))


def _assert_reporter_shape(reporter_type: type) -> None:
    assert getattr(reporter_type, "_is_protocol", False) is True
    method = reporter_type.report_scalar
    signature = inspect.signature(method)
    assert tuple(signature.parameters) == ("self", "group", "series", "value", "step")
    assert signature.parameters["self"].kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
    for name in ("group", "series", "value", "step"):
        assert signature.parameters[name].kind is inspect.Parameter.KEYWORD_ONLY
    hints = get_type_hints(method)
    assert hints["group"] is str
    assert hints["series"] is str
    assert hints["value"] == int | float
    assert hints["step"] is int
    assert hints["return"] is type(None)


def test_telemetry_reporter_public_shape_matches_frozen_contract() -> None:
    _assert_reporter_shape(SkeletonTelemetryReporter)
    _assert_reporter_shape(TelemetryReporter)


def test_context_shapes_are_exact_frozen_and_require_telemetry() -> None:
    train_fields = (
        "task", "corpus", "architecture", "model", "seed", "parameters", "telemetry", "work_dir"
    )
    evaluation_fields = ("task", "corpus", "model", "parameters", "telemetry", "work_dir")
    for cls in (SkeletonTrainContext, TrainContext):
        assert _field_names(cls) == train_fields
        assert cls.__dataclass_params__.frozen is True
        telemetry_field = next(field for field in dataclasses.fields(cls) if field.name == "telemetry")
        assert telemetry_field.default is dataclasses.MISSING
        assert telemetry_field.default_factory is dataclasses.MISSING
        assert cls.__annotations__["telemetry"] == "TelemetryReporter"
    for cls in (SkeletonEvaluationContext, EvaluationContext):
        assert _field_names(cls) == evaluation_fields
        assert cls.__dataclass_params__.frozen is True
        telemetry_field = next(field for field in dataclasses.fields(cls) if field.name == "telemetry")
        assert telemetry_field.default is dataclasses.MISSING
        assert telemetry_field.default_factory is dataclasses.MISSING
        assert cls.__annotations__["telemetry"] == "TelemetryReporter"


def test_valid_scalar_calls_are_recorded_in_exact_order() -> None:
    reporter = _RecordingTelemetryReporter()
    assert reporter.report_scalar(
        group="optimization", series="loss", value=1, step=0
    ) is None
    assert reporter.report_scalar(
        group="quality", series="score", value=-0.25, step=100
    ) is None
    assert reporter.accepted_count == 2
    assert reporter.accepted_events == (
        _AcceptedScalarEvent(group="optimization", series="loss", value=1, step=0),
        _AcceptedScalarEvent(group="quality", series="score", value=-0.25, step=100),
    )


def test_accepted_events_are_delivered_to_sink_in_exact_order() -> None:
    delivered: list[_AcceptedScalarEvent] = []
    reporter = _RecordingTelemetryReporter(sink=delivered.append)
    reporter.report_scalar(group="optimization", series="loss", value=1.25, step=3)
    reporter.report_scalar(group="validation", series="mean_iou", value=0.7, step=5)
    assert reporter.accepted_events == tuple(delivered)


def test_sink_failure_is_isolated_after_acceptance() -> None:
    def failing_sink(_event: _AcceptedScalarEvent) -> None:
        raise RuntimeError("delivery unavailable")

    reporter = _RecordingTelemetryReporter(sink=failing_sink)
    assert reporter.report_scalar(
        group="optimization", series="loss", value=1.25, step=3
    ) is None
    assert reporter.accepted_count == 1


def test_validation_failure_does_not_call_sink() -> None:
    delivered: list[_AcceptedScalarEvent] = []
    reporter = _RecordingTelemetryReporter(sink=delivered.append)
    with pytest.raises(ValueError, match="telemetry"):
        reporter.report_scalar(group="", series="loss", value=1.25, step=3)
    assert reporter.accepted_count == 0
    assert delivered == []


def test_reporter_without_sink_preserves_local_behavior() -> None:
    reporter = _RecordingTelemetryReporter()
    reporter.report_scalar(group="fixture", series="signal", value=1, step=0)
    assert reporter.accepted_count == 1


@pytest.mark.parametrize(
    "kwargs",
    [
        {"group": "", "series": "loss", "value": 1, "step": 0},
        {"group": "optimization", "series": "", "value": 1, "step": 0},
        {"group": 1, "series": "loss", "value": 1, "step": 0},
        {"group": "optimization", "series": object(), "value": 1, "step": 0},
        {"group": "optimization", "series": "loss", "value": True, "step": 0},
        {"group": "optimization", "series": "loss", "value": False, "step": 0},
        {"group": "optimization", "series": "loss", "value": "1", "step": 0},
        {"group": "optimization", "series": "loss", "value": float("nan"), "step": 0},
        {"group": "optimization", "series": "loss", "value": float("inf"), "step": 0},
        {"group": "optimization", "series": "loss", "value": float("-inf"), "step": 0},
        {"group": "optimization", "series": "loss", "value": 1, "step": True},
        {"group": "optimization", "series": "loss", "value": 1, "step": False},
        {"group": "optimization", "series": "loss", "value": 1, "step": -1},
        {"group": "optimization", "series": "loss", "value": 1, "step": 1.5},
        {"group": "optimization", "series": "loss", "value": 1, "step": "1"},
    ],
)
def test_scalar_validation_rejects_malformed_inputs_without_coercion(kwargs: dict[str, object]) -> None:
    reporter = _RecordingTelemetryReporter()
    with pytest.raises(ValueError, match="telemetry"):
        reporter.report_scalar(**kwargs)  # type: ignore[arg-type]
    assert reporter.accepted_count == 0
    assert reporter.accepted_events == ()


def test_telemetry_modules_have_no_backend_specific_dependencies() -> None:
    forbidden_exact = {"clearml", "boto3"}
    forbidden_fragments = ("mldb_v2.src.backend", ".queue", ".worker")
    for module in (skeleton_telemetry_module, telemetry_module):
        source = Path(module.__file__).read_text(encoding="utf-8")
        tree = ast.parse(source)
        imports: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module is not None:
                imports.append(node.module)
        assert all(name.split(".", 1)[0] not in forbidden_exact for name in imports)
        assert all(
            fragment not in name
            for name in imports
            for fragment in forbidden_fragments
        )
