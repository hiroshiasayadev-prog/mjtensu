"""MLDB v2 reusable Evaluation Protocol definition shape and private parsing."""

from pathlib import Path
from typing import Literal, Mapping, NotRequired, TypeAlias, TypedDict

from mldb_v2.src.catalog._executable_definition_loading import (
    _require_exact_keys,
    _require_string,
    _resolve_document,
    _validate_definition_id,
    _validate_implementation,
    _validate_parameter_declarations,
    _validate_status,
    _validate_task_reference,
    _validate_versioned_schema,
)
from mldb_v2.src.common.ids import EntityKind, EvaluationProtocolId, TaskId
from mldb_v2.src.common.parameters import PublicParameterDeclarations
from mldb_v2.src.verification.executable_integrity import ExecutableSource


class EvaluationProtocolImplementation(TypedDict):
    entrypoint: Literal["evaluate"]
    sha256: NotRequired[str]
    sources: NotRequired[list[ExecutableSource]]


class EvaluationMetricDeclaration(TypedDict):
    type: Literal["integer", "number"]
    required: bool
    description: NotRequired[str]
    preference: NotRequired[Literal["higher", "lower", "neutral"]]


class EvaluationArtifactDeclaration(TypedDict):
    format: str
    schema: str
    required: bool
    description: NotRequired[str]
    study_view: NotRequired[Literal["hidden", "select", "all"]]


EvaluationMetricDeclarations: TypeAlias = Mapping[str, EvaluationMetricDeclaration]
EvaluationArtifactDeclarations: TypeAlias = Mapping[str, EvaluationArtifactDeclaration]


class EvaluationProtocol(TypedDict):
    schema: Literal["mjtensu.mldb-v2/evaluation-protocol/v1"]
    id: EvaluationProtocolId
    status: Literal["draft", "sealed"]
    task: TaskId
    name: str
    description: str
    implementation: EvaluationProtocolImplementation
    parameters: PublicParameterDeclarations
    metrics: EvaluationMetricDeclarations
    artifacts: EvaluationArtifactDeclarations
    model_parameters: NotRequired[Mapping[str, str]]
    onnx_input_parameter: NotRequired[str]
    artifact_inputs: NotRequired[Mapping[str, str]]
    artifact_only: NotRequired[bool]


_REQUIRED_TOP_LEVEL = {
    "schema", "id", "status", "task", "name", "description",
    "implementation", "parameters", "metrics", "artifacts",
}
_OPTIONAL_TOP_LEVEL = {"model_parameters", "onnx_input_parameter", "artifact_inputs", "artifact_only"}


def _validate_metric_declarations(value: object) -> dict[str, EvaluationMetricDeclaration]:
    if type(value) is not dict:
        raise ValueError("metrics must be a mapping")
    result: dict[str, EvaluationMetricDeclaration] = {}
    for key, declaration in value.items():
        _require_string(key, label="metric name", nonempty=True)
        mapping = _require_exact_keys(
            declaration,
            required={"type", "required"},
            optional={"description", "preference"},
            label=f"metric {key}",
        )
        metric_type = mapping["type"]
        if type(metric_type) is not str or metric_type not in {"integer", "number"}:
            raise ValueError("metric type must be exactly integer or number")
        required = mapping["required"]
        if type(required) is not bool:
            raise ValueError("metric required must be a boolean")
        parsed: EvaluationMetricDeclaration = {"type": metric_type, "required": required}  # type: ignore[typeddict-item]
        if "description" in mapping:
            parsed["description"] = _require_string(
                mapping["description"], label="metric description"
            )
        if "preference" in mapping:
            preference = mapping["preference"]
            if type(preference) is not str or preference not in {"higher", "lower", "neutral"}:
                raise ValueError("metric preference must be exactly higher, lower, or neutral")
            parsed["preference"] = preference  # type: ignore[typeddict-item]
        result[key] = parsed
    return result


def _validate_artifact_declarations(value: object) -> dict[str, EvaluationArtifactDeclaration]:
    if type(value) is not dict:
        raise ValueError("artifacts must be a mapping")
    result: dict[str, EvaluationArtifactDeclaration] = {}
    for key, declaration in value.items():
        _require_string(key, label="artifact name", nonempty=True)
        mapping = _require_exact_keys(
            declaration,
            required={"format", "schema", "required"},
            optional={"description", "study_view"},
            label=f"artifact {key}",
        )
        artifact_format = _require_string(mapping["format"], label="artifact format", nonempty=True)
        schema = _validate_versioned_schema(mapping["schema"])
        required = mapping["required"]
        if type(required) is not bool:
            raise ValueError("artifact required must be a boolean")
        parsed: EvaluationArtifactDeclaration = {
            "format": artifact_format,
            "schema": schema,
            "required": required,
        }
        if "description" in mapping:
            parsed["description"] = _require_string(
                mapping["description"], label="artifact description"
            )
        if "study_view" in mapping:
            study_view = mapping["study_view"]
            if type(study_view) is not str or study_view not in {"hidden", "select", "all"}:
                raise ValueError("artifact study_view must be exactly hidden, select, or all")
            parsed["study_view"] = study_view  # type: ignore[typeddict-item]
        result[key] = parsed
    return result



def _validate_model_parameters(
    value: object,
    parameters: PublicParameterDeclarations,
) -> dict[str, str]:
    if type(value) is not dict:
        raise ValueError("model_parameters must be a mapping")
    result: dict[str, str] = {}
    seen_parameters: set[str] = set()
    for alias, parameter_name in value.items():
        _require_string(alias, label="model parameter alias", nonempty=True)
        parameter = _require_string(
            parameter_name, label=f"model parameter {alias}", nonempty=True
        )
        if parameter not in parameters:
            raise ValueError("model_parameters references an undeclared public parameter")
        if parameter in seen_parameters:
            raise ValueError("model_parameters must not reuse a public parameter")
        seen_parameters.add(parameter)
        result[alias] = parameter
    return result

def _parse_evaluation_protocol_document(
    document: object, *, expected_id: str
) -> EvaluationProtocol:
    mapping = _require_exact_keys(
        document,
        required=_REQUIRED_TOP_LEVEL,
        optional=_OPTIONAL_TOP_LEVEL,
        label="Evaluation Protocol",
    )
    if mapping["schema"] != "mjtensu.mldb-v2/evaluation-protocol/v1":
        raise ValueError("unsupported Evaluation Protocol schema")
    definition_id = _validate_definition_id(mapping["id"])
    if definition_id != expected_id:
        raise ValueError("Evaluation Protocol id does not match canonical path identity")
    status = _validate_status(mapping["status"])
    task = _validate_task_reference(mapping["task"])
    name = _require_string(mapping["name"], label="Evaluation Protocol name", nonempty=True)
    description = _require_string(
        mapping["description"], label="Evaluation Protocol description"
    )
    implementation = _validate_implementation(
        mapping["implementation"], entrypoint="evaluate", sealed=status == "sealed",
        namespace=definition_id.split("/", 1)[0],
    )
    parameters = _validate_parameter_declarations(mapping["parameters"])
    metrics = _validate_metric_declarations(mapping["metrics"])
    artifacts = _validate_artifact_declarations(mapping["artifacts"])
    model_parameters = (
        _validate_model_parameters(mapping["model_parameters"], parameters)
        if "model_parameters" in mapping
        else {}
    )
    onnx_input_parameter = mapping.get("onnx_input_parameter")
    if onnx_input_parameter is not None:
        onnx_input_parameter = _require_string(
            onnx_input_parameter, label="onnx_input_parameter", nonempty=True
        )
        if onnx_input_parameter not in parameters:
            raise ValueError("onnx_input_parameter must be a declared parameter")
        declaration = parameters[onnx_input_parameter]
        if declaration.get("type") != "string":
            raise ValueError("onnx_input_parameter requires a string parameter")
        if onnx_input_parameter in model_parameters.values():
            raise ValueError("onnx_input_parameter cannot be a model parameter")
    artifact_inputs = mapping.get("artifact_inputs", {})
    if type(artifact_inputs) is not dict or any(
        type(k) is not str or not k or type(v) is not str or not v
        for k, v in artifact_inputs.items()
    ):
        raise ValueError("artifact_inputs must map input aliases to nonempty artifact formats")
    artifact_only = mapping.get("artifact_only", False)
    if type(artifact_only) is not bool:
        raise ValueError("artifact_only must be a boolean")
    if artifact_only and not artifact_inputs:
        raise ValueError("artifact_only requires declared artifact_inputs")
    if artifact_only and (model_parameters or onnx_input_parameter):
        raise ValueError("artifact_only cannot use model_parameters or onnx_input_parameter")
    if not metrics and not artifacts:
        raise ValueError("Evaluation Protocol requires at least one metric or artifact")
    result: EvaluationProtocol = {
        "schema": "mjtensu.mldb-v2/evaluation-protocol/v1",
        "id": EvaluationProtocolId(definition_id),
        "status": status,  # type: ignore[typeddict-item]
        "task": TaskId(task),
        "name": name,
        "description": description,
        "implementation": implementation,  # type: ignore[typeddict-item]
        "parameters": parameters,
        "metrics": metrics,
        "artifacts": artifacts,
    }
    if model_parameters:
        result["model_parameters"] = model_parameters
    if onnx_input_parameter is not None:
        result["onnx_input_parameter"] = onnx_input_parameter
    if artifact_inputs:
        result["artifact_inputs"] = artifact_inputs
    if artifact_only:
        result["artifact_only"] = True
    return result


def _load_evaluation_protocol_definition(
    mldb_data_root: str | Path, protocol_id: EvaluationProtocolId | str
) -> EvaluationProtocol:
    expected_id = _validate_definition_id(protocol_id)
    document, _path = _resolve_document(
        mldb_data_root,
        kind=EntityKind.EVALUATION_PROTOCOL,
        entity_id=expected_id,
    )
    return _parse_evaluation_protocol_document(document, expected_id=expected_id)
