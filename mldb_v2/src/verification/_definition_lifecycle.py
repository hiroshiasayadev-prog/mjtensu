"""Private repository-backed definition validation and verification composition."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from mldb_v2.src.catalog.architecture import Architecture, _load_architecture_definition
from mldb_v2.src.catalog.architecture_build import _load_architecture_build
from mldb_v2.src.catalog.corpus import Corpus, _load_corpus
from mldb_v2.src.catalog.task import Task, _load_task
from mldb_v2.src.common.diagnostic import Diagnostic, _validate_diagnostic
from mldb_v2.src.common.ids import (
    ArchitectureId,
    CorpusId,
    DefinitionKind,
    EntityKind,
    EvaluationProtocolId,
    NamespaceId,
    StudyId,
    TaskId,
    TrainProtocolId,
    _validate_typed_reference,
)
from mldb_v2.src.common.parameters import _validate_value_against_declaration
from mldb_v2.src.evaluation.evaluate_interface import _load_evaluation_callable
from mldb_v2.src.evaluation.evaluation_protocol import (
    EvaluationProtocol,
    _load_evaluation_protocol_definition,
)
from mldb_v2.src.repository.listing import CanonicalRepositoryListing
from mldb_v2.src.repository.resolution import CanonicalRepositoryResolver
from mldb_v2.src.storage.corpus_manifest import _corpus_manifest_sha256, _parse_corpus_manifest
from mldb_v2.src.storage.object_bytes import _ObjectByteAccess
from mldb_v2.src.study._study_validation import _load_study_definition
from mldb_v2.src.study.study import Study
from mldb_v2.src.training.train_interface import _load_train_callable
from mldb_v2.src.training.train_protocol import TrainProtocol, _load_train_protocol_definition
from mldb_v2.src.verification._executable_asset_tests import (
    _RepositoryExecutableAssetTestVerifier,
    _default_pytest_runner,
)
from mldb_v2.src.verification._executable_integrity import _RepositoryExecutableIntegrityVerifier

from .definition_lifecycle import (
    DefinitionValidationRequest,
    DefinitionValidationResult,
    DefinitionVerificationRequest,
    DefinitionVerificationResult,
)


DefinitionValue = Task | Corpus | Architecture | TrainProtocol | EvaluationProtocol | Study

_KIND_TO_ENTITY = {
    DefinitionKind.TASK: EntityKind.TASK,
    DefinitionKind.CORPUS: EntityKind.CORPUS,
    DefinitionKind.ARCHITECTURE: EntityKind.ARCHITECTURE,
    DefinitionKind.TRAIN_PROTOCOL: EntityKind.TRAIN_PROTOCOL,
    DefinitionKind.EVALUATION_PROTOCOL: EntityKind.EVALUATION_PROTOCOL,
    DefinitionKind.STUDY: EntityKind.STUDY,
}
_EXECUTABLE_KINDS = {
    DefinitionKind.ARCHITECTURE,
    DefinitionKind.TRAIN_PROTOCOL,
    DefinitionKind.EVALUATION_PROTOCOL,
}


def _diagnostic(code: str, message: str) -> Diagnostic:
    return {"code": code, "message": message}


def _parse_request(request: object) -> tuple[DefinitionKind, str] | None:
    if type(request) is not dict or set(request) != {"kind", "id"}:
        return None
    raw_kind = request["kind"]
    if isinstance(raw_kind, DefinitionKind):
        kind = raw_kind
    elif type(raw_kind) is str:
        try:
            kind = DefinitionKind(raw_kind)
        except ValueError:
            return None
    else:
        return None
    entity_id = request["id"]
    if type(entity_id) is not str:
        return None
    return kind, entity_id


def _safe_diagnostics(value: object) -> list[Diagnostic]:
    if type(value) is not list:
        return []
    result: list[Diagnostic] = []
    for item in value:
        try:
            validated = _validate_diagnostic(item)
        except ValueError:
            continue
        if validated is not None:
            result.append(validated)
    return result


class _RepositoryDefinitionValidator:
    """Local-only reusable-definition validator over exact canonical resolution."""

    def __init__(self, mldb_data_root: str | Path) -> None:
        self._root = Path(mldb_data_root)
        self._resolver = CanonicalRepositoryResolver(self._root)

    def _load(self, kind: DefinitionKind, entity_id: str) -> DefinitionValue:
        if kind is DefinitionKind.TASK:
            return _load_task(self._resolver, TaskId(entity_id))
        if kind is DefinitionKind.CORPUS:
            return _load_corpus(self._resolver, CorpusId(entity_id))
        if kind is DefinitionKind.ARCHITECTURE:
            return _load_architecture_definition(self._root, ArchitectureId(entity_id))
        if kind is DefinitionKind.TRAIN_PROTOCOL:
            return _load_train_protocol_definition(self._root, TrainProtocolId(entity_id))
        if kind is DefinitionKind.EVALUATION_PROTOCOL:
            return _load_evaluation_protocol_definition(self._root, EvaluationProtocolId(entity_id))
        if kind is DefinitionKind.STUDY:
            return _load_study_definition(self._root, StudyId(entity_id))
        raise ValueError("unsupported definition kind")

    def validate(self, *, request: DefinitionValidationRequest) -> DefinitionValidationResult:
        parsed = _parse_request(request)
        if parsed is None:
            return {
                "valid": False,
                "diagnostics": [_diagnostic("definition_kind_invalid", "definition validation request is invalid")],
            }
        kind, entity_id = parsed
        try:
            self._load(kind, entity_id)
        except FileNotFoundError:
            return {
                "valid": False,
                "diagnostics": [_diagnostic("definition_not_found", "canonical definition was not found")],
            }
        except (OSError, UnicodeError, ValueError, TypeError):
            return {
                "valid": False,
                "diagnostics": [_diagnostic("definition_invalid", "canonical definition is invalid")],
            }
        return {"valid": True, "diagnostics": []}


@dataclass(frozen=True)
class _CorpusVerificationEvidence:
    manifest_sha256: str
    manifest_entries: int


@dataclass(frozen=True)
class _ModelLineage:
    model: str
    training_result: str
    task: str
    architecture: str


class _RepositoryDefinitionVerifier:
    """Read-only seal-gate verifier composed from W001 and T002-01..04 boundaries."""

    def __init__(
        self,
        *,
        repository_root: str | Path,
        mldb_data_root: str | Path,
        mldb_tests_root: str | Path,
        object_access: _ObjectByteAccess | None = None,
        integrity_verifier: object | None = None,
        asset_test_verifier: object | None = None,
        pytest_runner: object | None = None,
        architecture_interface_loader: Callable[[str | Path, str], object] = _load_architecture_build,
        train_interface_loader: Callable[[str | Path, str], object] = _load_train_callable,
        evaluation_interface_loader: Callable[[str | Path, str], object] = _load_evaluation_callable,
    ) -> None:
        self._repository_root = Path(repository_root)
        self._root = Path(mldb_data_root)
        self._tests_root = Path(mldb_tests_root)
        self._resolver = CanonicalRepositoryResolver(self._root)
        self._listing = CanonicalRepositoryListing(self._root)
        self._validator = _RepositoryDefinitionValidator(self._root)
        self._object_access = object_access
        self._integrity = integrity_verifier or _RepositoryExecutableIntegrityVerifier(
            self._repository_root, self._root
        )
        self._asset = asset_test_verifier or _RepositoryExecutableAssetTestVerifier(
            mldb_tests_root=self._tests_root,
            integrity_verifier=self._integrity,  # type: ignore[arg-type]
        )
        self._runner = pytest_runner or _default_pytest_runner()
        self._architecture_interface_loader = architecture_interface_loader
        self._train_interface_loader = train_interface_loader
        self._evaluation_interface_loader = evaluation_interface_loader

    def verify(self, *, request: DefinitionVerificationRequest) -> DefinitionVerificationResult:
        parsed = _parse_request(request)
        if parsed is None:
            return {
                "valid": False,
                "diagnostics": [_diagnostic("definition_kind_invalid", "definition verification request is invalid")],
            }
        kind, entity_id = parsed
        validation = self._validator.validate(request={"kind": kind, "id": entity_id})
        if not validation["valid"]:
            return {"valid": False, "diagnostics": list(validation["diagnostics"])}
        try:
            definition = self._validator._load(kind, entity_id)
        except Exception:
            return {
                "valid": False,
                "diagnostics": [_diagnostic("definition_invalid", "canonical definition became unavailable during verification")],
            }
        diagnostics = self._verify_loaded(kind, entity_id, definition)
        return {"valid": not diagnostics, "diagnostics": diagnostics}

    def _structure_diagnostics(self, kind: DefinitionKind, entity_id: str) -> list[Diagnostic]:
        namespace = entity_id.split("/", 1)[0]
        try:
            listing = self._listing.list_entities(
                kind=_KIND_TO_ENTITY[kind], namespace=NamespaceId(namespace)
            )
        except (OSError, UnicodeError, ValueError):
            return [_diagnostic("repository_structure_invalid", "canonical repository structure could not be inspected")]
        return [
            _diagnostic("repository_structure_invalid", issue["message"])
            for issue in listing["issues"]
        ]

    def _load_sealed_reference(
        self, kind: DefinitionKind, entity_id: str
    ) -> tuple[DefinitionValue | None, list[Diagnostic]]:
        try:
            definition = self._validator._load(kind, entity_id)
        except FileNotFoundError:
            return None, [_diagnostic("referenced_definition_missing", "referenced canonical definition was not found")]
        except (OSError, UnicodeError, ValueError, TypeError):
            return None, [_diagnostic("referenced_definition_invalid", "referenced canonical definition is invalid")]
        if definition.get("status") != "sealed":
            return None, [_diagnostic("referenced_definition_not_sealed", "referenced definition is not sealed")]
        return definition, []

    def _verify_sealed_reference(
        self, kind: DefinitionKind, entity_id: str
    ) -> tuple[DefinitionValue | None, list[Diagnostic]]:
        definition, diagnostics = self._load_sealed_reference(kind, entity_id)
        if diagnostics or definition is None:
            return None, diagnostics
        nested = self._verify_loaded(kind, entity_id, definition)
        if nested:
            return None, nested
        return definition, []

    def _verify_loaded(
        self, kind: DefinitionKind, entity_id: str, definition: DefinitionValue
    ) -> list[Diagnostic]:
        if kind is DefinitionKind.TASK:
            return self._structure_diagnostics(kind, entity_id)
        if kind is DefinitionKind.CORPUS:
            _evidence, diagnostics = self._inspect_corpus(entity_id, definition)  # type: ignore[arg-type]
            return diagnostics
        if kind in _EXECUTABLE_KINDS:
            return self._verify_executable(kind, entity_id, definition)
        if kind is DefinitionKind.STUDY:
            return self._verify_study(entity_id, definition)  # type: ignore[arg-type]
        return [_diagnostic("definition_kind_invalid", "unsupported definition kind")]

    def _verify_task_binding(self, task_id: str) -> tuple[Task | None, list[Diagnostic]]:
        definition, diagnostics = self._verify_sealed_reference(DefinitionKind.TASK, task_id)
        if diagnostics or definition is None:
            return None, diagnostics
        return definition, []  # type: ignore[return-value]

    def _inspect_corpus(
        self, entity_id: str, corpus: Corpus
    ) -> tuple[_CorpusVerificationEvidence | None, list[Diagnostic]]:
        _task, diagnostics = self._verify_task_binding(str(corpus["task"]))
        if diagnostics:
            return None, diagnostics
        namespace, local_id = entity_id.split("/", 1)
        manifest_path = self._root / namespace / "corpora" / f"{local_id}.manifest.jsonl"
        try:
            manifest_bytes = manifest_path.read_bytes()
            entries = _parse_corpus_manifest(manifest_bytes)
            manifest_sha256 = _corpus_manifest_sha256(manifest_bytes)
        except (OSError, UnicodeError, ValueError):
            return None, [_diagnostic("corpus_manifest_invalid", "Corpus manifest bytes are missing or invalid")]

        recorded_manifest = corpus["manifest"]
        if "sha256" in recorded_manifest and recorded_manifest["sha256"] != manifest_sha256:
            return None, [_diagnostic("corpus_manifest_invalid", "Corpus manifest SHA-256 does not match the definition")]
        if "entries" in recorded_manifest and recorded_manifest["entries"] != len(entries):
            return None, [_diagnostic("corpus_manifest_invalid", "Corpus manifest entry count does not match the definition")]

        structure = self._structure_diagnostics(DefinitionKind.CORPUS, entity_id)
        if structure:
            return None, structure

        if entries and self._object_access is None:
            return None, [_diagnostic("corpus_object_invalid", "Corpus object-byte access is not configured")]
        root_uri = corpus["storage"]["root_uri"]
        prefix = root_uri if root_uri.endswith("/") else root_uri + "/"
        for entry in entries:
            ref = {
                "uri": prefix + entry["path"],
                "bytes": entry["bytes"],
                "sha256": entry["sha256"],
            }
            try:
                assert self._object_access is not None
                self._object_access.read_verified(ref)
            except Exception:
                return None, [_diagnostic("corpus_object_invalid", "a Corpus manifest object failed exact byte verification")]

        try:
            builder_result = self._integrity.verify_corpus_builder(request={"corpus": entity_id})
        except Exception:
            return None, [_diagnostic("corpus_builder_invalid", "Corpus builder integrity could not be established")]
        if type(builder_result) is not dict or builder_result.get("valid") is not True:
            nested = _safe_diagnostics(builder_result.get("diagnostics") if type(builder_result) is dict else None)
            return None, [
                _diagnostic("corpus_builder_invalid", "Corpus builder integrity verification failed"),
                *nested,
            ]
        return _CorpusVerificationEvidence(manifest_sha256, len(entries)), []

    def _corpus_sealing_evidence(self, *, corpus_id: str) -> _CorpusVerificationEvidence:
        try:
            corpus = self._validator._load(DefinitionKind.CORPUS, corpus_id)
        except Exception as error:
            raise ValueError("Corpus verification failed") from error
        evidence, diagnostics = self._inspect_corpus(corpus_id, corpus)  # type: ignore[arg-type]
        if diagnostics or evidence is None:
            raise ValueError("Corpus verification failed")
        return evidence

    def _verify_executable(
        self, kind: DefinitionKind, entity_id: str, definition: DefinitionValue
    ) -> list[Diagnostic]:
        task_id = str(definition["task"])  # type: ignore[typeddict-item]
        _task, diagnostics = self._verify_task_binding(task_id)
        if diagnostics:
            return diagnostics
        structure = self._structure_diagnostics(kind, entity_id)
        if structure:
            return structure

        request_kind = kind.value
        try:
            integrity_result = self._integrity.verify_executable_definition(
                request={"kind": request_kind, "id": entity_id}
            )
        except Exception:
            return [_diagnostic("executable_integrity_failed", "executable integrity could not be established")]
        if type(integrity_result) is not dict or integrity_result.get("valid") is not True:
            nested = _safe_diagnostics(integrity_result.get("diagnostics") if type(integrity_result) is dict else None)
            return [
                _diagnostic("executable_integrity_failed", "executable integrity verification failed"),
                *nested,
            ]

        try:
            if kind is DefinitionKind.ARCHITECTURE:
                self._architecture_interface_loader(self._root, entity_id)
            elif kind is DefinitionKind.TRAIN_PROTOCOL:
                self._train_interface_loader(self._root, entity_id)
            else:
                self._evaluation_interface_loader(self._root, entity_id)
        except Exception:
            return [_diagnostic("executable_interface_failed", "executable callable interface verification failed")]

        try:
            asset_result = self._asset.verify(
                request={"kind": request_kind, "id": entity_id}, runner=self._runner
            )
        except Exception:
            return [_diagnostic("executable_asset_tests_failed", "executable asset-test verification could not be established")]
        if type(asset_result) is not dict or asset_result.get("valid") is not True:
            nested = _safe_diagnostics(asset_result.get("diagnostics") if type(asset_result) is dict else None)
            return [
                _diagnostic("executable_asset_tests_failed", "executable asset-test verification failed"),
                *nested,
            ]
        return []

    def _validate_axes(self, axes: object, declarations: object) -> list[Diagnostic]:
        assert type(axes) is dict and type(declarations) is dict
        for key, axis in axes.items():
            if key not in declarations:
                return [_diagnostic("parameter_key_unknown", "Study parameter axis uses an unpublished protocol key")]
            assert type(axis) is dict and type(axis["values"]) is list
            for value in axis["values"]:
                try:
                    _validate_value_against_declaration(value, declarations[key])
                except (TypeError, ValueError):
                    return [_diagnostic("parameter_value_invalid", "Study parameter axis value violates its protocol declaration")]
        return []

    def _load_model_lineage(self, model_id: str) -> tuple[_ModelLineage | None, list[Diagnostic]]:
        try:
            _validate_typed_reference(model_id)
            model = dict(self._resolver.resolve(kind=EntityKind.MODEL, entity_id=model_id))
        except FileNotFoundError:
            return None, [_diagnostic("model_lineage_invalid", "selected Model was not found")]
        except (OSError, UnicodeError, ValueError, TypeError):
            return None, [_diagnostic("model_lineage_invalid", "selected Model is invalid")]
        if set(model) != {"schema", "id", "training_result"} or model.get("schema") != "mjtensu.mldb-v2/model/v1":
            return None, [_diagnostic("model_lineage_invalid", "selected Model has invalid v1 shape")]
        try:
            training_result_id = _validate_typed_reference(model["training_result"])
        except ValueError:
            return None, [_diagnostic("model_lineage_invalid", "selected Model training_result reference is invalid")]
        try:
            training_result = dict(
                self._resolver.resolve(
                    kind=EntityKind.TRAINING_RESULT,
                    entity_id=training_result_id,
                )
            )
        except FileNotFoundError:
            return None, [_diagnostic("model_lineage_invalid", "selected Model Training Result was not found")]
        except (OSError, UnicodeError, ValueError, TypeError):
            return None, [_diagnostic("model_lineage_invalid", "selected Model Training Result is invalid")]
        if training_result.get("schema") != "mjtensu.mldb-v2/training-result/v1":
            return None, [_diagnostic("model_lineage_invalid", "selected Model Training Result schema is invalid")]
        if training_result.get("status") != "completed":
            return None, [_diagnostic("training_result_not_completed", "selected Model Training Result is not completed")]
        result = training_result.get("result")
        if type(result) is not dict or result.get("model") != model_id:
            return None, [_diagnostic("model_lineage_invalid", "selected Model Training Result success payload is invalid")]
        try:
            task_id = _validate_typed_reference(training_result.get("task"))
            architecture_id = _validate_typed_reference(training_result.get("architecture"))
        except ValueError:
            return None, [_diagnostic("model_lineage_invalid", "selected Model Training Result definition lineage is invalid")]
        return _ModelLineage(model_id, training_result_id, task_id, architecture_id), []

    def _study_reference(
        self,
        cache: dict[tuple[DefinitionKind, str], DefinitionValue],
        kind: DefinitionKind,
        entity_id: str,
    ) -> tuple[DefinitionValue | None, list[Diagnostic]]:
        key = (kind, entity_id)
        if key in cache:
            return cache[key], []
        definition, diagnostics = self._verify_sealed_reference(kind, entity_id)
        if diagnostics or definition is None:
            return None, diagnostics
        cache[key] = definition
        return definition, []

    def _verify_study(self, entity_id: str, study: Study) -> list[Diagnostic]:
        structure = self._structure_diagnostics(DefinitionKind.STUDY, entity_id)
        if structure:
            return structure
        cache: dict[tuple[DefinitionKind, str], DefinitionValue] = {}
        model = study["model"]
        model_task: str | None = None

        if "train" in model:
            source = model["train"]
            corpus, diagnostics = self._study_reference(
                cache, DefinitionKind.CORPUS, str(source["corpus"])
            )
            if diagnostics or corpus is None:
                return diagnostics
            protocol, diagnostics = self._study_reference(
                cache, DefinitionKind.TRAIN_PROTOCOL, str(source["protocol"])
            )
            if diagnostics or protocol is None:
                return diagnostics
            architectures: list[Architecture] = []
            for architecture_id in source["architectures"]:
                architecture, diagnostics = self._study_reference(
                    cache, DefinitionKind.ARCHITECTURE, str(architecture_id)
                )
                if diagnostics or architecture is None:
                    return diagnostics
                architectures.append(architecture)  # type: ignore[arg-type]
            model_task = str(corpus["task"])  # type: ignore[typeddict-item]
            if str(protocol["task"]) != model_task or any(str(item["task"]) != model_task for item in architectures):  # type: ignore[typeddict-item]
                return [_diagnostic("definition_task_mismatch", "training Study definitions do not reference one exact Task")]
            _task, diagnostics = self._study_reference(cache, DefinitionKind.TASK, model_task)
            if diagnostics:
                return diagnostics
            diagnostics = self._validate_axes(source["parameters"], protocol["parameters"])  # type: ignore[typeddict-item]
            if diagnostics:
                return diagnostics
        else:
            lineages: list[_ModelLineage] = []
            for model_id in model["existing"]:
                lineage, diagnostics = self._load_model_lineage(str(model_id))
                if diagnostics or lineage is None:
                    return diagnostics
                lineages.append(lineage)
            tasks = {lineage.task for lineage in lineages}
            if len(tasks) != 1:
                return [_diagnostic("definition_task_mismatch", "selected Models do not share one exact Task")]
            model_task = next(iter(tasks))
            _task, diagnostics = self._study_reference(cache, DefinitionKind.TASK, model_task)
            if diagnostics:
                return diagnostics
            for lineage in lineages:
                architecture, diagnostics = self._study_reference(
                    cache, DefinitionKind.ARCHITECTURE, lineage.architecture
                )
                if diagnostics or architecture is None:
                    return diagnostics
                if str(architecture["task"]) != model_task:  # type: ignore[typeddict-item]
                    return [_diagnostic("definition_task_mismatch", "Model lineage Architecture references a different Task")]

        assert model_task is not None
        for stage in study["evaluations"]:
            corpus, diagnostics = self._study_reference(
                cache, DefinitionKind.CORPUS, str(stage["corpus"])
            )
            if diagnostics or corpus is None:
                return diagnostics
            protocol, diagnostics = self._study_reference(
                cache, DefinitionKind.EVALUATION_PROTOCOL, str(stage["protocol"])
            )
            if diagnostics or protocol is None:
                return diagnostics
            if str(corpus["task"]) != model_task or str(protocol["task"]) != model_task:  # type: ignore[typeddict-item]
                return [_diagnostic("definition_task_mismatch", "Evaluation stage definitions reference a different Task")]
            diagnostics = self._validate_axes(stage["parameters"], protocol["parameters"])  # type: ignore[typeddict-item]
            if diagnostics:
                return diagnostics
        return []
