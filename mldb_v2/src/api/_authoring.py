"""Authoring/planning application composition over W001/W002/W003 boundaries."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

from mldb_v2.src.common.ids import (
    DefinitionKind,
    EntityKind,
    NamespaceId,
    StudyId,
    _validate_namespace_id,
    _validate_typed_reference,
)
from mldb_v2.src.repository.listing import CanonicalRepositoryListing
from mldb_v2.src.source.git_snapshot import _current_repository_commit
from mldb_v2.src.storage.object_bytes import _ObjectByteAccess
from mldb_v2.src.study._plan_build import _build_study_plan, _create_study_plan
from mldb_v2.src.study._planning_preflight import _StudyPlanningPreflight
from mldb_v2.src.study.plan import StudyPlan
from mldb_v2.src.verification._definition_lifecycle import (
    _RepositoryDefinitionValidator,
    _RepositoryDefinitionVerifier,
)
from mldb_v2.src.verification._definition_sealing import _RepositoryDefinitionSealer
from mldb_v2.src.verification.definition_lifecycle import (
    DefinitionSealer,
    DefinitionValidator,
    DefinitionVerifier,
)

from ._errors import (
    _ApplicationBoundaryError,
    _application_error,
    _raise_application_error,
    _translate_application_error,
)


_DEFINITION_TO_ENTITY = {
    DefinitionKind.TASK: EntityKind.TASK,
    DefinitionKind.CORPUS: EntityKind.CORPUS,
    DefinitionKind.ARCHITECTURE: EntityKind.ARCHITECTURE,
    DefinitionKind.TRAIN_PROTOCOL: EntityKind.TRAIN_PROTOCOL,
    DefinitionKind.EVALUATION_PROTOCOL: EntityKind.EVALUATION_PROTOCOL,
    DefinitionKind.STUDY: EntityKind.STUDY,
}
_SCHEMA_TO_DEFINITION = {
    "mjtensu.mldb-v2/task/v1": DefinitionKind.TASK,
    "mjtensu.mldb-v2/corpus/v1": DefinitionKind.CORPUS,
    "mjtensu.mldb-v2/architecture/v1": DefinitionKind.ARCHITECTURE,
    "mjtensu.mldb-v2/train-protocol/v1": DefinitionKind.TRAIN_PROTOCOL,
    "mjtensu.mldb-v2/evaluation-protocol/v1": DefinitionKind.EVALUATION_PROTOCOL,
    "mjtensu.mldb-v2/study/v1": DefinitionKind.STUDY,
}


def _invalid_request(message: str) -> _ApplicationBoundaryError:
    return _ApplicationBoundaryError(_application_error("invalid_request", message))


def _definition_kind(value: object) -> DefinitionKind:
    if isinstance(value, DefinitionKind):
        return value
    if type(value) is str:
        try:
            return DefinitionKind(value)
        except ValueError as error:
            raise _invalid_request("definition kind is invalid") from error
    raise _invalid_request("definition kind is invalid")


def _parse_scope(scope: Mapping[str, object] | None) -> tuple[DefinitionKind | None, NamespaceId | None, str | None]:
    if scope is None:
        return None, None, None
    if not isinstance(scope, Mapping) or not set(scope).issubset({"kind", "namespace", "id"}):
        raise _invalid_request("definition scope fields do not match contract")
    kind = _definition_kind(scope["kind"]) if "kind" in scope else None
    try:
        namespace = _validate_namespace_id(scope["namespace"]) if "namespace" in scope else None
    except ValueError as error:
        raise _invalid_request("definition namespace is invalid") from error
    entity_id = None
    if "id" in scope:
        if kind is None:
            raise _invalid_request("exact definition id requires explicit kind")
        try:
            entity_id = _validate_typed_reference(scope["id"])
        except ValueError as error:
            raise _invalid_request("definition id is invalid") from error
        entity_namespace = entity_id.split("/", 1)[0]
        if namespace is not None and str(namespace) != entity_namespace:
            raise _invalid_request("definition id conflicts with namespace selector")
        namespace = NamespaceId(entity_namespace)
    return kind, namespace, entity_id


def _failure_diagnostic(code: str, message: str) -> dict[str, str]:
    return {"code": code, "message": message}


def _result_shape(value: object, *, failure_code: str, failure_message: str) -> tuple[bool, list[dict[str, str]]]:
    if type(value) is not dict or set(value) != {"valid", "diagnostics"}:
        return False, [_failure_diagnostic(failure_code, failure_message)]
    valid = value.get("valid") is True
    diagnostics = value.get("diagnostics")
    if type(diagnostics) is not list:
        return False, [_failure_diagnostic(failure_code, failure_message)]
    bounded: list[dict[str, str]] = []
    for diagnostic in diagnostics:
        if type(diagnostic) is dict and set(diagnostic) == {"code", "message"} and type(diagnostic["code"]) is str and type(diagnostic["message"]) is str:
            bounded.append({"code": diagnostic["code"], "message": diagnostic["message"]})
    if len(bounded) != len(diagnostics):
        return False, [_failure_diagnostic(failure_code, failure_message)]
    return valid, bounded


class AuthoringPlanningService:
    """Thin Application-side composition for definition checks, sealing, and Study planning."""

    def __init__(
        self,
        *,
        repository_root: str | Path,
        mldb_data_root: str | Path,
        mldb_tests_root: str | Path,
        object_access: _ObjectByteAccess | None = None,
        validator: DefinitionValidator | None = None,
        verifier: DefinitionVerifier | None = None,
        sealer: DefinitionSealer | None = None,
    ) -> None:
        self._repository_root = Path(repository_root)
        self._root = Path(mldb_data_root)
        self._listing = CanonicalRepositoryListing(self._root)
        self._validator = validator or _RepositoryDefinitionValidator(self._root)
        self._verifier = verifier or _RepositoryDefinitionVerifier(
            repository_root=self._repository_root,
            mldb_data_root=self._root,
            mldb_tests_root=mldb_tests_root,
            object_access=object_access,
        )
        self._sealer = sealer or _RepositoryDefinitionSealer(
            repository_root=self._repository_root,
            mldb_data_root=self._root,
            mldb_tests_root=mldb_tests_root,
            definition_verifier=self._verifier,
        )

    def _targets(self, scope: Mapping[str, object] | None):
        kind, namespace, exact_id = _parse_scope(scope)
        try:
            if kind is not None:
                listing = self._listing.list_entities(
                    kind=_DEFINITION_TO_ENTITY[kind], namespace=namespace
                )
                targets = [(kind, item) for item in listing["items"]]
            else:
                listing = self._listing.list_entities(namespace=namespace)
                targets = []
                for item in listing["items"]:
                    item_kind = _SCHEMA_TO_DEFINITION.get(item.get("schema"))
                    if item_kind is not None:
                        targets.append((item_kind, item))
        except (OSError, UnicodeError, ValueError) as error:
            _raise_application_error(error)
        if exact_id is not None:
            targets = [(target_kind, item) for target_kind, item in targets if item.get("id") == exact_id]
            if not targets:
                _raise_application_error(FileNotFoundError(exact_id))
        return targets, [dict(issue) for issue in listing["issues"]]

    def validate_scope(self, *, scope: Mapping[str, object] | None = None):
        targets, repository_issues = self._targets(scope)
        items = []
        for kind, document in targets:
            entity_id = str(document["id"])
            try:
                result = self._validator.validate(request={"kind": kind, "id": entity_id})
                valid, diagnostics = _result_shape(
                    result,
                    failure_code="definition_validation_failed",
                    failure_message="definition validation returned an invalid result",
                )
            except Exception:
                valid, diagnostics = False, [_failure_diagnostic(
                    "definition_validation_failed", "definition validation failed"
                )]
            items.append({"kind": kind, "id": entity_id, "valid": valid, "diagnostics": diagnostics})
        return {"items": items, "repository_issues": repository_issues}

    def verify_scope(self, *, scope: Mapping[str, object] | None = None):
        targets, repository_issues = self._targets(scope)
        items = []
        for kind, document in targets:
            entity_id = str(document["id"])
            try:
                result = self._verifier.verify(request={"kind": kind, "id": entity_id})
                valid, diagnostics = _result_shape(
                    result,
                    failure_code="definition_verification_failed",
                    failure_message="definition verification returned an invalid result",
                )
            except Exception:
                valid, diagnostics = False, [_failure_diagnostic(
                    "definition_verification_failed", "definition verification failed"
                )]
            items.append({"kind": kind, "id": entity_id, "valid": valid, "diagnostics": diagnostics})
        return {"items": items, "repository_issues": repository_issues}

    def seal_scope(self, *, scope: Mapping[str, object], bulk: bool = False):
        if not scope:
            raise _invalid_request("seal_scope requires an explicit target selector")
        targets, _issues = self._targets(scope)
        if len(targets) > 1 and not bulk:
            raise _invalid_request("multiple seal targets require explicit bulk intent")
        results = []
        for kind, document in targets:
            entity_id = str(document["id"])
            try:
                self._sealer.seal(request={"kind": kind, "id": entity_id})
            except Exception as error:
                if not bulk:
                    _raise_application_error(error)
                mapped = _translate_application_error(error)
                results.append({"kind": kind, "id": entity_id, "sealed": False, "diagnostics": [mapped]})
                continue
            results.append({"kind": kind, "id": entity_id, "sealed": True, "diagnostics": []})
        return results

    def plan_study(self, *, study: StudyId) -> StudyPlan:
        try:
            study_id = StudyId(_validate_typed_reference(study))
        except ValueError as error:
            raise _invalid_request("Study id is invalid") from error
        try:
            planning = _StudyPlanningPreflight(
                mldb_data_root=self._root,
                verifier=self._verifier,
            ).prepare(study_id)
            commit = _current_repository_commit(self._repository_root)
            plan = _build_study_plan(
                repository_root=self._repository_root,
                mldb_data_root=self._root,
                planning=planning,
                selected_commit=commit,
            )
            return _create_study_plan(repository_root=self._repository_root, plan=plan)
        except Exception as error:
            _raise_application_error(error)

