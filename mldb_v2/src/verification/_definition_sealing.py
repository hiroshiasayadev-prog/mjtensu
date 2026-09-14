"""Private repository-backed reusable-definition sealing lifecycle."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Callable, Mapping, cast

from mldb_v2.src.catalog.architecture import _parse_architecture_document
from mldb_v2.src.catalog.corpus import _validate_corpus
from mldb_v2.src.catalog.task import _validate_task
from mldb_v2.src.catalog._core_definition_validation import _validate_versioned_entity_id
from mldb_v2.src.common.ids import DefinitionKind
from mldb_v2.src.evaluation.evaluation_protocol import _parse_evaluation_protocol_document
from mldb_v2.src.repository._process_lock import _process_file_lock
from mldb_v2.src.repository._yaml import _YamlError, _load_yaml
from mldb_v2.src.repository.canonical_writes import _atomic_replace_record
from mldb_v2.src.study._study_validation import _validate_study
from mldb_v2.src.training.train_protocol import _parse_train_protocol_document
from mldb_v2.src.verification._definition_lifecycle import (
    _RepositoryDefinitionVerifier,
    _parse_request,
)
from mldb_v2.src.verification._executable_integrity import (
    _RepositoryExecutableIntegrityVerifier,
)
from mldb_v2.src.verification.definition_lifecycle import (
    DefinitionSealingRequest,
    DefinitionSealingResult,
    SealableDefinition,
)

_DOMAIN_BY_KIND = {
    DefinitionKind.TASK: "tasks",
    DefinitionKind.CORPUS: "corpora",
    DefinitionKind.ARCHITECTURE: "architectures",
    DefinitionKind.TRAIN_PROTOCOL: "train_protocols",
    DefinitionKind.EVALUATION_PROTOCOL: "evaluation_protocols",
    DefinitionKind.STUDY: "studies",
}
_EXECUTABLE_KINDS = {
    DefinitionKind.ARCHITECTURE,
    DefinitionKind.TRAIN_PROTOCOL,
    DefinitionKind.EVALUATION_PROTOCOL,
}


def _definition_path(root: Path, *, kind: DefinitionKind, entity_id: str) -> Path:
    validated = _validate_versioned_entity_id(entity_id)
    namespace, local_id = validated.split("/", 1)
    return root / namespace / _DOMAIN_BY_KIND[kind] / f"{local_id}.yaml"


def _read_exact_document(path: Path) -> tuple[bytes, dict[str, object]]:
    try:
        payload = path.read_bytes()
    except FileNotFoundError:
        raise
    except OSError as error:
        raise ValueError("canonical definition cannot be read") from error
    try:
        value = _load_yaml(payload.decode("utf-8"))
    except (UnicodeError, _YamlError) as error:
        raise ValueError("canonical definition is invalid") from error
    if type(value) is not dict:
        raise ValueError("canonical definition root must be a mapping")
    return payload, value


def _validate_proposed(
    *, kind: DefinitionKind, entity_id: str, document: object
) -> SealableDefinition:
    if kind is DefinitionKind.TASK:
        return _validate_task(document, expected_id=entity_id)
    if kind is DefinitionKind.CORPUS:
        return _validate_corpus(document, expected_id=entity_id)
    if kind is DefinitionKind.ARCHITECTURE:
        return _parse_architecture_document(document, expected_id=entity_id)
    if kind is DefinitionKind.TRAIN_PROTOCOL:
        return _parse_train_protocol_document(document, expected_id=entity_id)
    if kind is DefinitionKind.EVALUATION_PROTOCOL:
        return _parse_evaluation_protocol_document(document, expected_id=entity_id)
    if kind is DefinitionKind.STUDY:
        return _validate_study(document, expected_id=entity_id)
    raise ValueError("unsupported definition kind")


def _verified_success(value: object) -> bool:
    return (
        type(value) is dict
        and set(value) == {"valid", "diagnostics"}
        and value.get("valid") is True
        and value.get("diagnostics") == []
    )


class _RepositoryDefinitionSealer:
    """Seal one exact verified draft through a short definition-specific lock."""
    def __init__(
        self,
        *,
        repository_root: str | Path,
        mldb_data_root: str | Path,
        mldb_tests_root: str | Path,
        definition_verifier: object | None = None,
        integrity_verifier: object | None = None,
        object_access: object | None = None,
        asset_test_verifier: object | None = None,
        pytest_runner: object | None = None,
        architecture_interface_loader: Callable[[str | Path, str], object] | None = None,
        train_interface_loader: Callable[[str | Path, str], object] | None = None,
        evaluation_interface_loader: Callable[[str | Path, str], object] | None = None,
    ) -> None:
        self._repository_root = Path(repository_root)
        self._root = Path(mldb_data_root)
        self._tests_root = Path(mldb_tests_root)
        self._lock_root = (
            self._repository_root / ".local" / "mldb_v2" / "definition_seal_locks"
        )

        if definition_verifier is None:
            integrity = integrity_verifier or _RepositoryExecutableIntegrityVerifier(
                self._repository_root, self._root
            )
            verifier_kwargs: dict[str, object] = {
                "repository_root": self._repository_root,
                "mldb_data_root": self._root,
                "mldb_tests_root": self._tests_root,
                "object_access": object_access,
                "integrity_verifier": integrity,
            }
            if asset_test_verifier is not None:
                verifier_kwargs["asset_test_verifier"] = asset_test_verifier
            if pytest_runner is not None:
                verifier_kwargs["pytest_runner"] = pytest_runner
            if architecture_interface_loader is not None:
                verifier_kwargs["architecture_interface_loader"] = architecture_interface_loader
            if train_interface_loader is not None:
                verifier_kwargs["train_interface_loader"] = train_interface_loader
            if evaluation_interface_loader is not None:
                verifier_kwargs["evaluation_interface_loader"] = evaluation_interface_loader
            definition_verifier = _RepositoryDefinitionVerifier(**verifier_kwargs)  # type: ignore[arg-type]
            self._integrity = integrity
        else:
            verifier_integrity = getattr(definition_verifier, "_integrity", None)
            if integrity_verifier is not None and verifier_integrity is not None:
                if integrity_verifier is not verifier_integrity:
                    raise ValueError(
                        "definition verifier and sealing integrity verifier must share one instance"
                    )
            self._integrity = integrity_verifier or verifier_integrity
            if self._integrity is None:
                raise TypeError("integrity_verifier is required for injected definition verifier")

        self._verifier = definition_verifier

    def _lock_key(self, path: Path) -> str:
        try:
            return path.resolve().relative_to(self._repository_root.resolve()).as_posix()
        except (OSError, ValueError):
            return path.absolute().as_posix()

    def _verification_authorized(self, *, kind: DefinitionKind, entity_id: str) -> None:
        try:
            result = self._verifier.verify(request={"kind": kind, "id": entity_id})
        except Exception as error:
            raise ValueError("definition verification failed") from error
        if not _verified_success(result):
            raise ValueError("definition verification failed")

    def _executable_sha256(self, *, kind: DefinitionKind, entity_id: str) -> str:
        method = getattr(self._integrity, "_executable_sealing_evidence", None)
        if not callable(method):
            raise ValueError("executable sealing evidence is unavailable")
        try:
            evidence = method(request={"kind": kind.value, "id": entity_id})
            digest = evidence.companion_sha256
        except Exception as error:
            raise ValueError("executable sealing evidence failed") from error
        if type(digest) is not str:
            raise ValueError("executable sealing evidence is invalid")
        return digest

    def _corpus_evidence(self, *, entity_id: str) -> tuple[str, int, str | None]:
        manifest_method = getattr(self._verifier, "_corpus_sealing_evidence", None)
        builder_method = getattr(self._integrity, "_corpus_builder_sealing_evidence", None)
        if not callable(manifest_method) or not callable(builder_method):
            raise ValueError("Corpus sealing evidence is unavailable")
        try:
            manifest = manifest_method(corpus_id=entity_id)
            builder = builder_method(request={"corpus": entity_id})
            manifest_sha256 = manifest.manifest_sha256
            manifest_entries = manifest.manifest_entries
            builder_sha256 = builder.builder_sha256
        except Exception as error:
            raise ValueError("Corpus sealing evidence failed") from error
        if type(manifest_sha256) is not str or type(manifest_entries) is not int:
            raise ValueError("Corpus sealing evidence is invalid")
        if builder_sha256 is not None and type(builder_sha256) is not str:
            raise ValueError("Corpus builder sealing evidence is invalid")
        return manifest_sha256, manifest_entries, builder_sha256

    def _proposed_document(
        self,
        *,
        kind: DefinitionKind,
        entity_id: str,
        draft: Mapping[str, object],
    ) -> dict[str, object]:
        proposed = cast(dict[str, object], copy.deepcopy(dict(draft)))
        proposed["status"] = "sealed"

        if kind in _EXECUTABLE_KINDS:
            implementation = proposed.get("implementation")
            if type(implementation) is not dict:
                raise ValueError("executable implementation is invalid")
            implementation["sha256"] = self._executable_sha256(
                kind=kind, entity_id=entity_id
            )
        elif kind is DefinitionKind.CORPUS:
            manifest_sha256, manifest_entries, builder_sha256 = self._corpus_evidence(
                entity_id=entity_id
            )
            manifest = proposed.get("manifest")
            if type(manifest) is not dict:
                raise ValueError("Corpus manifest is invalid")
            manifest["sha256"] = manifest_sha256
            manifest["entries"] = manifest_entries
            if "builder" in proposed:
                builder = proposed["builder"]
                if type(builder) is not dict or builder_sha256 is None:
                    raise ValueError("Corpus builder sealing evidence is invalid")
                builder["sha256"] = builder_sha256
            elif builder_sha256 is not None:
                raise ValueError("Corpus builder evidence exists without a builder declaration")
        return proposed

    def seal(
        self,
        *,
        request: DefinitionSealingRequest,
    ) -> DefinitionSealingResult:
        parsed = _parse_request(request)
        if parsed is None:
            raise ValueError("definition sealing request is invalid")
        kind, entity_id = parsed
        path = _definition_path(self._root, kind=kind, entity_id=entity_id)

        with _process_file_lock(lock_root=self._lock_root, key=self._lock_key(path)):
            authorized_bytes, current = _read_exact_document(path)
            if current.get("status") != "draft":
                raise ValueError("definition lifecycle conflict: definition is not a draft")

            self._verification_authorized(kind=kind, entity_id=entity_id)
            proposed = self._proposed_document(
                kind=kind,
                entity_id=entity_id,
                draft=current,
            )
            validated = _validate_proposed(
                kind=kind,
                entity_id=entity_id,
                document=proposed,
            )

            try:
                current_bytes = path.read_bytes()
            except OSError as error:
                raise ValueError("canonical definition became unavailable before sealing") from error
            if current_bytes != authorized_bytes:
                raise ValueError("definition lifecycle conflict: authorized draft became stale")

            _atomic_replace_record(path, proposed)
            return {"definition": validated}
