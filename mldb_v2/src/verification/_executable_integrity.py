"""Private repository-backed executable and Corpus-builder integrity verification."""

from __future__ import annotations

import hashlib
import os
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from mldb_v2.src.catalog.architecture import _load_architecture_definition
from mldb_v2.src.catalog.corpus import _load_corpus
from mldb_v2.src.common.diagnostic import Diagnostic
from mldb_v2.src.common.ids import CorpusId
from mldb_v2.src.evaluation.evaluation_protocol import _load_evaluation_protocol_definition
from mldb_v2.src.repository.resolution import CanonicalRepositoryResolver
from mldb_v2.src.storage._paths import _validate_safe_relative_path
from mldb_v2.src.training.train_protocol import _load_train_protocol_definition
from mldb_v2.src.verification.executable_integrity import (
    CorpusBuilderIntegrityRequest,
    ExecutableDefinitionIntegrityRequest,
    ExecutableIntegrityResult,
)
from mldb_v2.src.verification._source_imports import _scan_project_source_imports


_EXECUTABLE_DOMAIN = {
    "architecture": "architectures",
    "train_protocol": "train_protocols",
    "evaluation_protocol": "evaluation_protocols",
}
_EXECUTABLE_LOADERS: dict[str, Callable[[str | Path, str], dict[str, object]]] = {
    "architecture": _load_architecture_definition,
    "train_protocol": _load_train_protocol_definition,
    "evaluation_protocol": _load_evaluation_protocol_definition,
}


@dataclass(frozen=True)
class _ExecutableSealingEvidence:
    companion_sha256: str
    sources: tuple[tuple[str, str], ...]


@dataclass(frozen=True)
class _CorpusBuilderSealingEvidence:
    builder_sha256: str | None


def _diagnostic(code: str, message: str) -> Diagnostic:
    return {"code": code, "message": message}


def _resolved_directory(value: str | Path, *, label: str) -> Path:
    try:
        path = Path(value).resolve(strict=True)
    except OSError as error:
        raise ValueError(f"{label} must exist") from error
    if not path.is_dir():
        raise ValueError(f"{label} must be a directory")
    return path


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
def _hash_regular_beneath(
    root: Path,
    *,
    relative: str,
    missing_code: str,
    invalid_code: str,
    unreadable_code: str,
    label: str,
) -> tuple[str | None, Diagnostic | None]:
    target = root.joinpath(*relative.split("/"))
    try:
        resolved = target.resolve(strict=True)
    except FileNotFoundError:
        return None, _diagnostic(missing_code, f"{label} is missing")
    except OSError:
        return None, _diagnostic(unreadable_code, f"{label} cannot be resolved")
    if not resolved.is_relative_to(root):
        return None, _diagnostic(invalid_code, f"{label} escapes its configured root")
    try:
        mode = resolved.stat().st_mode
    except OSError:
        return None, _diagnostic(unreadable_code, f"{label} cannot be inspected")
    if not stat.S_ISREG(mode):
        return None, _diagnostic(invalid_code, f"{label} must be a regular file")
    try:
        return _sha256_file(resolved), None
    except OSError:
        return None, _diagnostic(unreadable_code, f"{label} cannot be read")


def _executable_relative(kind: str, entity_id: str) -> str:
    namespace, local_id = entity_id.split("/", 1)
    return f"{namespace}/{_EXECUTABLE_DOMAIN[kind]}/{local_id}.py"


def _builder_relative(corpus_id: str) -> str:
    namespace, local_id = corpus_id.split("/", 1)
    return f"{namespace}/corpora/{local_id}.py"
class _RepositoryExecutableIntegrityVerifier:
    """Read-only integrity verifier over one repository and canonical mldb_data tree."""

    def __init__(self, repository_root: str | Path, mldb_data_root: str | Path) -> None:
        self._repository_root = _resolved_directory(repository_root, label="repository root")
        self._mldb_data_root = _resolved_directory(mldb_data_root, label="mldb_data root")

    def _load_executable(
        self, request: ExecutableDefinitionIntegrityRequest
    ) -> tuple[dict[str, object] | None, str | None, str | None, list[Diagnostic]]:
        try:
            kind = request["kind"]
            entity_id = request["id"]
        except (KeyError, TypeError):
            return None, None, None, [_diagnostic("definition_invalid", "integrity request is invalid")]
        if type(kind) is not str or kind not in _EXECUTABLE_LOADERS or type(entity_id) is not str:
            return None, None, None, [_diagnostic("definition_invalid", "integrity request kind or id is invalid")]
        loader = _EXECUTABLE_LOADERS[kind]
        try:
            definition = loader(self._mldb_data_root, entity_id)
        except FileNotFoundError:
            return None, kind, entity_id, [_diagnostic("definition_not_found", "canonical executable definition was not found")]
        except (OSError, UnicodeError, ValueError):
            return None, kind, entity_id, [_diagnostic("definition_invalid", "canonical executable definition is invalid")]
        return definition, kind, entity_id, []

    def _inspect_executable(
        self, request: ExecutableDefinitionIntegrityRequest
    ) -> tuple[_ExecutableSealingEvidence | None, list[Diagnostic]]:
        definition, kind, entity_id, diagnostics = self._load_executable(request)
        if diagnostics:
            return None, diagnostics
        assert definition is not None and kind is not None and entity_id is not None
        implementation = definition["implementation"]
        assert type(implementation) is dict

        companion_relative = _executable_relative(kind, entity_id)
        companion_sha256, issue = _hash_regular_beneath(
            self._mldb_data_root,
            relative=companion_relative,
            missing_code="companion_missing",
            invalid_code="companion_invalid",
            unreadable_code="companion_unreadable",
            label="same-basename executable companion",
        )
        if issue is not None:
            diagnostics.append(issue)
        recorded_companion = implementation.get("sha256")
        if (
            issue is None
            and recorded_companion is not None
            and companion_sha256 != recorded_companion
        ):
            diagnostics.append(
                _diagnostic(
                    "companion_hash_mismatch",
                    "same-basename executable companion SHA-256 does not match the definition",
                )
            )

        source_evidence: list[tuple[str, str]] = []
        raw_sources = implementation.get("sources", [])
        assert type(raw_sources) is list
        for source in raw_sources:
            assert type(source) is dict
            source_path = source["path"]
            declared_sha256 = source["sha256"]
            assert type(source_path) is str and type(declared_sha256) is str
            source_evidence.append((source_path, declared_sha256))
            try:
                relative = _validate_safe_relative_path(
                    source_path, label="repository source path"
                )
            except ValueError:
                diagnostics.append(
                    _diagnostic("source_invalid", "declared project source path is invalid")
                )
                continue
            actual_sha256, source_issue = _hash_regular_beneath(
                self._repository_root,
                relative=relative,
                missing_code="source_missing",
                invalid_code="source_invalid",
                unreadable_code="source_unreadable",
                label=f"declared project source {relative}",
            )
            if source_issue is not None:
                diagnostics.append(source_issue)
                continue
            if actual_sha256 != declared_sha256:
                diagnostics.append(
                    _diagnostic(
                        "source_hash_mismatch",
                        f"declared project source {relative} SHA-256 does not match the definition",
                    )
                )

        if issue is None:
            companion_path = self._mldb_data_root.joinpath(*companion_relative.split("/"))
            try:
                import_scan = _scan_project_source_imports(
                    self._repository_root,
                    companion_path,
                )
            except (OSError, ValueError):
                diagnostics.append(
                    _diagnostic(
                        "source_import_scan_failed",
                        "project-owned executable imports could not be inspected",
                    )
                )
            else:
                declared_paths = {path for path, _sha256 in source_evidence}
                for forbidden_path in import_scan.forbidden_paths:
                    diagnostics.append(
                        _diagnostic(
                            "source_import_forbidden",
                            f"project-owned executable import is not an allowed source: {forbidden_path}",
                        )
                    )
                for required_path in import_scan.required_paths:
                    if required_path not in declared_paths:
                        diagnostics.append(
                            _diagnostic(
                                "source_declaration_missing",
                                f"project-owned executable source is undeclared: {required_path}",
                            )
                        )

        if diagnostics or companion_sha256 is None:
            return None, diagnostics
        return _ExecutableSealingEvidence(
            companion_sha256=companion_sha256,
            sources=tuple(source_evidence),
        ), []
    def verify_executable_definition(
        self,
        *,
        request: ExecutableDefinitionIntegrityRequest,
    ) -> ExecutableIntegrityResult:
        _evidence, diagnostics = self._inspect_executable(request)
        return {"valid": not diagnostics, "diagnostics": diagnostics}

    def _executable_sealing_evidence(
        self, *, request: ExecutableDefinitionIntegrityRequest
    ) -> _ExecutableSealingEvidence:
        evidence, diagnostics = self._inspect_executable(request)
        if diagnostics or evidence is None:
            raise ValueError("executable integrity verification failed")
        return evidence

    def _load_corpus(
        self, request: CorpusBuilderIntegrityRequest
    ) -> tuple[dict[str, object] | None, str | None, list[Diagnostic]]:
        try:
            corpus_id = request["corpus"]
        except (KeyError, TypeError):
            return None, None, [_diagnostic("definition_invalid", "integrity request is invalid")]
        if type(corpus_id) is not str:
            return None, None, [_diagnostic("definition_invalid", "Corpus id is invalid")]
        resolver = CanonicalRepositoryResolver(self._mldb_data_root)
        try:
            corpus = _load_corpus(resolver, CorpusId(corpus_id))
        except FileNotFoundError:
            return None, corpus_id, [_diagnostic("definition_not_found", "canonical Corpus definition was not found")]
        except (OSError, UnicodeError, ValueError):
            return None, corpus_id, [_diagnostic("definition_invalid", "canonical Corpus definition is invalid")]
        return corpus, corpus_id, []

    def _inspect_corpus_builder(
        self, request: CorpusBuilderIntegrityRequest
    ) -> tuple[_CorpusBuilderSealingEvidence | None, list[Diagnostic]]:
        corpus, corpus_id, diagnostics = self._load_corpus(request)
        if diagnostics:
            return None, diagnostics
        assert corpus is not None and corpus_id is not None
        relative = _builder_relative(corpus_id)
        lexical_builder = self._mldb_data_root.joinpath(*relative.split("/"))
        builder = corpus.get("builder")

        if builder is None:
            if os.path.lexists(lexical_builder):
                diagnostics.append(
                    _diagnostic(
                        "builder_unowned",
                        "same-basename Corpus Python sibling exists without a builder declaration",
                    )
                )
                return None, diagnostics
            return _CorpusBuilderSealingEvidence(builder_sha256=None), []

        assert type(builder) is dict
        builder_sha256, issue = _hash_regular_beneath(
            self._mldb_data_root,
            relative=relative,
            missing_code="builder_missing",
            invalid_code="builder_invalid",
            unreadable_code="builder_unreadable",
            label="same-basename Corpus builder",
        )
        if issue is not None:
            diagnostics.append(issue)
            return None, diagnostics

        recorded_sha256 = builder.get("sha256")
        if recorded_sha256 is not None and builder_sha256 != recorded_sha256:
            diagnostics.append(
                _diagnostic(
                    "builder_hash_mismatch",
                    "same-basename Corpus builder SHA-256 does not match the definition",
                )
            )
            return None, diagnostics
        assert builder_sha256 is not None
        return _CorpusBuilderSealingEvidence(builder_sha256=builder_sha256), []

    def verify_corpus_builder(
        self,
        *,
        request: CorpusBuilderIntegrityRequest,
    ) -> ExecutableIntegrityResult:
        _evidence, diagnostics = self._inspect_corpus_builder(request)
        return {"valid": not diagnostics, "diagnostics": diagnostics}

    def _corpus_builder_sealing_evidence(
        self, *, request: CorpusBuilderIntegrityRequest
    ) -> _CorpusBuilderSealingEvidence:
        evidence, diagnostics = self._inspect_corpus_builder(request)
        if diagnostics or evidence is None:
            raise ValueError("Corpus builder integrity verification failed")
        return evidence
