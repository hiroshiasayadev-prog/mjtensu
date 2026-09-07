"""Catalog-side Corpus metadata implementation for MLDB Wave I1-A."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Mapping

from ..common.errors import ValidationIssue, ValidationReport
from ..common.ids import CorpusId, TaskId


@dataclass(frozen=True, slots=True)
class CorpusArtifact:
    """Identity and integrity metadata for the materialized Corpus artifact."""

    format: Literal["sqlite"]
    sha256: str
    bytes: int | None = None


@dataclass(frozen=True, slots=True)
class CorpusDataSpec:
    """Selector for the concrete physical-data contract inside a Corpus artifact."""

    schema: str
    table: str


@dataclass(frozen=True, slots=True)
class Corpus:
    """Registered immutable Corpus metadata."""

    schema: Literal["mjtensu.mldb/corpus/v1"]
    id: CorpusId
    task: TaskId
    artifact: CorpusArtifact
    data: CorpusDataSpec
    representation: Mapping[str, object]
    builder_parameters: Mapping[str, object]
    splits: Mapping[str, int]
    description: str | None = None
    statistics: object | None = None
    origin: object | None = None


def validate_corpus(corpus: Corpus) -> ValidationReport:
    """Validate static Corpus metadata without resolving or opening runtime assets."""
    issues: list[ValidationIssue] = []

    if corpus.schema != "mjtensu.mldb/corpus/v1":
        issues.append(
            ValidationIssue(
                code="corpus.schema.unsupported",
                message="Corpus schema must be 'mjtensu.mldb/corpus/v1'.",
                path="schema",
            )
        )

    if corpus.artifact.format != "sqlite":
        issues.append(
            ValidationIssue(
                code="corpus.artifact.format.unsupported",
                message="Corpus v1 artifact format must be 'sqlite'.",
                path="artifact.format",
            )
        )

    if not _is_sha256(corpus.artifact.sha256):
        issues.append(
            ValidationIssue(
                code="corpus.artifact.sha256.invalid",
                message="Corpus artifact sha256 must be a 64-character hexadecimal digest.",
                path="artifact.sha256",
            )
        )

    artifact_bytes = corpus.artifact.bytes
    if artifact_bytes is not None and (
        type(artifact_bytes) is not int or artifact_bytes < 0
    ):
        issues.append(
            ValidationIssue(
                code="corpus.artifact.bytes.invalid",
                message="Corpus artifact bytes must be a non-negative integer when present.",
                path="artifact.bytes",
            )
        )

    if type(corpus.data.schema) is not str or corpus.data.schema == "":
        issues.append(
            ValidationIssue(
                code="corpus.data.schema.invalid",
                message="Corpus data schema must be a non-empty string.",
                path="data.schema",
            )
        )

    if type(corpus.data.table) is not str or corpus.data.table == "":
        issues.append(
            ValidationIssue(
                code="corpus.data.table.invalid",
                message="Corpus data table must be a non-empty string.",
                path="data.table",
            )
        )

    for split_name, count in corpus.splits.items():
        if type(split_name) is not str or split_name == "":
            issues.append(
                ValidationIssue(
                    code="corpus.split.name.invalid",
                    message="Corpus split names must be non-empty strings.",
                    path="splits",
                )
            )
        if type(count) is not int or count < 0:
            issues.append(
                ValidationIssue(
                    code="corpus.split.count.invalid",
                    message="Corpus split counts must be non-negative integers.",
                    path=f"splits[{split_name!r}]",
                )
            )

    return ValidationReport(tuple(issues))


def _is_sha256(value: object) -> bool:
    if type(value) is not str or len(value) != 64:
        return False
    return all(character in "0123456789abcdefABCDEF" for character in value)
