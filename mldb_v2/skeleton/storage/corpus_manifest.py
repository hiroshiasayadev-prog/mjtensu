"""Canonical MLDB v2 Corpus manifest entry shape."""

from typing import NotRequired, TypedDict


class CorpusManifestEntry(TypedDict):
    path: str
    bytes: int
    sha256: str
    split: NotRequired[str]
