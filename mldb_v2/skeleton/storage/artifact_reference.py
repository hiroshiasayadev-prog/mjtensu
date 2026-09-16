"""Backend-neutral immutable artifact reference shape."""

from typing import TypedDict


class ArtifactRef(TypedDict):
    uri: str
    bytes: int
    sha256: str
