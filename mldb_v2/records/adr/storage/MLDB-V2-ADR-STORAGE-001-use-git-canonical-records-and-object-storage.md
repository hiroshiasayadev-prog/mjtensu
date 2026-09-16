# MLDB-V2-ADR-STORAGE-001: Use Git for canonical records and object storage for large bytes

- **status**: accepted
- **date**: 2026-09-09
- **depends_on**:
  - MLDB-V2-ADR-ARCHITECTURE-001
- **supersedes**:
- **migrated_to_spec**: `spec:mldb.v2.storage`

## Context

Experiment definitions and formal scalar results are small and valuable enough to preserve in Git.
Corpora, learned weights, checkpoints, predictions, images, and other large artifacts are not
appropriate Git payloads. A self-hosted experiment database or server SSD must not be the sole copy
of either the experiment contract or the experiment result.

## Decision

Git-managed `mldb_data/` records are the canonical source for:

- namespaces and reusable definitions;
- immutable Study Plans;
- formal Training, Model, Evaluation, and Study records;
- failed and cancelled formal execution outcomes;
- artifact URIs, sizes, hashes, and lineage;
- Corpus manifests and their digests.

Large immutable bytes live in S3-compatible object storage.

Every canonical large-object reference MUST carry at least a URI, byte size, and SHA-256 digest.
Corpus identity is fixed to the materialized sample bytes by a canonical manifest whose own
SHA-256 is recorded in Corpus YAML.

The storage contract is S3-compatible and backend-neutral. MinIO is the initial deployment, not a
schema identity.

ClearML may mirror or cache the same artifacts, but ClearML storage identifiers are not canonical
MLDB identities.

## Rationale

A Git clone plus object-store backup is sufficient to reconstruct the formal experiment history
and repopulate a new visualization/execution backend. Large bytes remain efficient to transfer and
store.

## Rejected alternatives

### Store all result state only in ClearML

This makes backend database durability a requirement for reproducibility.

### Commit corpora and weights to Git

Large binary histories make normal repository operations impractical.

### Use object paths without content hashes

Mutable object keys would allow old records to silently refer to new bytes.

## Consequences

Artifact collection is not complete until content integrity is verified. Deleting or overwriting a
referenced object is corruption even if a backend still has metadata for it.
