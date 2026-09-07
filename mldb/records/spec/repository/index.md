# Overview: MLDB repository

- **id**: `spec:mldb.repository`
- **status**: draft
- **date**: 2026-09-04
- **parent**: `spec:mldb`

## What this is

Defines the repository-level placement boundary for MLDB machine-readable data and executable-asset tests.

This overview owns only physical roots, canonical placement responsibility, and routing to focused repository references. Entity semantics and file contents belong to entity-specific specifications.

## Current contract

MLDB uses two repository-level roots outside the Brewprint Design Records tree.

| root | responsibility |
|---|---|
| `mldb_data/` | Registered MLDB definitions, execution records, and MLDB-owned formal artifacts. |
| `mldb_tests/` | Asset-specific pytest code for executable MLDB definitions. |

`mldb/records/` remains the Brewprint-governed Design Records tree. It is not the storage root for runtime MLDB data.

Canonical entity lookup is derived from entity kind and entity ID using the placement rules in the repository layout reference.

## Non-goals

- Define YAML, SQLite, JSONL, or PyTorch artifact contents.
- Define entity lifecycle or ID grammar beyond its use in canonical paths.
- Define Python runtime package placement.
- Define MLDB runtime implementation tests.
- Define queue database, worker state, cache, or MLflow storage.

## Topics

| title | kind | ref | summary |
|---|---|---|---|
| Repository layout | Reference | `spec:mldb.repository.layout` | Canonical roots, entity-kind directories, file-vs-directory forms, sibling basenames, and executable-asset test locations. |

## Related specs

| ref | relation |
|---|---|
| `spec:mldb` | Parent MLDB specification overview. |
| `spec:mldb.runtime.asset_resolution` | Resolves typed entity references through the canonical repository placement defined here. |
