# Overview: MLDB v2 repository

- **id**: `spec:mldb.v2.repository`
- **status**: draft
- **date**: 2026-09-09
- **parent**: `spec:mldb.v2`

## What this is

Defines physical placement, typed lookup, deterministic inventory, and canonical writes for v2.

## Roots

| root | responsibility |
|---|---|
| `mldb_v2/records/` | Brewprint Design Records. |
| `mldb_v2/skeleton/` | Frozen public implementation shape after specs are reviewed. |
| `mldb_v2/src/` | MLDB v2 implementation. |
| `mldb_v2/tests/` | MLDB v2 implementation tests. |
| `mldb_tests/` | Namespace-first executable-asset contract tests used by sealing. |
| `mldb_data/` | Canonical ML definitions, plans, and formal execution history. |
| external S3-compatible storage | Large immutable bytes referenced by canonical records. |

## Topics
| ref | responsibility |
|---|---|
| `spec:mldb.v2.repository.layout` | Canonical roots, namespace-first paths, and sibling basename rules. |
| `spec:mldb.v2.repository.resolution` | Typed deterministic lookup and no-fallback resolution. |
| `spec:mldb.v2.repository.listing` | Deterministic broad inventory and structural diagnostics. |
| `spec:mldb.v2.repository.canonical_writes` | Crash-safe/idempotent canonical record writes, immutable validation ownership, and reconciliation. |
| `spec:mldb.v2.repository.mutation_coordination` | Short per-Study Result mutation serialization for concurrent progression callers. |
