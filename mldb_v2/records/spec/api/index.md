# Overview: MLDB v2 application API

- **id**: `spec:mldb.v2.api`
- **status**: draft
- **date**: 2026-09-09
- **parent**: `spec:mldb.v2`

## What this is

Defines one transport-independent application boundary used by the normal CLI, Python callers, and
any future HTTP/UI adapter.

The API owns MLDB workflow and discovery semantics. Adapters MUST NOT duplicate validation,
planning, result acceptance, readiness progression, canonical-history, or query/filter rules.

## Golden execution path

```text
run Study
  -> validate/plan
  -> create Study Result
  -> advance ready work
  -> collect/accept terminal outcomes
  -> advance newly-ready work
  -> close every planned stage
  -> terminal Study Result
```

The workflow is resumable from canonical Study Result + Plan state; no hidden in-memory workflow
state is authoritative.
## Topics

| ref | responsibility |
|---|---|
| `spec:mldb.v2.api.application_interface` | Public mutation/check operations and side-effect boundaries. |
| `spec:mldb.v2.api.query_interface` | Read-only discovery, listing, observation, logs, and diagnostics. |
| `spec:mldb.v2.api.study_driver` | Idempotent Study progression and resumable run/resume behavior. |
| `spec:mldb.v2.api.errors` | Stable public failure categories. |

## Interface policy

The normal human entrypoint is an installed `mldb` command. Direct execution of implementation
`.py` files is not part of the supported golden path.

Broad discovery/check operations are first-class API behavior; callers are not expected to discover
IDs by scanning `mldb_data/` themselves.

HTTP is optional. Adding a transport does not add new MLDB semantics.
