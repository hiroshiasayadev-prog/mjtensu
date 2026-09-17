# Overview: MLDB v2 application API

- **id**: `spec:mldb.v2.api`
- **status**: draft
- **date**: 2026-09-17
- **parent**: `spec:mldb.v2`

## What this is

Defines one transport-independent application boundary used by the normal CLI, Python callers, and
any future HTTP/UI adapter.

The API owns MLDB workflow and discovery semantics. Adapters MUST NOT duplicate validation,
planning, semantic readiness/gating, result acceptance, canonical-history, or query/filter rules.
Operational Task scheduling/retry/liveness belongs to the selected execution backend.

## Golden execution path

```text
run Study
  -> validate/plan
  -> create Study Result
  -> create/recover backend Study execution
  -> backend schedules semantically-released child work
  -> MLDB collects/accepts terminal candidates
  -> accepted canonical predecessors release downstream semantic gates
  -> close every planned stage
  -> terminal Study Result
```

The workflow is resumable from canonical Study Result + Plan state plus recoverable backend execution
identity; no local in-memory workflow state is authoritative.
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
