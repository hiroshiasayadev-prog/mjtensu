# Index: MLDB verification

- **id**: `spec:mldb.verification`
- **status**: draft
- **date**: 2026-09-04
- **parent**: `spec:mldb`

## What this is

Navigation for MLDB executable-asset verification contracts.

Verification complements runtime validation. It does not replace runtime schema, integrity, compatibility, lifecycle, or result checks.

## Topics

| title | kind | ref | summary |
|---|---|---|---|
| Executable asset verification | Reference | `spec:mldb.verification.executable_asset_tests` | Asset-specific pytest ownership, derived test locations, sealing gate, resolver usage, and runtime dependency boundary. |

## Related specs

| ref | relation |
|---|---|
| `spec:mldb.repository.layout` | Defines canonical `mldb_tests/` placement. |
| `spec:mldb.runtime.asset_resolution` | Defines the shared resolution path used by asset tests and execution tooling. |
| `spec:mldb.catalog.architecture_build` | Architecture executable boundary verified by Architecture asset tests. |
| `spec:mldb.training.train_interface` | Train Protocol executable boundary verified by Train Protocol asset tests. |
| `spec:mldb.evaluation.evaluate_interface` | Evaluation Protocol executable boundary verified by Evaluation Protocol asset tests. |
