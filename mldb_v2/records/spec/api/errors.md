# Contract: Application error model

- **id**: `spec:mldb.v2.api.errors`
- **status**: draft
- **date**: 2026-09-09
- **parent**: `spec:mldb.v2.api`
- **contract_class**: `interface`

## Stable categories

Public operations distinguish at least:

| code | meaning |
|---|---|
| `not_found` | Requested typed canonical object does not exist. |
| `invalid_request` | Request shape, selector, kind, filter, or identity is invalid. |
| `validation_failed` | Canonical/domain validation prevents the requested operation. |
| `not_sealed` | Formal planning/execution requires a sealed definition. |
| `source_not_pinned` | Required formal inputs do not match the selected Git commit. |
| `lifecycle_conflict` | Requested mutation conflicts with immutable/terminal state. |
| `backend_unavailable` | Required backend operation cannot currently be reached/performed. |
| `unsupported_capability` | Selected backend does not expose an optional requested capability. |
| `result_rejected` | Backend candidate failed formal MLDB result acceptance. |
| `internal_failure` | Unexpected application failure outside normal domain rejection. |

Validation/verification reports may return ordered per-target issues without raising one exceptional
application failure; bulk command exit status is adapter presentation policy defined by CLI spec.

## Public error value

Transport adapters preserve one backend-neutral application error value:

```yaml
code: not_found
message: <bounded human-readable summary>
```

`code` is exactly one of the stable categories above. `message` is explanatory text and is not parsed
for identity or lifecycle state. Stack traces, backend logs, credentials, and arbitrary exception
objects are not public error fields. Python/CLI/HTTP adapters may represent this value with their
native exception/exit/status mechanism without changing its semantic fields.
