# Contract: Backend attempt summary

- **id**: `spec:mldb.v2.results.attempt_summary`
- **status**: draft
- **date**: 2026-09-09
- **parent**: `spec:mldb.v2.results`
- **contract_class**: `value`

## Shape

Every attempt summary has exactly:

```yaml
backend: clearml
execution_id: <non-empty opaque backend id>
status: completed
started_at: 2026-09-09T09:00:00Z
ended_at: 2026-09-09T09:05:00Z
diagnostic: null
```

`backend` is a non-empty registered backend type name. `execution_id` is opaque and MUST NOT be
parsed for MLDB identity. `status` is exactly `completed`, `failed`, or `cancelled`.

`started_at` and `ended_at` are RFC3339 UTC timestamps using `Z`; either may be null only when the
backend cannot provide that timestamp. When both are present, `ended_at >= started_at`.

`diagnostic` is null for completed attempts and otherwise follows `spec:mldb.v2.common.diagnostic`.
Attempt summaries are ordered oldest-to-newest and preserve all terminal backend retries reported for
one logical stage. Backend logs, stack traces, resource telemetry, and heartbeat history are excluded.
