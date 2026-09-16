# Reference: Bounded diagnostic value

- **id**: `spec:mldb.v2.common.diagnostic`
- **status**: draft
- **date**: 2026-09-09
- **parent**: `spec:mldb.v2.common`
- **contract_class**: `value`

## Shape

A persisted diagnostic is either null or:

```yaml
code: <stable lowercase snake-case code>
message: <bounded human-readable message>
```

`code` is intended for programmatic classification and matches `[a-z][a-z0-9]*(?:_[a-z0-9]+)*`.
`message` is explanatory text, MUST NOT exceed 4096 bytes when UTF-8 encoded, and MUST NOT be parsed
as identity or lifecycle state.

## Boundary

Diagnostics intentionally exclude stack traces, full backend logs, environment dumps, secrets,
credentials, and unbounded process output. Those remain backend/logging concerns.

A domain-specific contract may restrict allowed codes further but MUST preserve this common shape.
