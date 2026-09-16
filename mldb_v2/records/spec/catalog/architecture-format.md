# Contract: Architecture format

- **id**: `spec:mldb.v2.catalog.architecture_format`
- **status**: draft
- **date**: 2026-09-09
- **parent**: `spec:mldb.v2.catalog`
- **contract_class**: `format`

## Meaning

Architecture owns one versioned unweighted model structure for one Task. Training policy and learned
weights remain outside Architecture.

## YAML

Schema is `mjtensu.mldb-v2/architecture/v1`.

Required fields are:

| field | contract |
|---|---|
| `schema` | Exact schema identifier. |
| `id` | Full versioned typed identity. |
| `status` | `draft` or `sealed`. |
| `task` | Typed Task reference. |
| `name` | Non-empty human name. |
| `family` | Non-empty searchable family identifier. |
| `description` | Human-readable summary. |
| `implementation.framework` | Exactly `pytorch`. |
| `implementation.entrypoint` | Exactly `build`. |
| `interface.input` | JSON-compatible mapping with non-empty `kind`. |
| `interface.output` | JSON-compatible mapping with non-empty `kind`. |
| `structure.summary` | Non-empty human-readable structural summary. |
Optional fields are `structure.traits` (unique non-empty strings) and `parameters` (descriptive
JSON-compatible mapping not supplied as `build()` arguments).

For `sealed`, `implementation.sha256` is required. `implementation.sources`, when needed, follows
`spec:mldb.v2.verification.executable_integrity`.

## Executable companion

Same-basename `<local-id>.py` is required and exposes `build` according to
`spec:mldb.v2.catalog.architecture_build`. The sibling may import declared normal project source but
must not designate a `tools/` script as implementation.

## Identity and validation

Local ID ends in `-v<positive-integer>`. Each Architecture references exactly one sealed Task before
formal planning. A sealed Architecture is immutable; any topology, forward/output meaning,
result-affecting companion/source, or coarse interface change requires a new revision.

Architecture does not own loss, optimizer, scheduler, augmentation, epochs, batch size, seed,
checkpoint selection, pretrained learned state, or evaluation behavior.

Reject unsupported schema/framework/entrypoint, invalid field types or ID/path/version, unresolved
Task, invalid sealed executable integrity, or unknown top-level keys in schema v1.
