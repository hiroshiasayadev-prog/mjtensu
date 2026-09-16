# Contract: Train Protocol format

- **id**: `spec:mldb.v2.training.train_protocol_format`
- **status**: draft
- **date**: 2026-09-09
- **parent**: `spec:mldb.v2.training`
- **contract_class**: `format`

## Meaning

Train Protocol defines reusable training procedure for compatible Architectures of one Task. Study
supplies Corpus, Architecture, seed, and resolved public parameter values.

## YAML

Schema is `mjtensu.mldb-v2/train-protocol/v1`.

Required fields are:

| field | contract |
|---|---|
| `schema` | Exact schema identifier. |
| `id` | Full versioned typed identity. |
| `status` | `draft` or `sealed`. |
| `task` | Typed Task reference. |
| `name` | Non-empty human name. |
| `description` | Human-readable procedure summary. |
| `implementation.entrypoint` | Exactly `train`. |
| `parameters` | Mapping of all caller-visible parameter declarations; empty is valid. |

Each `parameters.<key>` follows `spec:mldb.v2.common.public_parameters`. The training seed is a
separate integer execution input and is not duplicated as a universal public parameter.
For `sealed`, `implementation.sha256` is required. `implementation.sources`, when needed, follows
`spec:mldb.v2.verification.executable_integrity`.

## Executable companion

Same-basename `<local-id>.py` is required and exposes `train` according to
`spec:mldb.v2.training.train_interface`. It may import declared normal project source but must not
use a `tools/` script as the reusable protocol implementation.

The returned learned module and standardized canonical weights are defined by the Train interface and
`spec:mldb.v2.training.canonical_weights`; Train Protocol YAML does not declare an alternate learned
output format.

## Identity and validation

Local ID ends in `-v<positive-integer>`. Before formal planning, referenced Task is sealed and public
parameter declarations/defaults are valid. A sealed protocol is immutable; changing public parameter
keys/defaults/constraints or result-affecting executable behavior requires a new revision.

Training may emit arbitrary backend telemetry. Such telemetry is not automatically a formal
Evaluation Result.

Reject unsupported schema/entrypoint, invalid field types/ID/path/version, unresolved Task, malformed
parameter declarations, invalid sealed executable integrity, or unknown top-level keys in schema v1.
