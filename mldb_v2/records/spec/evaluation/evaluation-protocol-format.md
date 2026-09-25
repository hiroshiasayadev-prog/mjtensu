# Contract: Evaluation Protocol format

- **id**: `spec:mldb.v2.evaluation.evaluation_protocol_format`
- **status**: draft
- **date**: 2026-09-18
- **parent**: `spec:mldb.v2.evaluation`
- **contract_class**: `format`

## Meaning

Evaluation Protocol owns reusable post-training evaluation procedure and declares which outputs are
formal MLDB results.

## YAML

Schema is `mjtensu.mldb-v2/evaluation-protocol/v1`.

Required fields are:

| field | contract |
|---|---|
| `schema` | Exact schema identifier. |
| `id` | Full versioned typed identity. |
| `status` | `draft` or `sealed`. |
| `task` | Typed Task reference. |
| `name` | Non-empty human name. |
| `description` | Human-readable evaluation summary. |
| `implementation.entrypoint` | Exactly `evaluate`. |
| `parameters` | Public parameter declarations; empty is valid. |
| `metrics` | Mapping of formal scalar declarations; empty is valid. |
| `artifacts` | Mapping of formal artifact declarations; empty is valid. |

At least one entry across `metrics` and `artifacts` is required.
Each `parameters.<key>` follows `spec:mldb.v2.common.public_parameters`.

Each `metrics.<key>` is exactly a mapping with required `type` (`integer` or `number`) and required
boolean `required`; optional `description` and optional `preference` are allowed. `preference`, when
present, is exactly one of `higher`, `lower`, or `neutral`; omission is semantically equivalent to
`neutral`. `higher` means larger finite values are preferable for human comparison, `lower` means
smaller finite values are preferable, and `neutral` declares that the metric has no generic
better/worse direction. This metadata is comparison semantics only: it MUST NOT change result
acceptance, Study lifecycle, automatic model selection, or optimization behavior. `number` accepts
finite integer or floating values except boolean; `integer` accepts integer except boolean.

Each `artifacts.<key>` is exactly a mapping with required non-empty `format`, required non-empty
versioned `schema`, required boolean `required`, optional `description`, and optional `study_view`.
`study_view`, when present, is exactly one of `hidden`, `select`, or `all`; omission is semantically
equivalent to `hidden`. `hidden` keeps the artifact on the child Evaluation execution only, `select`
asks a Study-level UI to present one artifact surface with a trial/model selector, and `all` asks the
Study-level UI to mirror every available trial/model artifact directly. This field is presentation
semantics only: it MUST NOT alter artifact bytes, result acceptance, Study lifecycle, model selection,
or optimization behavior. A backend MAY omit a requested Study-level rendering when the artifact
format cannot be rendered, but that omission is observational and MUST NOT invalidate a canonical
Evaluation Result.

For `sealed`, `implementation.sha256` is required. `implementation.sources`, when present, declares exact same-namespace `lib/` helpers as defined by
`spec:mldb.v2.verification.executable_integrity`.

## Executable and lifecycle

Same-basename `<local-id>.py` is required and exposes `evaluate` according to
`spec:mldb.v2.evaluation.evaluate_interface`. Local ID ends in `-v<positive-integer>`.

A sealed protocol is immutable. Changing parameter keys/defaults/constraints, metric meaning/type/
requiredness/preference, artifact format/schema/requiredness/`study_view`, or result-affecting executable behavior
requires a new revision.

Backend scalars/plots not declared here remain telemetry only. Missing optional outputs are permitted
without a partial-success state; missing required outputs fail result acceptance.

Reject unsupported schema/entrypoint, malformed declarations, unresolved Task, invalid ID/path/version,
invalid sealed executable integrity, no formal outputs, or unknown top-level keys in schema v1.
