# Contract: Study format

- **id**: `spec:mldb.v2.study.study_format`
- **status**: draft
- **date**: 2026-09-09
- **parent**: `spec:mldb.v2.study`
- **contract_class**: `format`

## Meaning

Study is one reusable declarative experiment intent. It chooses model sources and one or more
Evaluation stages; deterministic materialization belongs to the Study Plan.

## YAML

Schema is `mjtensu.mldb-v2/study/v1`.

Required top-level fields are `schema`, full versioned `id`, `status`, non-empty `name`,
`description`, `model`, and non-empty `evaluations`. `status` is `draft` or `sealed`.

`model` contains exactly one of `train` or `existing`; both or neither are invalid.

## Training model source

`model.train` contains exactly:

| field | contract |
|---|---|
| `corpus` | One typed training Corpus ref. |
| `protocol` | One typed Train Protocol ref. |
| `architectures` | Non-empty authored-order list of unique Architecture refs. |
| `parameters` | Mapping from published Train Protocol keys to axis objects. |
| `seeds` | Non-empty authored-order list of unique integer seeds; boolean invalid. |

Each training parameter axis is exactly `{values: [...]}` with a non-empty list of unique
JSON-compatible values. Omitted protocol keys use protocol defaults and do not create axes.
## Existing-Model source

`model.existing` is a non-empty authored-order list of unique canonical Model refs. Every selected
Model must resolve to completed Training Result lineage and the same Task. Existing-Model Studies
create no Training Result and no new Model.

## Evaluation stages

Every `evaluations` item contains exactly:

| field | contract |
|---|---|
| `stage` | Unique lowercase kebab-case Study-local stage identifier. |
| `corpus` | One typed evaluation Corpus ref. |
| `protocol` | One typed Evaluation Protocol ref. |
| `parameters` | Mapping from published Evaluation Protocol keys to axis objects. |

Each Evaluation parameter axis is exactly `{values: [...]}` with a non-empty list of unique
JSON-compatible values. A fixed condition uses a singleton list. Omitted protocol keys use defaults.
Evaluation axes create coordinates beneath each model trial and never create retraining trials.

Every Study has at least one Evaluation stage; training-only and existing-Model no-op Studies are
invalid in schema v1.

## Compatibility

Before sealing/planning, the model Task is resolved and every selected Corpus/Architecture/Protocol
must reference that same Task. Referenced reusable definitions required for formal execution are
sealed and pass integrity rules.
Unknown parameter keys, duplicate decoded axis values under type-sensitive equality, empty axes,
duplicate Architecture/Model/seed values, invalid seed types, or incompatible Task relationships are
invalid.

## Identity and lifecycle

Local ID ends in `-v<positive-integer>`. A sealed Study is immutable; changing model source, grid
axes/values/order, seeds, Evaluation stages/order/axes, or referenced definitions requires a new
Study revision.

Study YAML contains no backend queue, worker, retry, attempt, backend Task ID, or live-progress
fields. Dynamic backend-generated HPO trials are outside Study schema v1.

Formal compilation order is defined only by `spec:mldb.v2.study.grid_expansion`.
Unknown top-level/model-source/evaluation-stage keys are invalid in schema v1.
