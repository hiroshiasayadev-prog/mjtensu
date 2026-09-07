# Contract: Study Run plan format

- **id**: `spec:mldb.study.plan_format`
- **status**: draft
- **date**: 2026-09-04
- **parent**: `spec:mldb.study`
- **contract_class**: `format`

## What this is

Defines the immutable `plan.jsonl` materialized for one Study Run.

The plan records the fully resolved execution intent for every Study-local trial without storing Queue or Worker state or concrete child Run IDs.

## Current contract

Canonical placement is:

```text
mldb_data/study_runs/<study-run-id>/plan.jsonl
```

The file contains exactly one JSON object per Study-local trial.

Training-derived row example:

```json
{"trial":"trial-0001","training":{"architecture":"plain-cnn-v1","corpus":"gray35-train-v1","protocol":"tile-classifier-standard-v1","seed":42,"parameters":{"epochs":150,"learning_rate":0.001}},"evaluations":[{"stage":"final-holdout","corpus":"gray35-final-holdout-v1","protocol":"tile-classifier-standard-eval-v1","parameters":{"batch_size":4096}}]}
```

Existing-Model row example:

```json
{"trial":"trial-0001","model":"mdl-20260901-001","evaluations":[{"stage":"revised-real-holdout","corpus":"gray35-real-holdout-v2","protocol":"tile-classifier-robustness-eval-v2","parameters":{"batch_size":2048}}]}
```

Every row contains:

| field | contract |
|---|---|
| `trial` | Study Run-local identifier `trial-NNNN`. |
| Model source | Exactly one of `training` or `model`. |
| `evaluations` | One or more fully resolved evaluation stage definitions for the trial Model. |

A `training` object contains:

| field | contract |
|---|---|
| `architecture` | One concrete Architecture ID selected from `model.train`. |
| `corpus` | Training Corpus ID from the Study. |
| `protocol` | Train Protocol ID from the Study. |
| `seed` | One concrete integer seed selected from the Study grid. Boolean is invalid. |
| `parameters` | Complete resolved Train Protocol public-parameter mapping. |

A `model` value is the exact existing Model ID selected from `model.existing`.

Each `evaluations` entry contains:

| field | contract |
|---|---|
| `stage` | Study-local evaluation-stage identifier. |
| `corpus` | Evaluation Corpus ID. |
| `protocol` | Evaluation Protocol ID. |
| `parameters` | Complete resolved Evaluation Protocol public-parameter mapping. |

## Rules

- Every materialized Study Model coordinate appears exactly once.
- Every row has exactly one Study Run-local trial ID.
- Trial IDs begin at `trial-0001` and increase without gaps across the canonical persisted row sequence.
- Trial IDs are stable only within one Study Run.
- Every row contains exactly one of `training` or `model`; both or neither are invalid.
- Training-derived rows preserve the exact fully resolved training coordinate from `model.train`.
- Existing-Model rows preserve the exact Model ID from `model.existing`.
- `training.seed` preserves the exact validated integer seed without coercion.
- `training.parameters` stores defaults plus the concrete Study grid coordinate, not only explicit Study values.
- `training.parameters` and every evaluation `parameters` mapping contain only JSON-compatible public parameter values.
- Every evaluation `parameters` mapping stores the complete resolved protocol mapping.
- Every row contains every Study evaluation stage in Study declaration order.
- Plan rows do not contain Training Run IDs or Evaluation Run IDs.
- Existing-Model rows intentionally contain the selected Model ID because that Model is an immutable Study input.
- Training-derived rows do not contain the future Model ID because the Model does not exist until training completes.
- Plan rows do not contain Queue job IDs, claims, attempts, leases, heartbeats, priorities, or retry state.
- Queue workers must not mutate `plan.jsonl` to record progress.
- Child retries do not add attempt records, replacement Run IDs, or retry counters to `plan.jsonl`.
- Once complete plan materialization succeeds and its integrity metadata is recorded, the plan is immutable.

For a training-derived row, evaluation work remains intended even before its Model exists. If training never produces a Model, those evaluation intents may remain blocked operationally without synthetic Evaluation Run records.

For an existing-Model row, the Model dependency is already satisfied at plan materialization time.

### Ordering

Persisted row order must match `spec:mldb.study.grid_expansion`.

| Study Model source | required row order |
|---|---|
| `model.train` | Canonical Cartesian-product coordinate order. |
| `model.existing` | Authored existing-Model list order. |

### Serialization

The complete file must be serialized deterministically for one materialized ordered plan so that its recorded SHA-256 protects exact persisted bytes.

Candidate: standardize UTF-8, one compact JSON object per line, LF line endings, and a fixed object-key ordering before implementation depends on byte-identical regeneration across runtime versions.

The candidate serialization details are not yet normative.

## Validation rules

- Reject blank or non-JSON non-empty lines.
- Reject a row missing `trial` or `evaluations`.
- Reject a row with both `training` and `model`.
- Reject a row with neither `training` nor `model`.
- Reject an empty `evaluations` list.
- Reject duplicate trial IDs.
- Reject a persisted row sequence that does not match the selected Study Model-source ordering.
- Reject trial IDs that do not correspond to the persisted row sequence.
- Reject duplicate materialized Model coordinates within one plan.
- Reject a training-derived row not produced by the referenced `model.train` grid.
- Reject an existing-Model row whose `model` is not selected by the referenced `model.existing` list.
- Reject a `training.seed` that is not an integer or is boolean.
- Reject a training parameter mapping that is not fully resolved against the selected Train Protocol.
- Reject a training parameter mapping containing a value outside the JSON-compatible public-parameter domain.
- Reject an evaluation parameter mapping that is not fully resolved against its selected Evaluation Protocol.
- Reject an evaluation parameter mapping containing a value outside the JSON-compatible public-parameter domain.
- Reject an evaluation stage not declared by the referenced Study.
- Reject Queue state or concrete child Run identity as part of the v1 plan contract.
- Reject persisted plan bytes whose SHA-256 or byte size disagrees with Study Run `run.yaml`.

## Boundary

| concern | owner |
|---|---|
| Study-authored Model source and stages | `spec:mldb.study.study_format`. |
| Model-source expansion and trial ordering | `spec:mldb.study.grid_expansion`. |
| Plan SHA-256, byte size, and row counts | `spec:mldb.study.study_run_format`. |
| Concrete Training Run child identity | Training Run lineage. |
| Existing Model identity and loading | `spec:mldb.model`. |
| Concrete Evaluation Run child identity | Evaluation Run lineage. |
| Operational blocked/queued/running state | `spec:mldb.orchestration`. |

## Related specs

| ref | relation |
|---|---|
| `spec:mldb.study` | Parent Study overview. |
| `spec:mldb.study.grid_expansion` | Produces the rows represented by this format. |
| `spec:mldb.runtime.public_parameters` | Resolves complete parameter mappings stored in each row. |
| `spec:mldb.model` | Defines existing Model identity used directly by existing-Model rows. |
| `spec:mldb.orchestration` | Consumes immutable plan intent without writing Queue state or retry attempts back into the plan. |
