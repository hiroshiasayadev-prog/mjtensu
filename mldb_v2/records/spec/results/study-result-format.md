# Contract: Study Result format

- **id**: `spec:mldb.v2.results.study_result_format`
- **status**: draft
- **date**: 2026-09-09
- **parent**: `spec:mldb.v2.results`
- **contract_class**: `format`

## Identity

Schema is `mjtensu.mldb-v2/study-result/v1`. `execution_key` is exactly 32 lowercase hexadecimal
characters encoding one UUID4 without hyphens. `start_study` persists
`<study-namespace>/run-<execution_key>` before backend work is admitted.

Reusing the same execution key with the same Plan and backend is idempotent. Reusing it with a
different Plan or backend is a lifecycle conflict. Every intentional new execution uses a fresh key.

Generated children always live in the Study Result namespace:

- Training Result: `<namespace>/run-<key>-<trial>-train`;
- Model: `<namespace>/run-<key>-<trial>-model`;
- Evaluation Result: `<namespace>/run-<key>-<trial>-<eval-coordinate>`.

Existing-Model trials create no Training Result or new Model.

## Top-level shape
```yaml
schema: mjtensu.mldb-v2/study-result/v1
id: rotated-fcos/run-abcd1234abcd1234abcd1234abcd1234
execution_key: abcd1234abcd1234abcd1234abcd1234
plan: rotated-fcos/spatial-screen-v2-plan-0123456789abcdef
study: rotated-fcos/spatial-screen-v2
source_commit: <full unabbreviated Git commit object id>
backend: clearml
created_at: 2026-09-09T09:00:00Z
status: submitted
diagnostic: null
trials: []
```

`created_at` is RFC3339 UTC using `Z`; it is allocated once with the initial Study Result and never
changes. `backend` is the generic backend type name only; endpoint, credentials, queue, worker, and
backend Task IDs are not Study Result fields.

Status is `submitted` or `cancelling` while non-terminal. Terminal status is exactly `completed`,
`completed_with_failures`, `failed`, or `cancelled`.

## Trial tree

The `trials` sequence exactly mirrors Plan trial order. Training-source trial:
```yaml
- trial: trial-0001
  training:
    disposition: pending
    result: null
    reason: null
  evaluations:
    - coordinate: eval-0001
      stage: final-holdout
      disposition: pending
      result: null
      reason: null
```

Existing-Model trial uses `training: null` and still contains every planned Evaluation coordinate.

A stage slot has exactly `disposition`, `result`, and `reason`; Evaluation slots additionally carry
`coordinate` and `stage`. Allowed combinations are:

- `pending`: `result=null`, `reason=null`;
- `completed|failed|cancelled`: `result` is the exact deterministic child Result ref, `reason=null`;
- `skipped`: `result=null`, `reason` is one stable code.

Initial stable skip reasons are `upstream_failed`, `upstream_cancelled`, `study_cancelled`, and
`global_failure`. Unknown reason codes are invalid in schema v1.
## Closure and mutation

Terminal closure requires every planned stage disposition to be non-`pending`.

- all planned stages `completed` -> Study `completed`;
- `cancelling` closes as `cancelled` after every planned stage is terminal;
- unrecoverable global progression/setup failure closes remaining pending stages as
  `skipped: global_failure` and Study as `failed`;
- otherwise, any terminal non-completed child/skip -> Study `completed_with_failures`.

A transient `backend_unavailable` error does not itself terminalize the Study. `diagnostic` is null
for ordinary `submitted`, `completed`, and `completed_with_failures`; `failed` requires a non-null
`spec:mldb.v2.common.diagnostic`. Cancellation may retain a nullable explanatory diagnostic.

Until terminal closure, only `status`, `diagnostic`, and stage-slot `disposition/result/reason` may
change. `schema`, identity, Plan/Study/source/backend/created_at, trial order, coordinates, and stage
names are immutable. Terminal Study Result is fully immutable.
