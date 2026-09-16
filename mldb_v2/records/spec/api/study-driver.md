# Contract: Resumable Study driver

- **id**: `spec:mldb.v2.api.study_driver`
- **status**: draft
- **date**: 2026-09-09
- **parent**: `spec:mldb.v2.api`
- **contract_class**: `lifecycle`

## One progression pass

`advance_study(study_result)` is idempotent and derives all action from the immutable Plan,
canonical accepted child results, current Study Result dispositions, and backend observations.

One pass performs, in order:

1. validate Study Result/Plan/source integrity required for progression;
2. observe/recover deterministic backend ownership for pending logical stage keys and identify any
   admitted work that is active or terminal;
3. collect terminal candidates and run formal result acceptance;
4. persist terminal Training/Evaluation Results and Models where accepted;
5. update the corresponding Study Result dispositions;
6. apply Plan-defined upstream skip rules and, when cancelling, use observed deterministic backend
   ownership to close only never-admitted pending stages as `skipped: study_cancelled`;
7. derive newly-ready pending coordinates;
8. idempotently admit those coordinates through the backend port;
9. close the Study Result if every planned coordinate is terminal.

The pass may make no change when backend work is still active.
## Run and resume

`run_study(study_ref, backend)` is the normal synchronous new-execution path:

```text
plan_study -> start_study -> advance/wait loop -> terminal Study Result
```

`resume_study(study_result_ref)` runs the same advance/wait loop for one already-persisted
non-terminal Study Result and creates no new execution identity.

A process interruption does not invalidate the execution. Backend work already admitted may continue;
semantic progression resumes only when another driver caller invokes `resume_study` or
`advance_study`.

`watch` is not a Study-driver operation. Read-only monitoring uses
`spec:mldb.v2.api.query_interface` and MUST NOT be implemented as an alias for progression.

## Concurrency and ownership

Progression MUST tolerate repeated or concurrent caller attempts without duplicate logical stage
admission or duplicate canonical child results. Canonical read/modify/write is coordinated by
`spec:mldb.v2.repository.mutation_coordination`; backend admission uses deterministic ownership
metadata derived from Study Result + trial + stage coordinate.

The driver does not choose GPU, queue priority, retry timing, or worker placement. Those remain
backend concerns.
