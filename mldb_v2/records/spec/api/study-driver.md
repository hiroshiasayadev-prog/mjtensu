# Contract: Resumable Study lifecycle / reconciliation

- **id**: `spec:mldb.v2.api.study_driver`
- **status**: draft
- **date**: 2026-09-17
- **parent**: `spec:mldb.v2.api`
- **contract_class**: `lifecycle`

## Purpose

The Study lifecycle service protects MLDB semantic gates and canonical history around a backend-owned Study execution. It is not a second queue or physical scheduler.

For backends with native Study execution containers, such as ClearML Pipeline, the backend owns physical Task creation/enqueue, queue/worker placement, retry/liveness, and UI grouping after MLDB releases a logical stage. MLDB owns candidate acceptance, canonical lineage, terminal dispositions, and release of semantic gates that require accepted canonical predecessors.

## One reconciliation pass

`advance_study(study_result)` is idempotent and derives action from the immutable Plan, canonical accepted child results, current Study Result dispositions, and backend Study/child observations.

One pass performs, in order:

1. validate Study Result/Plan/source integrity required for progression;
2. idempotently create or recover the backend Study execution for that exact Study Result;
3. observe child-stage state and collect newly terminal candidates;
4. run formal Training/Evaluation result acceptance;
5. persist accepted Training/Evaluation Results and Models, then update Study Result dispositions;
6. derive semantic downstream gates from the reconciled canonical state and notify/release the backend execution accordingly;
7. apply canonical upstream-failure/cancellation skip rules for work that must never execute;
8. close the Study Result when every planned coordinate is canonically terminal.

The pass may make no canonical change while backend work is active. Releasing a semantically ready logical stage through the backend port is allowed; MLDB itself must not create ClearML Tasks, choose queues/workers, or run a duplicate physical scheduler.

## Run and resume

`run_study(study_ref, backend)` remains the normal synchronous new-execution path:

```text
plan_study -> start_study -> ensure backend Study execution -> reconcile/wait -> terminal Study Result
```

`resume_study(study_result_ref)` recovers the same backend Study execution and resumes reconciliation; it creates no new execution identity.

A local process interruption does not cancel or invalidate a backend-owned Pipeline. Backend work may continue independently. Resuming MLDB reconnects to the same Study Result/Pipeline and reconciles any completed child work into canonical state.

`watch` is read-only and MUST NOT perform reconciliation or backend mutation.

## Concurrency and ownership

Progression MUST tolerate repeated/concurrent reconciliation attempts without duplicate backend Study execution, duplicate logical child work, or duplicate canonical child results.

Canonical read/modify/write remains coordinated by `spec:mldb.v2.repository.mutation_coordination`. Backend Study execution identity and child ownership are deterministic/recoverable operational mappings derived from the exact Study Result and Plan.

## Backend responsibility boundary

The lifecycle service does not choose GPU, queue priority, retry timing, worker placement, or child Task launch timing among already-eligible work. Those remain backend concerns.

The lifecycle service does decide whether a backend child result is formally acceptable and whether a downstream stage's semantic prerequisites are satisfied. A backend controller callback/hook may invoke this generic lifecycle boundary to bridge one completed child into canonical acceptance before releasing a dependent stage.
