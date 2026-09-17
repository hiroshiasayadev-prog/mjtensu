# MLDB-V2-TASK-011-03: Rework Study progression for backend Pipeline execution

- **status**: completed
- **date**: 2026-09-17
- **work_item**: MLDB-V2-WORK-011
- **task_type**: implementation
- **depends_on**: [MLDB-V2-TASK-011-02]
- **outputs**: semantic-gate/canonical-reconciliation progression over backend-owned child scheduling

## Goal

Remove MLDB's duplicate physical stage scheduler while preserving all semantic and canonical guarantees already enforced by W006.

## Work

- Keep Plan/readiness logic as the authority for when a logical child is semantically releasable.
- Let the backend Pipeline own physical Task creation/queueing after release.
- Preserve exact StageInput construction, source pinning, and runtime Model lineage.
- Collect terminal child candidates and run the existing Training/Evaluation result acceptance before opening dependent evaluation gates.
- Preserve child-before-parent canonical persistence and deterministic identities.
- Keep evaluation siblings independent after their common semantic predecessor is accepted.
- Make `run`/`resume` reconcile the same backend Study execution rather than create stage Tasks directly.

## Verification

Focused tests must prove that backend Task completion alone cannot release dependent Evaluation, no duplicate child is launched by repeated reconciliation, interruption/resume recovers the same Pipeline, and existing accepted Result/Model shapes stay unchanged.

## Completion evidence

Study progression now creates/recovers the backend Study execution before any child-stage observation/release. ClearML stores that recovered Pipeline identity and binds each semantically released child Task to the exact Pipeline node before enqueueing it; replays recover the same Pipeline/Task ownership without duplicate creation.

The existing MLDB readiness boundary remains authoritative: a completed backend training Task alone does not release Evaluation. Formal TrainingResult acceptance plus canonical Model lineage must succeed first. MLDB releases logical work through the backend port but does not create ClearML Tasks or choose queues/workers itself.

Verification:

- Pipeline/admission/Study-driver focused set: **70 passed**
- W011 + W006/Application integration join: **102 passed**
- explicit Pipeline ensure-before-admit ordering and child parent/`pipe:` binding: PASS
