# MLDB-V2-TASK-011-01: Amend ClearML Pipeline execution contracts

- **status**: completed
- **date**: 2026-09-17
- **work_item**: MLDB-V2-WORK-011
- **task_type**: specification
- **depends_on**: [MLDB-V2-WORK-010]
- **outputs**: amended backend/Study/API responsibility contracts, Pipeline mapping manual, W011 implementation plan

## Goal

Replace the post-W010 flat ClearML Task projection as the target design with one backend-native Study execution per MLDB Study Result, while preserving MLDB as the authority for experiment semantics and canonical history.

## Decisions

- One MLDB Study Result maps to one recoverable ClearML Pipeline Run/controller.
- Planned training/evaluation work remains backend-neutral MLDB stage semantics and executes as Pipeline child Tasks through the existing CommonExecutionHarness.
- ClearML owns physical child scheduling, queue/worker placement, operational retry/liveness, cancellation, logs, and Pipeline UI.
- MLDB owns Study/Plan semantics, source pinning, semantic release gates, accepted Model lineage, result acceptance, and canonical Training/Evaluation/Study Results.
- A downstream Evaluation may be predeclared in the Pipeline DAG but may not execute until MLDB has canonically accepted the required Training Result and Model.
- ClearML Task/Pipeline success never substitutes for canonical result acceptance.

## Amended sources

Formal/architecture/API/backend contracts were updated under `records/spec/` for the StudyResult-to-backend-Study-execution model, and `docs/CLEARML_PIPELINES.md` was added as the operational design guide. README/OPERATIONS/TELEMETRY/TROUBLESHOOTING now distinguish the W011 target from the still-flat current runtime.

Historical W005/W006/W010 task/work-item evidence was intentionally not rewritten. It remains evidence of the previous implementation state.

## Verification

- Targeted consistency search over active Specs/manuals finds no remaining flat-Task-only statements in the amended execution path.
- `git diff --check -- mldb_v2` passes.
- No runtime source, tests, canonical experiment records, or unrelated dirty files were modified by this task.

**Closure:** T011-01 is complete. Runtime implementation starts at T011-02.
