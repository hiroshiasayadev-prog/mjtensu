# PRODUCT-TASK-SCORING-001-08: Route Agari build-management authoring

- **status**: done
- **date**: 2026-08-26
- **work_item**: PRODUCT-WORK-SCORING-001
- **task_type**: coordination
- **estimate**: 0.25d
- **depends_on**:
  - PRODUCT-TASK-SCORING-001-01
- **outputs**:
  - PRODUCT-TASK-SCORING-001-08
  - PRODUCT-TASK-SCORING-001-09
  - PRODUCT-TASK-SCORING-001-02
  - PRODUCT-WORK-SCORING-001

## Goal

Route the accepted Agari source/artifact-management decision through one canonical authoring step before fork implementation begins.

## Work

- Added PRODUCT-TASK-SCORING-001-09 as the bounded authoring task for the accepted decision.
- Routed PRODUCT-TASK-SCORING-001-02 through that authoring task so fork implementation starts from the canonical ADR/Specification projection rather than only from decision-task evidence.
- Updated the parent Work Item task list, task flow, and task-candidate dependency view.

## Done condition

The authoring task exists, the parent Work Item registers it, and Agari core implementation depends on completion of that authoring task.

## Verification

- PRODUCT-TASK-SCORING-001-09 belongs to PRODUCT-WORK-SCORING-001 and depends on the completed decision plus this coordination task.
- PRODUCT-TASK-SCORING-001-02 depends on PRODUCT-TASK-SCORING-001-09.
- The Work Item task list and human-readable flow reflect the same route.

## Evidence

- PRODUCT-TASK-SCORING-001-01 is `done` with ADR routing `amend PRODUCT-ADR-SYSTEM-003` and normative target `spec:product.system.contracts.agari_fork`.
- PRODUCT-TASK-SCORING-001-09 is materialized as the authoring owner.
- No scoring-engine implementation is performed by this coordination task.
