# PRODUCT-TASK-UI-001-11: Verify UI review corrections

- **status**: not_started
- **date**: 2026-08-27
- **work_item**: PRODUCT-WORK-UI-001
- **task_type**: verification
- **estimate**: 0.5d
- **depends_on**:
  - PRODUCT-TASK-UI-001-08
  - PRODUCT-TASK-UI-001-09
  - PRODUCT-TASK-UI-001-10
- **finding_refs**:
  - PRODUCT-TASK-UI-001-07/F-MAJ-01
  - PRODUCT-TASK-UI-001-07/F-MAJ-02
  - PRODUCT-TASK-UI-001-07/F-MAJ-03
  - PRODUCT-TASK-UI-001-07/F-MIN-01
- **outputs**:
  - objective verification evidence for U07 corrections
  - PRODUCT-TASK-UI-001-11

## Goal

Verify that U08 through U10 resolve every U07 finding without regressing the accepted production scoring-flow behavior or architecture boundary.

## Work

- Re-run focused component and navigation tests affected by the correction Tasks.
- Re-run the deterministic fake-service browser scoring flow with fakes implementing the same public feature contracts used by production.
- Add and execute the missing Result -> Conditions edit -> cancel acceptance path and prove the prior Result/session remains unchanged.
- Prove accepted Result-origin correction recalculates and replaces Result while post-confirmed Recognition correction cannot restore stale score state.
- Prove Recognition UI composes against public Camera/Recognition types without a UI-private semantic service contract.
- Prove missing scoring-flow composition cannot fabricate valid/no-yaku/correction-readiness semantics.
- Run strict typecheck, lint/architecture, unit/component tests, production build, and affected Playwright flow.
- Record expected and observed results plus one overall verdict.

## Done condition

Every U07 finding has direct objective verification evidence and the complete affected production UI gate is green.

## Verification

| check | expected result | observed result |
|---|---|---|
| Result condition edit -> cancel preserves prior Result/session | PASS | not run |
| Result condition edit -> calculate replaces Result | PASS | not run |
| confirmed Recognition correction cannot restore stale Result | PASS | not run |
| Result dealer/child shortcut focuses seat wind | PASS | not run |
| Recognition UI/test fakes use public Camera/Recognition contracts | PASS | not run |
| production-compatible Recognition composition typechecks | PASS | not run |
| missing scoring-flow composition exposes no fabricated semantics | PASS | not run |
| focused UI/navigation/component tests | PASS | not run |
| fake-service Playwright scoring flow | PASS | not run |
| strict typecheck/lint/architecture/unit/build gate | PASS | not run |

## Evidence

- U07 recorded three major and one minor unresolved findings after U06 PASS.
- U11 is the objective correction gate for U08 through U10 and feeds the subsequent independent re-review.
