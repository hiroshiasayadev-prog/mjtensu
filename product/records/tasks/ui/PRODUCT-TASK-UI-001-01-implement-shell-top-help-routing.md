# PRODUCT-TASK-UI-001-01: Implement shell Top Help and routing

- **status**: not_started
- **date**: 2026-08-26
- **work_item**: PRODUCT-WORK-UI-001
- **task_type**: implementation
- **estimate**: 1d
- **depends_on**:
  - PRODUCT-TASK-SYSTEM-001-05
- **outputs**:
  - production UI shell/router implementation
  - PRODUCT-TASK-UI-001-01

## Goal

Implement the production application shell, Top and Help pages, route table/guards, and baseline history behavior required by the accepted screen flow.

## Work

- Implement routes for Top, Recognition, Conditions, Result, and Help.
- Implement Top primary actions and Help navigation/content shell.
- Guard Conditions and Result when no active scoring session exists and redirect to Top.
- Implement explicit new-recognition entry semantics and baseline route-history helpers needed by later page Tasks.
- Keep route URLs as navigation state rather than scoring-session source of truth.
- Add focused route/component tests with deterministic Application state fixtures.

## Implementation contract

| target | required change | acceptance criterion | verification |
|---|---|---|---|
| route table | Provide the five production routes defined by the architecture/screen-flow contracts. | Each route renders its owning page boundary and no route reconstructs session state from URL data. | Router tests with a memory router/test host. |
| route guards | Redirect Conditions/Result to Top when no active session exists. | Direct guarded-route entry without session cannot fabricate partial scoring state. | Guard tests. |
| Top/Help | Implement Top scoring entry and Help round-trip without implicit session creation/destruction. | Top -> Help -> Top preserves the expected Application state; starting Recognition is explicit. | Testing Library/router tests. |
| history foundation | Provide baseline navigation helpers compatible with later Recognition replace and Result correction routes. | Later pages can express replace/back outcomes without private router hacks. | Focused navigation helper tests. |

## Done condition

The production shell, Top, Help, route table, guards, and baseline navigation behavior satisfy the screen-flow/architecture contracts and pass focused component/router tests.

## Verification

- Run Top/Help component tests.
- Run route-guard tests with session present/absent.
- Run baseline route/history tests.
- Run strict typecheck/lint/architecture checks.

## Evidence

- `spec:product.ui.screen_flow` and `spec:product.system.architecture` define the route/history boundary.
- Execution results are recorded here when the Task is performed.
