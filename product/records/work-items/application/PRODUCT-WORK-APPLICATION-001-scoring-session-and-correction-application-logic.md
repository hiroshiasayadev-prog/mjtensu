# PRODUCT-WORK-APPLICATION-001: Scoring session and correction application logic

- **status**: not_started
- **date**: 2026-08-26
- **source_refs**:
  - `spec:product.application.scoring_session`
  - `spec:product.system.contracts.application_session_api`
  - `spec:product.system.contracts.scoring_condition_policy`
  - `spec:product.system.contracts.correction_editor_api`
  - `spec:product.system.contracts.testing_strategy`
- **impact_refs**: []
- **tasks**:
  - PRODUCT-TASK-APPLICATION-001-01
  - PRODUCT-TASK-APPLICATION-001-02
  - PRODUCT-TASK-APPLICATION-001-03
  - PRODUCT-TASK-APPLICATION-001-04
  - PRODUCT-TASK-APPLICATION-001-05
  - PRODUCT-TASK-APPLICATION-001-06

## Goal

Implement the page-independent scoring-session, shared condition policy, correction-draft semantics, and Zustand application-state binding required by the production flow.

## Boundary

This Work Item owns Application-layer session creation/mutation/calculation orchestration, shared condition normalization/availability semantics, correction-draft commands/validation/commit behavior, and cross-page application store integration.

It does not own camera/model inference, Agari scoring internals, page rendering, router presentation, or final production service composition.

## Impact Scope

| target | impact |
|---|---|
| production `application` module | Implement the scoring-session service and cross-page state orchestration. |
| condition policy implementation | Provide one shared normalization/availability implementation used by Application and UI. |
| correction implementation | Provide permissive local correction drafts and validated canonical commit conversion. |
| application tests | Add deterministic contract tests using fake ScoringService/Recognition-derived structures. |

## Task flow

```text
SYSTEM T05 bootstrap review PASS
   +-> A01 scoring-session service/state transitions
   +-> A02 shared scoring-condition policy
   +-> A03 correction draft/commands/validation/commit

A01 + A02 + A03 -> A04 Zustand/store binding and application orchestration
A04 -> A05 objective Application contract verification -> A06 independent integrated review
```

A01, A02, and A03 may proceed in parallel after the production bootstrap gate because their public contracts are already fixed.

## Task Candidates

| task | task type | responsibility | dependency |
|---|---|---|---|
| PRODUCT-TASK-APPLICATION-001-01 | implementation | Implement ScoringSessionService state transitions, defaults, winning-tile preservation, preview, calculate, and result invalidation. | SYSTEM T05 |
| PRODUCT-TASK-APPLICATION-001-02 | implementation | Implement the shared scoring-condition normalization and UI-availability policy. | SYSTEM T05 |
| PRODUCT-TASK-APPLICATION-001-03 | implementation | Implement correction draft commands, structural validation targeting, and validated structure commit. | SYSTEM T05 |
| PRODUCT-TASK-APPLICATION-001-04 | implementation | Bind the public Application services to the cross-page Zustand session store without storing lifecycle runtime resources. | A01, A02, A03 |
| PRODUCT-TASK-APPLICATION-001-05 | verification | Execute the objective Application contract suite against deterministic service fakes. | A04 |
| PRODUCT-TASK-APPLICATION-001-06 | review | Independently review the complete Application-layer implementation. | A05 |

## Completion Condition

- Scoring-session behavior matches the Application and system contracts.
- Initial condition/winning-tile defaults and later user correction semantics are covered by focused tests.
- One shared condition policy drives both normalization and UI availability semantics.
- Correction drafts allow temporary malformed local state but only commit supported complete winning structure.
- Zustand owns only cross-page Application state and does not contain camera/model/WASM lifecycle resources.
- The objective Application contract verification is PASS.
- The independent integrated review is PASS with no unresolved findings.

## Evidence

- The Application scoring-session Specification fixes session ownership and correction/recalculation behavior.
- The Application session, condition-policy, and correction-editor contracts fix the implementation-facing boundaries.
- The production testing strategy explicitly permits deterministic fake Scoring/Recognition dependencies so this Work Item can proceed independently of feature runtime completion.
