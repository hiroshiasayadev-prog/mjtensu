# PRODUCT-TASK-APPLICATION-001-04: Bind application session store

- **status**: not_started
- **date**: 2026-08-26
- **work_item**: PRODUCT-WORK-APPLICATION-001
- **task_type**: implementation
- **estimate**: 1d
- **depends_on**:
  - PRODUCT-TASK-APPLICATION-001-01
  - PRODUCT-TASK-APPLICATION-001-02
  - PRODUCT-TASK-APPLICATION-001-03
- **outputs**:
  - production Zustand application-session binding
  - PRODUCT-TASK-APPLICATION-001-04

## Goal

Bind the accepted Application services into one cross-page Zustand scoring-session store while preserving service/runtime ownership boundaries and keeping high-frequency/page-local state out of the global store.

## Work

- Implement `ScoringSessionState | null` ownership in the production Zustand store.
- Expose semantic application actions that delegate to ScoringSessionService and the shared condition policy rather than reproducing domain rules in store reducers.
- Keep camera streams, ONNX sessions, Agari WASM instances, realtime recognition updates, and correction drafts out of the global store.
- Implement new-recognition session disposal/reset and route-consumable session guards.
- Preserve latestResult invalidation/replace semantics emitted by the Application services.
- Add focused store tests for session creation/replacement/disposal and action delegation.

## Implementation contract

| target | required change | acceptance criterion | verification |
|---|---|---|---|
| Zustand session state | Store exactly the cross-page active scoring-session state or null. | Conditions/Result can consume one shared session while no-session state remains explicit. | Store unit tests. |
| semantic actions | Route session mutations through public Application services/policies. | Store code does not duplicate winning-tile, condition, correction, or scoring rules. | Action-delegation tests plus source review. |
| runtime-resource isolation | Keep camera/model/WASM/realtime/page-local correction state outside Zustand. | Store state remains serializable/product-semantic and contains no opaque lifecycle-managed runtime object. | State-shape tests and architecture/static checks. |
| restart/disposal | Provide the Application state transition needed by explicit new Recognition. | Starting a new recognition attempt clears the prior scoring session without creating a replacement until a new recognition commit exists. | Restart transition tests. |

## Done condition

The production global Application store holds only the accepted cross-page session state, delegates semantic behavior correctly, and passes focused state/action/resource-boundary tests.

## Verification

- Run store creation/mutation/reset tests.
- Run semantic action delegation tests.
- Inspect/test the store state shape for prohibited runtime resources.
- Run strict typecheck/lint/architecture checks.

## Evidence

- `spec:product.system.architecture` defines Zustand versus runtime/page-local ownership.
- `spec:product.system.contracts.application_session_api` defines the session service boundary.
- Execution results are recorded here when the Task is performed.
