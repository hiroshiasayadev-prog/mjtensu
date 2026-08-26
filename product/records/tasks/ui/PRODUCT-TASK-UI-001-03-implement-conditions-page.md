# PRODUCT-TASK-UI-001-03: Implement Conditions page

- **status**: not_started
- **date**: 2026-08-26
- **work_item**: PRODUCT-WORK-UI-001
- **task_type**: implementation
- **estimate**: 1.5d
- **depends_on**:
  - PRODUCT-TASK-SYSTEM-001-05
- **outputs**:
  - production Conditions page implementation
  - PRODUCT-TASK-UI-001-03

## Goal

Implement the production Conditions page for recognized-structure review, winning-tile selection, non-image condition input, live scoring preview, structural correction entry, and Calculate readiness.

## Work

- Render the recognized completed hand prominently with compact meld/dora presentation.
- Render the initial winning-tile selection supplied by Application and allow any completed-hand tile instance to be selected.
- Implement Ron/Tsumo, round wind, seat wind, riichi state, ippatsu, and secondary situational controls.
- Drive control availability from the shared scoring-condition policy rather than duplicating dependency rules in UI code.
- Render live `ScoringPreview` states including invalid-winning-shape, invalid-input, no-yaku, and ready yaku feedback.
- Keep structural correction entry separate from condition entry while using the shared correction editor component from U04 once available.
- Enable Calculate only when the current Application/scoring state is strict, consistent, winning-shape valid, and contains at least one scoring yaku.
- Add focused component tests using deterministic Application/Scoring state fakes.

## Implementation contract

| target | required change | acceptance criterion | verification |
|---|---|---|---|
| recognized structure + winning tile | Show hand/meld/dora state and make completed-hand instances selectable as winning tile. | Rightmost is only the initial default; user can select another duplicate-kind instance by app identity. | Component tests with duplicate tile identities. |
| condition controls | Render all accepted condition controls using shared policy availability/normalization behavior. | UI does not maintain a second dependency table; impossible dependent selections clear through Application/policy state. | Policy-backed interaction tests. |
| scoring preview | Show product-semantic preview states and awarded yaku feedback without Agari-specific data. | `役なし` is distinct from invalid structure/input; dora-only does not show scoring readiness. | Fake ScoringPreview matrix tests. |
| Calculate readiness | Expose Calculate only for a scoring-ready strict state. | Missing/contradictory/invalid-shape/no-yaku states cannot calculate; ready state invokes Application calculation. | Readiness/action tests. |
| correction entry | Integrate the shared tile correction editor surface without routing back to camera. | Structural edits remain local until valid commit and then replace Application structure. | Integration tests with U04 component once available. |

## Done condition

The Conditions page implements the accepted selection/condition/preview/readiness behavior, consumes the shared policy and correction boundaries, and passes focused deterministic component tests.

## Verification

- Run winning-tile selection tests including duplicate tile kinds.
- Run primary/secondary condition control interaction tests.
- Run preview-state/readiness matrix tests.
- Run calculate-action tests with fake Application/Scoring services.
- Run strict typecheck/lint/architecture checks.

## Evidence

- `spec:product.ui.pages.conditions` and `spec:product.ui.components.condition_controls` define the visible behavior.
- The Application/scoring condition contracts define the shared semantic state used by this page.
- Execution results are recorded here when the Task is performed.
