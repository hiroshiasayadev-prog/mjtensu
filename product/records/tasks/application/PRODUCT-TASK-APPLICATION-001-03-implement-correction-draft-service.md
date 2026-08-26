# PRODUCT-TASK-APPLICATION-001-03: Implement correction draft service

- **status**: completed
- **date**: 2026-08-26
- **work_item**: PRODUCT-WORK-APPLICATION-001
- **task_type**: implementation
- **estimate**: 1.5d
- **depends_on**:
  - PRODUCT-TASK-SYSTEM-001-05
- **outputs**:
  - production correction-draft implementation
  - PRODUCT-TASK-APPLICATION-001-03

## Goal

Implement the permissive correction draft, semantic edit commands, local issue targeting, and validated canonical structure commit required by the shared correction editor contract.

## Work

- Implement `CorrectionDraft` construction from a committed `RecognizedStructure` without adding null placeholder tiles.
- Implement add, replace, remove, move/reorder, meld-group add/remove, and kan-openness toggle commands.
- Preserve `TileInstanceId` on identity replacement; remove IDs only on deletion and create new IDs only on addition.
- Derive chi/pon from valid three-tile composition and kan from equal four-tile composition plus explicit openness; keep malformed groups invalid rather than assigning contradictory type metadata.
- Implement product-owned structural checks for unresolved/malformed groups and completed-hand count versus logical meld count.
- Use the public ScoringService winning-structure validation boundary for whole completed-hand validity rather than reimplementing a solver.
- Return correction issues targeted to completed hand, a specific meld group, or the whole winning structure.
- Commit only a supported complete winning structure; lack of yaku or missing Conditions must not block correction commit.
- Add focused command/identity/validation/commit tests using a fake ScoringService shape validator where appropriate.

## Implementation contract

| target | required change | acceptance criterion | verification |
|---|---|---|---|
| draft commands | Implement the accepted semantic correction commands over a permissive local draft. | Temporary broken counts/melds remain representable; commands modify only their intended semantic destination. | Table-driven command tests. |
| identity semantics | Preserve instance ID for replacement, remove ID on deletion, create ID on addition. | Winning-tile identity can survive pure tile-identity correction and cannot survive deletion as the same instance. | Identity lifecycle tests. |
| meld derivation | Derive chi/pon/kan semantics from member composition and explicit kan openness without redundant stored chi/pon type. | Valid sequence/equal triple/equal four cases derive correctly; malformed groups remain invalid. | Meld derivation tests. |
| local validation | Return product-semantic issues with the smallest useful target. | Count/meld/whole-shape failures point to completed hand, the relevant meld group, or whole structure without concrete scoring-library errors. | Validation-target tests. |
| validated commit | Commit only structurally valid supported winning shape through ScoringService shape validation; ignore yaku/condition readiness for correction validity. | Non-winning shape cannot commit; winning yaku-less shape can commit. | Fake shape-validator commit tests. |

## Done condition

The correction draft implementation supports all accepted edit operations, preserves instance identity semantics, returns targeted validation issues, and commits only supported winning structure while remaining independent of yaku/Conditions readiness.

## Verification

- Run correction command/reorder/move tests.
- Run TileInstanceId replacement/add/delete tests.
- Run meld derivation and kan-openness tests.
- Run local issue-target tests.
- Run winning-shape commit and yaku-independence tests.
- Run strict typecheck/lint/architecture checks.

## Evidence

- `spec:product.system.contracts.correction_editor_api` and the canonical tile model define the correction boundary.
- `spec:product.system.contracts.scoring_api` owns whole winning-shape validation.
- Implemented `product/frontend/src/application/correction-draft-service.ts` and exported the correction editor API from `product/frontend/src/application/index.ts`.
- Added focused coverage in `product/frontend/test/correction-draft-service.test.ts` for commands, identity lifecycle, meld derivation, kan openness, targeted validation, and commit gating.
- `npm test -- correction-draft-service.test.ts` passed on 2026-08-26.
- `npm run typecheck` passed on 2026-08-26.
- `npm test` passed on 2026-08-26.
- `npm run lint` passed on 2026-08-26.
