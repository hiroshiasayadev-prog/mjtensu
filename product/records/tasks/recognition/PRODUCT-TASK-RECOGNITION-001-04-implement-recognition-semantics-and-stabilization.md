# PRODUCT-TASK-RECOGNITION-001-04: Implement recognition semantics and stabilization

- **status**: completed
- **date**: 2026-08-26
- **work_item**: PRODUCT-WORK-RECOGNITION-001
- **task_type**: implementation
- **estimate**: 2d
- **depends_on**:
  - PRODUCT-TASK-RECOGNITION-001-02
  - PRODUCT-TASK-RECOGNITION-001-03
- **outputs**:
  - production recognition semantic reconstruction/stabilization implementation
  - PRODUCT-TASK-RECOGNITION-001-04

## Goal

Implement the semantic observation, ordering, meld grouping/reconstruction, capture eligibility, and three-consecutive-structure stabilization behavior that converts per-frame detections/classifications into committable recognition structures.

## Work

- Build per-frame tile observations with semantic region, preview-mappable box, recognized identity, or unresolved state.
- Order completed-hand and dora observations left-to-right.
- Group meld observations into stable rows supporting the accepted ±22.5° common tilt.
- Order meld groups top-to-bottom and members along each row.
- Reconstruct two same-base visible members as one logical concealed kan with four logical members without inventing hidden red status.
- Attach chi/pon/open-kan semantics only when current identities make the interpretation unambiguous; preserve malformed/illegal grouped identity state for downstream correction.
- Implement the 10-visible-non-dora / 2-completed-hand eligibility gate using actual observations rather than logical concealed-kan expansion.
- Implement semantic stabilization that ignores bbox jitter and commits after three consecutive equivalent eligible structures.
- Add focused grouping, concealed-kan, eligibility, and stabilization tests.

## Implementation contract

| target | required change | acceptance criterion | verification |
|---|---|---|---|
| observation/ordering | Produce live observations and ordered hand/dora semantic views from retained classified candidates. | Fixed candidate fixtures produce deterministic region/order/identity/unresolved outputs while preserving preview geometry separately. | Pure semantic-ordering tests. |
| meld grouping | Reconstruct spatial meld rows up to the accepted common tilt and keep separate rows distinct. | Grouping fixtures at horizontal and boundary tilt cases produce expected row membership/order; unstable grouping yields non-committable frame state. | Geometry/grouping tests. |
| concealed-kan reconstruction | Convert two same-base visible members into one four-member logical concealed kan while keeping only visible members associated with detector geometry and never inferring hidden red. | Normal/red visible-five fixtures produce the expected logical identities and no fabricated hidden red. | Concealed-kan semantic tests. |
| downstream-validity boundary | Do not reject stable grouped structure solely for no-yaku, non-winning shape, or currently illegal meld identities. | Fixtures with coherent geometry but invalid scoring composition remain representable for correction. | Semantic boundary tests. |
| eligibility/stabilization | Apply the accepted visible-observation minima and three-consecutive semantic-structure rule independent of bbox jitter. | Eligibility/count boundary cases and jitter/same/different structure sequences produce the specified stabilization state transitions. | Table-driven state-machine tests. |

## Done condition

Per-frame semantic reconstruction and temporal stabilization match the Recognition Specifications and pass focused deterministic tests for ordering, grouping, concealed kan, eligibility, downstream-validity separation, and three-result commit behavior.

## Verification

- Run semantic ordering and meld-grouping fixtures.
- Run concealed-kan reconstruction cases including red-five visibility cases.
- Run capture-eligibility boundary cases.
- Run stabilization sequence/state-machine cases including bbox-only jitter.
- Run strict typecheck/lint for touched Recognition code.

## Evidence

- `spec:product.recognition.runtime_recognition` defines ordering, meld grouping, eligibility, and stabilization semantics.
- `spec:product.recognition.pipeline` defines live-observation versus recognized-structure separation and downstream scoring-validity boundary.
- The production testing strategy requires these semantics at the deterministic unit layer.
- Production implementation authored in `product/frontend/src/recognition/semantics/` for frame observations/order, meld grouping/reconstruction, capture eligibility, semantic equality, and three-result stabilization.
- Focused deterministic coverage authored in `product/frontend/test/recognition-semantics.test.ts` and `product/frontend/test/recognition-stabilization.test.ts`.
- 2026-08-26 focused verification: `npm test -- recognition-semantics.test.ts recognition-stabilization.test.ts` PASS (2 files, 15 tests).
- 2026-08-26 architecture verification: `npm run lint` PASS (`Architecture import boundaries: OK (47 source files checked)`).
- 2026-08-26 strict typecheck: `npm run typecheck` PASS after the unrelated scoring-test mock typing issue was corrected in another session.
