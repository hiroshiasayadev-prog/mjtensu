# PRODUCT-TASK-SYSTEM-001-07: Correct Zustand runtime-resource guard

- **status**: done
- **date**: 2026-08-26
- **work_item**: PRODUCT-WORK-SYSTEM-001
- **task_type**: correction
- **estimate**: 0.5d
- **depends_on**:
  - PRODUCT-TASK-SYSTEM-001-05
- **finding_refs**:
  - PRODUCT-TASK-SYSTEM-001-05/F-MAJ-02
- **outputs**:
  - corrected production frontend state-ownership static gate
  - PRODUCT-TASK-SYSTEM-001-07

## Goal

Correct F-MAJ-02 by adding a deterministic static/build guard that rejects lifecycle-managed runtime resources from Zustand-owned Application state.

## Work

- Extend the existing architecture checker with a bounded state-ownership rule for Zustand store declarations.
- Cover the runtime-resource categories explicitly named by the accepted architecture/testing contracts, including ONNX inference sessions, browser media resources, and app-lifetime runtime/service references.
- Add bounded negative and positive probes proving the rule rejects prohibited store state without blocking ordinary semantic Application state.
- Keep the correction generic; do not implement the production scoring-session store or feature Application semantics.
- Run the focused architecture suite and ordinary bootstrap verification commands affected by the correction.

## Done condition

A deterministic architecture/static check fails for representative prohibited runtime-resource storage in Zustand state, accepts ordinary semantic state, and all directly affected verification passes.

## Verification

- Run `npm run lint`.
- Run `npm test` and confirm the architecture-boundary suite passes.
- Run `npm run typecheck`.
- Run `npm run build`.

## Evidence

- T05 F-MAJ-02 identified that the testing strategy requires a failing static/build guard for opaque runtime objects in Zustand session state, while the bootstrap gate had no such rule or negative proof.
- `product/frontend/scripts/check-architecture-imports.ts` now detects Application source that imports Zustand and rejects references to the runtime/resource categories explicitly represented by the accepted architecture: ONNX `InferenceSession`, browser media/frame resources, `CameraService`, `RecognitionModelAssets`, `RecognitionRuntime`, `ScoringService`, and `ScoringSessionService`.
- The rule is reported as `zustand-no-runtime-resource-state`; token scanning ignores comments and string literals rather than matching raw source text.
- `product/frontend/test/architecture-boundaries.test.ts` now includes negative probes for ONNX-session, browser-media, Recognition runtime, Camera service, and Scoring service state plus a positive semantic-state probe.
- No production Zustand store or Application feature state was introduced by this correction.
- `npm run lint` PASS: `Architecture import boundaries: OK (9 source files checked)`.
- `npm run typecheck` PASS with no reported TypeScript errors.
- `npm test` PASS under Vitest 4.1.11: 4 test files / 17 tests, including 13 architecture-boundary tests.
- `npm run build` PASS with Vite 8.2.2: 820 modules transformed and production `dist/` emitted successfully.
- F-MAJ-02 correction is implemented and its declared verification passes.
