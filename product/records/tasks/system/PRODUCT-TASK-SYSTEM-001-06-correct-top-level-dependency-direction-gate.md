# PRODUCT-TASK-SYSTEM-001-06: Correct top-level dependency-direction gate

- **status**: done
- **date**: 2026-08-26
- **work_item**: PRODUCT-WORK-SYSTEM-001
- **task_type**: correction
- **estimate**: 0.5d
- **depends_on**:
  - PRODUCT-TASK-SYSTEM-001-05
- **finding_refs**:
  - PRODUCT-TASK-SYSTEM-001-05/F-MAJ-01
- **outputs**:
  - corrected production frontend architecture gate
  - PRODUCT-TASK-SYSTEM-001-06

## Goal

Correct F-MAJ-01 by making the production architecture gate reject top-level module dependencies that violate `spec:product.system.architecture`, even when the target is imported through its public entry point.

## Work

- Extend the existing architecture checker with the accepted top-level dependency-direction policy.
- Preserve public-entry-only cross-module enforcement and existing concrete-library isolation rules.
- Add bounded positive and negative probes covering representative allowed and forbidden dependency directions.
- Run the focused architecture suite and ordinary bootstrap verification commands affected by the correction.
- Do not change feature semantics, public contracts, or unrelated bootstrap behavior.

## Done condition

The static architecture gate rejects contract-invalid top-level dependency directions through public entry points, accepts the directions allowed by the architecture contract, and all directly affected verification passes.

## Verification

- Run `npm run lint`.
- Run `npm test` and confirm the architecture-boundary suite passes.
- Run `npm run typecheck`.
- Run `npm run build`.

## Evidence

- T05 F-MAJ-01 identified that public-entry shape was enforced without enforcing the complete top-level dependency direction.
- `product/frontend/scripts/check-architecture-imports.ts` now defines an explicit allowed cross-module dependency map matching the architecture contract: `domain -> none`; `camera/recognition/scoring -> domain`; `application -> camera/domain/recognition/scoring`; `ui -> application/camera/domain/recognition/scoring`; and `app -> all other top-level modules`.
- The checker reports `top-level-dependency-direction` before public-entry-path validation when a public or private target module itself is forbidden.
- `product/frontend/test/architecture-boundaries.test.ts` now proves representative allowed Application/UI/app imports and rejects `domain -> ui`, `camera -> recognition`, `scoring -> application`, and `ui -> app` through public entry points.
- Existing Recognition -> UI, public-entry-only, ONNX Runtime, and Agari WASM rules remain in place.
- `npm run lint` PASS: `Architecture import boundaries: OK (9 source files checked)`.
- `npm run typecheck` PASS with no reported TypeScript errors.
- `npm test` PASS under Vitest 4.1.11: 4 test files / 17 tests, including 13 architecture-boundary tests.
- `npm run build` PASS with Vite 8.2.2: 820 modules transformed and production `dist/` emitted successfully.
- F-MAJ-01 correction is implemented and its declared verification passes.
