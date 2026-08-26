# PRODUCT-WORK-SYSTEM-001: Production frontend bootstrap and test harness

- **status**: in_progress
- **date**: 2026-08-26
- **source_refs**:
  - PRODUCT-ADR-SYSTEM-001
  - `spec:product.system.architecture`
  - `spec:product.system.contracts.testing_strategy`
- **impact_refs**: []
- **tasks**:
  - PRODUCT-TASK-SYSTEM-001-01
  - PRODUCT-TASK-SYSTEM-001-02
  - PRODUCT-TASK-SYSTEM-001-03
  - PRODUCT-TASK-SYSTEM-001-04
  - PRODUCT-TASK-SYSTEM-001-05
  - PRODUCT-TASK-SYSTEM-001-06
  - PRODUCT-TASK-SYSTEM-001-07

## Goal

Create the production PWA source/bootstrap boundary and the shared automated-test harness that later Recognition, Scoring, Application, and UI Work Items can implement against in parallel.

## Boundary

This Work Item owns the initial Vite/React/TypeScript production project, framework dependencies already selected by PRODUCT-ADR-SYSTEM-001, the top-level module/public-entry-point skeleton, shared test runners, and mechanically enforced architecture-import boundaries.

It does not own feature implementation, production ONNX model wiring, Agari scoring implementation, final PWA cache/update behavior, feature UI pages, or release/device acceptance.

## Impact Scope

| target | impact |
|---|---|
| production frontend root | Create the Vite + React + TypeScript strict project and selected frontend dependencies. |
| production source modules | Create `app`, `domain`, `camera`, `recognition`, `scoring`, `application`, and `ui` boundaries with public entry points. |
| test configuration | Configure Vitest, Testing Library, and Playwright foundations. |
| static architecture gate | Enforce public-entry-point-only cross-feature imports and concrete-library isolation rules. |

## Task flow

```text
T01 bootstrap production project/module skeleton
  +-> T02 test harness
  +-> T03 architecture/static enforcement

T01 + T02 + T03 -> T04 objective bootstrap verification -> T05 independent integrated review
                                                        +-> T06 correct F-MAJ-01
                                                        +-> T07 correct F-MAJ-02
```

T02 and T03 may proceed in parallel after T01 establishes the project root and package/tool configuration surface.

## Task Candidates

| task | task type | responsibility | dependency |
|---|---|---|---|
| PRODUCT-TASK-SYSTEM-001-01 | implementation | Bootstrap the selected production frontend stack and module/public-entry-point skeleton. | none |
| PRODUCT-TASK-SYSTEM-001-02 | implementation | Install/configure the shared Vitest, Testing Library, and Playwright harness with deterministic test support foundations. | T01 |
| PRODUCT-TASK-SYSTEM-001-03 | implementation | Implement mechanical architecture/static import-boundary enforcement. | T01 |
| PRODUCT-TASK-SYSTEM-001-04 | verification | Execute the objective bootstrap gate across build, typecheck, lint/architecture, unit/component smoke, and Playwright smoke. | T01, T02, T03 |
| PRODUCT-TASK-SYSTEM-001-05 | review | Independently review the complete bootstrap/test-harness implementation boundary. | T04 |
| PRODUCT-TASK-SYSTEM-001-06 | correction | Correct T05 F-MAJ-01 by enforcing accepted top-level module dependency direction. | T05 |
| PRODUCT-TASK-SYSTEM-001-07 | correction | Correct T05 F-MAJ-02 by adding the required Zustand runtime-resource state guard. | T05 |

## Completion Condition

- The production frontend builds from the selected repository-local source root.
- TypeScript strict checking is active.
- The seven top-level module boundaries and public entry points required by the architecture contract exist.
- Vitest, Testing Library, and Playwright can execute at least one non-placeholder smoke path.
- Architecture/static rules fail on prohibited private/concrete-library imports.
- PRODUCT-TASK-SYSTEM-001-04 records PASS for the objective bootstrap gate.
- The independent bootstrap review has no unresolved findings; if an earlier review returned NEEDS REVISION, every named finding is corrected and independently closed before this Work Item completes.

## Evidence

- PRODUCT-ADR-SYSTEM-001 fixes the production frontend stack.
- `spec:product.system.architecture` fixes module/dependency/public-entry-point boundaries.
- `spec:product.system.contracts.testing_strategy` fixes the production test toolchain and verification layers.
- T04 completed the objective bootstrap gate with PASS on all predefined checks.
- T05 returned NEEDS REVISION with F-MAJ-01 and F-MAJ-02; those findings are routed independently to T06 and T07.
