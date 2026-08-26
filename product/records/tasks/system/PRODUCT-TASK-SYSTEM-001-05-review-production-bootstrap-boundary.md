# PRODUCT-TASK-SYSTEM-001-05: Review production bootstrap boundary

- **status**: done
- **date**: 2026-08-26
- **work_item**: PRODUCT-WORK-SYSTEM-001
- **task_type**: review
- **estimate**: 0.5d
- **depends_on**:
  - PRODUCT-TASK-SYSTEM-001-04
- **outputs**:
  - PRODUCT-TASK-SYSTEM-001-05

## Goal

Independently judge whether the completed production bootstrap/test-harness state is semantically sound and safe as the common base for parallel feature Work Items.

## Work

- Review the implementation and verification Evidence from T01 through T04.
- Check conformance to PRODUCT-ADR-SYSTEM-001, system architecture, and production testing strategy.
- Check that bootstrap choices did not introduce hidden feature semantics or cross-module coupling.
- Record a PASS or NEEDS REVISION verdict and any named findings.

## Done condition

The review records exactly one integrated verdict, PASS or NEEDS REVISION, with complete finding Evidence for the production bootstrap boundary.

## Review verdict

**NEEDS REVISION**

The bootstrap implementation is minimal and otherwise aligned with the selected frontend stack, strict TypeScript configuration, module skeleton, and shared test-harness boundary. However, the architecture/static enforcement is not yet sufficient to make the accepted dependency/state-ownership constraints mechanically safe for parallel feature implementation.

## Findings

### F-MAJ-01: Top-level dependency direction is not mechanically enforced

- **severity**: major
- **authority**: `spec:product.system.architecture` dependency direction; PRODUCT-TASK-SYSTEM-001-03 Goal
- **evidence**: `product/frontend/scripts/check-architecture-imports.ts` rejects cross-module deep imports and has a dedicated Recognition -> UI rule, but a cross-module public-entry import is otherwise accepted regardless of importer/target direction. The current rule set therefore permits contract-invalid public-entry dependencies such as `domain -> ui`, `camera -> ui`, or `scoring -> ui` so long as the target public `index` is used.
- **risk**: Parallel feature Tasks can introduce architecture-invalid coupling while `npm run lint`, the architecture tests, and consequently the T04 bootstrap gate still pass.
- **required outcome**: Extend the architecture gate and bounded negative tests to enforce the accepted top-level dependency direction, not only public-entry-path shape and the one Recognition -> UI special case.
- **correction boundary**: SYSTEM bootstrap architecture/static enforcement. Do not change feature semantics or public contracts while correcting this finding.

### F-MAJ-02: Prohibited opaque runtime objects in Zustand have no required static/build guard

- **severity**: major
- **authority**: `spec:product.system.contracts.testing_strategy` Architecture/static test requirements; `spec:product.system.architecture` Runtime/service versus state ownership
- **evidence**: The testing strategy explicitly requires a failing static/build gate for storage of ONNX sessions, media resources, or other opaque runtime objects in the Zustand session store. The current architecture checker exposes only `cross-feature-public-entry`, `recognition-no-ui`, `ui-application-no-onnxruntime-web`, and `ui-application-no-concrete-agari-wasm`; the architecture-boundary tests contain no Zustand/runtime-object negative probe.
- **risk**: Later Application implementation can accidentally place lifecycle-managed resources in global session state without the common bootstrap gate detecting the ownership violation.
- **required outcome**: Add a deterministic static/build check and bounded negative proof covering the prohibited Zustand/runtime-resource state ownership case required by the testing strategy.
- **correction boundary**: SYSTEM bootstrap architecture/static enforcement/test harness only. Do not implement Application feature state as part of the correction.

## Verification

- Reviewed state is based on `main` HEAD `53e53fb4ab3924901d328145e300a3c99b1f7b7c`, whose HEAD log entry is `docs(system): record production bootstrap gate pass`; T04 at that state records all six objective checks PASS.
- PRODUCT-ADR-SYSTEM-001 stack selection matches `product/frontend/package.json`: Vite, React, Mantine, Zustand, React Router, strict TypeScript, and the deferred `vite-plugin-pwa` dependency are present. `vite.config.ts` does not activate service-worker lifecycle behavior prematurely.
- `tsconfig.json` has `strict: true`; all seven required source modules and public entry points exist.
- The production bootstrap remains intentionally minimal: `App.tsx` mounts Mantine and BrowserRouter only, while feature public entry points remain empty skeletons. No Recognition, Scoring, Application, camera-runtime, store, or PWA lifecycle semantics are embedded in the bootstrap.
- Shared test support remains generic and structurally typed; the bootstrap component, public-entry, fake-service, architecture, and Playwright smoke evidence is consistent with T01 through T04.
- F-MAJ-01 and F-MAJ-02 are tied directly to accepted architecture/testing contracts and identify missing enforcement rather than undefined design preferences.
- This review changes only this Task record and does not repair either finding or modify reviewed production artifacts.

## Evidence

- T01 through T04 are complete, and T04 records overall objective verification **PASS**.
- No finding was identified against the selected framework/tool stack, strict TypeScript setup, seven-module skeleton, minimal application bootstrap, deterministic generic test support, or bootstrap smoke execution.
- Two major findings remain in the common static-enforcement boundary: incomplete top-level dependency-direction enforcement (F-MAJ-01) and the missing required Zustand/runtime-resource ownership guard (F-MAJ-02).
- Integrated review verdict: **NEEDS REVISION**.
