# PRODUCT-TASK-SYSTEM-002-14: Correct Recognition meld-grouping tilt gate

- **status**: in_progress
- **date**: 2026-08-28
- **work_item**: PRODUCT-WORK-SYSTEM-002
- **task_type**: correction
- **estimate**: 0.25d
- **depends_on**:
  - PRODUCT-TASK-SYSTEM-002-04
- **finding_refs**:
  - PRODUCT-TASK-SYSTEM-002-04/F-MAJ-11
- **outputs**:
  - corrected meld-row grouping acceptance
  - captured iPhone bbox-geometry regression fixture
  - clarified Recognition tilt contract
  - PRODUCT-TASK-SYSTEM-002-14

## Goal

Correct the target-device Recognition failure where an otherwise stable `3 + 2` meld-row partition is rejected because detector bbox-center jitter makes the fitted angle of an individual short row exceed `±22.5°`.

## Work

- Keep the existing `±22.5°` bound on candidate/common row-direction search used to reconstruct meld rows.
- Remove the second hard rejection that requires every reconstructed row's least-squares `fittedRowAngle` to remain within `±22.5°`.
- Continue using each fitted row angle in partition scoring so row-direction disagreement remains a ranking penalty rather than an all-or-nothing rejection.
- Preserve the existing row-size, row-count, projected row-width, ambiguity, ordering, and meld-interpretation rules.
- Add a deterministic regression fixture using the exact meld bbox geometry from `mjtensu-recognition-debug-2026-08-27T17-37-35-246Z.json`. The five recognized meld observations must reconstruct as `7m 8m 9m` plus `1p 1p` and remain commit-eligible when the total visible non-dora minimum is met.
- Keep the existing synthetic beyond-supported-tilt case non-committable; this correction must not broaden the required common-direction search range.
- Clarify the Recognition specifications that `±22.5°` is the required common-row tilt/search support boundary, not a mandatory rejection threshold for the fitted angle of every already-stable row partition.
- Re-deploy and repeat the physical target-device arrangement that produced the captured failure before closing the finding.

## Done condition

The captured `3 + 2` meld geometry is reconstructed into two stable groups without weakening the common-direction search bound, focused Recognition semantic tests pass, and target-device re-verification no longer remains on the Recognition page solely because an individual fitted row angle exceeds `±22.5°`.

## Verification

From `product/frontend`:

- `npx vitest run test/recognition-semantics.test.ts test/recognition-stabilization.test.ts`
- `npm run typecheck`

Target-device:

- use the detector-promotion build/model set selected by PRODUCT-TASK-SYSTEM-002-13;
- repeat the physical `7m 8m 9m` / concealed-kan `1p 1p` arrangement from the recorded capture;
- confirm the live frame can progress to stabilization/commit when the visible non-dora count gate is satisfied.

## Evidence

- Debug capture `mjtensu-recognition-debug-2026-08-27T17-37-35-246Z.json` contains five valid meld observations and reports `commitEligibility.reason = unresolved-meld-geometry`.
- The exact recognized meld observations are `7m`, `8m`, `9m`, `1p`, `1p`; their geometry admits the intended `3 + 2` row partition under the existing projected-row clustering tolerance.
- The fitted angle of the three-member row is approximately `-35.15°` and the two-member row approximately `-23.04°`, demonstrating that short-row bbox-center jitter can exceed the required common-row support angle even when the spatial partition remains unambiguous.
- The pre-correction implementation applies `MAX_TILT_RADIANS` twice: once while selecting candidate/common row directions and again as a hard gate on each fitted row angle. The latter gate is redundant with the scoring residual and causes the captured false rejection.
- Correction authored on 2026-08-28: `meld-grouping.ts` now keeps `fittedRowAngle` only as an angle-residual scoring input and rejects only an undefined fit; candidate/common direction search remains bounded by `MAX_TILT_RADIANS`.
- `test/recognition-semantics.test.ts` now contains the exact five captured meld bounding boxes as a regression fixture and expects the `7m 8m 9m` / `1p 1p` `3 + 2` partition to remain commit-eligible with the visible-count gate satisfied.
- 2026-08-28 deterministic verification: `npx vitest run test/recognition-semantics.test.ts test/recognition-stabilization.test.ts` — **PASS**, 2 files / 16 tests (`recognition-semantics`: 12, `recognition-stabilization`: 4).
- 2026-08-28 strict typecheck: `npm run typecheck` — **PASS** (`tsconfig.app.json` and `tsconfig.test.json`, no errors).
- Desktop/deterministic T14 gates are complete. T14 remains `in_progress` only for target-device re-deployment/re-verification of the physical `7m 8m 9m` / `1p 1p` arrangement on the detector-promotion build.
