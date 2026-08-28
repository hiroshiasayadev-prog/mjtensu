# PRODUCT-TASK-SYSTEM-002-14: Correct Recognition meld-row grouping

- **status**: in_progress
- **date**: 2026-08-28
- **work_item**: PRODUCT-WORK-SYSTEM-002
- **task_type**: correction
- **estimate**: 0.5d
- **depends_on**:
  - PRODUCT-TASK-SYSTEM-002-04
- **finding_refs**:
  - PRODUCT-TASK-SYSTEM-002-04/F-MAJ-11
- **outputs**:
  - robust bounded-orientation meld-row grouping
  - captured iPhone bbox-geometry regression fixture
  - clarified Recognition tilt contract
  - PRODUCT-TASK-SYSTEM-002-14

## Goal

Correct the target-device Recognition failure where valid meld observations remain non-committable because the existing angle-candidate plus `v`-sorted greedy row cutting is too sensitive to detector bbox-center jitter. Replace that row assignment with bounded common-direction candidate search and explicit small-set row-partition scoring while preventing near-vertical/rotated layouts from being accepted.

## Work

- Expand the supported common meld-row direction from `±22.5°` to `±45°`; directions outside that range remain unsupported.
- Generate common-direction candidates from pairwise bbox-center line angles within the supported range, with deterministic angle de-duplication and a horizontal fallback.
- Project bbox centers onto common-direction coordinates `(u, v)` for each candidate direction.
- Replace the previous `v`-sorted greedy cut with explicit enumeration of spatial row candidates containing `2..4` observations.
- Treat row membership as a complete-linkage-style geometric constraint: all members must fit inside a bounded perpendicular spread, and adjacent members along `u` must remain within a bounded tile-width-relative gap. A single bridge observation must not be sufficient to join otherwise separate rows.
- Search exact covers of all meld observations using at most four row candidates. Require separate row centers to remain spatially distinguishable so one physical four-member row is not split into multiple two-member rows.
- Rank valid partitions using normalized perpendicular residual, common-angle residual, adjacent-gap regularity/large-gap penalties, and a small group-count penalty. Choose the unique best spatial partition; tile/scoring legality is not part of the grouping score.
- Do not apply a separate hard rejection to each reconstructed row's least-squares fitted angle; detector bbox-center jitter on a short row is handled by partition score instead.
- Preserve top-to-bottom group ordering, left-to-right member ordering, concealed-kan reconstruction, chi/pon/open-kan interpretation, and downstream correction/scoring boundaries.
- Keep a deterministic regression fixture using the exact five meld bboxes from `mjtensu-recognition-debug-2026-08-27T17-37-35-246Z.json`; it must reconstruct `7m 8m 9m` plus `1p 1p`.
- Add boundary coverage at `±45°` and keep a synthetic `50°` layout non-committable.
- Expose the selected stable common meld-row angle as live snapshot metadata without adding it to semantic stabilization equality or commit eligibility.
- Add non-blocking capture guidance: show `牌の並びを水平にすると認識が安定します` after three consecutive frames above `30°`; once visible, clear only after three consecutive frames below `25°`. Missing/unstable angle frames reset the pending transition counter but do not clear an already-visible warning.
- Re-deploy and repeat the physical target-device arrangement before closing the finding.

## Done condition

The captured `3 + 2` meld geometry and deterministic synthetic rows through `±45°` reconstruct stably, geometry beyond the supported common direction remains non-committable, focused Recognition semantic/stabilization tests and strict typecheck pass, and target-device re-verification no longer remains on Recognition because of meld-row grouping for the recorded arrangement.

## Verification

From `product/frontend`:

- `npx vitest run test/recognition-semantics.test.ts test/recognition-stabilization.test.ts test/recognition-page.test.tsx test/recognition-services.test.ts`
- `npm run typecheck`

Target-device:

- use the detector-promotion build/model set selected by PRODUCT-TASK-SYSTEM-002-13;
- repeat the physical `7m 8m 9m` / concealed-kan `1p 1p` arrangement from the recorded capture;
- confirm the live frame can progress to stabilization/commit when the visible non-dora count gate is satisfied.

## Evidence

- Debug capture `mjtensu-recognition-debug-2026-08-27T17-37-35-246Z.json` contains five valid meld observations and reports `commitEligibility.reason = unresolved-meld-geometry`.
- The exact recognized meld observations are `7m`, `8m`, `9m`, `1p`, `1p`.
- Initial analysis found fitted short-row angles of approximately `-35.15°` and `-23.04°`. The first correction removed only the redundant per-row `±22.5°` fitted-angle hard gate while leaving the original greedy row assignment intact.
- That narrow correction passed deterministic verification on 2026-08-28 (`2` files / `16` tests and strict typecheck), but target-device re-verification showed no practical improvement. Those results therefore verify only the removed gate and do not close F-MAJ-11.
- Follow-up inspection identified the broader failure mode: the original implementation sorts projected observations by `v` and greedily appends to only the current row, then rejects the entire common-angle candidate if any resulting row has fewer than two or more than four members. This makes row assignment order-sensitive and can create singleton rows from bbox-center jitter even when a valid `3 + 2` partition exists.
- The current correction replaces that greedy cut with bounded common-direction search plus complete-linkage-style `2..4` row-candidate enumeration and exact-cover partition scoring. Common direction is now bounded to `±45°`; individual row fitted angles remain scoring inputs rather than hard gates.
- `test/recognition-semantics.test.ts` retains the original exact five captured meld bboxes and now covers deterministic row reconstruction at `0°`, `±22.5°`, and `±45°`, with `50°` remaining non-committable.
- Follow-up target capture `mjtensu-recognition-debug-2026-08-28T13-55-32-749Z` provides a second exact `7m 8m 9m` / `1p 1p` geometry whose stable selected common direction is above `30°`; that geometry is also pinned as a regression fixture so live tilt guidance is anchored to observed device geometry rather than only synthetic rows.
- 2026-08-28 deterministic verification of the expanded correction: `npx vitest run test/recognition-semantics.test.ts test/recognition-stabilization.test.ts` — **PASS**, 2 files / 18 tests (`recognition-semantics`: 14, `recognition-stabilization`: 4).
- 2026-08-28 strict typecheck: `npm run typecheck` — **PASS** (`tsconfig.app.json` and `tsconfig.test.json`, no errors).
- Follow-up capture-guidance correction exposes the selected stable common angle on each non-empty meld snapshot and adds the non-blocking `>30° x3` show / `<25° x3` clear hysteresis. Missing common-angle frames reset only the pending transition counter and never change recognition eligibility.
- 2026-08-29 deterministic verification of the capture-guidance follow-up: `npx vitest run test/recognition-semantics.test.ts test/recognition-stabilization.test.ts test/recognition-page.test.tsx test/recognition-services.test.ts` — **PASS**, 4 files / 48 tests (`recognition-semantics`: 15, `recognition-stabilization`: 4, `recognition-page`: 13, `recognition-services`: 16).
- 2026-08-29 strict typecheck after the guidance follow-up: `npm run typecheck` — **PASS** (`tsconfig.app.json` and `tsconfig.test.json`, no errors).
- Desktop/deterministic gates are complete for the grouping rewrite and non-blocking tilt guidance. T14 remains `in_progress` only for target-device re-deployment and physical warning/grouping re-verification.
