# PRODUCT-TASK-UI-001-12: Re-review production UI

- **status**: done
- **date**: 2026-08-27
- **work_item**: PRODUCT-WORK-UI-001
- **task_type**: review
- **estimate**: 0.5d
- **depends_on**:
  - PRODUCT-TASK-UI-001-11
- **finding_refs**:
  - PRODUCT-TASK-UI-001-07/F-MAJ-01
  - PRODUCT-TASK-UI-001-07/F-MAJ-02
  - PRODUCT-TASK-UI-001-07/F-MAJ-03
  - PRODUCT-TASK-UI-001-07/F-MIN-01
- **outputs**:
  - final correction review verdict for PRODUCT-WORK-UI-001
  - PRODUCT-TASK-UI-001-12

## Goal

Independently re-review the corrected production UI and decide whether every U07 finding is resolved and PRODUCT-WORK-UI-001 is contract-conformant and ready for real-service integration.

## Work

- Review U08 through U11 implementation and verification Evidence against the original U07 findings.
- Re-check Result-origin condition correction/cancel semantics and stale-result boundaries against screen-flow and Conditions contracts.
- Re-check Recognition page dependencies against the public Camera/Recognition contracts and composition-root architecture.
- Re-check production UI service contexts for fabricated feature semantics.
- Re-check Result dealer/child shortcut focus behavior.
- Confirm the corrected UI remains semantically thin over Application/feature services and preserves high-frequency/runtime state ownership boundaries.
- Record exactly one integrated PASS or NEEDS REVISION verdict and any remaining/new findings without repairing them inside this Task.

## Done condition

The re-review records one final integrated verdict with complete finding/evidence traceability. PASS requires all U07 findings to be demonstrably resolved with no new unresolved production UI boundary finding.

## Review verdict

**PASS**

The U11-verified production UI is contract-conformant for the reviewed scoring-flow boundary. All four U07 findings are closed by the corrected implementation plus current-state verification evidence, and independent re-inspection found no new unresolved production UI boundary finding.

PRODUCT-WORK-UI-001 is eligible to close.

## U07 finding closure

| finding | verdict | independent re-review evidence |
|---|---|---|
| F-MAJ-01: Result-origin Conditions correction is not an isolated correction transaction | **CLOSED** | Result-origin navigation carries an explicit correction marker. `ConditionsPage` switches that path to the injected pure `ScoringSessionService`, so winning-tile/condition edits stay page-local and do not mutate canonical Application state. The mode omits structural correction, exposes cancel, and commits only the calculated winning-tile/condition values back through Application semantic actions before canonical recalculation. U11 directly verifies cancel preservation, accepted replacement, and the post-recognition stale-result boundary. |
| F-MAJ-02: Recognition page duplicates and diverges from the public Camera/Recognition contracts | **CLOSED** | `recognition-page.tsx` imports `CameraService` / `CameraSession` only from `@/camera` and Recognition runtime/realtime/snapshot semantic types only from `@/recognition`. The page consumes public `bbox`, `classification`, and meld `interpretation` values directly. The remaining `RecognitionPageServices` type is only a page composition bundle of those public services, not a second semantic Camera/Recognition API. U11 verifies the same public contracts are used by production UI and test/E2E fixtures. |
| F-MAJ-03: Production UI context contains fallback scoring semantics | **CLOSED** | `scoring-flow-services.tsx` now contains only a nullable injected service-reference context. No production fallback creates fabricated `valid`, `no-yaku`, calculation, or correction-readiness semantics. `pages.tsx` renders an explicit unavailable-service state before scoring/correction behavior when composition is absent. U11 directly verifies that missing composition cannot surface fabricated score semantics. |
| F-MIN-01: Result `親` / `子` shortcut does not actually focus the seat-wind control | **CLOSED** | The Result shortcut navigates with `focus: 'seatWind'`; Conditions consumes that route state and `RadioButtonSet` focuses and visibly emphasizes the `自風` fieldset without introducing a separate dealer flag. Dealer/non-dealer remains derived from `conditions.seatWind`. U11 verifies focus behavior in component/browser coverage. |

## Integrated boundary review

### Result-origin correction and stale-result semantics — PASS

- Result-origin condition correction is explicitly distinguishable from initial Conditions and from the repair continuation entered after a confirmed Recognition correction.
- Result-origin edits are performed against a local session value with the pure injected session service; the canonical Application session and its existing `latestResult` remain untouched until acceptance.
- Cancel replaces the correction history entry with Result without committing local edits.
- Accepted correction replays only winning-tile and conditions changes through Application-owned semantic actions, then recalculates canonically; recognized structure is preserved on the `条件を修正` path.
- Conditions reached after a confirmed Recognition correction uses the store-backed semantic session service and has no cancel-to-old-Result action. Because structure replacement has already invalidated the old result, browser history cannot restore that stale score as current.

### Recognition public-service boundary — PASS

- Recognition UI consumes Camera and Recognition dependencies exclusively through their top-level public entries.
- The page-local camera-to-recognition frame projection adds only the visible fixed semantic region layout to a public `CameraFrame`; it does not import detector, classifier, stabilizer, ONNX, or other private Recognition implementation modules.
- Live overlay rendering reads public frame snapshot semantics and retains no duplicate page-owned observation/update service contract.
- Camera session, Recognition run, current snapshot, and startup/recovery state remain page-local rather than being promoted into Application/Zustand state.

### Scoring-flow composition and semantic thinness — PASS

- Production UI service context contains references only and does not construct fallback scoring/correction judgments.
- Conditions and Recognition-correction pages fail visibly closed when the scoring-flow service bundle is absent.
- Conditions delegates condition availability/normalization to the Application policy and preview/calculation to Application/Scoring services.
- Tile correction remains delegated to the Application correction-editor service; UI does not implement winning-shape or scoring rules.
- Result presentation consumes calculated product values and does not reconstruct yaku, fu, limits, payment, or dealer state from concrete scoring-library internals.

### Architecture and state ownership — PASS

- Current UI imports cross-feature dependencies through public `@/application`, `@/camera`, `@/domain`, `@/recognition`, and `@/scoring` entries; no reviewed UI code reaches into private feature implementation paths.
- No direct UI import of `onnxruntime-web` or concrete Agari/WASM implementation is present in the reviewed surface.
- Stable runtime/service references are kept in React composition/context boundaries, while high-frequency Recognition snapshot/camera-run state and uncommitted correction state stay local to their owning UI surfaces.
- The current architecture test mechanically covers public-entry imports, forbidden dependency directions, concrete runtime-library isolation, and runtime-resource exclusion from Zustand; U11 records that gate passing for the verified candidate.

## Verification

- U07, U08, U09, U10, U11, the UI Work Item, the relevant UI page/screen-flow contracts, Camera/Recognition public contracts, architecture contract, and current production UI source were reviewed together.
- Direct inspection of `navigation.ts`, `pages.tsx`, `conditions-page.tsx`, `recognition-page.tsx`, `scoring-flow-services.tsx`, `App.tsx`, Camera/Recognition public entries, and the focused architecture/navigation/Result/shell tests matches the corrected state described by U11; no reviewed post-U11 semantic divergence was observed.
- F-MAJ-01 and F-MIN-01 are structurally closed by explicit Result-correction route state, local session editing, semantic Application commit, cancel behavior, structural-editor exclusion, and consumed seat-wind focus.
- F-MAJ-02 is structurally closed by direct use of the public Camera/Recognition service and semantic types from their top-level entries.
- F-MAJ-03 is structurally closed by the nullable injected scoring-flow context and explicit unavailable-service presentation.
- U11 is accepted as the objective current-state gate: strict typecheck PASS; lint/architecture PASS with 52 source files checked; Vitest PASS with 29/29 files and 306/306 tests; production build PASS; E2E build PASS; focused fake-service Playwright PASS with 14/14 tests.
- Git worktree comparison is unavailable through the configured Git inspection tool for this repository, so this review does not claim a mechanically proven byte-for-byte U11 tree identity. The reviewed current source contents do match the concrete corrected forms and test assertions named by U11.
- No finding was repaired or self-closed by implementation changes inside U12.

## Findings

**None.**

All U07 findings are closed by the corrected current source plus U11 verification evidence. No new blocking or non-blocking production UI boundary finding remains.

## Evidence

- U07 recorded **NEEDS REVISION** with F-MAJ-01 through F-MAJ-03 and F-MIN-01.
- U08 corrected Result-origin condition-edit transaction/cancel semantics and seat-wind shortcut focus; its final focused Vitest, typecheck/lint, and fake-service Playwright evidence is PASS.
- U09 removed the page-private Camera/Recognition semantic contract divergence and bound Recognition UI/fakes to the public feature APIs; its focused contract/architecture and browser-flow evidence is PASS.
- U10 removed production scoring fallback semantics and made missing scoring-flow composition explicit; its focused UI/type/lint/browser evidence is PASS.
- U11 reran the complete corrected UI acceptance gate and recorded overall **PASS**, including 29/29 Vitest files / 306/306 tests and 14/14 focused Playwright cases after the E2E-mode build.
- Independent current-source review confirms the corrected responsibility split remains coherent: UI owns presentation/navigation/local drafts, Application owns canonical session transitions, Camera/Recognition own runtime semantics, and Scoring owns scoring judgments.
- Final integrated re-review verdict: **PASS**.
