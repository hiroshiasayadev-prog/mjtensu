# PRODUCT-TASK-UI-001-07: Review production UI

- **status**: done
- **date**: 2026-08-26
- **work_item**: PRODUCT-WORK-UI-001
- **task_type**: review
- **estimate**: 0.5d
- **depends_on**:
  - PRODUCT-TASK-UI-001-06
- **outputs**:
  - PRODUCT-TASK-UI-001-07

## Goal

Independently judge whether the complete production UI is contract-conformant, semantically thin over Application/feature services, and ready for real-service integration.

## Work

- Review U01 through U06 implementation and verification Evidence.
- Check conformity to screen-flow, page, component, Application, and architecture contracts.
- Check that UI code does not duplicate condition, correction, scoring, or recognition semantics owned by lower layers.
- Check direct concrete-library imports and high-frequency/runtime state ownership boundaries.
- Record PASS or NEEDS REVISION and any named findings without repairing them inside this Task.

## Done condition

The review records one integrated PASS or NEEDS REVISION verdict with complete findings/evidence for the production UI boundary.

## Review verdict

**NEEDS REVISION**

The production UI has broad page/component coverage and U06 proves the exercised fake-service happy/recovery flows, route guards, stale-result invalidation, and strict architecture/type/lint/test gates. The integrated boundary is not yet contract-conformant, however: Result-origin Conditions editing does not preserve an unchanged Result until confirmation, Recognition consumes UI-private service/update shapes rather than the public Camera/Recognition contracts, and the production UI service context contains fallback scoring semantics. These findings prevent the UI Work Item from being considered ready for real-service integration.

## Findings

### F-MAJ-01: Result-origin Conditions correction is not an isolated correction transaction

- **severity**: major
- **authority**: `spec:product.ui.screen_flow` Result recovery/back-navigation rules; `spec:product.ui.pages.conditions` Correction-mode behavior; PRODUCT-TASK-UI-001-03 and PRODUCT-TASK-UI-001-05 correction acceptance criteria
- **evidence**: `product/frontend/src/ui/navigation.ts` routes Result condition correction to the ordinary `/conditions` route without a correction-mode marker other than the optional seat-wind focus hint. `product/frontend/src/ui/pages.tsx` mounts the same `ConditionsPage` for both initial scoring and Result-origin correction and provides no cancel action or Result-origin draft boundary. In `product/frontend/src/ui/conditions-page.tsx`, every winning-tile, condition, and committed structure edit calls `replaceSession()`, which immediately calls `onSessionChange`; production wires that callback to `installScoringSession`. `ScoringSessionService.update()` invalidates `latestResult` for these mutations, so the previously valid Result is destroyed as soon as the user edits rather than only after correction is confirmed/recalculated. The same page also keeps the structural correction editor available on the `条件を修正` path even though screen-flow defines condition correction as preserving recognized structure.
- **U06 gap**: U06 verifies Result -> Conditions -> Result recalculation, but its check matrix does not verify cancelling Result-origin condition edits and returning to the unchanged Result. The green E2E therefore does not cover this required semantic branch.
- **risk**: A user entering `条件を修正` cannot safely abandon edits and recover the unchanged score. Back/navigation after an edit can only see a session whose old result has already been invalidated, and the condition-correction path can also mutate recognized structure despite the separate Recognition-correction flow.
- **required outcome**: Introduce an explicit Result-origin Conditions correction mode with uncommitted local/session-draft state, an actual cancel path that returns the unchanged Result, and commit/recalculation semantics that install changes only when accepted. Keep recognized structure preserved for condition correction or otherwise align the visible flow with the accepted screen-flow contract.
- **correction boundary**: UI/Application interaction boundary for Conditions correction. Do not move scoring validity or correction-domain logic into UI while correcting this finding.

### F-MAJ-02: Recognition page duplicates and diverges from the public Camera/Recognition contracts

- **severity**: major
- **authority**: `spec:product.system.architecture` Public entry points / Composition root; `spec:product.system.contracts.camera_api`; `spec:product.system.contracts.recognition_api` UI access rule; PRODUCT-TASK-UI-001-02 Goal and implementation contract
- **evidence**: `product/frontend/src/ui/recognition-page.tsx` defines UI-owned `RecognitionPageCameraService`, `RecognitionPageCameraSession`, `RecognitionPageCameraFrame`, `RecognitionPageRealtimeRecognizer`, `RecognitionPageRealtimeUpdate`, `RecognitionObservation`, and meld-observation types instead of consuming the public feature types. The production `product/frontend/src/camera/index.ts` currently exports no Camera API. More importantly, the current public Recognition implementation exports `RealtimeRecognizer` / `RealtimeRecognitionUpdate` whose `FrameRecognitionSnapshot` uses `TileObservation.bbox` plus `classification` and `MeldGroupObservation.interpretation` semantic objects, while the UI-private snapshot expects `observation.box`, direct `tile`, a UI-invented meld-group `id`, and a string/null interpretation. A production `RealtimeRecognizer` therefore cannot be wired directly to `RecognitionPageServicesProvider` without an additional shape-conversion adapter.
- **U06 gap**: `product/frontend/test/e2e/fake-flow-main.tsx` implements the UI-private Recognition interfaces, so the 12/12 Playwright result proves the page against those private shapes, not compatibility with the accepted public Recognition contract.
- **risk**: Real-service integration is blocked or forced to add an unnecessary semantic translation layer in the composition root, and future Recognition contract changes can leave UI tests green while production integration fails.
- **required outcome**: Expose/implement the Camera public contract and make Recognition UI consume the public Camera/Recognition service and semantic update types from their top-level entries. Any presentation-only projection should occur after receiving those public values, not by defining a second page-level service contract.
- **correction boundary**: Camera/Recognition public-contract implementation plus UI consumption seam. Do not move recognition scheduling, classification, stabilization, or grouping semantics into UI.

### F-MAJ-03: Production UI context contains fallback scoring semantics

- **severity**: major
- **authority**: `spec:product.system.architecture` Concrete-library isolation / Composition root / runtime-service ownership; PRODUCT-WORK-UI-001 completion condition; PRODUCT-TASK-UI-001-07 Work item requiring the UI to remain semantically thin
- **evidence**: `product/frontend/src/ui/scoring-flow-services.tsx` defines `deferredScoringService` inside the production UI module. Its `validateWinningStructure()` reports every structure as valid, `preview()` always reports `no-yaku`, and `calculate()` throws. The UI then constructs default `ScoringSessionService` and `CorrectionEditorService` instances from those fabricated scoring responses when no provider is supplied. By contrast, Recognition uses an explicit missing-services state rather than inventing successful feature semantics.
- **risk**: Missing composition wiring is silently converted into false product semantics. In particular, correction readiness can be reported from a fake `valid` structure result and callers can observe `役なし` instead of a clear unavailable-service boundary. Fake-service E2E supplies a provider and therefore cannot expose this production fallback behavior.
- **required outcome**: Remove semantic fake behavior from the production UI default. Require composition-root injection or expose an explicit unavailable/deferred dependency state that cannot be mistaken for a real scoring/correction judgment.
- **correction boundary**: UI service-provider/composition seam only; scoring rules remain owned by Scoring/Application.

### F-MIN-01: Result `親` / `子` shortcut does not actually focus the seat-wind control

- **severity**: minor
- **authority**: `spec:product.ui.pages.result` dealer/non-dealer shortcut; `spec:product.ui.components.condition_controls` Edit focus; PRODUCT-TASK-UI-001-05 implementation contract
- **evidence**: `navigateToConditionCorrection(navigate, 'seatWind')` stores `{ focus: 'seatWind' }` in route state, but `ConditionsPage` never reads route state and `ConditionsPageView` exposes no focus/emphasis input. The Result action therefore reaches Conditions but does not perform the required focused-seat-wind behavior.
- **risk**: The shortcut is functionally indistinguishable from ordinary `条件を修正`, so the intended compact dealer/child correction affordance is incomplete.
- **required outcome**: Consume the route focus hint in Conditions and visibly/emphatically target the seat-wind control without introducing a second dealer-state source of truth.
- **correction boundary**: UI navigation/presentation only.

## Verification

- U01 through U05 all record completed implementation evidence; U06 records overall **PASS**, including 12/12 focused Playwright fake-service flow tests and the final strict typecheck/lint/architecture/unit-test gates.
- The reviewed production UI surface is the same set of UI modules and fake-flow seam named by U06 Evidence: shell/routes, Recognition, Conditions, shared correction editor, Recognition correction, Result presentation, Application-state provider, and scoring-flow service provider. This review did not modify any reviewed production source or test artifact.
- `spec:product.system.architecture` conformance is positive for the checked direct-import/state-ownership concerns: production UI imports cross-feature code through public top-level entries, has no direct `onnxruntime-web` or concrete Agari/WASM import, keeps realtime Recognition snapshot/camera-session state page-local, keeps correction drafts component-local, and keeps runtime service references in context rather than Zustand.
- Conditions uses the shared `ScoringConditionPolicy` for availability/normalization rather than maintaining a second condition dependency table. Tile correction delegates command/update/validation/commit semantics to `CorrectionEditorService` rather than implementing a winning-hand solver in the component.
- Result presentation consumes product `ScoringCalculation` / payment / fu / yaku values and does not reconstruct scoring-library internals.
- F-MAJ-01 is traced to explicit Result-origin cancel/preservation requirements that are absent from the current Conditions implementation and absent from the U06 check matrix.
- F-MAJ-02 is traced by comparing the UI-private Recognition service/update interfaces directly with the currently exported public Recognition contract shapes and the empty Camera entry point.
- F-MAJ-03 is directly present in the production UI service-provider implementation and is independent of real WASM integration being outside this Work Item.
- F-MIN-01 is directly traceable from Result navigation state production to the absence of any Conditions-side consumer.
- Findings are intentionally not repaired or self-closed inside this review Task.

## Evidence

- This Task is the independent integrated review for PRODUCT-WORK-UI-001.
- Positive evidence: U06 overall PASS; no direct concrete-library imports from UI; public-entry-only cross-feature imports are mechanically green; high-frequency recognition/camera state and correction drafts stay outside Zustand; condition-policy and correction-service semantics are delegated to lower layers; Result uses product scoring result types.
- Unresolved findings: F-MAJ-01 Result-origin Conditions transaction/cancel semantics; F-MAJ-02 public Camera/Recognition contract divergence; F-MAJ-03 fallback scoring semantics in production UI context; F-MIN-01 unconsumed seat-wind focus hint.
- Integrated review verdict: **NEEDS REVISION**.
