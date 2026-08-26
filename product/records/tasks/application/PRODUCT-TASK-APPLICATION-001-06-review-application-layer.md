# PRODUCT-TASK-APPLICATION-001-06: Review Application layer

- **status**: done
- **date**: 2026-08-26
- **work_item**: PRODUCT-WORK-APPLICATION-001
- **task_type**: review
- **estimate**: 0.5d
- **depends_on**:
  - PRODUCT-TASK-APPLICATION-001-05
- **outputs**:
  - PRODUCT-TASK-APPLICATION-001-06

## Goal

Independently judge whether the complete Application implementation is contract-conformant, free of duplicated feature semantics, and ready for production UI/service integration.

## Work

- Review A01 through A05 implementation and verification Evidence.
- Check conformity to Application session, condition-policy, correction-editor, scoring, and architecture contracts.
- Check that UI/runtime-specific concerns have not leaked into Application state/services.
- Check that no second scoring or recognition solver exists in the Application layer.
- Record PASS or NEEDS REVISION and any named findings without repairing them in this Task.

## Done condition

The review records one integrated PASS or NEEDS REVISION verdict with complete findings/evidence for the Application boundary.

## Review verdict

**NEEDS REVISION**

The core Application services are well separated: scoring-session transitions delegate scoring, condition normalization is centralized, correction delegates whole-hand winning-shape determination to `ScoringService`, and Zustand contains no camera/model/WASM lifecycle resources. No second winning-hand, yaku, fu, point, or recognition solver was found in Application.

The integrated boundary is not yet review-PASS, however. The A05 verification evidence no longer describes the current Application source after a later Scoring change, the public Zustand surface still exposes an unrestricted whole-session replacement path that bypasses the semantic mutation boundary, and the current correction issue union has drifted from the accepted correction-editor contract.

## Findings

### F-MAJ-01: Current Application source is not the A05 verified state

- **severity**: major
- **authority**: this Task's Verification requirement to confirm the reviewed state is the A05 verified state; `spec:product.system.contracts.testing_strategy` separation of objective verification from independent review
- **evidence**: A05 recorded its PASS evidence at 2026-08-26 23:34 with `7 files / 98 tests`, typecheck, and architecture lint. The current `product/frontend/src/application/correction-draft-service.ts` was modified later at 2026-08-26 23:50 as part of the Scoring S05 integration path. S05 subsequently records focused verification including `correction-draft-service.test.ts`, but it does not rerun or replace the complete A05 Application acceptance gate.
- **risk**: A06 cannot truthfully inherit A05's integrated PASS for the current Application implementation; cross-service regressions in session/store/condition/Result-correction behavior after the Scoring API change are not covered by the recorded A05 snapshot.
- **required outcome**: Re-run the complete A05 Application acceptance gate against the current source and update A05 evidence, or otherwise restore and identify the exact A05-reviewed source state before re-review.
- **correction boundary**: Application verification evidence/state alignment. Do not self-close this finding inside A06.

### F-MAJ-02: Public store API permits bypassing Application session invariants

- **severity**: major
- **authority**: `spec:product.system.contracts.application_session_api` UI access rule and result-invalidation contract; PRODUCT-TASK-APPLICATION-001-04 semantic-action/store-binding contract
- **evidence**: `product/frontend/src/application/application-store.ts` publicly exposes `installScoringSession(session)` as an exact replacement operation. It accepts an arbitrary `ScoringSessionState`, including `winningTileId`, normalized/un-normalized conditions, rule profile, and `latestResult`, without delegating to `ScoringSessionService` or checking whether the result still matches the installed state. The method is part of exported `ApplicationStoreState` rather than a private test/composition seam.
- **risk**: UI/integration code can replace canonical Application state and retain or inject a stale `latestResult`, bypassing the contract that all score-relevant mutations go through Application semantics and invalidate prior results. Correctness therefore depends on caller discipline instead of the Application boundary.
- **required outcome**: Remove or narrow whole-session installation from the public UI-facing mutation surface, or make it a clearly non-production/private construction seam; production mutations should go through the semantic create/update/calculate/reset Application actions.
- **correction boundary**: Application store/API binding. Do not repair UI flow or service composition inside this review.

### F-MAJ-03: CorrectionIssue public API has drifted from the accepted correction-editor contract

- **severity**: major
- **authority**: `spec:product.system.contracts.correction_editor_api` public `CorrectionIssue` union
- **evidence**: the accepted contract defines `CorrectionIssue` as `completed-hand-count | invalid-meld | not-winning-shape`. Current `product/frontend/src/application/correction-draft-service.ts` additionally exports `invalid-completed-hand-tile` and maps Scoring's `completed-hand-tile` validation issue to that new Application discriminant. This is a cross-module public contract expansion not reflected in the accepted correction-editor contract.
- **risk**: Application and UI can compile against semantics that the authoritative contract does not declare, making independent implementations/tests and future exhaustive consumers disagree about the supported correction issue surface.
- **required outcome**: Reconcile the contract and implementation explicitly: either accept/document the additional product-semantic correction issue or map the Scoring validation outcome into the already accepted correction issue vocabulary.
- **correction boundary**: correction-editor public contract/Application mapping; no Scoring solver behavior should move into Application.

## Verification

- A01 scoring-session source was traced against `spec:product.application.scoring_session` and `spec:product.system.contracts.application_session_api`: initial defaults, rightmost winning-tile selection, stable-ID preservation/fallback, result invalidation, preview delegation, and calculation delegation are present. No concrete Agari behavior is implemented there.
- A02 condition-policy source was traced against `spec:product.system.contracts.scoring_condition_policy`: normalization and availability share one rule table, selected impossible dependent values are cleared, and no structure/yaku/scoring solver logic is present.
- A03 correction source was traced against `spec:product.system.contracts.correction_editor_api` and `spec:product.system.contracts.scoring_api`: temporary malformed drafts and local meld derivation are Application-owned; whole winning-shape validity remains delegated to `ScoringService.validateWinningStructure()`. F-MAJ-03 is the identified public-contract drift.
- A04 store source was traced against `spec:product.system.architecture`: Zustand data ownership is limited to `activeScoringSession`; service/policy dependencies are closure-owned and no camera, ONNX, realtime-recognition, Agari-WASM, or correction-draft runtime resource is stored. F-MAJ-02 remains on the public exact-install mutation seam.
- A05 records objective PASS for its then-current state, but F-MAJ-01 prevents treating that evidence as verification of the current post-S05 Application source.
- No second winning-hand decomposition, yaku detection, fu calculation, point/payment calculation, or recognition solver was found under `product/frontend/src/application/`. The local chi/pon/kan composition derivation is explicitly owned by the correction-editor contract and is not a duplicate winning-hand solver.
- This review changes only this Task record and does not repair or self-close any finding.

## Evidence

- A01 through A05 records were reviewed together with the current Application source, canonical tile model, Application session contract, condition-policy contract, correction-editor contract, Scoring API, testing strategy, and system architecture.
- The core Application responsibility split is otherwise coherent and free of concrete Agari/ONNX/browser-runtime leakage.
- Three unresolved major findings remain: stale A05 verification provenance (F-MAJ-01), a public whole-session invariant bypass (F-MAJ-02), and correction-editor public-contract drift (F-MAJ-03).
- Integrated review verdict: **NEEDS REVISION**.
- Findings and verdict are recorded here when executed.
