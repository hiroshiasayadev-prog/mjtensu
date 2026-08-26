# PRODUCT-TASK-APPLICATION-001-10: Re-review Application layer

- **status**: done
- **date**: 2026-08-27
- **work_item**: PRODUCT-WORK-APPLICATION-001
- **task_type**: review
- **estimate**: 0.5d
- **depends_on**:
  - PRODUCT-TASK-APPLICATION-001-09
- **finding_refs**:
  - PRODUCT-TASK-APPLICATION-001-06/F-MAJ-01
  - PRODUCT-TASK-APPLICATION-001-06/F-MAJ-02
  - PRODUCT-TASK-APPLICATION-001-06/F-MAJ-03
- **outputs**:
  - final Application integrated review verdict
  - PRODUCT-TASK-APPLICATION-001-10

## Goal

Independently re-review the corrected and currently verified Application implementation and determine whether PRODUCT-WORK-APPLICATION-001 can close with no unresolved A06 findings.

## Work

- Review A07 and A08 corrections and A09 current-state verification evidence.
- Re-check the complete Application boundary against session, condition-policy, correction-editor, Scoring API, architecture, and testing contracts.
- Confirm F-MAJ-01 is closed by current-state acceptance evidence rather than historical A05 evidence.
- Confirm F-MAJ-02 is closed structurally so ordinary production UI callers cannot bypass session/result invariants through whole-state replacement.
- Confirm F-MAJ-03 is closed with one accepted correction issue vocabulary across contract, Application, and UI consumers.
- Re-check that Application contains no second recognition/scoring solver and no browser/model/WASM lifecycle resource ownership.
- Record one PASS or NEEDS REVISION verdict with named findings. Do not repair findings inside this review.

## Done condition

The re-review records one integrated PASS or NEEDS REVISION verdict for the exact A09-verified state, and every A06 finding is either explicitly closed by evidence or remains named and unresolved.

## Review verdict

**PASS**

The A09-verified Application boundary is contract-conformant and all three A06 major findings are closed. No new blocking or non-blocking findings were identified in the reviewed Application boundary. This review made no implementation, contract, or test repair.

PRODUCT-WORK-APPLICATION-001 is eligible to close from the Application integrated-review perspective.

## A06 finding closure

| finding | verdict | independent re-review evidence |
|---|---|---|
| F-MAJ-01: stale Application verification provenance | **CLOSED** | A09 executed the complete current-state Application acceptance gate after A07/A08 and after its own `pages.tsx` type-narrowing correction: 7 files / 101 tests PASS, strict typecheck PASS, lint/architecture PASS. Direct inspection of the current Application/store/integration state matches the specific corrected state described by the final A09 evidence; no reviewed post-A09 divergence was observed. Historical A05 evidence is not used as the acceptance authority. |
| F-MAJ-02: public whole-session invariant bypass | **CLOSED** | `ApplicationStoreState` exposes only semantic create/update/preview/calculate/reset operations and contains no `installScoringSession()` or equivalent whole-session mutable action. Exact-state hydration is confined to the explicitly named construction/test `ApplicationStoreHydrationState` argument to `createApplicationStore()`. Recognition confirmation creates the canonical session through `createScoringSession()`, and production correction/calculation commits flow back through store semantic actions backed by the configured Application session port. Ordinary UI code cannot replace `activeScoringSession` wholesale through the mutable store surface. |
| F-MAJ-03: correction issue contract drift | **CLOSED** | The accepted correction-editor contract and exported Application `CorrectionIssue` union now share exactly `completed-hand-count | invalid-completed-hand-tile | invalid-meld | not-winning-shape`. Scoring `completed-hand-tile` is mapped to `invalid-completed-hand-tile` targeted at `completed-hand`, and the UI exhaustively presents the same discriminant. Whole winning-shape determination remains delegated to `ScoringService.validateWinningStructure()`. |

## Integrated boundary review

### Scoring session and store boundary — PASS

- `ScoringSessionService.create()` owns the product defaults, rightmost completed-hand winning-tile initialization, supplied rule profile, and `latestResult: null` initialization.
- `select-winning-tile`, `replace-structure`, `replace-conditions`, and `replace-rule-profile` all return a new session with `latestResult: null`.
- Structure replacement preserves the winning `TileInstanceId` only while that instance remains in the completed hand and otherwise selects the defined completed-hand default.
- Condition replacement delegates normalization to the shared condition policy rather than duplicating availability rules.
- Preview and calculation translate Application state into the public Scoring boundary and delegate to `ScoringService`; Application does not calculate yaku, fu, limits, payments, or points.
- Zustand-owned data remains limited to `activeScoringSession`; configured services and policies are closure-owned dependencies rather than mutable store data.
- No production mutable API was found that can inject an arbitrary `latestResult` or otherwise restore a stale result after a score-relevant semantic mutation.

### Condition policy — PASS

- Availability and normalization use the same `SCORING_CONDITION_RULES` authority.
- Impossible selected dependent conditions are cleared until the draft reaches a stable normalized state.
- The policy contains only product input-consistency semantics and no hand-shape, yaku, fu, limit, payment, or recognition logic.

### Correction editor boundary — PASS

- Uncommitted correction state remains a permissive `CorrectionDraft` outside canonical scoring-session state.
- Identity replacement preserves `TileInstanceId`; add/remove owns instance lifetime changes as required by the correction contract.
- Chi/pon/kan materialization is limited to correction-owned local meld composition semantics; it is not a second winning-hand solver.
- Local malformed-count/meld checks run before whole winning-shape validation.
- Whole winning-shape validity remains delegated to `ScoringService.validateWinningStructure()` and is normalized into the accepted correction issue vocabulary.
- Commit returns a canonical `RecognizedStructure` only when validation permits commit; an invalid draft cannot mutate canonical session state.

### UI/Application integration relevant to the reviewed invariants — PASS

- Recognition confirmation invokes the Application store `createScoringSession()` action instead of installing a caller-constructed session.
- The store-backed scoring adapter ignores caller-owned session objects for mutation authority and executes update/preview/calculate against the current canonical store state.
- Result-origin condition editing may use an Application `ScoringSessionService` locally as a transactional draft calculation seam, but it does not expose or use a whole-session store setter; successful commit is replayed through semantic store actions before the canonical calculation is installed.
- Recognition-origin correction commits structure through the semantic `replace-structure` path, so the prior result is invalidated before preview/recalculation routing.
- `ConditionsPageView` / `RecognitionCorrectionPageView` optional `onSessionChange` callbacks remain component-level seams; the production page binding does not connect them to an arbitrary whole-session store replacement action.

### Solver and runtime-resource isolation — PASS

- No camera, browser media, ONNX runtime/session, recognition runtime, model asset, or concrete Agari/WASM lifecycle resource is stored or owned by the Application Zustand state.
- Application imports public domain/scoring semantics only; no direct concrete Agari WASM or ONNX implementation dependency was found in the Application module.
- No second recognition pipeline, winning-hand decomposition solver, yaku detector, fu calculator, limit classifier, payment calculator, or point calculator was found under `src/application/`.
- The architecture gate recorded by A09 passed against the current candidate and mechanically covers public-entry imports, concrete runtime isolation, and prohibited Zustand runtime-resource ownership.

## Verification

- A06, A07, A08, A09, and the Application Work Item were reviewed together with the current Application source and the accepted Application-session, correction-editor, Scoring API, architecture, and testing-strategy contracts.
- The current `application-store.ts` has no `installScoringSession()` mutable action and exposes construction-only hydration separately from `ApplicationStoreState`.
- `application-store.test.ts` explicitly covers absence of whole-session replacement, session-port delegation, stale-result invalidation propagation, no-session failure behavior, and semantic-only Zustand data ownership.
- The current correction contract, `correction-draft-service.ts`, correction service tests, and tile-correction UI all use the reconciled four-kind `CorrectionIssue` vocabulary including `invalid-completed-hand-tile`.
- A09 final current-state evidence is accepted as the objective gate for this review: focused suite PASS (7 files / 101 tests), strict typecheck PASS, and architecture/lint PASS (52 source files checked).
- The reviewed source and integration contents are consistent with the corrected state named in A09. This review does not substitute A05 or the earlier pre-fix A09 run for that final evidence.
- No finding was repaired or self-closed by implementation changes inside A10.

## Findings

**None.**

All A06 findings are closed by current-state evidence and the independently inspected corrected boundary. No new Application-layer contract or architecture finding remains.

## Evidence

- A06 recorded NEEDS REVISION with three major findings: stale verification provenance, public whole-session store replacement, and correction issue contract drift.
- A07 removed the whole-session mutable store action and routed canonical production mutations through Application semantic actions.
- A08 accepted and documented `invalid-completed-hand-tile` as the product correction semantic for Scoring `completed-hand-tile` validation.
- A09 reran the complete predefined Application acceptance command set after the corrections and recorded final PASS for tests, strict typecheck, and lint/architecture.
- Independent source review confirms the Application responsibility split remains coherent: session transitions and orchestration in Application, winning-shape/scoring authority in Scoring, correction draft semantics in the correction service, and lifecycle runtime ownership outside Zustand.
- Final integrated re-review verdict: **PASS**.
