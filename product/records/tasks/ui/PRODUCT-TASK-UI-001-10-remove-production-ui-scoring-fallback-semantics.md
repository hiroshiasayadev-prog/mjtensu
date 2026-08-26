# PRODUCT-TASK-UI-001-10: Remove production UI scoring fallback semantics

- **status**: in_progress
- **date**: 2026-08-27
- **work_item**: PRODUCT-WORK-UI-001
- **task_type**: correction
- **estimate**: 0.5d
- **depends_on**:
  - PRODUCT-TASK-UI-001-07
- **finding_refs**:
  - PRODUCT-TASK-UI-001-07/F-MAJ-03
- **outputs**:
  - explicit production scoring-flow dependency boundary
  - PRODUCT-TASK-UI-001-10

## Goal

Correct F-MAJ-03 by removing fabricated scoring/correction semantics from the production UI dependency context and making missing production wiring explicit.

## Work

- Remove the UI-owned fallback `ScoringService` that reports arbitrary structures as valid and previews every state as `no-yaku`.
- Make `ScoringFlowServicesProvider` production dependencies explicit and impossible to confuse with a real scoring judgment when no provider is present.
- Prefer required composition-root injection or an explicit unavailable-service UI state over semantic fallback values.
- Keep fake Scoring/Application behavior in test-only fixtures and E2E composition, not production UI modules.
- Preserve stable service-reference/context ownership and keep runtime services out of Zustand.
- Update affected shell/page/component tests to provide explicit deterministic services where required.

## Done condition

Production UI cannot emit `valid`, `no-yaku`, calculation, or correction-readiness behavior from fabricated fallback scoring semantics; pages either receive real/injected public services or expose a clearly unavailable dependency state.

## Verification

- Verify rendering without production scoring-flow composition cannot be mistaken for a valid/no-yaku scoring state.
- Verify normal composed Conditions, correction, and Result flows still consume injected Application/Scoring services.
- Verify test-only fakes remain outside production UI source.
- Confirm service references remain context/composition owned and are not added to Zustand.
- Run affected UI tests, `npm run typecheck`, and `npm run lint`.

## Evidence

- U07 F-MAJ-03 identified `src/ui/scoring-flow-services.tsx` as defining a production fallback service whose validation and preview methods fabricate successful product-semantic outcomes.
- `src/ui/scoring-flow-services.tsx` now owns only a nullable service-reference context; the production `deferredScoringService`, fabricated `valid` / `no-yaku` responses, and UI-side `createCorrectionEditorService(...)` fallback construction are removed.
- `src/app/App.tsx` now exposes `scoringFlowServices` as an explicit composition-root input and installs `ScoringFlowServicesProvider` only when those services are supplied.
- `src/ui/pages.tsx` renders an explicit `点数計算サービスを利用できません。` state for Conditions and Recognition-correction routes when scoring-flow composition is absent, before preview, calculation, or correction validation can run.
- `test/shell-routing.test.tsx` now supplies deterministic scoring/correction services for normal composed route coverage and adds missing-composition checks that assert the unavailable state and absence of `役なし`, calculation, or correction-confirm actions.
- `test/result-page.test.tsx` now explicitly supplies the scoring-flow services needed by Result-origin navigation into Conditions and Recognition correction.
- Existing fake scoring behavior remains in `test/e2e/fake-flow-main.tsx`; no fake Scoring semantics remain in production UI source.
- Service references remain owned by React context/composition and were not added to Zustand.
- Command verification is pending: affected Vitest tests, `npm run typecheck`, and `npm run lint` must pass before this Task is marked `done`.
- `spec:product.system.architecture` requires real runtime/service construction in the composition root and UI consumption through service references rather than hidden feature semantics.
- PRODUCT-WORK-UI-001 requires UI to remain semantically thin over Application/feature services.
