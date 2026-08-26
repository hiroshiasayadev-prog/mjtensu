# PRODUCT-TASK-SYSTEM-002-06: Review integrated release

- **status**: not_started
- **date**: 2026-08-26
- **work_item**: PRODUCT-WORK-SYSTEM-002
- **task_type**: review
- **estimate**: 0.5d
- **depends_on**:
  - PRODUCT-TASK-SYSTEM-002-05
- **outputs**:
  - PRODUCT-TASK-SYSTEM-002-06

## Goal

Independently judge whether the fully integrated production PWA and all required browser, PWA, real-device, and performance Evidence satisfy the accepted release boundary.

## Work

- Review I01 through I05 implementation/verification Evidence and the exact production build under review.
- Confirm every prerequisite feature Work Item ended in independent PASS before integration.
- Check the production composition, asset provenance, PWA lifecycle, real-service browser behavior, iPhone functional acceptance, and performance gate against accepted contracts.
- Confirm no required release check was silently waived, replaced by a lower-level test, or satisfied by stale evidence from a different build/model set.
- Record PASS or NEEDS REVISION and named findings without repairing them inside this Task.

## Done condition

The review records one integrated PASS or NEEDS REVISION verdict with complete findings/evidence for the production release boundary.

## Verification

- Confirm the reviewed production build/model-set/Agari artifact identities match I03 through I05 Evidence.
- Confirm all release-gate checks required by `spec:product.system.contracts.testing_strategy` are PASS or have an explicitly accepted external resolution.
- Confirm the 100 ms cadence performance gate is not inferred from detector-only/desktop evidence.
- Confirm findings are not repaired or self-closed inside this review.

## Evidence

- This Task is the final independent integrated review for PRODUCT-WORK-SYSTEM-002.
- The final release verdict and any named findings are recorded here when executed.
