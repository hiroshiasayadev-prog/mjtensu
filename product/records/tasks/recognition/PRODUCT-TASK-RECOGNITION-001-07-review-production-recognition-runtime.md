# PRODUCT-TASK-RECOGNITION-001-07: Review production recognition runtime

- **status**: completed
- **date**: 2026-08-26
- **work_item**: PRODUCT-WORK-RECOGNITION-001
- **task_type**: review
- **estimate**: 0.5d
- **depends_on**:
  - PRODUCT-TASK-RECOGNITION-001-06
- **outputs**:
  - PRODUCT-TASK-RECOGNITION-001-07

## Goal

Independently judge whether the complete production Recognition implementation is semantically sound, contract-conformant, and ready for final product integration.

## Work

- Review R01 through R06 implementation and verification Evidence.
- Check conformity to the Recognition ADRs/specifications, model-runtime/public-API contracts, and testing strategy.
- Check that scoring validity has not leaked back into Recognition acceptance.
- Check that private ONNX/runtime details remain isolated and lifecycle ownership is consistent.
- Record PASS or NEEDS REVISION and any named findings without repairing them inside this Task.

## Done condition

The review records one integrated PASS or NEEDS REVISION verdict with complete findings/evidence for the production Recognition boundary.

## Verification

- Confirm the reviewed state is the state verified by R06.
- Trace each substantive review judgment to an accepted Recognition/system contract or a clearly documented implementation risk.
- Confirm the review does not self-repair or self-close findings.

## Evidence

### Integrated review: 2026-08-27

**Verdict: PASS**

R01 through R06 are complete and internally consistent. R06 verified the reviewed production state with the pinned three-model set: 11 focused Recognition test files / 90 tests PASS, strict typecheck/architecture PASS, production E2E build PASS, and the frozen Chromium real-ONNX gate PASS for the detector, v3_jp189 35-class base classifier, and warm-augmented RGB red-five classifier.

The integrated review checked the current migrated Specifications as the implementation authority rather than applying superseded detail from older ADR text. In particular, `spec:product.recognition.runtime_recognition` explicitly states that all three semantic capture regions are always active and that empty dora/meld regions do not require enable/disable controls. The production pipeline's always-enabled three-region composition therefore conforms to the current contract.

The reviewed implementation conforms to the substantive Recognition boundary requirements:

- the fixed visible semantic regions are mapped into the accepted `320 x 320` detector composite without exposing composite coordinates through the public `RecognitionFrame` API;
- the 35-class base classifier owns `invalid/background`, and only base `5m` / `5p` / `5s` reach the RGB red-five specialist;
- invalid/background observations remain available as live unresolved feedback but do not become recognized tiles or count toward capture eligibility;
- detector duplicate suppression, semantic region assignment, completed-hand/dora ordering, meld grouping/reconstruction, and concealed-kan reconstruction are separated at the expected pipeline stages;
- capture eligibility uses the specified visible-observation minima (`10` non-dora total, `2` completed-hand) and does not call scoring/winning-shape/yaku validity;
- spatially coherent but scoring-invalid meld identities remain representable for downstream correction rather than being rejected by Recognition;
- temporal stabilization compares semantic draft structure rather than detector-box jitter and confirms only after three consecutive equivalent eligible results;
- realtime scheduling uses the accepted `100 ms` request cadence with a single acceptance-owning evaluation and no required stale-frame queue;
- route/run stop releases realtime work without disposing the app-lifetime model sessions, while `RecognitionRuntime.dispose()` owns final model-session disposal;
- ONNX Runtime objects remain below the public Recognition contracts; UI-facing Recognition code consumes public semantic/runtime types rather than ORT sessions or model internals;
- the production model set is source-pinned to one coherent detector/base/red-five artifact set and R06 verified those exact artifacts through the public runtime.

No unresolved review finding was identified. No implementation repair was performed inside this review Task.
