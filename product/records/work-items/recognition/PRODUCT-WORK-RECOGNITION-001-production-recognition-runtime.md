# PRODUCT-WORK-RECOGNITION-001: Production recognition runtime

- **status**: in_progress
- **date**: 2026-08-26
- **source_refs**:
  - PRODUCT-ADR-RECOGNITION-002
  - PRODUCT-ADR-RECOGNITION-004
  - `spec:product.recognition.runtime_recognition`
  - `spec:product.recognition.pipeline`
  - `spec:product.system.contracts.recognition_api`
  - `spec:product.system.contracts.model_runtime`
  - `spec:product.system.contracts.testing_strategy`
- **impact_refs**: []
- **tasks**:
  - PRODUCT-TASK-RECOGNITION-001-01
  - PRODUCT-TASK-RECOGNITION-001-02
  - PRODUCT-TASK-RECOGNITION-001-03
  - PRODUCT-TASK-RECOGNITION-001-04
  - PRODUCT-TASK-RECOGNITION-001-05
  - PRODUCT-TASK-RECOGNITION-001-06
  - PRODUCT-TASK-RECOGNITION-001-07

## Goal

Implement the production browser recognition runtime from fixed camera/composite input through NanoDet, both C8 classifiers, semantic reconstruction, stabilization, and the public Recognition API.

## Boundary

This Work Item owns production recognition model/runtime infrastructure, detector/classifier preprocessing and decoding, duplicate suppression, semantic ordering/grouping/reconstruction, realtime scheduling/stabilization, and feature-level recognition integration tests.

It does not own camera-page presentation, Application scoring-session state, Scoring, model-training research, PWA release integration, or final iPhone release acceptance/performance sign-off.

## Impact Scope

| target | impact |
|---|---|
| production `recognition` module | Implement the public recognition API and private ONNX/model pipeline. |
| recognition model artifacts/manifest support | Bind detector, 35-class base classifier, and red-five classifier to production runtime contracts. |
| recognition fixtures/tests | Add fixed-image/tensor and semantic fixtures for deterministic runtime verification. |

## Task flow

```text
SYSTEM T05 bootstrap review PASS
   +-> R01 model assets/runtime sessions
   +-> R02 NanoDet preprocessing/decode/postprocess
   +-> R03 C8 base/red-five preprocessing/inference

R02 + R03 -> R04 semantic ordering/meld reconstruction/stabilization
R01 + R04 -> R05 public RecognitionRuntime/realtime composition
R05 -> R06 objective recognition integration verification -> R07 independent integrated review
```

R01, R02, and R03 may proceed in parallel after the production bootstrap gate. R02 and R03 do not wait for one another.

## Task Candidates

| task | task type | responsibility | dependency |
|---|---|---|---|
| PRODUCT-TASK-RECOGNITION-001-01 | implementation | Implement model-set asset/session initialization, runtime-spec dispatch, provider fallback, retry, and app-lifetime session ownership. | SYSTEM T05 |
| PRODUCT-TASK-RECOGNITION-001-02 | implementation | Implement fixed-composite NanoDet preprocessing, decode, region assignment, and detector duplicate suppression. | SYSTEM T05 |
| PRODUCT-TASK-RECOGNITION-001-03 | implementation | Implement 35-class grayscale C8 and RGB red-five preprocessing/inference/result mapping. | SYSTEM T05 |
| PRODUCT-TASK-RECOGNITION-001-04 | implementation | Implement ordered semantic observations, meld grouping/reconstruction, eligibility, and three-result stabilization. | R02, R03 |
| PRODUCT-TASK-RECOGNITION-001-05 | implementation | Compose the public one-frame/realtime recognition services and scheduler/lifecycle behavior. | R01, R04 |
| PRODUCT-TASK-RECOGNITION-001-06 | verification | Run actual-artifact and public-runtime contract verification with bounded fixed fixtures. | R05 |
| PRODUCT-TASK-RECOGNITION-001-07 | review | Independently review the complete production Recognition implementation. | R06 |

## Completion Condition

- All three production ONNX roles load through the model-runtime contract.
- NanoDet and both classifier paths match their accepted preprocessing/output contracts.
- Invalid/background, duplicate suppression, ordering, meld reconstruction, capture eligibility, and stabilization semantics match the Recognition Specifications.
- Realtime recognition exposes the public API without leaking ONNX Runtime types.
- Focused automated tests required by the production testing strategy pass.
- Actual-artifact contract verification is PASS.
- The independent integrated review is PASS with no unresolved findings.

## Evidence

- PRODUCT-ADR-RECOGNITION-002 fixes the production capture/composite layout.
- PRODUCT-ADR-RECOGNITION-004 fixes the 35-class base-classifier pipeline and downstream scoring-validity boundary.
- The Recognition and model-runtime Specifications define the public implementation contract.
- The production testing strategy defines lower-level and actual-artifact verification responsibilities.
- PRODUCT-TASK-RECOGNITION-001-06 completed on 2026-08-27 with the L2 actual-artifact/public-runtime gate PASS: 11 focused Recognition test files / 90 tests PASS, strict typecheck/architecture PASS, and the frozen Chromium real-ONNX gate PASS for the detector, v3_jp189 base classifier, and warm-augmented RGB red-five classifier. PRODUCT-TASK-RECOGNITION-001-07 remains the final independent integrated review before this Work Item can be completed.
