# PRODUCT-TASK-RECOGNITION-001-06: Verify production recognition runtime

- **status**: not_started
- **date**: 2026-08-26
- **work_item**: PRODUCT-WORK-RECOGNITION-001
- **task_type**: verification
- **estimate**: 1d
- **depends_on**:
  - PRODUCT-TASK-RECOGNITION-001-05
- **outputs**:
  - PRODUCT-TASK-RECOGNITION-001-06

## Goal

Execute one objective Recognition acceptance gate using the real production ONNX artifacts plus bounded fixed fixtures and the public production runtime.

## Work

- Load the production detector, 35-class base classifier, and red-five classifier through the real model runtime.
- Verify artifact manifest/runtime-spec compatibility and provider initialization on the supported browser test environment.
- Run bounded fixed image/tensor fixtures through each actual model path and compare normalized semantic output to expected contract results.
- Run bounded full-pipeline fixtures through the public one-frame Recognition service.
- Execute the complete focused Recognition test suite from R01 through R05.
- Record expected/observed results and an overall PASS, FAIL, or validly BLOCKED verdict.

## Done condition

Every predefined real-artifact/public-runtime Recognition check has an observed result and the overall verification verdict is PASS, FAIL, or validly BLOCKED.

## Verification

| check | expected result |
|---|---|
| detector ONNX load/runtime-spec validation | PASS |
| base C8 ONNX load/runtime-spec validation | PASS |
| red-five ONNX load/runtime-spec validation | PASS |
| bounded detector fixture inference/decode | expected semantic candidate output |
| bounded base-classifier fixture inference | expected 35-class semantic result |
| bounded red-five fixture inference | expected ordinary/red semantic result |
| bounded full one-frame pipeline fixture | expected observations/recognized snapshot |
| complete Recognition focused test suite | PASS |
| strict typecheck/lint/architecture gate | PASS |

The overall result is PASS only when every required check is PASS.

## Evidence

- This Task is the L2 actual-artifact gate required by `spec:product.system.contracts.testing_strategy`.
- Model-set version, artifact hashes, runtime/provider selections, fixture identities, and observed results are recorded here when executed.
- Target-device end-to-end performance remains owned by PRODUCT-WORK-SYSTEM-002 rather than this verification.
