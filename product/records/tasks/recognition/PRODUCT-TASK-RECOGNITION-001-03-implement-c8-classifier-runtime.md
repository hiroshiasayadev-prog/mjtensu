# PRODUCT-TASK-RECOGNITION-001-03: Implement C8 classifier runtime

- **status**: not_started
- **date**: 2026-08-26
- **work_item**: PRODUCT-WORK-RECOGNITION-001
- **task_type**: implementation
- **estimate**: 1.5d
- **depends_on**:
  - PRODUCT-TASK-SYSTEM-001-05
- **outputs**:
  - production C8 classifier pipeline implementation
  - PRODUCT-TASK-RECOGNITION-001-03

## Goal

Implement production crop preprocessing and inference/result mapping for the grayscale 35-class base classifier and RGB red-five specialist.

## Work

- Implement aspect-preserving 64x64 crop preprocessing with the accepted resize/letterbox/fill policy.
- Implement grayscale base-classifier input normalization and 35-class result mapping.
- Treat invalid/background as a base-classifier outcome that does not become a recognized tile.
- Invoke the RGB red-five specialist only when the base result is `5m`, `5p`, or `5s`.
- Preserve ordinary-versus-red identity without changing the 34 base-kind vocabulary.
- Add deterministic preprocessing/tensor fixtures and classifier-result mapping tests.

## Implementation contract

| target | required change | acceptance criterion | verification |
|---|---|---|---|
| crop preprocessing | Produce the accepted 64x64 aspect-preserving letterboxed classifier crop and checkpoint normalization inputs. | Fixed image fixtures produce deterministic expected pixel/tensor values within declared numeric tolerance. | Focused preprocessing fixture tests. |
| 35-class base mapping | Map classifier logits to 34 base tile identities plus invalid/background. | Every class label maps exhaustively; invalid/background produces unresolved/non-tile observation state rather than a tile identity. | Exhaustive class-map tests. |
| red-five specialist | Run RGB red-five inference only for base `5m`, `5p`, or `5s` and refine ordinary/red identity. | Non-five base results never invoke the specialist; all three five suits map normal/red outcomes correctly. | Invocation/mapping tests with fake classifier sessions. |
| concrete runtime isolation | Keep ONNX tensors/session types private while exposing only semantic classifier outcomes to the pipeline. | Public Recognition types contain no ORT-specific values. | Typecheck/architecture checks. |

## Done condition

Both production classifier paths and their preprocessing/result mapping satisfy the Recognition pipeline contract and pass deterministic focused tests without requiring actual model accuracy evaluation.

## Verification

- Run crop preprocessing parity fixtures.
- Run exhaustive 35-class mapping tests.
- Run red-five conditional-invocation/result tests.
- Run strict typecheck/lint/architecture checks.
- Defer actual production classifier ONNX loading/prediction fixtures to R06.

## Evidence

- PRODUCT-ADR-RECOGNITION-004 fixes the integrated invalid/background 35-class base classifier.
- `spec:product.recognition.pipeline` fixes red-five specialist placement.
- The production testing strategy requires preprocessing parity and conditional-specialist tests.
- Execution results are recorded here when the Task is performed.
