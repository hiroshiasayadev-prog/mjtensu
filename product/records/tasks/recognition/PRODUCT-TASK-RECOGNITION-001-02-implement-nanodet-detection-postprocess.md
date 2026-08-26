# PRODUCT-TASK-RECOGNITION-001-02: Implement NanoDet detection and postprocess

- **status**: completed
- **date**: 2026-08-26
- **work_item**: PRODUCT-WORK-RECOGNITION-001
- **task_type**: implementation
- **estimate**: 1.5d
- **depends_on**:
  - PRODUCT-TASK-SYSTEM-001-05
- **outputs**:
  - production NanoDet pipeline implementation
  - PRODUCT-TASK-RECOGNITION-001-02

## Goal

Implement the deterministic fixed-composite NanoDet preprocessing, output decoding, semantic-region assignment, and detector-duplicate suppression path used by production Recognition.

## Work

- Implement the ADR-002 fixed semantic-region to 320x320 composite mapping.
- Port/adapt the proven NanoDet preprocessing and output decode behavior into production Recognition.
- Keep detector confidence/NMS configuration private and configurable at the runtime implementation boundary where current contracts permit it.
- Assign decoded candidates to completed-hand, dora, or meld semantic regions and reject padding/separator/outside candidates.
- Implement duplicate grouping/suppression using the accepted overlap and highest-confidence winner semantics.
- Add focused geometry/decode/duplicate tests, including neighboring non-overlapping candidates that must remain distinct.

## Implementation contract

| target | required change | acceptance criterion | verification |
|---|---|---|---|
| fixed composite | Map the three visible capture regions to the ADR-002 320x320 detector input without changing product-visible region semantics. | Known camera-region fixture coordinates map to the specified composite coordinates and back consistently. | Pure coordinate/composite unit tests. |
| NanoDet preprocess/decode | Implement production detector normalization/input and output decoding compatible with the selected NanoDet artifact contract. | Fixed tensor/output fixtures decode to expected candidate boxes/confidences/classes. | Deterministic decoder tests plus later real-artifact R06 tests. |
| region assignment | Classify retained detections into hand/dora/meld regions and reject padding/separator/outside candidates. | Boundary fixtures produce the expected semantic region or rejection state. | Region-boundary table tests. |
| duplicate suppression | Group detector duplicates under the accepted overlap rule and retain the highest-confidence candidate while preserving distinct neighbors. | Duplicate, tie, chain-component, and non-overlap fixtures return the expected winner set. | Focused postprocess unit tests. |

## Done condition

The production detector path produces deterministic region-assigned, duplicate-suppressed candidate boxes from the fixed composite and passes all focused preprocessing/decode/geometry/postprocess tests.

## Verification

- Run NanoDet preprocessing/decode unit tests.
- Run fixed-layout coordinate and region-assignment tests.
- Run duplicate-suppression matrix tests.
- Run strict typecheck/lint for the touched Recognition code.
- Defer actual production ONNX inference compatibility to R06.

## Evidence

- PRODUCT-ADR-RECOGNITION-002 defines the fixed composite geometry.
- PRODUCT-ADR-RECOGNITION-004 and `spec:product.recognition.pipeline` define the detector/postprocess placement.
- The production testing strategy requires deterministic duplicate/geometry tests below full-model integration.
- Production implementation:
  - `product/frontend/src/recognition/detector/fixed-composite.ts`
  - `product/frontend/src/recognition/detector/nanodet.ts`
  - `product/frontend/src/recognition/detector/duplicate-suppression.ts`
  - `product/frontend/src/recognition/detector/detection-postprocessor.ts`
- Focused verification on 2026-08-26: 4 Vitest files, 28 tests passed for fixed geometry/composition, preprocessing/decode/NMS, semantic-region boundaries, duplicate suppression, and integrated postprocess.
- Full frontend verification on 2026-08-26: 10 Vitest files / 83 tests passed; strict app/test typecheck passed; architecture lint passed; production Vite build passed.
- Actual production ONNX artifact inference remains intentionally deferred to PRODUCT-TASK-RECOGNITION-001-06.
