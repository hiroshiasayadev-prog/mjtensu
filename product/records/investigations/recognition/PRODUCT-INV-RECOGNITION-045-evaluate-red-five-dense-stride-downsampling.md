# PRODUCT-INV-RECOGNITION-045: Evaluate dense-stride downsampling for RED5

- **status**: concluded
- **date**: 2026-10-16
- **trigger**: PRODUCT-INV-RECOGNITION-043 found that dense Full6 recovered RED5 quality but cost about 8.3 ms iPhone p50, while DW/PW was faster but lost accuracy. This suggested that RED5 may require dense field/channel mixing but not high late-stage spatial resolution. Learned stride downsampling was tested to reduce spatial compute without factorizing the convolutions.
- **scope**: At fields 6/12/24/48, compare the dense Full baseline against a stride-safe variant and a stride-aggressive variant while holding batch 128, seed 42, training recipe, and full evaluation stack fixed.
- **non_scope**: Batch-size optimization; DW/PW substitution; width changes; color auxiliary features; quantization; or production promotion.
- **source_refs**:
  - PRODUCT-INV-RECOGNITION-043
  - PRODUCT-INV-RECOGNITION-044
- **follow_up_candidates**:
  - Carry stride-aggressive forward to a dedicated batch-size sweep.
  - Do not expect stride placement alone to reduce ONNX download size because parameter count remains effectively unchanged.

## Investigation scope

Test the hypothesis that RED5 needs channel/field representation more than fine spatial detail.

All candidates keep dense equivariant convolution and fields 6/12/24/48. Stage 1 remains full with the existing first max-pool. The variants change later spatial reduction:

- baseline: historical pooling topology;
- stride-safe: learned stride reduction through stages 2 and 3, retaining an 8x8 late map;
- stride-aggressive: learned stride reduction through stages 2, 3, and 4, producing a 4x4 late map.

All three have 77,852 trainable parameters.

## What was investigated

| evidence | ref |
|---|---|
| Study | `tile-classifier/red-five-c8-rgb-full-stride-batch128-screen-v1` |
| Study Plan | `tile-classifier/red-five-c8-rgb-full-stride-batch128-screen-v1-plan-5e44de9b6f44a4f8` |
| Study Result | `tile-classifier/run-ab70af18fff94c1fba3714649dd7f0ec` |
| source commit | `8d32577ce579ffc129eeb37556e35aa841bae9c7` |
| runtime registry | `8` |

The comparison retrained all three architectures at batch 128 so the architecture effect is internally controlled. Training used 50 epochs, seed 42, AdamW 0.001 / 0.0001, accepted rotation and warm-light augmentation, AMP, and TF32.

## Findings

### More aggressive spatial reduction improved both speed and observed quality at batch 128

| model | final spatial map | specialist errors | manual | warm | CPU p50 | iPhone p50 |
|---|---:|---:|---:|---:|---:|---:|
| dense Full baseline | historical | 8 | 99.17% | 100% | 2.160 ms | 8.080 ms |
| stride-safe | 8x8 | 7 | 100% | 100% | 1.208 ms | 4.465 ms |
| stride-aggressive | **4x4** | **4** | **100%** | **100%** | **0.762 ms** | **2.620 ms** |

The aggressive candidate cuts iPhone p50 by about 68% relative to the same-batch dense baseline and also has fewer audited specialist errors.

This is strong evidence against the idea that RED5 needs to preserve large late spatial maps. In this experiment, the 4x4 late representation is both faster and more accurate than the less aggressive alternatives.

### Learned stride reduces compute, not download size

| model | ONNX bytes |
|---|---:|
| dense baseline | 3,527,023 |
| stride-safe | 3,526,581 |
| stride-aggressive | 3,526,581 |

The ONNX artifacts are effectively the same size. Stride changes where the dense kernels execute; it does not remove their parameters.

Therefore the benefit is runtime compute. A separate factorization step is still required to materially reduce model download size.

### Batch 128 is not the RED5 quality optimum

The historical Full6 batch-1024 model had only two specialist errors, while the retrained same architecture at batch 128 has eight and also misses two manual examples.

This differs from the base 35-class classifier, where batch 128 had previously been the stronger quality setting. The base-classifier batch heuristic therefore does not transfer directly to RED5.

The architecture conclusion remains valid within this Study because all three candidates were trained at the same batch 128. The batch itself requires a separate controlled sweep.

## Recommendation

Select stride-aggressive as the dense RED5 topology for the next experiment.

Keep fields 6/12/24/48 and dense field/channel mixing, reduce the late spatial map to 4x4, and tune optimizer batch independently.

## Evidence boundary

The iPhone stages originally failed while the device trust/tunnel path was unavailable. They were later retried against the same immutable StageInputs and completed after the iPhone path was recovered. Final StudyResult status is `completed`.

The Study establishes the architecture direction at batch 128. It does not establish batch 128 as the preferred RED5 optimizer setting.

## Follow-up judgment candidates

- Sweep training batch on stride-aggressive only.
- After the best batch is selected, test selective late DW/PW to reduce parameter/download size.

## Follow-up artifact candidates

No production promotion is established by this investigation.

## Open questions

- Which training batch recovers or exceeds the historical Full6 quality on the aggressive topology?
- Once spatial resolution is 4x4, how much late dense convolution can be removed without losing RED5 discrimination?
