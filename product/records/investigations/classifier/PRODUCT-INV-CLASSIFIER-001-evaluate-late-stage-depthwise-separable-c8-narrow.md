# PRODUCT-INV-CLASSIFIER-001: Evaluate late-stage depthwise-separable C8 narrow

- **status**: concluded
- **date**: 2026-10-06
- **trigger**: PRODUCT-INV-RECOGNITION-032 showed that C8 narrow retained most C8 robustness while reducing iPhone p50 from 13.29 ms to 3.435 ms. The next question was whether late equivariant spatial mixing could be made cheaper without discarding C8 symmetry.
- **scope**: Replace both stage 3 and stage 4 3x3 equivariant convolutions in C8 narrow with field-wise grouped DW3x3 followed by equivariant PW1x1. Keep the accepted random360 e150 seed42 recipe and full evaluation suite fixed.
- **non_scope**: Stage-local attribution, further width reduction, C4/C2 comparison, quantization, detector changes, production promotion, or multi-seed architecture confirmation.
- **source_refs**:
  - PRODUCT-INV-RECOGNITION-032
  - PRODUCT-INV-RECOGNITION-033
- **follow_up_candidates**:
  - Compare stage3-only and stage4-only substitution under the same controls.
- **supersedes**:
  - PRODUCT-INV-RECOGNITION-036
- **follow_up_results**:
  - PRODUCT-INV-CLASSIFIER-002

## Investigation scope

Test whether both late C8 narrow stages can use field-wise depthwise-separable equivariant blocks without losing the quality profile that made C8 narrow attractive.

The baseline uses C8 regular fields 4/8/16/32.

The candidate changes only stages 3 and 4.

Each changed stage uses field-wise grouped DW3x3 followed by equivariant PW1x1.

Equivariant BatchNorm and ReLU follow both operations.

GroupPooling, global spatial average pooling, and the 35-logit classifier head remain unchanged.

## Out of scope

- Stage-local attribution.
- Earlier-stage substitution.
- Further C8 field-width reduction.
- C4 or C2 comparison.
- Quantization.
- Detector changes.
- Production promotion.
- Multi-seed architecture confirmation.

## Background

PRODUCT-INV-RECOGNITION-032 reduced C8 equivariant tensor width by half.

C8 narrow kept strong dense-angle and full-class robustness.

C8 narrow also reduced iPhone p50 from 13.29 ms to 3.435 ms.

The remaining latency was still above the Plain-family target.

The late 3x3 equivariant convolutions were plausible remaining cost centers.

## What was investigated

| item | baseline | candidate |
|---|---|---|
| C8 regular fields | 4/8/16/32 | 4/8/16/32 |
| stage 3 spatial block | equivariant 3x3 | field-wise DW3x3 + equivariant PW1x1 |
| stage 4 spatial block | equivariant 3x3 | field-wise DW3x3 + equivariant PW1x1 |
| GroupPooling | unchanged | unchanged |
| classifier head | unchanged | unchanged |

Training controls were fixed:

- Corpus: `tile-classifier/gray35-jp500-seed42-v3-jp189-v1`.
- Train Protocol: `tile-classifier/tile-shape-train-gpu-v6`.
- Epochs: 150.
- Effective batch: 128.
- Optimizer: AdamW.
- Learning rate: 0.001.
- Weight decay: 0.0001.
- Augmentation: `random360-only-v1`.
- Seed: 42.
- AMP: enabled.
- TF32: enabled.

Evidence:

| evidence | ref |
|---|---|
| Study | `tile-classifier/c8-narrow-late2-dw3-pw1-random360-e150-full-eval-v1` |
| Study Plan | `tile-classifier/c8-narrow-late2-dw3-pw1-random360-e150-full-eval-v1-plan-8ed3b75da8913b03` |
| Study Result | `tile-classifier/run-10ddcc8ecfb9481c80133ffc5ebb8c94` |
| ClearML controller | `1d9f9878c53c4a958c9ec9ab93fe5a47` |
| Architecture | `tile-classifier/tile-c8-gray35-narrow-late2-dw3-pw1-v1` |
| source commit | `21156c61c7f0acff85260bca00eb12cc3ea8d9c0` |
| runtime registry | `8` |
| training worker | `dev-wsl-gpu3060` / RTX 3060 |

The run completed training and all eight evaluation stages.

## Findings

### Latency

| metric | C8 narrow baseline | late2 DW3/PW1 | change |
|---|---:|---:|---:|
| CPU p50 | 0.9170 ms | **0.5526 ms** | -39.7% |
| iPhone p50 | 3.435 ms | **1.690 ms** | -50.8% |
| iPhone p95 | 3.520 ms | **1.830 ms** | -48.0% |

The candidate removes about half of the remaining iPhone browser inference latency.

### Quality

| metric | C8 narrow baseline | late2 DW3/PW1 |
|---|---:|---:|
| dense64 manual angle mean | **0.979201** | 0.967813 |
| full-class mean condition accuracy | **0.957037** | 0.930815 |
| full-class worst condition accuracy | **0.844444** | 0.791111 |
| Manzu mean condition accuracy | **0.973050** | 0.889362 |
| Manzu worst condition accuracy | **0.893617** | 0.702128 |
| validity balanced accuracy | **0.933911** | 0.903858 |
| all-real accuracy | **0.999855** | 0.999702 |
| all-real worst-class recall | **0.998808** | 0.998316 |

The largest regression appears in the Manzu diagnostic.

Manzu mean condition accuracy drops by 0.083688.

Manzu worst condition accuracy drops by 0.191489.

The regression is broad enough that latency gain alone does not make the combined substitution attractive.

### Evidence boundary

The completed StudyResult is visible in ClearML.

All nine child Tasks completed.

The Study, Plan, and Architecture definitions are recoverable from source commit `21156c61c7f0acff85260bca00eb12cc3ea8d9c0`.

The completed StudyResult YAML is not present in the current working-tree `mldb_data` snapshot.

The immutable StudyResult ID and ClearML controller ID remain the evidence references.

## Cross-cutting observations

Late equivariant spatial mixing accounts for a material share of C8 narrow latency.

C8 symmetry alone does not explain the remaining runtime cost.

At least one late full equivariant 3x3 also carries important discriminative capacity.

The combined candidate establishes a useful latency bound for the C8 narrow family.

The combined candidate does not identify which late stage causes the quality regression.

## Follow-up judgment candidates

- Determine whether stage 3 or stage 4 causes most of the quality loss.
- Compare selective separation against further C8 field-width reduction.
- Treat 1.690 ms iPhone p50 as a feasible lower-latency reference, not as an accepted architecture target.

## Recommendation

Stage-local ablation appears preferable to adopting the combined stage3+stage4 substitution.

A useful selective candidate should preserve more of the baseline Manzu and validity behavior.

The selective candidate should also retain a material share of the measured latency reduction.

## Follow-up artifact candidates

- PRODUCT-INV-CLASSIFIER-002 for stage-local attribution.
- A later classifier Investigation comparing stage3-only separation against narrower C8 field schedules.
- A later model-selection artifact if one classifier candidate is adopted.

## Open questions

- Which late stage causes most of the robustness loss?
- Can one-stage substitution approach 2.5 ms iPhone p50 without the combined quality collapse?
- Does further field-width reduction produce a better quality-latency frontier than selective separation?
