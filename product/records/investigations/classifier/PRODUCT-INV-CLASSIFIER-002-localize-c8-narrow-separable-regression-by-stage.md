# PRODUCT-INV-CLASSIFIER-002: Localize C8 narrow separable regression by stage

- **status**: concluded
- **date**: 2026-10-06
- **trigger**: PRODUCT-INV-CLASSIFIER-001 reduced C8 narrow iPhone p50 from 3.435 ms to 1.690 ms, but the combined stage3+stage4 substitution caused large Manzu and full-class regressions. A controlled stage-local study was required to attribute the regression.
- **scope**: Compare stage3-only and stage4-only field-wise DW3x3 plus equivariant PW1x1 substitutions under one two-trial Study. Hold all other architecture, training, seed, and evaluation conditions fixed.
- **non_scope**: Combined late2 retest, earlier-stage substitution, further width changes, quantization, multi-seed confirmation, detector work, or production promotion.
- **source_refs**:
  - PRODUCT-INV-RECOGNITION-032
  - PRODUCT-INV-RECOGNITION-033
  - PRODUCT-INV-CLASSIFIER-001
- **follow_up_candidates**:
  - Compare stage3-only separation against narrower C8 field schedules.
  - Investigate stage4 only if a new mechanism addresses its quality loss.
- **supersedes**:
  - PRODUCT-INV-RECOGNITION-037

## Investigation scope

Identify which late C8 narrow stage causes the accuracy loss observed under combined field-wise depthwise separation.

Trial 1 changes only stage 3 from the C8 narrow baseline.

Trial 2 changes only stage 4.

Both trials preserve C8 regular fields 4/8/16/32.

Both trials preserve GroupPooling, global spatial average pooling, and the 35-logit classifier head.

## Out of scope

- Combined stage3+stage4 retest.
- Stage 1 or stage 2 substitution.
- Further C8 field-width reduction.
- Quantization.
- Multi-seed architecture confirmation.
- Detector work.
- Production promotion.

## Background

PRODUCT-INV-CLASSIFIER-001 changed two late stages at once.

The combined candidate nearly halved C8 narrow iPhone latency.

The same candidate caused a large Manzu regression.

The combined experiment could not attribute the loss to stage 3, stage 4, or their interaction.

A stage-local ablation was therefore required.

## What was investigated

| trial | stage 3 | stage 4 |
|---|---|---|
| C8 narrow baseline | equivariant 3x3 | equivariant 3x3 |
| stage3-only | field-wise DW3x3 + equivariant PW1x1 | equivariant 3x3 |
| stage4-only | equivariant 3x3 | field-wise DW3x3 + equivariant PW1x1 |

Training controls matched PRODUCT-INV-CLASSIFIER-001:

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
| Study | `tile-classifier/c8-narrow-stage3-stage4-dw3-pw1-random360-e150-full-eval-v1` |
| Study Plan | `tile-classifier/c8-narrow-stage3-stage4-dw3-pw1-random360-e150-full-eval-v1-plan-a1606e5b1e7aae34` |
| Study Result | `tile-classifier/run-626b78a7fc4d4150b3e86260143b1d9b` |
| ClearML controller | `5d05c851172c4d828c6b5f980464402b` |
| stage3 Architecture | `tile-classifier/tile-c8-gray35-narrow-stage3-dw3-pw1-v1` |
| stage4 Architecture | `tile-classifier/tile-c8-gray35-narrow-stage4-dw3-pw1-v1` |
| source commit | `be9e51c8a3e7f2f529507347af08f57f389cf3cc` |
| runtime registry | `8` |

The stage3 trial trained on `precision5820-gpu3060`.

The stage4 trial trained on `dev-wsl-gpu3060`.

Both workers used RTX 3060 class GPUs.

PRODUCT-INV-RECOGNITION-033 established identical-weight reproducibility for one corrected seeded Plain Plan across these workers.

Small architecture deltas still remain subject to normal run-to-run caution.

Both trials completed training and all eight evaluation stages.

## Findings

### Quality attribution

| metric | C8 narrow baseline | stage3-only | stage4-only |
|---|---:|---:|---:|
| dense64 manual angle mean | 0.979201 | **0.983437** | 0.979549 |
| full-class mean condition accuracy | 0.957037 | **0.962519** | 0.931111 |
| full-class worst condition accuracy | 0.844444 | **0.893333** | 0.777778 |
| Manzu mean condition accuracy | **0.973050** | 0.964539 | 0.933333 |
| Manzu worst condition accuracy | **0.893617** | 0.829787 | 0.787234 |
| validity balanced accuracy | **0.933911** | 0.903858 | 0.917090 |
| all-real accuracy | 0.999855 | **0.999893** | 0.999731 |
| all-real worst-class recall | 0.998808 | **0.998834** | 0.998627 |

Stage3-only stays near or above the baseline on dense-angle, full-class, and all-real metrics.

Stage3-only still regresses Manzu worst-condition accuracy and validity balanced accuracy.

Stage4-only is materially worse on full-class and Manzu metrics.

The stage4 result moves toward the same failure profile as the combined late2 candidate.

Most observed separable quality damage therefore localizes to stage 4.

### Latency

| metric | C8 narrow baseline | stage3-only | stage4-only | combined late2 |
|---|---:|---:|---:|---:|
| CPU p50 | 0.9170 ms | 0.7318 ms | **0.7010 ms** | 0.5526 ms |
| iPhone p50 | 3.435 ms | 2.550 ms | **2.520 ms** | 1.690 ms |
| iPhone p95 | 3.520 ms | 2.690 ms | **2.590 ms** | 1.830 ms |

Stage3-only reduces iPhone p50 by 0.885 ms against the C8 narrow baseline.

Stage4-only is only 0.030 ms faster than stage3-only at iPhone p50.

That latency difference is small compared with the observed quality difference.

### Evidence boundary

The completed two-trial Study is present in ClearML.

All eighteen child Tasks completed.

The Study, Plan, and Architecture definitions are recoverable from source commit `be9e51c8a3e7f2f529507347af08f57f389cf3cc`.

The completed StudyResult YAML is not present in the current working-tree `mldb_data` snapshot.

The immutable StudyResult and controller IDs remain the evidence references.

## Cross-cutting observations

The two late C8 narrow stages are not interchangeable.

Stage 4 is substantially more sensitive to replacing full equivariant 3x3 spatial mixing.

The combined failure does not show that field-wise separation is generally incompatible with C8 narrow.

Stage3-only demonstrates a selective path with material latency savings and much smaller quality loss.

Stage3-only still exposes residual weakness in Manzu worst-condition accuracy and validity rejection.

## Follow-up judgment candidates

- Keep stage3-only as the selective-separable comparison reference.
- Compare stage3-only against further C8 field-width reduction.
- Revisit stage4 only with a materially different mechanism or hypothesis.
- Require another investigation before treating stage3-only as a production candidate.

## Recommendation

Stage3-only appears preferable to stage4-only as the selective-separable reference.

Further C8 width reduction appears more useful than repeating the same stage4 substitution.

A later comparison should place stage3-only and narrower C8 candidates on the same quality-latency frontier.

## Follow-up artifact candidates

- A classifier Investigation for further C8 field-width reduction.
- A classifier Investigation comparing stage3-only against narrower C8 candidates.
- A later model-selection artifact if one classifier candidate is adopted.

## Open questions

- Can stage3-only recover the remaining Manzu and validity gap with a targeted change?
- Does further C8 width reduction beat stage3-only on the same quality-latency frontier?
- Is stage4 sensitivity caused mainly by reduced spatial mixing, reduced cross-field mixing before GroupPooling, or both?
