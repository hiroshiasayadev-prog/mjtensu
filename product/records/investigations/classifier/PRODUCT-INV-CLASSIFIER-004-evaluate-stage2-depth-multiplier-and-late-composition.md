# PRODUCT-INV-CLASSIFIER-004: Evaluate stage2 depth multiplier and late composition

- **status**: concluded
- **date**: 2026-10-06
- **trigger**: C8 narrow retained strong robustness but remained relatively expensive at 3.435 ms iPhone p50. Replacing both late full equivariant 3x3 stages with multiplier-1 depthwise-separable blocks reduced latency to 1.690 ms but caused substantial quality regression. PRODUCT-INV-CLASSIFIER-003 then showed that restoring limited multiplier capacity at stages 3 and 4 could recover much of that quality at modest latency cost while stage 2 remained full. This suggested a compute-budget reallocation strategy: reduce the earlier, larger-spatial stage 2, then spend part of the saved budget on late-stage multiplier capacity. A stage2-only screen identified multiplier 2 as the most balanced earlier-stage setting, motivating a controlled stage3/stage4 multiplier sweep with stage2 fixed at multiplier 2.
- **scope**: First compare stage2 depth multipliers 1, 2, and 4 while stages 3 and 4 remain full equivariant convolutions. Then fix stage2 multiplier 2 and compare stage3/stage4 pairs 1/2, 2/1, and 2/2 under the same training and full eight-stage evaluation controls.
- **non_scope**: Stage 1 changes, stage2 multipliers above 4, late multipliers above 2, kernel-size changes, intermediate-activation changes, multi-seed finalist confirmation, detector work, or production promotion.
- **source_refs**:
  - PRODUCT-INV-RECOGNITION-032
  - PRODUCT-INV-RECOGNITION-033
  - PRODUCT-INV-CLASSIFIER-002
  - PRODUCT-INV-CLASSIFIER-003
- **follow_up_candidates**:
  - Keep stage2 multiplier 2 as the balanced earlier-stage reduction reference.
  - Treat 2/2/2 as the current leading combined C8 candidate and retain 2/2/1 as the faster alternative.
  - Perform finalist reproducibility confirmation before production promotion.

## Investigation scope

Determine whether the remaining full stage 2 convolution can be replaced by a field-wise depthwise-separable block without giving back the quality preserved by C8 narrow.

The first Study isolates stage 2.

Stages 3 and 4 remain full equivariant 3x3 convolutions while stage2 multiplier 1, 2, and 4 are compared.

The second Study is a composition follow-up.

It fixes stage2 multiplier 2 and then applies the same stage3/stage4 multiplier pairs used in PRODUCT-INV-CLASSIFIER-003.

Multiplier notation in the composition Study is ordered as stage2/stage3/stage4.

## Out of scope

- Stage 1 substitution.
- Stage2 multipliers above 4.
- Stage3 or stage4 multipliers above 2.
- Kernel-size changes.
- Removing or relocating intermediate BatchNorm/ReLU.
- Multi-seed finalist confirmation.
- Detector work.
- Production promotion.

## Background

The original C8 narrow baseline had strong quality but an iPhone p50 of 3.435 ms.

Replacing both stage 3 and stage 4 full equivariant 3x3 convolutions with multiplier-1 field-wise DW3x3 plus equivariant PW1x1 reduced iPhone p50 to 1.690 ms, but the quality loss was too large.

PRODUCT-INV-CLASSIFIER-003 then screened stage3/stage4 multiplier pairs while stage 2 remained full.

Its 1/2 and 2/2 candidates recovered much of the lost quality at 1.925 ms and 2.060 ms iPhone p50 respectively.

That result suggested that full equivariant connectivity did not need to be restored everywhere. Limited multiplier capacity could instead be allocated where it mattered.

Stage 2 remained attractive as the next cost target because it operates at a larger spatial resolution than stages 3 and 4.

The working hypothesis was therefore a compute-budget reallocation: save compute by separating stage 2, then use part of the saved budget to retain enough multiplier capacity in the later stages.

The stage2-only screen was used first to choose the earlier-stage operating point. Multiplier 2 provided the best balance between latency reduction and tail behavior, so the composition follow-up fixed stage2 at multiplier 2 and repeated the late 1/2, 2/1, and 2/2 allocation screen.

## What was investigated

Training controls were fixed across both Studies:

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

Stage2 screen evidence:

| evidence | ref |
|---|---|
| Study | `tile-classifier/c8-narrow-stage2-dw-multiplier-screen-random360-e150-full-eval-v1` |
| Study Plan | `tile-classifier/c8-narrow-stage2-dw-multiplier-screen-random360-e150-full-eval-v1-plan-0f73d82c8406de5b` |
| Study Result | `tile-classifier/run-cadd627d258947f5a0c5d4466956e9c6` |
| source commit | `f0d313f4ff90f97c040258c5b166f30a79f2effe` |
| runtime registry | `8` |

The stage2 screen completed all three trainings and all 24 evaluation stages.

Composition follow-up evidence:

| evidence | ref |
|---|---|
| Study | `tile-classifier/c8-narrow-stage2m2-stage34-dw-multiplier-screen-random360-e150-full-eval-v1` |
| Study Plan | `tile-classifier/c8-narrow-stage2m2-stage34-dw-multiplier-screen-random360-e150-full-eval-v1-plan-ddbe2a9634fb6698` |
| Study Result | `tile-classifier/run-bdcce50e63fb4665bebbd8d3b78ea243` |
| source commit | `d9aac51e074d4274c997e29dbb2974bdea8383e7` |
| runtime registry | `8` |

The composition follow-up completed all three trainings and all 24 evaluation stages with no failures.

C8 narrow baseline quality comes from `tile-classifier/run-ac1a2514af214cb69977972c0278ab39` trial 0002.

Baseline latency uses the v3 recovery result `tile-classifier/run-5f7624b9bea84d71821ab0d588e01fee` trial 0002.

## Findings

### Stage2 multiplier 2 is the best balanced isolated setting

| metric | full stage2 | stage2 m1 | stage2 m2 | stage2 m4 |
|---|---:|---:|---:|---:|
| dense64 manual angle mean | 0.979201 | **0.984861** | 0.980208 | 0.980903 |
| full-class mean condition accuracy | 0.957037 | 0.962815 | 0.961333 | **0.964000** |
| full-class worst condition accuracy | 0.844444 | 0.875556 | 0.882222 | **0.902222** |
| full-class worst class-condition recall | 0.272727 | 0.333333 | **0.578947** | 0.333333 |
| Manzu mean condition accuracy | 0.973050 | **0.975887** | 0.964539 | 0.973050 |
| Manzu worst condition accuracy | 0.893617 | 0.893617 | 0.872340 | **0.914894** |
| validity balanced accuracy | **0.933911** | 0.905054 | 0.889429 | 0.919483 |
| all-real accuracy | 0.999855 | 0.999892 | 0.999906 | **0.999920** |
| all-real worst-class recall | 0.998808 | 0.998886 | **0.999300** | 0.999093 |
| all-real worst class-orientation recall | 0.803571 | 0.837209 | **0.860465** | 0.814815 |

No stage2 multiplier dominates every quality metric.

Multiplier 1 is the fastest isolated stage2 setting and is strong on several mean metrics, but its full-class worst class-condition recall is only 0.333333.

Multiplier 4 has the strongest full-class worst-condition and Manzu worst-condition scores, but its tail metrics are not uniformly better and its latency nearly returns to the full-stage baseline.

Multiplier 2 has the strongest full-class worst class-condition recall and all-real worst-class recall of the three stage2 candidates.

It also retains a meaningful latency reduction.

### Stage2 multiplier 4 gives back most of the latency benefit

| metric | full stage2 | stage2 m1 | stage2 m2 | stage2 m4 |
|---|---:|---:|---:|---:|
| CPU p50 | 0.9170 ms | **0.8553 ms** | 0.9093 ms | 1.2169 ms |
| iPhone p50 | 3.435 ms | **2.715 ms** | 2.965 ms | 3.400 ms |
| iPhone p95 | 3.520 ms | **2.830 ms** | 3.080 ms | 3.630 ms |

Stage2 multiplier 4 is effectively back in the C8 narrow full-stage latency class on iPhone.

Multiplier 2 therefore forms the more useful isolated knee: it preserves stronger tail behavior than multiplier 1 while still saving about 0.47 ms versus the full C8 narrow baseline.

### Stage2 multiplier 2 does not compose neutrally with the late candidates

The composition follow-up produced:

| metric | 2/1/2 | 2/2/1 | 2/2/2 |
|---|---:|---:|---:|
| dense64 manual angle mean | 0.974965 | 0.974583 | **0.978750** |
| full-class mean condition accuracy | 0.930963 | 0.944444 | **0.951259** |
| full-class worst condition accuracy | 0.786667 | 0.835556 | **0.851111** |
| full-class worst class-condition recall | 0.250000 | 0.333333 | 0.333333 |
| Manzu mean condition accuracy | 0.916312 | **0.951773** | 0.930496 |
| Manzu worst condition accuracy | 0.680851 | **0.829787** | 0.787234 |
| validity balanced accuracy | 0.871411 | 0.903858 | **0.917090** |
| all-real accuracy | 0.999793 | 0.999838 | **0.999860** |
| all-real worst-class recall | 0.999144 | 0.999411 | **0.999508** |
| all-real worst class-orientation recall | 0.782609 | 0.813953 | **0.851852** |

The stage2-m2 plus stage3-m1 combination is clearly weak.

The earlier stage3/stage4 1/2 candidate with full stage 2 had full-class mean 0.952741.

After stage 2 is also separated, the 2/1/2 composition falls to 0.930963 full-class mean.

This is evidence of a meaningful interaction between earlier and later depthwise separation.

Once stage2 multiplier 2 is active, stage3 multiplier 2 appears necessary to avoid the strongest collapse.

### 2/2/2 is the strongest combined quality candidate at 1.54 ms

| metric | 2/1/2 | 2/2/1 | 2/2/2 |
|---|---:|---:|---:|
| CPU p50 | **0.5750 ms** | 0.5822 ms | 0.6912 ms |
| iPhone p50 | **1.380 ms** | 1.410 ms | 1.540 ms |
| iPhone p95 | **1.460 ms** | 1.470 ms | 1.660 ms |

All three compositions are below the original 1.690 ms multiplier-1 late2 reference from PRODUCT-INV-CLASSIFIER-001.

Among the viable stage3-m2 candidates, 2/2/2 costs 0.130 ms more iPhone p50 than 2/2/1.

That small latency increase buys the better result on most of the broad quality measures in the full evaluation suite:

- dense64 manual angle mean: 0.978750 versus 0.974583;
- full-class mean condition accuracy: 0.951259 versus 0.944444;
- full-class worst condition accuracy: 0.851111 versus 0.835556;
- validity balanced accuracy: 0.917090 versus 0.903858;
- all-real accuracy: 0.999860 versus 0.999838;
- all-real worst-class recall: 0.999508 versus 0.999411;
- all-real worst class-orientation recall: 0.851852 versus 0.813953.

The full-class worst class-condition recall is tied at 0.333333.

On this evidence, 2/2/2 is the leading combined architecture candidate rather than merely one side of an equal Pareto split.

### The Manzu difference is a geometric stress-test difference, not a front-facing real-crop difference

The Manzu diagnostic is the main exception to the broader 2/2/2 advantage.

Its aggregate metrics favor 2/2/1:

| metric | 2/2/1 | 2/2/2 |
|---|---:|---:|
| front-facing accuracy | 0.978723 | 0.978723 |
| mean condition accuracy | **0.951773** | 0.930496 |
| worst condition accuracy | **0.829787** | 0.787234 |

The front-facing result is identical: both models classify 46 of 47 target crops correctly.

The difference appears only after the diagnostic applies artificial geometric perturbations such as affine compression, perspective yaw, recropping, and pixel shifts.

Training used `random360-only-v1`; these affine, perspective, recrop, and shift transformations were not explicit training augmentations.

Across all 47 crops and 15 diagnostic conditions, 2/2/1 makes 34 errors out of 705 evaluations while 2/2/2 makes 49.

The worst-condition values also have a small absolute sample count: 2/2/1 is 39/47 correct and 2/2/2 is 37/47 correct.

2/2/2 additionally shows a stronger tendency to predict `invalid` under these artificial perturbations: 20 such errors versus 5 for 2/2/1 across the 705 diagnostic evaluations.

This behavior is consistent with a more conservative rejection response to unfamiliar geometry, but the diagnostic does not establish that interpretation causally.

The Manzu mean and worst metrics should therefore be treated as geometric robustness stress-test evidence, not as equivalent evidence to front-facing accuracy or exhaustive real-crop accuracy.

The current `confusion_plot` for the Manzu diagnostic contains only the front-facing condition. This explains why its confusion matrix looks nearly identical between 2/2/1 and 2/2/2 even when the aggregate perturbation metrics differ.

The Manzu stress-test result is worth retaining as a caveat, but it is not sufficient evidence to demote 2/2/2 given the identical front-facing result and the broader full-class, validity, and all-real advantage.

## Cross-cutting observations

Stage2 separation and stage3 separation are not independent optimizations.

Reducing both to low multiplier capacity compounds information loss.

The result supports the original budget-reallocation hypothesis: reducing stage 2 creates enough latency headroom to retain multiplier 2 at both later stages while still reaching 1.54 ms iPhone p50.

The 2/2/2 result is broadly stronger than 2/2/1 across dense-angle, full-class, validity, and exhaustive real-crop metrics.

The Manzu diagnostic is an important interpretation boundary because its aggregate gap is driven by artificial geometric perturbations rather than a difference on the unperturbed front-facing crops.

Artificial perturbation robustness should remain visible, but it should not silently outweigh exhaustive real-crop evidence when selecting the architecture candidate.

## Follow-up judgment candidates

- Use stage2 multiplier 2 as the balanced earlier-stage reduction reference.
- Reject stage2/stage3/stage4 2/1/2 as too aggressive.
- Treat 2/2/2 as the current leading combined C8 candidate.
- Retain 2/2/1 as the faster 1.41 ms alternative and as evidence that the artificial geometric stress test favors a different stage4 allocation.
- Do not treat the Manzu perturbation aggregate as equivalent to front-facing or exhaustive real-crop evidence.
- Confirm the leading candidate across additional seeds before production promotion.

## Recommendation

Stage2 multiplier 2 appears to be the best continuation point from the isolated stage2 screen.

When stage2 multiplier 2 is used, stage3 should also remain at multiplier 2.

Do not continue the 2/1/2 allocation.

Use 2/2/2 as the current leading C8 narrow depthwise-separable candidate.

It reaches 1.54 ms iPhone p50 while winning most of the broad full-evaluation quality comparisons against 2/2/1.

Retain 2/2/1 as a faster 1.41 ms alternative, but do not promote it over 2/2/2 solely because of the Manzu aggregate perturbation metrics.

The Manzu front-facing accuracy is identical between the two models, while the observed mean/worst difference comes from artificial affine, perspective, recrop, and shift stress conditions that were not explicit training augmentations.

Before production promotion, confirm the leading architecture with finalist reproducibility evidence and continue to keep perturbation stress behavior visible as a separate robustness signal.

## Follow-up artifact candidates

- A finalist reproducibility Investigation for 2/2/2, with 2/2/1 retained as the faster comparison candidate.
- A later architecture Investigation only if further latency or robustness improvement is required after finalist confirmation.
- A later model-selection artifact only after reproducibility and production-relevant evaluation are accepted.

## Open questions

- Does the 2/2/2 broad quality advantage reproduce across additional seeds?
- Is the stronger `invalid` response under artificial Manzu perturbations stable across seeds, and does it represent useful rejection or merely a diagnostic artifact?
- How should artificial geometric stress-test results be weighted relative to front-facing and exhaustive real-crop evidence in later model selection?
- How close can this C8-family path approach the Plain 0.5x latency class without losing the C8 robustness advantage?
