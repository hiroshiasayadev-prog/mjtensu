# PRODUCT-INV-CLASSIFIER-003: Evaluate C8 narrow late depth multipliers

- **status**: concluded
- **date**: 2026-10-06
- **trigger**: PRODUCT-INV-CLASSIFIER-002 localized most of the late depthwise-separable quality loss to stage 4. The next question was whether giving each input field more than one spatially filtered intermediate field could recover quality without restoring the full equivariant 3x3.
- **scope**: Compare stage3/stage4 depth-multiplier pairs 1/2, 2/1, and 2/2 while stage 2 remains the full C8 narrow equivariant convolution. Keep architecture width, training recipe, seed, and the full eight-stage evaluation suite fixed.
- **non_scope**: Stage 2 substitution, stage 1 changes, multipliers above 2, kernel-size changes, intermediate-activation changes, multi-seed confirmation, detector work, or production promotion.
- **source_refs**:
  - PRODUCT-INV-RECOGNITION-032
  - PRODUCT-INV-RECOGNITION-033
  - PRODUCT-INV-CLASSIFIER-001
  - PRODUCT-INV-CLASSIFIER-002
- **follow_up_candidates**:
  - Test whether stage 2 can also use a depthwise-separable multiplier while preserving the late-stage quality-latency frontier.
  - If stage 2 can be reduced, repeat the late multiplier comparison under that cheaper earlier-stage context.

## Investigation scope

Test whether the stage 4 regression identified by PRODUCT-INV-CLASSIFIER-002 is primarily a consequence of using only one spatially filtered intermediate field per input field.

The candidate block keeps C8 equivariance and uses field-wise grouped DW3x3 followed by equivariant PW1x1.

A depth multiplier of 2 means each input regular field produces two independently filtered regular fields before the pointwise mixing step.

The Study compares:

| trial | stage 2 | stage 3 | stage 4 |
|---|---|---:|---:|
| 1/2 | full equivariant 3x3 | DW multiplier 1 | DW multiplier 2 |
| 2/1 | full equivariant 3x3 | DW multiplier 2 | DW multiplier 1 |
| 2/2 | full equivariant 3x3 | DW multiplier 2 | DW multiplier 2 |

## Out of scope

- Stage 2 depthwise-separable substitution.
- Stage 1 changes.
- Depth multipliers above 2.
- 5x5 or other kernel-size changes.
- Removing or relocating the intermediate BatchNorm/ReLU.
- Multi-seed architecture confirmation.
- Detector work.
- Production promotion.

## Background

PRODUCT-INV-CLASSIFIER-001 showed that replacing both late full equivariant 3x3 stages with multiplier-1 field-wise DW3x3 plus equivariant PW1x1 reduced iPhone p50 from 3.435 ms to 1.690 ms but caused a large quality regression.

PRODUCT-INV-CLASSIFIER-002 then showed that stage 4 was substantially more sensitive than stage 3.

The multiplier-1 block gives each input field one spatially filtered intermediate field before pointwise mixing.

A plausible explanation was therefore not that field-wise separation itself was unusable, but that one spatial view per input field was too restrictive at the late stages.

## What was investigated

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

Primary evidence:

| evidence | ref |
|---|---|
| Study | `tile-classifier/c8-narrow-dw-multiplier2-screen-random360-e150-full-eval-v1` |
| Study Plan | `tile-classifier/c8-narrow-dw-multiplier2-screen-random360-e150-full-eval-v1-plan-f3b643b2f64ab0dc` |
| Study Result | `tile-classifier/run-4def379d2179423bbbb409f7c63aa6c1` |
| source commit | `06fd762e113b11d93edefd61e198484c5de6eb93` |
| runtime registry | `8` |

Architectures:

- `tile-classifier/tile-c8-gray35-narrow-dwm12-v1`.
- `tile-classifier/tile-c8-gray35-narrow-dwm21-v1`.
- `tile-classifier/tile-c8-gray35-narrow-dwm22-v1`.

The run completed all three trainings and all 24 evaluation stages.

C8 narrow baseline quality comes from `tile-classifier/run-ac1a2514af214cb69977972c0278ab39` trial 0002.

The baseline latency comparison uses the v3 recovery Study `tile-classifier/run-5f7624b9bea84d71821ab0d588e01fee` trial 0002.

## Findings

### Late multiplier allocation changes the quality profile materially

| metric | full/full | 1/2 | 2/1 | 2/2 |
|---|---:|---:|---:|---:|
| dense64 manual angle mean | 0.979201 | 0.976979 | 0.973194 | 0.977674 |
| full-class mean condition accuracy | 0.957037 | 0.952741 | 0.931704 | 0.954370 |
| full-class worst condition accuracy | 0.844444 | **0.880000** | 0.802222 | 0.868889 |
| Manzu mean condition accuracy | **0.973050** | 0.965957 | 0.939007 | 0.953191 |
| Manzu worst condition accuracy | **0.893617** | **0.893617** | 0.787234 | 0.787234 |
| validity balanced accuracy | **0.933911** | 0.873804 | 0.902661 | 0.887036 |
| all-real accuracy | 0.999855 | 0.999785 | 0.999705 | **0.999888** |
| all-real worst-class recall | 0.998808 | 0.999177 | 0.996587 | **0.999201** |
| all-real worst class-orientation recall | 0.803571 | 0.851852 | 0.743590 | **0.906667** |

The 2/1 condition is the weakest allocation.

Increasing stage 3 to multiplier 2 does not compensate for leaving stage 4 at multiplier 1.

Both 1/2 and 2/2 are much stronger than 2/1.

This supports the hypothesis that stage 4 needs more than one spatially filtered intermediate field.

The comparison between 1/2 and 2/2 is not one-sided.

The 1/2 condition preserves the baseline Manzu worst-condition score and has the best full-class worst-condition score.

The 2/2 condition has the strongest all-real accuracy, worst-class recall, and worst class-orientation recall.

### Multiplier 2 at the late stages remains inexpensive

| metric | full/full | 1/2 | 2/1 | 2/2 |
|---|---:|---:|---:|---:|
| CPU p50 | 0.9170 ms | **0.5916 ms** | 0.6328 ms | 0.6461 ms |
| iPhone p50 | 3.435 ms | 1.925 ms | **1.915 ms** | 2.060 ms |
| iPhone p95 | 3.520 ms | 2.010 ms | **1.980 ms** | 2.170 ms |

Moving stage 3 from multiplier 1 to multiplier 2 while keeping stage 4 at multiplier 2 increases iPhone p50 by only about 0.135 ms.

The 1/2 and 2/2 candidates therefore occupy a much faster latency class than the full C8 narrow baseline while retaining substantially more quality than the original multiplier-1 late2 candidate.

### The evidence does not support a single winner between 1/2 and 2/2

The 1/2 candidate is faster and materially stronger on Manzu.

The 2/2 candidate is stronger on the exhaustive all-real population and several tail metrics.

The Study used one seeded training run per architecture.

The observed trade should therefore be treated as a Pareto split rather than evidence that either architecture universally dominates the other.

## Cross-cutting observations

The late-stage failure mode is not explained by depthwise separation alone.

The number of spatially filtered intermediate fields matters.

Stage 4 is the critical allocation point in this screen.

Stage 3 multiplier 2 has a small latency cost, but its benefit depends on the metric being optimized.

The result also shows that multiplier count is a usable architecture budget: additional late multiplier capacity costs much less latency than restoring full equivariant 3x3 connectivity.

## Follow-up judgment candidates

- Keep both 1/2 and 2/2 as late-stage Pareto references.
- Treat 2/1 as a rejected allocation for this architecture family.
- Test earlier-stage cost reduction before spending more budget on late multipliers.
- Defer multi-seed finalist confirmation until the architecture search has narrowed further.

## Recommendation

Do not return to multiplier-1 stage 4 as the main path.

Use stage 4 multiplier 2 as the minimum late-stage reference for further depthwise-separable C8 narrow work.

Keep stage3/stage4 1/2 as the Manzu/latency-oriented reference and 2/2 as the all-real/tail-oriented reference.

The next useful experiment is to reduce stage 2 and then determine whether the late multiplier budget can be retained or increased without losing the latency advantage.

## Follow-up artifact candidates

- PRODUCT-INV-CLASSIFIER-004 for the stage 2 multiplier screen and the stage2-m2 composition follow-up.
- A later classifier Investigation for stage 4 multipliers above 2 if the combined stage2-m2 path remains quality-limited.
- A later finalist reproducibility Investigation after the architecture frontier is narrower.

## Open questions

- Can stage 2 be made depthwise-separable without destroying the late 1/2 or 2/2 quality profile?
- Under a cheaper stage 2, does stage 3 multiplier 2 become necessary?
- Would stage 4 multiplier 3 or 4 recover the remaining Manzu gap at a still-useful latency?
- How much of the remaining variance between 1/2 and 2/2 is architecture effect versus single-seed training variation?
