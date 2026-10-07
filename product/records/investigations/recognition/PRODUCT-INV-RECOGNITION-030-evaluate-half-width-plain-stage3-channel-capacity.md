# PRODUCT-INV-RECOGNITION-030: Evaluate half-width Plain stage3 channel capacity

- **status**: concluded
- **date**: 2026-10-05
- **trigger**: PRODUCT-INV-RECOGNITION-027 restored seeded initialization and left a large robustness gap between fast Plain classifiers and C8. A direct stage3 channel-capacity test was needed before attributing that gap to equivariance.
- **scope**: Keep the pure 0.5x Plain topology unchanged and widen only stage3 from 64 channels to 80 or 96 while holding the other stages, pooling, head, training recipe, seed, and evaluation suite fixed.
- **non_scope**: Stage1 or stage2 widening, combined widening, depthwise or pointwise substitutions, learned downsampling, late expansion, augmentation changes, multi-seed confirmation, production promotion, or explaining C8 behavior.
- **source_refs**:
  - PRODUCT-INV-RECOGNITION-018
  - PRODUCT-INV-RECOGNITION-021
  - PRODUCT-INV-RECOGNITION-027
- **follow_up_candidates**:
  - Test stage1 and stage2 channel capacity independently.
  - Revisit stage3 width only if a later topology changes the surrounding operator structure.

## Investigation scope

Test whether the pure 0.5x Plain classifier is limited by its 16x16 stage3 width.

The baseline channel schedule is `16/32/64/96`.

The experiment changes only stage3 width.

## Out of scope

- Stage1 or stage2 width changes.
- Combined stage widening.
- DW/PW substitution.
- Learned downsampling.
- Late expansion.
- Training-recipe changes.
- Multi-seed confirmation.
- Production promotion.

## Background

The seed-corrected replay showed that Plain quality does not increase enough through broad width scaling to explain the C8 robustness advantage.

A narrower hypothesis remained possible: the 64-channel stage3 feature map might be the local bottleneck.

The stage3 screen isolates that hypothesis without changing the Plain operator family.

## What was investigated

| condition | stage1 | stage2 | stage3 | stage4 |
|---|---:|---:|---:|---:|
| baseline | 16 | 32 | 64 | 96 |
| stage3-80 | 16 | 32 | 80 | 96 |
| stage3-96 | 16 | 32 | 96 | 96 |

Every block remains full convolution + batch normalization + SiLU.

MaxPool downsampling and the classifier head remain unchanged.

Training controls were:

- Corpus: `tile-classifier/gray35-jp500-seed42-v3-jp189-v1`.
- Train Protocol: `tile-classifier/tile-shape-train-gpu-v6`.
- Epochs: 150.
- Batch: 128.
- Learning rate: 0.001.
- Weight decay: 0.0001.
- Augmentation: `random360-only-v1`.
- Seed: 42.
- AMP: enabled.
- TF32: enabled.

Formal evidence:

| evidence | ref |
|---|---|
| Study | `tile-classifier/plain-half-stage3-channel-screen-v1` |
| Study Plan | `tile-classifier/plain-half-stage3-channel-screen-v1-plan-e21c8deba17770d9` |
| Study Result | `tile-classifier/run-ebb90392c6ad4c19865e6a2315074c55` |
| source commit | `af2d1aed7c520f6d74135fb321a21411092390e6` |
| runtime registry | `8` |
| completion | 3/3 training and 24/24 evaluations completed |

## Findings

| metric | stage3=64 baseline | stage3=80 | stage3=96 |
|---|---:|---:|---:|
| dense64 manual-angle mean | **0.963785** | 0.959931 | 0.949861 |
| full-class mean accuracy | **0.921333** | 0.916741 | 0.900296 |
| full-class worst accuracy | 0.791111 | **0.795556** | 0.728889 |
| Manzu mean accuracy | 0.912057 | 0.889362 | **0.919149** |
| validity balanced accuracy | 0.903858 | 0.884644 | **0.920679** |
| all-real accuracy | 0.999540 | **0.999622** | 0.998153 |
| all-real worst-class recall | 0.996881 | **0.998264** | 0.983820 |
| CPU p50 | **0.2702 ms** | 0.3397 ms | 0.3622 ms |
| iPhone p50 | **1.06 ms** | 1.17 ms | 1.29 ms |

Stage3=80 increases CPU p50 by about 26% and iPhone p50 by about 10%.

The added width does not improve the broad robustness metrics.

Stage3=96 increases latency further and loses angle, full-class mean, full-class worst, and all-real accuracy.

The experiment does not support stage3 width as the primary capacity bottleneck of the pure 0.5x Plain topology.

Worker assignment was heterogeneous.

The baseline trained on an RTX 3090.

The stage3-80 and stage3-96 trials trained on RTX 3060 workers.

Train-gpu-v6 controls initialization and requests deterministic cuDNN behavior.

Bitwise-identical training across heterogeneous GPUs has not been established.

Small deltas therefore remain hardware-confounded.

The larger observation is still useful: stage3 widening did not produce a consistent gain large enough to justify the latency cost.

## Cross-cutting observations

The result weakens the hypothesis that the 0.5x Plain model mainly lacks late spatial channels.

The result also makes broad global-width scaling less attractive.

Earlier channel placement remains a separate hypothesis.

## Follow-up judgment candidates

- Whether stage1 or stage2 width is a more productive capacity target.
- Whether stage3 widening should remain closed for the current pure Plain topology.
- Whether later comparisons need worker-pinned training before interpreting small quality deltas.

## Recommendation

Stage3 widening appears lower value than testing earlier channel placement.

The current evidence does not justify a finer stage3 width sweep.

## Follow-up artifact candidates

- A separate Investigation for stage1 and stage2 channel placement.
- A later worker-controlled confirmation only if stage3 becomes relevant again.

## Open questions

- Does stage2 width recover quality more efficiently than stage3 width?
- Does the stage3 conclusion persist under worker-pinned training?
- Is the remaining Plain-to-C8 gap primarily architectural rather than capacity-driven?
