# PRODUCT-INV-RECOGNITION-031: Evaluate half-width Plain early channel capacity

- **status**: concluded
- **date**: 2026-10-05
- **trigger**: PRODUCT-INV-RECOGNITION-030 did not find a useful stage3 widening point. The remaining simple capacity hypothesis was that the 0.5x Plain classifier loses representation earlier, especially around the 32x32 stage2 feature map.
- **scope**: Starting from pure Plain 0.5x `16/32/64/96`, widen stage1 or stage2 independently while leaving stage3, stage4, pooling, full convolutions, classifier head, training recipe, seed, and evaluation suite unchanged.
- **non_scope**: Combined stage1+stage2 widening, stage3 widening, DW/PW substitution, learned downsampling, late expansion, group-equivariant architectures, multi-seed confirmation, or production promotion.
- **source_refs**:
  - PRODUCT-INV-RECOGNITION-027
  - PRODUCT-INV-RECOGNITION-030
- **follow_up_candidates**:
  - Use `16/64/64/96` as the strongest observed pure-Plain width-placement reference.
  - Compare further Plain work against equivariant models rather than widening every stage.

## Investigation scope

Test where additional channel capacity is useful in the pure 0.5x Plain backbone.

Stage1 and stage2 are varied independently.

All later widths and operators remain fixed.

## Out of scope

- Combined stage1+stage2 widening.
- Stage3 widening.
- DW/PW substitution.
- Learned downsampling.
- Late expansion.
- Group-equivariant models.
- Training-recipe changes.
- Multi-seed confirmation.
- Production promotion.

## Background

PRODUCT-INV-RECOGNITION-030 did not support stage3 as the main Plain capacity bottleneck.

A direct early-stage screen was needed before treating Plain as architecturally saturated.

Stage1 tests initial feature-bank capacity at 64x64.

Stage2 tests representation capacity at 32x32 without widening later stages.

## What was investigated

| condition | stage1 | stage2 | stage3 | stage4 |
|---|---:|---:|---:|---:|
| baseline | 16 | 32 | 64 | 96 |
| S1-24 | 24 | 32 | 64 | 96 |
| S1-32 | 32 | 32 | 64 | 96 |
| S2-48 | 16 | 48 | 64 | 96 |
| S2-64 | 16 | 64 | 64 | 96 |

Only the named stage width changes.

The operator family remains full convolution + batch normalization + SiLU + MaxPool.

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
| Study | `tile-classifier/plain-half-early-channel-screen-v1` |
| Study Plan | `tile-classifier/plain-half-early-channel-screen-v1-plan-fd06bad926dd8d62` |
| Study Result | `tile-classifier/run-96cd149330af4e019d1353a2919e55f1` |
| source commit | `42593474924a19420ab0a21af0ed15a4589463bc` |
| runtime registry | `8` |
| completion | 5/5 training and 40/40 evaluations completed |

## Findings

| metric | baseline | S1-24 | S1-32 | S2-48 | S2-64 |
|---|---:|---:|---:|---:|---:|
| dense64 angle mean | **0.963785** | 0.953125 | 0.952292 | 0.962292 | 0.963507 |
| full-class mean | 0.921333 | 0.908148 | 0.904444 | 0.923111 | **0.928148** |
| full-class worst | 0.791111 | 0.764444 | 0.771111 | 0.791111 | **0.797778** |
| Manzu mean | 0.912057 | 0.882270 | 0.893617 | 0.903546 | **0.929078** |
| Manzu worst | 0.659574 | 0.595745 | 0.510638 | 0.659574 | **0.723404** |
| validity balanced | 0.903858 | 0.887036 | 0.902661 | 0.888233 | **0.915894** |
| all-real accuracy | 0.999540 | 0.999272 | **0.999570** | 0.999458 | 0.999567 |
| CPU p50 | **0.2699 ms** | 0.4657 ms | 0.3893 ms | 0.3417 ms | 0.4205 ms |
| iPhone p50 | **1.00 ms** | 1.26 ms | 1.46 ms | 1.31 ms | 1.60 ms |

Stage1 widening does not produce a useful trade.

Both stage1 variants are slower and lose broad robustness.

The result provides no evidence that 16 channels at 64x64 is the relevant Plain bottleneck.

Stage2 behaves differently.

S2-48 changes little beyond a small full-class mean increase.

S2-64 is the strongest observed pure-Plain channel-placement candidate.

Observed changes from baseline to S2-64 are:

| metric | baseline | S2-64 | delta |
|---|---:|---:|---:|
| full-class mean | 0.921333 | 0.928148 | +0.006815 |
| full-class worst | 0.791111 | 0.797778 | +0.006667 |
| Manzu mean | 0.912057 | 0.929078 | +0.017021 |
| Manzu worst | 0.659574 | 0.723404 | +0.063830 |
| validity balanced | 0.903858 | 0.915894 | +0.012036 |
| dense64 angle mean | 0.963785 | 0.963507 | -0.000278 |

The deployment cost is substantial for the Plain family.

CPU p50 rises by about 56%.

iPhone p50 rises by about 60%.

The all-real difference is small.

### Context against Plain 1.0x

The seed-corrected replay contains a Plain 1.0x reference with the same nominal 150-epoch, batch-128, random360, seed-42 recipe.

| metric | S2-64 | Plain 1.0x corrected reference |
|---|---:|---:|
| dense64 angle mean | 0.963507 | 0.963542 |
| full-class mean | **0.928148** | 0.925630 |
| full-class worst | 0.797778 | **0.806667** |
| Manzu mean | **0.929078** | 0.920567 |
| validity balanced | **0.915894** | 0.887036 |
| all-real accuracy | **0.999567** | 0.999473 |
| CPU p50 | **0.4205 ms** | 0.8673 ms |
| iPhone p50 | **1.60 ms** | 3.395 ms |

The Plain 1.0x row is a cross-Study reference.

The comparison is contextual evidence rather than an isolated same-run causal result.

Worker assignment was heterogeneous.

The baseline and S2-48 trained on RTX 3090.

S1-24, S1-32, and S2-64 trained on RTX 3060 workers.

Bitwise-identical training across heterogeneous GPUs has not been established.

The exact size of small deltas therefore remains hardware-confounded.

The useful directional observation is narrower.

Stage1 widening does not help.

S2-64 is the only early-width condition with a broad observed quality improvement large enough to remain interesting despite a large latency increase.

## Cross-cutting observations

Plain quality does not respond uniformly to channel width.

Adding width at stage1 or stage3 is ineffective in the tested ranges.

Stage2=64 is materially stronger.

The result argues against treating Plain quality as a simple global-width problem.

The 1.0x schedule appears to spend capacity in stages that are not required to recover most of its observed quality.

## Follow-up judgment candidates

- Whether `16/64/64/96` should remain the main pure-Plain quality reference.
- Whether further Plain work is worth its latency cost relative to equivariant models.
- Whether a worker-controlled rerun is needed before using small quality deltas for model selection.

## Recommendation

`16/64/64/96` appears preferable as the current pure-Plain channel-placement reference.

Further broad Plain widening appears lower value than testing whether equivariance can be retained at lower cost.

## Follow-up artifact candidates

- A cyclic-equivariant width-reduction Investigation.
- A later worker-controlled finalist comparison if production selection is opened.

## Open questions

- How much of the S2-64 gain persists under worker-pinned training?
- Can the S2-64 quality level be retained after replacing expensive later operations?
- Does a narrowed equivariant model dominate S2-64 on both robustness and latency?
