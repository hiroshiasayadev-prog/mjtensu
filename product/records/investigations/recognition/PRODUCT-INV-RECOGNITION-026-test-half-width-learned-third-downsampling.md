# PRODUCT-INV-RECOGNITION-026: Test learned third downsampling in half-width Plain classifier

- **status**: in_progress
- **date**: 2026-10-04
- **trigger**: The original trigger was a concentrated 5p/7p/8p failure pattern observed in a v5-trained half-width Plain model. PRODUCT-INV-RECOGNITION-027 later established that the v5 comparison had uncontrolled Architecture initialization, so the original failure pattern remains historical motivation rather than valid causal evidence.
- **scope**: Preserve the 16/32/64/96 spatial channel schedule and the selected late DW3x3/PW256 tail, then test whether replacing only the third 16x16->8x8 MaxPool with learned depthwise or depthwise-separable stride-2 downsampling improves seed-corrected robustness and exhaustive real-crop recall.
- **non_scope**: Widening above 0.5x, changing the training Corpus, changing random360, moving other pooling stages, changing the late 96->256 capacity, production promotion, multi-seed confirmation, or proving a general theory of pooling.
- **source_refs**:
  - PRODUCT-INV-RECOGNITION-018
  - PRODUCT-INV-RECOGNITION-019
  - PRODUCT-INV-RECOGNITION-020
  - PRODUCT-INV-RECOGNITION-021
  - PRODUCT-INV-RECOGNITION-025
  - PRODUCT-INV-RECOGNITION-027
- **follow_up_candidates**:
  - Keep the seed-corrected DW+PW learned-downsample candidate as a low-cost Plain reference.
  - Revisit channel placement separately rather than attributing the historical pinzu collapse to MaxPool.

## Investigation scope

Test whether a learned third downsampling operator improves the selected half-width Plain classifier.

Keep the nominal 16/32/64/96 spatial-width schedule unchanged.

Keep the late 8x8 DW3x3 refinement and PW1x1 96->256 expansion unchanged.

## Out of scope

- Widening above the 0.5x spatial schedule.
- Changing the training Corpus.
- Changing random360.
- Moving the first or second pooling stage.
- Changing late 96->256 capacity.
- Multi-seed confirmation.
- Production promotion.
- A general conclusion about MaxPool.

## Background

PRODUCT-INV-RECOGNITION-025 originally exposed a severe 5p/7p/8p failure pattern in the selected half-width Plain model.

The original pattern suggested that local circle-like features survived while whole-tile layout did not.

That evidence motivated a test of the third 16x16->8x8 downsampling operation.

PRODUCT-INV-RECOGNITION-027 later found a seed bug in the v5 training lifecycle.

The declared seed was applied after Architecture construction.

The original failure comparison therefore cannot support architecture-causal conclusions.

The present record keeps the original question open and uses the v6 replay as the corrected evidence.

## What was investigated

Three downsampling conditions were compared.

| condition | third 16x16 -> 8x8 downsampling |
|---|---|
| baseline | MaxPool2d(2,2) |
| learned-DW | DW3x3 stride2, 64 -> 64 |
| learned-DW+PW | DW3x3 stride2, 64 -> 64, then PW1x1 64 -> 64 |

All three retain:

- Spatial widths 16/32/64/96.
- 8x8 DW3x3 96 -> 96 refinement.
- Late PW1x1 96 -> 256 expansion.
- GAP.
- 256-unit classifier head.

Training controls were:

- Corpus: `tile-classifier/gray35-jp500-seed42-v3-jp189-v1`.
- Train Protocol: `tile-classifier/tile-shape-train-gpu-v6` for corrected evidence.
- Epochs: 150.
- Batch: 128.
- Learning rate: 0.001.
- Weight decay: 0.0001.
- Augmentation: `random360-only-v1`.
- Seed: 42.
- AMP: enabled.
- TF32: enabled.

Historical v5 evidence:

| evidence | ref |
|---|---|
| Study | `tile-classifier/plain-w500-third-downsample-failure-recovery-v1` |
| Study Plan | `tile-classifier/plain-w500-third-downsample-failure-recovery-v1-plan-acc15ab0757679ce` |
| Study Result | `tile-classifier/run-b49bb08b07964cd1a2e562fc1ca26971` |
| source commit | `5ee2e1587ecae4b477accfeeb2f6b2fa800978a4` |

The historical v5 Study remains traceability evidence only.

Seed-corrected evidence comes from the consolidated replay:

| evidence | ref |
|---|---|
| Study | `tile-classifier/v6-replay-all-v5-affected-conditions-v1` |
| Study Plan | `tile-classifier/v6-replay-all-v5-affected-conditions-v1-plan-c5a2cec921f8f82b` |
| Study Result | `tile-classifier/run-869049bb06b64f8b8db36b6a7d824309` |
| source commit | `a0162c0a22d0e2b5df89cb0fe4f4ca1a035c7fa5` |
| runtime registry | `8` |
| baseline trial | `trial-0039` |
| learned-DW trial | `trial-0040` |
| learned-DW+PW trial | `trial-0041` |

## Findings

Seed-corrected results are:

| metric | baseline MaxPool | learned-DW | learned-DW+PW |
|---|---:|---:|---:|
| dense64 manual-angle mean | 0.974792 | 0.976181 | **0.981042** |
| full-class mean accuracy | 0.931704 | 0.921037 | **0.936741** |
| full-class worst accuracy | 0.760000 | 0.742222 | **0.811111** |
| Manzu mean accuracy | 0.916312 | 0.907801 | **0.917731** |
| Manzu worst accuracy | 0.680851 | 0.553191 | **0.702128** |
| validity balanced accuracy | 0.933911 | **0.935108** | **0.935108** |
| all-real accuracy | 0.999818 | 0.999759 | **0.999834** |
| all-real worst-class recall | 0.998471 | 0.998583 | **0.999067** |
| CPU p50 | **0.3049 ms** | 0.3059 ms | 0.3150 ms |
| iPhone p50 | 1.20 ms | 1.24 ms | **1.19 ms** |

The corrected replay does not reproduce the historical catastrophic pinzu collapse.

Every corrected condition has all-real worst-class recall above 0.998.

The historical collapse is not a stable property of the half-width topology.

The learned-DW condition is not attractive.

The condition is approximately latency-neutral and loses full-class and Manzu robustness.

The learned-DW+PW condition is more useful.

The condition improves dense-angle mean, full-class mean, full-class worst, Manzu mean, Manzu worst, validity balanced accuracy, and all-real accuracy.

CPU p50 changes from about 0.305 ms to 0.315 ms.

Measured iPhone p50 is effectively unchanged.

The result does not support the original hypothesis that fixed third-stage MaxPool caused the historical severe failure.

Worker assignment was heterogeneous.

The corrected baseline and learned-DW+PW trials trained on RTX 3090.

The learned-DW trial trained on RTX 3060.

Bitwise-identical training across heterogeneous GPUs has not been established.

Small deltas therefore remain hardware-confounded.

## Cross-cutting observations

The seed correction changes the interpretation more than the downsampling operator does.

The original severe class-localized failure was not reproducible.

DW-only learned downsampling does not help.

DW+PW learned downsampling is a modest quality improvement at small deployment cost.

The evidence does not identify third-stage pooling as the dominant Plain bottleneck.

## Follow-up judgment candidates

- Whether the learned-DW+PW candidate should remain a Plain reference.
- Whether channel placement is a more useful next question than downsampling replacement.
- Whether a worker-pinned confirmation is needed before using small deltas for selection.

## Recommendation

The learned-DW+PW condition appears preferable to the DW-only condition.

The result does not justify treating MaxPool replacement as the main Plain architecture direction.

Channel-placement experiments appear more informative.

## Follow-up artifact candidates

- A separate channel-placement Investigation.
- A worker-controlled finalist comparison if the learned-DW+PW candidate remains relevant.

## Open questions

- Does the learned-DW+PW advantage persist under worker-pinned training?
- Does the candidate remain competitive after stronger Plain channel placement is considered?
- Is the remaining robustness gap primarily architectural rather than downsampling-related?
- Should this Investigation be concluded after a later lifecycle judgment?
