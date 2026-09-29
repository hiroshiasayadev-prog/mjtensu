# PRODUCT-INV-RECOGNITION-019: Ablate Plain stage spatial mixing

- **status**: concluded
- **date**: 2026-09-29
- **trigger**: PRODUCT-INV-RECOGNITION-018 found that widening above the favorable `16/32/64/96` schedule incurs a large CPU latency penalty. The next question was whether selected full 3x3 convolutions could be simplified instead.
- **scope**: Replace selected stage 3x3 convolutions in `0.5x + late256` with ordinary 1x1 convolutions to identify where spatial mixing is necessary.
- **non_scope**: Depthwise-separable replacement, added capacity, width changes, production promotion, and multi-seed confirmation.
- **source_refs**:
  - PRODUCT-INV-RECOGNITION-018
- **follow_up_candidates**:
  - PRODUCT-INV-RECOGNITION-020
- **follow_up_results**:
  - PRODUCT-INV-RECOGNITION-020

## Investigation scope

Keep channels, pooling, late expansion, training, and evaluation fixed.

Remove neighboring-pixel spatial mixing from selected stages by replacing full 3x3 convolutions with ordinary 1x1 convolutions.

## Out of scope

- Depthwise-separable replacement.
- Added capacity.
- Channel-width changes.
- Production model promotion.
- Multi-seed statistical confirmation.

## Background

PRODUCT-INV-RECOGNITION-018 found no attractive intermediate width above `0.5x`.

The remaining speed question was whether all full 3x3 stage convolutions were necessary inside the favorable half-width topology.

## What was investigated

| variant | stage kernels after stem |
|---|---|
| baseline | `3x3 / 3x3 / 3x3` |
| stage2 1x1 | `1x1 / 3x3 / 3x3` |
| stage3 1x1 | `3x3 / 1x1 / 3x3` |
| stage4 1x1 | `3x3 / 3x3 / 1x1` |
| stage3+4 1x1 | `3x3 / 1x1 / 1x1` |

| evidence | ref |
|---|---|
| MLDB Study | `tile-classifier/plain-stage-spatial-mixing-ablation-v1` |
| StudyResult | `tile-classifier/run-61942c2852a94301a983c87e27472017` |
| source commit | `db6d431faf949b8edc752898cf1b07f446e8e2b3` |

## Findings

| variant | CPU p50 | full mean acc. | full worst | full mean margin | Manzu mean | Manzu margin |
|---|---:|---:|---:|---:|---:|---:|
| baseline | `0.302 ms` | `0.867` | `0.673` | `4.68` | `0.867` | `3.11` |
| stage2 `1x1` | `0.239 ms` | `0.860` | `0.656` | `4.31` | `0.818` | `2.46` |
| stage3 `1x1` | `0.243 ms` | `0.806` | `0.582` | `3.07` | `0.708` | `1.16` |
| stage4 `1x1` | `0.258 ms` | `0.842` | `0.640` | `3.42` | `0.801` | `1.94` |
| stage3+4 `1x1` | `0.203 ms` | `0.707` | `0.500` | `1.63` | `0.447` | `-0.29` |

Stage3 and stage4 spatial mixing are important under this topology.

Removing both later 3x3 stages causes the largest quality collapse.

Stage2 is the least damaging location. Its CPU p50 drops by about `0.063 ms`, but Manzu accuracy and margin still fall materially.

## Cross-cutting observations

The fastest architecture is not the most useful one.

Later low-resolution 3x3 stages still carry important discriminative information.

Stage2 is the only tested location where the tradeoff is close enough to justify another experiment.

## Follow-up judgment candidates

- Whether a stage2 depthwise 3x3 can restore spatial filtering more cheaply than the full convolution.
- Whether a pointwise channel mixer after depthwise filtering preserves enough quality.

## Recommendation

A focused stage2 `DW3x3 + PW1x1` substitution appears worth testing.

Further pointwise-only ablations do not appear useful.

## Follow-up artifact candidates

- PRODUCT-INV-RECOGNITION-020 for stage2 depthwise-separable substitution.

## Open questions

- Can depthwise spatial filtering recover the stage2 1x1 quality loss?
- Is the remaining latency gain large enough to justify the factorized operator sequence?
