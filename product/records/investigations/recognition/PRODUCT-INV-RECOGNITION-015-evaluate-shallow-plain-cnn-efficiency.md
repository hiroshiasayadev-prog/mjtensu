# PRODUCT-INV-RECOGNITION-015: Evaluate shallow Plain-CNN efficiency

- **status**: concluded
- **date**: 2026-09-29
- **trigger**: Coarse architecture screen requested after Plain random360 showed lower CPU latency than deeper ShuffleNet candidates despite higher MAC count.
- **scope**: Evaluate which shallow Plain-CNN architecture factors materially affect CPU latency, classification margin, and robustness under a controlled MLDB screen.
- **non_scope**: Final production model selection, statistical confirmation across seeds, dense64 acceptance, iPhone runtime acceptance, detector changes, and augmentation changes.
- **source_refs**:
  - PRODUCT-INV-RECOGNITION-008
- **follow_up_candidates**:
  - Narrow channel-width knee around 0.625x to 0.875x while retaining MaxPool.
  - Test late 1x1 expansion on the stronger MaxPool backbone.

## Investigation scope

Compare channel width, downsampling method, delayed downsampling, and cheap late channel expansion while holding corpus, random360 training recipe, evaluation protocols, and CPU latency environment fixed.

## Out of scope

Production adoption, multi-seed statistical confirmation, dense64/iPhone acceptance, detector changes, and augmentation changes.

## Background

Plain random360 measured faster than much lower-MAC ShuffleNet variants on batch-1 ORT CPU. A coarse shallow-CNN sweep was run to identify which architectural factors are worth refining next.

## What was investigated

| factor | conditions | intent |
|---|---|---|
| channel width | `1.0x`, `0.75x`, `0.5x` | reduce compute without adding operators |
| downsampling | MaxPool vs stride-2 Conv | test whether learned stride can replace pooling cheaply |
| delayed downsampling | late-narrow | preserve spatial detail longer |
| late expansion | late-narrow vs late-expand | add capacity only at small spatial size |

All candidates used the same corpus, `random360-only-v1`, 100 epochs, seed 42, and common MLDB evaluations.
CPU latency used `old-gpu3090`, ORT CPU, batch 1, and 1/1 threads.

| evidence | ref |
|---|---|
| MLDB Study | `tile-classifier/plain-shallow-architecture-screen-v1` |
| StudyResult | `tile-classifier/run-e2c05368602845ecb253a9a3c4cc6665` |
| source commit | `29162c5d2fff0c9cf2e16b8b2e73e77a64772944` |
| ClearML controller | `6a952630fa1e4f7bb9ef58a486ebeb9b` |

## Findings

| model | CPU p50 | angle mean | full cond. margin | manzu cond. margin | manzu worst | 6m neighbor confusion |
|---|---:|---:|---:|---:|---:|---:|
| Plain `1.0x` | `0.868 ms` | `0.946` | `5.53` | `4.57` | `0.617` | `0.167` |
| Plain `0.75x` | `0.673 ms` | `0.936` | `4.52` | `3.18` | `0.617` | `0.167` |
| Plain `0.5x` | `0.269 ms` | `0.912` | `3.43` | `1.86` | `0.340` | `0.500` |
| late-narrow | `0.808 ms` | `0.851` | — | — | `0.362` | `0.667` |
| late-expand | `0.826 ms` | `0.941` | — | — | `0.532` | `0.333` |
| `0.75x` strided | `0.615 ms` | `0.818` | — | — | `0.319` | `1.000` |

Findings large enough to affect the next experiment:

- **Channel width is the strongest speed lever.** `0.5x` is about `3.2x` faster than baseline, but margin and worst-case robustness degrade sharply.
- **`0.75x` preserves substantially more robustness.** It is about `22%` faster than baseline and matches baseline Manzu worst accuracy and neighbor-confusion rate in this screen.
- **Stride-2 Conv is not an effective MaxPool replacement here.** At the same `0.75x` width and essentially the same MAC count, robustness drops sharply for only a small latency gain.
- **Late capacity is cheap.** A late `1x1` expansion recovers much of late-narrow's lost accuracy for about `0.02 ms` additional latency.

Findings not useful for selecting the next architecture in this screen:

- JP angle accuracy is saturated at roughly `0.996-0.999`.
- Validity-rejection metrics do not separate candidates consistently.
- Small latency differences such as `0.808` vs `0.826 ms` are negligible relative to the accuracy change.
- Single-seed differences are not statistical significance claims.

## Cross-cutting observations

The measured tradeoff is not explained by MAC count alone. Keeping a shallow graph while reducing width gives large latency gains, whereas changing downsampling semantics can materially damage robustness even when MAC count is unchanged.

## Follow-up judgment candidates

- Whether the useful width knee lies near `0.625x`, `0.75x`, or `0.875x`.
- Whether late expansion can restore margin on a narrowed MaxPool backbone cheaply enough to improve the tradeoff.

## Recommendation

A follow-up screen centered on MaxPool Plain variants appears preferable: `0.625x`, `0.75x`, `0.875x`, plus `0.75x + late 1x1 expansion`. The stride-only and late-narrow branches do not currently appear worth extending.

## Follow-up artifact candidates

- A follow-up MLDB Study for the narrowed width/late-expansion screen.
- A follow-up Investigation if the narrowed screen requires a new architecture judgment.

## Open questions

- Where does the margin/robustness knee occur between `0.5x` and `1.0x`?
- Can late expansion recover margin without materially increasing CPU/WASM latency?
