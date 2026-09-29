# PRODUCT-INV-RECOGNITION-016: Evaluate late channel expansion on narrowed Plain CNN

- **status**: concluded
- **date**: 2026-09-29
- **trigger**: PRODUCT-INV-RECOGNITION-015 found channel width to be the strongest Plain-CNN latency lever and suggested restoring lost capacity with a late 1x1 expansion after MaxPool downsampling.
- **scope**: Compare `0.5x` and `0.75x` MaxPool Plain backbones with and without late 1x1 channel expansion against the `1.0x` Plain baseline, including full-class occlusion-margin diagnostics.
- **non_scope**: Production adoption, multi-seed confirmation, iPhone/WASM acceptance, augmentation changes, and downsampling alternatives.
- **source_refs**:
  - PRODUCT-INV-RECOGNITION-015
- **follow_up_candidates**:
  - Sweep late-expansion width around the `0.75x` backbone.
  - Confirm the leading candidate across seeds and iPhone/WASM runtime.
- **follow_up_results**:
  - PRODUCT-INV-RECOGNITION-017

## Investigation scope

Test whether expensive early/spatial channel capacity can be reduced and cheaply restored after the feature map reaches 8x8.

## Out of scope

Production selection, statistical confirmation, iPhone/WASM acceptance, augmentation changes, and alternative downsampling structures.

## Background

INV-015 showed that narrowing Plain CNN channels substantially reduced CPU latency, while excessive narrowing reduced margin and robustness. A late 1x1 expansion was therefore tested as a cheap way to restore channel capacity without restoring expensive high-resolution convolutions.

## What was investigated

| model | late expansion | intent |
|---|---|---|
| Plain `1.0x` | none | baseline |
| Plain `0.5x` | none | aggressive width reduction |
| Plain `0.5x + 1x1` | `96 -> 160` at 8x8 | test low-cost capacity recovery |
| Plain `0.75x` | none | moderate width reduction |
| Plain `0.75x + 1x1` | `144 -> 192` at 8x8 | test low-cost capacity recovery |

All candidates used the same corpus, `random360-only-v1`, 100 epochs, seed 42, and common evaluation protocols. The added full-class occlusion evaluation masked every 8x8 region for every `manual_val` sample and measured the resulting true-class logit-margin drop.

| evidence | ref |
|---|---|
| MLDB Study | `tile-classifier/plain-width-late-expansion-screen-v1` |
| StudyResult | `tile-classifier/run-de1563ab21f7445fbcfa1fbea9e31d83` |
| source commit | `dff4f4cf40d1bbb2b1fbb65b4a20ced314b70ed6` |
| ClearML controller | `6ddbf2f8941b44f1a41bc72f8636f9d4` |

## Findings

| model | CPU p50 | manual angle mean | full front acc. | full margin | worst condition acc. | Manzu margin | Manzu worst | 6m neighbor confusion |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `1.0x` | `0.966 ms` | `0.944` | `0.951` | `6.60` | `0.711` | `4.96` | `0.511` | `0.167` |
| `0.5x` | `0.296 ms` | `0.891` | `0.880` | `4.24` | `0.569` | `1.18` | `0.447` | `1.000` |
| `0.5x + 1x1` | `0.327 ms` | `0.939` | `0.940` | `5.93` | `0.647` | `3.14` | `0.574` | `0.167` |
| `0.75x` | `0.694 ms` | `0.937` | `0.929` | `5.94` | `0.671` | `3.95` | `0.489` | `0.333` |
| `0.75x + 1x1` | `0.704 ms` | `0.966` | `0.964` | `7.45` | `0.742` | `5.38` | `0.617` | `0.000` |

- **Late 1x1 expansion is an efficient capacity-recovery mechanism in this screen.** `0.75x -> 0.75x+1x1` adds about `0.010 ms` p50 latency while materially improving margin and robustness.
- **`0.75x + 1x1` is near-dominant over the `1.0x` Plain baseline in this single-seed Study.** It is about `27%` faster by p50 while improving manual angle mean, full-class accuracy, full/Manzu margin, worst-condition accuracy, and measured 6m neighbor confusion.
- **The same mechanism also rescues much of the `0.5x` degradation.** `0.5x+1x1` remains far faster than baseline and substantially improves the corresponding `0.5x` accuracy and margin results.
- **The gain is not explained by stronger dependence on one masked region.** For `0.75x+1x1`, full-class occlusion peak drop (`6.48` vs `7.72`) and positive mean drop (`0.98` vs `1.20`) are lower than `1.0x`, while effective patch count (`18.51` vs `18.86`) and top-patch share (`0.154` vs `0.157`) remain similar.
- The screen remains single-seed evidence; it does not establish statistical superiority.

## Cross-cutting observations

The result supports allocating less capacity to high-resolution spatial convolutions and adding channel-mixing capacity after downsampling. The occlusion result is consistent with the improved margin not requiring a more spatially concentrated cue, but occlusion is diagnostic rather than causal proof.

## Follow-up judgment candidates

- Whether `0.75x + late 1x1` should replace `1.0x` Plain as the primary architecture candidate.
- Where the useful late-expansion width lies around `160-256` output channels.

## Recommendation

A focused `0.75x` late-expansion-width sweep appears preferable to further stride/downsampling experiments. The leading configuration should then receive multi-seed and iPhone/WASM confirmation before any production-selection judgment.

## Follow-up artifact candidates

- MLDB Study sweeping late 1x1 output width on the `0.75x` MaxPool backbone.
- Production-selection investigation after multi-seed and device-runtime evidence exists.

## Open questions

- Is `192` near the late-expansion capacity knee, or can a smaller expansion retain the same margin gain?
- Does the near-dominant result persist across seeds and iPhone/WASM execution?
