# PRODUCT-INV-RECOGNITION-017: Refine half-width Plain CNN capacity recovery and device latency

- **status**: concluded
- **date**: 2026-09-30
- **trigger**: PRODUCT-INV-RECOGNITION-016 showed that late channel expansion can recover much of the robustness lost by narrowing the Plain CNN. Follow-up work then explored where the useful width/capacity knee lies, whether expensive spatial convolutions can be simplified, and whether the resulting fast Plain candidates remain attractive on the actual iPhone ORT Web runtime.
- **scope**: Consolidate the post-INV-016 Plain-CNN experiments covering spatial-width/late-channel sweeps, fine width refinement, spatial-mixing ablation, depthwise-separable substitution, late spatial refinement, and corrected iPhone latency measurement.
- **non_scope**: Production model promotion, detector changes, augmentation changes, multi-seed statistical confirmation, quantization, and further architecture micro-optimization beyond the tested family.
- **source_refs**:
  - PRODUCT-INV-RECOGNITION-015
  - PRODUCT-INV-RECOGNITION-016
- **decision**: Retain `tile-plain-gray35-w500-late256-late-dw3-pw1-v1` as the leading fast Plain-CNN candidate from this line of investigation. Stop speed-first micro-optimization for now; the remaining measured quality gap to the 1.0x Plain reference is small relative to the latency reduction.

## Investigation question

Can the aggressively narrowed `0.5x` Plain backbone retain approximately 1.0x-Plain classification quality by spending capacity only at low spatial resolution, while preserving the large CPU and iPhone latency advantage?

The relevant fast backbone is:

```text
64x64: 1  -> 16   Conv5x5, pool
32x32: 16 -> 32   Conv3x3, pool
16x16: 32 -> 64   Conv3x3, pool
 8x8 : 64 -> 96   Conv3x3
 8x8 : 96 -> 256  late PW1x1
GAP -> Linear 256 -> 256 -> 35
```

The strongest refinement found in this investigation keeps that outer width schedule unchanged and adds only:

```text
8x8: 96 -> 96   DW3x3
     96 -> 256  PW1x1
```

immediately before global pooling.

## Canonical experiment trail

| experiment | Study | StudyResult | source commit | purpose |
|---|---|---|---|---|
| spatial-width × late-channel grid | `tile-classifier/plain-spatial-late-channel-grid-screen-v1` | `tile-classifier/run-1d73b84383734a2c9e8e9a282056bcaf` | `d2431e2a717393def4b9cb8835cd112c9a4c69e5` | locate useful width/late-capacity regions |
| fine spatial-width screen | `tile-classifier/plain-spatial-width-fine-screen-v1` | `tile-classifier/run-bc6fe4f59d1946daa7dbd95f8ac85d1c` | `be302e6fa078f030d7f7342e14d31ef4f676e27b` | refine the `0.5x` to `0.625x` region and characterize latency discontinuity |
| stage spatial-mixing ablation | `tile-classifier/plain-stage-spatial-mixing-ablation-v1` | `tile-classifier/run-61942c2852a94301a983c87e27472017` | `db6d431faf949b8edc752898cf1b07f446e8e2b3` | test whether stage 3x3 spatial mixing can be removed |
| stage2 depthwise-separable replacement | `tile-classifier/plain-stage2-depthwise-separable-screen-v1` | `tile-classifier/run-93c70cce7fae4c05b4d211326c147130` | `e2b6b03ed095d43bc9fb784604ad5e7a4fd2f02e` | restore cheap spatial mixing after the stage2 1x1 ablation |
| capacity-recovery screen | `tile-classifier/plain-capacity-recovery-screen-v1` | `tile-classifier/run-01fc07b6027d4f00a1c8dd15119779ce` | `6dd26897dc99d0ceea6b69b095af7104496f9d99` | compare late DW refinement with temporary stage3 expansion |
| corrected iPhone latency review | `tile-classifier/plain-capacity-recovery-iphone-latency-v2-review-v1` | `tile-classifier/run-bde5ea6da58047eeab6760c1eac402e1` | `0d5bb0c7d6e57f360e30787763591598c49c99a1` | re-evaluate the same trained models with higher-resolution Safari timing |

Unless otherwise noted, architecture screens used the same `gray35-jp500-seed42-v3-jp189-v1` corpus, `random360-only-v1`, 100 epochs, seed 42, batch 512, AdamW `lr=0.001`, weight decay `0.0001`, AMP, and TF32. Results therefore isolate architecture changes reasonably well, but remain single-seed evidence.

## 1. Spatial width and late capacity

The broad grid confirmed that late channel capacity is much cheaper than restoring width throughout the spatial backbone.

| representative model | CPU p50 | angle mean | full mean condition acc. | full worst condition acc. | full mean margin |
|---|---:|---:|---:|---:|---:|
| Plain `1.0x` | `0.873 ms` | `0.946` | `0.901` | `0.749` | `5.65` |
| `0.5x` | `0.275 ms` | `0.905` | `0.808` | `0.560` | `3.04` |
| `0.5x + late192` | `0.370 ms` | `0.938` | `0.873` | `0.687` | `4.78` |
| `0.5x + late224` | `0.327 ms` | `0.942` | `0.881` | `0.684` | `5.00` |
| `0.5x + late256` | `0.331 ms` | `0.945` | `0.885` | `0.709` | `5.17` |
| `0.625x + late192` | `0.697 ms` | `0.958` | `0.908` | `0.756` | `5.77` |
| `0.75x + late192` | `0.707 ms` | `0.962` | `0.909` | `0.749` | `6.06` |
| `0.875x + late224` | `0.962 ms` | `0.966` | `0.925` | `0.784` | `6.54` |

The quality ceiling continues to rise with wider spatial backbones, but the CPU cost rises much faster than the late-expansion cost. `0.5x + late256` therefore became the main efficiency baseline rather than an attempt to maximize absolute single-seed accuracy.

## 2. Fine width refinement exposed a runtime cliff

The fine screen tested small increments above the aligned `0.5x` width schedule while keeping late256 fixed.

| model | representative channel schedule | CPU p50 | full mean acc. | full worst | Manzu mean |
|---|---|---:|---:|---:|---:|
| `0.5x + late256` | `16/32/64/96` | `0.302 ms` | `0.889` | `0.722` | `0.882` |
| `0.53125x + late256` | `17/34/68/102` | `0.541 ms` | `0.875` | `0.700` | `0.861` |
| `0.5625x + late256` | wider than 0.53125x | `0.580 ms` | `0.887` | `0.693` | `0.871` |
| `0.59375x + late256` | `19/38/76/114` | `0.605 ms` | `0.902` | `0.733` | `0.901` |
| `0.625x + late256` | wider again | `0.711 ms` | `0.896` | `0.747` | `0.855` |

A very small width increase from `16/32/64/96` to `17/34/68/102` nearly doubled measured CPU latency without a corresponding quality gain. This strongly suggests backend/kernel shape effects matter more than smooth MAC-count scaling in this range. The exact ORT/MLAS blocking mechanism was not profiled, so the investigation does not claim a specific alignment rule.

The practical conclusion is narrower: the `16/32/64/96` schedule is unusually favorable on the measured ORT CPU path and should not be casually widened one channel group at a time.

## 3. Removing spatial mixing was fast but destructive

The stage ablation kept the `0.5x + late256` channel schedule fixed and replaced selected full 3x3 convolutions with ordinary pointwise 1x1 convolutions. The replacement preserved channel mixing but removed neighboring-pixel spatial mixing at that stage.

| variant | CPU p50 | full mean acc. | full worst | full mean margin | Manzu mean | Manzu margin |
|---|---:|---:|---:|---:|---:|---:|
| baseline `3x3/3x3/3x3` | `0.302 ms` | `0.867` | `0.673` | `4.68` | `0.867` | `3.11` |
| stage2 `1x1` | `0.239 ms` | `0.860` | `0.656` | `4.31` | `0.818` | `2.46` |
| stage3 `1x1` | `0.243 ms` | `0.806` | `0.582` | `3.07` | `0.708` | `1.16` |
| stage4 `1x1` | `0.258 ms` | `0.842` | `0.640` | `3.42` | `0.801` | `1.94` |
| stage3+4 `1x1` | `0.203 ms` | `0.707` | `0.500` | `1.63` | `0.447` | `-0.29` |

The 16x16 and 8x8 spatial stages are not redundant. Removing both late spatial convolutions is especially destructive. Stage2 is the least damaging place to remove a 3x3, but even there the speed gain comes with a meaningful Manzu robustness/margin loss.

This changed the optimization goal from “remove more 3x3 work” to “preserve the fast 0.5x backbone and restore capacity cheaply after downsampling.”

## 4. Replacing stage2 with DW3x3 + PW1x1 did not recover enough quality

A MobileNet-style stage2 replacement restored per-channel 3x3 spatial filtering while retaining a pointwise 16->32 channel mixer.

| model | CPU p50 | angle mean | full mean acc. | full worst | full mean margin | Manzu mean | Manzu worst |
|---|---:|---:|---:|---:|---:|---:|---:|
| `0.5x + late256` | `0.302 ms` | `0.949` | `0.886` | `0.687` | `5.18` | `0.840` | `0.511` |
| stage2 `DW3x3 + PW1x1` | `0.250 ms` | `0.936` | `0.863` | `0.662` | `4.05` | `0.799` | `0.574` |

The depthwise-separable replacement is faster, but it does not consistently restore the quality lost by factorizing the full convolution. Some worst-case behavior improves relative to the pure stage2 1x1 ablation, but global accuracy and margin remain below the unchanged full-convolution backbone.

The result does not support replacing the existing stage2 full 3x3 merely to save another roughly `0.05 ms` on CPU.

## 5. Late depthwise spatial refinement was the useful recovery mechanism

The capacity-recovery screen stopped replacing spatial convolutions and instead added inexpensive spatial processing after the feature map had already reached 8x8.

Three conditions were compared:

```text
baseline:
  ... -> Conv3x3 64->96 -> PW1x1 96->256 -> GAP

late DW refinement:
  ... -> Conv3x3 64->96 -> DW3x3 96->96 -> PW1x1 96->256 -> GAP

stage3 temporary expansion:
  16x16: DW3x3 32->32 -> PW1x1 32->128
   8x8 : PW1x1 128->64 -> Conv3x3 64->96 -> PW1x1 96->256
```

| model | CPU p50 | angle mean | full front | full mean | full worst | full mean margin | Manzu mean | Manzu worst | Manzu margin |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `0.5x + late256` | `0.309 ms` | `0.952` | `0.956` | `0.881` | `0.680` | `4.83` | `0.845` | `0.468` | `2.89` |
| `0.5x + late DW3x3 + PW256` | `0.316 ms` | `0.962` | `0.967` | `0.906` | `0.702` | `6.23` | `0.888` | `0.596` | `5.27` |
| stage3 expand128/compress64 | `0.274 ms` | `0.943` | `0.942` | `0.867` | `0.673` | `4.69` | `0.861` | `0.660` | `3.10` |

The late DW refinement adds only about `0.007 ms` CPU p50 in this run while improving every listed global quality/margin metric over the same-run `0.5x + late256` baseline. The temporary 128-channel stage3 expansion is faster on CPU and has a strong Manzu worst result, but it does not recover global accuracy/margin as effectively.

This is the strongest architecture result from the post-INV-016 sequence: cheap spatial refinement is useful when **added at 8x8**, whereas factorizing/replacing earlier full spatial convolutions is not.

## 6. iPhone latency v1 was too coarsely quantized for this model family

The original `tile-shape-ort-web-iphone-latency-v1` protocol measured each awaited `InferenceSession.run` separately with `performance.now()`. On the dedicated iPhone environment, the browser was not a secure or cross-origin-isolated context, and observed samples collapsed almost entirely to integer-millisecond buckets.

For the capacity-recovery run, raw v1 samples were approximately:

| model | dominant raw samples | v1 p50 | v1 mean |
|---|---|---:|---:|
| baseline | mostly `1 ms`, occasional `2 ms` | `1.0 ms` | `1.08 ms` |
| late DW refinement | roughly `1 ms` / `2 ms`, more `2 ms` | `2.0 ms` | `1.585 ms` |
| stage3 expand128 | mostly `1 ms`, some `0 ms` | `1.0 ms` | `0.97 ms` |

The `0 ms` observations prove that single-run timing resolution was insufficient for sub-ms/low-ms architecture ranking. The v1 values remain historical measurements, but they should not be used for fine-grained comparison inside this fast Plain family.

A sealed v2 protocol was therefore created rather than modifying v1. It times `100` sequential awaited runs as one block, divides block elapsed time by `100`, repeats that for `40` blocks, and computes p50/p95/mean over the per-run block averages.

## 7. Corrected iPhone latency shows the late refinement cost is small

The three already-trained capacity-recovery models were re-evaluated without retraining using `tile-shape-ort-web-iphone-latency-v2`.

| model | iPhone v2 p50 | p95 | mean | relative p50 vs baseline |
|---|---:|---:|---:|---:|
| `0.5x + late256` | `1.07 ms` | `1.11 ms` | `1.076 ms` | baseline |
| `0.5x + late DW3x3 + PW256` | `1.12 ms` | `1.17 ms` | `1.131 ms` | about `+4.7%` |
| stage3 expand128/compress64 | `0.95 ms` | `1.00 ms` | `0.959 ms` | about `-11%` |

The old apparent `1 ms -> 2 ms` p50 regression for late DW refinement was therefore primarily a timer-quantization artifact. The corrected device cost is about `0.05 ms` p50 relative to the same-run baseline, which is small compared with the quality/margin improvement.

The result also reinforces that arithmetic/MAC count alone is not a reliable predictor of ORT Web/WASM latency. The stage3 expanded variant is faster than the simpler baseline on this device despite its extra pointwise operators.

## 8. Comparison with the original 1.0x Plain random360 reference

The closest existing 1.0x Plain reference is the INV-016 baseline from `tile-classifier/run-de1563ab21f7445fbcfa1fbea9e31d83`. It used the same corpus family, random360 recipe, 100 epochs, and seed 42, but it is a **different Study/training execution**, so small differences must not be treated as statistical superiority.

| metric | Plain `1.0x` reference | `0.5x + late DW refinement` | observation |
|---|---:|---:|---|
| CPU p50 | `0.966 ms` | `0.316 ms` | about `3.1x` lower latency |
| manual angle mean | `0.944` | `0.962` | no observed degradation |
| full front accuracy | `0.951` | `0.967` | no observed degradation |
| full mean condition accuracy | `0.882` | `0.906` | no observed degradation |
| full worst condition accuracy | `0.711` | `0.702` | about `-0.9` percentage point |
| full mean condition margin | `5.12` | `6.23` | no observed degradation |
| Manzu mean condition accuracy | `0.868` | `0.888` | no observed degradation |
| Manzu worst condition accuracy | `0.511` | `0.596` | no observed degradation |

Within the current single-seed evidence, the only listed metric lower than the 1.0x reference is full worst-condition accuracy, and the difference is small (`0.711 -> 0.702`). This does **not** prove that the half-width model is statistically better than 1.0x; it does show that the originally feared large quality loss from `0.5x` is not present after late capacity/spatial recovery.

## Cross-cutting conclusions

The post-INV-016 experiments support a consistent architectural picture. Width reduction is still the dominant latency lever, but blindly removing full spatial convolutions damages the classifier. The favorable `16/32/64/96` channel schedule should be preserved because slightly wider schedules exhibit a large measured CPU latency discontinuity. Capacity added only after reaching 8x8 is comparatively cheap. In particular, adding a depthwise 3x3 refinement before the existing late 96->256 pointwise expansion materially improves margin and robustness for almost no CPU cost and only a small corrected iPhone latency increase.

The investigation therefore does not justify further speed-first simplification at this time. The current `0.5x + late DW3x3 + PW256` model is already around three times faster than the historical 1.0x Plain CPU reference while showing only a small observed worst-condition difference and no broad degradation in the other recorded quality metrics.

## Decision

`tile-classifier/tile-plain-gray35-w500-late256-late-dw3-pw1-v1` is the leading Plain-CNN candidate produced by this investigation line.

Further architecture work should be driven by a concrete deployment failure or a measured quality deficit rather than by an abstract goal of reducing another few hundredths of a millisecond. The stage3-expand128 branch and earlier 1x1/depthwise-separable replacement branches remain useful negative/diagnostic evidence but are not preferred continuation paths.

This decision is **not production promotion**. The currently deployed classifier remains unchanged. If a later promotion decision is opened, it should use the corrected iPhone latency protocol and may add multi-seed confirmation if the expected deployment risk warrants it.

## Closed questions

The useful width knee is not a smooth scalar width between `0.5x` and `0.625x`; measured runtime strongly favors the aligned `16/32/64/96` schedule. Early/stage spatial mixing cannot be removed cheaply enough with plain 1x1 or the tested stage2 DW+PW replacement. Late spatial refinement at 8x8 is a better use of compute. The corrected iPhone measurement shows that its runtime penalty is small enough not to change the architecture judgment.
