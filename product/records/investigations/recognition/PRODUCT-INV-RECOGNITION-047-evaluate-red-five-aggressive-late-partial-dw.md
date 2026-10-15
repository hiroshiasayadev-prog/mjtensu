# PRODUCT-INV-RECOGNITION-047: Evaluate late partial DW/PW on aggressive RED5

- **status**: concluded
- **date**: 2026-10-16
- **trigger**: PRODUCT-INV-RECOGNITION-045 and PRODUCT-INV-RECOGNITION-046 produced a dense stride-aggressive RED5 model with zero audited specialist errors at batch 256 and about 2.62 ms iPhone p50, but its ONNX remained about 3.53 MB because dense stage parameters were unchanged. The next question was whether late convolutions could be factorized after the spatial map had already been reduced to 8x8/4x4.
- **scope**: At aggressive stride, fields 6/12/24/48, batch 256, seed 42, and the accepted RED5 recipe, compare stage4-only versus stages3+4 field-wise DW3x3 + equivariant PW1x1, each with DW intermediate multiplier 1 and 2; run exhaustive quality, ONNX parity/size, CPU latency, and iPhone latency.
- **non_scope**: Earlier-stage factorization; width reduction; color auxiliary features; quantization; threshold tuning; multi-seed confirmation; or production promotion.
- **source_refs**:
  - PRODUCT-INV-RECOGNITION-036
  - PRODUCT-INV-RECOGNITION-037
  - PRODUCT-INV-RECOGNITION-045
  - PRODUCT-INV-RECOGNITION-046
- **follow_up_candidates**:
  - Keep stage3+4 DW/PW x1 as the leading RED5 mobile candidate.
  - Test whether fields 4/8/16/32 can now work under aggressive stride plus dense early mixing.
  - Perform product-level replacement validation before promotion.

## Investigation scope

Reduce RED5 download size after the dense-stride experiments established that fine late spatial resolution is unnecessary.

The dense aggressive reference has 77,852 parameters. Its stage4 full equivariant convolution alone contains 55,296 parameters, so late factorization has large size leverage.

Four candidates were tested:

- stage4 DW/PW, intermediate multiplier 1;
- stage4 DW/PW, intermediate multiplier 2;
- stages3+4 DW/PW, intermediate multiplier 1;
- stages3+4 DW/PW, intermediate multiplier 2.

The early dense stages remain intact. The final field schedule stays 6/12/24/48 and the spatial stride schedule remains aggressive.

## What was investigated

| evidence | ref |
|---|---|
| Study | `tile-classifier/red-five-c8-rgb-aggressive-partial-dw-screen-v1` |
| Study Plan | `tile-classifier/red-five-c8-rgb-aggressive-partial-dw-screen-v1-plan-1f45ab6c393511c5` |
| Study Result | `tile-classifier/run-b18af9873dd84e788890646e4f81f3cc` |
| source commit | `5631351f38498e6d2c0fc09c60ffc383ec3de188` |
| runtime registry | `8` |

Training used batch 256, 50 epochs, seed 42, the accepted rotation/warm-light augmentation, AdamW 0.001 / 0.0001, AMP, and TF32.

## Findings

### Stage3+4 x1 is the clear Pareto candidate

| model | params | specialist errors | ONNX | CPU p50 | iPhone p50 | iPhone p95 |
|---|---:|---:|---:|---:|---:|---:|
| dense aggressive reference | 77,852 | **0** | 3.527 MB | 0.762 ms | 2.620 ms | 2.750 ms |
| stage4 DW/PW x1 | 32,972 | 3 | 1.224 MB | 0.665 ms | 2.175 ms | 2.320 ms |
| stage4 DW/PW x2 | 43,388 | 3 | 1.575 MB | 0.699 ms | 2.240 ms | 2.390 ms |
| **stage3+4 DW/PW x1** | **22,052** | **1** | **0.662 MB** | **0.568 ms** | **1.735 ms** | **1.840 ms** |
| stage3+4 DW/PW x2 | 35,372 | 2 | 1.115 MB | 0.608 ms | 1.865 ms | 2.000 ms |

All four partial-DW candidates retain 241/241 manual and 24/24 warm accuracy.

The x1 intermediate width wins over x2 in both stage-local families. Doubling the DW intermediate fields increases size and latency without improving audited specialist error count.

Unlike the earlier C8-narrow late-DW experiment, factorizing stages 3 and 4 is now successful. The difference is structural: dense early mixing is retained and spatial resolution has already been reduced aggressively before the late factorized blocks.

### Product-scale comparison is unusually favorable

The current production RED5 runtime model is the historical dense Full8 RGB warm-augmentation model:

| metric | current production RED5 | stage3+4 DW/PW x1 |
|---|---:|---:|
| audited specialist errors | 1 | **1** |
| manual | 241/241 | **241/241** |
| warm | 24/24 | **24/24** |
| params | 138,106 | **22,052** |
| ONNX | 6,256,955 bytes | **662,382 bytes** |
| CPU p50 | 3.736 ms | **0.568 ms** |

The candidate is about 9.45x smaller by ONNX bytes and about 6.6x faster in the direct single-thread CPU measurement while matching the production model's observed audited specialist error count.

A separately trained Full8 width-control model with the same dense architecture class measured 14.03 ms iPhone p50. The candidate measures 1.735 ms. This is an architecture-class latency comparison rather than an exact production-weight iPhone replay, but it shows the scale of the mobile reduction.

### Why late DW/PW now works

The combined evidence supports a more specific RED5 requirement than “C8 must remain full-convolution everywhere.”

RED5 appears to require:

- dense early RGB/field mixing;
- enough intermediate width, with 6/12/24/48 sufficient;
- aggressive spatial reduction before the final representation;
- but not dense 3x3 mixing in every late block once the representation is already compact.

This explains why the original all-late-DW width experiments could regress while the new stage3+4 x1 candidate recovers production-class quality.

## Recommendation

Select `stage3+4 DW/PW x1`, fields 6/12/24/48, aggressive stride, batch 256 as the leading RED5 mobile candidate.

Do not select the x2 variants. Their extra intermediate capacity is not justified by quality or latency.

Before production promotion, run product-level replacement validation and confirm that the remaining single audited specialist error does not expose a deployment-specific failure mode.

A final width-reduction experiment at 4/8/16/32 is also reasonable because the old 37-error result was obtained under a materially different topology. The new architecture preserves dense early mixing and may therefore tolerate a narrower field schedule.

## Evidence boundary

The production CPU comparison uses the actual production ONNX artifact under ORT CPU single-thread batch1. The production iPhone ONNX was not successfully replayed end-to-end during this session. The 14.03 ms iPhone reference is a completed Full8 width-control EvaluationResult with the same dense architecture family and essentially the same model size, not the exact production weight file.

Several iPhone evaluations initially failed because the old iPhone infrastructure was unstable. Device trust was repaired, then a separate `iphone-usbmuxd-healer` fault was found repeatedly restarting host usbmuxd and invalidating long runs. Reported latency values above are only from completed MLDB EvaluationResults after the benchmark path was stabilized. This infrastructure issue does not affect the CPU, ONNX, or exhaustive-quality evidence.

## Follow-up judgment candidates

- Evaluate 4/8/16/32 under the new aggressive + dense-early + late-DW topology.
- Run the candidate through the product integration/functional path as a replacement RED5 specialist.
- If the candidate survives product validation, promote its runtime artifact and retire the 6.26 MB historical Full8 model.

## Follow-up artifact candidates

No production promotion is established by this investigation.

## Open questions

- Can 4/8/16/32 retain the one-error quality point under this new topology while reducing the 0.662 MB artifact further?
- Does the candidate preserve behavior on deployment video outside the persisted specialist corpus?
- Is a multi-seed confirmation warranted before final runtime promotion?
