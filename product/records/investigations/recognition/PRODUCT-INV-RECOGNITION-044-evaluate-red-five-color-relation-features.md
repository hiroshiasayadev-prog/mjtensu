# PRODUCT-INV-RECOGNITION-044: Evaluate explicit red-color relation features for RED5 DW/PW

- **status**: concluded
- **date**: 2026-10-16
- **trigger**: PRODUCT-INV-RECOGNITION-043 showed that RED5 DW 6/12/24/48 had eight specialist errors while dense Full6 had only two. The DW6 errors were all false-normal misses, suggesting that field-wise factorization might be discarding useful cross-channel color relationships. An explicit redness feature was tested as a cheaper way to restore that signal without returning to full convolution.
- **scope**: Add one model-internal color-relation map to raw RGB before the same C8 DW2/2/2 fields 6/12/24/48 backbone; compare Cr, normalized red, ExR, and NRG under the accepted RED5 recipe and full deployment evaluation.
- **non_scope**: Threshold tuning; learned color transforms; topology changes outside the input feature; stride changes; width changes; quantization; or production promotion.
- **source_refs**:
  - PRODUCT-INV-RECOGNITION-006
  - PRODUCT-INV-RECOGNITION-043
- **follow_up_candidates**:
  - Do not continue hand-designed redness-feature tuning unless a later failure analysis specifically points back to input color representation.
  - Prefer topology changes that preserve cross-field mixing directly.

## Investigation scope

Test whether an explicit per-pixel redness/color-relation channel can recover the RED5 signal lost by the DW/PW backbone.

All candidates keep raw RGB as the external interface. The auxiliary map is generated inside the model after augmentation and is normalized together with the image-derived input fields.

The four feature definitions were:

- Cr-like chroma: `0.5R - 0.418688G - 0.081312B + 0.5`, clamped to `[0,1]`.
- normalized red: `R / (R + G + B)`.
- ExR: `1.4r - g` using normalized RGB chromaticity.
- NRG: `(R - G) / (R + G)`.

## What was investigated

Accuracy evidence:

| evidence | ref |
|---|---|
| feature screen Study | `tile-classifier/red-five-c8-rgb-redfeature-dw222-screen-v1` |
| Study Result | `tile-classifier/run-6f8ac329ba55471795484d1e81d3fd16` |
| source commit | `29dce44d9c6b93e40fa5240da851087290d819d5` |

Deployment evidence:

| evidence | ref |
|---|---|
| full-eval Study | `tile-classifier/red-five-c8-rgb-redfeature-dw222-full-eval-v1` |
| Study Result | `tile-classifier/run-08ac11bc2db84ed1bf06665dabfa43b5` |
| source commit | `20711a493a74ef1e661d190adc9c9879bc2c1b17` |

All models used fields 6/12/24/48, DW multiplier 2 in stages 2-4, batch 1024, 50 epochs, seed 42, and the accepted warm-light/rotation training recipe.

## Findings

### Cr verifies the color-signal hypothesis, but does not solve the model-selection problem

| feature | specialist errors | false-normal rate | false-red rate | manual | warm |
|---|---:|---:|---:|---:|---:|
| RGB DW6 baseline | 8 | 0.00027654 | 0 | 100% | 100% |
| RGB + Cr | **6** | **0** | 0.00006885 | 100% | 100% |
| RGB + normalized red | 14 | 0.00013827 | 0.00011474 | 100% | 100% |
| RGB + ExR | 14 | 0.00010370 | 0.00012622 | 100% | 100% |
| RGB + NRG | 24 | 0.00006913 | 0.00025244 | 100% | 100% |

Cr removes the false-normal failure mode from the original DW6 result. All eight original DW6 misses become red-positive under the Cr candidate, including the difficult `jp:train:601330` example.

Cr nevertheless introduces six normal-to-red false positives. It therefore supports the hypothesis that the factorized network lost useful redness information, but hand-injecting that signal only moves the error boundary rather than restoring the dense Full6 behavior.

The other three hand-designed features are worse on the audited specialist metric.

### Color features do not buy mobile speed

| feature | ONNX | CPU p50 | iPhone p50 |
|---|---:|---:|---:|
| RGB DW6 baseline | 1.022 MB | 1.332 ms | **3.430 ms** |
| RGB + Cr | 1.029 MB | 1.420 ms | 3.815 ms |
| RGB + normalized red | 1.028 MB | 1.337 ms | 3.790 ms |
| RGB + ExR | 1.029 MB | **1.331 ms** | **3.775 ms** |
| RGB + NRG | 1.028 MB | 1.346 ms | 3.795 ms |

The added feature channel leaves model size roughly unchanged but increases iPhone latency relative to the RGB DW6 baseline. None approaches dense Full6 quality.

### The normalized-red exhaustive failure was infrastructure, not missing evidence

The later full-eval Study's normalized-red exhaustive stage failed at its original large evaluation batch because of CUDA memory pressure. The earlier completed feature-screen Study already contains a successful audited exhaustive result for the same accepted trained candidate: 14 specialist errors with manual and warm accuracy both 100%.

The deployment stages later completed, including ONNX parity, CPU latency, and recovered iPhone latency.

## Recommendation

Do not select a hand-designed redness auxiliary channel as the main RED5 continuation path.

Retain Cr as evidence that color/channel relationships matter. Address that requirement structurally by preserving dense early mixing rather than adding a fourth handcrafted input feature.

## Evidence boundary

The full-eval Study retains `completed_with_failures` because the normalized-red exhaustive attempt failed at its original evaluation batch. Accuracy for that candidate is taken from the completed earlier feature-screen Study, while its ONNX/CPU/iPhone measurements come from the later deployment Study.

Recovered iPhone values are from completed MLDB EvaluationResults after device trust was restored.

## Follow-up judgment candidates

- Preserve RGB-only input for the next architecture screen.
- Test spatial downsampling while retaining dense equivariant convolution.

## Follow-up artifact candidates

No production promotion is established by this investigation.

## Open questions

- How much of the original DW regression is specifically caused by loss of cross-field mixing versus loss of late spatial mixing?
- Can dense early mixing plus aggressive spatial reduction make explicit color features unnecessary?
