# PRODUCT-INV-RECOGNITION-043: Evaluate red-five C8 width and convolution form

- **status**: concluded
- **date**: 2026-10-16
- **trigger**: The initial narrow RED5 C8 DW/PW model at fields 4/8/16/32 was small and fast but produced 37 exhaustive specialist errors. It was unclear whether RED5 primarily needed more channel capacity, full cross-field convolution, or both. A controlled width and convolution-form comparison was required before continuing architecture optimization.
- **scope**: Compare RED5 C8 DW/PW fields 6/12/24/48 versus 8/16/32/64, then compare dense full-convolution fields 6/12/24/48 versus 8/16/32/64 under the same RED5 recipe; recover ONNX, CPU, and iPhone latency for the width candidates.
- **non_scope**: Spatial-stride changes; color auxiliary features; batch-size tuning; late-stage selective DW/PW; quantization; multi-seed confirmation; or production promotion.
- **source_refs**:
  - PRODUCT-INV-RECOGNITION-006
  - PRODUCT-INV-RECOGNITION-032
  - PRODUCT-INV-RECOGNITION-036
- **follow_up_candidates**:
  - Keep fields 6/12/24/48 as the RED5 width reference.
  - Preserve dense field/channel mixing while seeking latency reduction through spatial downsampling rather than further widening.
  - Treat 8/16/32/64 as a quality ceiling, not the preferred mobile operating point.

## Investigation scope

Determine why RED5 moved away from the original C8-narrow fields 4/8/16/32 and whether the added capacity should come from width, dense convolution, or both.

C8 field counts are not raw tensor-channel counts. One C8 regular field contains eight tensor channels. Therefore:

| C8 fields | tensor channels |
|---|---|
| 4/8/16/32 | 32/64/128/256 |
| 6/12/24/48 | 48/96/192/384 |
| 8/16/32/64 | 64/128/256/512 |

The 6-field schedule is therefore not an unaligned six-channel implementation. Its actual equivariant tensor widths are all multiples of 16.

## What was investigated

Primary accuracy evidence:

| evidence | ref |
|---|---|
| DW width Study | `tile-classifier/red-five-c8-rgb-dw222-width-screen-v1` |
| DW Study Result | `tile-classifier/run-9c32e61c4b694c63b1c3b0a87e76b8b3` |
| DW source commit | `0d66bf1eca170a8f3f4cda742b68711592c986e1` |
| Full width Study | `tile-classifier/red-five-c8-rgb-full-width-control-v1` |
| Full Study Result | `tile-classifier/run-af5ff3c29adc4dda98bd701b2f941fc2` |
| Full source commit | `a57a8b9b987d87963a2941e2ab6e0c2d2ef2c092` |

The two historical width Studies trained 50 epochs with batch 1024, seed 42, the accepted RED5 warm-light augmentation, residual rotation augmentation, and the audited exhaustive diagnostic.

The original width Studies did not include iPhone latency. Existing accepted models were therefore source-pinned and reused without retraining in two latency-only Studies:

| evidence | ref |
|---|---|
| DW latency Study | `tile-classifier/red-five-c8-rgb-dw222-width-latency-v1` |
| DW latency Study Result | `tile-classifier/run-fbee690b3ed14d7a89f68f6b9179cf2f` |
| Full latency Study | `tile-classifier/red-five-c8-rgb-full-width-latency-v1` |
| Full latency Study Result | `tile-classifier/run-33dd69e125994572989e3b8172566426` |
| latency source commit | `50d9f8775f8bd67d4c01944962ebd8fb43fa33c0` |

The latency Studies used parity-checked ONNX, ORT CPU single-thread batch1, and iPhone Safari ORT Web WASM-SIMD with one thread.

## Findings

### Width alone does not solve the DW/PW regression

| model | params | specialist errors | manual | warm | ONNX | CPU p50 | iPhone p50 |
|---|---:|---:|---:|---:|---:|---:|---:|
| DW 4/8/16/32 historical narrow | 18,286 | 37 | 100% | 100% | — | — | 1.925 ms |
| DW 6/12/24/48 | 35,300 | **8** | 100% | 100% | 1.022 MB | **1.332 ms** | **3.430 ms** |
| DW 8/16/32/64 | 57,690 | 23 | 100% | 100% | 1.703 MB | 1.859 ms | 5.465 ms |

Moving from four to six fields materially improves RED5 quality. Moving from six to eight does not continue that trend; it regresses from 8 to 23 specialist errors while increasing iPhone p50 from 3.43 to 5.47 ms.

The six-field DW model is therefore the best of these DW width points, but its remaining eight errors show that width is not the complete explanation.

### Restoring dense convolution removes most of the error

| model | params | specialist errors | manual | warm | ONNX | CPU p50 | iPhone p50 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Full 6/12/24/48 | 77,852 | **2** | 100% | 100% | 3.527 MB | **2.110 ms** | **8.300 ms** |
| Full 8/16/32/64 | 138,106 | **1** | 100% | 100% | 6.257 MB | 3.788 ms | 14.030 ms |

At the same 6/12/24/48 width, restoring dense equivariant 3x3 convolutions reduces specialist errors from 8 to 2. This is much larger than the benefit from widening the DW model.

The result indicates that RED5 needs cross-field/channel mixing more than it needs the maximum 8/16/32/64 width.

### Why 6/12/24/48 became the reference width

The 6-field schedule is the useful middle point:

- 4/8/16/32 was too narrow in the original all-late-DW topology at 37 errors.
- 6/12/24/48 recovered most of the missing capacity.
- 8/16/32/64 only improves dense Full accuracy from 2 errors to 1 while increasing ONNX from 3.53 MB to 6.26 MB and iPhone p50 from 8.30 ms to 14.03 ms.
- On the DW path, 8/16/32/64 is worse in both quality and latency than 6/12/24/48.

The evidence therefore supports 6/12/24/48 as the RED5 architecture-search reference width rather than treating 8/16/32/64 as a default.

## Recommendation

Keep fields 6/12/24/48 for subsequent RED5 architecture work.

Do not widen to 8/16/32/64 merely to recover the last dense-model error. The mobile latency and download-size cost is disproportionate.

The next optimization should preserve dense early field/channel mixing while reducing spatial work. This motivates a learned-stride downsampling experiment rather than another width sweep.

## Evidence boundary

The 4/8/16/32 latency value comes from the earlier narrow latency recovery and is contextual because its architecture/training lineage predates the two width-control Studies. The 6-versus-8 comparisons are from accepted models in the same width Studies and were re-evaluated without retraining.

The iPhone latency recovery required repairing the old iPhone runner path. During recovery, an independent usbmuxd-healer fault repeatedly restarted usbmuxd and invalidated several attempts. Final reported latency values come only from completed MLDB EvaluationResults after the runner was stabilized.

## Follow-up judgment candidates

- Test dense learned stride downsampling at fields 6/12/24/48.
- Keep the Full8 result as a quality ceiling and production-size reference.
- Avoid more width points until topology changes have been evaluated.

## Follow-up artifact candidates

No production promotion is established by this investigation.

## Open questions

- Can spatial downsampling retain the dense Full6 quality while approaching DW-class latency?
- Once early dense mixing is preserved, can late DW/PW be reintroduced without recreating the original eight-error regression?
