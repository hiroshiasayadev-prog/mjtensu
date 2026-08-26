# PRODUCT-INV-RECOGNITION-002: Validate NanoDet PWA runtime on iPhone 13

- **status**: investigating
- **date**: 2026-08-04
- **trigger**: PRODUCT-ADR-RECOGNITION-001 assigns one recognition frame every 100 milliseconds, but the selected tile-region detector has not been measured in an iPhone Safari PWA.
- **scope**: Validate whether the epoch-40 NanoDet-Plus-m 320 single-class detector can execute near 10 Hz in an iPhone 13 PWA while returning correctly aligned camera-preview bounding boxes.
- **non_scope**: Tile classification, tile ordering, hand interpretation, point calculation, temporal stabilization, final capture UI, and the three-region composite input are excluded.
- **source_refs**:
  - PRODUCT-ADR-RECOGNITION-001
  - PRODUCT-ADR-RECOGNITION-002
  - PRODUCT-INV-RECOGNITION-001
  - tools/recognition/nanodet/configs/e1_nanodet_plus_m_320_stage40_amp_resume.yml
  - tools/recognition/nanodet/export_and_validate_e1_320_onnx.py
  - tools/recognition/pwa_detector_probe
- **follow_up_candidates**:
  - Validate the fixed completed-hand, dora-indicator, and meld composite input from PRODUCT-ADR-RECOGNITION-002 after the full-frame runtime probe passes.
- **related_adrs**:
  - PRODUCT-ADR-RECOGNITION-001
  - PRODUCT-ADR-RECOGNITION-002

## Investigation scope

This investigation measures the first deployment boundary of the staged recognition pipeline:

```text
rear camera
  -> full camera frame
  -> 320 x 320 detector input
  -> NanoDet browser inference
  -> confidence filtering and NMS
  -> camera-preview bounding-box overlay
```

The initial measurement uses the complete camera frame.
It deliberately precedes the semantic three-region capture layout adopted by PRODUCT-ADR-RECOGNITION-002 so that model execution cost, frame scheduling, output decoding, and coordinate mapping can be established without additional composition logic.

The target device is iPhone 13 running Safari or an installed PWA using the same WebKit runtime.
The target detector cadence is one request every 100 milliseconds.

## Out of scope

- Mahjong tile classification.
- Japanese tile-type recognition.
- Tile ordering or meld grouping.
- Hand-validity checks.
- Yaku, fu, han, or point calculation.
- Temporal stabilization.
- Capture confirmation behavior.
- The completed-hand, dora-indicator, and meld region layout.
- Region enable and disable controls.
- Black-fill composition for disabled regions.
- Final visual design.

## Model and export contract

Use the epoch-40 `model_best` artifact from:

```text
/srv/data/mjtensu/.local/recognition/nanodet_runs/
E1_plus_m_320_amp30_seed42
```

The model condition is:

| property | value |
|---|---|
| architecture | NanoDet-Plus-m |
| detector input | `320 x 320` |
| classes | One `mahjong_tile` class |
| training epoch | 40 |
| validation AP at maximum 200 detections | 0.975 |
| AP50 | 0.990 |
| AP75 | 0.990 |

Before export, inventory the actual `model_best` file structure and record artifact sizes and SHA-256 values.
Do not infer the checkpoint filename from the run-directory name.

Use NanoDet v1.0.0 at commit `d3fb34fa91d6020f273d6d063bf324dcd97bac12` and its own `tools/export_onnx.py`.
The exported ONNX input must be fixed to `[1, 3, 320, 320]`.

The NanoDet-Plus-m 320 browser output contract is:

| property | value |
|---|---|
| output shape | `[1, 2125, 33]` |
| points | `40 x 40 + 20 x 20 + 10 x 10 + 5 x 5 = 2125` |
| channels | One exported sigmoid class score plus four eight-bin regression distributions |
| strides | `8`, `16`, `32`, `64` |
| center prior | `(column * stride, row * stride)` |
| regression decode | Softmax expectation over bins `0..7`, multiplied by stride |
| NMS IoU | `0.6` |
| maximum detections | `200` |

The output shape and postprocess contract must be derived from the pinned implementation rather than assumed from another NanoDet version.

## Export parity validation

Feed one identical preprocessed image tensor to the PyTorch model and ONNX Runtime Python.

Validate:

- Fixed ONNX input shape.
- ONNX graph validity.
- Runtime output shape.
- Raw export-contract tensor numerical equivalence.
- Confidence-filtered and NMS-processed bounding-box set equivalence.

The PyTorch comparison applies sigmoid only to the class channel before raw-output comparison because NanoDet's ONNX export path includes that activation while its ordinary evaluation forward returns class logits.
Both runtime outputs then use the same source-derived distance decode and NMS contract.

No browser artifact may be treated as validated until this parity check passes.

## PWA prototype contract

The browser probe must implement:

1. Rear-camera startup through `getUserMedia`.
2. A full-screen native `playsInline` video preview.
3. Direct complete-frame conversion to `320 x 320`.
4. NanoDet ONNX execution inside the browser.
5. Adjustable confidence filtering and IoU `0.6` NMS.
6. Bounding-box mapping back to visible preview coordinates.
7. Current confidence and detection-count display.
8. A 100-millisecond detector request cadence.
9. At most one active inference.
10. No queued camera image while inference is active.
11. Preview rendering independent from detector completion rate.
12. Installable PWA metadata.
13. Offline caching of the application, model, and runtime artifacts.

The current E1 validation configuration uses `keep_ratio: false`.
The full-frame probe therefore stretches the complete source frame directly to `320 x 320` and must invert that transform before mapping through the preview's centered `object-fit: cover` geometry.

A busy scheduler tick increments the dropped-frame count and discards its image.
Busy ticks may collapse into one latest-frame request, but the current video frame must be sampled only when that request starts; no prior frame bitmap or tensor may be retained as a queue entry.

## Execution providers

Attempt automatic initialization in this order:

1. WebGL.
2. WASM SIMD with one thread.
3. WASM SIMD with multiple threads when cross-origin isolation is available.

Do not assume WebGPU support.
A forced provider measurement must fail visibly rather than silently substitute another provider.
Each provider must be tested after a page reload because the browser runtime is initialized once per page load.

The multi-threaded WASM condition requires a hosting environment that returns the cross-origin isolation headers needed for `crossOriginIsolated` to become true.

## Telemetry

Display on screen and write to the console:

- Selected execution provider.
- Camera frame size.
- Preprocess milliseconds.
- Inference milliseconds.
- Decode and NMS milliseconds.
- End-to-end milliseconds.
- Rolling median.
- Rolling p95.
- Effective detector Hz.
- Current detection count.
- Dropped-frame count.

Measure each stage separately around its own work.
Do not include model initialization or warm-up in the rolling detector measurements.

A sixty-second provider run must also report:

- Sample count.
- Whole-run median and p95.
- First-ten-second median.
- Last-ten-second median.
- Percentage change from the first to the last window.
- Dropped-frame count.

Browser APIs do not expose iPhone temperature.
Thermal acceptance therefore combines the timing trend with manual observation of device heat, display dimming, and operating-system behavior.

## Acceptance

The full-frame probe passes only when the following have been measured on iPhone 13:

- Rear camera starts.
- Bounding boxes align with tile faces, including near preview edges.
- No inference request backlog forms.
- The preview and telemetry controls remain responsive during inference.
- Median end-to-end latency is near the 100-millisecond budget or the measured miss is explicitly judged.
- Effective detector rate is reported rather than inferred only from inference duration.
- Sixty-second first-window and last-window timing is recorded.
- Physical heat tendency is recorded manually.
- WebGL and WASM SIMD single-thread results are recorded from the same scene and threshold.
- WASM multi-thread is recorded when a cross-origin-isolated host is available.
- The fastest practical provider is selected from measured results.

Passing this investigation establishes only the full-frame deployment runtime and coordinate contract.
It does not establish accuracy for the fixed composite input from PRODUCT-ADR-RECOGNITION-002.

## What was implemented

The following implementation artifacts were authored:

| artifact | responsibility |
|---|---|
| `tools/recognition/nanodet/export_and_validate_e1_320_onnx.py` | Inventory model artifacts, invoke the pinned official exporter, validate ONNX shape, compare PyTorch and ONNX Runtime raw output, compare final bounding boxes, and copy the validated model into the PWA. |
| `tools/recognition/pwa_detector_probe/src/nanodet.ts` | E1 preprocessing, source-derived NanoDet output decode, confidence filtering, and NMS. |
| `tools/recognition/pwa_detector_probe/src/runtime.ts` | Provider initialization, forced-provider measurement, and automatic fallback. |
| `tools/recognition/pwa_detector_probe/src/telemetry.ts` | Rolling and sixty-second timing statistics. |
| `tools/recognition/pwa_detector_probe/src/main.ts` | Camera lifecycle, latest-frame scheduling, inference, overlay mapping, telemetry UI, and provider-run persistence. |
| `tools/recognition/pwa_detector_probe/vite.config.ts` | HTTPS development support, cross-origin isolation headers, install metadata, and offline precaching. |
| `tools/recognition/pwa_detector_probe/README.md` | Server export, deployment, provider measurement, and acceptance procedure. |

## Current state

Implementation authoring is complete for the first probe.
Execution evidence is not yet available.

The Windows repository did not contain the server run's `model_best` artifact.
Therefore the following remain pending on the actual execution environments:

- Server-side model artifact inventory.
- ONNX export.
- PyTorch and ONNX Runtime Python parity report.
- Dependency installation and production PWA build.
- Trusted HTTPS deployment.
- iPhone 13 camera and overlay validation.
- Provider-specific sixty-second measurements.
- Heat observation and final provider judgment.

The requested repository-root `prompt_chappy.md` was not present during authoring.
No alternative authoring-policy location was inferred.

## Open questions

- Whether WebGL or WASM SIMD produces the lower sustained end-to-end latency on iPhone 13.
- Whether multi-threaded WASM improves inference enough to justify cross-origin-isolated hosting.
- Whether the epoch-40 ONNX graph requires browser-specific graph or operator changes.
- Whether sustained execution near 10 Hz produces unacceptable heat or timing degradation.
- Whether the full-frame latency remains representative after the three-region composite preprocessing is added.
