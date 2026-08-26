# NanoDet iPhone PWA detector probe

This prototype measures whether the epoch-40 `NanoDet-Plus-m 320` tile-region detector can run near 10 Hz in an iPhone 13 PWA while keeping the native camera preview smooth.

It intentionally implements only:

```text
rear camera -> centered 1:1 crop -> resize to 320 x 320 -> NanoDet -> decode/NMS -> bbox overlay in the same square
```

Tile classification, hand interpretation, scoring, temporal stabilization, and the three-region composite layout are excluded from this probe.

## 1. Export and validate the model on the NanoDet server

The export helper first inventories the actual `model_best` artifact, records sizes and SHA-256 values, verifies the pinned NanoDet v1.0.0 commit, and then invokes NanoDet's own `tools/export_onnx.py` with a fixed `320 x 320` input.

From `/srv/data/mjtensu`, in the verified NanoDet Python environment:

```bash
python tools/recognition/nanodet/export_and_validate_e1_320_onnx.py
```

Defaults:

```text
run:
  /srv/data/mjtensu/.local/recognition/nanodet_runs/E1_plus_m_320_amp30_seed42

config:
  tools/recognition/nanodet/configs/e1_nanodet_plus_m_320_stage40_amp_resume.yml

ONNX output:
  .local/recognition/browser_probe/nanodet-plus-m-320.onnx

PWA model copy:
  tools/recognition/pwa_detector_probe/public/models/nanodet-plus-m-320.onnx

parity report:
  .local/recognition/browser_probe/e1-320-onnx-parity.json
```

When `--image-path` is omitted, the helper selects a validation image whose annotation count is nearest to fourteen. An explicit image can be used instead:

```bash
python tools/recognition/nanodet/export_and_validate_e1_320_onnx.py \
  --image-path /srv/data/mjtensu/data/<image>
```

The helper rejects the artifact when any of the following is not established:

- NanoDet source is not commit `d3fb34fa91d6020f273d6d063bf324dcd97bac12`.
- ONNX input is not fixed `[1, 3, 320, 320]`.
- Runtime output is not `[1, 2125, 33]`.
- PyTorch export-contract output and ONNX Runtime Python output are not numerically equivalent within the configured tolerance.
- The decoded/NMS detection sets are not equivalent.

The postprocess contract implemented by both the Python validator and browser is:

- One exported sigmoid class score followed by 32 raw regression logits.
- 2,125 center priors from strides `8`, `16`, `32`, and `64`.
- Prior coordinate `(column * stride, row * stride)` with no half-stride offset.
- Four eight-bin distributions, softmax expectation over bins `0..7`, multiplied by stride.
- Adjustable confidence threshold, class-agnostic NMS IoU `0.6`, maximum 200 detections.

## 2. Install and build the PWA

Use Node.js 20.19 or newer.

```bash
cd tools/recognition/pwa_detector_probe
npm install
npm run build
```

`npm run build` generates the 192-pixel and 512-pixel install icons and copies the ONNX Runtime Web WASM and module artifacts into `public/ort` before Vite builds the application. The generated service worker precaches the application, ONNX model, and runtime artifacts.

The build must contain the validated model before it is deployed:

```text
public/models/nanodet-plus-m-320.onnx
```

## 3. Serve over trusted HTTPS

Camera access requires a secure context. For the actual iPhone measurement, deploy `dist` to a trusted HTTPS origin.

The host must return these headers when testing multi-threaded WASM:

```text
Cross-Origin-Opener-Policy: same-origin
Cross-Origin-Embedder-Policy: require-corp
Cross-Origin-Resource-Policy: same-origin
```

The Vite development and preview configuration already emits these headers. Its generated local certificate is useful for desktop development, but a trusted HTTPS deployment is preferable for iPhone measurement.

Development server:

```bash
npm run dev
```

Production preview after `npm run build`:

```bash
npm run preview
```

## 4. Provider measurements

A runtime instance is initialized only once per page load. Test each provider after a full reload by selecting it in the panel, or by opening the corresponding URL:

```text
?provider=webgl
?provider=wasm-simd
?provider=wasm-threaded
```

The default URL attempts initialization in this order:

```text
WebGL -> WASM SIMD with one thread -> WASM SIMD multi-threaded
```

The threaded condition is rejected when `crossOriginIsolated` is false. A forced provider URL does not silently substitute another provider; this prevents invalid comparison results.

For each provider:

1. Reload with the provider forced.
2. Start the rear camera.
3. Confirm that the visible preview is the centered 1:1 camera crop and that boxes align with tiles near all four square edges.
4. Set the confidence threshold required for the measurement and keep it unchanged across providers.
5. Start `Run 60s measurement`.
6. Leave the same scene and camera position unchanged for sixty seconds.
7. Record physical heat, display dimming, or OS warnings manually.

The latest result is retained in local storage and shown beside the other provider results.

## Scheduling and coordinate behavior

The detector scheduler issues a tick every 100 milliseconds. Only one inference may be active. A tick arriving while inference is active increments `Dropped ticks`; its camera image is discarded. Busy ticks are coalesced into one latest-frame request, and that request samples the current video frame only after the active inference finishes. No old image is queued.

The `<video>` preview is not rendered through the processing canvas. It remains a native `playsInline` video element inside a square viewport, so preview smoothness is independent of detector completion rate.

The visible video uses centered `object-fit: cover` inside that 1:1 viewport. Preprocessing computes the same centered square from the native camera frame, crops only that square, and resizes it to `320 x 320`. Decoded coordinates therefore map directly from model coordinates to the visible square. Pixels outside the centered square are neither displayed nor evaluated.

## Telemetry

The panel and console report:

- Selected execution provider.
- Native camera frame size and centered square crop size.
- Preprocess milliseconds.
- Inference milliseconds.
- Decode/NMS milliseconds.
- End-to-end milliseconds.
- Rolling median over the latest 120 completed detections.
- Rolling p95 over the latest 120 completed detections.
- Effective detector Hz over the latest five seconds.
- Current detection count.
- Dropped scheduler ticks.

A 60-second run also reports the first-ten-second and last-ten-second median and their percentage difference. Browser APIs do not expose iPhone temperature, so the thermal judgment must combine this slowdown signal with manual physical observation.

## Acceptance record

Record each item as pass, fail, or not measured:

```text
[ ] rear camera starts on iPhone 13
[ ] bbox aligns with tile faces across the visible preview
[ ] no inference request backlog forms
[ ] telemetry panel remains operable during inference
[ ] median end-to-end latency is near 100 ms
[ ] 60-second first/last-window slowdown is acceptable
[ ] physical heat trend is acceptable
[ ] WebGL result recorded
[ ] WASM SIMD single-thread result recorded
[ ] WASM multi-thread result recorded when cross-origin isolation is available
[ ] fastest practical provider identified from the same scene and threshold
```
