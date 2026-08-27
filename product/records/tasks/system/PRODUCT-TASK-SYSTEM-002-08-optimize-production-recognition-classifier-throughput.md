# PRODUCT-TASK-SYSTEM-002-08: Optimize production Recognition classifier throughput

- **status**: completed
- **date**: 2026-08-27
- **work_item**: PRODUCT-WORK-SYSTEM-002
- **task_type**: correction
- **estimate**: 1d
- **depends_on**:
  - PRODUCT-TASK-SYSTEM-002-01
  - PRODUCT-TASK-SYSTEM-002-02
- **finding_refs**:
  - PRODUCT-TASK-SYSTEM-002-04/F-MAJ-04
- **outputs**:
  - batched production tile-classifier inference path
  - target-device performance-ready production Recognition build
  - PRODUCT-TASK-SYSTEM-002-08

## Goal

Remove the per-candidate sequential classifier bottleneck discovered during iPhone 13 acceptance and make the complete production Recognition path suitable for objective I05 target-device cadence measurement.

## Work

- Replace detector-candidate `for` + awaited batch-1 base-classifier inference with one bounded batched base-classifier invocation per evaluated frame.
- Preserve candidate/result correspondence and all existing classification semantics, including invalid/background output handling and observation ordering.
- Collect only base `5m` / `5p` / `5s` candidates and invoke the red-five specialist in one bounded batch for those candidates rather than one inference per candidate.
- Extend classifier preprocessing/tensor construction to produce `[N,C,64,64]` batches while preserving current crop resize, letterbox, normalization, red-five refinement, and artifact contracts.
- Measure or otherwise isolate preprocessing cost sufficiently to determine whether JavaScript crop/Lanczos preprocessing remains a material bottleneck after inference batching.
- Do not silently change model artifacts, recognition semantics, stabilization rules, or the accepted 100 ms request cadence in this correction Task.
- Add focused tests covering batched tensor shape, result ordering, invalid/background handling, red-five selective batching, zero red-five candidates, and representative multi-candidate frames.
- Expose/retain enough diagnostics for I05 to record the actual selected execution provider per production model on iPhone 13.

## Done condition

The production pipeline performs at most one base-classifier inference and at most one red-five-classifier inference per evaluated frame, preserves existing recognition output semantics, passes focused regression/build verification, and is ready for I05 target-device timing measurement.

## Verification

- Run focused classifier/preprocessing/production-pipeline tests.
- `npm run typecheck`
- `npm run lint`
- `npm test`
- `npm run build`
- Target-device cadence PASS/FAIL remains owned by PRODUCT-TASK-SYSTEM-002-05.

## Evidence

- PRODUCT-TASK-SYSTEM-002-04/F-MAJ-04 records roughly `1.2 fps` observed target-device behavior.
- Before this correction, `src/recognition/production-pipeline.ts` awaited `classifyDetection` inside a loop over detector candidates.
- Before this correction, classifier tensor construction always used batch size `1` even though the exported classifier runtime contract supports a batch dimension.
- `spec:product.system.contracts.testing_strategy` requires the complete production path to sustain the accepted 100 ms request cadence or return to performance implementation/spec resolution.

### Implementation: 2026-08-27

- `src/recognition/classifier/preprocessing.ts` now builds batched NCHW tensors for both classifier models. Base crops are packed as `[N,1,64,64]`; red-five crops are packed as `[N,3,64,64]`. Existing grayscale/RGB conversion, Lanczos resize, border-median letterbox fill, normalization, and single-crop helpers remain unchanged semantically.
- `src/recognition/classifier/runtime.ts` now provides `classifyBatch()`. One base-classifier invocation covers every candidate in the frame, base results are mapped back by batch index, only base `5m` / `5p` / `5s` results are collected for one red-five batch, and refined results are written back to their original candidate positions. An empty batch invokes neither classifier.
- `src/recognition/production-pipeline.ts` now extracts all candidate crops before classification and invokes the batched classifier path once per evaluated frame instead of awaiting classifier inference inside the detection loop. The ONNX session adapter validates classifier output length against `batchSize * classCount`, preserving model-incompatible handling for malformed outputs.
- Production evaluation timing now isolates detector preprocessing/inference/postprocessing, crop extraction, base-classifier preprocessing/inference, and red-five preprocessing/inference. This separates the JavaScript crop/Lanczos cost from model execution so I05 can determine whether preprocessing remains material after batching.
- `createRecognitionRuntimeComposition()` retains the latest 120 evaluation timing samples and exposes them together with `RecognitionModelRuntime` diagnostics, including the actual selected execution provider and failed-provider fallbacks for each production model. The public Recognition runtime contract exposes this diagnostics shape without introducing ORT-specific values.
- Focused regression coverage was extended for batched tensor shape, candidate/result ordering, invalid handling, selective red-five batching, zero red-five candidates, zero detector candidates, one-base/one-red-five production-frame inference counts, preprocessing timing observability, and selected-provider diagnostics.

### Verification: 2026-08-27

User-executed verification from `product/frontend`:

- `npx vitest run test/recognition-c8-classifier.test.ts test/recognition-services.test.ts` — **PASS**, 2/2 files and 28/28 tests.
- `npm run typecheck` — **PASS**.
- `npm run lint` — **PASS**, architecture import boundaries OK across 58 source files.
- `npm test` — **PASS**, 35/35 files and 333/333 tests.
- `npm run build` — **PASS**, Vite 8.2.2 production build completed and PWA `generateSW` emitted `dist/sw.js` and `dist/workbox-2fbc6a65.js`.
- Production build emitted `assets/index-DSAVR7Gc.js`; production asset-manifest identity remains `production-assets-e15bf73e46ef0d48.json` because this correction does not change pinned Recognition/Agari artifacts.
- Vite native-config-loader extension notices and the >500 kB chunk warning are non-blocking build warnings and do not invalidate the classifier-throughput correction criteria.

All deterministic implementation/regression/build gates are PASS. The production pipeline is ready for I05 target-device timing and selected-provider measurement. Target-device cadence PASS/FAIL remains owned by PRODUCT-TASK-SYSTEM-002-05 and is not inferred from these desktop checks.
