# PRODUCT-TASK-SYSTEM-002-13: Promote real-capture detector to production

- **status**: in_progress
- **date**: 2026-08-28
- **work_item**: PRODUCT-WORK-SYSTEM-002
- **task_type**: correction
- **estimate**: 0.5d
- **depends_on**:
  - PRODUCT-TASK-SYSTEM-002-12
- **finding_refs**:
  - PRODUCT-TASK-SYSTEM-002-04/F-MAJ-10
- **outputs**:
  - production detector artifact replaced by the accepted real-capture fine-tune
  - updated Recognition model-set/provenance identity
  - exact-capture detector regression evidence
  - PRODUCT-TASK-SYSTEM-002-13

## Goal

Replace the composite-augmented production NanoDet artifact with the real-capture fine-tuned detector after exact-runtime comparison shows materially better target-domain meld recall with no held-out composite meld regression.

## Work

- Promote `.local/recognition/nanodet_runs/E1_plus_m_320_real_capture_ft10_l10_seed42/model_best/nanodet-plus-m-320-real-capture-ft10-l10.onnx` into the canonical `vendor/recognition-models` production package.
- Compute and pin the promoted artifact SHA-256/byte identity; do not copy an unverified detector into the production manifest.
- Advance the Recognition model-set identity to `recognition-v2-2026-08-28` while leaving the accepted tile classifier, red-five classifier, runtime specs, and provider preference unchanged.
- Update `vendor/recognition-models/provenance.json` with the selected fine-tune source run and exact-runtime selection evidence.
- Remove the superseded composite-augmented detector payload from the canonical production vendor directory after verifying its known baseline SHA-256, so the public production package does not ship an unused detector artifact.
- Keep the historical comparison tooling independent of the current vendor pin by comparing the two `.local` run artifacts directly.
- Update the actual-artifact tests/E2E model-set expectation to the promoted model set.
- Re-run the captured iPhone detector tensor against the current production artifact. The bounded regression must retain at least six meld-region candidates after production post-processing and must not retain a meld box wider or taller than 60 composite pixels for this recorded failure case.
- Build the production PWA and verify the generated production asset manifest changes identity with the new Recognition model set.
- Re-deploy and repeat iPhone 13 meld recognition before closing the target-device finding.

## Done condition

The real-capture fine-tune is the only vendored production detector, its exact SHA-256 is content-addressed by `recognition-v2-2026-08-28`, production artifact/runtime gates pass, the captured failure tensor satisfies the bounded meld-localization regression, and target-device re-verification shows the promoted detector rather than the superseded model set.

## Verification

From repository root:

- `python tools/recognition/nanodet/promote_real_capture_finetune.py`
- `python tools/compare_recognition_debug.py mjtensu-recognition-debug-2026-08-27T15-49-35-764Z`

From `product/frontend`:

- `npx vitest run test/recognition-production-model-artifacts.test.ts test/pwa-production-assets.test.ts test/recognition-duplicate-suppression.test.ts test/recognition-services.test.ts`
- `npm run typecheck`
- `npm run build`
- `npm run build:e2e`
- `npx playwright test test/e2e/recognition-production-artifacts.spec.ts`

Target-device:

- verify debug/runtime identity reports `recognition-v2-2026-08-28` and the selected detector provider;
- repeat the physical meld arrangement that produced the recorded 2026-08-27 failure and confirm distinct tile-localized detector feedback is retained after the corrected duplicate policy.

## Evidence

### Selection decision: 2026-08-28

The model decision was rerun at the exact deployed detector operating point rather than inferred from aggregate COCO AP. Both artifacts used confidence threshold `0.35`, IoU NMS `0.60`, fixed semantic-region assignment, merged-bridge rejection, and greedy pairwise duplicate suppression.

Held-out real-capture validation (`8` images):

| model | TP | FP | FN | precision | recall | F1 | clean | meld TP/FP/FN | meld F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---|---:|
| composite-augmented baseline | 124 | 3 | 4 | 0.9764 | 0.9688 | 0.9725 | 4/8 | 8 / 2 / 4 | 0.7273 |
| real-capture fine-tune | 128 | 3 | 0 | 0.9771 | 1.0000 | 0.9884 | 6/8 | 12 / 1 / 0 | 0.9600 |

Held-out composite validation (`71` images):

| model | TP | FP | FN | precision | recall | F1 | clean | meld TP/FP/FN | meld F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---|---:|
| composite-augmented baseline | 1044 | 28 | 8 | 0.9739 | 0.9924 | 0.9831 | 45/71 | 156 / 1 / 0 | 0.9968 |
| real-capture fine-tune | 1045 | 23 | 7 | 0.9785 | 0.9933 | 0.9858 | 48/71 | 156 / 1 / 0 | 0.9968 |

The fine-tune therefore removes all four held-out real-capture false negatives, improves real-capture meld F1 from `0.7273` to `0.9600`, preserves perfect meld recall on the held-out composite set, and slightly improves the composite-set overall F1. This resolves the earlier aggregate-AP ambiguity in favor of the real-capture fine-tune for the deployed runtime operating point.

The exact 2026-08-27 iPhone failure tensor reinforces the selection: the composite-augmented detector retained only four meld candidates after the corrected post-process, including a `108.98 x 50.13` merged localization, while the real-capture fine-tune retained seven tile-scale meld candidates on the same tensor.

`tools/recognition/nanodet/promote_real_capture_finetune.py` owns the mechanical promotion so the selected ONNX bytes are copied, hashed, content-addressed, recorded in provenance, and the known superseded vendored detector is removed without manually inventing an artifact identity.

### Artifact promotion: 2026-08-28

The promotion script completed successfully from repository root.

- model set: `recognition-v2-2026-08-28`
- source artifact: `.local/recognition/nanodet_runs/E1_plus_m_320_real_capture_ft10_l10_seed42/model_best/nanodet-plus-m-320-real-capture-ft10-l10.onnx`
- canonical production artifact: `vendor/recognition-models/nanodet-plus-m-320-real-capture-ft10-l10.onnx`
- SHA-256: `9587a02dd1bbccfc14a925dc69c66b3c4a34ab628552b840ec113f7899dbf883`
- bytes: `5,597,449`
- runtime spec remains `nanodet-plus-m-320-v1`
- provider preference remains `wasm-simd -> wasm-threaded -> webgl`
- `production-model-set.json` now content-addresses the promoted detector by the exact SHA-256 above.
- `vendor/recognition-models/provenance.json` records the same artifact identity, source run, and exact-runtime selection evidence.
- The superseded `nanodet-plus-m-320-composite-augmented.onnx` production-vendor payload was removed by the promotion script after baseline identity validation.

The remaining T13 gates are exact captured-tensor regression, focused production-asset/runtime tests, production/E2E builds, browser real-artifact execution, and iPhone 13 re-verification.

### Deterministic verification: 2026-08-28

User-executed deterministic verification completed successfully after artifact promotion.

- Exact captured-tensor replay: **PASS**. Current production model set `recognition-v2-2026-08-28` retains `7` meld-region candidates on the recorded iPhone failure tensor and the bounded oversized-box check reports `oversized=[]`. The previous composite-augmented baseline still reproduces only `4` retained meld candidates including the `108.98 x 50.13` merged localization, confirming the regression fixture distinguishes the superseded and promoted detector artifacts.
- Focused Vitest gate: **PASS**, 4 files / 32 tests:
  - `test/recognition-production-model-artifacts.test.ts`
  - `test/pwa-production-assets.test.ts`
  - `test/recognition-duplicate-suppression.test.ts`
  - `test/recognition-services.test.ts`
- `npm run typecheck`: **PASS**.
- `npm run build`: **PASS**. Production PWA build emitted `production-assets-973ac08b074fc268.json`, changing the build-asset identity from the prior detector pin. Main application bundle for this build: `assets/index-CLYLTmbD.js`. PWA `generateSW` completed and emitted `dist/sw.js` plus `dist/workbox-2fbc6a65.js`.
- `npm run build:e2e`: **PASS**. E2E build emitted the same `production-assets-973ac08b074fc268.json` asset identity and the Recognition real-artifact harness.
- `npx playwright test test/e2e/recognition-production-artifacts.spec.ts`: **PASS**, 1 Chromium test. The public production Recognition runtime reports model set `recognition-v2-2026-08-28`; detector, base classifier, and red-five classifier all initialize on `wasm-simd` with no failed providers. Frozen base/red-five/blank-frame fixture semantics remain unchanged.
- Vite future native-config-loader notices and the >500 kB chunk-size notice are non-blocking warnings and do not invalidate the detector-promotion acceptance criteria.

All desktop/browser deterministic T13 gates are PASS. T13 remains `in_progress` only for target-device redeployment and iPhone 13 re-verification of model-set identity and physical meld recognition.
