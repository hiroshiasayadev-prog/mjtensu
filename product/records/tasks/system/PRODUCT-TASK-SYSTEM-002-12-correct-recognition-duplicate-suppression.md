# PRODUCT-TASK-SYSTEM-002-12: Correct Recognition duplicate suppression

- **status**: in_progress
- **date**: 2026-08-28
- **work_item**: PRODUCT-WORK-SYSTEM-002
- **task_type**: correction
- **estimate**: 0.5d
- **depends_on**:
  - PRODUCT-TASK-SYSTEM-002-04
- **finding_refs**:
  - PRODUCT-TASK-SYSTEM-002-04/F-MAJ-09
- **outputs**:
  - corrected production detector duplicate resolution
  - clarified canonical Recognition duplicate-resolution contract
  - PRODUCT-TASK-SYSTEM-002-12

## Goal

Correct the target-device Recognition failure where a large merged detector box transitively connects several spatially distinct tile detections and the current duplicate-suppression implementation keeps only one confidence winner.

## Work

- Clarify `spec:product.recognition.pipeline` so duplicate resolution preserves separate candidates when a larger merged box overlaps multiple candidates that are not themselves duplicates.
- Reject a larger merged bridge before detector-confidence winner selection when it substantially covers at least two smaller candidates that do not substantially overlap one another.
- Use detector confidence only to resolve remaining pairwise duplicate alternatives.
- Remove the transitive connected-component behavior that can collapse `A -> bridge -> B` even when `A` and `B` are not duplicate-overlapping.
- Preserve same-region scope, deterministic confidence/tie-break behavior, and the existing overlap metric/current production threshold.
- Add focused unit coverage for a high-confidence merged bridge, a transitive bridge, ordinary duplicate confidence selection, neighboring distinct detections, semantic-region isolation, and the exact captured iPhone meld candidate geometry.
- Update the temporary debug comparison utility to reproduce the corrected production postprocess when re-decoding the captured raw detector output.
- Do not silently rewrite the semantics of existing immutable detector-crop review datasets. `tools/recognition/detector_duplicate_groups.py` remains historical dataset-construction behavior unless a later versioned dataset rebuild explicitly adopts the corrected policy.
- Re-run the captured iPhone failure after deployment. This correction does not claim that the production detector model itself has sufficient meld localization quality; model-selection follow-up remains separate if bad merged detections remain after post-processing is corrected.

## Done condition

The canonical Recognition contract and production runtime agree on merged-bridge and pairwise duplicate behavior, focused automated verification passes, and the captured iPhone failure no longer collapses all meld-region candidates to one box solely because of duplicate suppression.

## Verification

- `npx vitest run test/recognition-duplicate-suppression.test.ts test/recognition-services.test.ts`
- `npm run typecheck`
- `npm run build`
- Re-run `tools/compare_recognition_debug.py` or an equivalent exact-tensor comparison against the captured iPhone debug case and confirm that duplicate suppression does not reduce seven post-NMS meld candidates to one transitive component winner.
- Run `python tools/recognition/nanodet/compare_runtime_models.py` to compare the production detector and real-capture fine-tune on the existing held-out real/composite datasets using the corrected runtime threshold, region assignment, merged-bridge rejection, and pairwise duplicate policy before making any detector-artifact replacement decision.
- Re-deploy and repeat target-device meld recognition before closing F-MAJ-09.

## Evidence

- The 2026-08-27 iPhone debug capture used model set `recognition-v1-2026-08-27` with detector provider `wasm-simd`.
- The captured detector input reconstructed from the saved composite matched the browser tensor within `2.384185791015625e-07` maximum absolute error.
- iPhone WASM and desktop CPU production ONNX raw detector outputs matched within `1.0251998901367188e-05` maximum absolute error, ruling out a material provider-specific detector-output divergence for this frame.
- Ordinary IoU NMS retained seven meld-region candidates. The existing duplicate stage then joined all seven into one transitive component and retained only detection `1950` at confidence `0.594521`.
- The captured component included clear bridge evidence such as detection `2085` versus `1215` with IoU `0.144` but intersection/smaller-area overlap `1.000`, and `2085` versus `1222` with IoU `0.157` and overlap `1.000`. Those smaller detections are not one tile merely because the large box contains both.
- The pre-correction implementation in `product/frontend/src/recognition/detector/duplicate-suppression.ts` explicitly built connected components from the overlap relation and selected one highest-confidence member per component.
- The superseded PRODUCT-ADR-RECOGNITION-003 recorded that earlier connected-component policy. The current canonical Specifications did not preserve that detailed algorithm, and the target-device evidence demonstrates that it violates the intended distinct-candidate preservation behavior.

### Correction verification: 2026-08-28

- Production duplicate resolution now rejects merged bridge candidates before confidence selection and then performs deterministic greedy pairwise duplicate suppression; it no longer collapses a transitive overlap component to one winner.
- `npx vitest run test/recognition-duplicate-suppression.test.ts test/recognition-detection-postprocessor.test.ts test/recognition-services.test.ts` — **PASS**, 3 files / 26 tests.
- `npm run typecheck` — **PASS**.
- `npm run build` — **PASS**, Vite 8.2.2 production PWA build completed and generated `sw.js` / Workbox assets. Application bundle for this verification build: `assets/index-BYs4cec1.js`.
- Vite future native-config-loader notices and the >500 kB chunk notice are non-blocking and unrelated to this correction.
- Exact captured-tensor replay with the corrected policy confirms the original catastrophic collapse is removed: ordinary IoU NMS retains 7 meld candidates; merged-bridge rejection removes detections `1910` and `2085`; greedy pairwise duplicate suppression then retains 4 meld candidates: `1215`, `1222`, `1890`, and `1950`.
- iPhone WASM raw output and desktop production ONNX remain effectively identical on the captured tensor, so this corrected result is attributable to post-processing policy rather than provider differences.
- The correction does **not** close the target-device meld-recognition finding by itself. The captured meld image contains multiple distinct physical tiles, while the current production detector still leaves only four retained meld candidates and one retained candidate (`1950`) remains an obviously large merged localization (`108.98 x 50.13` in composite coordinates). A later model-selection/recognition-quality decision must therefore be based on broader evidence rather than treating the duplicate-policy fix as sufficient detector accuracy.
- The real-capture fine-tune remains materially different on this exact tensor: after the corrected runtime postprocess it produces seven tile-sized meld candidates, but this single capture is not sufficient evidence to replace the production detector because it may still contain an extra/duplicate detection.
- Existing threshold-sweep evidence at the **current production detector threshold `0.35`**, before the corrected duplicate stage, already favors the real-capture fine-tune at the runtime operating point despite its lower aggregate COCO-style AP: held-out real captures improve from `TP=126 / FP=9 / FN=2 / F1=0.9582` to `TP=128 / FP=4 / FN=0 / F1=0.9846`; composite validation improves from `TP=1044 / FP=44 / FN=8 / F1=0.9757` to `TP=1046 / FP=36 / FN=6 / F1=0.9803`. The earlier decision to retain the composite-augmented detector weighted aggregate AP, but those AP differences include low-score calibration and higher-IoU localization behavior that are not identical to the deployed `0.35` runtime operating point.
- `tools/recognition/nanodet/compare_runtime_models.py` was added to remove that remaining comparison mismatch by evaluating both artifacts at the exact production threshold with semantic-region assignment and the corrected merged-bridge/pairwise duplicate policy on the existing held-out real and composite validation sets.
- Exact-runtime model comparison completed on 2026-08-28. On held-out real captures, the production detector produced `TP=124 / FP=3 / FN=4 / F1=0.9725 / clean=4/8`, while the real-capture fine-tune produced `TP=128 / FP=3 / FN=0 / F1=0.9884 / clean=6/8`. Meld-only performance improved from `TP=8 / FP=2 / FN=4 / F1=0.7273` to `TP=12 / FP=1 / FN=0 / F1=0.9600`.
- On composite validation, the production detector produced `TP=1044 / FP=28 / FN=8 / F1=0.9831 / clean=45/71`, while the real-capture fine-tune produced `TP=1045 / FP=23 / FN=7 / F1=0.9858 / clean=48/71`. Meld-only performance was identical at `TP=156 / FP=1 / FN=0 / F1=0.9968` for both models.
- The exact-runtime comparison therefore shows no meaningful held-out composite meld regression from the real-capture fine-tune and a large held-out real-capture meld-recall improvement. The model-selection question is no longer blocked by aggregate-AP ambiguity; the real-capture fine-tune is the stronger candidate for the deployed runtime operating point.
- T12 remains `in_progress` only until the corrected production build is re-executed on iPhone 13. The remaining detector-artifact replacement is routed separately to PRODUCT-TASK-SYSTEM-002-13 rather than expanding this duplicate-suppression correction beyond its stated scope.
