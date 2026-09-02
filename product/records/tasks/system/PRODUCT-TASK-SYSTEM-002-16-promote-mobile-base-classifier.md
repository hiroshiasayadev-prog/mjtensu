# PRODUCT-TASK-SYSTEM-002-16: Promote mobile base classifier

- **status**: in_progress
- **date**: 2026-09-02
- **work_item**: PRODUCT-WORK-SYSTEM-002
- **task_type**: correction
- **depends_on**:
  - PRODUCT-INV-RECOGNITION-011
- **outputs**:
  - MobileNetV3-Small 1.0x production base-classifier binding
  - updated production model-set/provenance identity
  - production artifact regression evidence
  - PRODUCT-TASK-SYSTEM-002-16

## Trigger

PRODUCT-INV-RECOGNITION-011 completed the architecture comparison and direct iPhone 13 ONNX Runtime Web A/B benchmark. MobileNetV3-Small 1.0x retained essentially the same dense-angle manual-domain accuracy as the current Plain e150 classifier while reducing isolated target-device batch inference by about `1.5-1.7x` across `N=1..24`.

Representative iPhone production diagnostics after classifier-preprocessing optimization show base inference around `62 ms` for 18 candidates, now the largest single measured stage. The direct benchmark predicts that MobileNetV3-Small 1.0x should reduce this stage to roughly the mid-30 ms range at a comparable batch size.

## Goal

Replace the production Plain e150 base classifier with the selected MobileNetV3-Small 1.0x artifact without changing the existing `[N,1,64,64] -> [N,35]` classifier input/output contract, normalization, label mapping, invalid/background semantics, red-five refinement, or execution-provider preference.

## Work

- Bind `tile-mobilenet-v3-small-1.0x-random360-e150.onnx` as the production `tile-classifier` artifact.
- Preserve the current gray64 normalization and runtime preprocessing contract.
- Update model-set version, content-addressed URL, SHA-256 identity, provenance, and production-artifact tests.
- Keep provider preference `wasm-simd -> wasm-threaded -> webgl` unchanged.
- Update architecture-specific specification wording where it incorrectly names C8 as the current base classifier.
- Run focused production-model artifact tests, classifier/runtime tests, typecheck, and production build.
- Re-measure iPhone 13 base inference and complete-pipeline timing after deployment.

## Done condition

- Production build integrity validates the MobileNet artifact against the committed SHA-256.
- Production Recognition initializes and classifies through the existing contract without semantic regression.
- Focused tests/typecheck/build pass.
- iPhone diagnostics confirm the expected material base-inference reduction versus the current ~62 ms / 18-candidate observation.

## Evidence

INV-011 selected MobileNetV3-Small 1.0x with dense-angle metrics:

- manual mean: `0.94872` versus Plain `0.94743`;
- manual worst: `0.93111` versus Plain `0.94000`;
- JP mean: `0.99948` versus Plain `0.99959`;
- direct iPhone WASM-SIMD median: `30 ms` versus Plain `51 ms` at batch 16;
- direct iPhone WASM-SIMD median: `46 ms` versus Plain `79 ms` at batch 24.

The vendored candidate artifact size is `6,223,234` bytes. The user-verified local SHA-256 is `6c93615cf6b7ffcb829ce49e91c9f6557b499aeeae51129cd8411b665331af76`.

## Implementation: 2026-09-02

The production model set is now wired to `tile-mobilenet-v3-small-1.0x-random360-e150.onnx` with the verified SHA-256 and model-set version `recognition-v4-2026-09-02`. `vendor/recognition-models/provenance.json` records the INV-011 source run, artifact size, dense-angle selection metrics, and direct iPhone WASM-SIMD batch-16/batch-24 evidence.

Because the base-classifier runtime contract describes gray64 input, normalization, dynamic batch, and 35 logits rather than a backbone architecture, the stale `c8-tile-35-v1` name is replaced by the architecture-neutral `gray64-tile-35-v1`. The normalization values and `[N,1,64,64] -> [N,35]` contract are unchanged. Runtime/unit/e2e expectations and current Recognition spec/ADR wording are updated consistently; historical C8 evidence remains identified as historical rather than rewritten.

Implementation wiring is complete.

## Verification: 2026-09-02

User-executed verification from `product/frontend`:

- focused Vitest set covering classifier preprocessing/runtime, browser model cache, production model artifacts, PWA production assets, and production Recognition services — **PASS**, 6/6 files and 51/51 tests;
- `npm run typecheck` — **PASS**;
- `npm run build` — **PASS**, emitting content-versioned production asset manifest `production-assets-3cea6089b4d96f6a.json`;
- `npm run build:browser-verification` — **PASS**;
- `npx playwright test test/e2e/recognition-production-artifacts.spec.ts` — **PASS**.

The real-artifact browser verification initialized all three production models on `wasm-simd`, reported model set `recognition-v4-2026-09-02`, reported the base classifier runtime spec as `gray64-tile-35-v1`, executed the MobileNet base classifier to a finite 35-logit output, executed the red-five classifier to the expected fixture result, and completed the blank-frame production pipeline fixture.

All desktop/browser integrity and semantic-regression gates are PASS. I16 remains `in_progress` only because its Done condition requires post-deployment iPhone timing confirming the expected material reduction from the prior ~62 ms / 18-candidate base-inference observation.
