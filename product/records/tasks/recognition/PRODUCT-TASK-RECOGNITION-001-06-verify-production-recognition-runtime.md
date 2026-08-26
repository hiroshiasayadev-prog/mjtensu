# PRODUCT-TASK-RECOGNITION-001-06: Verify production recognition runtime

- **status**: completed
- **date**: 2026-08-26
- **work_item**: PRODUCT-WORK-RECOGNITION-001
- **task_type**: verification
- **estimate**: 1d
- **depends_on**:
  - PRODUCT-TASK-RECOGNITION-001-05
- **outputs**:
  - PRODUCT-TASK-RECOGNITION-001-06

## Goal

Execute one objective Recognition acceptance gate using the real production ONNX artifacts plus bounded fixed fixtures and the public production runtime.

## Work

- Load the production detector, 35-class base classifier, and red-five classifier through the real model runtime.
- Verify artifact manifest/runtime-spec compatibility and provider initialization on the supported browser test environment.
- Run bounded fixed image/tensor fixtures through each actual model path and compare normalized semantic output to expected contract results.
- Run bounded full-pipeline fixtures through the public one-frame Recognition service.
- Execute the complete focused Recognition test suite from R01 through R05.
- Record expected/observed results and an overall PASS, FAIL, or validly BLOCKED verdict.

## Done condition

Every predefined real-artifact/public-runtime Recognition check has an observed result and the overall verification verdict is PASS, FAIL, or validly BLOCKED.

## Verification

| check | expected result |
|---|---|
| detector ONNX load/runtime-spec validation | PASS |
| base C8 ONNX load/runtime-spec validation | PASS |
| red-five ONNX load/runtime-spec validation | PASS |
| bounded detector fixture inference/decode | expected semantic candidate output |
| bounded base-classifier fixture inference | expected 35-class semantic result |
| bounded red-five fixture inference | expected ordinary/red semantic result |
| bounded full one-frame pipeline fixture | expected observations/recognized snapshot |
| complete Recognition focused test suite | PASS |
| strict typecheck/lint/architecture gate | PASS |

The overall result is PASS only when every required check is PASS.

## Evidence

- This Task is the L2 actual-artifact gate required by `spec:product.system.contracts.testing_strategy`.
- Model-set version, artifact hashes, runtime/provider selections, fixture identities, and observed results are recorded here when executed.
- Target-device end-to-end performance remains owned by PRODUCT-WORK-SYSTEM-002 rather than this verification.

### Verification attempt: 2026-08-27

**Overall verdict: BLOCKED**

R06 cannot execute the required real-artifact/public-runtime acceptance gate from the current repository state because the authoritative production recognition model set is not materialized. The fake/synthetic R01-R05 coverage is present, but promoting those results to this L2 gate would not verify the production artifacts required by this Task.

| check | observed result |
|---|---|
| detector ONNX load/runtime-spec validation | **BLOCKED** — detector ONNX candidates exist under `.local/recognition/nanodet_runs/` and recognition tooling, but no build-pinned production model-set manifest identifies one candidate, URL, SHA-256, runtime spec, and provider preference as the production detector artifact. |
| base C8 ONNX load/runtime-spec validation | **BLOCKED** — selected 35-class checkpoint exists at `.local/recognition/tile_classifier_runs/gray64_c8_rot22p5_bs512_gray35_v2_seed42/best.pt` (7.71 MB), but no classifier ONNX artifact is present anywhere in the current mjtensu tree. `tools/recognition/export_c8_classifiers_onnx.py` provides the accepted export/parity path, but the production ONNX output is not materialized. |
| red-five ONNX load/runtime-spec validation | **BLOCKED** — no red-five training run/checkpoint directory and no red-five ONNX artifact are present in the current local `.local/recognition` tree. PRODUCT-INV-RECOGNITION-006 records the accepted warm-augmented RGB specialist on the Linux training host, but that artifact is not available here as a production browser asset. |
| production model-set manifest compatibility | **BLOCKED** — `src/recognition/model-runtime/manifest.ts` implements validation only. No committed/materialized production manifest data was found under `product/frontend`, and `createProductionRecognitionRuntime()` still requires callers to supply a `RecognitionModelSetManifest`. |
| bounded detector fixture inference/decode | **BLOCKED** — no authoritative production detector artifact/model-set identity is available for the required real-runtime execution. |
| bounded base-classifier fixture inference | **BLOCKED** — production base-classifier ONNX is absent. |
| bounded red-five fixture inference | **BLOCKED** — production red-five ONNX is absent. |
| bounded full one-frame pipeline fixture | **BLOCKED** — the production three-model set cannot initialize; `product/frontend/test/fixtures/` currently contains only the scoring golden fixture and no fixed real-artifact Recognition fixture set. |
| complete Recognition focused test suite | not re-executed as an R06 acceptance result because the required real-artifact gate is not runnable. R05 records the latest focused deterministic result as PASS: 6 files / 50 tests on 2026-08-27. |
| strict typecheck/lint/architecture gate | not promoted to an R06 PASS without the required real-artifact checks. R05 records both typecheck and lint/architecture PASS on 2026-08-27. |

Repository artifact inventory performed for this attempt found detector `.onnx` files only; no base-classifier or red-five-classifier `.onnx` file exists under `C:\Users\imved\projects\mjtensu`. The local `.local/recognition` root contains `tile_classifier_runs` and `red_five_datasets`, but no `red_five_runs`; the selected 35-class run contains only `best.pt`.

This is a valid verification blocker rather than a model mismatch verdict: R06 must not invent hashes, expected predictions, or a production model identity from research artifacts.

To resume R06, first materialize the accepted three-model production set: export the selected 35-class checkpoint to ONNX with its parity metadata, obtain/export the accepted warm-augmented RGB red-five checkpoint, designate the accepted detector artifact, and create the build-pinned `RecognitionModelSetManifest` containing exact artifact URLs, SHA-256 values, runtime specs (`nanodet-plus-m-320-v1`, `c8-tile-35-v1`, `c8-red-five-v1`), and provider preferences. Then add the bounded fixed Recognition fixtures/expected semantic results and execute every check in this Task through `createProductionRecognitionRuntime()`.

### Artifact materialization update: 2026-08-27

The classifier-artifact blockers from the initial attempt are now resolved locally.

- The selected production base classifier is now `gray64_c8_rot22p5_bs512_gray35_v3_jp189_seed42/best.pt` (epoch 45), superseding the v2 checkpoint selection recorded during the initial blocked attempt.
- Its deployment export is `.local/recognition/tile_classifier_runs/gray64_c8_rot22p5_bs512_gray35_v3_jp189_seed42/tile-c8-gray35-v3-jp189.onnx`, SHA-256 `b8a8fa3ff6c6d1e944a7593fa0afc947e0cd2513fb79ca46e5f8fcd6e19c97d0`. Export parity is `allclose=true` with zero prediction mismatches. The runtime-spec normalization has been updated to the checkpoint metadata values: mean `0.6815832403977466`, std `0.2725553681973969`.
- The selected warm-augmented RGB red-five deployment export is `.local/recognition/red_five_runs/c8_rgb_cr_ycr_warmaug_seed42/c8_rgb_warmaug_rot22p5_seed42/red-five-c8-rgb-warmaug.onnx`, SHA-256 `c2b780f682d84bf186db90290050f8b05016c3e8058de559eea679a28eeb80c6`. Export parity is `allclose=true` with zero prediction mismatches.

R06 remains **BLOCKED**, but no longer because classifier ONNX artifacts are missing. The remaining acceptance blockers are the authoritative detector selection/build pin, the production `RecognitionModelSetManifest`, and the bounded fixed real-artifact Recognition fixtures required to execute the full public-runtime gate.

### v3_jp189 runtime-spec verification: 2026-08-27

The production base-classifier runtime normalization was updated to match the selected `gray64_c8_rot22p5_bs512_gray35_v3_jp189_seed42/best.pt` export metadata.

- `npm test -- recognition-model-runtime.test.ts` — **PASS**: 1 file / 12 tests.
- `npm run typecheck` — **PASS**.
- `npm run lint` — **PASS**: `Architecture import boundaries: OK (52 source files checked)`.

This verifies the code-owned `c8-tile-35-v1` normalization binding for the v3_jp189 checkpoint and preserves the existing red-five normalization contract. It does not by itself satisfy the R06 real-artifact/public-runtime acceptance gate, so the Task status remains `blocked` pending the remaining manifest/detector/fixture work.

### Production model-set materialization: 2026-08-27

The detector identity and production model-set declaration are now fixed in source rather than left implicit.

- The production detector is `nanodet-plus-m-320-composite-augmented.onnx`, SHA-256 `4768daa5cb44e7bee37fbb69c36063800164d9e9e8c852e5b3c77bc88ce9ac76`, 5,597,449 bytes, runtime spec `nanodet-plus-m-320-v1`. PRODUCT-ADR-RECOGNITION-002 records this pin and explicitly leaves the later mixed-result real-capture fine-tune as evaluation evidence rather than silently changing production identity.
- `vendor/recognition-models/provenance.json` now declares one coherent `recognition-v1-2026-08-27` three-model set with exact artifact names, SHA-256 values, byte sizes, runtime specs, checkpoint provenance, and classifier normalization metadata.
- `src/recognition/model-runtime/production-model-set.ts` now exposes the build-pinned `PRODUCTION_RECOGNITION_MODEL_SET` using Vite asset URLs for the three committed vendor ONNX files and the accepted provider preference `wasm-simd -> wasm-threaded -> webgl`.
- `test/recognition-production-model-artifacts.test.ts` verifies manifest validation plus SHA-256/byte identity against the committed vendor package.
- `test/e2e/recognition-production-artifacts.html` and `recognition-production-artifacts-main.ts` provide a browser-only real-artifact harness. It initializes the public `createProductionRecognitionRuntime()` path, evaluates a bounded blank-frame full-pipeline fixture, then separately executes deterministic base/red-five preprocessing tensors through real initialized classifier sessions and reports semantic labels/logits and provider diagnostics.
- `test/e2e/recognition-production-artifacts.spec.ts` drives that harness in Chromium. Its first execution is intentionally used to record/freeze the exact deterministic classifier labels/logits and blank-frame semantic snapshot before the final R06 verdict.

The remaining execution step is to copy the already-verified `.local` ONNX bytes into the canonical `vendor/recognition-models/` package, run the artifact/unit/build/browser gate, freeze the first observed bounded fixture outputs as expected values, and re-run the gate. Until that observed execution completes, R06 remains `blocked` rather than claiming PASS from source declarations alone.

### First real-artifact execution and fixture freeze: 2026-08-27

The canonical vendor package was materialized and the first browser execution completed successfully. The earlier artifact-availability and model-set blockers are therefore cleared; R06 is now `in_progress` pending the frozen-fixture rerun and the complete focused Recognition suite.

Observed gate results from `product/frontend`:

- `npm test -- recognition-production-model-artifacts.test.ts recognition-model-runtime.test.ts` — **PASS**: 2 files / 14 tests.
- `npm run typecheck` — **PASS**.
- `npm run lint` — **PASS**: `Architecture import boundaries: OK (53 source files checked)`.
- `npm run build:e2e` — **PASS**.
- `npx playwright test test/e2e/recognition-production-artifacts.spec.ts` — **PASS**: 1 Chromium test.

The browser harness loaded all three production ONNX artifacts through the public production runtime. Provider selection was `wasm-simd` for detector, tile classifier, and red-five classifier, with no failed providers.

The first observed bounded fixture outputs were frozen into `test/e2e/recognition-production-artifacts.spec.ts` as exact expected results:

- deterministic base-classifier fixture -> semantic label `invalid`; 35 rounded logits are asserted exactly in the E2E test.
- deterministic red-five fixture -> semantic label `red`; rounded logits `[-8.893241, 9.968656]`.
- blank 320x320 full-pipeline fixture -> zero observations, zero meld groups, empty completed hand/dora/meld draft, and commit eligibility `{ kind: 'ineligible', reason: 'insufficient-visible-tiles' }`.
- the same blank-frame execution exercises the real detector session plus production preprocessing and detector decode/postprocess path; the expected semantic candidate result is an empty candidate/observation set.

The successful harness reported model-set version `recognition-v1-2026-08-27` and runtime specs `nanodet-plus-m-320-v1`, `c8-tile-35-v1`, and `c8-red-five-v1` for the three initialized sessions.

Before declaring final PASS, rerun the E2E test against these now-frozen expected values and execute the complete current `recognition*.test.ts*` focused suite plus strict typecheck/lint. No model/artifact blocker remains.

### Final-gate rerun: 2026-08-27

The frozen real-artifact browser gate remained stable and passed exactly, but the first complete focused-suite rerun exposed one stale v2 normalization literal in `test/recognition-services.test.ts` rather than a production-runtime failure.

- `npm test -- recognition` — **89/90 tests PASS; 1 FAIL**. The only failure was the one-frame composition assertion still expecting the superseded v2 base normalization `(mean=0.68306223733377514, std=0.27237886485683077)`. The observed tensor value matched the selected v3_jp189 runtime normalization instead.
- `npm run typecheck` — **PASS**.
- `npm run lint` — **PASS**: `Architecture import boundaries: OK (53 source files checked)`.
- `npm run build:e2e` — **PASS**.
- `npx playwright test test/e2e/recognition-production-artifacts.spec.ts` — **PASS** against the frozen expectations. All three models selected `wasm-simd` with no failed providers; base fixture remained `invalid`, red-five fixture remained `red`, and the blank-frame snapshot remained exactly unchanged.

The stale unit-test expectation has now been updated to the production v3_jp189 values `mean=0.6815832403977466`, `std=0.2725553681973969`. No production code was changed for this correction.

### Final verification verdict: 2026-08-27

**Overall verdict: PASS**

The complete focused Recognition suite was rerun after the stale v2 expectation was corrected:

- `npm test -- recognition` — **PASS**: 11 test files / 90 tests.
- `npm run typecheck` — **PASS** on the final production implementation; the only subsequent change was the numeric expected normalization literal in the test described above.
- `npm run lint` — **PASS**: `Architecture import boundaries: OK (53 source files checked)`; no production architecture change followed this result.
- `npm run build:e2e` — **PASS**.
- `npx playwright test test/e2e/recognition-production-artifacts.spec.ts` — **PASS** against the frozen real-artifact expectations.

Final required-check results:

| check | observed result |
|---|---|
| detector ONNX load/runtime-spec validation | **PASS** — pinned `nanodet-plus-m-320-composite-augmented.onnx`, SHA-256 `4768daa5cb44e7bee37fbb69c36063800164d9e9e8c852e5b3c77bc88ce9ac76`, runtime spec `nanodet-plus-m-320-v1`; browser provider `wasm-simd`. |
| base C8 ONNX load/runtime-spec validation | **PASS** — pinned `tile-c8-gray35-v3-jp189.onnx`, SHA-256 `b8a8fa3ff6c6d1e944a7593fa0afc947e0cd2513fb79ca46e5f8fcd6e19c97d0`, runtime spec `c8-tile-35-v1`; browser provider `wasm-simd`. |
| red-five ONNX load/runtime-spec validation | **PASS** — pinned `red-five-c8-rgb-warmaug.onnx`, SHA-256 `c2b780f682d84bf186db90290050f8b05016c3e8058de559eea679a28eeb80c6`, runtime spec `c8-red-five-v1`; browser provider `wasm-simd`. |
| bounded detector fixture inference/decode | **PASS** — fixed blank 320x320 frame executes the real detector/preprocess/decode/postprocess path and yields the frozen empty candidate/observation set. |
| bounded base-classifier fixture inference | **PASS** — deterministic fixture yields frozen semantic label `invalid` and the exact 35 rounded logits asserted by the E2E test. |
| bounded red-five fixture inference | **PASS** — deterministic fixture yields frozen semantic label `red` with logits `[-8.893241, 9.968656]`. |
| bounded full one-frame pipeline fixture | **PASS** — frozen result: no observations/meld groups, empty draft, commit eligibility `ineligible / insufficient-visible-tiles`. |
| complete Recognition focused test suite | **PASS** — 11 files / 90 tests. |
| strict typecheck/lint/architecture gate | **PASS**. |

R06 therefore satisfies the L2 actual-artifact/public-runtime acceptance gate. Target-device performance and final iPhone acceptance remain outside this Task as defined.
