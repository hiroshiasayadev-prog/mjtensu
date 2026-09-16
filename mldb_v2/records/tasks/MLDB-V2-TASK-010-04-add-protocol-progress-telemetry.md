# MLDB-V2-TASK-010-04: Add protocol-specific progress telemetry

- **status**: completed
- **date**: 2026-09-15
- **work_item**: MLDB-V2-WORK-010
- **task_type**: verification
- **depends_on**: [MLDB-V2-TASK-010-02]
- **outputs**: reviewed classifier/detector telemetry-enabled Train Protocol revisions and integrated closure evidence

## Boundary

Close the protocol-specific telemetry work already implemented in the classifier and detector lanes. Do not add generic metric policy, backend SDK dependencies, new training semantics, or a detector-specific bookkeeping Task. T010-05 retains actual ClearML GPU/chart verification and W010 closure ownership.

## Classifier Protocol

Historical `tile-classifier/tile-shape-train-gpu-v3` remains tracked, sealed, clean, and public-verification valid. Telemetry-enabled `tile-classifier/tile-shape-train-gpu-v4` is sealed and differs from v3 only by epoch loss accumulation and one report after each completed epoch.

V4 telemetry semantics are exactly:
- `group = optimization`
- `series = cross_entropy_loss`
- `step = epoch + 1` (1-based completed epoch)
- `value = sample-weighted mean` of per-batch mean cross entropy over the completed epoch

The v4 focused test also proves same-seed learned-state equality with v3. No ClearML/backend/storage/transport import exists in the companion.

## Detector Protocol

Historical `rotated-fcos/rotated-fcos-train-gpu-v2` remains tracked, sealed, clean, and public-verification valid. Telemetry-enabled `rotated-fcos/rotated-fcos-train-gpu-v3` is sealed and differs from v2 only by reporting the already-computed validation key after each completed validation epoch and before the unchanged early-stop check.

V3 telemetry semantics are exactly:
- `group = validation`
- `series = f1`, `recall`, `mean_iou`, `loss`
- one point per series per completed validation epoch
- `step = epoch + 1` (1-based completed epoch)
- `loss = -key[3]`, presenting positive mean validation loss while leaving the model-selection tuple unchanged

Focused tests prove exact series/order, positive presentation loss, best-key/best-state preservation, stopping-epoch reporting without phantom future steps, and reporter failure propagation. No ClearML/backend/storage/transport import exists in the companion.

## Genericity and definition integrity

The generic `mldb_v2/src` and `mldb_v2/skeleton` telemetry/runtime surfaces contain none of the classifier/detector fixed telemetry series/group names. No universal loss, accuracy, epoch, classifier, or detector metric policy was introduced; each Protocol chooses its own observations and cadence.

Exact companion SHA-256 values match their sealed YAML implementation hashes:
- classifier v3: `4fa93f8397e3d7039b1d0e38db20364605fd0593f57f76d2cace81b5e09432f8`
- classifier v4: `6034c1a26b9ff7b05919f7abb3b8921565080b4492cedbb3c13a862bc548a234`
- detector v2: `ccabb6c437fb520b88ec0470f17d9f54655820bd9d51bd0e4053ed9107bf4441`
- detector v3: `b890ae242c27af9044c9e448ff14d13e4715d250303a61da25aa9c737d9f2ebc`

Public `mldb verify train-protocol ... --json` reports `valid: true`, empty diagnostics, and no repository issues for all four definitions.

## Verification evidence — 2026-09-15

- Classifier v4 focused telemetry tests: **5 passed in 5.01s**.
- Detector v3 focused telemetry tests: **5 passed in 3.74s**.
- Integrated protocol + executable loading/integrity/asset + definition lifecycle/sealing suite: **249 passed, 1 skipped in 33.06s**.
- `py_compile` for telemetry-enabled companions and focused tests: PASS.
- Companion import/dependency scan for ClearML/backend/storage/transport concerns: PASS.
- Generic fixed metric/cadence policy scan over telemetry and Train/Evaluation context/runtime surfaces: PASS.
- Exact SHA/sealed-status check and old/new parameter-surface equality: PASS.
- Owned-file trailing-whitespace scan: PASS.

## Record normalization

The mistaken `MLDB-V2-TASK-010-04A-add-classifier-progress-telemetry.md` parallel-lane record was renamed and consolidated into this sole formal `MLDB-V2-TASK-010-04` record. Its classifier evidence is preserved here together with detector closure evidence. No `04B` Task record exists or was created.

T010-04 is `completed`. W010 remains `planned` until T010-05 performs bounded actual ClearML GPU/chart integration verification.
