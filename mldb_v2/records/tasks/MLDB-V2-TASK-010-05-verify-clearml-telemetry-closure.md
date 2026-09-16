# MLDB-V2-TASK-010-05: Verify actual ClearML telemetry closure

- **status**: completed
- **date**: 2026-09-15
- **work_item**: MLDB-V2-WORK-010
- **task_type**: verification
- **depends_on**: [MLDB-V2-TASK-010-03, MLDB-V2-TASK-010-04]
- **outputs**: focused conformance, bounded actual ClearML GPU/chart verification, W010 closure evidence

## Boundary

Verify the completed backend-neutral telemetry path against actual ClearML GPU execution without adding new telemetry semantics or changing canonical Result authority. W009 remains completed/PASS and is not reopened by this task.

## Focused/fake conformance

Focused closure coverage passed **18 tests in 7.82s**. The selected tests cover accepted scalar recording, Training zero-event enforcement, Training delivery-failure isolation, validated Evaluation metric automatic projection, Evaluation delivery-failure isolation, exact ClearML scalar mapping, CommonExecutionHarness telemetry threading, detector v3 telemetry semantics, and classifier v4 telemetry semantics.

No production ClearML outage or credential destruction was used for failure-isolation verification.

## Bounded actual Study preparation

Two new bounded Studies were authored from the W009 production shapes without mutating historical Studies:

- `rotated-fcos/w010-telemetry-smoke-v1`: W009 detector architecture/corpus/evaluation, Train Protocol `rotated-fcos/rotated-fcos-train-gpu-v3`, seed 42, 3 epochs.
- `tile-classifier/w010-telemetry-smoke-v1`: W009 classifier architecture/corpus/evaluation, Train Protocol `tile-classifier/tile-shape-train-gpu-v4`, seed 42, 3 epochs, `cache_device=cuda` preserved.

Both Study definitions pass structural `mldb validate study ... --json` validation.

## Resolved source-pinning blocker and actual closure

The blocker was repaired on feature branch `feature/w010-telemetry-e2e` by committing only the required T010 execution source and bounded Study inputs. Pinned source commit is `0cc3828702bbb7630675ba3de3e62a6a0b79bfb7`, pushed to the matching remote branch. Formal plans for both W010 Studies resolve to that exact commit; `source_not_pinned` is no longer present.

Actual ClearML execution used queue `default`, worker `bugrat-gpu0`, and one `NVIDIA GeForce RTX 3090` (24 GB). Detector Training Task `64622d3c84dc41b48bccdeab8cf7abe2` and Evaluation Task `52bfc1104c3a44289bcd2eaa10007de2` completed. Training Charts contain `validation/{f1,loss,mean_iou,recall}` at steps 1,2,3; Evaluation group `standard-operating-point` contains validated numeric metrics at step 0. Canonical StudyResult `rotated-fcos/run-82b12decf06840708fefc5dde4ccaf52` is `completed`; its TrainingResult, Model, and EvaluationResult are completed. Weights are `s3://mldb-w004-smoke/mldb-v2/35091979e61327a62bcaa013b2dca619d46d28d661959adf02a4058bbc6c8344/weights.pt`, 910414 bytes, SHA-256 `0919e3db9a25f5185fd5e19210c2b9b2e299b34bac7a3435663e5774c5d14c7a`.

Classifier Training Task `6aac714935e149e29d2dff604c190680` and Evaluation Task `29d064e1dc0a48119072a1a10a449206` completed on the same worker/GPU. Training Charts contain `optimization/cross_entropy_loss` at steps 1,2,3; Evaluation group `angle-robustness` contains validated numeric metrics at step 0. Canonical StudyResult `tile-classifier/run-bc661cc52608479b9edcf0cdb91933e2` is `completed`; its TrainingResult, Model, and EvaluationResult are completed. Training preserved `cache_device=cuda`. Weights are `s3://mldb-w004-smoke/mldb-v2/ecd3c40453a6684f0cf4a38566be79671385309ed77028462b28151e748da773/weights.pt`, 1504834 bytes, SHA-256 `3a4824f29a736f2b228f18e75e756ec64878a3c20d4b16fdc090fedcdb248109`.

Canonical Result shapes remain unchanged and contain no telemetry payloads; telemetry exists only as backend operational projection. The final full regression passed: `1384 passed, 3 skipped in 363.73s`. W009 remains completed/PASS.

**Closure:** T010-05 is completed. W010 is completed. Execution telemetry and observability is closed.
