# PRODUCT-INV-RECOGNITION-024: Evaluate NanoDet effective batch size

- **status**: concluded
- **date**: 2026-10-04
- **trigger**: PRODUCT-INV-RECOGNITION-023 found that reducing effective batch improved Plain classifier robustness, reinforcing earlier evidence that training batch size should not be chosen primarily for GPU throughput. NanoDet therefore required the same training-recipe check before architecture comparisons continued.
- **scope**: Compare NanoDet Plus M320 GhostPAN baseline training at physical and effective batch 24, 48, and 96 for 60 epochs on the corrected Japanese-riichi COCO-only corpus, using native NanoDet training and the product-facing bbox-pathology evaluation.
- **non_scope**: Batch sizes below 24, optimizer-update-matched comparisons, NanoDet-R or theta branches, architecture changes, joint or fine-tune data recipes, learning-rate scaling, augmentation changes, multi-seed confirmation, and production promotion.
- **source_refs**:
  - PRODUCT-INV-RECOGNITION-022
  - PRODUCT-INV-RECOGNITION-023
- **follow_up_candidates**:
  - Lower-batch NanoDet sweep at batch 6 and 12
  - Optimizer-update-matched NanoDet batch-size comparison

## Investigation scope

Determine whether the effective batch size used for NanoDet materially changes detector quality under a fixed 60-epoch training recipe.

The experiment compares batch 24, 48, and 96.

The architecture, source corpus, optimizer settings, augmentation, seed, pretrained initialization, and native NanoDet training semantics remain fixed.

The primary product-facing evaluation is bbox-pathology-v2.

## Out of scope

- Batch 6 or 12.
- Equalizing optimizer-update count across batch sizes.
- NanoDet-R or angle regression.
- GhostPAN or backbone architecture changes.
- Joint corrected-COCO plus product-domain training.
- Fine-tuning from a corrected-COCO checkpoint.
- Learning-rate scaling.
- Augmentation changes.
- Multi-seed confirmation.
- Production promotion.
- GPU-efficiency optimization.

## Background

PRODUCT-INV-RECOGNITION-023 found that reducing the Plain classifier effective batch improved robustness and fine-grained confusion metrics.

That result reinforced an older project warning from PRODUCT-INV-RECOGNITION-005: large training batches can be counterproductive even when they are operationally convenient.

The detector training recipe had also been using relatively large batches where GPU utilization was an operational concern.

Before comparing additional detector architectures, the training recipe therefore needed a direct batch-size check.

The detector experiment used the corrected Japanese-riichi COCO source prepared after PRODUCT-INV-RECOGNITION-022.

The first source-training implementation was excluded from evidence because its handwritten EMA update did not match native NanoDet ExpMovingAverager semantics.

The valid batch sweep uses native NanoDet TrainingTask, native EMA, native optimizer and cosine schedule, and native COCO evaluation.

## What was investigated

The primary sealed Study was:

| evidence | ref |
|---|---|
| Study | nanodet/coco-batch-sweep-v1 |
| StudyResult | nanodet/run-7eeac0f4e5df4950a43d302ad10a1820 |
| source commit | 2b37441953c97f112f1ba1bad6d67fa349475a10 |

The original batch-96 child was killed by host RAM OOM on dev-wsl-gpu3060 after about two epochs.

That partial execution produced no authoritative MLDB harness projection or final weights and is not used as model evidence.

Batch 96 was rerun through:

| recovery evidence | ref |
|---|---|
| Study | nanodet/coco-batch96-recovery-v1 |
| StudyResult | nanodet/run-efb80d49a87c408bb633077bd7d5d2c1 |
| source commit | 285ec4d153861cbd96167a7c59d0ba69f67896d2 |

The recovery completed all 60 epochs and bbox-pathology-v2.

The three valid conditions share:

- NanoDet Plus M320;
- ShuffleNetV2 1.0x backbone;
- GhostPAN baseline neck;
- corrected coco_mahjong_jp_v2 train/valid source;
- official NanoDet Plus M320 pretrained initialization;
- 60 epochs;
- learning rate 0.001;
- weight decay 0.05;
- cosine minimum learning rate 0.00005;
- 500 warmup steps;
- seed 42;
- no gradient accumulation;
- physical batch equal to effective optimizer batch.

The product-facing diagnostic uses the frozen 79-image nanodet/mahjong-composite-pathology-320-v1 validation split.

## Findings

### Native COCO validation

AP50 saturated at approximately 0.9901 for all three conditions and did not separate the recipes.

Small-object AP showed a strong batch-size gradient.

| batch | AP small at epoch 60 | AP medium at epoch 60 |
|---:|---:|---:|
| 24 | **0.428908** | **0.988908** |
| 48 | 0.399231 | 0.988297 |
| 96 | 0.301552 | 0.986295 |

Batch 96 is materially weaker on small-object AP.

Batch 24 is also better than batch 48 on the same metric.

### Product-facing bbox pathology

The product duplicate-suppression stage produced:

| metric | bs24 | bs48 | bs96 |
|---|---:|---:|---:|
| precision | 0.709698 | **0.710928** | 0.631706 |
| recall | **0.955085** | 0.948305 | 0.888136 |
| F1 | **0.814306** | 0.812636 | 0.738288 |
| duplicate GT rate | **0.251695** | 0.297458 | 0.318644 |
| multi-GT prediction rate | **0.031486** | 0.036213 | 0.037372 |
| spurious prediction rate | 0.012594 | 0.014612 | **0.010247** |
| missed GT rate | **0.006780** | 0.018644 | 0.031356 |
| bbox excess p95 | 28 | **27** | 35 |
| bbox excess max | 38 | **33** | 43 |
| prediction count | 1,588 | **1,574** | 1,659 |

Ground-truth count is 1,180.

Batch 24 and 48 are close on product F1.

Batch 24 has the best recall, duplicate rate, multi-GT rate, and miss rate.

Batch 48 has slightly better precision and bbox-excess counts.

Batch 96 is clearly weaker across the main product-facing quality metrics.

### Candidate-count behavior

The batch-96 regression is not explained by simply producing more raw candidates.

Raw candidate p95 was:

| batch | raw candidate p95 |
|---:|---:|
| 24 | 480 |
| 48 | 466 |
| 96 | 379 |

Batch 96 generated fewer raw candidates while still missing more GT and producing worse product-facing F1.

The regression therefore reflects prediction quality and allocation, not only candidate volume.

## Cross-cutting observations

The detector result is directionally consistent with PRODUCT-INV-RECOGNITION-023.

In both model families, the larger effective batch was not the strongest quality recipe.

For NanoDet, AP50 is too saturated to diagnose the difference.

Small-object AP and the product-facing pathology metrics are more discriminative.

The 60-epoch comparison does not isolate batch size from optimizer-update count.

With the same dataset and epoch count, smaller batches perform more optimizer updates.

The current evidence therefore supports the practical statement that smaller batches are better under this 60-epoch recipe.

The current evidence does not yet establish whether smaller-batch gradient statistics or additional optimizer updates are the main cause.

The improvement continues through batch 24.

That trend makes a follow-up at batch 12 and 6 reasonable, but those conditions are outside this investigation.

## Follow-up judgment candidates

- Whether batch 24 should become the default starting point for the next NanoDet architecture experiments.
- Whether batch 12 or 6 improves the fixed-epoch recipe further.
- Whether batch-size ordering persists when optimizer-update count is matched.
- Whether the preferred corrected-COCO batch recipe should be fixed before joint and fine-tune data-recipe comparisons.

## Recommendation

Batch 96 does not appear appropriate as the quality-first baseline for this NanoDet training recipe.

Batch 24 or 48 appears preferable within the tested range.

Batch 24 has the stronger product-facing error profile, while batch 48 is close on aggregate F1.

A lower-batch follow-up appears justified because the measured trend has not yet reached a clear quality knee.

A final causal claim about batch size should wait for the optimizer-update-matched comparison.

## Follow-up artifact candidates

- A follow-up investigation for batch 6 and 12 under the fixed 60-epoch recipe.
- The same follow-up should include optimizer-update-matched conditions so batch-size effects can be separated from update-budget effects.
- A later detector training-recipe ADR only if one batch policy is adopted across subsequent NanoDet experiments.

## Open questions

- Does the fixed-epoch improvement continue below batch 24?
- Does the ordering remain when optimizer-update count is matched?
- Is the batch-24 advantage stable across additional seeds?
- Does the preferred batch remain best after product-domain joint training or fine-tuning is introduced?
