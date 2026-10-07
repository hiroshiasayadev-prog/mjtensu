# PRODUCT-INV-RECOGNITION-028: Evaluate lower NanoDet batches and update budget

- **status**: concluded
- **date**: 2026-10-05
- **trigger**: PRODUCT-INV-RECOGNITION-024 found that reducing NanoDet effective batch from 96 through 48 to 24 improved small-object COCO validation and product-facing bbox pathology. The improvement had not reached a clear knee at batch 24, while the fixed-epoch comparison also changed optimizer-update count. Lower batches and an update-budget comparison were therefore required.
- **scope**: Extend the corrected-COCO-only NanoDet Plus M320 GhostPAN baseline from batch 24 to batch 12 and 6, compare all three under the 60-epoch recipe, and compare batch 6, 12, and 24 at the same nominal total optimizer-update count.
- **non_scope**: Batch sizes above 24, architecture changes, NanoDet-R or theta branches, joint or fine-tune data recipes, learning-rate scaling, augmentation changes, multi-seed confirmation, production promotion, and redesign of the native NanoDet epoch-based detach schedule.
- **source_refs**:
  - PRODUCT-INV-RECOGNITION-024
- **follow_up_candidates**:
  - Training-dynamics investigation around batch schedule, detach timing, and checkpoint selection
  - Corrected-COCO to product-domain joint or fine-tune recipe investigation

## Investigation scope

Determine whether lowering NanoDet effective batch below 24 improves the practical 60-epoch training recipe.

Separate that practical question from a second question: whether the batch-24 result can be explained only by a different number of optimizer updates.

The fixed-epoch comparison uses batch 6, 12, and 24 for 60 epochs.

The nominal matched-update comparison uses:

| batch | epochs | steps per epoch | nominal total updates |
|---:|---:|---:|---:|
| 6 | 15 | 2,024 | 30,360 |
| 12 | 30 | 1,012 | 30,360 |
| 24 | 60 | 506 | 30,360 |The batch-24 result is the valid result already recorded by PRODUCT-INV-RECOGNITION-024.

## Out of scope

- Repeating batch 48 or 96.
- Architecture changes.
- NanoDet-R or angle regression.
- Joint corrected-COCO plus product-domain training.
- Fine-tuning on product-domain composites.
- Learning-rate scaling.
- Augmentation changes.
- Multi-seed confirmation.
- Production promotion.
- Changing the native NanoDet detach schedule as part of this experiment.

## Background

PRODUCT-INV-RECOGNITION-024 showed a clear quality loss at batch 96.

Batch 48 and 24 were substantially stronger, and batch 24 improved small-object AP plus several product-facing error metrics.

That result justified testing batch 12 and 6.

However, a fixed-epoch batch sweep changes the number of optimizer steps per epoch.

For the corrected COCO train split of 12,144 images, the practical 60-epoch recipes perform approximately:

| batch | updates per epoch | updates in 60 epochs |
|---:|---:|---:|
| 6 | 2,024 | 121,440 |
| 12 | 1,012 | 60,720 |
| 24 | 506 | 30,360 |A second comparison therefore reduced epoch counts for batch 6 and 12 so all three conditions reached 30,360 nominal optimizer updates.

The comparison uses the native NanoDet trainer.

Each training sample is reshuffled each epoch and passes through stochastic scale, stretch, translate, flip, brightness, contrast, and saturation augmentation.

The learning-rate scheduler is cosine and advances once per epoch.

NanoDet Plus also changes auxiliary-training behavior at `detach_epoch=10`. From epoch 10 onward, the auxiliary path uses detached backbone and FPN features.

This epoch-based transition means the matched-total-update comparison does not equalize every training dynamic.

Before detach, the nominal update counts are approximately:

| batch | updates before epoch-10 detach |
|---:|---:|
| 6 | 20,240 |
| 12 | 10,120 |
| 24 | 5,060 |

The matched-update experiment therefore tests a practical training-budget hypothesis. It is not a fully isolated causal batch-size experiment.

## What was investigated

### Fixed 60-epoch lower-batch Study

| evidence | ref |
|---|---|
| Study | `nanodet/coco-small-batch-fixed60-v1` |
| StudyResult | `nanodet/run-be68d960c5fe4ef8914f2e6bb2fd35cd` |
| source commit | `b54dad53e9d6ff61de0cc98bc323033bb02da8b9` |This Study trained batch 6 and 12 for 60 epochs and ran `bbox-pathology-v2`.

The batch-24 reference comes from PRODUCT-INV-RECOGNITION-024.

### Nominal matched-update Studies

| condition | Study | StudyResult |
|---|---|---|
| bs6 x 15 epochs | `nanodet/coco-bs6-update-matched-v1` | `nanodet/run-1b0d9e34378149bea97dacd824d03c9b` |
| bs12 x 30 epochs | `nanodet/coco-bs12-update-matched-v1` | `nanodet/run-cefde4b4f0be468ca99563588e95a49b` |
| bs24 x 60 epochs | reference from PRODUCT-INV-RECOGNITION-024 | `nanodet/run-7eeac0f4e5df4950a43d302ad10a1820` |

The bs12 matched Study initially left its bbox-pathology task in ClearML `created` state after the parent Pipeline became non-responsive.

The training result and model artifact remained valid.

The existing MLDB-owned evaluation task was later enqueued on `old-gpu3090`, completed successfully, and was reconciled into the canonical StudyResult.

No retraining was required.

## Findings

### Fixed 60-epoch comparison

Native COCO validation showed:

| batch | AP small at epoch 60 | AP medium at epoch 60 |
|---:|---:|---:|
| 6 | 0.453051 | **0.989997** |
| 12 | **0.483306** | 0.989991 |
| 24 | 0.428908 | 0.988908 |The lower batches improved small-object AP relative to batch 24.

Batch 12 produced the strongest measured AP-small value.

The product-facing bbox pathology showed a different ordering:

| metric | bs6 | bs12 | bs24 |
|---|---:|---:|---:|
| precision | 0.706943 | 0.698639 | **0.709698** |
| recall | 0.897458 | 0.913559 | **0.955085** |
| F1 | 0.790889 | 0.791774 | **0.814306** |
| duplicate GT rate | **0.214407** | 0.218644 | 0.251695 |
| multi-GT prediction rate | 0.022029 | **0.017498** | 0.031486 |
| spurious prediction rate | 0.012684 | 0.014906 | **0.012594** |
| missed GT rate | 0.031356 | 0.029661 | **0.006780** |
| bbox excess p95 | **26** | **26** | 28 |
| bbox excess max | **35** | 41 | 38 |
| prediction count after product | 1,498 | 1,543 | 1,588 |

Batch 6 and 12 reduced duplicate-related pathology.

They also lost materially more GT detections.

Batch 24 retained substantially higher product-facing recall and F1.

The fixed-epoch result therefore exposes a trade-off rather than a monotonic smaller-batch win.

### Nominal matched-update comparison

All three conditions nominally perform 30,360 optimizer updates.| condition | AP small at final validation | product precision | product recall | product F1 | duplicate GT | miss |
|---|---:|---:|---:|---:|---:|---:|
| bs6 x 15ep | **0.442072** | 0.716968 | 0.905932 | 0.800449 | **0.211864** | 0.033051 |
| bs12 x 30ep | 0.413617 | **0.720105** | 0.928814 | 0.811251 | 0.263559 | 0.022034 |
| bs24 x 60ep | 0.428908 | 0.709698 | **0.955085** | **0.814306** | 0.251695 | **0.006780** |

Equalizing nominal total optimizer updates does not remove the batch-24 product-facing advantage.

The result does not support the explanation that batch 24 only won because its fixed-epoch run received more or fewer total updates.

Batch 6 retains the lowest duplicate rate, but its recall and miss rate remain weaker.

Batch 12 moves closer to batch 24 under the matched budget, but batch 24 still has the strongest F1 and recall.

### Training duration within the lower-batch recipes

The product-facing result for the lower batches is not improved by simply extending the same source-domain training budget.

| batch | shorter matched-budget run | product F1 | 60-epoch run | product F1 |
|---:|---:|---:|---:|---:|
| 6 | 15 epochs | **0.800449** | 60 epochs | 0.790889 |
| 12 | 30 epochs | **0.811251** | 60 epochs | 0.791774 |

These are separate runs with different cosine schedule horizons.

They are not early checkpoints from the corresponding 60-epoch runs.

The result therefore does not prove ordinary late-epoch overfitting.

It does show that more corrected-COCO-only optimization under the lower-batch recipes does not monotonically improve the product-domain pathology metric.## Cross-cutting observations

The best source-domain and product-domain metrics no longer select the same batch.

Batch 12 is strongest on AP small at 60 epochs.

Batch 24 is strongest on product-facing recall and F1.

Batch 6 and 12 reduce duplicate behavior but trade that gain for more misses.

This divergence suggests that corrected-COCO validation alone is insufficient for selecting the final product-facing detector recipe.

The lower-batch results also show that total optimizer-update count is not a sufficient explanation for the batch ordering.

However, the native NanoDet `detach_epoch=10` transition remains epoch-based.

The nominal matched-update conditions therefore differ substantially in the number of updates performed before auxiliary features are detached.

The cosine scheduler is also epoch-based, although scaling the epoch horizon with batch size keeps its normalized progress broadly aligned with total training progress.

The current evidence supports a practical knee near batch 24 for the corrected-COCO-only product-facing baseline.

The current evidence does not establish a universal optimal batch size.

A later training-dynamics experiment could test whether low-duplicate behavior from smaller batches and high-recall behavior from batch 24 can be combined through staged batch scheduling, detach-timing control, checkpoint selection, or weight interpolation.

## Follow-up judgment candidates

- Whether batch 24 should be the quality-first corrected-COCO baseline for subsequent NanoDet architecture comparisons.
- Whether product-facing checkpoint selection should become a separate held-out criterion rather than relying only on source-domain COCO validation.
- Whether staged training can combine the lower duplicate rate of batch 6/12 with the recall of batch 24.- Whether `detach_epoch` should be expressed in optimizer-update units for future causal batch-size experiments.
- Whether the next highest-value experiment is data-recipe transfer from corrected COCO into the product-domain composite distribution.

## Recommendation

Batch 24 appears preferable as the next corrected-COCO-only NanoDet baseline when product-facing behavior is the primary criterion.

Batch 12 appears preferable only if source-domain AP-small is treated as the dominant objective.

Further reduction below batch 12 does not appear useful as a simple fixed recipe.

Further batch-only sweeping appears lower value than testing training dynamics or product-domain transfer.

Any claim that batch 24 is intrinsically optimal should wait for experiments that control the epoch-based detach transition and other schedule effects.

## Follow-up artifact candidates

- A training-dynamics investigation for staged batch size, detach timing, checkpoint trajectory, or weight interpolation.
- A data-recipe investigation for corrected-COCO pretraining followed by product-domain fine-tuning.
- A later training-recipe ADR if one batch and scheduling policy is adopted for subsequent detector experiments.

## Open questions

- Can the lower duplicate rate of batch 6/12 be combined with batch-24 recall without adding inference cost?
- At which training epochs do product-facing duplicate rate and recall diverge for each batch?
- How much of the matched-budget result is caused by the fixed epoch-10 auxiliary detach boundary?
- Would expressing detach timing in optimizer updates change the batch ordering?
- Does weight interpolation between batch-12 and batch-24 checkpoints preserve the strengths of both models?
- Does product-domain fine-tuning remove the COCO AP-small versus product-pathology disagreement?
