# PRODUCT-INV-RECOGNITION-039: Compare NanoDet source-only, mixed, and deployment fine-tune training

- **status**: concluded
- **date**: 2026-10-07
- **trigger**: Corrected-jp_v2 source-only training, joint deployment replay, and deployment fine-tuning had all been executed as separate NanoDet experiments, but their results had not been compared as one training-data strategy investigation. Functional-video review also showed that none of the recipes made meld detection usable despite strong static metrics.
- **scope**: Compare the corrected-jp_v2-only batch-24 NanoDet baseline, joint corrected-jp_v2 plus deployment replay training, and 15-epoch deployment fine-tuning with corrected-jp_v2 replay 0/256/512/1024. Compare same-threshold bbox-pathology metrics, layout-disjoint real-capture metrics where available, production-equivalent functional-video metrics, and human review of meld detector overlays.
- **non_scope**: NanoDet architecture changes, NanoDet-R or OBB angle branches, classifier architecture changes, score-threshold optimization, NMS tuning, synthetic Blender corpus evaluation, physical-iPhone latency, or production promotion.
- **source_refs**:
  - PRODUCT-INV-RECOGNITION-010
  - PRODUCT-INV-RECOGNITION-022
  - PRODUCT-INV-RECOGNITION-024
  - PRODUCT-INV-RECOGNITION-029
  - PRODUCT-INV-RECOGNITION-035
- **follow_up_candidates**:
  - Expand meld-focused detector training coverage with positive layout variation and explicit no-meld hard negatives.
  - Re-evaluate the expanded corpus against the current joint-mixed and deployment-fine-tune references.
  - Separate remaining data-coverage failures from architecture failures before using NanoDet-R as the explanation for current meld pathology.

## Investigation scope

Compare three ways of introducing deployment-domain evidence into the current AABB NanoDet Plus M320 detector.

The comparison asks whether corrected jp_v2 should remain the only training source, whether deployment examples should be mixed into full training, or whether a corrected-jp_v2 model should first be trained and then adapted through a short deployment fine-tune.

The investigation also records whether any of the three strategies solves the meld failures visible in production-equivalent video.

## Out of scope

- Selecting or promoting a production detector.
- Changing GhostPAN, the NanoDet head, assignment, loss, or input size.
- Testing NanoDet-R, rotated boxes, or a theta branch.
- Retuning score threshold or NMS as part of this comparison.
- Comparing tile-classifier architectures.
- Evaluating the later Blender synthetic corpus.
- Defining the final composition or size of a future meld-focused corpus.

## Background

PRODUCT-INV-RECOGNITION-022 corrected legacy COCO annotation semantics before the current NanoDet training line.

PRODUCT-INV-RECOGNITION-024 then established the corrected-jp_v2 batch-24 model as the strongest practical source-only baseline under the 60-epoch recipe.

The next experiments introduced deployment-domain evidence in two different ways.

The joint-mixed run trained from the official NanoDet pretrained checkpoint using corrected jp_v2, fixed-layout composite replay, and repeated annotated real iPhone deployment composites in one 60-epoch run.

The fine-tune run instead started from the completed corrected-jp_v2 batch-24 model and adapted it for 15 epochs using fixed deployment composites and repeated real captures. Corrected-jp_v2 replay was swept at 0, 256, 512, and 1024 images per epoch to measure source-retention effects.

These experiments were originally executed for their local questions. This investigation compares them after the fact because the product question is broader: which training-data strategy produces a detector that localizes individual tiles reliably in the deployment layout?

Functional-video evidence makes that comparison necessary. Aggregate static metrics improved substantially, but meld overlays still showed split detections, duplicate detections, and background false positives that are not acceptable in the application.

## What was investigated

### Training strategies

| strategy | canonical evidence | initialization | training data | schedule |
|---|---|---|---|---|
| corrected jp_v2 only | `nanodet/run-7eeac0f4e5df4950a43d302ad10a1820` | official NanoDet Plus M320 pretrained | corrected jp_v2 | 60 epochs, batch 24 |
| joint deployment mix | `nanodet/run-57df6b047bba465f98ac45f0a6ae4ae3` | official NanoDet Plus M320 pretrained | corrected jp_v2 + one-pass fixed-layout composite replay + 20x replay of 32 annotated real iPhone composites | 60 epochs, batch 24 |
| deployment fine-tune | `nanodet/run-dae482c625b44a7081449ce018c83a13` | completed corrected-jp_v2 batch-24 model | fixed deployment composites + 20x real replay + corrected-jp_v2 replay sweep | 15 epochs, batch 24 |

The corrected-jp_v2-only model came from `nanodet/coco-batch-sweep-v1`, source commit `2b37441953c97f112f1ba1bad6d67fa349475a10`.

The joint-mixed Study was `nanodet/deployment-replay-bs24-v1`, source commit `3649ad03ea3d0e169eee2f5523a538facec52535`, runtime registry 8.

The deployment-fine-tune Study was `nanodet/jp-v2-deployment-finetune-replay-sweep-v1`, source commit `70b8951b8f9429a60b9c50837e1684889df43491`, runtime registry 8.

The fine-tune conditions were:

| trial | corrected-jp_v2 replay images per epoch |
|---|---:|
| trial-0001 | 0 |
| trial-0002 | 256 |
| trial-0003 | 512 |
| trial-0004 | 1024 |

The four fine-tune models were replayed through the sealed five-video detector-primary functional evaluation by `nanodet/run-4d3c8f082f734e5fb04d593e65888c10`, source commit `90fe8c9fb8c6b0c111a2c7736dc1293695c50f9b`, runtime registry 8.

### Evaluation boundary

The direct bbox-pathology comparison uses score threshold 0.35, NMS IoU 0.6, and the product duplicate-suppression pass on `nanodet/mahjong-composite-pathology-320-v1`.

The layout-disjoint real-capture comparison uses `nanodet/deployment-real-quality-v1` at score threshold 0.35. No result under that exact protocol exists for the corrected-jp_v2-only model, so the source-only row is not invented.

The directly comparable current functional-video results for joint mix and fine-tune also use detector threshold 0.35. The historical corrected-jp_v2-only functional run in PRODUCT-INV-RECOGNITION-035 used threshold 0.575 and is therefore contextual evidence rather than a same-operating-point comparison.

Human review covered the rendered detector overlays in addition to the aggregate metrics.

## Findings

### Same-threshold bbox pathology strongly favors deployment-domain training

At score threshold 0.35, the corrected-jp_v2-only model is much weaker on the product-facing pathology corpus than either deployment-domain strategy.

| strategy | jp replay | precision | recall | F1 | duplicate GT | spurious | miss |
|---|---:|---:|---:|---:|---:|---:|---:|
| jp_v2 only | n/a | 0.709698 | 0.955085 | 0.814306 | 0.251695 | 0.012594 | 0.006780 |
| joint mix | n/a | **0.939759** | **0.991525** | **0.964948** | **0.064407** | 0.027309 | **0.000000** |
| fine-tune | 0 | 0.851933 | 0.989831 | 0.915719 | 0.121186 | 0.030635 | **0.000000** |
| fine-tune | 256 | 0.879038 | **0.991525** | 0.931900 | 0.111864 | 0.023291 | **0.000000** |
| fine-tune | 512 | 0.906202 | 0.990678 | 0.946559 | 0.094915 | 0.017829 | **0.000000** |
| fine-tune | 1024 | 0.916144 | 0.990678 | 0.951954 | 0.086441 | 0.013323 | 0.002542 |

Adding deployment-domain data is therefore not a marginal change. Joint mixed training raises after-product F1 from 0.814306 to 0.964948 and reduces duplicate-GT rate from 0.251695 to 0.064407.

Within the fine-tune sweep, increasing corrected-jp_v2 replay improves the pathology profile almost monotonically. F1 rises from 0.915719 with no jp_v2 replay to 0.951954 at 1024 replay images. Duplicate-GT rate falls from 0.121186 to 0.086441.

The fine-tune trend indicates that retaining source-domain exposure matters. Pure deployment adaptation is not the strongest fine-tune recipe.

The best fine-tune condition still does not surpass joint mixed training on this evaluation. Joint mix retains higher F1 and lower duplicate-GT rate at the same detector threshold.

### Fine-tuning is strongest on the small layout-disjoint real holdout

The joint-mixed and fine-tune models were evaluated on the same held-out real-capture protocol.

| strategy | jp replay | F1 | precision | recall | clean image rate | GT coverage p10 | crop purity p10 | meld F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| joint mix | n/a | 0.984615 | 0.969697 | 1.000000 | 0.625 | 0.911292 | 0.913112 | 1.000000 |
| fine-tune | 0 | 0.984615 | 0.969697 | 1.000000 | 0.625 | 0.902245 | 0.925723 | 0.960000 |
| fine-tune | 256 | 0.984615 | 0.969697 | 1.000000 | 0.625 | 0.905837 | **0.932806** | 1.000000 |
| fine-tune | 512 | **0.996109** | **0.992248** | 1.000000 | **0.875** | 0.911750 | 0.928671 | 1.000000 |
| fine-tune | 1024 | **0.996109** | **0.992248** | 1.000000 | **0.875** | **0.918561** | 0.927379 | 1.000000 |

The 512- and 1024-replay fine-tunes are strongest on this holdout. Both reach F1 0.996109 and clean-image rate 0.875.

The 1024-replay model also has the best GT-coverage p10 at 0.918561.

This holdout contains only eight real captures from two held-out layouts. The near-saturated result is useful evidence for local crop quality but is not sufficient to override the larger pathology corpus or the five-video evaluation.

### Joint mixed training is strongest on the directly comparable functional-video metrics

The joint-mixed model and all four fine-tune candidates were evaluated through the same detector-primary five-video protocol at threshold 0.35.

| strategy | jp replay | all-region semantic F1 | meld semantic F1 | frame semantic exact | GT-exact streak3 take rate | product-confirmed exact take rate |
|---|---:|---:|---:|---:|---:|---:|
| joint mix | n/a | **0.899078** | **0.817069** | **0.260521** | **0.800** | **0.800** |
| fine-tune | 0 | 0.882343 | 0.739568 | 0.205745 | 0.600 | 0.400 |
| fine-tune | 256 | 0.890125 | 0.770229 | 0.219773 | 0.600 | 0.600 |
| fine-tune | 512 | 0.891827 | 0.778240 | **0.221109** | 0.600 | 0.400 |
| fine-tune | 1024 | 0.891457 | 0.783073 | 0.219773 | **0.800** | 0.600 |

More corrected-jp_v2 replay again helps the fine-tune meld metric. Meld semantic F1 rises from 0.739568 at zero replay to 0.783073 at 1024 replay images.

The joint-mixed model remains better than every fine-tune candidate on all-region semantic F1, meld semantic F1, frame semantic exact rate, and product-confirmed exact take rate.

The result is consistent with the bbox-pathology comparison: the best deployment fine-tune is competitive, but it is not an across-the-board replacement for joint mixed training.

### The historical source-only functional result was already poor on meld scenes

PRODUCT-INV-RECOGNITION-035 evaluated the corrected-jp_v2-only detector at threshold 0.575 rather than 0.35.

That run reached frame semantic exact rate 0.171009 and only one of five takes achieved an exact three-frame streak. The aggregate meld-exact rate was 0.229793, but the number was dominated by the no-meld take.

Among takes that contained melds, t2, t3, and t5 had zero meld-exact frames. t4 had only 48 exact meld frames out of 299.

The different threshold prevents a direct numerical comparison with the current 0.35 runs. The historical result still establishes that source-only training did not produce usable meld behavior.

### Human overlay review shows a tile-localization failure that aggregate metrics understate

Human review of the detector overlays found that meld behavior remains unusable even for the numerically stronger current models.

Representative failures include:

- one `4p` tile being split into two separate detections that are classified as `2p` and `2p`;
- multiple overlapping boxes being emitted over a white tile;
- many small boxes being emitted on background structures with no tile present;
- meld rows being partially localized even when the detector reacts to the correct general area.

These failures are more severe than a small aggregate F1 difference suggests. A detector that reacts to the approximate meld region but does not preserve one-box-per-tile localization cannot supply reliable classifier crops or stable meld semantics.

The visual pattern suggests a data-coverage hypothesis rather than proving one. The current detector may be learning that tiles are likely to exist in particular parts of the meld region while receiving insufficient evidence about individual tile boundaries and true negative meld backgrounds.

The hypothesis is consistent with both split/duplicate detections and background false positives. The current experiments do not isolate the cause strongly enough to claim that the hypothesis is proven.

## Cross-cutting observations

Deployment-domain evidence clearly helps. Both joint training and fine-tuning materially improve the product-facing detector compared with corrected jp_v2 only.

Fine-tuning is not uniformly superior to joint training. The 512/1024 fine-tunes are best on the small real-capture holdout, while joint mixed training is better on same-threshold bbox pathology and five-video semantic behavior.

Within fine-tuning, source replay is valuable. Increasing corrected-jp_v2 replay reduces duplicates and raises pathology F1 without creating a corresponding video regression.

None of the current strategies resolves the central meld failure seen in overlays. The difference between F1 0.95 and 0.96 does not make the detector usable when one physical tile can become multiple boxes or background can generate tile-sized predictions.

The current evidence therefore points beyond recipe weighting alone. The next data investigation should distinguish object-level tile-boundary coverage from region-location shortcuts.

A particularly important missing negative may be a valid capture with the meld destination region present but no meld tiles in it. If current training disproportionately associates the meld destination with positive tiles, explicit empty-meld examples can test whether background false positives are partly caused by that shortcut.

Positive meld coverage should also include white tiles, four-tile groups, varied spacing, angle, edge proximity, and the specific backgrounds that produced false detections in video.

## Follow-up judgment candidates

- Whether joint mixed training should remain the current AABB reference because it is strongest on same-threshold pathology and functional-video metrics.
- Whether the 1024-replay fine-tune should remain a secondary reference because it is strongest or tied strongest on the layout-disjoint real holdout.
- Whether the next detector iteration should prioritize meld-focused corpus expansion before another optimizer, replay-ratio, or architecture sweep.
- Whether empty-meld hard negatives and mined video false-positive backgrounds materially reduce small background boxes and duplicate detections.
- Whether remaining meld pathology after corpus expansion is strong enough evidence to justify an architecture change such as NanoDet-R.

## Recommendation

Joint mixed training appears preferable as the current AABB comparison reference because it has the strongest same-threshold bbox-pathology and functional-video results among the completed recipes.

The 1024-replay fine-tune appears useful as a second reference because it has the strongest fine-tune pathology profile and ties for the best held-out real-capture F1.

Neither model appears suitable as evidence that the meld problem is solved. Human overlay review shows structural localization failures that remain unacceptable for the product.

The next major experiment appears more useful as a meld-focused corpus intervention than as another narrow replay-ratio sweep.

That corpus should include explicit no-meld examples, hard-negative backgrounds from observed false positives, and broader positive variation around individual meld tile boundaries. White tiles and four-tile groups should be represented deliberately because they appear in observed failure modes.

The next comparison should keep the current joint-mixed and strongest fine-tune models as fixed historical references and use the same bbox-pathology, real-capture, and functional-video evaluations. A data intervention should be judged by whether it reduces the observed split, duplicate, and background-box failures, not only by aggregate F1.

## Follow-up artifact candidates

- A meld-focused detector-corpus investigation covering positive layout diversity, explicit empty-meld negatives, and mined hard-negative backgrounds.
- A new sealed corpus definition if that investigation produces an accepted dataset composition.
- A follow-up NanoDet training comparison that reuses the current joint-mixed and 1024-replay fine-tune results as historical references.
- A later NanoDet-R comparison only after the data-coverage question is tested or explicitly accepted as unresolved.

## Open questions

- How often is the meld destination empty in representative product usage, and what empty-meld fraction should be present in training?
- Are split detections primarily caused by insufficient tile-boundary variation, annotation geometry, or the AABB detector architecture?
- Which background structures are responsible for the repeated small false boxes in video, and can they be mined into stable hard negatives?
- Does increased white-tile and four-tile-group coverage remove the observed duplicate and split detections?
- After corpus expansion, does joint training remain stronger than deployment fine-tuning on functional video?
- Does any residual pathology justify NanoDet-R, or can the existing AABB architecture become usable with better data coverage?
