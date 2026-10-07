# PRODUCT-INV-RECOGNITION-029: Evaluate NanoDet score threshold across small-batch models

- **status**: concluded
- **date**: 2026-10-05
- **trigger**: PRODUCT-INV-RECOGNITION-028 found that batch 6 and 12 reduced duplicate-related pathology but lost materially more GT detections than batch 24 at the common score threshold of 0.35. A threshold sweep was required to determine whether those misses reflected lower confidence calibration rather than weaker detection behavior.
- **scope**: Reuse the existing corrected-COCO-only NanoDet batch-6, batch-12, and batch-24 60-epoch Models and evaluate score thresholds from 0.10 through 0.40 in 0.05 increments with the same bbox-pathology-v2 corpus, NMS, and product suppression logic.
- **non_scope**: Retraining, changing batch size or optimizer updates, score thresholds above 0.40, NMS tuning, architecture changes, NanoDet-R, data-recipe changes, production promotion, and selecting a final deployment threshold.
- **source_refs**:
  - PRODUCT-INV-RECOGNITION-028
- **follow_up_candidates**:
  - Fine score-threshold sweep above 0.40 on the batch-24 baseline
  - Joint score-threshold and NMS tuning only if score-threshold refinement leaves material pathology unresolved

## Investigation scope

Test whether the product-facing recall deficit of the batch-6 and batch-12 NanoDet models is mainly a score-calibration effect.

The working hypothesis was that smaller-batch models might emit useful boxes at lower confidence.

If so, lowering score threshold should recover recall while preserving some of their lower duplicate rate.

The comparison reuses existing 60-epoch models without retraining.

## Out of scope

- Any new training.
- Batch-size or update-budget changes.
- Thresholds above 0.40.
- NMS IoU changes.
- Product duplicate-suppression logic changes.
- Architecture or data-recipe changes.
- Production threshold selection.

## Background

PRODUCT-INV-RECOGNITION-028 showed a trade-off at the shared score threshold of 0.35.

Batch 6 and 12 produced fewer duplicate GT assignments than batch 24.

They also produced materially lower recall and higher miss rates.

This left two materially different explanations.

The lower-batch models could be detecting the same tiles with lower confidence, in which case threshold tuning could recover them.

Alternatively, the models could be producing a genuinely different prediction distribution that threshold tuning cannot repair.

A direct evaluation-only sweep can separate those explanations without retraining.## What was investigated

The sealed MLDB Study was:

| evidence | ref |
|---|---|
| Study | `nanodet/coco-small-batch-score-threshold-sweep-v1` |
| StudyResult | `nanodet/run-1c538203af1a4d4b8167758c940e84e6` |
| source commit | `1aa5f0fe8d4cde0a5ea21b10c4089b7cc109df8f` |

The Study reused three canonical 60-epoch Models:

- batch 6;
- batch 12;
- batch 24.

Each model was evaluated at:

`0.10 / 0.15 / 0.20 / 0.25 / 0.30 / 0.35 / 0.40`.

The following settings remained fixed:

- corpus: `nanodet/mahjong-composite-pathology-320-v1`;
- Evaluation Protocol: `nanodet/bbox-pathology-v2`;
- NMS IoU threshold: 0.6;
- max detections: 200;
- pathology overlap threshold: 0.25;
- duplicate overlap threshold: 0.8;
- product duplicate-suppression behavior.

The Study completed all 21 planned evaluations.

## Findings

### Recall could not be recovered by lowering threshold

The best observed product recall for each model across the tested threshold range was:

| batch | best recall | threshold |
|---:|---:|---:|
| 6 | 0.914407 | 0.20 |
| 12 | 0.927119 | 0.20 or 0.25 |
| 24 | **0.958475** | 0.10 or 0.25 |

Batch 6 and 12 never approached the batch-24 recall level.

Lowering threshold therefore did not recover the missing detections implied by the 0.35 comparison.

The lower-batch recall deficit is not explained by a simple global confidence shift.

### Lower thresholds traded misses for duplicate and spurious predictions

For batch 12:

| threshold | recall | F1 | duplicate GT | spurious | miss |
|---:|---:|---:|---:|---:|---:|
| 0.10 | 0.925424 | 0.604149 | 0.470339 | 0.122793 | 0.003390 |
| 0.20 | **0.927119** | 0.716907 | 0.333898 | 0.035791 | 0.011017 |
| 0.30 | 0.918644 | 0.764996 | 0.266949 | 0.021765 | 0.023729 |
| 0.35 | 0.913559 | 0.791774 | 0.218644 | 0.014906 | 0.029661 |
| 0.40 | 0.896610 | **0.806095** | **0.200000** | **0.013841** | 0.045763 |

Lowering threshold recovered only a small amount of recall.

The cost was a large increase in duplicate and spurious predictions.

Batch 6 showed the same pattern.### Batch 24 remained the strongest operating curve

For batch 24:

| threshold | precision | recall | F1 | duplicate GT | miss |
|---:|---:|---:|---:|---:|---:|
| 0.10 | 0.417343 | 0.958475 | 0.581491 | 0.494915 | 0.000000 |
| 0.20 | 0.589034 | 0.955932 | 0.728918 | 0.336441 | 0.000847 |
| 0.30 | 0.666470 | 0.955085 | 0.785092 | 0.278814 | 0.003390 |
| 0.35 | 0.709698 | 0.955085 | 0.814306 | 0.251695 | 0.006780 |
| 0.40 | **0.747502** | 0.950847 | **0.837001** | **0.215254** | 0.009322 |

Raising threshold from 0.35 to 0.40 improved batch-24 F1 from 0.814306 to 0.837001.

Duplicate GT rate fell from 0.251695 to 0.215254.

Recall changed only from 0.955085 to 0.950847.

The highest tested threshold therefore improved the aggregate product-facing operating point.

### Every model still improved in F1 at the upper sweep boundary

The best F1 for all three models occurred at threshold 0.40.

| batch | F1 at 0.35 | F1 at 0.40 |
|---:|---:|---:|
| 6 | 0.790889 | **0.806736** |
| 12 | 0.791774 | **0.806095** |
| 24 | 0.814306 | **0.837001** |

The sweep did not identify an interior F1 optimum.

The useful threshold range therefore extends above 0.40.

## Cross-cutting observations

The result rejects the simple explanation that batch 6 and 12 are merely more conservative versions of batch 24.

Their missing product detections cannot be recovered by accepting lower-confidence candidates without a large increase in duplicate and spurious predictions.

Batch 24 therefore remains preferable across the observed operating curves rather than only at the original threshold of 0.35.

Score threshold is nevertheless an important independent inference hyperparameter.

The original 0.35 threshold was not optimal for the batch-24 model on this pathology corpus.

The best tested batch-24 point occurred at the upper boundary, so a follow-up sweep above 0.40 is justified.

The threshold result does not change the source-domain AP-small findings from PRODUCT-INV-RECOGNITION-028.

It changes the interpretation of the product-facing recall gap: the gap reflects model prediction behavior more strongly than a simple confidence-calibration offset.

## Follow-up judgment candidates

- Whether the batch-24 product operating point continues improving above score threshold 0.40.
- Whether a finer threshold search identifies a stable F1/recall/duplicate knee.
- Whether one fixed product threshold remains appropriate after detector architecture or data-recipe changes.
- Whether NMS tuning is still necessary after the score threshold is selected.

## Recommendation

Batch 24 appears preferable to batch 6 and 12 across the tested score-threshold range.

Lowering threshold is not a useful remedy for the lower-batch recall deficit.

Threshold 0.35 also appears unnecessarily permissive for the current batch-24 model.

A finer batch-24 threshold sweep above 0.40 appears preferable to further threshold tuning of batch 6 or 12.

## Follow-up artifact candidates

- A batch-24-only fine score-threshold Study above 0.40 using the existing Model.
- A later detector runtime/default-threshold decision record if one operating point is adopted for product use.

## Open questions

- Where does batch-24 F1 peak above threshold 0.40?
- How much recall is acceptable to trade for lower duplicate rate in the product runtime?
- Does the preferred threshold remain stable after detector architecture or training-data changes?
- Would NMS tuning materially improve the selected threshold operating point, or merely move the same duplicate/recall trade-off?
