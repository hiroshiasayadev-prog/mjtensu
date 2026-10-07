# PRODUCT-INV-RECOGNITION-034: Refine NanoDet batch-24 score threshold

- **status**: concluded
- **date**: 2026-10-06
- **trigger**: PRODUCT-INV-RECOGNITION-029 found that the corrected-COCO NanoDet batch-24 model remained the strongest small-batch model and that its product-facing F1 was still increasing at the upper tested score threshold of 0.40. A finer sweep above 0.40 was required before using one threshold in functional video evaluation.
- **scope**: Reuse the existing NanoDet batch-24 60-epoch Model and evaluate score thresholds 0.400 through 0.600 in 0.025 increments under the same bbox-pathology-v2 corpus, NMS, and product duplicate-suppression rules.
- **non_scope**: Retraining, NMS tuning, architecture changes, data-recipe changes, thresholds outside 0.400-0.600, functional-video behavior, physical-device latency, or production-default promotion.
- **source_refs**:
  - PRODUCT-INV-RECOGNITION-028
  - PRODUCT-INV-RECOGNITION-029
- **follow_up_candidates**:
  - Test the bbox-selected operating point in the production-equivalent functional-video pipeline.
  - Revisit threshold selection using region-semantic E2E diagnostics rather than bbox-pathology F1 alone.

## Investigation scope

Locate the practical product-facing operating point of the current corrected-COCO NanoDet batch-24 model above the previous 0.40 sweep boundary.

The Study is evaluation-only. No model weights are changed.

## What was investigated

| evidence | ref |
|---|---|
| Study | `nanodet/coco-bs24-score-threshold-fine-v1` |
| Study Plan | `nanodet/coco-bs24-score-threshold-fine-v1-plan-8229e4744f72c15c` |
| Study Result | `nanodet/run-9d252f906256452d9ae4bd580016359a` |
| Model | `nanodet/run-7eeac0f4e5df4950a43d302ad10a1820-trial-0001-model` |
| source commit | `398481a4680238fcdbe5ba7eb27d5dc2a608814f` |
| runtime registry | `8` |

The fixed evaluation settings were:

- corpus: `nanodet/mahjong-composite-pathology-320-v1`;
- Evaluation Protocol: `nanodet/bbox-pathology-v2`;
- split: val;
- NMS IoU threshold: 0.6;
- max detections: 200;
- pathology overlap threshold: 0.25;
- duplicate overlap threshold: 0.8.

## Findings

### After-product operating curve

| threshold | precision | recall | F1 | duplicate GT | spurious | miss | bbox excess p95 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.400 | 0.747502 | 0.950847 | 0.837001 | 0.215254 | 0.012658 | 0.009322 | 25 |
| 0.425 | 0.771291 | 0.951695 | 0.852049 | 0.196610 | 0.012363 | 0.011017 | 22 |
| 0.450 | 0.790436 | 0.952542 | 0.863951 | 0.176271 | 0.011955 | 0.012712 | 19 |
| 0.475 | 0.815273 | 0.950000 | 0.877495 | 0.152542 | 0.011636 | 0.016102 | 17 |
| 0.500 | 0.830330 | 0.937288 | 0.880573 | 0.133898 | 0.012012 | 0.026271 | 14 |
| 0.525 | 0.839354 | 0.925424 | 0.880290 | 0.120339 | 0.011530 | 0.034746 | 12 |
| 0.550 | 0.858964 | 0.913559 | 0.885421 | 0.103390 | 0.010359 | 0.046610 | 9 |
| 0.575 | 0.883779 | 0.895763 | **0.889731** | 0.074576 | 0.010870 | 0.062712 | 5 |
| 0.600 | **0.899480** | 0.879661 | 0.889460 | **0.062712** | 0.010399 | 0.083051 | **4** |

### Bbox-pathology F1 peaks at 0.575

Threshold 0.575 produced the highest observed after-product F1, 0.889731.

Threshold 0.600 was essentially tied on F1 at 0.889460, but it reduced recall to 0.879661 and increased missed-GT rate to 0.083051.

Threshold 0.550 retained materially more recall at 0.913559 while giving F1 0.885421.

The useful operating region is therefore around 0.55-0.60 rather than a single sharply dominant point.

### Raising threshold suppresses duplicate pathology but buys that improvement with misses

From 0.400 to 0.575:

- duplicate-GT rate falls from 0.215254 to 0.074576;
- bbox-excess p95 falls from 25 to 5;
- precision rises from 0.747502 to 0.883779;
- recall falls from 0.950847 to 0.895763;
- missed-GT rate rises from 0.009322 to 0.062712.

The sweep therefore exposes a real recall-versus-duplicate trade rather than a free threshold improvement.

### Thresholding does not remove the structural error population

Affected-image rate remains 0.506329 from threshold 0.525 through 0.600.

Even near the best F1 point, about half of the pathology-corpus images still contain at least one tracked issue. Threshold selection alone is not a substitute for improving detector behavior.

## Cross-cutting interpretation

PRODUCT-INV-RECOGNITION-029 correctly identified that the useful threshold range extended above 0.40.

The fine sweep identifies 0.575 as the bbox-pathology F1 optimum for this Model and corpus, but it does not establish 0.575 as the product runtime optimum.

The bbox-pathology metric operates on annotated boxes. Product recognition additionally depends on region coverage, classification of detector crops, meld grouping, commit eligibility, and three-frame stabilization.

## Recommendation

Use 0.575 as a deliberately chosen first functional-video probe because it is the best bbox-pathology F1 point, not because it is accepted as the production default.

Keep 0.55 and 0.50 as plausible E2E alternatives because they preserve more recall.

Do not promote 0.575 from this Investigation alone.

## Follow-up artifact candidates

- A production-equivalent functional-video evaluation at the bbox-selected threshold.
- A later threshold comparison using region-semantic detector diagnostics so a wrong tile identity or duplicate cannot be hidden by raw box counts.

## Open questions

- Does 0.575 remain useful when evaluated on the five-video product corpus?
- Is the product optimum lower because missed detections are more damaging than duplicate boxes after downstream suppression?
- How much of the remaining affected-image population is threshold-insensitive structural detector failure?
