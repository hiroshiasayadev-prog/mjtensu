# PRODUCT-INV-RECOGNITION-035: Evaluate NanoDet batch-24 threshold 0.575 in functional video

- **status**: concluded
- **date**: 2026-10-06
- **trigger**: PRODUCT-INV-RECOGNITION-034 identified score threshold 0.575 as the current NanoDet batch-24 bbox-pathology F1 optimum. The next question was whether that operating point also behaves well when the detector is embedded in the production TypeScript recognition pipeline with the current C8 classifier, red-five specialist, meld semantics, and three-consecutive stabilizer.
- **scope**: Replay the sealed five-video iPhone 13 corpus at deterministic 100 ms source-video intervals using the current C8 classifier, corrected-COCO NanoDet batch-24 detector at threshold 0.575, red-five C8 specialist, production semantic grouping, and production stabilizer.
- **non_scope**: Physical-iPhone latency, detector retraining, classifier comparison, threshold sweep, NMS tuning, manual correction UI, or production-default promotion.
- **source_refs**:
  - PRODUCT-INV-RECOGNITION-005
  - PRODUCT-INV-RECOGNITION-024
  - PRODUCT-INV-RECOGNITION-034
- **follow_up_candidates**:
  - Add region-level ordered semantic detector diagnostics so misses, substitutions, and duplicates are separated.
  - Re-evaluate lower detector thresholds such as 0.50 and 0.55 with those diagnostics.

## Investigation scope

Measure actual recognition behavior of the bbox-selected detector threshold in the production-equivalent browser pipeline.

The classifier stack is fixed rather than a comparison factor. The purpose is to observe whether the detector operating point supplies usable crops to the rest of the product pipeline.

## What was investigated

| evidence | ref |
|---|---|
| historical Study | `mldb-smoke/c8-bs24-threshold0575-functional-video-v1` |
| Study Result | `mldb-smoke/run-779b0cf5763049ee9e6b0c0edd657a26` |
| source commit | `f7818c831dd273f731e8976bc41627c8deb3f9ff` |
| classifier Model | `tile-classifier/run-ac1a2514af214cb69977972c0278ab39-trial-0001-model` |
| detector Model | `nanodet/run-7eeac0f4e5df4950a43d302ad10a1820-trial-0001-model` |
| red-five Runtime Model | `recognition-runtime/red-five-c8-rgb-warmaug-v1` |
| corpus | `tile-classifier/recognition-e2e-night-iphone13-v1` |
| Evaluation Protocol | `tile-classifier/tile-shape-recognition-functional-video-v5` |
| detector threshold | 0.575 |
| runtime registry | `8` |

### Historical namespace note

This run was incorrectly authored under `mldb-smoke` even though it is an ML performance experiment rather than an infrastructure smoke test.

The historical StudyResult ID is immutable evidence and is retained exactly as executed. Later authoring/routing work moved this experiment class toward the detector experiment domain; this Investigation does not rewrite the completed history.

## Findings

### Aggregate product behavior is poor

| metric | result |
|---|---:|
| frame semantic exact rate | 0.171009 |
| completed-hand exact rate | 0.575150 |
| dora exact rate | 0.637275 |
| meld exact rate | 0.229793 |
| eligible frame rate | 0.438878 |
| takes with GT-exact streak3 | 1 / 5 |
| takes product-confirmed | 3 / 5 |
| takes product-confirmed and exact | 1 / 5 |

Only t1 reaches a correct production confirmation.

t2 and t4 reach stable product confirmations that are wrong. The three-consecutive stabilizer therefore behaves as designed but cannot distinguish a stable wrong perception from a correct one.

### Per-take behavior

| take | hand exact | dora exact | meld exact | eligible | product confirmation |
|---|---:|---:|---:|---:|---|
| t1 | 257/299 | 272/299 | 296/299 | 275/299 | correct at 3.9 s |
| t2 | 257/299 | 265/299 | 0/299 | 247/299 | wrong at 2.7 s |
| t3 | 282/300 | 138/300 | 0/300 | 12/300 | none |
| t4 | 0/299 | 275/299 | 48/299 | 123/299 | wrong at 17.1 s |
| t5 | 65/300 | 4/300 | 0/300 | 0/300 | none |

### Aggregate meld exact rate hides how bad the meld-containing takes are

t1 contains no meld GT, so empty-prediction frames count as meld-exact and contribute 296 exact meld frames.

For the four takes that actually contain melds:

- t2: 0/299 meld-exact frames;
- t3: 0/300;
- t4: 48/299;
- t5: 0/300.

The headline aggregate meld-exact rate of 0.229793 is therefore dominated by the no-meld take and materially understates the failure on actual meld scenes.

### Trace evidence separates two failure modes

Raw detector-region counts from the completed prediction trace show:

| take | meld detections/frame mean | meld zero-frame rate | interpretation |
|---|---:|---:|---|
| t2 | 2.120 | 9.0% | boxes often exist, but the three-tile meld is not semantically recovered |
| t3 | 0.163 | 88.3% | detector mostly misses the meld region |
| t4 | 1.344 | 26.8% | partial detections plus wrong grouping/identity lead to only 48 exact frames |
| t5 | 0.053 | 95.3% | detector effectively fails to see the two meld groups |

t3 and t5 are overwhelmingly detector-recall failures at threshold 0.575.

t2 demonstrates that raw detection count alone is not a sufficient metric: a scene may contain several boxes while the recognized tile identities and resulting meld are still wrong.

### The bbox-pathology optimum is not the E2E optimum

PRODUCT-INV-RECOGNITION-034 selected 0.575 only because it maximized bbox-pathology after-product F1.

In functional video, that threshold sacrifices enough recall to make difficult scenes unusable. t3 has only 12 eligible frames and t5 has none.

The result rejects any direct inference that the bbox-pathology F1 optimum is automatically the best product threshold.

## Cross-cutting interpretation

The current C8 classifier and red-five specialist are fixed dependencies in this run. The observed failure is not evidence that their architecture should be changed.

For detector diagnosis, however, plain box counts are also insufficient. A prediction such as GT `4p` becoming predicted `2p, 2p` must be penalized as a substitution plus an extra false prediction rather than treated as approximately correct because the count is close.

The next detector evaluation therefore needs region-level ordered semantic matching using the fixed classifier as a trusted probe.

## Recommendation

Do not promote score threshold 0.575 as the product default.

Keep the current detector Model as the reference while improving the evaluation first.

The next functional-video evaluation should report, per region, ordered semantic correct/substitution/insertion/deletion counts and derived precision/recall/F1. Meld evaluation must score actual detected member observations rather than inferred four-tile kan expansion.

After that diagnostic exists, compare lower thresholds such as 0.50 and 0.55 against 0.575 before deciding whether threshold adjustment is enough or detector retraining/architecture work is required.

## Follow-up artifact candidates

- A new functional-video Evaluation Protocol revision with ordered region-semantic detector diagnostics.
- A threshold comparison using that Protocol.
- Detector/data work focused specifically on meld-region recall if the lower-threshold comparison does not recover t3/t5.

## Open questions

- How much meld recall is recovered at 0.50 or 0.55 without unacceptable duplicate/substitution growth?
- Which meld failures are true misses versus malformed/partial boxes that the fixed classifier turns into the wrong tile?
- Does the detector require explicit meld-domain training changes after threshold effects are separated?
