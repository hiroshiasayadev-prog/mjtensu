# PRODUCT-INV-RECOGNITION-025: Characterize classifier failures with exhaustive real-crop recall

- **status**: concluded
- **date**: 2026-10-04
- **trigger**: Existing aggregate classifier evaluations could report very high overall accuracy while still hiding concentrated class-specific failures. The annotation-audit workflow also showed that running real crops through trained classifiers can expose failure structure that is difficult to infer from aggregate metrics alone. A dedicated exhaustive real-crop recall/failure inventory was therefore needed before deciding whether the next classifier change should target data, training method, or architecture.
- **scope**: Add an exhaustive real-crop recall/failure-inventory evaluation and compare C8, Plain 1.0x, and Plain 0.5x + late DW3x3/PW256 under a common batch-128, random360, 150-epoch condition. Analyze per-class recall, confusion concentration, and cross-model failure overlap.
- **non_scope**: Re-deciding the effective-batch-size result, selecting a production model, changing the training corpus, correcting additional annotations, changing random360, proving the causal mechanism of any architecture failure, or performing the follow-up width/capacity experiment.
- **source_refs**:
  - PRODUCT-INV-RECOGNITION-018
  - PRODUCT-INV-RECOGNITION-021
  - PRODUCT-INV-RECOGNITION-022
  - PRODUCT-INV-RECOGNITION-023
- **follow_up_candidates**:
  - Controlled Plain 0.5x spatial-capacity recovery focused on the observed pinzu failure modes, while preserving awareness of the known latency discontinuity immediately above the 0.5x channel schedule.

## Investigation scope

Determine whether exhaustive real-crop recall and sample-level failure inventory reveal classifier defects that are obscured by the existing aggregate evaluation suite.

The experiment deliberately does not reopen the batch-size investigation from PRODUCT-INV-RECOGNITION-023.

Batch 128 is used here as a common training condition so that the three architectures can be compared under a more nearly aligned optimization setup.

The three trained architectures are:

- C8 gray35;
- Plain 1.0x;
- Plain 0.5x + late DW3x3 refinement + PW1x1 96->256 expansion.

All three use the same frozen training Corpus, random360-only-v1 augmentation, 150 epochs, AdamW learning rate 0.001, weight decay 0.0001, seed 42, AMP, TF32, and existing checkpoint-selection semantics.

The current standard classifier evaluations were also rerun in the same Study so unexpected regressions remained visible. This investigation, however, is specifically about the added all-real-crop recall/failure inventory.

## Out of scope

- Re-evaluating whether batch 128 is globally optimal.
- Changing the frozen training Corpus.
- Treating classifier disagreement as authority to rewrite ground truth.
- Changing random360 or the augmentation family.
- Selecting a production classifier.
- Proving that reduced channel width is the causal mechanism of the observed half-width failure.
- Running the spatial-width recovery experiment.
- Profiling the exact ONNX Runtime / MLAS kernel reason for known channel-width latency discontinuities.

## Background

PRODUCT-INV-RECOGNITION-023 showed that the historical effective batch 512 was not a good quality-oriented default for the Plain random360 line and that batch 128 produced stronger robustness metrics than batch 256.

That result did not answer a separate question: whether the existing evaluation suite was adequately exposing concentrated model failures.

PRODUCT-INV-RECOGNITION-022 had already demonstrated the practical value of full-crop classifier screening during annotation audit. Model predictions were not treated as ground truth, but exhaustive crop-level disagreement was effective at surfacing both annotation defects and model blind spots.

The present investigation generalizes that idea into a reusable classifier Evaluation rather than an annotation-audit-only tool.

A further motivation came from the reduced-width Plain line.

PRODUCT-INV-RECOGNITION-021 found that the Plain 0.5x + late DW3x3/PW256 architecture recovered strong conventional metrics at very low latency. Those conventional metrics did not establish that the model preserved all class-specific behavior present in wider models.

## What was investigated

The sealed MLDB Study was:

| evidence | ref |
|---|---|
| Study | `tile-classifier/c8-plain-batch128-all-real-crop-failure-inventory-v1` |
| completed StudyResult | `tile-classifier/run-05410d8f6a104ee09b35d1386fbcdd08` |
| source commit | `4799074f7e307836f9a9e80706fb7eecf40f1fff` |
| exhaustive Corpus | `tile-classifier/all-real-crops-gray64-v1` |
| Evaluation | `tile-classifier/tile-shape-all-real-crop-recall-v1` |

All three training trials and all eight planned evaluations completed.

The all-real-crop Corpus contains 1,318,142 valid classifier crops:

| source partition | crops |
|---|---:|
| jp train | 1,207,280 |
| jp valid | 71,745 |
| jp test | 36,925 |
| manual capture | 2,192 |
| **total** | **1,318,142** |

The Corpus applies the reviewed jp_v2 correction set from the annotation audit:

- 46 reviewed label corrections;
- one reviewed non-tile annotation removed;
- red fives folded to their base five classes.

The input to each model is the same frozen 64x64 grayscale, aspect-preserving, border-median-letterboxed crop representation.

Because the population intentionally includes train-source crops, this Evaluation is a coverage/failure-inventory diagnostic rather than an independent generalization estimate.

For every crop the Evaluation records:

- true label;
- top-1 predicted label;
- confidence;
- top-2 margin;
- source and partition;
- crop geometry;
- failure status.

It also emits complete per-sample predictions so exact failure-set intersections can be compared across architectures.

## Findings

### Aggregate all-real-crop recall hides materially different failure populations

All three models exceed 99.9% aggregate accuracy over the exhaustive population, but their absolute failure counts and failure structure are very different.

| model | accuracy | macro recall | worst-class recall | errors |
|---|---:|---:|---:|---:|
| C8 | 0.999936 | 0.999935 | 0.999067 | **85** |
| Plain 1.0x | 0.999668 | 0.999668 | 0.997396 | **437** |
| Plain 0.5x + late DW/PW256 | 0.999041 | 0.999039 | 0.979986 | **1,264** |

The aggregate percentages alone make all three models appear nearly saturated.

The failure inventory shows that this interpretation is incomplete.

The half-width model has roughly three times as many errors as Plain 1.0x and roughly fifteen times as many as C8 over the same frozen crop population.

This supports keeping exhaustive real-crop recall/failure inventory as a classifier diagnostic in addition to conventional aggregate accuracy and robustness metrics.

### C8 retains strong exhaustive recall

C8 produced 85 failures over the complete population.

Its dominant confusion was:

- `white -> invalid`: 36 cases.

The remaining failures were sparse across classes.

C8 therefore establishes that the exhaustive population is not intrinsically too difficult for the current classifier task. Large concentrated failure modes observed in the reduced-width Plain model are not shared by all architectures.

### Plain 1.0x shows limited but visible concentrated failures

Plain 1.0x produced 437 failures.

Notable concentrations included:

- `white -> invalid`: 88;
- `1s -> 1p`: 78.

The model remains strong overall, but the inventory exposes class-specific behavior that is much less visible in aggregate accuracy.

### Plain 0.5x + late DW/PW256 has a severe pinzu-specific failure mode

The strongest finding is specific to the selected half-width Plain candidate.

Its pinzu errors are highly concentrated:

| true class | total errors | dominant confusion |
|---|---:|---|
| 5p | 118 | 115 -> 1p |
| 7p | 197 | 161 -> 1p |
| 8p | 773 | 323 -> invalid, 263 -> 1p, 183 -> white |

By contrast:

- 1p has 3 errors;
- 2p has 0;
- 3p has 0;
- 4p has 3;
- 6p has 4;
- 9p has 2.

This is not a uniform failure to recognize the pinzu suit.

The failure is concentrated in classes where identity depends strongly on the number and global arrangement of repeated, locally similar circle primitives.

The pattern is consistent with the model retaining local circle-like evidence while failing to preserve enough representation of the complete spatial arrangement.

This interpretation is a hypothesis, not a demonstrated causal mechanism.

### The half-width failure is not explained by GAP alone

C8, Plain 1.0x, and Plain 0.5x + late DW/PW256 all end with global spatial average pooling.

The 8p result differs sharply:

| model | 8p errors / 38,623 |
|---|---:|
| C8 | 0 |
| Plain 1.0x | 8 |
| Plain 0.5x + late DW/PW256 | **773** |

The common use of global average pooling therefore does not explain the half-width collapse by itself.

A more plausible architecture hypothesis is that the `16/32/64/96` half-width spatial backbone compresses representational capacity too aggressively before the late expansion.

The late architecture performs:

```text
16 -> 32 -> 64 -> 96 spatial channels
                      |
                   8x8 map
                      |
              DW3x3 96 -> 96
              PW1x1 96 -> 256
                      |
                     GAP
```

The late `96 -> 256` expansion can add channel-mixing capacity after spatial downsampling, but it cannot reconstruct information that was already discarded by the narrower earlier spatial backbone.

The fact that Plain 1.0x does not show comparable 5p/7p/8p collapse under the same training condition supports testing this spatial-capacity hypothesis next.

### Cross-model overlap separates common blind spots from architecture-specific failures

All three models failed on the same crop in 40 cases.

In 26 of those cases, all three models produced the same wrong class.

The largest shared patterns were:

- `white -> invalid`: 18;
- `9m -> invalid`: 3;
- `4m -> green`: 2;
- isolated shared cases including `8s -> 5s`, `7p -> 8p`, and `9p -> 9s`.

Pairwise failure intersections were:

| pair | shared failures |
|---|---:|
| C8 and Plain 1.0x | 63 |
| C8 and Plain 0.5x late | 40 |
| Plain 1.0x and Plain 0.5x late | 126 |

These common failures remain useful evidence for future data/training analysis.

They are not the primary next action from this investigation because the dominant actionable defect is the much larger architecture-specific pinzu collapse in the half-width Plain candidate.

### Crop-aspect slicing is diagnostic but should not be interpreted as tile orientation

The Evaluation also groups source crops by bbox aspect ratio:

- portrait;
- near-square;
- landscape.

This is only a crop-shape proxy. It is not a semantic rotation label.

Near-square crops were rare: 1,691 of 1,318,142 samples.

They contained a disproportionate number of difficult, strongly oblique, perspective-distorted, or ambiguous crops, and some remaining annotation/crop-quality issues.

However, most model failures occurred in ordinary portrait or landscape buckets:

| model | portrait errors | landscape errors | near-square errors |
|---|---:|---:|---:|
| C8 | 35 | 38 | 12 |
| Plain 1.0x | 176 | 173 | 88 |
| Plain 0.5x late | 456 | 747 | 61 |

The main failure finding therefore cannot be reduced to abnormal crop aspect or annotation quality.

Aspect slicing is useful as a diagnostic lens, but it should not be treated as a rotation-performance metric.

## Cross-cutting observations

### Aggregate accuracy is insufficient for classifier selection

The main methodological result is that aggregate accuracy above 99.9% can coexist with a severe class-localized failure mode.

The half-width candidate looked broadly competitive under previous conventional evaluations, yet exhaustive recall exposed 773 8p failures and large 5p/7p confusion toward 1p.

A classifier candidate should therefore not be considered well characterized from aggregate accuracy alone.

At minimum, the evaluation suite should retain:

- per-class exhaustive recall;
- total failure count;
- dominant confusion destinations;
- complete sample-level error inventory.

Cross-model failure intersections are particularly useful when several architecture families are available because they help distinguish shared upstream blind spots from architecture-specific behavior.

### The next problem is Plain 0.5x spatial capacity, not corpus correction

Some shared failures and some unusual crop-aspect cases are suitable candidates for later data review.

They do not explain the dominant half-width pinzu failure.

C8 and Plain 1.0x process the same exhaustive population without a comparable collapse.

The next architecture investigation should therefore focus first on whether the half-width Plain spatial backbone is too narrow to retain global pinzu arrangement information.

### Width recovery must account for the known 0.5x latency cliff

PRODUCT-INV-RECOGNITION-018 already measured the region immediately above the `16/32/64/96` half-width schedule.

Representative CPU p50 results were:

| spatial width | representative channels | CPU p50 |
|---|---|---:|
| 0.50000x | 16/32/64/96 | 0.302 ms |
| 0.53125x | 17/34/68/102 | 0.541 ms |
| 0.56250x | slightly wider | 0.580 ms |
| 0.59375x | 19/38/76/114 | 0.605 ms |
| 0.62500x | wider again | 0.711 ms |

The largest latency jump occurs immediately above 0.5x.

The exact ORT/MLAS kernel cause was not profiled, but the result is empirical and large enough that a follow-up should not assume latency scales smoothly with channel count.

A spatial-capacity recovery experiment should therefore evaluate both failure recovery and actual runtime latency rather than treating fractional width as a simple continuous cost knob.

## Follow-up judgment candidates

The next classifier architecture investigation should test whether restoring spatial-backbone capacity removes the 5p/7p/8p failure concentration.

The controlled comparison should preserve, as far as practical:

- the current frozen training Corpus;
- random360-only-v1;
- batch 128;
- 150 epochs;
- seed 42;
- optimizer settings;
- late channel capacity;
- the full existing evaluation suite;
- all-real-crop recall/failure inventory.

Primary evidence should include:

- total exhaustive real-crop errors;
- 5p recall and confusion destinations;
- 7p recall and confusion destinations;
- 8p recall and confusion destinations;
- whether `->1p`, `->invalid`, and `->white` collapse disappears;
- recall across the remaining classes;
- existing robustness diagnostics;
- CPU and iPhone latency.

The experiment should be designed around the known backend-sensitive latency cliff above 0.5x rather than assuming that a fine fractional-width sweep will provide proportionally fine latency steps.

A useful outcome would be evidence that some restored spatial capacity sharply reduces the pinzu failure population while retaining a meaningful deployment-latency advantage.

If width recovery does not remove the failure, later investigation can reopen other mechanisms such as spatial aggregation, pooling, or training-distribution effects.

## Recommendation

Retain exhaustive all-real-crop recall/failure inventory as part of classifier evaluation.

Do not rely on aggregate accuracy alone when comparing future classifier candidates.

Do not treat the current Plain 0.5x + late DW/PW256 candidate as sufficiently characterized by its previously strong aggregate metrics. Its concentrated 5p/7p/8p failure mode requires follow-up before any production-selection judgment.

Prioritize a controlled spatial-capacity recovery investigation for the Plain half-width line.

Treat the reduced spatial-channel hypothesis as the leading explanation to test, not as an established causal conclusion.

Do not make corpus correction the primary follow-up from this investigation.

## Follow-up artifact candidates

- A new classifier architecture investigation testing spatial-capacity recovery of the Plain 0.5x late-DW/PW256 line, with all-real-crop failure inventory as a primary diagnostic.
- A later data/training investigation for cross-architecture common failures only if those failures remain material after the architecture-specific half-width problem is addressed.

## Open questions

- Is the 5p/7p/8p collapse primarily caused by insufficient spatial-backbone channel capacity?
- How much spatial capacity must be restored before the failure population disappears?
- Can the failure be removed without giving up the half-width model's deployment-latency advantage?
- Is there a structural alternative to simple width increase that preserves the favorable `16/32/64/96` runtime shape while restoring global-layout representation?
- If the pinzu collapse persists after capacity recovery, is the remaining limitation caused by spatial aggregation, training distribution, or another architecture mechanism?
