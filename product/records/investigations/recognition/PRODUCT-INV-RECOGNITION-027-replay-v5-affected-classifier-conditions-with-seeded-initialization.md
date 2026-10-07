# PRODUCT-INV-RECOGNITION-027: Replay v5-affected classifier conditions with seeded initialization

- **status**: concluded
- **date**: 2026-10-05
- **trigger**: Review of repeated Plain-classifier runs found that `tile-shape-train-gpu-v5` applied the declared Study seed only after MLDB had already constructed `context.model`. Conv/Linear and other Architecture initialization was therefore not controlled by the declared seed, so single-run architecture and batch comparisons using v5 could be confounded by different initial weights.
- **scope**: Correct classifier initialization so the declared Study seed controls Architecture construction, preserve the historical training condition attached to every affected comparison, deduplicate identical conditions, and replay all 42 unique v5-affected classifier training conditions in one MLDB Study with the same full eight-evaluation suite.
- **non_scope**: Multi-seed statistical estimation, changing historical epoch/batch/augmentation conditions, selecting a production classifier, redesigning the classifier architecture, proving bitwise equality across heterogeneous GPU workers, detector experiments, or rewriting prior Investigation records in place.
- **source_refs**:
  - PRODUCT-INV-RECOGNITION-014
  - PRODUCT-INV-RECOGNITION-015
  - PRODUCT-INV-RECOGNITION-016
  - PRODUCT-INV-RECOGNITION-017
  - PRODUCT-INV-RECOGNITION-018
  - PRODUCT-INV-RECOGNITION-019
  - PRODUCT-INV-RECOGNITION-020
  - PRODUCT-INV-RECOGNITION-021
  - PRODUCT-INV-RECOGNITION-023
  - PRODUCT-INV-RECOGNITION-025
  - PRODUCT-INV-RECOGNITION-026
- **follow_up_candidates**:
  - Revisit Plain spatial-channel capacity using only seed-corrected evidence.
  - Update prior Investigation wording where an old causal interpretation is contradicted by the replay.

## Investigation scope

Determine which classifier conclusions survive after removing an initialization-seed confound in the generic GPU training path.

This investigation is a replay and evidence-revalidation exercise.

It does not make all architectures begin from the same tensor values. Different architectures cannot share one common parameter tensor.

Instead, the declared `seed=42` is applied before Architecture construction so that each Architecture receives its own deterministic initialization under the same declared seed. Repeating the same Architecture and seed therefore starts from the same initialized state, while different Architectures remain structurally different.

The replay preserves each historical condition rather than normalizing all trials to one new recipe.

For example:

- the ShuffleNet spatial screen remains 150 epochs / batch 512 / `inv013-mix-heavy-v1`;
- the earlier Plain architecture screens remain 100 epochs / batch 512 / `random360-only-v1`;
- the batch-size replay remains 150 epochs with batch 512/256/128;
- the exhaustive-recall and learned-downsampling comparisons remain 150 epochs / batch 128 / `random360-only-v1`.

The purpose is to remove the initialization confound while preserving the experiment that was originally authored.

## Root cause

The v5 execution order was:

1. MLDB runtime resolved the Architecture;
2. MLDB runtime constructed `context.model`;
3. the Train Protocol was invoked;
4. `tile-shape-train-gpu-v5` called its seed function;
5. training began.

The declared seed therefore controlled the later data-order and augmentation paths but did not control the already-completed model initialization.

The issue was directly demonstrated on the selected Plain 0.5x late-DW/PW256 Architecture.

Without seeding before Architecture construction, independent processes produced different initial state hashes.

When `torch.manual_seed(42)` was applied before `build()`, independent processes produced the same initial state hash.

Specialized historical C8 and MobileNet Train Protocols already contained architecture-specific workarounds that rebuilt their models after seeding. The generic v5 path did not.

This explains why the project could contain seeded training metadata while still allowing uncontrolled initial weights in generic v5 comparisons.

## Correction

The correction has two layers.

### Seed Architecture construction in MLDB runtime

The training runtime now applies the declared training seed immediately before fresh Architecture construction.

The runtime seeds:

- Python `random`;
- NumPy;
- PyTorch CPU RNG;
- PyTorch CUDA RNGs when CUDA is available.

This is the authoritative fix for Architecture initialization and also covers Architecture implementations whose custom modules do not expose a normal PyTorch `reset_parameters()` interface.

That detail matters for C8 / escnn.

### Add train-gpu-v6

`tile-classifier/tile-shape-train-gpu-v6` preserves the v5 training semantics but adds seed-corrected initialization behavior inside the Train Protocol and requests deterministic cuDNN algorithm selection.

For ordinary Plain Conv/BatchNorm/Linear models, v6 reinitializes standard modules after the seed has been fixed.

The runtime pre-build seeding remains the more general guarantee because custom Architecture modules such as escnn layers cannot be assumed to expose the same reset interface.

The replay source commit is:

`a0162c0a22d0e2b5df89cb0fe4f4ca1a035c7fa5`

Unit and integration tests cover:

- explicit seeded Architecture construction in the runtime;
- deterministic same-seed model initialization;
- explicit non-Cartesian Study training cases;
- preservation of existing Study grid behavior.

The relevant MLDB test subset completed with 146 passing tests before the replay was submitted.

## Why an explicit-case Study was required

The affected historical experiments do not form one Cartesian parameter grid.

Different Architectures were historically trained with different combinations of:

- epoch budget;
- batch size;
- augmentation recipe.

A normal Study matrix would create many combinations that never existed and were not part of the damaged evidence.

MLDB Study support was therefore extended with an explicit training-case form.

Each case declares exactly:

- Architecture;
- parameter overrides;
- seed.

This allowed all affected historical conditions to be represented inside one StudyResult without manufacturing extra cross-product trials.

## Replay population

The following affected Study definitions were collected:

| source Study | authored trials before cross-Study deduplication |
|---|---:|
| `shufflenet-spatial-screen-v1` | 4 |
| `plain-shallow-architecture-screen-v1` | 6 |
| `plain-width-late-expansion-screen-v1` | 5 |
| `plain-spatial-late-channel-grid-screen-v1` | 17 |
| `plain-spatial-width-fine-screen-v1` | 6 |
| `plain-stage-spatial-mixing-ablation-v1` | 5 |
| `plain-stage2-depthwise-separable-screen-v1` | 2 |
| `plain-capacity-recovery-screen-v1` | 3 |
| `plain-random360-e150-batch-sweep-v1` | 2 |
| `c8-plain-batch128-all-real-crop-failure-inventory-v1` | 3 |
| `plain-w500-third-downsample-failure-recovery-v1` | 3 |
| `plain-random360-e150-reproduction-v1` | 1 |

Several Studies reuse identical Architecture/training conditions.

After exact deduplication by Architecture, full resolved training parameters, and seed, the replay contains **42 unique training conditions**.

## Formal replay evidence

| evidence | ref |
|---|---|
| Study | `tile-classifier/v6-replay-all-v5-affected-conditions-v1` |
| Study Plan | `tile-classifier/v6-replay-all-v5-affected-conditions-v1-plan-c5a2cec921f8f82b` |
| StudyResult | `tile-classifier/run-869049bb06b64f8b8db36b6a7d824309` |
| source commit | `a0162c0a22d0e2b5df89cb0fe4f4ca1a035c7fa5` |
| runtime registry | 8 |
| training trials | 42 |

All 42 training trials completed.

Each trial was assigned the same eight-stage evaluation suite:

1. dense64 angle robustness;
2. full-class diagnostic;
3. validity/rejection;
4. Manzu diagnostic v4;
5. full-class occlusion;
6. ONNX CPU latency;
7. ORT Web iPhone latency v2;
8. exhaustive all-real-crop recall/failure inventory.

Of 336 planned Evaluation stages:

- 326 completed;
- 10 failed;
- all 10 remaining failures are ORT Web iPhone latency v2 stages.

Every non-iPhone Evaluation completed for every training condition.

The StudyResult therefore records `completed_with_failures`, but the initialization/quality/CPU conclusions below are backed by complete evidence across all 42 conditions.

iPhone comparisons are incomplete for the ten affected trials and must not be inferred from missing values.

## Findings

### The initialization bug materially affected individual model outcomes

The seed correction changed the exact trained models enough that some previous class-localized failure populations did not reproduce.

The strongest example is PRODUCT-INV-RECOGNITION-025.

The original v5 run reported a catastrophic all-real failure concentration in the selected Plain 0.5x late-DW/PW256 model, including hundreds of 5p/7p/8p failures.

The seed-corrected replay does **not** reproduce that catastrophic collapse.

Under the replay:

| model | all-real accuracy |
|---|---:|
| C8, e150 bs128 | **0.9999363** |
| Plain 0.5x late-DW/PW256, e150 bs128 | 0.9998179 |
| Plain 1.0x, e150 bs128 | 0.9994727 |

The exhaustive Evaluation remains useful, and C8 remains strongest, but the original specific 5p/7p/8p collapse cannot be treated as a stable Architecture property.

That old failure inventory was a real observation of the old trained weights, not reliable causal evidence about the Architecture family.

### C8 remains exceptionally strong after seed correction

C8 remains the strongest all-real model in the 42-condition replay.

Its main replay metrics are:

| metric | C8 e150 bs128 |
|---|---:|
| dense manual angle mean | 0.981215 |
| full mean-condition accuracy | 0.966815 |
| full worst-condition accuracy | 0.926667 |
| Manzu mean-condition accuracy | 0.977305 |
| all-real accuracy | **0.999936** |
| CPU p50 | 3.300 ms |
| iPhone v2 p50 | 13.3 ms |

The replay therefore preserves the qualitative conclusion that C8 provides unusually strong recognition robustness, while also preserving its severe deployment-latency cost.

### ShuffleNet spatial-retention trend survives

The four ShuffleNet spatial conditions reproduce the original spatial-retention direction strongly.

| condition | angle mean | full mean | full worst | Manzu mean | CPU p50 |
|---|---:|---:|---:|---:|---:|
| s2 + pool | 0.9333 | 0.9104 | 0.8267 | 0.8681 | 0.304 ms |
| s2 + no pool | 0.9727 | 0.9711 | 0.9467 | 0.9716 | 0.441 ms |
| s1 + pool | 0.9725 | 0.9699 | 0.9533 | 0.9716 | 0.541 ms |
| s1 + no pool | **0.9831** | **0.9779** | **0.9733** | **0.9986** | 1.230 ms |

Reducing early spatial compression remains strongly associated with higher robustness.

The best ShuffleNet condition is also substantially heavier than the fast Plain line, so this replay does not make it a deployment-default candidate.

### Plain late expansion remains useful

The seed-corrected replay preserves the benefit of adding later capacity to reduced-width Plain models.

Representative pairs:

| comparison | full mean before | full mean after | angle before | angle after |
|---|---:|---:|---:|---:|
| half -> half late-expand | 0.8359 | **0.8841** | 0.9095 | **0.9423** |
| 0.75x -> 0.75x late-expand | 0.8674 | **0.9009** | 0.9350 | **0.9573** |

Late capacity therefore remains useful.

It does not establish that late expansion alone is enough to match the strongest wider/spatially richer models.

### Plain width trend survives broadly

The replayed Plain late256 width family shows a broad quality increase as spatial channels become wider, despite local non-monotonic points.

| width | representative channels | full mean | full worst | angle mean | CPU p50 |
|---|---|---:|---:|---:|---:|
| 0.50000x | 16/32/64/96 | 0.8713 | 0.6667 | 0.9500 | 0.300 ms |
| 0.53125x | 17/34/68/102 | 0.8881 | 0.7267 | 0.9466 | 0.540 ms |
| 0.56250x | wider | 0.8924 | 0.7111 | 0.9479 | 0.614 ms |
| 0.59375x | 19/38/76/114 | 0.8861 | 0.7156 | 0.9489 | 0.605 ms |
| 0.62500x | 20/40/80/120 | 0.9033 | 0.7378 | 0.9547 | 0.696 ms |
| 0.75000x | 24/48/96/144 | 0.9098 | **0.7578** | **0.9627** | 0.714 ms |
| 0.87500x | wider | **0.9141** | 0.7533 | 0.9615 | 1.007 ms |

The known CPU latency discontinuity immediately above 0.5x also reproduces.

This keeps spatial-channel capacity as a credible follow-up hypothesis.

### Replacing full spatial mixing with 1x1 remains harmful

The stage spatial-mixing ablation reproduces strongly.

Baseline `w500-late256`:

- angle mean 0.9500;
- full mean 0.8713;
- all-real 0.998949.

Replacing later full 3x3 stages with 1x1 progressively degrades quality.

The combined stage3+stage4 1x1 condition falls to:

- angle mean 0.8020;
- full mean 0.6917;
- all-real 0.985161.

The prior conclusion that later spatial mixing matters therefore survives the initialization correction.

### Stage2 depthwise-separable substitution remains a speed/quality trade

Replacing stage2 with DW3x3+PW1x1 reduces CPU p50 from about 0.300 ms to 0.248 ms.

Quality is mixed:

- full mean is similar/slightly higher;
- angle mean is lower;
- Manzu mean is lower;
- all-real accuracy is lower.

The seed-corrected evidence still does not support treating this factorization as a free simplification.

### Late-DW capacity recovery remains beneficial

For the 100-epoch/batch-512 capacity-recovery group:

| model | angle mean | full mean | full worst | Manzu mean | all-real | CPU p50 |
|---|---:|---:|---:|---:|---:|---:|
| 0.5x + late256 | 0.9500 | 0.8713 | 0.6667 | 0.8468 | 0.998949 | 0.300 ms |
| + late DW3x3 | **0.9565** | **0.8936** | **0.7111** | **0.8865** | **0.999046** | 0.316 ms |
| stage3 expand128 | 0.9466 | 0.8767 | 0.6822 | 0.8156 | 0.998512 | **0.280 ms** |

The earlier result that the late depthwise refinement is the stronger quality-recovery option survives.

### Batch-size direction survives strongly

The same Plain 1.0x e150 Architecture was replayed at batch 512, 256, and 128 under seed-corrected initialization.

| batch | angle mean | full mean | full worst | Manzu mean | all-real |
|---:|---:|---:|---:|---:|---:|
| 512 | 0.9467 | 0.9009 | 0.7467 | 0.9092 | 0.998804 |
| 256 | 0.9611 | 0.9124 | 0.7733 | 0.8965 | 0.999431 |
| 128 | **0.9635** | **0.9256** | **0.8067** | **0.9206** | **0.999473** |

The Manzu mean at 256 is locally below 512, but the main robustness and exhaustive metrics support the same overall direction as PRODUCT-INV-RECOGNITION-023.

Batch 128 remains the strongest of these three replayed conditions.

The earlier batch-size judgment therefore survives the seed correction.

### Learned third downsampling result becomes interpretable

The original PRODUCT-INV-RECOGNITION-026 Study was confounded by uncontrolled initialization.

The seed-corrected replay gives:

| condition | angle mean | full mean | full worst | Manzu mean | all-real | CPU p50 |
|---|---:|---:|---:|---:|---:|---:|
| MaxPool baseline | 0.9748 | 0.9317 | 0.7600 | 0.9163 | 0.999818 | 0.305 ms |
| learned DW | 0.9762 | 0.9210 | 0.7422 | 0.9078 | 0.999759 | 0.306 ms |
| learned DW+PW | **0.9810** | **0.9367** | **0.8111** | **0.9177** | **0.999834** | 0.315 ms |

The DW-only replacement does not improve the overall quality surface.

The DW+PW replacement improves the principal angle, full-class, worst-condition, and exhaustive metrics over the fixed-MaxPool baseline at a small CPU cost.

This is now valid same-seed initialization-controlled evidence.

It still does not prove that downsampling is the primary root cause of the Plain quality gap.

## Cross-cutting interpretation

The replay changes the status of the affected historical evidence in an important way.

The project should distinguish:

1. **observations of historical trained weights**, which remain factually true for those artifacts;
2. **causal conclusions about Architecture or batch changes**, which required revalidation because initialization was uncontrolled.

Many high-level directions survive:

- preserving spatial information helps;
- wider Plain spatial channels generally help;
- late expansion helps;
- later full spatial mixing matters;
- late DW refinement remains useful;
- batch 128 remains preferable to 256/512 in the replayed Plain e150 comparison;
- C8 remains exceptionally strong;
- learned DW+PW third downsampling is better than the fixed-MaxPool baseline in its controlled replay.

The most important old conclusion that does **not** survive cleanly is the claim that the selected Plain 0.5x late-DW/PW256 Architecture intrinsically exhibits the catastrophic 5p/7p/8p collapse recorded in the original PRODUCT-INV-RECOGNITION-025 run.

That exact failure population was highly initialization-dependent.

The exhaustive all-real Evaluation remains valuable, but the old specific failure inventory must not be used as stable Architecture-level evidence.

## Recommendation

Treat the seed-corrected replay as the controlling evidence for classifier comparisons that previously relied on `tile-shape-train-gpu-v5`.

Do not discard the historical runs; retain them as observations of specific trained artifacts.

Do not use their single-run quality differences as causal Architecture/batch evidence where the seed-corrected replay now exists.

Preserve these revalidated directions for future work:

- batch 128 as the strongest replayed Plain e150 batch among 512/256/128;
- wider spatial channels as a credible Plain quality lever;
- late capacity as useful but not a substitute for all earlier spatial capacity;
- full spatial mixing as important;
- C8 and spatially conservative ShuffleNet variants as quality references;
- learned DW+PW final downsampling as a valid candidate improvement over the 0.5x late-DW baseline.

Do not treat the historical 5p/7p/8p collapse as an established property of the 0.5x Architecture.

## Follow-up artifact candidates

- Corrective notes in prior affected Investigations pointing to PRODUCT-INV-RECOGNITION-027 as controlling replay evidence.
- A new Plain-capacity Investigation testing where additional spatial channels are most valuable under the corrected seeded training path.
- Completion/retry of the ten missing iPhone v2 stages only if a complete device-latency ranking across all 42 conditions is required.

## Open questions

- Which Plain spatial stage gains the most robustness per added deployment cost?
- Can the strong C8 / ShuffleNet robustness be approached by increasing Plain intermediate representation width without inheriting their latency cost?
- How much of the remaining quality gap is channel capacity versus topology/depth/spatial retention?
- Does the learned DW+PW final downsampling remain useful once earlier/middle spatial capacity is increased?
- Are additional seeds needed before a production-promotion decision, even though the initialization confound is now removed?
