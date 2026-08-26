# PRODUCT-INV-RECOGNITION-006: Validate red-five classification and warm-light augmentation

- **status**: completed
- **date**: 2026-08-14
- **trigger**: PRODUCT-INV-RECOGNITION-005 validated a grayscale 34-class C8 base-tile classifier while explicitly leaving `5m/red5m`, `5p/red5p`, and `5s/red5s` discrimination to a separate specialist. The remaining question was whether red-five discrimination benefits from a hand-designed red-sensitive input representation, and whether the specialist can remain robust when deployment lighting shifts strongly toward warm colors.
- **scope**: Compare RGB, Cr-only, and Y+Cr inputs for a small 64 x 64 C8 binary red-five classifier; evaluate the models over the available six-label red-five crop corpus; audit disagreements against the physical tile content where source labels are wrong; evaluate an unseen warm-light capture set; test whether stochastic warm-light augmentation can recover that domain without training on the real warm captures; and compare parameter count and forward latency before selecting the current input representation.
- **non_scope**: Base 34-class tile identity, generic invalid or unknown crop rejection, NanoDet region-detection accuracy, end-to-end detector-to-classifier accuracy, actual iPhone/WebGL latency, ONNX or browser export, arbitrary illuminants outside the tested warm-light transformation family, temporal stabilization, hand interpretation, or score calculation.
- **source_refs**:
  - PRODUCT-ADR-RECOGNITION-001
  - PRODUCT-INV-RECOGNITION-004
  - PRODUCT-INV-RECOGNITION-005
  - tools/recognition/build_red_five_dataset.py
  - tools/recognition/build_red_five_classifier_dataset.py
  - tools/recognition/evaluate_red_five_redness_baseline.py
  - tools/recognition/red_five_classifier.py
  - tools/recognition/train_red_five_classifier.py
  - tools/recognition/run_red_five_input_sweep.py
  - tools/recognition/evaluate_red_five_all_models.py
  - tools/recognition/train_red_five_classifier_warm_aug.py
  - tools/recognition/run_red_five_warm_aug_sweep.py
  - tools/recognition/evaluate_red_five_warm_holdout.py
  - .local/recognition/red_five_datasets/red_five_all.sqlite
  - .local/recognition/red_five_datasets/rgb64_binary_jp5000_seed42.sqlite
  - /srv/data/mjtensu/.local/recognition/red_five_runs/c8_rgb_cr_ycr_seed42/all_samples_evaluation.json
  - /srv/data/mjtensu/.local/recognition/red_five_runs/c8_rgb_cr_ycr_seed42/all_samples_errors.jsonl
  - /srv/data/mjtensu/.local/recognition/red_five_datasets/warm_red_five_24.sqlite
  - /srv/data/mjtensu/.local/recognition/red_five_runs/c8_rgb_cr_ycr_warmaug_seed42/warm_holdout_evaluation.json
  - /srv/data/mjtensu/.local/recognition/red_five_runs/c8_rgb_cr_ycr_warmaug_seed42/all_samples_warmaug_evaluation.json
  - /srv/data/mjtensu/.local/recognition/red_five_runs/c8_rgb_cr_ycr_warmaug_seed42/all_samples_warmaug_errors.jsonl
- **follow_up_candidates**:
  - Export the selected RGB red-five specialist to the intended browser/mobile runtime and benchmark complete crop preprocessing plus inference on the iPhone 13 rather than extrapolating from RTX 3090 forward timing.
  - Preserve a small real capture holdout when a new lighting failure is discovered, model that condition with physically plausible augmentation, and verify the retrained model against the untouched real holdout before collecting a large condition-specific training corpus.
  - Extend the photometric augmentation family only when real failure evidence requires it; candidates include spatially varying colored illumination, specular clipping, camera white-balance errors, and sensor-specific tone curves.
  - Integrate the accepted red-five specialist after base `5m/5p/5s` recognition and measure end-to-end detector-to-base-classifier-to-red-five accuracy.
- **related_adrs**:
  - PRODUCT-ADR-RECOGNITION-001

## Investigation scope

This investigation evaluates the color-sensitive specialist that follows the grayscale base-tile classifier validated in PRODUCT-INV-RECOGNITION-005.

The base classifier intentionally maps red and non-red fives to the same shape classes:

```text
red5m -> 5m
red5p -> 5p
red5s -> 5s
```

The specialist therefore receives only crops already belonging to one of the three five classes and makes a binary decision:

```text
normal five
red five
```

The primary representation question was whether it is useful to discard RGB information before the learned classifier.
Red-five discrimination appears naturally suited to a red-sensitive color component, so Cr-only input was expected to be a plausible lower-dimensional alternative to raw RGB.
Y+Cr was included to test whether retaining luminance together with red chroma provides a useful compromise.

The investigation also became a lighting-domain experiment after a dedicated warm-light capture set exposed a severe failure in the initially strong RGB and Cr models.
The real warm captures were deliberately kept out of subsequent training so that they remained a genuine external holdout for the augmentation experiment.

## Red-five crop corpus

The six relevant source labels are:

```text
5m, red5m
5p, red5p
5s, red5s
```

The full red-five evaluation corpus contains 116,083 detector-derived crops:

| target | crops |
|---|---:|
| normal five | 87,154 |
| red five | 28,929 |
| total | 116,083 |

The source-label counts are:

| source label | crops |
|---|---:|
| `5m` | 29,129 |
| `red5m` | 9,676 |
| `5p` | 28,921 |
| `red5p` | 9,659 |
| `5s` | 29,104 |
| `red5s` | 9,594 |

The full corpus contains both Japanese-source crops and manual deployment captures.
The manual portion contains 241 relevant crops, including bright, dark, shadowed, and other deployment-capture conditions.

A compact training database was created at:

```text
.local/recognition/red_five_datasets/rgb64_binary_jp5000_seed42.sqlite
```

Its training policy uses 5,000 JP training crops for each `(suit, is_red)` group, giving 30,000 JP training crops across the six groups.
Manual training crops are split by capture rather than by individual crop and are represented repeatedly during training so that the much larger JP corpus does not eliminate the deployment-domain influence.

The principal compact-dataset settings are:

| setting | value |
|---|---:|
| image size | 64 x 64 RGB |
| JP train samples per `(suit, is_red)` | 5,000 |
| manual train fraction | 0.80 by capture |
| manual train repeat | 20 |
| seed | 42 |

The compact database stores preprocessed RGB bytes even for the Cr and Y+Cr models so that all three representations are derived from exactly the same source image at training time.

## Classifier architecture and input representations

All representation comparisons use the same C8 rotation-equivariant architecture.
Only the input representation changes.

```text
64 x 64 input
  -> C8 equivariant convolution blocks
     regular fields: 8 / 16 / 32 / 64
  -> GroupPooling
  -> spatial global pooling
  -> binary 2-logit head
```

The three input modes are:

| mode | channels | definition |
|---|---:|---|
| RGB | 3 | RGB channels scaled to `[0, 1]` |
| Cr | 1 | `0.5R - 0.418688G - 0.081312B + 0.5` |
| Y+Cr | 2 | BT.601-like luminance plus the same Cr component |

The input channels are scalar fields under spatial rotation.
The C8 backbone is otherwise identical across the three models.

Training uses:

| setting | value |
|---|---:|
| optimizer | AdamW |
| learning rate | 0.001 |
| weight decay | 0.0001 |
| batch size | 1,024 |
| evaluation batch size | 4,096 |
| residual rotation augmentation | +/-22.5 degrees |
| seed | 42 |
| AMP | enabled |
| TF32 | enabled |

Best-checkpoint selection uses JP validation only, averaged over the configured 0, 15, 30, and 45 degree evaluations.
Manual validation is an external holdout and is not used to choose `best.pt`.

## Initial representation comparison

The first RGB/Cr/Y+Cr sweep did not include the later warm-light augmentation.
All three models reached extremely high source-label accuracy on the 116,083-crop corpus.
At this stage Cr appeared especially attractive because it reduced the task to one red-sensitive channel and made only a handful of disagreements.

Error inspection also showed that source-label accuracy was no longer a sufficient interpretation by itself.
Several apparent classifier errors were actually annotation defects in the Japanese source data.
The confirmed defects relevant to the final comparison are described later in this investigation.

A simple fixed redness statistic had previously failed some dark and partially shadowed manual examples, while the learned C8 color classifier handled the original manual domain much better.
This established that red-five discrimination benefits from learned spatial structure rather than only from a global red-pixel threshold.

## Warm-light external holdout

A separate full-tile catalog had been captured under explicitly warm lighting:

```text
tile-catalog-warm-4-v2
layout: tile-catalog-layout-raw-001
```

The catalog contains four captures of the same physical tile layout:

- Warm, normal brightness, no shadow, front view.
- Warm, dim, no shadow, front view.
- Warm, normal brightness, partial shadow, front view.
- Warm, normal brightness, no shadow, slight camera angle.

For red-five evaluation this yields six relevant tiles per capture:

```text
5m, red5m, 5p, red5p, 5s, red5s
```

The resulting external holdout contains 24 crops.
These 24 real warm-light crops were not present in the compact training dataset used for the initial sweep.

The initial models performed as follows:

| mode | correct | accuracy |
|---|---:|---:|
| RGB | 12 / 24 | 50.0% |
| Cr | 12 / 24 | 50.0% |
| Y+Cr | 23 / 24 | 95.83% |

RGB and Cr classified all twelve normal fives as red.
The Cr model was not merely close to the binary threshold.
Its normal-five red probabilities under the warm captures ranged approximately from:

```text
0.848408 .. 0.999999
```

The corresponding red-five probabilities were also near one.
The representation had therefore collapsed under the warm-light domain rather than suffering from a tunable `0.5` threshold problem.

This result overturned the initial assumption that a red-sensitive one-channel projection was automatically robust for red-five discrimination.
Warm illumination shifts the entire crop toward the same color direction that Cr is intended to expose for red ink.
After the RGB information has been projected into a single Cr channel, the classifier has less evidence available to distinguish a global illuminant shift from a locally red tile marking.

Y+Cr retained much stronger warm-light robustness in this first test, but its behavior on the larger corpus still contained more false-red classifications than RGB or Cr.
The warm holdout therefore motivated augmentation rather than immediate Y+Cr selection.

## Warm-light augmentation experiment

The 24 real warm crops were deliberately kept out of training.
Instead, the existing training RGB images were stochastically transformed before conversion into RGB, Cr, or Y+Cr model inputs.

The augmentation is applied independently per sample with probability `0.50`.
For selected samples it varies a warm-strength scalar and applies RGB white-balance-like gains plus an exposure change.
The configured limits are:

| parameter | range or limit |
|---|---:|
| augmentation probability | 0.50 |
| warm strength | 0.10 .. 1.00 |
| red gain at maximum strength | 1.50 |
| green gain at maximum strength | 1.08 |
| blue gain at maximum strength | 0.45 |
| exposure | 0.65 .. 1.15 |

The three models were retrained from scratch rather than fine-tuned from the original checkpoints.
This preserves a clean comparison between representations and avoids interpreting adaptation from one previous solution as an intrinsic property of the input mode.

The real warm-light capture policy is explicitly recorded by the trainer as:

```text
external_holdout_not_used_for_training_or_checkpoint_selection
```

The augmentation is not claimed to reproduce the exact spectral distribution of the physical light source.
It is a deliberately simple family of plausible channel-balance and exposure perturbations intended to cover the observed warm-domain shift without consuming the real holdout.

## Warm-light holdout after augmentation

After retraining with stochastic warm-light augmentation, every input representation classified all 24 untouched real warm-light crops correctly:

| mode | best epoch | correct | accuracy |
|---|---:|---:|---:|
| Cr | 5 | 24 / 24 | 100% |
| RGB | 15 | 24 / 24 | 100% |
| Y+Cr | 25 | 24 / 24 | 100% |

The probability margins were also wide:

| mode | maximum `p(red)` among normal fives | minimum `p(red)` among red fives |
|---|---:|---:|
| Cr | 0.012970 | 0.976423 |
| RGB | 0.003173 | 0.999999 |
| Y+Cr | 0.000569 | 1.000000 |

The result is therefore stronger than 24 predictions barely crossing a `0.5` threshold.
For this captured warm-light domain, the stochastic synthetic transform moved all three learned decision functions to a large separation between normal and red fives.

Before augmentation, RGB and Cr were each only 50% accurate on exactly this holdout.
No real warm-light holdout crop was added to the training data to obtain the 100% result.

This is evidence that a physically plausible augmentation family can supply useful illumination invariance without requiring a large condition-specific real training corpus.
It is not a proof of invariance to arbitrary unseen lighting.
The demonstrated claim is limited to the tested real warm-light captures and to the relationship between that domain and the RGB gain/exposure perturbation family used here.

## Full-corpus evaluation after warm-light augmentation

The retrained models were evaluated again over all 116,083 relevant crops at zero-degree inference.
Using the stored source labels literally, the results were:

| mode | TP | TN | FP | FN | source-label errors | accuracy |
|---|---:|---:|---:|---:|---:|---:|
| Cr | 28,924 | 87,154 | 0 | 5 | 5 | 99.99569% |
| RGB | 28,928 | 87,151 | 3 | 1 | 4 | 99.99655% |
| Y+Cr | 28,928 | 87,143 | 11 | 1 | 12 | 99.98966% |

The 76,251 JP training crops not selected into the compact experiment database provide a particularly useful large unseen subset.
Their source-label results were:

| mode | errors | accuracy |
|---|---:|---:|
| Cr | 1 | 99.99869% |
| RGB | 4 | 99.99475% |
| Y+Cr | 12 | 99.98426% |

Manual-source results were:

| mode | correct | accuracy |
|---|---:|---:|
| Cr | 240 / 241 | 99.5851% |
| RGB | 241 / 241 | 100% |
| Y+Cr | 241 / 241 | 100% |

The source-label table is reproducible but must not be interpreted as the final physical-tile error count because the corpus contains confirmed wrong labels.

## Human audit of source-label disagreements

Visual and capture-layout review identified the following source annotation defects that materially affect interpretation of the three models.

### Red five pinzu stored as normal five manzu

The following records are physically `red5p` but are stored as `5m`:

```text
jp:train:106889
jp:train:106988
```

A third crop:

```text
jp:train:106570
```

comes from the same tile placement under an extremely dark capture.
The crop by itself is effectively too dark for reliable visual color judgment, but the capture-layout provenance identifies the position as the same red-five pinzu tile.
It is therefore treated as `red5p` for the audited interpretation.

On these three crops the warm-augmented RGB and Y+Cr models predict red, while Cr predicts normal.
The stored labels consequently make RGB and Y+Cr appear wrong and Cr appear correct even though the physical-tile interpretation is the opposite.

### Normal two manzu stored as red five souzu

The following records are physically `2m` but are stored as `red5s`:

```text
jp:train:1097575
jp:train:1186804
jp:train:48361
```

The warm-augmented Cr model predicts normal on all three and is therefore physically correct despite being counted as a source-label error.
RGB and Y+Cr predict red and are therefore physically wrong despite agreeing with the stored source label.

### Remaining disagreements

The source-label defects above account for the known hidden reversals between stored labels and physical-tile interpretation.
The remaining audited disagreements are treated as genuine model errors.
Notable examples include:

```text
Cr:
  jp:train:1186817        red5m -> normal, p(red)=0.368931
  manual red5m crop       red5m -> normal, p(red)=0.411273

RGB:
  jp:train:601330         red5m -> normal, p(red)=0.338241
```

Y+Cr retains multiple false-red predictions on genuine normal `5m` and `5s` crops in addition to one red-five miss.

After accounting for the confirmed source-label defects and the capture-layout interpretation of `jp:train:106570`, the observed physical-tile error counts remain:

| mode | observed physical-tile errors | dominant failure pattern |
|---|---:|---|
| RGB | 4 | three false-red normal tiles plus one red-five miss |
| Cr | 5 | red-five misses; no observed false-red normal tile in the audited set |
| Y+Cr | 12 | predominantly false-red normal `5m/5s`, plus one red-five miss |

An important lesson is that the same source-label defects can both create false model errors and hide real model errors.
At this accuracy level, model selection must use reviewed crop content and provenance rather than only aggregate source-label accuracy.

## Representation behavior

### Cr is an effective feature but an unnecessary bottleneck

Cr initially looked like the natural representation for this task.
It is directly sensitive to red-vs-non-red color differences, requires only one input channel, and produced excellent results on the original corpus.

The warm-light experiment exposed the cost of that manual information reduction.
A global warm shift and a local red marking can both increase the Cr response.
Once RGB has been collapsed into Cr, the downstream model cannot reconstruct the discarded cross-channel evidence that may distinguish those causes.

Warm-light augmentation taught the Cr model to solve the tested warm domain, but the final full-corpus audit still did not make Cr more accurate than RGB.

### Y+Cr retains illumination evidence but produces more false-red decisions

Y+Cr was substantially more robust than RGB and Cr on the warm holdout before warm augmentation, reaching 23/24 while the other two were at 12/24.
This supports the idea that retaining luminance helps the classifier distinguish global illumination from local chromatic structure.

The larger corpus, however, shows a consistent cost.
After the same warm augmentation, Y+Cr still produced twelve audited physical-tile errors, predominantly normal fives classified as red.
Its warm-domain advantage therefore does not justify selecting it over RGB once RGB receives adequate photometric augmentation.

### RGB lets the model learn the useful color evidence without an explicit projection

Warm-augmented RGB reaches 24/24 on the untouched warm-light holdout, 241/241 on the manual corpus, and the lowest audited physical-tile error count over the 116,083-crop corpus.

The result does not mean that RGB is inherently invariant to illumination.
The unaugmented RGB model failed the warm holdout at 50%.
The successful condition is specifically:

```text
raw RGB signal
+ learned C8 spatial classifier
+ residual rotation augmentation
+ stochastic warm-light photometric augmentation
```

The experiment therefore favors preserving the available RGB evidence and teaching the model the expected nuisance variation rather than projecting the signal into a supposedly task-specific color component before learning.

## Parameter count and forward benchmark

Reducing the input from three channels to one or two channels was also evaluated as a potential deployment optimization.
The effect on this architecture is minimal because the input channel count changes only the first equivariant convolution; the later C8 field widths are unchanged.

Measured parameter counts and training-checkpoint sizes were:

| mode | parameters | trainable | `best.pt` size |
|---|---:|---:|---:|
| RGB | 138,106 | 138,106 | 7.658 MiB |
| Cr | 137,930 | 137,930 | 7.643 MiB |
| Y+Cr | 138,018 | 138,018 | 7.650 MiB |

The `best.pt` values are training-checkpoint artifact sizes and include more than deployable model weights, so they are not an exported mobile-model size benchmark.
The parameter counts directly show that the architectural size reduction is only about 0.13% from RGB to Cr.

RTX 3090 forward timing was:

| batch | RGB ms/batch | Cr ms/batch | Y+Cr ms/batch |
|---:|---:|---:|---:|
| 1 | 0.991 | 0.988 | 0.988 |
| 32 | 1.385 | 1.329 | 1.346 |
| 256 | 6.314 | 5.918 | 6.102 |

At the deployment-relevant single-inference scale, the observed difference is approximately measurement noise.
At larger GPU batches Cr is several percent faster, but the absolute difference remains small and does not include the RGB-to-Cr preprocessing step required before inference.

Actual mobile and WebGL execution may have different kernel, texture-packing, and memory behavior, so the RTX measurement does not prove equal iPhone latency.
It does show that the current architecture obtains no large computational or storage benefit from removing RGB channels.
There is therefore no measured efficiency advantage large enough to compensate for the lower audited accuracy of Cr or Y+Cr.

## Findings

### Hand-designed red-sensitive projection did not improve the final specialist

Cr and Y+Cr are reasonable representations for the semantic task, and both produced useful intermediate results.
Neither representation produced a final accuracy, model-size, or forward-latency advantage over RGB after the warm-light robustness problem was addressed consistently.

The tested one- and two-channel projections therefore add representation complexity without a demonstrated deployment benefit.

### Photometric augmentation can remove a severe real lighting-domain failure without training on that real domain

The clearest positive result of the investigation is the warm-light augmentation experiment.

Before augmentation:

```text
RGB   12 / 24
Cr    12 / 24
Y+Cr  23 / 24
```

After training from scratch with stochastic synthetic warm-light transforms and still withholding all 24 real warm crops:

```text
RGB   24 / 24
Cr    24 / 24
Y+Cr  24 / 24
```

The wide post-augmentation probability margins further support that the models learned a stable separation on this holdout rather than exploiting an accidental threshold crossing.

This gives a practical future workflow for newly observed lighting failures:

```text
observe a real failure condition
  -> keep a small real sample as holdout
  -> design a physically plausible augmentation family that spans the failure
  -> retrain from scratch or under a controlled retraining protocol
  -> evaluate on the untouched real holdout
  -> collect a large real condition-specific corpus only if the synthetic coverage is insufficient
```

The workflow is an evidence-backed strategy, not a guarantee that every lighting domain can be simulated adequately.

### Data-quality auditing remains necessary at near-perfect classifier accuracy

The full-corpus comparison contains source labels that are physically wrong.
Some wrong labels inflate the measured error count, while others hide a classifier mistake by agreeing with the wrong stored target.

The final representation judgment therefore uses the manually audited physical-tile interpretation rather than blindly ranking models by the database target alone.

### The remaining red-five specialist error is small relative to unresolved pipeline risks

The selected RGB condition shows four observed physical-tile errors across 116,083 relevant crops after label audit, while also passing the available manual and real warm-light holdouts.

Further small improvements to this same binary task are unlikely to be the highest-value recognition work unless later end-to-end evidence shows a systematic red-five failure.
Invalid-crop handling, detector localization, mobile integration, and complete pipeline evaluation remain more consequential unresolved risks.

## Judgment

Adopt RGB as the current red-five specialist input representation.

The accepted experimental baseline is:

```text
64 x 64 RGB input
C8 rotation-equivariant binary classifier
8 / 16 / 32 / 64 regular fields
GroupPooling
normal-five vs red-five output
+/-22.5 degree residual rotation augmentation
stochastic warm-light RGB gain/exposure augmentation
batch size 1,024
JP-validation-only checkpoint selection
manual and real warm-light captures excluded from checkpoint selection
```

Do not add a mandatory RGB-to-Cr or RGB-to-Y+Cr projection before this classifier.
The measured parameter and latency savings are negligible in the current architecture, and both projected representations produced more audited physical-tile errors than RGB in the final comparison.

Do not treat the warm-light result as proof of universal illumination invariance.
Treat it as evidence that a small untouched real holdout plus a physically plausible synthetic augmentation family can be sufficient to recover a large domain-specific failure without collecting a large training corpus in that same condition.

When future lighting-domain failures are found, prefer first to reproduce the nuisance variation synthetically while preserving real examples for validation.
Escalate to substantial condition-specific data collection when synthetic augmentation fails to cover the observed real distribution.

## Impact on PRODUCT-ADR-RECOGNITION-001

PRODUCT-INV-RECOGNITION-004 showed that deterministic three-color quantization could discard class-relevant information before learning.
PRODUCT-INV-RECOGNITION-005 then showed that the 34 base tile classes can be recognized with a compact grayscale learned classifier while red-five discrimination remains a separate responsibility.

This investigation completes that split for the tested red-five task.
The red-five specialist does not benefit from mandatory hand-designed Cr or Y+Cr input projection once realistic photometric augmentation is included.
The evidence instead favors preserving RGB for the color-sensitive specialist and learning robustness to nuisance illumination from augmentation.

The staged detector -> base-shape classifier -> red-five specialist architecture remains supported.
A later ADR amendment or superseding ADR can replace the earlier mandatory normalization/quantization interpretation with the now-validated learned-classifier input policies:

```text
base tile identity: grayscale C8 classifier
red-five discrimination: RGB C8 specialist with photometric augmentation
```

Actual mobile-runtime selection remains subject to target-device export and end-to-end latency measurement.
