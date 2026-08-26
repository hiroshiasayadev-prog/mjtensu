# PRODUCT-INV-RECOGNITION-004: Evaluate deterministic three-color tile normalization

- **status**: completed
- **date**: 2026-08-06
- **trigger**: PRODUCT-ADR-RECOGNITION-001 assigns illumination normalization, palette inference, and multi-level color quantization before tile classification, but the information-preservation value of deterministic white, black, and red conversion had not been validated against real dark and shadowed captures.
- **scope**: Evaluate whether fixed CIELAB thresholds or adaptive Sauvola-based processing can convert detector-derived Japanese riichi tile crops into a stable white, black, and red representation without discarding class-relevant markings.
- **non_scope**: Train or select the final tile classifier, prove that raw RGB alone is sufficient, define the final unknown-class taxonomy, segment the exact tile face, or supersede PRODUCT-ADR-RECOGNITION-001.
- **source_refs**:
  - PRODUCT-ADR-RECOGNITION-001
  - PRODUCT-INV-RECOGNITION-003
  - tools/recognition/build_tile_crop_dataset.py
  - tools/recognition/build_color_trial_sample.py
  - tools/recognition/run_lab_threshold_trial.py
  - tools/recognition/run_sauvola_trial.py
  - tools/recognition/tests/test_color_trial.py
  - .local/recognition/color_trials/sample_seed42.summary.json
  - .local/recognition/color_trials/lab_fixed_seed42/summary.json
  - .local/recognition/color_trials/lab_fixed_seed42/contact_sheet.png
  - .local/recognition/color_trials/sauvola_seed42/summary.json
  - .local/recognition/color_trials/sauvola_seed42/contact_sheet.png
  - .local/recognition/color_trials/sauvola_seed42/red_labels_lowest_red_ratio.png
  - .local/recognition/color_trials/sauvola_seed42/non_red_labels_highest_red_ratio.png
  - .local/recognition/color_trials/sauvola_seed42/red_surface_rejections.png
- **follow_up_candidates**:
  - Compare a raw-RGB tile classifier with weak and strong capture-condition augmentation using a capture-level manual split.
  - Compare raw RGB against raw RGB plus a continuous locally normalized lightness channel without irreversible thresholding.
  - Add a small jointly trained illumination-normalization front end only if the augmented raw-RGB classifier remains specifically weak on dark captures.
  - Revisit the mandatory quantization stages in PRODUCT-ADR-RECOGNITION-001 after classifier evidence is available.
- **related_adrs**:
  - PRODUCT-ADR-RECOGNITION-001

## Investigation scope

This investigation tested whether deterministic preprocessing could remove lighting and color variation before a small tile classifier while preserving the visual information needed to distinguish Japanese riichi tile classes.

The evaluated output representation contained exactly three colors:

```text
white: tile-face or non-marking region
black: black, blue, or green markings and other non-red dark content
red: red printed markings
```

The investigation deliberately used detector-derived crops rather than ideal tile-face masks.
A deployment detector bounding box may include tile sides, table background, shadow, neighboring content, or a back-facing tile.
A useful preprocessing stage must therefore either tolerate those conditions or expose a reliable rejection signal.

The investigation did not train a classifier over the generated images.
Its judgment is based on reproducible pixel statistics and visual review of information preservation in diagnostic contact sheets.

## Dataset

A persistent crop database was created from the Japanese source dataset and the manually captured deployment-layout dataset.
A deterministic compact sample was then generated for preprocessing trials.

| source | sample policy | crops |
|---|---|---:|
| `jp` | Seed-42 reservoir sample of 100 training crops for each of 37 canonical labels | 3,700 |
| `manual` | All available manually captured crops | 1,968 |
| total |  | 5,668 |

The compact database is:

```text
.local/recognition/color_trials/sample_seed42.sqlite
```

The original PNG bytes are copied unchanged into the sample database.
Sampling does not apply normalization, resizing, color conversion, or augmentation.

The source composition is intentionally imbalanced by label in the manual subset because it retains all available deployment captures rather than manufacturing a balanced evaluation set.
The sample is suitable for inspecting preprocessing behavior but does not define a final classifier benchmark split.

## Experiment sequence

### Fixed CIELAB threshold baseline

The first baseline converted every crop to CIELAB and applied one global threshold set to every image:

| parameter | value |
|---|---:|
| dark lightness threshold | `L* <= 50` |
| red threshold | `a* >= 18` |
| green threshold folded into black | `a* <= -18` |
| minimum chroma | `15` |

Red was emitted for pixels meeting the red and chroma conditions.
Black was emitted for remaining dark pixels and sufficiently chromatic green pixels.
All remaining pixels became white.

No tile-face mask, white-reference estimation, local illumination correction, exposure correction, or background removal was applied.

### Sauvola lightness baseline

The fixed lightness threshold was replaced with a per-pixel Sauvola threshold over CIELAB `L*`.
The default trial used:

| parameter | value |
|---|---:|
| local window | `15 x 15` |
| `k` | `0.20` |
| lightness dynamic range | `50` |

This changed the black decision from an absolute lightness test to a local contrast test.
A pixel could remain white in a dark crop when it was not substantially darker than its neighborhood, while a printed stroke could become black when it was locally darker than the surrounding tile face.

### Crop-relative red detection

Normalized-RGB red thresholds improved some images but remained sensitive to warm lighting and low intensity.
The trial was therefore extended to estimate a crop-relative neutral color reference:

1. Remove pixels classified as locally dark by Sauvola.
2. Keep the brighter side of the remaining `L*` distribution.
3. Keep the lower-chroma side of that bright subset.
4. Estimate median `a*` and `b*`.
5. Reject color outliers through a median-absolute-deviation-derived distance limit.
6. Treat the refined median as a neutral reference rather than as a proven tile-face white point.
7. Detect red from positive `delta a*`, relative chroma, and normalized-RGB red dominance.

The default relative-red conditions were:

| parameter | value |
|---|---:|
| neutral lightness quantile | `0.45` |
| neutral chroma quantile | `0.40` |
| minimum neutral-candidate fraction | `0.05` |
| maximum neutral-reference spread | `10` |
| minimum `delta a*` | `8` |
| minimum relative chroma | `10` |
| minimum normalized-RGB red dominance | `0.025` |

When the neutral-reference checks failed, the implementation fell back to the earlier absolute normalized-RGB red rule.

### Red-surface rejection

Visual review exposed a different failure: a red tile back or red background region could satisfy the crop-relative red rule even though it was not red ink printed on a white tile face.

A connected-component filter was added to reject red regions that appeared to be a surface rather than ink:

| rule | value |
|---|---:|
| reject the complete raw-red mask above crop fraction | `0.40` |
| reject one red component above crop fraction | `0.30` |
| reject a red component touching more than this many crop sides | `1` |

Rejected red pixels were folded into black.
The contact sheets retained both the raw-red mask and the post-filter ink-red mask so that the rejection could be inspected directly.

## Reproducible outputs

The fixed-threshold trial writes:

```text
.local/recognition/color_trials/lab_fixed_seed42/
├─ contact_sheet.png
├─ metrics.csv
└─ summary.json
```

The adaptive trial writes:

```text
.local/recognition/color_trials/sauvola_seed42/
├─ contact_sheet.png
├─ manual_highest_black_ratio.png
├─ red_labels_lowest_red_ratio.png
├─ non_red_labels_highest_red_ratio.png
├─ red_surface_rejections.png
├─ metrics.csv
└─ summary.json
```

The adaptive contact sheet shows, for a representative JP crop and manual crop of each label:

```text
original | L* | Sauvola | neutral reference mask | raw red | ink red | final
```

The representative is selected near the median final black ratio for each source and label rather than from the most successful example.

## Quantitative observations

The aggregate output-pixel ratios changed as follows:

| method | source | white | black | red |
|---|---|---:|---:|---:|
| fixed CIELAB | JP | 76.07% | 18.96% | 4.96% |
| fixed CIELAB | manual | 52.38% | 45.11% | 2.51% |
| Sauvola plus relative red | JP | 69.10% | 24.03% | 6.87% |
| Sauvola plus relative red | manual | 76.02% | 19.66% | 4.33% |

The fixed CIELAB baseline exposed a large domain shift.
Manual crops became substantially more black and less red than the JP crops because absolute thresholds interpreted dark tile faces as black and failed to retain weak red under low illumination.

Sauvola substantially reduced the aggregate manual black ratio and increased the aggregate manual white ratio.
On those aggregate statistics alone, the adaptive pipeline appeared to correct much of the dark-domain failure.

That apparent improvement did not establish information preservation.
Visual review found dark crops where a marking visible in the original image was only partially retained or disappeared from the final three-color result.
The local threshold could model the dark tile face as background, but when the marking contrast approached the local illumination variation, it could not distinguish the intended stroke from noise or shading.

The neutral-reference implementation reported a reliable reference for every crop in both aggregate source groups.
Visual failures nevertheless remained.
This shows that the implemented reliability test measured whether the selected color cluster was compact and sufficiently large; it did not prove that the cluster represented the tile face or that it supported correct red-ink interpretation.

The aggregate white, black, and red ratios are therefore descriptive diagnostics only.
They cannot serve as acceptance metrics for tile recognition because a low-area class-defining stroke may disappear while the overall pixel ratios move in an apparently favorable direction.

## Qualitative findings

### Fixed color thresholds are not viable as a shared deployment rule

The JP sample has relatively regular exposure and background conditions.
The same fixed `L*`, `a*`, and chroma thresholds applied to manual crops produced black tile faces, weakened red markings, and large source-dependent output distributions.

Adjusting the fixed threshold toward the manual domain would move the failure into brighter crops rather than create an invariant representation.

### Sauvola improves broad foreground-background separation but can erase weak class evidence

Sauvola is better suited than an absolute lightness threshold to uneven illumination.
It retained usable black-white structure in many manual crops and prevented whole dark tile faces from becoming black.

It still makes an irreversible local binary decision.
In dark, blurred, or weak-contrast crops, a class-relevant stroke can fall on the background side of the local threshold and disappear.
Once discarded, the downstream classifier cannot recover it from shape context or neighboring pixels.

### Red cannot be defined only as a pixel color

Crop-relative color correction improved red detection under warm or dark conditions, but it introduced a semantic ambiguity.
A red printed marking and a red tile back can have similar relative color evidence.
The preprocessing code does not know whether red belongs to ink, tile material, or background.

Connected-component size and border-contact rules can reject obvious red surfaces.
Those rules remain geometric heuristics rather than semantic understanding.
They can also reject valid large or border-adjacent red markings when detector cropping, rotation, or perspective changes the component geometry.

### Detector crops do not provide the preconditions assumed by palette inference

The experiment did not receive a guaranteed tile-face-only image.
The bounding box can contain background, tile edge, shadow, another tile, or a back-facing surface.
Consequently:

- The brightest neutral cluster is not guaranteed to be the tile face.
- The largest bright area is not guaranteed to be the tile face.
- The image center is not guaranteed to be white tile material.
- The crop perimeter cannot be discarded safely because valid markings may approach it.
- A compact neutral cluster is not proof of a correct illuminant reference.

A deterministic palette stage would need either a reliable face segmentation contract or enough learned shape and semantic context to distinguish these cases.
The latter moves the responsibility toward a learned recognizer rather than a simple normalization rule.

### Three-color conversion removes information before its value has been demonstrated

The original RGB crop contains continuous brightness, hue, local contrast, blur, and shape evidence.
The tested pipeline reduces that evidence to three discrete colors before the classifier can evaluate it.

The dark failure examples show that the discarded information can remain perceptible and potentially useful to a learned classifier even when deterministic thresholding cannot assign it confidently.
This does not prove that raw RGB classification will meet the product target, but it does show that mandatory irreversible quantization requires stronger evidence than was available in PRODUCT-ADR-RECOGNITION-001.

## Judgment

Do not adopt the tested fixed-CIELAB or Sauvola-based three-color conversion as the mandatory sole input to the tile classifier.

Retain the implementation and outputs as a reproducible deterministic baseline and diagnostic tool.
Sauvola or continuous local-lightness normalization may still be useful as:

- An auxiliary classifier channel that does not replace the original RGB crop.
- One augmentation or alternate view during training.
- A diagnostic visualization for capture quality.
- A comparison baseline against learned illumination invariance.

The next classifier investigation should preserve raw RGB and compare at least:

| condition | input and training policy |
|---|---|
| A | Raw RGB with weak geometric and photometric augmentation. |
| B | Raw RGB with stronger exposure, gamma, shadow, color-temperature, noise, blur, compression, and detector-crop jitter augmentation. |
| C | Raw RGB plus a continuous locally normalized lightness channel. |
| D | The current deterministic three-color representation as a measured baseline. |

Manual crops must not be mixed with the approximately 1.3 million JP crops in their natural ratio.
The training sampler must deliberately preserve manual-domain influence, and validation must split by original capture or capture group rather than by individual crop.

A learnable illumination-normalization front end is a follow-up condition, not the starting architecture.
It becomes justified only when the raw-RGB classifier with strong capture augmentation remains specifically weak on dark or shadowed manual strata.
If added, it should be optimized jointly with tile classification and must preserve access to the original RGB signal.

## Impact on the staged pipeline

PRODUCT-ADR-RECOGNITION-001 currently describes shared illumination normalization, shared palette inference, and multi-level color quantization as mandatory stages before the tiny CNN.
This investigation provides counter-evidence to that mandatory information-reduction path.

The investigation does not by itself supersede the staged detector-then-classifier architecture.
It supports keeping NanoDet as a generic tile-region detector while moving illumination, color, and shape invariance into the tile classifier or into a jointly trained front end.

A later classifier comparison should determine whether PRODUCT-ADR-RECOGNITION-001 requires amendment or supersession for the normalization and quantization stages.

## Value of the failed approach

The experiment narrowed the design space in several useful ways:

- It demonstrated the fixed-threshold domain shift on the actual JP and deployment-capture corpora.
- It showed that local adaptive thresholding can improve aggregate appearance while still destroying low-area discriminative evidence.
- It separated red-color detection from red-ink interpretation and exposed the need for shape or semantic context.
- It showed that a heuristic neutral-reference confidence score can be internally consistent while semantically wrong.
- It produced a deterministic baseline, per-crop metrics, and failure-oriented contact sheets for later learned-model comparisons.
- It established that preprocessing quality must be judged by downstream recognition, not by output-pixel proportions or visual cleanliness alone.

The result is therefore a useful negative finding rather than an abandoned implementation attempt.
