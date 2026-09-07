# PRODUCT-INV-RECOGNITION-013: Evaluate perspective-aware tile-classifier augmentation

- **status**: completed
- **date**: 2026-09-03
- **area**: recognition
- **depends_on**:
  - PRODUCT-INV-RECOGNITION-012
- **related_tasks**:
  - PRODUCT-TASK-SYSTEM-002-16
  - PRODUCT-TASK-SYSTEM-002-05

## Trigger

PRODUCT-INV-RECOGNITION-012 showed that preserving a larger late feature map materially improves offline dense-angle and crop-perturbation accuracy while retaining a useful mobile latency advantage in the `mobile-tile-f8-r1` candidate. However, live iPhone Recognition still exhibits a clear view-angle-dependent failure: tiles classify reliably when close to front-facing, while oblique views can make fine-grained manzu identity unstable, including observed `6m -> 5m/7m` errors.

The existing `random360` augmentation is not missing. It assigns each training sample a deterministic pseudo-random angle in `[-180°, +180°)` for each epoch and rotates the already-cropped `64 x 64` classifier image with an affine image-plane rotation. Evaluation likewise sweeps image-plane rotation densely.

That transformation does not reproduce the full image formation seen by the production camera. An oblique physical tile can undergo perspective foreshortening, anisotropic scale, shear-like apparent geometry, unequal edge lengths, and a changed axis-aligned detector crop. The resulting classifier crop can therefore differ materially from a front-facing crop that was merely rotated in 2D.

INV-012 deliberately held augmentation fixed so the architecture comparison remained controlled. This image-formation mismatch is therefore a separate investigation rather than an extension of INV-012.

## Question

Can a perspective/foreshortening-aware training and evaluation distribution recover stable tile identity under realistic oblique camera views without changing the production `[N,1,64,64] -> [N,35]` classifier contract or sacrificing the useful iPhone latency of the selected mobile architecture?

## Hypothesis

The dominant remaining live failure is a training/evaluation distribution gap rather than insufficient in-plane rotation invariance.

A classifier trained only on image-plane `random360` sees rotated versions of an already-normalized crop. It does not learn the shape changes caused by viewing a planar tile from an oblique camera pose or by taking an axis-aligned crop after that transformation. Training with bounded projective/anisotropic geometry that better approximates those effects should improve `5m/6m/7m` separation under oblique views while preserving front-facing accuracy.

The most important augmentation is not arbitrary visual distortion. It is a bounded approximation of the production sequence:

```text
front-facing tile crop / tile content
  -> camera-like in-plane rotation + foreshortening / projective warp
  -> axis-aligned detector-style recrop with small bbox jitter
  -> existing production classifier resize / letterbox semantics
  -> gray64 normalization
```

## Baselines

Keep explicit baselines so architecture and augmentation effects are not mixed:

1. **`mobile-tile-f8-r1` + current `random360`** — selected INV-012 architecture and current live failure reference.
2. **Plain e150 + current `random360`** — historical accuracy/rollback reference.

Do not defer Plain to a later follow-up. Run a controlled `2 architectures x 4 augmentations` matrix from the start so the experiment can separate augmentation effects from architecture effects:

```text
Plain  x A0 / A1 / A2 / A3
f8-r1 x A0 / A1 / A2 / A3
```

The existing 150-epoch A0 checkpoints may be reused when their dataset, normalization, optimizer/training contract, and checkpoint-selection policy match this investigation. A `--retrain-a0` mode should remain available for a fully fresh eight-run matrix. A1/A2/A3 are trained independently for both architectures under the same seed and training contract.

## Controlled variables

For the Plain/f8-r1 controlled comparison preserve:

- source dataset and split identity from INV-012;
- grayscale `64 x 64` classifier input;
- 35-class label order including `invalid/background`;
- current gray64 normalization;
- each architecture definition unchanged within its four augmentation conditions: historical Plain or INV-012 f8-r1 with its late `8 x 8` endpoint and one terminal repeat;
- optimizer family, effective batch, nominal 150 epochs, seed, checkpoint selection policy, and ONNX export contract where practical;
- production provider target `wasm-simd`, one thread;
- red-five specialist and detector behavior unchanged.

The independent variable is the geometric augmentation/image-formation simulation.

## Augmentation conditions

Use a small controlled progression rather than combining many unrelated augmentations at once.

### A0 — current reference

`random360` exactly as implemented today: affine image-plane rotation of the cached classifier crop with bilinear sampling and border padding.

### A1 — anisotropic affine geometry

Add bounded non-uniform scale and shear/foreshortening-like affine geometry to the current rotation path. This condition tests whether simple directional compression/stretching is enough before introducing full projective warping.

Initial bounds are experiment parameters, not product constants. Start conservatively and record them explicitly; reasonable first-pass ranges are approximately:

- in-plane rotation: retain the existing random rotation distribution or a separately justified production-weighted range;
- `scale_x`: roughly `0.75 .. 1.25`;
- `scale_y`: roughly `0.85 .. 1.15`;
- small bounded shear.

If reviewed real captures show narrower or asymmetric geometry, calibrate the bounds from those captures rather than preserving arbitrary symmetric ranges.

### A2 — perspective / homography geometry

Apply a bounded four-corner projective warp representing a planar tile viewed obliquely. Parameterize the transform so it can produce unequal top/bottom or left/right edge lengths rather than merely rotating or shearing the image.

Prefer camera-like quadrilateral perturbation or homography parameters whose severity can be reported in interpretable terms. Avoid extreme synthetic warps that would not occur inside the supported capture surface.

### A3 — perspective plus detector-style recrop

Apply the selected projective transform on a larger working canvas, derive the transformed tile/content quadrilateral, form an axis-aligned bounding rectangle, apply bounded bbox translation/scale jitter, then pass that crop through the same classifier preprocessing semantics used by production.

This condition is important because the live classifier does not receive a perspective-warped canonical square directly; it receives an axis-aligned crop produced downstream of detector localization.

A3 is the preferred final synthetic approximation if it can be implemented without changing the production runtime contract.

## Real-crop evidence

Synthetic geometry alone is not an acceptance gate. Build or reuse a small reviewed real detector-crop holdout containing the actual failure surface.

At minimum include:

- front-facing examples that remain correct;
- moderate oblique examples from both left/right or near/far directions where available;
- multiple `5m`, `6m`, and `7m` crops, especially observed `6m -> 5m/7m` failures;
- neighboring correctly recognized suit tiles so the benchmark does not overfit one hand;
- exact production crop extraction/preprocessing when possible.

The holdout must remain separate from training when used for promotion decisions. If live failure crops are added to training, keep distinct reviewed captures for final acceptance.

## Evaluation

### 1. Existing continuity metrics

Retain the INV-012/INV-011 dense in-plane angle evaluation to ensure the new augmentation does not destroy existing rotation robustness or front-facing accuracy.

Record at least:

- manual dense-angle mean/worst accuracy;
- zero-degree accuracy;
- per-class confusion, especially `5m`, `6m`, and `7m`.

### 2. Perspective sweep

Create a deterministic evaluation grid independent from the random training sampler. Sweep bounded perspective/anisotropic severity and direction so results expose a narrow failure surface rather than averaging it away.

Report:

- mean and worst accuracy by perspective condition;
- severity at first material degradation;
- `5m/6m/7m` confusion by condition;
- front-facing versus oblique delta.

### 3. Real detector-crop holdout

Evaluate A0/A1/A2/A3 surviving models on the same reviewed real holdout. This is the primary offline promotion evidence because the current bug appears only when the physical camera view becomes oblique.

### 4. Live iPhone acceptance

For the best candidate, reproduce the same physical hand/view transition that exposed the failure:

- confirm close-to-front-facing behavior remains correct;
- tilt/view the hand through representative oblique angles;
- explicitly inspect the known `6m` failure surface and neighboring manzu identities;
- record production base-classifier inference and total pipeline timing so robustness is not gained by accidentally changing deployment behavior.

## Decision criteria

A perspective-aware replacement is acceptable only if all of the following hold:

1. The known oblique live failure is materially reduced or eliminated on repeated iPhone observations, especially `6m -> 5m/7m`.
2. The reviewed real detector-crop holdout improves materially over f8-r1 `random360` without an obvious new within-suit confusion surface.
3. Front-facing and dense in-plane rotation accuracy do not regress materially.
4. The production `[N,1,64,64] -> [N,35]` runtime contract, normalization, label mapping, and provider policy remain unchanged.
5. Target-device inference cost remains in the same architecture-dependent range; augmentation is a training-time change and should not add runtime operators.

Do not accept a candidate solely from synthetic perspective metrics. The live iPhone view-angle failure is the trigger and therefore the final acceptance surface.

## Expected interpretation

Possible outcomes:

- **A1 is sufficient**: the dominant gap is directional compression/scale rather than full projective geometry; prefer the simpler augmentation.
- **A2 improves but A3 is required**: projective shape change matters, and detector-style axis-aligned recrop is a significant part of the production distribution gap.
- **synthetic perspective improves but real crops remain weak**: the transform model is still not representative enough; prioritize real detector-crop training/evaluation rather than increasing arbitrary warp severity.
- **f8-r1 and Plain both improve similarly**: augmentation/image formation is the dominant factor and architecture is secondary.
- **f8-r1 remains weak while Plain is robust under identical perspective-aware training**: architecture still contributes materially and a later architecture decision may be required.
- **no augmentation fixes the live failure**: investigate detector localization/crop rectification or explicit geometric normalization before further classifier augmentation work.

## Implementation: 2026-09-03

The first INV-013 experiment implementation is:

- `tools/recognition/perspective_classifier_augmentation.py`
- `tools/recognition/run_perspective_classifier_experiment.py`
- `tools/recognition/tests/test_perspective_classifier_experiment.py`

The runner defines the full eight-condition Plain/f8-r1 x A0/A1/A2/A3 matrix. By default A0 reuses the existing 150-epoch Plain and f8-r1 checkpoints only when their recorded dataset, class order, gray64 normalization, epoch count, effective batch, optimizer hyperparameters, seed, and `random360` augmentation match the current experiment; otherwise A0 is retrained automatically. `--retrain-a0` forces a completely fresh eight-condition training matrix. A1/A2/A3 are always trained per architecture with the same frozen v3 dataset, gray64 normalization, 150-epoch default, AdamW/cosine schedule, seed, effective batch, and existing `0/15/30/45°` checkpoint-selection policy.

Training geometry is sample/epoch deterministic and uses one shared geometry stream across architectures/conditions so corresponding samples see comparable random factors. A0 preserves the existing `random360` affine-grid implementation exactly. A1 adds bounded anisotropic scale and shear. A2 adds structured perspective yaw/pitch plus bounded keystone displacement. A3 embeds the already-cached 64x64 crop on a larger replicate-padded canvas, applies the projective transform, derives the transformed content's axis-aligned bbox, adds bounded detector-style center/scale jitter, and letterboxes that recrop back to the unchanged 64x64 classifier contract.

A3 is explicitly a synthetic proxy: because the frozen dataset contains classifier-ready 64x64 crops, pixels outside the historical crop cannot be reconstructed. The implementation therefore does not claim to recreate the original camera frame. The optional `--real-holdout-database` path evaluates a separate reviewed SQLite holdout of production-preprocessed gray64 detector crops and remains the primary offline bridge to the live failure surface.

Each successful condition records:

- historical dense in-plane angle evaluation;
- an independent deterministic perspective evaluation grid containing front-facing, anisotropic compression, left/right yaw, pitch, and detector-style recrop cases;
- explicit `5m/6m/7m` confusion and worst `6m -> 5m/7m` rate;
- optional reviewed real-holdout accuracy/confusion;
- ONNX dynamic-batch parity, graph statistics, and CPU batch benchmark.

### Follow-up mixture sweep

The isolated A0/A1/A2/A3 comparison is an augmentation ablation, not the intended final training distribution. In particular, an A3-only model never receives an explicit canonical/original branch during training. The follow-up runner therefore trains stochastic mixtures of `Original + A0 + A1 + A2 + A3` and compares the same recipes for Plain and f8-r1:

- `tools/recognition/run_perspective_classifier_mixture_experiment.py`
- `tools/recognition/tests/test_perspective_classifier_mixture_experiment.py`

The initial ratio sweep is:

| recipe | Original | A0 | A1 | A2 | A3 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `mix-light` | 30% | 25% | 15% | 20% | 10% |
| `mix-mid` | 20% | 20% | 15% | 30% | 15% |
| `mix-heavy` | 10% | 15% | 15% | 35% | 25% |

Branch selection is deterministic by sample/epoch and uses the same choice stream for both architectures. A0/A1/A2/A3 geometry likewise reuses the shared sample/epoch geometry stream so Plain and f8-r1 see matched augmentation draws.

The mixture runner also corrects A3's detector-style bbox derivation for the compact dataset's pre-existing 64x64 letterbox. It reconstructs the actual content rectangle from each row's `original_width`/`original_height` using the same aspect-preserving resize rounding as the dataset builder, then derives the transformed detector-style bbox from that content rectangle rather than from the entire already-letterboxed 64x64 square. The original INV-013 isolated-run behavior remains unchanged unless these content extents are passed explicitly, preserving reproducibility of the first eight-condition result.

## Conclusion

INV-013 concludes that perspective/foreshortening-aware augmentation is effective and that the original `random360`-only training distribution was a real contributor to the live oblique-view regression. The isolated A1/A2 results materially improved synthetic perspective robustness, and the follow-up stochastic mixture of `Original + A0 + A1 + A2 + A3` avoided the severe canonical/dense-angle regression observed when stronger transforms were trained as exclusive conditions.

The strongest mixture results confirm that keeping canonical and progressively distorted views in the same training distribution is the appropriate direction. In particular, `plain-mix-heavy` retained manual dense-angle mean/worst accuracy `0.9571180556 / 0.9488888889` while reaching perspective mean/worst accuracy `0.9579797980 / 0.9355555556`; its worst measured `6m -> 5m/7m` rate was `0.0`. `f8-r1-mix-heavy` achieved still stronger aggregate perspective performance (`0.9749494949` mean / `0.9622222222` worst) and preserved its deployment-speed advantage, demonstrating that the perspective-aware mixture materially improves the selected mobile family as well.

However, perspective-aware augmentation does **not** fully resolve the fine-grained manzu confusion that triggered this investigation. Under the same `mix-heavy` training distribution, Plain eliminated the measured `6m -> 5m/7m` confusion while f8-r1 still recorded a worst rate of `0.1666666667`. The same pattern is visible across the mixture sweep: all three Plain recipes recorded `0.0`, whereas f8-r1 retained `0.3333333333 / 0.1666666667 / 0.1666666667` for light/mid/heavy respectively.

Because the architectures were held fixed inside this investigation, INV-013 does not identify which f8-r1 layer or operator causes that residual failure. It does establish that augmentation/image-formation mismatch is not a sufficient remaining explanation: after applying the same perspective-aware training distribution, the residual `5m/6m/7m` confusion is architecture-dependent in the measured comparison. This is consistent with the possibility that the current MobileNetV3-derived bottleneck/depthwise/global-pooling topology loses some fine local spatial discrimination needed to separate neighboring manzu classes, but that mechanism must be tested in a separate architecture investigation rather than asserted here.

Therefore:

1. Treat perspective-aware stochastic mixture training as the preferred augmentation strategy for subsequent classifier candidates; do not return to A1/A2/A3-exclusive training as a production recipe.
2. Do not consider INV-013 alone sufficient to accept `f8-r1` as the final production classifier despite its improved aggregate perspective metrics and latency.
3. Re-open the classifier architecture question in a separate investigation focused on preserving fine local discriminative structure while retaining the mobile latency advantage, using the perspective mixture recipe as a fixed training control.
4. Keep Plain mixture results as the accuracy reference demonstrating that the observed `5m/6m/7m` failure is solvable under the present data/augmentation formulation.

INV-013 is therefore complete: perspective-aware augmentation materially improves robustness, but the remaining fine-grained confusion is not fully explained by perspective distribution mismatch and requires architecture-level follow-up.

## Non-goals

- changing NanoDet architecture or detector training;
- introducing oriented bounding boxes into production inside this investigation;
- changing the production classifier input/output contract;
- changing red-five classification;
- broad neural architecture search;
- arbitrary photometric augmentation unrelated to observed evidence;
- adding runtime perspective rectification before proving a training-distribution fix is insufficient.

## Deliverables

- reproducible perspective-aware augmentation implementation with explicit parameters;
- deterministic perspective/foreshortening evaluation sweep;
- reviewed real detector-crop holdout covering front-facing and oblique views;
- complete Plain/f8-r1 x A0/A1/A2/A3 comparison;
- per-class `5m/6m/7m` confusion evidence;
- final iPhone live acceptance or rejection;
- explicit recommendation for production model training and any next investigation if augmentation alone is insufficient.
