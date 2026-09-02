# PRODUCT-INV-RECOGNITION-011: Evaluate mobile-oriented base-tile classifiers

- **status**: completed
- **date**: 2026-09-02
- **trigger**: PRODUCT-INV-RECOGNITION-008 promoted the 150-epoch `PlainTileShapeClassifier` random360 checkpoint as the speed-oriented base-classifier candidate and the production model set now uses that Plain ONNX artifact. iPhone 13 runtime instrumentation nevertheless shows that base-classifier inference remains substantially slower than the much larger NanoDet detector artifact. Code inspection confirmed that batching is already correct: all detector crops are passed to one dynamic `[N,1,64,64]` classifier invocation. The remaining mismatch is architectural. The Plain model is a small conventional dense-convolution CNN, while the detector uses ShuffleNetV2, GhostPAN, and depthwise convolution designed for efficient mobile inference. Artifact bytes therefore do not predict deployed latency in the current runtime.
- **scope**: On the frozen gray64 v3 35-class corpus, compare the accepted Plain random360 e150 classifier against mobile-oriented grayscale classifiers built from ShuffleNetV2 and MobileNetV3-Small families. Keep the current per-crop dynamic-batch input contract unchanged so the experiment isolates classifier architecture rather than changing the Recognition pipeline. Train all new candidates with the accepted deterministic random360 policy and 150-epoch schedule, run the same dense-angle accuracy protocol used by INV-007/008, export dynamic-batch ONNX, verify parity, characterize graph/MAC/activation cost, benchmark single-thread desktop ORT, and benchmark ONNX Runtime Web on iPhone 13 using the current production execution-provider path.
- **non_scope**: changing NanoDet, merging detection and tile identity into one multi-class detector, packing several tile crops into one spatial composite image, shared-backbone multi-slot classifiers, red-five classification, detector crop extraction changes, dataset rebuild/relabeling, quantization, pruning, Web Worker classifier parallelism, WASM threaded execution tuning, temporal tracking/cache reuse, scoring, or UI behavior. Those are follow-ups only if a mobile-oriented per-crop classifier remains insufficient.
- **source_refs**:
  - PRODUCT-INV-RECOGNITION-007
  - PRODUCT-INV-RECOGNITION-008
  - PRODUCT-INV-RECOGNITION-009
  - PRODUCT-TASK-SYSTEM-002-08
  - tools/recognition/tile_shape_classifier.py
  - tools/recognition/run_rotation_classifier_experiment.py
  - product/frontend/src/recognition/classifier/runtime.ts
  - product/frontend/src/recognition/classifier/preprocessing.ts
  - product/frontend/src/recognition/model-runtime/onnx-session-factory.ts
  - product/frontend/src/recognition/model-runtime/production-model-set.json
  - tools/recognition/nanodet/configs/e1_nanodet_plus_m_320_real_capture_ft10_l10.yml
  - .local/recognition/tile_classifier_datasets/gray35_jp500_seed42_v3_jp189.sqlite
- **planned_outputs**:
  - tools/recognition/mobile_classifier_experiment_models.py
  - tools/recognition/run_mobile_classifier_experiment.py
  - tools/recognition/tests/test_mobile_classifier_experiment.py
  - .local/recognition/mobile_classifier_experiment/
  - browser/iPhone benchmark evidence recorded in this investigation
  - PRODUCT-INV-RECOGNITION-011
- **related_adrs**:
  - PRODUCT-ADR-RECOGNITION-004

## Question

Determine whether replacing the conventional dense-convolution Plain classifier with a mobile-oriented CNN materially reduces real browser/iPhone base-classifier latency while retaining enough arbitrary-angle tile-classification accuracy to remain a production candidate.

The investigation should answer:

1. Is the current Plain classifier slow primarily because its dense convolution topology is poorly matched to mobile/WASM execution rather than because dynamic batching itself is broken?
2. Can ShuffleNetV2 or MobileNetV3-Small reduce base-classifier latency at the same `[N,1,64,64] -> [N,35]` runtime contract?
3. Which width/model variant lies on the best accuracy-versus-iPhone-latency Pareto frontier?
4. If a mobile-oriented classifier is still too slow, is there enough evidence to justify a more invasive shared-backbone/multi-tile architecture or browser-level parallel execution next?

## Current production/runtime evidence

The current production model set binds the base classifier to:

```text
tile-plain-gray35-random360-e150.onnx
```

The Plain model is approximately 1.5 MB, while the current NanoDet detector is several times larger as a serialized artifact. Despite that, iPhone diagnostics show the base classifier consuming substantially more inference time than NanoDet under normal multi-tile candidate counts.

This is not evidence that the classifier batch is accidentally serialized into one ORT invocation per crop. The current runtime already builds one tensor:

```text
[N, 1, 64, 64]
```

and invokes the base classifier once per Recognition evaluation. Red-five candidates are separately batched into one specialist invocation.

The current Plain topology is a conventional dense CNN:

```text
64x64 grayscale
  -> Conv 1 -> 32, 5x5
  -> MaxPool
  -> Conv 32 -> 64, 3x3
  -> MaxPool
  -> Conv 64 -> 128, 3x3
  -> MaxPool
  -> Conv 128 -> 192, 3x3
  -> AdaptiveAvgPool2d(1)
  -> 192 -> 256 -> 35
```

Its ordinary convolution workload is roughly 55 million MAC per crop. A representative 16-18 crop frame therefore approaches one billion dense-convolution MAC before red-five classification.

By contrast, the current NanoDet detector uses a mobile-oriented topology:

```text
ShuffleNetV2 1.0x backbone
GhostPAN
Depthwise-enabled feature pyramid/head paths
```

The key hypothesis of this investigation is therefore not "smaller ONNX is faster". It is:

> A classifier designed around depthwise separable convolution, channel shuffle, inverted residuals, and mobile-oriented channel schedules may execute substantially faster than the existing dense Plain CNN even when both models receive the same total crop pixels.

## Frozen dataset and classifier contract

Use the same frozen classifier corpus as INV-007/008:

```text
.local/recognition/tile_classifier_datasets/
  gray35_jp500_seed42_v3_jp189.sqlite
```

Recorded split sizes remain:

| split | samples |
|---|---:|
| train | 19,593 |
| jp_val | 6,800 |
| manual_val | 450 |
| total | 26,843 |

Do not rebuild or relabel the corpus.

Every compared classifier must preserve the current runtime input/output contract:

```text
64 x 64 grayscale uint8 letterboxed crop
  -> float32 [0,1]
  -> frozen v3 normalization
  -> dynamic batch [N,1,64,64]
  -> [N,35] logits
```

Use the frozen normalization:

```text
mean = 0.6815832403977466
std  = 0.2725553681973969
```

The 35 classes remain the canonical 34 base tile identities plus `invalid`. Red-five identity remains outside this investigation.

Do not spatially concatenate crops for the first experiment. The purpose is to measure whether mobile operators alone solve the runtime problem without changing Recognition semantics or crop independence.

## Reference candidate

Use the already accepted INV-008 150-epoch Plain result as the fixed baseline. Do not retrain it merely for this investigation.

Relevant accepted metrics are:

| model | manual mean | manual worst | JP mean | JP worst | ORT CPU median b16 |
|---|---:|---:|---:|---:|---:|
| Plain random360 e150 | 0.94743 | 0.94000 | 0.99959 | 0.99794 | 17.86 ms |
| production C8 reference | 0.97222 | 0.96000 | 0.99984 | 0.99941 | 52.68 ms |

The Plain e150 artifact is the comparison baseline because it is the currently selected speed-oriented classifier and because its graph is already known to be much faster than C8 on desktop ORT.

## Candidate architectures

Compare the following new mobile-oriented candidates from scratch.

### ShuffleNetV2 0.5x

Use the standard ShuffleNetV2 stage/block design at width multiplier 0.5x, adapted only for the product contract:

- input channels: 1 instead of 3;
- input resolution: 64x64;
- output head: 35 logits;
- no ImageNet pretrained weights;
- preserve channel-split, channel-shuffle, depthwise 3x3, and pointwise 1x1 behavior.

Do not replace the ShuffleNet blocks with a custom "ShuffleNet-like" approximation. The purpose of this row is to test the mobile operator family already proven deployable in the project's NanoDet path.

### ShuffleNetV2 1.0x

Use the same standard topology at width multiplier 1.0x. This variant is especially relevant because the current NanoDet detector already uses a ShuffleNetV2 1.0x backbone successfully in the browser runtime.

The classifier implementation need not share weights or code with NanoDet, but architectural fidelity should make its deployment behavior interpretable against the existing detector evidence.

### MobileNetV3-Small 0.5x

Use a standard MobileNetV3-Small topology at width multiplier 0.5x, adapted to one grayscale input channel and 35 output classes. Preserve its inverted residual, depthwise convolution, squeeze-and-excitation, and hard-swish/hard-sigmoid structure where applicable.

### MobileNetV3-Small 1.0x

Use the standard MobileNetV3-Small topology at width multiplier 1.0x with the same grayscale/head adaptations.

No candidate uses pretrained weights. Pretraining would introduce a separate RGB/input-domain transfer variable and is not needed to answer the runtime architecture question.

## Architecture fidelity and implementation rules

Prefer well-established reference topology definitions, but do not make the experiment depend on an online model download. If the installed torchvision version provides the required architecture and permits deterministic one-channel/head adaptation without hidden pretrained assets, reuse its implementation. Otherwise implement the documented standard topology locally in the experiment module.

Record for every candidate:

- stage widths;
- stride/downsampling schedule at 64x64;
- block count;
- activation family;
- presence/count of depthwise convolutions;
- parameter count;
- ordinary Conv/Gemm MAC estimate;
- ONNX operator histogram.

The 64x64 adaptation must not silently remove major standard blocks solely to win latency. If a topology becomes pathological at 64x64 because of excessive downsampling, document the exact issue and make only the minimum defensible stride adaptation, recorded explicitly in the result.

## Training protocol

Train every new candidate with the accepted INV-008 Plain e150 learning contract so runtime architecture is the main changed variable.

Use:

| setting | value |
|---|---:|
| epochs | 150 |
| augmentation | deterministic continuous random360 |
| angle distribution | `Uniform(-180,+180)` |
| effective batch size | 512 |
| optimizer | AdamW |
| learning rate | 0.001 |
| weight decay | 0.0001 |
| scheduler | cosine annealing, `T_max=150` |
| seed | 42 |
| AMP | enabled when numerically safe |
| TF32 | enabled on RTX 3090 path |

Preserve deterministic angle assignment keyed by `(seed, epoch, sample_id)` exactly as in INV-007/008 so each architecture sees the same rotated training view for a given source sample and epoch.

If a model cannot fit physical batch 512, use gradient accumulation to preserve effective batch 512. Record physical micro-batch and peak training VRAM, but training throughput does not decide deployment selection.

Unlike INV-008, do not stage candidate elimination at 50 epochs. Plain was proven materially undertrained at 50 epochs; using a short horizon as an early rejection gate would risk eliminating a mobile architecture for optimization-speed differences rather than architecture quality.

## Checkpoint selection and dense-angle evaluation

Use the same checkpoint-selection protocol as INV-007/008.

During training, evaluate `manual_val` at:

```text
0 / 15 / 30 / 45 deg
```

and select the best checkpoint by mean accuracy over those angles.

After training, evaluate the selected checkpoint on the same deterministic dense rotation grid:

```text
0.000
5.625
11.250
...
354.375 deg
```

for 64 orientations on both:

```text
manual_val: 450 x 64
jp_val:   6,800 x 64
```

Record at least:

- mean accuracy;
- worst-angle accuracy and angle;
- best-angle accuracy;
- standard deviation across angle;
- 0-degree accuracy;
- total errors;
- per-class accuracy;
- confusion matrix.

Reuse the Plain e150 dense-angle result from INV-008 as the comparison baseline rather than recomputing it unless verification is needed because evaluation code changed.

## ONNX deployment gate

Every candidate must export to the exact production-style dynamic-batch contract:

```text
[N,1,64,64] -> [N,35]
```

Use the repository's existing PyTorch/ONNX environment and opset 16 unless a documented compatibility reason requires otherwise.

For each candidate:

1. export ONNX with dynamic batch axis only;
2. run `onnx.checker`;
3. compare PyTorch and ORT CPU logits on deterministic parity inputs;
4. require zero argmax mismatches;
5. record maximum/mean absolute logit error;
6. run an ONNX Runtime Web smoke load using the same runtime package used by the frontend;
7. record any operators that force provider fallback or fail in WASM/WebGL.

A model that is fast in PyTorch but cannot execute through the current PWA ORT path is not a candidate.

## Static deployment characterization

For the fixed ONNX graph of each architecture record:

- trainable parameter count;
- ONNX bytes;
- Conv/Gemm MAC estimate per crop;
- estimated MAC at batch 16;
- operator histogram;
- count of depthwise convolutions;
- count of 1x1 pointwise convolutions;
- inferred peak intermediate-tensor bytes at batch 16 where possible.

Do not rank candidates by ONNX bytes or parameter count. These are explanatory metrics only.

The main cost evidence is measured latency. INV-011 is specifically triggered by the observation that a smaller serialized Plain classifier can be slower than the larger mobile-oriented detector.

## Desktop ORT screening benchmark

Use the same fixed desktop benchmark contract as INV-007/008 so historical results remain comparable.

ONNX Runtime `CPUExecutionProvider`:

```text
intra_op_num_threads = 1
inter_op_num_threads = 1
execution_mode = sequential
batch = 16
shape = [16,1,64,64]
warm-up = 100
measurement = 1000
```

Record:

- mean ms/batch;
- median ms/batch;
- p95 ms/batch;
- ms/image;
- CPU identity;
- ORT version;
- thread configuration.

The existing Plain e150 value of approximately 17.86 ms/batch is the screening baseline.

Desktop timing is not sufficient for selection. A candidate may use operators whose relative performance differs materially in ORT Web/WASM.

## iPhone 13 ONNX Runtime Web benchmark

Unlike INV-007/008, direct iPhone measurement is part of this investigation rather than a deferred promotion gate. The performance question originates from the production browser runtime, so the final Pareto decision must use the actual deployment engine.

Benchmark each deployment-valid candidate on iPhone 13 using the same application/browser build and runtime configuration.

At minimum record the base-classifier inference time for fixed synthetic/deterministic batches:

```text
N = 1
N = 4
N = 8
N = 16
```

If practical, also measure `N = 24` because a full Recognition frame can exceed 16 candidates when hand, dora, and meld regions are all populated.

For every batch size record:

- selected execution provider;
- warm-up policy;
- median inference ms;
- p95 inference ms;
- ms/image;
- whether the run is `wasm-simd`, `wasm-threaded`, or `webgl`;
- `navigator.hardwareConcurrency`;
- whether the page is `crossOriginIsolated`.

The current production provider preference chooses `wasm-simd` first and configures that provider with `numThreads=1`. Keep that behavior for the primary comparison. Do not change thread count or add worker parallelism while comparing architectures, because that would confound model topology with runtime scheduling.

After the fixed-batch benchmark, integrate the leading candidate into a temporary/dev model set and record real Recognition diagnostics on representative iPhone 13 frames:

- detector inference ms;
- base-classifier preprocessing ms;
- base-classifier inference ms;
- candidate count;
- total pipeline ms.

The purpose is to confirm that a synthetic batch win translates into the actual pipeline and that classifier inference no longer dominates unexpectedly.

## Primary result tables

Accuracy:

| architecture | width | manual mean | manual worst | JP mean | JP worst |
|---|---:|---:|---:|---:|---:|
| Plain e150 reference | existing | 0.94743 | 0.94000 | 0.99959 | 0.99794 |
| ShuffleNetV2 | 0.5x | | | | |
| ShuffleNetV2 | 1.0x | | | | |
| MobileNetV3-Small | 0.5x | | | | |
| MobileNetV3-Small | 1.0x | | | | |

Static/desktop deployment:

| architecture | params | ONNX bytes | MAC/image | peak act b16 | ORT CPU median b16 | p95 b16 |
|---|---:|---:|---:|---:|---:|---:|
| Plain e150 | | 1,495,802 | ~55M | | 17.86 ms | |
| ShuffleNetV2 0.5x | | | | | | |
| ShuffleNetV2 1.0x | | | | | | |
| MobileNetV3-Small 0.5x | | | | | | |
| MobileNetV3-Small 1.0x | | | | | | |

Direct iPhone ORT Web latency:

| architecture | provider | N=1 median | N=4 | N=8 | N=16 | N=24 |
|---|---|---:|---:|---:|---:|---:|
| Plain e150 | | | | | | |
| ShuffleNetV2 0.5x | | | | | | |
| ShuffleNetV2 1.0x | | | | | | |
| MobileNetV3-Small 0.5x | | | | | | |
| MobileNetV3-Small 1.0x | | | | | | |

## Interpretation and selection rule

The investigation should identify an empirical Pareto frontier, not force one winner through an arbitrary weighted score.

Primary dimensions are:

1. dense arbitrary-angle `manual_val` mean/worst accuracy;
2. iPhone 13 ORT Web latency at realistic multi-crop batch sizes;
3. ONNX/Web deployment reliability;
4. desktop single-thread latency and static compute only as supporting explanation.

A mobile architecture is a strong production replacement candidate if it preserves most of the Plain e150 manual-domain accuracy while producing a clear iPhone latency reduction at `N=8-16` and remaining stable across angle.

Do not dismiss a candidate solely because its ONNX file is larger than Plain. Conversely, do not promote a tiny model if the actual WASM kernel/operator mix makes it slower.

If the fastest mobile candidate loses substantial manual-domain accuracy, compare the 0.5x/1.0x width frontier before changing augmentation or dataset composition.

If no tested mobile architecture materially improves iPhone latency over Plain, conclude that the bottleneck is not solved by standard mobile CNN operators and move to a separate investigation for one of:

- a shared-backbone classifier that spatially packs multiple known tile slots into one image and emits multiple class heads;
- merging tile identity into the detector itself;
- Web Worker / multi-session parallel classification;
- WASM threaded execution/provider tuning;
- quantization.

Do not mix those approaches into INV-011.

## Implementation boundary

Do not modify the accepted INV-007 experiment implementation in a way that changes historical reproducibility.

Create dedicated experiment code:

```text
tools/recognition/
  mobile_classifier_experiment_models.py
  run_mobile_classifier_experiment.py
  tests/test_mobile_classifier_experiment.py
```

The new runner may reuse frozen dataset loading, deterministic random360 augmentation, dense-angle evaluation, ONNX parity, graph-statistics, and benchmark helpers from existing experiment code where doing so does not mutate INV-007 semantics.

The production `PlainTileShapeClassifier` remains unchanged during the experiment. Production model promotion is a later explicit action after the investigation concludes.

## Required preflight tests

Before starting full 150-epoch training, require fast tests that verify:

- every candidate accepts `[N,1,64,64]` for `N > 1`;
- every candidate returns exactly `[N,35]`;
- dynamic batch survives ONNX export;
- ONNX Runtime CPU inference succeeds at `N=1` and `N=16`;
- PyTorch/ORT argmax parity is exact on deterministic inputs;
- ShuffleNet channel shuffle actually permutes the intended groups;
- depthwise layers use `groups == channels` where expected;
- MobileNet squeeze-and-excitation and hard activations survive export without unsupported custom operators;
- training augmentation remains deterministic across candidate architectures for a fixed `(seed, epoch, sample_id)`.

A candidate that fails the ONNX smoke gate should be fixed before expensive training rather than discovered after 150 epochs.

## Expected decision

INV-011 should end with one of these concrete outcomes:

1. **Mobile classifier wins**: one ShuffleNetV2/MobileNetV3 candidate retains acceptable dense manual accuracy and materially reduces iPhone multi-crop inference latency; carry it to an explicit production-promotion task.
2. **Width trade-off remains**: a smaller mobile candidate is fast but inaccurate and a larger candidate is accurate but slower; retain both on the Pareto frontier and decide from actual pipeline frame budget.
3. **Standard mobile CNN is insufficient**: none materially improves real iPhone latency at acceptable accuracy; close this path and open a separate investigation for shared-backbone multi-tile classification, detector-integrated classification, or browser-level parallel execution.

## Implementation record: 2026-09-02

The INV-011 experiment implementation is isolated from the accepted INV-007/008 model definitions and runner:

```text
tools/recognition/mobile_classifier_experiment_models.py
tools/recognition/run_mobile_classifier_experiment.py
tools/recognition/tests/test_mobile_classifier_experiment.py
```

`mobile_classifier_experiment_models.py` contains local standard-topology implementations of:

- ShuffleNetV2 0.5x with stage repeats `4 / 8 / 4` and channels `24 / 48 / 96 / 192 / 1024`;
- ShuffleNetV2 1.0x with stage repeats `4 / 8 / 4` and channels `24 / 116 / 232 / 464 / 1024`;
- MobileNetV3-Small 0.5x;
- MobileNetV3-Small 1.0x.

All four models accept grayscale `[N,1,64,64]` tensors and emit `[N,35]` logits. ShuffleNetV2 preserves channel split/shuffle and true depthwise convolutions. MobileNetV3-Small preserves inverted residuals, depthwise convolutions, squeeze-and-excitation, ReLU/HardSwish activations, and the standard MobileNetV3 BatchNorm settings (`eps=0.001`, `momentum=0.01`). No torchvision model download or pretrained weight is required.

The INV-011 runner reuses only generic frozen protocol helpers from `run_rotation_classifier_experiment.py`: v3 SQLite loading, deterministic per-`(seed, epoch, sample_id)` random360 assignment, tensor rotation, checkpoint-angle evaluation, dense 64-angle evaluation, ONNX export/parity helpers, graph shape/MAC inspection, and single-thread ORT CPU benchmarking. It does not add candidates to the INV-007 condition matrix or change historical INV-007 behavior.

Additional INV-011 deployment evidence includes:

- ONNX dynamic-batch smoke execution at `N=1` and `N=16`;
- exact output-shape check `[N,35]`;
- PyTorch/ORT argmax parity and float tolerance gate;
- ONNX depthwise/pointwise convolution counts;
- the current Plain e150 production artifact re-benchmarked under the same desktop ORT settings when available;
- resume only when the exact INV-011 implementation version and training contract match.

Run the preflight before starting the long training matrix:

```bash
PY=/srv/bugrat/data-lv/mjtensu/nanodet/nanodet/.venv/bin/python
"$PY" tools/recognition/tests/test_mobile_classifier_experiment.py
```

Then run the complete four-candidate 150-epoch matrix:

```bash
PY=/srv/bugrat/data-lv/mjtensu/nanodet/nanodet/.venv/bin/python
mkdir -p /srv/data/mjtensu/.local/recognition/mobile_classifier_experiment
nohup "$PY" tools/recognition/run_mobile_classifier_experiment.py \
  --database /srv/data/mjtensu/.local/recognition/tile_classifier_datasets/gray35_jp500_seed42_v3_jp189.sqlite \
  --output-root /srv/data/mjtensu/.local/recognition/mobile_classifier_experiment \
  > /srv/data/mjtensu/.local/recognition/mobile_classifier_experiment/run.log 2>&1 &
```

The runner output is designed to be resumable per candidate. Do not use `--overwrite-completed` unless intentionally retraining the exact completed condition.

## Desktop experiment result: 2026-09-02

All four mobile-oriented candidates completed training, dense-angle evaluation, ONNX export/parity, dynamic-batch smoke execution, graph characterization, and the fixed single-thread ORT CPU batch-16 benchmark under implementation version `inv011-mobile-v1`.

Accuracy:

| architecture | width | manual mean | manual worst | JP mean | JP worst |
|---|---:|---:|---:|---:|---:|
| Plain e150 reference | existing | 0.94743 | **0.94000** | 0.99959 | 0.99794 |
| ShuffleNetV2 | 0.5x | 0.87510 | 0.84667 | 0.99875 | 0.99809 |
| ShuffleNetV2 | 1.0x | 0.93785 | 0.92000 | 0.99937 | 0.99882 |
| MobileNetV3-Small | 0.5x | 0.92174 | 0.90222 | 0.99900 | 0.99794 |
| MobileNetV3-Small | 1.0x | **0.94872** | 0.93111 | **0.99948** | **0.99897** |

Desktop deployment cost:

| architecture | width | ONNX bytes | known Conv/Gemm MAC/image | depthwise convs | pointwise convs | ORT CPU median b16 | p95 b16 |
|---|---:|---:|---:|---:|---:|---:|---:|
| ShuffleNetV2 | 0.5x | 1,620,082 | 505,856 | 19 | 36 | 1.90 ms | 1.92 ms |
| ShuffleNetV2 | 1.0x | 5,250,977 | 964,992 | 19 | 36 | 6.22 ms | 6.45 ms |
| MobileNetV3-Small | 0.5x | 1,709,726 | 1,620,160 | 11 | 40 | **1.39 ms** | **1.41 ms** |
| MobileNetV3-Small | 1.0x | 6,223,234 | 5,233,408 | 11 | 40 | 3.12 ms | 3.15 ms |

The Plain e150 ONNX artifact was not present in the server repository used by this run, so it was not re-benchmarked in the same process. The historical INV-008 fixed desktop result remains 17.86 ms/batch-16 and is used only as prior comparative evidence, not as a same-run measurement.

The desktop result is decisive enough to reduce the iPhone candidate set. ShuffleNetV2 0.5x is both less accurate and slower than MobileNetV3-Small 0.5x. ShuffleNetV2 1.0x is likewise less accurate and slower than MobileNetV3-Small 1.0x. Neither ShuffleNet variant remains on the measured Pareto frontier.

MobileNetV3-Small 0.5x is the fastest graph but loses about 2.57 percentage points of dense manual mean accuracy and about 3.78 points of worst-angle accuracy versus Plain e150. It remains useful as a speed boundary but is not the primary production candidate.

MobileNetV3-Small 1.0x is the leading candidate. Relative to Plain e150 it improves dense manual mean slightly (`0.94872` versus `0.94743`) while losing about 0.89 percentage point at the worst angle (`0.93111` versus `0.94000`). Its fixed desktop ORT CPU batch-16 median is 3.12 ms. Compared with the historical Plain e150 result of 17.86 ms, this is approximately 5.7x lower latency, although that ratio must be confirmed in the same browser/device runtime before promotion.

**Desktop-stage decision:** carry only MobileNetV3-Small 1.0x as the primary iPhone 13 replacement candidate. Keep Plain e150 as the production/reference baseline. Do not spend iPhone benchmark effort on the ShuffleNet variants unless later evidence invalidates the MobileNet result. The remaining INV-011 gate is direct ONNX Runtime Web measurement on iPhone 13 and representative Recognition-frame timing.

### iPhone A/B benchmark harness

Do not replace the production classifier before the direct device comparison. A browser-verification-only harness is added instead:

```text
product/frontend/test/e2e/mobile-classifier-benchmark.html
product/frontend/test/e2e/mobile-classifier-benchmark-main.ts
```

`vite.config.ts` exposes that page only in the existing `browser-verification` multi-page build. The harness loads both:

```text
tile-plain-gray35-random360-e150.onnx
tile-mobilenet-v3-small-1.0x-random360-e150.onnx
```

and configures ONNX Runtime Web identically for both models:

```text
provider = wasm-simd
wasm.proxy = false
wasm.simd = true
wasm.numThreads = 1
executionMode = sequential
graphOptimizationLevel = all
```

It benchmarks deterministic synthetic tensors at:

```text
N = 1 / 4 / 8 / 16 / 24
warm-up = 10 runs per batch
measurement = 50 runs per batch
```

and renders median, p95, and median ms/image plus a JSON report containing `navigator.hardwareConcurrency`, `crossOriginIsolated`, and user-agent evidence. This isolates architecture/runtime performance from camera, detector, crop extraction, and classifier preprocessing while using the same ORT Web WASM configuration as the production `wasm-simd` path.

The MobileNet artifact is intentionally not bound into `production-model-set.json` yet. Promotion remains gated on this iPhone comparison.

## iPhone 13 ONNX Runtime Web result: 2026-09-02

The direct same-device A/B benchmark completed on iPhone 13 / Safari with the production-style single-thread WASM SIMD configuration:

```text
provider = wasm-simd
wasm.numThreads = 1
wasm.proxy = false
hardwareConcurrency = 4
crossOriginIsolated = false
warm-up = 10
measurement = 50
```

Measured latency:

| batch | Plain e150 median | Plain p95 | MobileNetV3-Small 1.0x median | Mobile p95 | speedup |
|---:|---:|---:|---:|---:|---:|
| 1 | 3 ms | 4 ms | 2 ms | 2 ms | 1.50x |
| 4 | 12 ms | 13 ms | 8 ms | 8 ms | 1.50x |
| 8 | 24 ms | 25 ms | 15 ms | 16 ms | 1.60x |
| 16 | 51 ms | 53 ms | 30 ms | 31 ms | 1.70x |
| 24 | 79 ms | 82 ms | 46 ms | 47 ms | 1.72x |

The iPhone result confirms that the desktop ordering generalizes to the actual ORT Web deployment path, but not at the same magnitude: the desktop historical comparison suggested about 5.7x while the direct iPhone gain is about 1.5-1.7x over the measured batch range. MobileNet's median cost stays near 1.9-2.0 ms/image at larger batches, versus about 3.0-3.3 ms/image for Plain.

Combined with dense-angle accuracy, MobileNetV3-Small 1.0x is the clear replacement candidate: manual mean is slightly higher than Plain (`0.94872` versus `0.94743`) while manual worst is lower by about 0.89 percentage point (`0.93111` versus `0.94000`). The speed gain is real on the target device and does not require worker parallelism or threaded WASM.

The same iPhone diagnostics also clarify that classifier inference is not the dominant remaining frame cost. A representative 17-candidate production frame showed base-classifier preprocessing around 161 ms versus base-classifier inference around 66 ms. The standalone Plain benchmark predicts roughly 50-60 ms inference at that batch size, so the production inference measurement is broadly consistent with the isolated benchmark. The next performance investigation should therefore focus on classifier preprocessing rather than further classifier-architecture work.

## Conclusion

INV-011 is complete.

- Dynamic batching was already functioning correctly; one classifier inference is issued for the full crop batch.
- Standard mobile-oriented topology does improve the actual iPhone ORT Web path.
- ShuffleNetV2 0.5x/1.0x are dominated by the corresponding MobileNetV3-Small candidates in the desktop comparison and require no further device work.
- MobileNetV3-Small 0.5x is the speed boundary but gives up too much manual-domain accuracy for the primary candidate.
- MobileNetV3-Small 1.0x preserves essentially the same mean accuracy as Plain and reduces iPhone batch inference by about 1.5-1.7x.
- Production promotion remains a separate explicit change, but there is no further architecture-selection question to answer in this investigation.
- The dominant next optimization target is the existing gray64 classifier preprocessing path, especially per-crop software resampling.

**Decision:** close INV-011 with MobileNetV3-Small 1.0x as the selected classifier replacement candidate. Open/follow a separate preprocessing-performance task or investigation before considering workers, shared-backbone classification, or further model-family changes.
