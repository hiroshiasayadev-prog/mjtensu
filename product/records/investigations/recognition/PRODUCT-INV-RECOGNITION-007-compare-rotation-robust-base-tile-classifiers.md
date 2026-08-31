# PRODUCT-INV-RECOGNITION-007: Compare rotation-robust base-tile classifiers

- **status**: completed
- **date**: 2026-08-30
- **trigger**: iPhone 13 timing instrumentation showed that the production grayscale C8 base-classifier inference cost scales strongly with the number of detector candidates. The accepted C8 model is highly accurate, but its exported representation carries the C8 orientation dimension through the convolutional backbone and is therefore substantially more expensive at runtime than its trainable-parameter count suggests. Before optimizing that implementation further, compare the accepted C8 approach against a conventional CNN with continuous rotation augmentation and against three published rotation-aware alternatives whose deployment graphs may have different compute characteristics.
- **scope**: On the frozen 35-class gray64 v3 classifier corpus, compare production C8, a conventional CNN, RotEqNet-style vector-field convolution, RIC-CNN rotation-invariant coordinate convolution, and SConv sorting convolution. Train the planned augmentation/no-augmentation conditions, evaluate every condition on the same deterministic dense rotation sweep, export deployment candidates to ONNX, verify PyTorch-to-ONNX parity, characterize static graph/workload properties, and benchmark ONNX Runtime CPU inference under one fixed representative batch size.
- **non_scope**: NanoDet accuracy or cadence, red-five classification, detector crop generation, relabeling or rebuilding the frozen v3 compact corpus, production model promotion, browser/iPhone timing before a desktop candidate is selected, quantization, pruning, temporal tracking/cache optimization, scoring, or UI behavior.
- **source_refs**:
  - PRODUCT-INV-RECOGNITION-005
  - PRODUCT-ADR-RECOGNITION-004
  - PRODUCT-TASK-SYSTEM-002-08
  - tools/recognition/tile_shape_classifier.py
  - tools/recognition/train_tile_shape_classifier.py
  - tools/recognition/export_c8_classifiers_onnx.py
  - .local/recognition/tile_classifier_datasets/gray35_jp500_seed42_v3_jp189.sqlite
  - .local/recognition/tile_classifier_datasets/gray35_jp500_seed42_v3_jp189.summary.json
  - .local/recognition/tile_classifier_runs/gray64_c8_rot22p5_bs512_gray35_v3_jp189_seed42/tile-c8-gray35-v3-jp189.onnx.metadata.json
  - product/frontend/src/recognition/model-runtime/production-model-set.json
- **planned_outputs**:
  - tools/recognition/rotation_classifier_experiment_models.py
  - tools/recognition/run_rotation_classifier_experiment.py
  - tools/recognition/tests/test_rotation_classifier_experiment.py
  - .local/recognition/rotation_classifier_experiment/
  - PRODUCT-INV-RECOGNITION-007
- **follow_up_candidates**:
  - Benchmark the selected Pareto-frontier candidates in ONNX Runtime Web on iPhone 13 with the actual selected execution provider.
  - If all float32 candidates remain too slow, evaluate INT8 or mixed quantization only after preserving the same dense-angle accuracy protocol.
  - If rotation-aware architectures remain expensive, evaluate temporal crop-result reuse/tracking independently from classifier architecture.
- **related_adrs**:
  - PRODUCT-ADR-RECOGNITION-004

## Question

Determine which base-tile classifier architecture gives the best deployment trade-off between:

1. arbitrary-angle 35-class accuracy on the already accepted crop domain; and
2. deployed inference cost for the multi-crop batches produced by Recognition.

The investigation must not assume that low parameter count implies low runtime cost. The production C8 model is the motivating counterexample: its source parameterization is compact, while the exported tensor graph expands regular C8 fields into ordinary orientation channels throughout the backbone.

The comparison therefore treats measured deployed latency as the decisive compute evidence and uses parameter/MAC-style metrics only to explain that latency.

## Frozen corpus and input contract

Do not rebuild the classifier dataset for this investigation. Use the existing compact v3 database directly:

```text
.local/recognition/tile_classifier_datasets/
  gray35_jp500_seed42_v3_jp189.sqlite
```

Its recorded composition is:

| split | samples |
|---|---:|
| train | 19,593 |
| jp_val | 6,800 |
| manual_val | 450 |
| total | 26,843 |

The v3 corpus was produced by copying v2 unchanged and appending 189 human-reviewed JP detector crops to `train` only:

| appended decision | samples |
|---|---:|
| invalid | 180 |
| valid tile | 9 |
| total | 189 |

The validation sets are therefore frozen across the v2 -> v3 extension and must remain untouched here.

Every compared model receives the same classifier input contract:

```text
64 x 64 grayscale uint8 letterboxed crop
  -> float32 [0,1]
  -> normalize with the frozen v3 train statistics
  -> 35 logits
```

The production-v3 normalization recorded by the selected checkpoint is:

```text
mean = 0.6815832403977466
std  = 0.2725553681973969
```

The 35 classes are the canonical 34 base tile identities plus `invalid` in the existing database order. Red-five identity remains outside this investigation.

## Existing production reference

The selected production artifact is v3, not v4:

```text
.local/recognition/tile_classifier_runs/
  gray64_c8_rot22p5_bs512_gray35_v3_jp189_seed42/
    tile-c8-gray35-v3-jp189.onnx
```

`production-model-set.json` currently binds the base classifier role to that artifact. Its recorded source checkpoint is epoch 45.

The accepted C8 architecture is:

```text
64 x 64 grayscale
  -> C8 regular fields 8 / 16 / 32 / 64
  -> GroupPooling
  -> spatial global pooling
  -> 35-class head
```

The accepted training policy uses batch 512 and random residual rotation in `[-22.5 deg, +22.5 deg]`. PRODUCT-INV-RECOGNITION-005 already showed that C8 without residual-angle augmentation has a substantial accuracy trough between discrete group orientations.

The existing v3 production checkpoint is reused as the `C8-production` reference rather than retrained merely to reproduce an artifact that is already deployed.

## Candidate methods

Compare five architecture families.

### C8

Use the existing accepted C8 implementation and field schedule. The purpose of this row is to anchor accuracy and deployed cost against the production model, not to redesign C8 in this investigation.

### Plain CNN

Use the conventional small gray64 CNN topology already represented by `PlainTileShapeClassifier` as the ordinary-convolution baseline. Continuous random rotation augmentation is allowed to carry the entire rotation-robustness burden in the augmented condition.

### RotEqNet-style vector-field CNN

Implement the rotating-convolution plus orientation-pooling mechanism described by Marcos et al., ICCV 2017. A canonical filter is evaluated at multiple orientations; orientation pooling retains the strongest response and its orientation as a vector field consumed by deeper RotEqNet layers.

Use `R = 17` sampled filter orientations for the initial comparison, matching the paper's MNIST-rot classification setting. The implementation must preserve vector-field semantics rather than approximating the method as ordinary max pooling over unrelated rotated feature maps.

External references:

- Marcos, Volpi, Komodakis, Tuia, "Rotation Equivariant Vector Field Networks", ICCV 2017, https://openaccess.thecvf.com/content_ICCV_2017/html/Marcos_Rotation_Equivariant_Vector_ICCV_2017_paper.html
- PyTorch remake referenced by the original authors: https://github.com/COGMAR/RotEqNet

### RIC-CNN

Implement Rotation-Invariant Coordinate Convolution as described by Mo and Zhao. The sampling offsets are deterministic functions of spatial position relative to the image/feature-map center rather than learned offsets. The implementation should use the paper's deformable-convolution equivalence or an exactly equivalent tensor formulation.

The investigation specifically tests whether the method remains useful on detector crops where the physical tile may be imperfectly centered, because the theoretical invariance is defined for rotations around the input center.

External references:

- Mo and Zhao, "RIC-CNN: Rotation-Invariant Coordinate Convolutional Neural Network", Pattern Recognition 146 (2024) 109994, https://doi.org/10.1016/j.patcog.2023.109994
- Author implementation: https://github.com/HanlinMo/Rotation-Invariant-Coordinate-Convolutional-Neural-Network

### SConv

Implement the published Sorting Convolution using the polar/ring sampling and ring-sorting strategy selected by the authors for arbitrary-angle rotation invariance. Do not replace it with simple square-neighborhood sorting: the paper explicitly introduces polar sampling because ordinary square-grid neighborhoods do not remain the same sample set under arbitrary rotations.

External references:

- Mo and Zhao, "Sorting Convolution Operation for Achieving Rotational Invariance", IEEE Signal Processing Letters 31 (2024) 1199-1203, https://doi.org/10.1109/LSP.2024.3381909
- Author implementation: https://github.com/HanlinMo/Sorting-Convolution-Operation-for-Achieving-Rotational-Invariance

## Experiment-only implementation boundary

Do not extend `train_tile_shape_classifier.py` into a multi-paper research harness and do not change the production classifier implementation merely to run this comparison.

Create dedicated experiment code:

```text
tools/recognition/
  rotation_classifier_experiment_models.py
  run_rotation_classifier_experiment.py
  tests/test_rotation_classifier_experiment.py
```

The experiment harness may import the accepted C8/plain definitions where exact reuse prevents accidental architecture drift, but experiment-specific training, dense-angle evaluation, ONNX export, graph inspection, and latency benchmarking belong to the new runner.

The three published alternative operators belong in the experiment model module and must not enter production runtime code unless a later decision explicitly promotes one.

## Architecture-comparison rule

Do not force every published method into one nominal backbone when that would remove method-specific structure. The comparison instead has two controlled groups plus the production reference.

C8 remains the accepted 8 / 16 / 32 / 64 regular-field production architecture. It is intentionally not resized because the question is whether an alternative beats the model that is actually deployed.

Plain, RIC-CNN, and SConv are treated as replaceable-convolution comparisons on the compact product backbone. Plain and SConv use stage widths 32 / 64 / 128 / 192 and the 5/3/3/3 spatial-kernel schedule. RIC-CNN uses the same widths/downsampling/head but uses 3x3 RIC-C at all four stages, matching the published RIC-C operator's 3x3 definition rather than inventing a 5x5 version. RIC-C and SConv are explicitly presented as compact-backbone operator comparisons, not as reproductions of the authors' ResNet/MNIST classifier topologies.

RotEqNet is different: its vector-field topology is itself part of the published method and must not be replaced by the generic compact backbone. The comparison therefore uses the public PyTorch remake's MNIST-rot feature topology:

```text
RotConv 1 -> 6,  9 x 9, padding 4, 17 orientations
VectorMaxPool 2
VectorBatchNorm 6
RotConv 6 -> 16, 9 x 9, padding 4, 17 orientations
VectorMaxPool 2
VectorBatchNorm 16
RotConv 16 -> 32, 9 x 9, padding 1, 17 orientations
Vector2Magnitude
1 x 1 Conv 32 -> 128
BatchNorm + ReLU + Dropout2d(0.7)
1 x 1 Conv 128 -> 35
```

On the author's 28 x 28 MNIST input the third 9 x 9 RotConv leaves a 1 x 1 map. The investigation keeps the product's fixed 64 x 64 input contract, which leaves a 10 x 10 class map; the adaptation adds only `AdaptiveAvgPool2d(1)` after the class map to produce 35 logits. This adaptation must be recorded with the result.

Architecture sizes are therefore not normalized to equal parameter count or equal theoretical compute. The deployed latency/accuracy Pareto comparison measures the actual cost of each defensible implementation rather than hiding method cost through architecture resizing.

## Training matrix

Run these ten accuracy conditions:

| condition | architecture | training rotation |
|---|---|---|
| `c8-noaug` | C8 | none |
| `c8-production` | C8 v3 existing checkpoint | existing random `[-22.5,+22.5]` |
| `plain-noaug` | Plain CNN | none |
| `plain-random360` | Plain CNN | continuous random full rotation |
| `roteqnet-noaug` | RotEqNet | none |
| `roteqnet-random360` | RotEqNet | continuous random full rotation |
| `riccnn-noaug` | RIC-CNN | none |
| `riccnn-random360` | RIC-CNN | continuous random full rotation |
| `sconv-noaug` | SConv | none |
| `sconv-random360` | SConv | continuous random full rotation |

The C8 augmented reference intentionally retains the already accepted residual `+/-22.5 deg` policy rather than inventing a new C8 training condition for the initial matrix. The other four families receive identical full-range random augmentation in their augmented conditions. A separate C8 `random360` ablation may be added later only if the initial results make it relevant.

## Continuous random rotation policy

For every `random360` condition, do not materialize an expanded dataset and do not enumerate a fixed angle grid during training.

Each original training sample contributes one view per epoch. Its angle is drawn from:

```text
Uniform(-180 deg, +180 deg)
```

The angle assignment must be deterministic for a given `(seed, epoch, sample_id)` so that:

- rerunning one condition produces the same augmented examples;
- different architecture families see the same angle for the same source sample and epoch;
- batch ordering or micro-batch changes do not silently change the augmentation corpus.

Use the same bilinear image-rotation and border-padding semantics for every architecture. Rotation augmentation is applied before normalization.

The no-augmentation rows receive the original stored gray64 crop only.

## Common training protocol

Unless an operator makes a setting impossible, use the accepted C8 training baseline:

| setting | value |
|---|---:|
| epochs | 50 |
| effective batch size | 512 |
| optimizer | AdamW |
| learning rate | 0.001 |
| weight decay | 0.0001 |
| scheduler | cosine annealing |
| seed | 42 |
| AMP | enabled when numerically/operator safe |
| TF32 | enabled on the RTX 3090 training path |

If a specialized architecture cannot fit batch 512 in memory, reduce only the physical micro-batch and use gradient accumulation to preserve effective batch 512. Do not quietly change the optimizer-update batch size, because PRODUCT-INV-RECOGNITION-005 already found material accuracy degradation at larger C8 batch sizes.

Training throughput and VRAM are recorded as secondary engineering metrics but do not decide deployment selection; training is offline.

## Best-checkpoint selection

Preserve the existing checkpoint-selection principle so model families are not selected with different validation objectives.

During training, run the established manual validation sweep at:

```text
0 / 15 / 30 / 45 deg
```

on full-sweep epochs and select `best.pt` by mean `manual_val` accuracy across those four angles.

The dense 360-degree evaluation below is a final comparison and must not be used differently for different architectures to cherry-pick checkpoints.

## Dense deterministic rotation evaluation

After selecting a checkpoint, evaluate every condition using the exact same frozen validation samples and deterministic angle grid.

Primary grid:

```text
0.000
5.625
11.250
16.875
...
354.375 deg
```

This is 64 equally spaced orientations over 360 degrees.

Evaluate both:

```text
manual_val: 450 base crops x 64 angles
jp_val:   6,800 base crops x 64 angles
```

Do not apply random augmentation during evaluation. Run checkpoint-selection validation and the final dense-angle comparison in float32 even when AMP is used for training, so architecture-dependent mixed-precision behavior does not contaminate the accuracy comparison against float32 ONNX deployment graphs.

For each split and angle record at least:

- sample count;
- correct count;
- accuracy;
- macro accuracy;
- per-class accuracy;
- confusion matrix.

For each model condition summarize:

- mean accuracy across the 64 angles;
- worst-angle accuracy and its angle;
- best-angle accuracy;
- standard deviation across angles;
- 0-degree accuracy;
- total errors across the deterministic sweep.

If two leading candidates remain indistinguishable on the primary grid, add a second grid shifted by half a step:

```text
2.8125 + n * 5.625 deg
```

for another 64 orientations. Do not run this extra sweep merely to increase result volume when the first comparison is already decisive.

## ONNX deployment gate

Accuracy alone does not make a candidate relevant to the production PWA. Every architecture that remains in the comparison must have a deployment-form ONNX graph.

For each architecture:

1. export a dynamic-batch `[N,1,64,64] -> [N,35]` ONNX model;
2. run `onnx.checker`;
3. compare PyTorch and ONNX Runtime CPU logits on a fixed deterministic 16-sample parity batch;
4. require zero argmax mismatches and a documented numerical tolerance;
5. record the opset and any operator/runtime constraints.

Prefer standard ONNX operators. If a faithful architecture cannot be represented by the intended ONNX/ONNX Runtime path without a custom runtime extension, record that as a deployment failure rather than timing a materially different substitute implementation.

In particular, investigate and record the actual exported form of:

- RotEqNet rotated-filter/orientation-pooling operations;
- RIC-CNN coordinate sampling / `DeformConv`-equivalent path;
- SConv polar sampling and sorting path.

The experiment implementation intentionally targets the repository's existing PyTorch 1.13 / ONNX opset-16 environment. RIC-C is therefore exported as its fixed coordinate-sampling equivalent (`GridSample` for the nine center-radial samples followed by a packed 1x1 weighted sum) rather than requiring a newer native ONNX `DeformConv` exporter. SConv is expressed with polar `GridSample`, ring-wise full `TopK` sorting, and the packed weighted sum. These are deployment formulations of the published operators, not claims that they are the fastest possible native implementations.

Do not infer browser suitability merely because `torch.onnx.export` creates a file.

## Static graph and workload characterization

Parameter count alone is insufficient. For each exported architecture record:

- trainable parameter count;
- ONNX file bytes;
- ONNX operator histogram;
- inferred peak intermediate-tensor bytes at batch 16 where shape inference permits it;
- ordinary Conv/Gemm MACs where meaningful.

For specialized operators, also record method-specific work counters rather than forcing every operation into a misleading single MAC number:

- RotEqNet: number of sampled filter orientations and resulting convolution work;
- RIC-CNN: deformable/bilinear sample count and associated convolution work;
- SConv: sampled values per location/ring and sort/TopK workload.

The final report may include a rough total-operation estimate, but it must distinguish estimates from exact graph counts.

## CPU inference benchmark

Benchmark the exported ONNX graph, not PyTorch eager mode.

Use ONNX Runtime `CPUExecutionProvider` with:

```text
intra_op_num_threads = 1
inter_op_num_threads = 1
execution_mode = sequential
```

Use one representative fixed benchmark batch:

```text
batch = 16
shape = [16,1,64,64]
```

Candidate-count scaling has already been demonstrated on the production classifier, so this investigation does not repeat a broad batch-size sweep. The fixed batch exists to compare architecture cost under a common multi-crop workload.

Use the same deterministic input batch for every architecture and run, at minimum:

```text
warm-up      = 100 runs
measurement  = 1000 runs
```

Record:

- mean ms/batch;
- median ms/batch;
- p95 ms/batch;
- ms/image.

Record CPU identity, ONNX Runtime version, opset, and thread settings alongside the timing. Desktop CPU timing is a screening/relative metric, not an iPhone performance claim.

## Result tables

The accuracy result table is condition-specific because training augmentation can change weights:

| architecture | train rotation | manual mean | manual worst | JP mean | JP worst |
|---|---|---:|---:|---:|---:|
| C8 | none | | | | |
| C8 | production +/-22.5 | | | | |
| Plain | none | | | | |
| Plain | random360 | | | | |
| RotEqNet | none | | | | |
| RotEqNet | random360 | | | | |
| RIC-CNN | none | | | | |
| RIC-CNN | random360 | | | | |
| SConv | none | | | | |
| SConv | random360 | | | | |

The deployment-cost table is architecture-specific. No-augmentation and random360 variants of one architecture have the same graph topology, so do not duplicate identical structural rows unless trained weights materially change serialized size:

| architecture | params | ONNX bytes | key operators/work | peak activation b16 | ORT CPU median b16 | p95 b16 |
|---|---:|---:|---|---:|---:|---:|
| C8 | | | | | | |
| Plain | | | | | | |
| RotEqNet | | | | | | |
| RIC-CNN | | | | | | |
| SConv | | | | | | |

Also produce an angle-versus-accuracy curve for each accuracy condition. A single mean number must not hide periodic troughs like those already observed for C8 without residual augmentation.

## Interpretation rule

Do not choose a winner from paper claims, parameter count, theoretical invariance, or desktop latency alone.

The investigation judgment should identify the empirical Pareto frontier across:

- arbitrary-angle validation robustness;
- ONNX deployability/parity;
- single-thread ONNX Runtime CPU batch-16 latency;
- graph/memory complexity.

A model that is fastest but has a material angle-specific accuracy trough is not automatically preferable. A model that is mathematically rotation-invariant but requires unsupported/custom deployment operators is not a production candidate. A plain CNN is not privileged: it is one of the five architecture families and wins only if the measured evidence supports it.

The final promotion decision remains separate. Leading candidates must subsequently be measured in ONNX Runtime Web on iPhone 13 because desktop CPU timing establishes only a useful relative screening signal.

## Implementation fidelity gates

Before accepting comparative results from the three new architecture families, record evidence that the implemented operator matches the published method rather than only its name.

### RotEqNet

Verify on fixed tensors that:

- canonical filters are evaluated at all configured orientations;
- orientation pooling selects the maximum response orientation;
- the selected response is represented as the expected vector field;
- deeper rotating convolutions transform both vector components consistently with the sampled filter angle.

Use the authors' original/PyTorch implementation as a behavioral reference where practical.

### RIC-CNN

Verify that:

- sampling coordinates depend on spatial position relative to feature-map center;
- offsets are fixed/non-learned;
- the convolution samples the rotation-invariant coordinate neighborhood described by the paper;
- a controlled center-rotation test exhibits the expected invariance up to interpolation/numerical error.

### SConv

Verify that:

- arbitrary-angle sampling uses the intended polar/ring layout;
- sorting occurs within the intended rings;
- sorted samples are mapped to convolution weights in deterministic ring order;
- a controlled rotated input yields the expected near-invariant local response up to interpolation/numerical error.

Unit tests for these properties belong in `test_rotation_classifier_experiment.py` and are required before expensive training runs begin.

## Execution record

### Experiment harness implementation: 2026-08-30

Dedicated experiment files were added without changing the accepted production trainer/runtime:

```text
tools/recognition/rotation_classifier_experiment_models.py
tools/recognition/run_rotation_classifier_experiment.py
tools/recognition/tests/test_rotation_classifier_experiment.py
```

The runner is designed for an unattended single invocation. It executes the requested condition matrix sequentially, records a failed condition and continues to the next one, writes an aggregate summary after every condition, and skips already completed conditions when the same command is resumed. Training starts at physical batch 512 and automatically halves the physical micro-batch on CUDA OOM while preserving effective optimizer batch 512 by gradient accumulation.

Implementation-specific operator forms are:

- RotEqNet: 17 bilinearly rotated canonical filter banks, orientation maximum selection, and two-component vector-field propagation; deeper layers rotate/mix both learned vector-filter components.
- RIC-CNN: fixed center-radial 3x3 sampling coordinates at each feature resolution, implemented as nine bilinear `GridSample` operations plus the learned 3x3 weight tensor reshaped to the equivalent packed 1x1 weighted sum.
- SConv: kernel-dependent polar rings (`8*r` samples at radius `r`), bilinear `GridSample`, full ring-wise `TopK` ordering, and the learned kernel tensor reshaped to the equivalent packed 1x1 weighted sum.

### First overnight execution finding: 2026-08-30

The first unattended execution is **not accepted as investigation evidence**. Multiple harness/fidelity defects were found and must be corrected before comparative results are used.

1. `c8-noaug` completed its 50 training epochs but failed immediately afterward while reconstructing `best.pt`. The runner built the new C8 instance on CPU, loaded the checkpoint, called `model.eval()`, and only then called `.cuda()` in the caller. `escnn` performs R2Conv basis expansion when entering eval mode; because the same process had already trained a CUDA C8 model, basis-expansion state could be CUDA-resident while the newly loaded weights were still CPU-resident. The resulting `einsum` mixed `cuda:0` and `cpu`. This is a runner checkpoint-device-order bug, not a C8 model/training failure. The loader is changed to move the complete model to its requested final device **before** entering eval mode, and a CUDA C8 checkpoint round-trip regression test is required.
2. The initial RotEqNet experiment implementation preserved the rotating-convolution/vector-field idea but changed the method-specific network topology to the generic compact 5/3/3/3 four-stage comparison backbone. That result is not a faithful RotEqNet comparison and all results from that implementation are excluded. The replacement restores the public PyTorch topology (6/16/32 vector channels, three 9x9 RotConv layers, pool-before-vector-BN order, vector-BN momentum/initialization, and the 1x1/Dropout head) with only the explicit 64x64 class-map average described above. A discrete quarter-turn end-to-end invariance preflight is required before retraining it.
3. The initial SConv implementation correctly used polar `8*r` sampling and ring-wise sorting, but concatenated the values as `[center, ring1, ring2, ...]` and directly flattened the conventional row-major kernel weights. The paper instead requires each sorted ring to be arranged back into the corresponding square-grid ring in row-major order before applying `W(P)`. The initial implementation therefore paired sorted values with the wrong kernel positions. A deterministic ring-to-row-major gather is now explicit; all SConv results from the first implementation are excluded.
4. RIC-C's center-radial geometry was consistent with the author implementation on code review, including the 3x3 row-major offset order. The fixed-grid implementation now also reproduces the author's four-decimal angle rounding. Before accepting RIC-C results, a numerical preflight compares the experiment's nine-`GridSample` formulation directly against `torchvision.ops.deform_conv2d` constructed from the author's fixed offsets using identical input and weights.

The C8 CUDA checkpoint-device regression has since passed on the Linux training server. The runner also preserves failed run directories so that a fully completed C8/Plain training phase can be recovered after a downstream evaluation/export failure rather than deleting 50 epochs of valid work. Recovery is intentionally restricted to the unchanged accepted C8/Plain implementations; old RotEqNet/RIC-CNN/SConv artifacts are not reused. An implementation-version gate also prevents completed results from the invalid research-model implementation from being silently skipped on resume.

No second overnight matrix should begin until the complete fidelity/preflight suite, including RIC numerical equivalence, RotEqNet quarter-turn invariance, SConv row-major mapping/invariance, C8 checkpoint reload, and ONNX/ORT smoke export, passes on the training server.

The frozen v3 compact database remains sufficient; no corpus rebuild is required.

### Accepted fidelity-v2 execution: 2026-08-31

After the fidelity/device/export fixes and preflight suite passed, the complete ten-condition matrix was rerun under implementation version `inv007-fidelity-v2`. All ten conditions completed and `deployment_failure_count` was zero.

Dense rotation accuracy:

| condition | manual mean | manual worst | JP mean | JP worst |
|---|---:|---:|---:|---:|
| `c8-noaug` | 0.86188 | 0.73556 | 0.95446 | 0.83147 |
| `c8-production` | **0.97222** | 0.96000 | **0.99984** | 0.99941 |
| `plain-noaug` | 0.42205 | 0.17333 | 0.51361 | 0.14559 |
| `plain-random360` | 0.91118 | 0.88889 | 0.99898 | 0.99162 |
| `roteqnet-noaug` | 0.71260 | 0.34222 | 0.84162 | 0.47632 |
| `roteqnet-random360` | 0.97170 | 0.96222 | 0.99959 | 0.99926 |
| `riccnn-noaug` | 0.93500 | 0.89778 | 0.98544 | 0.94265 |
| `riccnn-random360` | 0.97111 | **0.96444** | 0.99983 | **0.99971** |
| `sconv-noaug` | 0.89514 | 0.86000 | 0.98592 | 0.95574 |
| `sconv-random360` | 0.95722 | 0.94667 | 0.99949 | 0.99926 |

Deployment cost on the fixed single-thread ORT CPU batch-16 benchmark:

| architecture | representative ONNX bytes | ORT CPU median b16 | p95 b16 |
|---|---:|---:|---:|
| C8 | 6,261,185 | 52.68 ms | 53.12 ms |
| Plain | 1,495,802 | **17.78 ms** | **17.89 ms** |
| RotEqNet | 640,454 | 195.57 ms | 196.84 ms |
| RIC-CNN | 2,013,089 | 149.56 ms | 150.36 ms |
| SConv | 2,536,002 | 300.18 ms | 301.47 ms |

The ordinary Conv/Gemm MAC estimator does not represent the specialized operators fairly: RotEqNet reports zero known MACs and RIC-CNN/SConv report only the ordinary packed convolution/Gemm portion while omitting `GridSample`, orientation/filter expansion, `TopK`, gather, and related work. The measured ORT latency and operator histogram are therefore the decisive deployment-cost evidence for those methods.

## Conclusion

The empirical Pareto frontier from INV-007 contains **production C8** and **Plain + random360**.

Production C8 remains the best accuracy point in the tested matrix: 0.97222 dense manual mean accuracy with 0.96000 worst-angle accuracy. RotEqNet-random360 and RIC-CNN-random360 reach essentially the same accuracy range, but their deployment graphs are approximately 3.7x and 2.8x slower than C8 respectively in the fixed ORT CPU benchmark. SConv-random360 is both less accurate and substantially slower than C8. None of the three research rotation-aware alternatives therefore improves the production accuracy/runtime trade-off.

Plain-random360 is the only alternative that changes the trade-off materially. At the common 50-epoch INV-007 budget it reaches 0.91118 manual mean / 0.88889 worst while running at 17.78 ms per batch 16 versus 52.68 ms for C8, approximately **2.96x faster** in the desktop ORT screening benchmark. Its JP validation accuracy is already effectively saturated. The remaining weakness is the manual real-crop rotation domain, not basic JP classification capability.

The no-augmentation rows also confirm that architecture-level rotation handling is real but not sufficient to determine deployment value. RIC-CNN and SConv retain strong arbitrary-angle accuracy without random360 augmentation, while Plain collapses without it; nevertheless their deployment cost prevents them from displacing C8. Continuous random360 augmentation is therefore the most promising route for the ordinary compact CNN because it obtains strong robustness while preserving the simplest and fastest deployment graph.

**Decision:** do not promote RotEqNet, RIC-CNN, or SConv. Keep production C8 as the accepted production classifier after INV-007. Carry Plain-random360 forward as the sole speed-oriented follow-up candidate and determine whether its 50-epoch accuracy gap is training-horizon limited before judging the architecture ceiling. That follow-up is PRODUCT-INV-RECOGNITION-008. Direct iPhone timing remains a separate promotion gate; desktop ORT CPU timing is only comparative screening evidence.
