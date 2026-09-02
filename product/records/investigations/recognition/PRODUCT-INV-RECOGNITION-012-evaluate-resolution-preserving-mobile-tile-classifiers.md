# PRODUCT-INV-RECOGNITION-012: Evaluate resolution-preserving mobile tile classifiers

- **status**: in_progress
- **date**: 2026-09-02
- **area**: recognition
- **depends_on**:
  - PRODUCT-INV-RECOGNITION-011
- **related_tasks**:
  - PRODUCT-TASK-SYSTEM-002-16

## Trigger

PRODUCT-INV-RECOGNITION-011 selected MobileNetV3-Small 1.0x as a mobile-oriented replacement candidate for the Plain random360 e150 base tile classifier. The direct iPhone 13 ONNX Runtime Web benchmark confirmed a material inference-latency advantage, and the promoted model-set subsequently measured about `35 ms` base-classifier inference for roughly 17 candidates.

However, live iPhone recognition exposed a practical accuracy regression that was not visible in the existing dense-angle classifier benchmark. A representative hand containing visually ordinary manzu tiles produced multiple within-suit identity errors such as `2m -> 7m` and `6m -> 7m`, while neighboring tiles remained correctly classified. The failure pattern is consistent with loss of fine stroke-level information rather than a global label-ordering error.

The promoted MobileNetV3-Small topology was adapted from the standard architecture without changing its spatial downsampling schedule. For the fixed `64 x 64` classifier input, that schedule reduces the feature map approximately as:

```text
64 -> 32 -> 16 -> 8 -> 4 -> 2 -> global average pool
```

By contrast, the current Plain reference preserves spatial resolution to approximately `8 x 8` before global pooling:

```text
64 -> 32 -> 16 -> 8 -> global average pool
```

This creates a concrete hypothesis: the standard MobileNetV3-Small downsampling schedule is too aggressive for a `64 x 64` fine-grained tile-shape task, even though its depthwise/pointwise operator family remains attractive for mobile latency.

## Question

Can a MobileNet-style classifier retain most of the iPhone inference advantage while recovering Plain-level live robustness by preserving a larger final feature map and spending compute on repeated same-resolution blocks instead of additional spatial downsampling?

## Hypothesis

For this task, preserving `8 x 8` or at least `4 x 4` feature maps through the late backbone will improve fine-grained manzu/pinzu/souzu discrimination and robustness to detector-crop perturbations relative to the promoted standard MobileNetV3-Small `2 x 2` endpoint.

Repeated blocks at one resolution are expected to be useful rather than redundant because each block has independent learned weights and can increase receptive field / feature abstraction without discarding additional spatial samples. Depthwise-separable inverted residual blocks should allow this extra depth at substantially lower MAC cost than the Plain reference.

## Baselines

Keep both existing models as explicit baselines:

1. **Plain random360 e150** — current accuracy reference and rollback-safe production candidate.
2. **MobileNetV3-Small 1.0x standard** — INV-011 speed reference and live-regression reference.

Do not compare only among new mobile variants; the investigation succeeds only if a candidate is evaluated against both accuracy and target-device latency baselines.

## Candidate matrix

Use a small controlled search rather than a broad NAS sweep. Keep input, labels, normalization, augmentation, optimizer family, training duration, dataset, and export contract identical to INV-011 wherever possible.

Primary factors:

- **late spatial endpoint**: `8 x 8` or `4 x 4` before global pooling;
- **same-resolution repeat count**: `1`, `2`, or `3` late blocks at the preserved resolution.

Initial six conditions:

| condition | spatial schedule concept | late repeats | intent |
|---|---|---:|---|
| `mobile-tile-f8-r1` | stop downsampling at `8 x 8` | 1 | cheapest 8x8-preserving variant |
| `mobile-tile-f8-r2` | stop downsampling at `8 x 8` | 2 | moderate same-resolution depth |
| `mobile-tile-f8-r3` | stop downsampling at `8 x 8` | 3 | strongest 8x8 late representation |
| `mobile-tile-f4-r1` | stop downsampling at `4 x 4` | 1 | cheap compromise |
| `mobile-tile-f4-r2` | stop downsampling at `4 x 4` | 2 | moderate 4x4 depth |
| `mobile-tile-f4-r3` | stop downsampling at `4 x 4` | 3 | deeper 4x4 compromise |

The implementation may reuse MobileNetV3-Small inverted-residual, SE, Hardswish, and depthwise/pointwise operators, but must make the spatial schedule explicit rather than implicitly inheriting the ImageNet-oriented stride pattern.

## Controlled variables

Unless a candidate cannot train/export under the existing contract, preserve:

- source dataset: `gray35_jp500_seed42_v3_jp189.sqlite`;
- input: grayscale `64 x 64`;
- output: 35 logits in the existing class-label order;
- normalization: current gray64 mean/std;
- augmentation: `random360`;
- effective batch and optimizer setup from INV-011;
- nominal training duration: 150 epochs;
- checkpoint selection/evaluation angles from INV-011;
- ONNX opset 16 and dynamic batch `[N,1,64,64] -> [N,35]`;
- iPhone benchmark provider: production-equivalent `wasm-simd`, one thread.

Do not change crop preprocessing, class weighting, hard-negative composition, or detector behavior inside this investigation. Those would confound the architectural question.

## Evaluation

### 1. Existing classifier-domain accuracy

Repeat INV-011-compatible dense-angle evaluation for:

- manual validation mean/worst accuracy;
- JP validation mean/worst accuracy;
- per-angle results sufficient to detect a narrow angular failure mode.

These metrics are necessary for continuity but are not sufficient for promotion.

### 2. Fine-grained confusion analysis

Record per-class confusion, especially within each suit. Explicitly inspect errors among visually similar numbered tiles, including the observed manzu failure surface.

At minimum report:

- `2m`, `6m`, `7m` confusion counts/rates;
- worst within-suit confusion pairs;
- invalid/background confusion where relevant.

### 3. Detector-crop robustness

Add a bounded evaluation derived from real detector crops or deterministic perturbations of reviewed real crops so the investigation covers the distribution gap missed by INV-011.

Perturbations should focus on realistic runtime variation rather than arbitrary image corruption:

- small x/y crop translation;
- small scale / bbox expansion-contraction;
- slight blur or resampling variation if supported by recorded live evidence;
- existing realistic rotation range / dense-angle sweep.

Keep this evaluation identical across Plain, standard MobileNetV3-Small, and all new candidates.

### 4. Deployment cost

For every successful candidate record:

- ONNX bytes;
- parameter count;
- estimated MACs per sample;
- depthwise / pointwise / ordinary convolution counts;
- final pre-pool feature-map resolution and channel count;
- desktop ORT CPU batch benchmark for continuity.

### 5. iPhone 13 benchmark

Benchmark only candidates that survive the accuracy/robustness gates. Use the same browser harness/runtime configuration as INV-011 and report at least representative batches around the production range, including `N=16` and `N=24`.

Then validate the best candidate through the actual production pipeline on live iPhone frames, not only isolated random tensors.

## Decision criteria

A replacement candidate must satisfy all of the following:

1. No material regression versus Plain on dense-angle manual-domain accuracy.
2. No obvious live-like fine-grained confusion regression of the kind observed with standard MobileNetV3-Small.
3. Detector-crop robustness materially closer to Plain than to the regressed standard MobileNet baseline.
4. A meaningful iPhone inference advantage over Plain remains after preserving spatial resolution.
5. Existing production preprocessing/output contracts remain unchanged.

Do not promote a candidate merely because it wins desktop ORT or isolated iPhone random-tensor latency.

A candidate around `40-50 ms` at production-sized iPhone batches may still be preferable to a `~35 ms` candidate if it restores Plain-like recognition quality; exact acceptance is based on measured end-to-end tradeoff rather than a hardcoded latency threshold.

## Expected interpretation

Possible outcomes:

- **8x8-preserving candidate wins**: supports the hypothesis that the standard `1/32` spatial reduction was the main architectural mismatch.
- **4x4 candidate wins**: suggests moderate spatial preservation is sufficient and gives a better speed/accuracy balance.
- **all mobile variants remain weak on live-like crops**: reject this MobileNet family for the production base classifier and retain Plain while pursuing other operator/backbone designs.
- **new candidates match Plain but lose nearly all mobile speed advantage**: retain Plain because the architectural complexity no longer buys a useful deployment benefit.

## Deliverables

- implementation of the controlled resolution-preserving MobileNet-style candidate family;
- reproducible six-condition training/evaluation runner;
- dense-angle and per-class confusion reports;
- live-like detector-crop robustness benchmark;
- ONNX deployment/parity and graph-cost report;
- iPhone isolated and production-pipeline timing for finalists;
- explicit promotion / rejection decision with evidence.

## Implementation: 2026-09-02

The investigation implementation is now scaffolded as separate INV-012 tooling so the completed INV-011 experiment remains reproducible and unchanged:

- `tools/recognition/resolution_preserving_mobile_models.py`
- `tools/recognition/run_resolution_preserving_mobile_experiment.py`
- `tools/recognition/tests/test_resolution_preserving_mobile_experiment.py`

The candidate topology preserves the standard MobileNetV3-Small 16 -> 24 -> 40 -> 48 -> 96 channel progression and operator family, but makes spatial downsampling explicit. The `f8` family changes the standard 24 -> 40 transition to stride 1 and never downsamples below 8x8. The `f4` family keeps that transition at stride 2 and never downsamples below 4x4. In both families the later 48 -> 96 transition is stride 1. The `r1/r2/r3` factor controls how many independent same-resolution 96-channel terminal blocks are retained, so the experiment varies late spatial resolution and late depth without changing width, input/output contract, normalization, or training corpus.

The runner reuses INV-011/INV-007 training, dense-angle evaluation, ONNX parity, graph-cost, and CPU benchmark helpers. It additionally records zero-degree class confusion and a deterministic manual-validation robustness proxy consisting of identity, +/-2 px x/y translations, +/-6% content scale, and a light 3x3 blur. The proxy is explicitly recorded as an approximation applied after the cached classifier-crop boundary; it is not treated as a substitute for a reviewed real detector-crop holdout or final live iPhone acceptance.

Plain e150 and standard MobileNetV3-Small 1.0x are evaluated as explicit baselines when their local checkpoints are available, using the same robustness proxy. Their existing dense-angle evidence remains the reference rather than being silently recomputed into a different benchmark definition.

## Measured finalist results: 2026-09-03

The completed f8 finalists show that preserving late spatial resolution materially improves the classifier-domain and crop-perturbation metrics relative to the standard MobileNetV3-Small baseline.

| condition | manual mean | manual worst | crop robustness mean | crop robustness worst | iPhone N=16 median | iPhone N=24 median |
|---|---:|---:|---:|---:|---:|---:|
| Plain e150 | `0.94743` | `0.94000` | reference | reference | `50.56 ms` | `77.52 ms` |
| standard MobileNetV3-Small 1.0x | `0.9487153` | `0.9311111` | live-regressed | live-regressed | `30.40 ms` | `44.02 ms` |
| `mobile-tile-f8-r1` | `0.9715625` | `0.9533333` | `0.9666667` | `0.9600000` | `39.84 ms` | `58.18 ms` |
| `mobile-tile-f8-r2` | `0.9740625` | `0.9666667` | `0.9675000` | `0.9511111` | `62.34 ms` | `91.54 ms` |

For f8-r1 the worst deterministic crop-perturbation condition was `shift-x-plus-2px`; for f8-r2 it was `shift-y-plus-2px`. The vendored f8-r1 ONNX artifact is `3,873,724` bytes and the user-verified SHA-256 is `5039c044a490b44e8c645ead5a3280293f78c3c43db9baabd9f07219ff883a7e`.

The isolated iPhone benchmark used production-equivalent WASM-SIMD with `numThreads=1`, `wasmProxy=false`, 10 warmup runs, and 50 measurement runs. At production-sized batches f8-r1 is slower than the standard MobileNet speed reference but remains clearly faster than Plain, while f8-r2 loses that deployment advantage and is slower than Plain.

## Provisional decision

Select `mobile-tile-f8-r1` as the production finalist and reject f8-r2 for promotion. f8-r1 gives the better measured deployment tradeoff: substantially stronger dense-angle/crop robustness than the standard MobileNet candidate while retaining a meaningful iPhone latency advantage over Plain. The standard MobileNet remains rejected because live iPhone recognition exposed fine-grained manzu identity errors despite its superior isolated latency.

The investigation remains `in_progress` until f8-r1 is exercised through the actual production Recognition pipeline on the live crop distribution that exposed the standard-MobileNet regression. A successful isolated benchmark is not sufficient to close INV-012.

## Non-goals

- changing the detector;
- changing classifier preprocessing semantics;
- retraining on a materially different corpus before the architecture comparison is complete;
- broad neural architecture search;
- replacing the separate red-five specialist;
- optimizing WebGPU or threaded WASM in this investigation.
