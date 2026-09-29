# PRODUCT-INV-RECOGNITION-014: Evaluate ShuffleNet spatial-retention tile classifiers

- **status**: in_progress
- **date**: 2026-09-17
- **area**: recognition
- **depends_on**:
  - PRODUCT-INV-RECOGNITION-013
- **related_tasks**:
  - PRODUCT-TASK-SYSTEM-002-16
- **source_refs**:
  - `mldb_data/tile-classifier/studies/shufflenet-spatial-screen-v1.yaml`
  - `mldb_data/tile-classifier/studies/shufflenet-spatial-angle-robustness-v2.yaml`
  - `product/frontend/src/recognition/model-runtime/production-model-set.json`
  - PRODUCT-INV-RECOGNITION-012
  - PRODUCT-INV-RECOGNITION-013

## Trigger

INV-013 showed that the perspective-aware `mix-heavy` training distribution materially improves robustness, but the MobileNet-derived `f8-r1` family still retains a measurable fine-grained manzu failure surface. Under the same augmentation family, Plain eliminated the measured `6m -> 5m/7m` confusion while `f8-r1` retained a worst rate of `0.1666666667`.

This re-opened the architecture question. A controlled ShuffleNetV2 0.5x screen was therefore run to test whether preserving more spatial resolution before the standard ShuffleNet stages improves tile discrimination without redesigning the block family.

The screen is useful accuracy evidence, but it did **not** yet include deployment-cost evaluation against the classifier currently shipped by mjtensu. For this architecture screen, reproducible desktop single-thread ONNX Runtime latency is the required performance gate. iPhone 13 and full production-pipeline timing are deferred promotion checks rather than INV-014 closure requirements.
## Question

Can a ShuffleNetV2 0.5x classifier with less aggressive stem downsampling remove the residual fine-grained manzu confusion while retaining a useful latency advantage over heavier accuracy-oriented classifiers and the current dev `mobile-tile-f8-r1` reference?

## Controlled architecture screen

Keep the ShuffleNetV2 0.5x stage/block topology fixed and vary only two stem decisions:

| condition | conv1 stride | maxpool | 64x64 size entering stage2 |
|---|---:|---|---:|
| `s2-pool` | 2 | 3x3 stride 2 | 16x16 |
| `s2-nopool` | 2 | removed | 32x32 |
| `s1-pool` | 1 | 3x3 stride 2 | 32x32 |
| `s1-nopool` | 1 | removed | 64x64 |

All four candidates use the same ShuffleNetV2 x0.5 stage2/stage3/stage4 topology, class count, gray64 contract, corpus, 150-epoch training duration, seed 42, and INV-013 `mix-heavy` augmentation recipe. The experiment therefore isolates early spatial downsampling rather than changing the ShuffleNet block family.

## Baselines

Two baselines are required and serve different purposes:

1. **Actual production**: `tile-c8-gray35-v3-jp189.onnx`, SHA-256 `b8a8fa3ff6c6d1e944a7593fa0afc947e0cd2513fb79ca46e5f8fcd6e19c97d0`, runtime spec `c8-tile-35-v1`, bound by `/persist/srv-bugrat/mjtensu-product/compose.prod.yaml` deployment assets.
2. **Current dev reference**: `mobile-tile-f8-r1.onnx`, SHA-256 `5039c044a490b44e8c645ead5a3280293f78c3c43db9baabd9f07219ff883a7e`, currently bound by the `mjtensu-dev` model set.
3. **Plain mix-heavy** from INV-013. This remains the accuracy reference because it demonstrated `0.0` worst measured `6m -> 5m/7m` confusion under the perspective-aware training distribution.

A fair architecture comparison should evaluate `f8-r1` under the same current `mix-heavy` protocol where practical. The actual production C8 remains the deployment replacement baseline; dev f8-r1 timing is supporting architecture evidence, not production state.
## Measured screen evidence

Canonical MLDB result `tile-classifier/run-70f6ddae5a5446478dbd0e15a9523d52` completed all four training trials plus angle-robustness and manzu-diagnostic evaluation.

| condition | manual angle mean | front accuracy | mean condition accuracy | worst condition accuracy | worst 6m-neighbor confusion |
|---|---:|---:|---:|---:|---:|
| `s2-pool` | `0.9494444` | `0.9574468` | `0.8893617` | `0.7234042` | `0.5000000` |
| `s2-nopool` | `0.9783333` | `0.9787234` | `0.9645390` | `0.9148936` | `0.1666667` |
| `s1-pool` | `0.9783333` | `1.0000000` | `0.9787234` | `0.9361702` | `0.0000000` |
| `s1-nopool` | `0.9822222` | `1.0000000` | `0.9929078` | `0.9574468` | `0.0000000` |

The result is directionally strong: removing one early downsampling step improves robustness, and removing both tested stem downsampling operations produces the strongest measured classifier-domain result. The evidence supports the narrower statement that early spatial compression is harmful for this 64x64 tile task under the tested ShuffleNet topology.

All four ShuffleNet candidates have the same parameter count: `377,235`. The dev `mobile-tile-f8-r1` architecture has `965,251` parameters. Parameter count alone is not a latency result; operator mix and retained activation resolution can dominate WASM/mobile cost.

## Existing dev performance reference

INV-012 recorded `mobile-tile-f8-r1` on iPhone 13 / ONNX Runtime Web / `wasm-simd`, one thread, as approximately:

- `N=16`: `39.84 ms` median
- `N=24`: `58.18 ms` median
- ONNX bytes: `3,873,724`

These values are a dev-runtime reference, not the current production deployment and not measurements of the ShuffleNet candidates.
## Missing performance work

The current MLDB screen did **not** measure or record:

- ONNX export bytes for the four ShuffleNet candidates;
- ONNX parity against the PyTorch checkpoints;
- operator/MAC/activation-cost breakdown;
- desktop single-thread ORT latency at production-relevant dynamic batches;
- controlled accuracy/robustness evaluation of the current `f8-r1` architecture under the same `mix-heavy` protocol.

These are INV-014 closure requirements. iPhone 13 ONNX Runtime Web latency and end-to-end production Recognition timing are deferred until a candidate survives the desktop performance/accuracy screen. In particular, `s1-nopool` retains much larger intermediate feature maps than standard ShuffleNet, so its low parameter count does not imply low runtime cost.

## Closure plan

1. Export at least `s1-pool` and `s1-nopool` to dynamic-batch ONNX `[N,1,64,64] -> [N,35]`; export the other two if needed for a complete Pareto view.
2. Verify numerical parity and record model bytes, parameter count, operator counts, estimated MACs, and peak/intermediate activation sizes.
3. Add a dedicated MLDB deployment-performance Evaluation Protocol so these measurements are canonical model-evaluation results rather than ad-hoc local notes.
4. In that evaluation, benchmark desktop ORT single-thread at the same representative batch sizes used in INV-011/012 and evaluate the dev `mobile-tile-f8-r1` and actual production C8 with the identical benchmark implementation/environment.
5. Compare accuracy/robustness and desktop deployment cost to select or reject a ShuffleNet candidate for further promotion testing.
6. Defer iPhone 13 and full mjtensu Recognition-pipeline timing until a candidate passes INV-014; those checks are required before production model-set promotion, not before this investigation can close.

## Decision criteria

A ShuffleNet candidate may advance beyond INV-014 only if it simultaneously:

- preserves the measured `0.0` six-neighbor confusion surface seen in the best screen conditions or otherwise materially improves on the current f8-r1 failure surface;
- does not materially regress angle/perspective robustness versus the strongest controlled references;
- preserves the existing gray64 35-class runtime contract;
- has competitive desktop single-thread ORT latency versus the same-run `f8-r1` reference and actual production C8 baseline.

Advancing beyond INV-014 is not production promotion. iPhone 13 and live full-pipeline verification remain later promotion gates because ORT CPU timing is expected to be directionally useful but does not guarantee identical WASM/mobile operator ranking.

Until these gates are completed, the current conclusion is architectural only: reducing early ShuffleNet downsampling improves offline fine-grained tile robustness, but **no production speed/accuracy tradeoff has yet been established**.