# PRODUCT-INV-RECOGNITION-008: Evaluate longer training for the Plain random360 tile classifier

- **status**: completed
- **date**: 2026-08-31
- **trigger**: PRODUCT-INV-RECOGNITION-007 found that the ordinary `PlainTileShapeClassifier` trained with deterministic continuous `random360` augmentation is the only non-C8 architecture on the deployment Pareto frontier. At 50 epochs it is approximately 3x faster than production C8 in the fixed ORT CPU batch-16 benchmark, but trails C8 by about 6 percentage points on dense `manual_val` rotation accuracy. Its checkpoint-selection validation continued to improve materially during the final ten epochs, so the 50-epoch budget may be truncating useful learning rather than exposing a hard architecture limit.
- **scope**: Hold the Plain architecture, frozen v3 corpus, random360 augmentation semantics, optimizer, effective batch, seed, validation protocol, ONNX export, and deployment benchmark constant. Compare the accepted 50-epoch result against longer cosine schedules, starting with 100 epochs and conditionally extending to 150 epochs only if the 100-epoch run shows meaningful continued improvement.
- **non_scope**: C8 retraining, other rotation-aware architectures, detector changes, dataset rebuild/relabeling, new augmentation families, classifier width/depth changes, quantization, pruning, red-five classification, production promotion, or iPhone runtime measurement before the longer-training result is known.
- **source_refs**:
  - PRODUCT-INV-RECOGNITION-007
  - PRODUCT-INV-RECOGNITION-005
  - tools/recognition/tile_shape_classifier.py
  - tools/recognition/run_rotation_classifier_experiment.py
  - .local/recognition/tile_classifier_datasets/gray35_jp500_seed42_v3_jp189.sqlite
  - .local/recognition/rotation_classifier_experiment/plain-random360/
- **planned_outputs**:
  - .local/recognition/plain_random360_epoch_sweep/e100/
  - .local/recognition/plain_random360_epoch_sweep/e150/ if the staged continuation gate is met
  - PRODUCT-INV-RECOGNITION-008
- **related_adrs**:
  - PRODUCT-ADR-RECOGNITION-004

## Question

Determine whether the 50-epoch `plain-random360` result from PRODUCT-INV-RECOGNITION-007 is still optimization/data-view limited, or whether its approximately 91% dense manual rotation accuracy is already near the practical ceiling of the current Plain architecture.

The investigation should answer:

1. Does increasing the training horizon from 50 to 100 epochs materially improve dense arbitrary-angle `manual_val` accuracy?
2. If improvement is still active at 100 epochs, does extending to 150 epochs provide additional useful gain?
3. Does any gain preserve the Plain model's deployment advantage over production C8?
4. Is the remaining gap to C8 small enough to justify subsequent Plain-specific model/augmentation optimization?

## Existing 50-epoch evidence

PRODUCT-INV-RECOGNITION-007 completed all ten conditions with deployment parity passing. The relevant Pareto-frontier results were:

| condition | manual mean | manual worst | JP mean | JP worst | ORT CPU median b16 |
|---|---:|---:|---:|---:|---:|
| `plain-random360` | 0.91118 | 0.88889 | 0.99898 | 0.99162 | 17.78 ms |
| `c8-production` | 0.97222 | 0.96000 | 0.99984 | 0.99941 | 52.68 ms |

Thus Plain is approximately 2.96x faster in the fixed desktop ORT CPU benchmark, while trailing C8 by about 6.10 percentage points in dense manual mean accuracy and 7.11 points at the worst angle.

The 50-epoch `plain-random360` checkpoint-selection history is not flat at the end of training:

| epoch | train accuracy | manual 0/15/30/45 mean |
|---:|---:|---:|
| 20 | 0.96984 | 0.73944 |
| 25 | 0.97591 | 0.79944 |
| 30 | 0.98229 | 0.80833 |
| 35 | 0.98581 | 0.80556 |
| 40 | 0.98821 | 0.87000 |
| 45 | 0.98989 | 0.89722 |
| 50 | 0.99005 | 0.90222 |

The curve is noisy, but epochs 35 -> 40 -> 45 -> 50 show a late increase of roughly 9.7 percentage points in the checkpoint-selection score. The final five epochs still add about 0.5 point from epoch 45 to 50. This is enough evidence to test a longer schedule, but not enough to assume that another 50 epochs will close the C8 gap.

## Frozen experimental contract

Use the same frozen v3 database as PRODUCT-INV-RECOGNITION-007:

```text
.local/recognition/tile_classifier_datasets/
  gray35_jp500_seed42_v3_jp189.sqlite
```

Keep unchanged:

- architecture: existing `PlainTileShapeClassifier`;
- 64x64 grayscale input and frozen v3 normalization;
- train split: 19,593;
- manual validation split: 450;
- JP validation split: 6,800;
- augmentation: deterministic `Uniform(-180,+180)` angle per `(seed, epoch, sample_id)`;
- optimizer: AdamW;
- learning rate: `1e-3`;
- weight decay: `1e-4`;
- effective batch: 512;
- seed: 42;
- AMP/TF32 policy: same as INV-007;
- checkpoint-selection angles: 0/15/30/45 degrees;
- final dense evaluation: 64 angles, 5.625-degree spacing, on both validation splits;
- ONNX opset/parity gate and ORT CPU batch-16 benchmark: same as INV-007.

Do not rebuild the corpus and do not change augmentation semantics between epoch budgets.

## Training-horizon rule

The longer runs are **new training runs from initialization**, not continuation of the 50-epoch checkpoint.

This matters because the runner uses cosine annealing with `T_max = epochs`. A 50-epoch run has already annealed its learning rate to the end of a 50-epoch schedule. Simply loading epoch 50 and continuing for another 50 epochs would not be equivalent to asking whether a 100-epoch cosine schedule performs better.

Compare:

| run | epochs | cosine `T_max` | role |
|---|---:|---:|---|
| existing INV-007 baseline | 50 | 50 | reference; do not retrain |
| `plain-random360-e100` | 100 | 100 | required |
| `plain-random360-e150` | 150 | 150 | conditional |

Because deterministic random360 angles are keyed by `(seed, epoch, sample_id)`, the 100- and 150-epoch runs see the same augmented sample/angle sequence for their shared epoch numbers. The intended difference is the longer optimization schedule, principally the slower cosine learning-rate decay and the additional later-epoch rotation views.

## Staged execution gate

Run 100 epochs first. Do not automatically spend another full training budget on 150 epochs.

Run the 150-epoch condition only if the 100-epoch result shows evidence that useful learning remains active. Treat either of the following as sufficient evidence:

- dense `manual_val` mean accuracy improves by at least `+0.5` percentage point versus the 50-epoch baseline; or
- the selected/best checkpoint occurs after epoch 70 and the late checkpoint-selection trajectory is still improving rather than having clearly plateaued.

Stop at 100 epochs if the dense manual mean is effectively flat/degraded and the best checkpoint occurs well before the end of the schedule.

The `+0.5 pp` threshold is an engineering continuation gate, not a statistical significance claim. Its purpose is to avoid another long run for a change too small to matter against the current approximately 6.1-point C8 gap.

## Primary comparison metrics

The primary metric is the same as INV-007:

- dense 64-angle `manual_val` mean accuracy.

Also compare:

- dense `manual_val` worst-angle accuracy;
- dense `manual_val` standard deviation across angle;
- JP mean/worst accuracy;
- best checkpoint epoch and 0/15/30/45 checkpoint-selection score;
- training accuracy/loss trajectory;
- ONNX parity;
- ORT CPU batch-16 median/p95 latency.

Latency should remain structurally identical for all Plain checkpoints. Re-benchmark the longer checkpoint to catch accidental graph/export drift, but do not interpret weight-only timing noise as an architectural change.

## Interpretation rules

### Evidence that 50 epochs was too short

The case for longer Plain training is supported if 100 epochs produces a clear increase in dense manual mean/worst accuracy while JP remains saturated and deployment cost remains unchanged.

A late best checkpoint strengthens that interpretation because it shows the longer cosine schedule is still converting additional augmented views into validation robustness.

### Evidence that the Plain architecture is near its present ceiling

Treat the current Plain formulation as effectively saturated if:

- 100 epochs changes dense manual mean by less than about 0.5 point;
- the best checkpoint occurs substantially before the end of the schedule; and
- the late validation trajectory is flat/noisy without a sustained upward envelope.

In that case, do not keep increasing epochs mechanically. The next experiment should change model capacity, regularization, or training data/augmentation rather than spending more compute on the same schedule.

### Relation to production C8

The goal is not necessarily to beat C8 in this investigation. The decision-relevant question is whether Plain can reduce enough of the current manual accuracy gap to justify its approximately 3x deployment-speed advantage.

Interpret the end state roughly as:

- if Plain remains around 91-92% manual mean, retain C8 as the clearly safer accuracy choice;
- if Plain reaches the mid-90% range while preserving worst-angle robustness, it becomes a strong candidate for targeted follow-up and iPhone timing;
- if Plain approaches the approximately 97% C8 manual mean, prioritize direct iPhone comparison before further architecture work.

These bands are investigation guidance, not production acceptance thresholds.

## Execution plan

Reuse the validated INV-007 runner rather than introducing another trainer. Use separate output roots so different epoch budgets cannot be mistaken for resumed copies of one another.

Required 100-epoch run:

```bash
PY=/srv/bugrat/data-lv/mjtensu/nanodet/nanodet/.venv/bin/python
mkdir -p /srv/data/mjtensu/.local/recognition/plain_random360_epoch_sweep

nohup "$PY" tools/recognition/run_rotation_classifier_experiment.py \
  --conditions plain-random360 \
  --epochs 100 \
  --database /srv/data/mjtensu/.local/recognition/tile_classifier_datasets/gray35_jp500_seed42_v3_jp189.sqlite \
  --output-root /srv/data/mjtensu/.local/recognition/plain_random360_epoch_sweep/e100 \
  > /srv/data/mjtensu/.local/recognition/plain_random360_epoch_sweep/e100.log 2>&1 &
```

If the staged continuation gate is met, run 150 epochs from initialization with a fresh output root. The parent directory already exists from the 100-epoch command above; if running it independently, create it first with the same `mkdir -p` command.

```bash
nohup "$PY" tools/recognition/run_rotation_classifier_experiment.py \
  --conditions plain-random360 \
  --epochs 150 \
  --database /srv/data/mjtensu/.local/recognition/tile_classifier_datasets/gray35_jp500_seed42_v3_jp189.sqlite \
  --output-root /srv/data/mjtensu/.local/recognition/plain_random360_epoch_sweep/e150 \
  > /srv/data/mjtensu/.local/recognition/plain_random360_epoch_sweep/e150.log 2>&1 &
```

Do not use `--overwrite-completed` on either output root unless rerunning that exact epoch budget intentionally.

## 100-epoch result: 2026-08-31

The required 100-epoch run completed successfully with ONNX deployment parity and the same Plain deployment graph.

| run | manual mean | manual worst | JP mean | JP worst | ORT CPU median b16 |
|---|---:|---:|---:|---:|---:|
| 50 epochs | 0.91118 | 0.88889 | 0.99898 | 0.99162 | 17.78 ms |
| 100 epochs | 0.93288 | 0.92222 | 0.99943 | 0.99588 | 17.73 ms |
| production C8 reference | 0.97222 | 0.96000 | 0.99984 | 0.99941 | 52.68 ms |

Relative to the 50-epoch Plain baseline, the 100-epoch schedule improved dense `manual_val` mean accuracy by **+2.17 percentage points** and worst-angle accuracy by **+3.33 points**. JP mean also increased by about +0.04 point. The deployment graph size/MAC estimate remained unchanged and measured ORT CPU batch-16 median latency remained effectively identical, as expected for a weight-only training change.

The improvement is far above the investigation's `+0.5 pp` staged-continuation threshold. The gap to production C8 has therefore narrowed from about 6.10 to **3.93 percentage points** on manual mean accuracy and from about 7.11 to **3.78 points** at the worst angle, while Plain remains approximately **2.97x faster** in the fixed desktop ORT CPU benchmark.

This is direct evidence that the original 50-epoch Plain result was still materially training-horizon limited. It does not establish that 150 epochs will continue the same rate of improvement, but it satisfies the predefined gate without needing to infer from noisy checkpoint history.

**Decision:** execute the staged 150-epoch run from initialization with `T_max=150` before changing architecture or augmentation.

## 150-epoch result: 2026-08-31

The staged 150-epoch run also completed successfully with ONNX deployment parity and the same Plain deployment graph.

| run | manual mean | manual worst | JP mean | JP worst | ORT CPU median b16 |
|---|---:|---:|---:|---:|---:|
| 50 epochs | 0.91118 | 0.88889 | 0.99898 | 0.99162 | 17.78 ms |
| 100 epochs | 0.93288 | 0.92222 | 0.99943 | 0.99588 | 17.73 ms |
| 150 epochs | 0.94743 | 0.94000 | 0.99959 | 0.99794 | 17.86 ms |
| production C8 reference | 0.97222 | 0.96000 | 0.99984 | 0.99941 | 52.68 ms |

Relative to 100 epochs, the 150-epoch schedule improved dense `manual_val` mean accuracy by **+1.45 percentage points** and worst-angle accuracy by **+1.78 points**. Relative to the original 50-epoch baseline, the total gains are **+3.63 points** in manual mean and **+5.11 points** at the worst angle. JP remains effectively saturated and also improved slightly.

The remaining gap to production C8 is now approximately **2.48 percentage points** on manual mean and **2.00 points** at the worst angle. The measured ORT CPU batch-16 median remains effectively unchanged; Plain is still approximately **2.95x faster** than the production C8 reference in the fixed desktop benchmark.

The reduction in gain from +2.17 points (50 -> 100) to +1.45 points (100 -> 150) is consistent with diminishing returns.

The 150-epoch checkpoint-selection history resolves whether another pure epoch extension is justified. The selected best score was **0.95333 at epoch 125**. Later full-sweep scores were 0.94500 at epoch 130, 0.94333 at epoch 140, 0.94556 at epoch 145, and 0.94500 at epoch 150. No later checkpoint exceeded epoch 125, and the last 25 epochs show a noisy plateau rather than a rising upper envelope. Training accuracy was already approximately 99.9% throughout this region.

This means the 150-epoch schedule successfully moved the useful optimization region later than the shorter runs and materially improved the final dense result, but the evidence does **not** support mechanically extending the same experiment to 200 epochs. The current Plain formulation is now reasonably treated as training-horizon saturated under this optimizer/augmentation/model configuration.

## Conclusion

INV-008 concludes that the original 50-epoch Plain result was materially undertrained, while the 150-epoch schedule is sufficient for the current configuration.

- 50 -> 100 epochs produced +2.17 pp dense manual mean accuracy.
- 100 -> 150 epochs produced a further +1.45 pp.
- The 150-epoch best checkpoint occurred at epoch 125 and was not surpassed over the final 25 epochs.
- The selected 150-epoch model reaches 0.94743 manual mean and 0.94000 manual worst while retaining approximately 2.95x lower desktop ORT CPU batch-16 latency than production C8.
- The remaining C8 gap is approximately 2.48 pp in manual mean and 2.00 pp at the worst angle.

**Decision:** stop the pure epoch sweep at 150. If the Plain path is pursued further, change a substantive lever such as model capacity, regularization, or augmentation/data composition rather than extending the same cosine schedule to 200 epochs. Direct iPhone timing of the 150-epoch Plain model versus production C8 is also justified because Plain is now in the mid-90% manual-accuracy range while retaining the large deployment-cost advantage.

## Expected decision

This investigation should end with one of three concrete conclusions:

1. **50 epochs was materially undertrained**: longer schedule closes a meaningful fraction of the C8 manual-accuracy gap; continue Plain optimization.
2. **100 epochs helps but is still improving**: execute/assess the staged 150-epoch run before changing architecture.
3. **Plain has plateaued near the current result**: stop spending epochs and move to a different lever if the approximately 3x runtime advantage is still worth pursuing.
