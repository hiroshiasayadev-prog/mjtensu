# PRODUCT-INV-RECOGNITION-023: Evaluate Plain random360 effective batch size

- **status**: concluded
- **date**: 2026-10-04
- **trigger**: Earlier classifier work showed that very large batches weaken validation accuracy and that this task is sensitive to loss of fine local discriminative structure. The Plain random360 training line still retained effective batch 512 as a high-throughput GPU-oriented baseline, so the batch size itself required re-evaluation before further architecture conclusions were trusted.
- **scope**: Compare effective batch 256 and 128 against the historical Plain random360 e150 batch-512 reference while keeping architecture, corpus, optimizer hyperparameters, augmentation, epoch budget, seed, preprocessing, checkpoint-selection semantics, AMP, and TF32 unchanged.
- **non_scope**: Architecture changes, augmentation changes, learning-rate scaling, epoch-budget changes, optimizer changes, multi-seed confirmation, detector training, production promotion, and GPU-throughput optimization.
- **source_refs**:
  - PRODUCT-INV-RECOGNITION-005
  - PRODUCT-INV-RECOGNITION-008
  - PRODUCT-INV-RECOGNITION-012
  - PRODUCT-INV-RECOGNITION-013
  - PRODUCT-INV-RECOGNITION-014
- **follow_up_candidates**:
  - PRODUCT-INV-RECOGNITION-024
  - Cross-architecture confirmation of lower effective batch size if later classifier selection requires it

## Investigation scope

Evaluate whether the effective batch size inherited by the Plain random360 classifier is unnecessarily large for model quality.

The experiment keeps the accepted Plain random360 e150 recipe fixed and changes only optimizer-step batch size from the historical 512 reference to 256 and 128.

The investigation focuses on classifier quality. Training throughput and GPU utilization are operational context rather than selection goals.

## Out of scope

- Changing the Plain architecture.
- Changing the random360 augmentation distribution.
- Changing the 150-epoch training budget.
- Scaling learning rate with batch size.
- Changing AdamW or weight decay.
- Multi-seed confirmation.
- Detector training.
- Production promotion.
- Optimizing GPU training throughput.

## Background

PRODUCT-INV-RECOGNITION-005 already found that increasing C8 batch size from 512 to 1,024, 2,048, and 4,096 progressively reduced manual-validation accuracy.

The same investigation found no material elapsed-training-time improvement from those larger batches.

Later classifier work established a separate but relevant property of the task.

PRODUCT-INV-RECOGNITION-012 observed live within-suit errors such as 2m -> 7m and 6m -> 7m. The failure pattern was consistent with loss of fine stroke-level information.

PRODUCT-INV-RECOGNITION-013 retained a measurable 5m/6m/7m confusion surface after perspective-aware augmentation and identified fine local spatial discrimination as a remaining architecture concern.

PRODUCT-INV-RECOGNITION-014 then showed that reducing early spatial compression improved fine-grained classifier robustness under a controlled ShuffleNet screen.

These records do not establish that large optimizer batches erase local features.

They do establish that:

- very large batches have already hurt classifier validation quality in this project;
- fine local discrimination matters materially for this tile task;
- the effective batch of 512 should not be preserved only because it is convenient for GPU throughput.

The working hypothesis was that a smaller effective batch may improve the optimization outcome for fine-grained tile discrimination.

## What was investigated

The sealed MLDB Study was:

| evidence | ref |
|---|---|
| Study | tile-classifier/plain-random360-e150-batch-sweep-v1 |
| completed StudyResult | tile-classifier/run-1316c835f3334066ab720751cc269742 |
| source commit | d2e2c464d494a2bd4dd9e6e034668d37b5745735 |

The two new training conditions were:

| condition | effective batch | epochs |
|---|---:|---:|
| bs256 | 256 | 150 |
| bs128 | 128 | 150 |

No gradient accumulation was used to preserve 512.

Both trials kept these controls fixed:

- Plain 32/64/128/192 classifier architecture;
- frozen gray35 v3_jp189 corpus;
- deterministic random360-only-v1;
- AdamW learning rate 0.001;
- weight decay 0.0001;
- seed 42;
- AMP enabled;
- TF32 enabled;
- checkpoint evaluation every five epochs;
- the current full-class, validity/rejection, dense-angle, Manzu, and CPU-latency evaluation suite.

The historical bs512 reference is PRODUCT-INV-RECOGNITION-008.

The bs512 reference is not a same-Study rerun. Comparisons against bs512 therefore use it as historical evidence rather than same-run statistical proof.

## Findings

### Historical batch-512 reference

PRODUCT-INV-RECOGNITION-008 used effective batch 512 for the accepted Plain random360 e150 run.

Its selected checkpoint reported:

| metric | bs512 historical reference |
|---|---:|
| dense manual angle mean | 0.94743 |
| dense manual worst angle | 0.94000 |
| selected best epoch | 125 |

### Same-Study batch-256 versus batch-128 result

Both new trials completed training and every planned evaluation.

| metric | bs256 | bs128 |
|---|---:|---:|
| dense manual angle mean | 0.963368 | **0.964444** |
| full front accuracy | 0.964444 | **0.966667** |
| full mean-condition accuracy | 0.924000 | **0.928296** |
| full worst-condition accuracy | 0.795556 | **0.822222** |
| full worst-class condition recall | **0.200000** | 0.090909 |
| full 6m -> 5m/7m confusion max | 0.166667 | **0.000000** |
| Manzu mean-condition accuracy | 0.912057 | **0.936170** |
| Manzu worst-condition accuracy | 0.659574 | **0.744681** |
| Manzu 6m neighbor confusion max | 0.166667 | **0.000000** |
| validity balanced accuracy | 0.901465 | 0.901465 |
| validity ROC AUC | 0.975105 | **0.978020** |
| CPU latency p50 | 0.8751 ms | 0.8745 ms |

The bs128 trial improved the main robustness metrics over bs256.

The strongest improvements were in worst-condition and Manzu behavior rather than zero-degree or aggregate-validity metrics.

CPU inference latency remained effectively unchanged because training batch size does not change the deployed architecture.

The current sweep is directionally consistent with the historical bs512 result. The historical comparison is not same-run evidence, but it provides no reason to retain 512 as a quality-oriented default.

## Cross-cutting observations

The batch-size result aligns with two earlier evidence streams.

First, PRODUCT-INV-RECOGNITION-005 already showed that larger optimizer batches can weaken this classifier family.

Second, PRODUCT-INV-RECOGNITION-012 through PRODUCT-INV-RECOGNITION-014 showed that fine-grained tile identity depends on preserving local discriminative information.

The current experiment does not prove that large batches directly discard local features.

The current experiment does show that reducing batch size improves several metrics that expose fine-grained and worst-case discrimination.

The result is therefore more informative than front accuracy alone.

The batch-size choice should be treated as a model-quality hyperparameter rather than a GPU-utilization setting.

## Follow-up judgment candidates

- Whether effective batch 128 should replace 512 as the default starting point for future Plain-family classifier experiments.
- Whether the batch-size effect persists across C8 and newer Plain variants.
- Whether additional batch reduction below 128 is useful after cross-architecture confirmation.
- Whether later comparisons should equalize optimizer-update count when epoch count is held fixed.

## Recommendation

Effective batch 128 appears preferable to 256 within this controlled Plain random360 e150 sweep.

The historical 512 default no longer appears justified as a quality-first choice.

Future classifier experiments should avoid preserving 512 solely for GPU throughput.

Any default-policy change across all classifier architectures should use follow-up cross-architecture evidence rather than this single Plain Study alone.

## Follow-up artifact candidates

- PRODUCT-INV-RECOGNITION-024 for the detector-side batch-size check triggered by this classifier result.
- A classifier follow-up investigation if batch-size behavior is compared across C8, Plain 1.0x, and reduced-width Plain variants.
- A later training-recipe ADR only if a project-wide effective-batch default is adopted.

## Open questions

- Does the improvement persist across classifier architectures rather than only Plain random360?
- How far below effective batch 128 does useful improvement continue?
- How much of any further gain comes from smaller-batch optimization dynamics versus the increased number of optimizer updates per epoch?
- Does lower batch size improve reviewed real-crop failure recall in the same direction as the existing robustness metrics?
