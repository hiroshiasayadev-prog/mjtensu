# PRODUCT-INV-RECOGNITION-009: Characterize local angular instability of the Plain random360 classifier

- **status**: completed
- **date**: 2026-08-31
- **trigger**: iPhone 13 testing of the Plain random360 e150 candidate showed visibly worse recognition stability than production C8, especially for meld crops. Small physical angle changes could change tile identity and prevent semantic stabilization even though PRODUCT-INV-RECOGNITION-008 measured 0.94743 mean / 0.94000 worst dense manual accuracy on the 64-angle 5.625-degree grid. The deployed symptom suggests that the coarse dense sweep may hide high-frequency local angular instability at sub-grid angle changes.
- **scope**: Measure prediction stability of the accepted Plain random360 e150 checkpoint under a fine local rotation sweep, first on the frozen `manual_val` gray64 inputs and against production C8 as a reference. Record per-angle accuracy, adjacent-angle prediction flip rate, per-sample class changes, and the fraction of center-correct samples that remain correct throughout the local window. Use this evidence to decide whether the next Plain experiment should change rotation augmentation coverage rather than model capacity or epoch count.
- **non_scope**: Retraining in this investigation, detector changes, meld grouping changes, red-five classification, changing semantic stabilization thresholds, production promotion, or concluding that a specific augmentation policy is superior before measuring the local-instability symptom.
- **source_refs**:
  - PRODUCT-INV-RECOGNITION-007
  - PRODUCT-INV-RECOGNITION-008
  - tools/recognition/run_rotation_classifier_experiment.py
  - .local/recognition/tile_classifier_datasets/gray35_jp500_seed42_v3_jp189.sqlite
  - .local/recognition/plain_random360_epoch_sweep/e150/plain-random360/training/best.pt
  - .local/recognition/tile_classifier_runs/gray64_c8_rot22p5_bs512_gray35_v3_jp189_seed42/best.pt
- **planned_outputs**:
  - tools/recognition/analyze_local_angular_stability.py
  - .local/recognition/plain_random360_local_angular_stability/
  - PRODUCT-INV-RECOGNITION-009

## Question

Does the Plain random360 e150 classifier exhibit materially worse **local angular smoothness** than production C8, despite its strong average accuracy on the existing 5.625-degree dense sweep?

The practical failure mode is not merely a wrong classification at one absolute orientation. Recognition stabilization requires the same physical crop to remain semantically consistent across successive camera frames. A classifier that alternates between tile identities when the crop rotates by only one or two degrees can therefore fail the product even when its aggregate angle-grid accuracy looks acceptable.

## First-stage evaluation contract

Use the frozen `manual_val` split from the v3 compact database. This avoids detector/crop-extraction differences while testing the classifier symptom directly on the same gray64 input contract used by INV-007/008.

Compare:

1. Plain random360 e150 selected checkpoint, best epoch 125;
2. production C8 v3 checkpoint, epoch 45.

Use a local sweep centered on the stored crop orientation:

```text
-10, -9, -8, ... 0, ... +8, +9, +10 degrees
```

That is 21 deterministic views at one-degree spacing. Rotation must reuse the INV-007 tensor rotation semantics (`affine_grid` / bilinear `grid_sample` / border padding / `align_corners=False`) so the comparison is not confounded by a second image-rotation implementation.

The first-stage sweep is deliberately much finer than the existing 5.625-degree evaluation grid. If the symptom is real, this should expose short-period prediction changes that a coarse grid can miss.

## Metrics

For each model record at least:

- accuracy at every angle;
- adjacent-angle prediction flip rate, e.g. fraction of samples whose argmax changes from `3 deg` to `4 deg`;
- mean and maximum distinct predicted classes per sample across the 21-angle window;
- fraction of samples whose predicted class is constant across the full window;
- 0-degree accuracy;
- fraction of samples correct at 0 degrees and correct at every angle in the window;
- conditional local robustness: among samples correct at 0 degrees, fraction that remain correct at every angle in the window.

The plot should show at minimum:

1. accuracy versus local angle for Plain and C8;
2. adjacent-angle flip rate versus local angle for Plain and C8.

Also persist a per-sample CSV so unstable crops can be inspected directly rather than reducing the experiment to one mean curve.

## Interpretation

Evidence for an augmentation-coverage problem is strongest if Plain has similar 0-degree accuracy to its expected manual-domain level but materially higher one-degree flip rate / lower conditional local robustness than C8.

A smooth per-angle mean curve by itself is not enough to reject the symptom: different samples can flip at different angles and average out. The per-sample stability metrics are therefore required.

If Plain is locally smooth on the frozen `manual_val` inputs but remains unstable on iPhone meld crops, the next step should move the same fine-angle sweep onto captured deployment crops. That would point toward crop-domain differences such as translation, perspective, background contamination, overlap, or detector-box jitter rather than random360 angle coverage alone.

If Plain is measurably locally unstable already on `manual_val`, follow up with a separate training investigation comparing the current IID `Uniform(-180,+180)` one-view-per-sample-per-epoch policy against a coverage-controlled policy such as stratified angular bins plus intra-bin jitter. Do not assume that policy wins before this diagnostic establishes the failure mode.

## Server execution

The analysis script consumes the already trained checkpoints and does not retrain either model.

```bash
PY=/srv/bugrat/data-lv/mjtensu/nanodet/nanodet/.venv/bin/python; "$PY" tools/recognition/analyze_local_angular_stability.py --database /srv/data/mjtensu/.local/recognition/tile_classifier_datasets/gray35_jp500_seed42_v3_jp189.sqlite --plain-checkpoint /srv/data/mjtensu/.local/recognition/plain_random360_epoch_sweep/e150/plain-random360/training/best.pt --c8-checkpoint /srv/data/mjtensu/.local/recognition/tile_classifier_runs/gray64_c8_rot22p5_bs512_gray35_v3_jp189_seed42/best.pt --output-root /srv/data/mjtensu/.local/recognition/plain_random360_local_angular_stability
```

Expected primary outputs:

```text
.local/recognition/plain_random360_local_angular_stability/
  summary.json
  per_angle.csv
  per_sample.csv
  local_angular_stability.svg
```

## Result: 2026-08-31

The local one-degree sweep completed on all 450 `manual_val` samples.

| model | 0 deg accuracy | local mean | local worst | mean adjacent flip | stable prediction fraction | all-angle correct given 0-deg correct |
|---|---:|---:|---:|---:|---:|---:|
| Plain random360 e150 | 0.96000 | 0.95249 | 0.94667 | 0.00511 | 0.95111 | 0.96296 |
| production C8 | 0.97778 | 0.97725 | 0.97111 | 0.00200 | 0.98667 | 0.99318 |

Plain is measurably less locally smooth than C8: its mean adjacent one-degree prediction flip rate is about 2.6x higher and its full-window stable-prediction fraction is lower by about 3.56 percentage points. This confirms that the Plain classifier gives up some of C8's local angular stability.

However, the magnitude is not sufficient to explain the deployment symptom by itself. Even Plain keeps one constant predicted class across the full ±10-degree window for 95.11% of frozen manual samples, and among samples that are correct at 0 degrees, 96.30% remain correct at every angle in the window. The iPhone failure therefore cannot be attributed solely to one- or two-degree classifier rotation sensitivity without additional deployment-domain evidence.

Subsequent iPhone debug inspection showed badly localized detector boxes, including completed-hand crops whose source boxes contain substantial background or incomplete tile faces. Production `extractCrop` uses the detector `sourceBox` exactly, with no expansion, rectification, or classifier-side recovery. Detector localization error therefore propagates directly into the classifier input.

## Conclusion

INV-009 confirms a real but secondary Plain local-angular weakness. The stronger current hypothesis for the severe iPhone instability is detector localization / crop-domain error, especially in dense and rotated layouts. Do not start a Plain angular-augmentation retrain solely from this result. First characterize and improve the detector training distribution and geometric augmentation under PRODUCT-INV-RECOGNITION-010.
