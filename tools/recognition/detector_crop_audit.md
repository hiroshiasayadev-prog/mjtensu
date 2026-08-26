# Detector crop review / gray35 loop

## Duplicate handling: one shared implementation

Duplicate handling is defined only by `tools/recognition/detector_duplicate_groups.py`.

For every raw `candidate` bbox in the same `capture_id + region`:

```text
overlap(A, B) = intersection_area(A, B) / min(area(A), area(B))
duplicate edge when overlap >= 0.80
```

Connected bboxes form one duplicate cluster. Each cluster keeps exactly one bbox: the candidate with the highest NanoDet detector confidence. Ties use the lower detection index. All other members are losers.

A shot may have multiple independent duplicate clusters. Example:

```text
{A, B}       -> winner A
{C, D, E}    -> winner E
F            -> singleton

normal review candidates = A, E, F
duplicate audit items     = [A + B], [E + C + D]
```

`review_detector_crop_audit.py`, `build_gray35_classifier_dataset.py`, `audit_detector_crop_classifier.py`, and detector dataset generation all use the shared duplicate implementation. They do not trust an existing `postprocess_decision` table to decide current winner/loser membership.

The detector DB may still preserve `postprocess_decision` as generated audit data, but raw `candidate` geometry + detector confidence are the source from which duplicate groups are recomputed.

## Human review

Run:

```powershell
python tools\recognition\review_detector_crop_audit.py
```

The normal page shows only:

```text
non-duplicate singleton candidates
+
one final winner from each duplicate cluster
```

Human labels on that page are gray35 training truth:

```text
valid -> one canonical base tile label
invalid -> background | partial_tile | multi_tile | other
```

Duplicate losers never appear in the normal review page and never become gray35 invalid examples merely because they lost duplicate suppression.

The `重複除去audit` page shows one duplicate cluster per review item:

```text
KEEP / WINNER: detector-confidence maximum
REMOVE 1..N: all other members of that cluster
```

Confirming a duplicate cluster is an audit action only. It is never a classifier training label.

Existing human crop reviews are kept in `reviews.<detector_run_key>.sqlite` and continue to be keyed by `candidate_id`. Old duplicate-audit confirmations that used previous pair IDs are ignored naturally because new confirmations use stable cluster IDs.

## Build gray35

```powershell
python tools\recognition\build_gray35_classifier_dataset.py --force
```

The builder recomputes duplicate groups from raw detector candidates and appends only human-reviewed final winners/singletons. Reviewed loser rows, including stale reviews from an older UI, are ignored.

## Classifier audit

```bash
python tools/recognition/audit_detector_crop_classifier.py \
  --database .local/recognition/detector_crop_dataset/dataset.sqlite \
  --checkpoint .local/recognition/tile_classifier_runs/gray64_c8_rot22p5_bs512_gray35_seed42/best.pt
```

Classifier inference also recomputes duplicate groups and runs only on final winners/singletons. Predictions remain in `classifier_audit.sqlite` and never become human truth automatically.

## Tests

The duplicate behavior is locked by tests for:

- a 2-bbox cluster and a separate 3-bbox cluster in the same shot -> two winners
- transitive 3-bbox duplicate clustering
- region isolation
- detector-confidence winner selection
- normal review containing winner + singleton but not loser
- gray35 builder ignoring deliberately incorrect legacy `postprocess_decision` rows
