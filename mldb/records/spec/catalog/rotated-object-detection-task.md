# Contract: Rotated-object-detection Task target

- **id**: `spec:mldb.catalog.rotated_object_detection_task`
- **status**: draft
- **date**: 2026-09-09
- **parent**: `spec:mldb.catalog`
- **contract_class**: `format`

## What this is

Defines the concrete Task target contract selected when `target.type` is
`rotated-object-detection`.

The contract describes semantic object classes and rotated-rectangle target meaning.
It does not define image encoding, Corpus storage, model tensors, training loss, or
post-processing thresholds.

## Current contract

A rotated-object-detection Task uses:

```yaml
target:
  type: rotated-object-detection
  labels: [mahjong_tile]
  geometry:
    format: cx-cy-w-h-angle-deg
    angle_period_deg: 180
```

`labels` is the normative ordered object-class vocabulary for this Task revision.
Unlike a categorical-classification target, the prediction unit is a set of zero or
more object instances, each carrying one label and one rotated rectangle.

`geometry.format` is exactly `cx-cy-w-h-angle-deg`. Coordinates and dimensions are
expressed in the semantic image coordinate system supplied by the selected Corpus.
`angle_period_deg` is a positive integer and defines geometric angle periodicity.
For the mjtensu OBB contract it is `180`.

## Validation rules

- `labels` must contain unique non-empty strings.
- `geometry.format` must be `cx-cy-w-h-angle-deg`.
- `geometry.angle_period_deg` must be a positive integer.
- Changing labels, label ordering, geometry format, or angle periodicity is a semantic
  target change and requires a new Task identity/revision.
- Generic Task validation must continue to tolerate other non-categorical target
  structures that are not selected by this concrete target type.

## Boundary

| concern | owner |
|---|---|
| Semantic object labels and OBB meaning | This contract. |
| Materialized image/annotation storage | Rotated-object-detection Corpus contract. |
| Dense detector tensor interface | Architecture. |
| Assignment, loss, augmentation, thresholds | Train/Evaluation Protocols. |
