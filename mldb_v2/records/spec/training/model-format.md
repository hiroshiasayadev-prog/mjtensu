# Contract: Model format

- **id**: `spec:mldb.v2.training.model_format`
- **status**: draft
- **date**: 2026-09-09
- **parent**: `spec:mldb.v2.training`
- **contract_class**: `format`

## Shape

Model is the immutable learned identity produced by exactly one completed Training Result. Schema is
`mjtensu.mldb-v2/model/v1`.

```yaml
schema: mjtensu.mldb-v2/model/v1
id: rotated-fcos/run-abcd-trial-0001-model
training_result: rotated-fcos/run-abcd-trial-0001-train
```

These are the only v1 top-level fields. Architecture, Task, Corpus, Train Protocol, parameters,
seed, weights, source commit, and backend provenance remain authoritative in the referenced immutable
Training Result and are not duplicated.

## Identity

The Model ID is deterministic from Study Result + trial according to
`spec:mldb.v2.results.study_result_format`. Model lives in the Study Result namespace regardless of
where referenced Architecture/Corpus/Protocol definitions live.
## Invariants and loading

One completed Training Result produces exactly one Model. Failed/cancelled Training Results produce
none. A Model is immutable immediately after creation and has no independent revision, quality,
promotion, deployment, or mutable `best` status.

Model loading resolves the completed Training Result, verifies its canonical weights ArtifactRef,
resolves the selected Architecture, and follows `spec:mldb.v2.training.canonical_weights` strict
loading rules. Downstream evaluation refers to Model rather than Training Result.

A later Study may use an existing Model without retraining. Backend model IDs are provenance only and
do not participate in Model identity.
