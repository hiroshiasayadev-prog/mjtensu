# MLDB-ADR-SCHEMA-014: Reject duplicate Study grid values

- **status**: accepted
- **date**: 2026-09-04
- **depends_on**: MLDB-ADR-SCHEMA-009, MLDB-ADR-SCHEMA-013
- **supersedes**:
- **migrated_to_spec**: 2026-09-04

## Context

Study v1 materializes a deterministic Cartesian product and assigns persistent Study Run-local `trial-NNNN` lineage to the generated training coordinates.

MLDB-ADR-SCHEMA-009 already rejects duplicate training seeds, but it does not define whether duplicate Architecture IDs or duplicate values within one training parameter axis are valid.

Allowing duplicates creates multiple list positions that describe the same effective training coordinate. Tooling would then need to choose between preserving duplicate coordinates, silently deduplicating them, or rejecting the Study during plan validation. Those choices produce different trial counts and different `trial-NNNN` lineage.

Study v1 does not need duplicate grid entries to express repeated stochastic execution because training seed is already a first-class grid axis.

## Decision

Study v1 rejects duplicate entries in every authored training grid axis.

The following are invalid:

- duplicate Architecture IDs in `training.architectures`;
- duplicate values within one `training.parameters.<key>.values` list;
- duplicate values in `training.seeds`.

Duplicate detection for Architecture IDs uses exact ID string equality.

Duplicate detection for seeds uses exact integer value equality.

Duplicate detection for public parameter values compares the YAML-decoded value including scalar type. Values with different decoded types are not duplicates merely because a language runtime might coerce them to equal values. Nested sequence and mapping values, where permitted by the public-parameter contract, use recursive type-sensitive value equality.

For example:

```yaml
architectures:
  - plain-cnn-v1
  - plain-cnn-v1
```

is invalid.

```yaml
parameters:
  learning_rate:
    values: [0.001, 0.0003, 0.001]
```

is invalid.

```yaml
seeds: [42, 43, 42]
```

is invalid.

A valid Study therefore contains each authored coordinate value at most once per axis. Cartesian expansion produces exactly one trial for each effective coordinate.

This decision does not deduplicate invalid Study input automatically. Validation rejects the Study before a valid Study Run plan is materialized.

If a user wants repeated training under otherwise identical settings, the Study must express that repetition through distinct training seeds. An explicitly repeated execution of an already completed coordinate outside one Study Run remains a higher-level orchestration operation and creates another Run rather than duplicate grid entries.

## Rationale

Rejecting duplicates keeps Study trial count, deterministic ordering, and `trial-NNNN` lineage unambiguous.

Automatic deduplication would make authored input differ silently from materialized intent. Preserving duplicates would create multiple indistinguishable coordinates whose only distinction was list position, which is not useful experiment semantics for Study v1.

Training seed already provides the intended mechanism for stochastic repeated trials while retaining explicit provenance.

Type-sensitive parameter comparison avoids accidental duplicate classification caused by host-language coercion rules.

## Rejected alternatives

### Preserve duplicate coordinates as separate trials

This would allow identical Architecture, parameter, and seed combinations to receive different trial IDs without any semantic difference in their planned training inputs.

Study v1 uses explicit seed variation instead.

### Silently deduplicate repeated values

This would change the authored trial count and plan without reporting invalid input.

Study validation rejects duplicates rather than rewriting the grid.

### Reject only duplicate seeds

Duplicate Architecture IDs and duplicate parameter values can create the same ambiguity in coordinate multiplicity and trial numbering.

All training axes therefore use the same uniqueness principle.

## Consequences

Study validation must reject duplicate Architecture IDs, duplicate values within each training parameter axis, and duplicate seeds before valid plan materialization.

Grid expansion may assume that each authored axis contains unique values.

A valid `plan.jsonl` contains no duplicate effective training coordinates originating from repeated axis entries.

The canonical Study trial-ordering rule from MLDB-ADR-SCHEMA-013 operates over these unique authored axis values.

## Evidence

The ADR-to-spec review identified that the draft Study specs referred to distinct Cartesian coordinates and duplicate-coordinate rejection even though MLDB-ADR-SCHEMA-009 explicitly prohibited only duplicate seeds. This decision closes that ambiguity before Study materializer implementation begins.
