# MLDB-ADR-SCHEMA-013: Define deterministic Study trial ordering

- **status**: accepted
- **date**: 2026-09-04
- **depends_on**: MLDB-ADR-SCHEMA-009, MLDB-ADR-SCHEMA-010
- **supersedes**:
- **migrated_to_spec**: 2026-09-04

## Context

MLDB-ADR-SCHEMA-009 requires deterministic Cartesian-product Study expansion, and MLDB-ADR-SCHEMA-010 assigns sequential Study Run-local identifiers such as `trial-0001` to the materialized training trials.

Those decisions do not define the canonical axis nesting order used to enumerate one Study grid.

Without a fixed ordering, two conforming materializers could produce the same set of training coordinates but assign different `trial-NNNN` identifiers. Training Run and Evaluation Run lineage persists those trial identifiers, so the ordering is part of the durable Study Run contract rather than a presentation detail.

The ordering should preserve meaningful author-provided list order while avoiding dependence on YAML mapping insertion order for parameter-axis identity.

## Decision

Study v1 enumerates training trials using the following canonical Cartesian-product order, from outermost to innermost axis:

```text
Architecture declaration order
  -> training parameter keys in locale-independent ascending lexicographic order
    -> each parameter's values in declaration order
      -> training seeds in declaration order
```

Training seed is therefore the innermost axis.

Parameter-key ordering compares the exact parameter key strings and must not depend on locale, YAML parser insertion behavior, filesystem order, hash-map iteration, or worker scheduling.

The order in which keys appear under `training.parameters` in Study YAML does not affect Study semantics or trial numbering.

For example, given:

```yaml
architectures:
  - plain-cnn-v1
  - mobilenet-v1

parameters:
  learning_rate:
    values: [0.001, 0.0003]
  batch_size:
    values: [512, 1024]

seeds: [42, 43]
```

parameter keys are enumerated as:

```text
batch_size
learning_rate
```

and the first coordinates are:

```text
trial-0001 plain-cnn-v1 / batch_size=512 / learning_rate=0.001  / seed=42
trial-0002 plain-cnn-v1 / batch_size=512 / learning_rate=0.001  / seed=43
trial-0003 plain-cnn-v1 / batch_size=512 / learning_rate=0.0003 / seed=42
trial-0004 plain-cnn-v1 / batch_size=512 / learning_rate=0.0003 / seed=43
trial-0005 plain-cnn-v1 / batch_size=1024 / learning_rate=0.001 / seed=42
```

Sequential `trial-NNNN` identifiers are assigned in this generated training-coordinate order, beginning at `trial-0001` with no gaps.

Evaluation stages do not affect training trial numbering. Within every materialized trial row, evaluation stages preserve the declaration order from the Study's `evaluations` list.

This decision defines semantic trial ordering only. It does not require a particular JSON object-key ordering, whitespace policy, line-ending convention, or other byte-level `plan.jsonl` serialization rule.

## Rationale

Architecture, parameter-value, seed, and evaluation-stage lists are intentional ordered author inputs, so preserving their declared order keeps the materialized plan easy to inspect.

Sorting parameter keys makes equivalent Study definitions produce the same trial ordering even when YAML mapping keys are written in a different order.

Using a locale-independent lexical order avoids host-specific ordering behavior.

Keeping seeds innermost groups stochastic repeats of one otherwise identical training coordinate together, which makes trial plans easier to inspect and compare.

Separating semantic trial ordering from byte-level JSONL serialization fixes durable lineage without unnecessarily freezing serialization details that are not yet needed for interoperability.

## Rejected alternatives

### Use Study YAML parameter declaration order

This would make reordering mapping keys change every downstream `trial-NNNN` assignment even when the experiment grid is semantically unchanged.

### Sort every axis value

Architecture, parameter-value, and seed list order may be intentionally chosen by the author. Sorting those values would discard that authored ordering and would require generic comparison semantics across arbitrary parameter values.

### Leave ordering implementation-defined

The resulting coordinate set would be equivalent, but Study-local trial identities could differ between materializers. Because child Run lineage persists those identifiers, implementation-defined ordering is not acceptable.

### Couple trial ordering to byte-level JSONL canonicalization

Stable trial identity requires semantic coordinate order, but it does not require byte-identical JSON serialization across implementations. Byte-level canonicalization remains a separate decision.

## Consequences

Study materialization must sort `training.parameters` keys using the canonical locale-independent lexical ordering before Cartesian expansion.

Architecture order, each parameter `values` order, seed order, and evaluation-stage order remain as authored in their respective YAML lists.

Reordering only the keys beneath `training.parameters` does not change the materialized trial sequence.

Changing Architecture order, a parameter `values` order, or seed order changes trial numbering and therefore changes the materialized plan of a newly executed Study revision or Study Run.

Study Run plan validation can reconstruct the expected coordinate sequence and verify that each persisted `trial-NNNN` corresponds to its row position and canonical Study expansion order.

## Evidence

The pre-implementation spec review identified Study canonical ordering as unresolved while `trial-NNNN` is persisted by Training Run and Evaluation Run lineage. The existing Study grid spec already marked a candidate axis order and explicitly required the ordering to be fixed before independent materializers rely on stable trial identifiers.
