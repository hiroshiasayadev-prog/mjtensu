# MLDB-ADR-SCHEMA-019: Define Training seed type

- **status**: accepted
- **date**: 2026-09-04
- **depends_on**: MLDB-ADR-SCHEMA-004, MLDB-ADR-SCHEMA-005, MLDB-ADR-SCHEMA-009, MLDB-ADR-SCHEMA-010
- **supersedes**:
- **migrated_to_spec**: 2026-09-04

## Context

MLDB already treats training seed as a first-class execution input rather than as a Train Protocol public parameter.

Training Run persists the concrete seed, Study uses seed as one Cartesian grid axis, Study Run plan persists the selected seed for each trial, and `TrainContext` supplies that same value to Train Protocol code.

The existing records do not define the persisted seed value type. Without a shared type contract, values such as floating-point numbers, strings, booleans, or implementation-specific numeric objects could enter different parts of the execution path inconsistently.

MLDB needs one small stable seed-domain rule without introducing a generic random-number-generator or framework-specific seed policy.

## Decision

Training seed in MLDB v1 is an integer value.

Boolean values are not valid training seeds even in host languages where boolean is an integer subtype.

MLDB v1 defines no universal minimum or maximum seed value. Framework-specific or protocol-specific seed-range requirements may reject a concrete execution value when required by that implementation, but they do not change the persisted MLDB seed type.

The exact integer seed is preserved without coercion across:

```text
Study training.seeds
  -> Study Run plan training.seed
  -> Training Run execution.seed
  -> TrainContext.seed
```

A direct Training Run request is subject to the same integer-and-not-boolean rule before Run allocation.

Study materialization rejects a Study whose `training.seeds` list contains any non-integer value or boolean value.

Training Run validation rejects an `execution.seed` outside this domain.

Study Run plan validation rejects a `training.seed` outside this domain.

`TrainContext.seed` is supplied as the validated integer value and must not be derived from a Train Protocol public parameter.

## Rationale

Seed is execution identity input and should have one stable representation across direct Runs, Study plans, and executable protocol context.

Restricting the value to integer matches the intended random-seed role while avoiding ambiguous coercions such as `"42"`, `42.0`, or `true`.

Excluding boolean explicitly is necessary because some implementation languages treat booleans as integer subtypes.

Avoiding a universal numeric range keeps MLDB independent from the exact accepted ranges of PyTorch, NumPy, third-party trainers, or future training implementations.

## Rejected alternatives

### Accept any JSON-compatible scalar

This would allow strings, floating-point values, or null as seeds and would force individual executors to invent coercion rules.

### Treat boolean as integer

Values such as `true` and `false` are semantically boolean rather than authored numeric seeds. Accepting them would make YAML and host-language type behavior leak into the execution contract.

### Standardize a fixed integer range

Different random-number-generator implementations may support different seed domains. MLDB v1 only needs a stable persisted type, not a universal RNG specification.

### Move training seed into Train Protocol parameters

Training seed is already a first-class Training Run and Study axis with lifecycle and lineage meaning. Moving it into protocol parameters would weaken that shared execution identity contract.

## Consequences

Direct Training Run preflight must reject a non-integer or boolean seed before allocating a Training Run.

Study validation and materialization must reject invalid seed values before creating a complete plan.

Training Run, Study Run plan, and TrainContext implementations must preserve the validated integer value without string or floating-point coercion.

Protocol implementations remain responsible for applying `context.seed` to their concrete RNGs and for any implementation-specific range checks that are not standardized by MLDB.

## Evidence

Training Run already persists `execution.seed`, Study already declares `training.seeds` as an explicit Cartesian axis, Study Run plan persists `training.seed`, and TrainContext already types `seed` as `int`. The consistency review identified that the persisted value type itself had not been made normative.
