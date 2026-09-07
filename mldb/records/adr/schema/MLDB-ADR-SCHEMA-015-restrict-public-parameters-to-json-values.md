# MLDB-ADR-SCHEMA-015: Restrict public parameters to JSON-compatible values

- **status**: accepted
- **date**: 2026-09-04
- **depends_on**: MLDB-ADR-SCHEMA-004, MLDB-ADR-SCHEMA-007, MLDB-ADR-SCHEMA-009, MLDB-ADR-SCHEMA-010
- **supersedes**:
- **migrated_to_spec**: 2026-09-04

## Context

Train Protocol and Evaluation Protocol public parameters are authored in YAML and copied into Training Run, Evaluation Run, and Study Run plan records as fully resolved values.

Earlier contracts described public parameter values as arbitrary YAML-compatible values. Study Run, however, persists the fully resolved parameter mappings inside `plan.jsonl`. YAML supports values that do not have a lossless representation in the standard JSON data model, including non-string mapping keys and non-finite numeric values.

Without a common persistence domain, one runtime could reject such values, another could stringify them, and another could apply implementation-specific conversion before writing `plan.jsonl`. That would make Study materialization and child Run intent non-portable even when the same protocol declarations are used.

MLDB does not need a general-purpose parameter type system in v1. It only needs one lossless value domain shared by YAML-authored definitions and JSONL materialized plans.

## Decision

MLDB v1 public parameter values are restricted to the JSON-compatible data model.

A public parameter value may recursively contain only:

- `null`;
- boolean;
- string;
- finite integer or finite floating-point number;
- array of JSON-compatible values;
- object whose keys are strings and whose values are JSON-compatible values.

Values outside this domain are invalid public parameter values.

In particular, MLDB v1 rejects:

- NaN;
- positive or negative infinity;
- mappings with non-string keys;
- YAML-specific scalar/object types that cannot be represented losslessly in JSON;
- implementation-specific objects or tagged values.

YAML remains an authoring syntax for Task-related assets, Train Protocol, Evaluation Protocol, Study, and other MLDB records. Restricting parameter values to the JSON-compatible domain does not require those records themselves to be stored as JSON.

The same value must survive this path without semantic conversion:

```text
Protocol YAML default or caller-supplied Study/Run value
  -> YAML decoding
  -> public parameter resolution
  -> Training Run / Evaluation Run YAML
  -> Study Run plan.jsonl when applicable
  -> protocol context parameters
```

Generic MLDB resolution must not coerce parameter values between strings, numbers, booleans, arrays, or objects. It validates persistence compatibility and otherwise preserves the decoded value.

This decision does not introduce generic parameter declarations such as `type`, numeric ranges, enums, coercion rules, or schema validation. Such metadata remains advisory unless a later contract gives it normative meaning.

## Rationale

The JSON-compatible domain is sufficient for current sweep parameters such as learning rate, batch size, epoch count, thresholds, booleans, string modes, and small structured options while remaining directly persistable in `plan.jsonl`.

Using one shared value domain avoids introducing a custom YAML-to-JSON conversion layer and guarantees that a Study plan can record the exact resolved parameter intent supplied to child Runs.

Rejecting non-finite numbers also avoids dependence on permissive non-standard JSON encoders for `NaN` and infinity.

Keeping the rule to persistence compatibility rather than a full parameter type system preserves the existing ownership boundary: each executable protocol still decides the domain-specific meaning and validation of its published values.

## Rejected alternatives

### Allow arbitrary YAML values and define conversion rules into JSON

This would require MLDB to define conversions for YAML-specific values and could alter parameter meaning during Study materialization.

### Store Study Run plans as YAML instead of JSONL

The existing Study Run contract intentionally uses one JSON object per trial in `plan.jsonl`. Changing the plan format is unnecessary when current parameter needs fit the JSON data model.

### Add a general parameter schema/type system

Current automation needs only named published values and defaults. Generic types, ranges, enums, and coercion would add policy not required for current experiments.

## Consequences

Train Protocol and Evaluation Protocol validators must reject a public parameter `default` outside the JSON-compatible domain.

Study validators must reject grid values or fixed evaluation parameter values outside the JSON-compatible domain.

Direct Training Run and Evaluation Run launch preflight must reject caller-supplied parameter values outside the JSON-compatible domain before Run allocation.

Fully resolved Training Run, Evaluation Run, and Study Run plan parameter mappings are therefore losslessly representable in both YAML and JSON.

Protocol-specific validation may impose narrower rules after common parameter resolution, but it must not rely on values that violate this common persistence domain.

## Evidence

The Study Run contract persists complete resolved Train Protocol and Evaluation Protocol parameter mappings in `plan.jsonl`, while the common public-parameter spec previously admitted arbitrary YAML-compatible values. The mismatch was identified during the ADR-to-spec consistency review before MLDB runtime implementation.
