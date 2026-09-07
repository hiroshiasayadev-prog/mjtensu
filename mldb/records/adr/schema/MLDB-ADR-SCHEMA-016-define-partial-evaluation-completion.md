# MLDB-ADR-SCHEMA-016: Define partial Evaluation Run completion

- **status**: accepted
- **date**: 2026-09-04
- **depends_on**: MLDB-ADR-SCHEMA-007, MLDB-ADR-SCHEMA-008, MLDB-ADR-SCHEMA-010
- **supersedes**:
- **migrated_to_spec**: 2026-09-04

## Context

Evaluation Protocol declares a stable formal result surface consisting of scalar metrics and structured artifacts.

MLDB-ADR-SCHEMA-007 and MLDB-ADR-SCHEMA-008 distinguish valid formal outputs from malformed outputs and already allow omission of artifacts declared `required: false`.

Real evaluation conditions can also make an otherwise valid declared metric or diagnostic output genuinely unavailable for one concrete Run. Examples include a metric whose denominator has no qualifying samples, a best-operating-point metric whose qualification condition has no matching point, or a diagnostic artifact that cannot be produced for the evaluated sample composition.

Treating every such case as a failed Evaluation Run would discard useful valid results and conflate an incomplete but successfully executed evaluation with evaluator failure or contract corruption.

At the same time, silently omitting declared metrics would weaken the formal result contract. MLDB therefore needs an explicit representation for legitimate output unavailability and a terminal state that distinguishes complete success from usable partial completion.

## Decision

Add `completed_partial` as a terminal Evaluation Run status.

Evaluation Run v1 statuses become:

```text
running
completed
completed_partial
failed
cancelled
```

`completed` means evaluation execution succeeded and its formal result contains every declared scalar metric plus every required structured artifact, with no returned optional formal artifact rejected during validation.

`completed_partial` means evaluation execution succeeded and the Run retains valid formal results, but one or more declared scalar metrics or optional formal artifacts are unavailable or were rejected without invalidating the primary evaluation execution.

`failed` remains reserved for execution failure or a contract violation that prevents the formal result from being trusted.

### Explicit unavailable-output reporting

Extend the conceptual Evaluation Result contract with explicit unavailable-output reports:

```python
@dataclass(frozen=True)
class UnavailableOutput:
    output: str
    type: str
    message: str

@dataclass(frozen=True)
class EvaluationResult:
    metrics: Mapping[str, int | float]
    artifacts: Mapping[str, Path]
    unavailable_outputs: Sequence[UnavailableOutput]
```

The concrete runtime classes may differ, but the semantic fields are fixed.

`output` identifies exactly one declared formal output using:

```text
metrics.<metric-key>
artifacts.<artifact-key>
```

`type` is a non-empty short machine-readable reason string supplied by the Evaluation Protocol. MLDB v1 does not define a universal reason enum.

`message` is a concise human-readable explanation.

An output must not be both returned and reported unavailable.

Unknown or undeclared output references are contract violations and fail the Evaluation Run.

### Metric presence semantics

Every scalar metric declared by the sealed Evaluation Protocol must have exactly one outcome in the returned Evaluation Result:

- a valid finite numeric value under `metrics`; or
- one explicit entry under `unavailable_outputs`.

A declared metric that is neither returned nor explicitly reported unavailable is an invalid silent omission and fails the Evaluation Run.

A metric reported unavailable produces `completed_partial` when all remaining completion-critical contracts succeed.

A metric that is returned but has an invalid value such as boolean, string, NaN, or infinity remains a contract violation and fails the Evaluation Run. Invalid returned values must not be converted into unavailability by generic runtime tooling.

### Structured artifact semantics

Artifacts declared `required: true` remain completion-critical.

A required artifact that is missing, explicitly unavailable, or returned but invalid fails the Evaluation Run.

Artifacts declared `required: false` remain optional in the declaration contract.

Their outcomes are:

| optional artifact outcome | Evaluation Run consequence |
|---|---|
| omitted without an unavailable report | Valid omission; does not make the Run partial. |
| explicitly reported unavailable | `completed_partial` when all completion-critical outputs are valid. |
| returned and valid | Import and record normally. |
| returned but invalid | Reject the artifact, record `validation_issues`, and finish `completed_partial` when all completion-critical outputs are valid. |

This preserves the distinction between an artifact that the protocol simply does not choose to emit for a Run and one that was expected by the concrete execution but could not be produced or validated successfully.

### Partial result persistence

Both `completed` and `completed_partial` Evaluation Runs persist accepted scalar metrics and accepted formal artifact metadata under `result`.

A `completed_partial` Run additionally records the reason for incompleteness through at least one of:

- `unavailable_outputs`;
- `validation_issues` describing a rejected returned optional formal artifact.

A `completed_partial` Run must retain at least one accepted formal metric or accepted formal artifact. If no trustworthy formal output remains, the Evaluation Run is `failed` rather than `completed_partial`.

A `completed` Run must not contain an unresolved declared metric and must not have a returned optional formal artifact rejected from the formal result.

### Failure precedence

`completed_partial` is not a recovery state for arbitrary validation errors.

The Run remains `failed` when any failure condition applies, including:

- executable loading, context construction, or `evaluate(context)` failure;
- undeclared returned metric or artifact keys;
- undeclared unavailable-output references;
- a declared metric silently omitted without an unavailable report;
- an invalid returned scalar metric;
- a required artifact missing, unavailable, or invalid;
- failure to import or persist completion-critical formal result material.

When both a partial condition and a failure condition occur, `failed` takes precedence.

### Study Run interpretation

`completed_partial` is terminal child work and does not block sibling evaluations or unrelated trials.

A Study Run that otherwise processes its complete plan normally but contains one or more `completed_partial` child Evaluation Runs terminates as `completed_with_failures` rather than `completed`.

Study Run summary information may count `completed_partial` Evaluation Runs separately from fully `completed`, `failed`, `cancelled`, and blocked evaluation work.

## Rationale

A separate partial-completion state preserves useful measurements without pretending that a Run satisfied its complete formal result surface.

Explicit unavailable-output reporting keeps the result contract machine-checkable. Generic tooling can distinguish a protocol-authorized inability to compute one output from a programming error that silently forgot a declared metric.

Keeping invalid returned metric values as failures avoids masking implementation defects. A protocol that knows a metric is not computable should report that fact explicitly rather than emit NaN or omit the key.

Required artifacts remain strict because their declaration promises that successful evaluation includes them. Optional artifacts retain a lighter contract while still surfacing concrete production or validation problems through `completed_partial`.

The Study-level interpretation makes large sweeps resilient while preserving the distinction between complete and incomplete evaluation coverage.

## Rejected alternatives

### Treat every declared metric as unconditionally required and fail when one cannot be computed

This would discard otherwise valid evaluation results for legitimate data-dependent edge cases and overstate the severity of partial output availability.

### Permit silent omission of declared metrics

This would make an implementation bug indistinguishable from intentional unavailability and would weaken automated comparison of Evaluation Runs.

### Encode unavailable metrics as NaN

MLDB formal scalar metrics are finite numeric values. NaN also propagates poorly through sorting, dashboards, SQL, and external sinks and does not carry a reason.

### Add `required` to every metric declaration

The current need is to distinguish complete from partial concrete results, not to add another authored requiredness dimension. Every declared metric remains part of the protocol's formal surface and must be either returned or explicitly unavailable.

### Treat optional artifact validation failure as full completion

A returned artifact that fails its declared schema is materially different from a valid intentional omission. Marking the Run partial preserves that distinction without discarding valid primary results.

## Consequences

Evaluation Protocol tooling must support explicit unavailable-output reports in Evaluation Result.

Evaluation result validation must enforce exactly-one metric outcome, required-artifact strictness, optional-artifact partial semantics, and failure precedence.

Evaluation Run format and lifecycle must recognize `completed_partial` as terminal and persist accepted partial results plus incompleteness reasons.

Study Run lifecycle and optional summary counts must treat partial child evaluations as non-complete child outcomes without cancelling independent work.

External visualization and comparison tooling can distinguish fully comparable Evaluation Runs from Runs whose formal result surface is incomplete.

## Evidence

Existing project evaluations include data-dependent values such as qualification-based best-threshold summaries and class-conditional extrema that may legitimately be undefined for some concrete evaluation populations. Existing detector evaluation also produces rich optional diagnostics separately from primary scalar metrics. These cases benefit from preserving valid results while explicitly marking incomplete formal output coverage.
