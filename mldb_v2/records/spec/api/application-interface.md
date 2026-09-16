# Contract: Application interface

- **id**: `spec:mldb.v2.api.application_interface`
- **status**: draft
- **date**: 2026-09-09
- **parent**: `spec:mldb.v2.api`
- **contract_class**: `interface`

## Public mutation/check operations

| operation | request | response responsibility |
|---|---|---|
| `validate_scope` | optional kind/namespace/exact ref selector | ordered validation report; read-only |
| `verify_scope` | optional kind/namespace/exact ref selector | complete verification report; read-only |
| `seal_scope` | exact target or explicit bulk selector | sealed definition summaries |
| `plan_study` | sealed Study ref | immutable Study Plan identity/summary |
| `start_study` | Plan ref + backend name + UUID4 execution key | idempotently allocated Study Result identity |
| `advance_study` | Study Result ref | one idempotent progression-pass summary |
| `run_study` | sealed Study ref + backend name | fresh execution driven to terminal |
| `resume_study` | non-terminal Study Result ref | existing execution driven to terminal |
| `rerun_study` | Study Result ref + optional backend | fresh execution from the exact referenced Plan |
| `cancel_study` | Study Result ref | cancellation request outcome |

Read/discovery operations are defined by `spec:mldb.v2.api.query_interface`.
## Scope semantics

Empty `validate_scope` means repository v2 structure plus every discoverable v2 definition. Empty
`verify_scope` means every sealable v2 definition. Namespace and kind selectors narrow those sets;
an exact ref selects one target. Bulk reports continue through failures by default.

`seal_scope` never interprets omitted target as implicit bulk mutation. More than one resolved target
requires explicit bulk intent from the adapter request.

## Public report shapes

`validate_scope` and `verify_scope` return the same ordered report shape:

```yaml
items:
  - kind: architecture
    id: rotated-fcos/stem-s1-v1
    valid: true
    diagnostics: []
repository_issues: []
```

Each item has exactly `kind`, typed `id`, boolean `valid`, and ordered bounded diagnostics.
`repository_issues` contains structural inventory issues not owned by one valid typed definition and
is empty when none exist. The selected order follows the canonical repository listing order.

`seal_scope` returns ordered per-target items with exactly `kind`, typed `id`, boolean `sealed`, and
ordered diagnostics. Bulk sealing is not a cross-file transaction: targets are processed in canonical
order, a failed target is reported without rolling back already-sealed earlier targets, and no failed
target is rewritten as sealed. Single-target lifecycle conflicts may still surface through the public
error model.

`plan_study` returns the exact immutable `StudyPlan` value it created or found idempotently.
`start_study`, `run_study`, `resume_study`, and `rerun_study` return the exact canonical `StudyResult`
value at their documented return point; `run`/`resume`/`rerun` therefore return terminal values.

`cancel_study` returns exactly:

```yaml
study_result: rotated-fcos/run-...
outcome: accepted
status: cancelling
```

`outcome` is `accepted`, `already_cancelling`, or `already_terminal`; `status` is the canonical
Study Result status after the cancellation request handling performed by that call.

## Execution semantics

`run_study` creates a fresh execution key, plans the current sealed Study, starts that Plan, and
repeatedly invokes the same progression primitive until terminal.

`resume_study` operates on one existing Study Result and creates no new execution identity.
`rerun_study` starts a fresh execution from the exact immutable Plan referenced by the source Study
Result; it does not recompile the current Study definition.

`start_study` is an application primitive, not necessarily a normal human CLI command. It returns
after the Study Result is durably persisted; initial backend admission occurs only through
`advance_study`.
## Progression response

`advance_study` response has exactly these semantic fields: `study_result`, canonical `status`,
boolean `changed`, ordered `admitted` stage keys, ordered `finalized_results` refs, ordered
`finalized_models` refs, ordered currently `active` stage keys observed during that pass, and boolean
`terminal`. Concrete Python container classes are frozen by Skeleton. `active` is observational
convenience and not canonical history.

## Side-effect boundaries

Validation/query/verification operations are read-only; verification may execute definition tests.
`seal_scope` mutates only selected draft definition lifecycle/integrity metadata. `plan_study` writes
only an immutable Plan. `advance_study` is the sole normal progression mutation primitive after
Study Result allocation.

`run_study` and `resume_study` are loops over existing primitives, not alternate state machines.
`cancel_study` changes `submitted` to `cancelling` idempotently and delegates active-work
cancellation through the backend port; later advancement performs collection/terminal closure.

## Failure and authority

Validation failure before Study Result allocation creates no execution history. After allocation,
setup/backend/acceptance failures remain observable through that Study Result rather than deleting it.
Public failures follow `spec:mldb.v2.api.errors`.

Canonical MLDB records are authoritative; backend state informs progression/observation only and
never directly rewrites Plan semantics or formal results.
