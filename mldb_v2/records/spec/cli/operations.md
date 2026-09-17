# Contract: CLI operations

- **id**: `spec:mldb.v2.cli.operations`
- **status**: draft
- **date**: 2026-09-17
- **parent**: `spec:mldb.v2.cli`
- **contract_class**: `api`

## Supported entrypoint

Normal users invoke an installed repository command named `mldb`. Direct execution of implementation
`.py` files is not the supported experiment workflow.

The CLI follows `spec:mldb.v2.cli.selectors_output` and delegates semantics to
`spec:mldb.v2.api.application_interface`.

## Discover and inspect

```text
mldb ps [--all] [selectors]
mldb get <resource> [typed-id] [selectors]
mldb describe <resource> <typed-id>
mldb status [study-result-ref]
```

`ps` is the common execution view. Without `--all` it shows active/non-terminal Study Results; with
`--all` it includes terminal recent history subject to selectors/limit.
`get` lists when ID is omitted and returns one canonical object when ID is supplied. Initial resource
names include `namespaces`, `definitions`, `tasks`, `corpora`, `architectures`, `train-protocols`,
`evaluation-protocols`, `studies`, `plans`, `runs`, `training-results`, `models`, and
`evaluation-results`. `definitions` is an aggregate read-only view over reusable definitions.

`describe` gives a human-oriented detailed view of one exact object and may include resolved lineage
or references; it is read-only.

`status` without a Study Result ref returns the same active execution summary class as `ps`.
With a ref it returns detailed canonical progress plus current backend observational state.

## Authoring checks

```text
mldb validate [<kind> [typed-id]] [--namespace <namespace>]
mldb verify   [<kind> [typed-id]] [--namespace <namespace>]
mldb seal <kind> <typed-id>
mldb seal [<kind>] --namespace <namespace> --all
mldb plan <study-ref>
```

`validate` with no kind/ID validates repository v2 structure and every discoverable v2 definition.
`verify` with no kind/ID runs the complete verification gate for every sealable v2 definition.
A Namespace selector narrows either command; a kind narrows to that definition kind; an exact ID
narrows to one target.
Kind spellings are `task`, `corpus`, `architecture`, `train-protocol`, `evaluation-protocol`, and
`study`. Kind is never guessed from a typed ID by scanning domains.

`seal` is mutating. One target is explicit by default; multi-target sealing requires `--all` and an
explicit selector. `plan` requires one Study because execution intent must never be selected
implicitly.

## Execute and control

```text
mldb run <study-ref> [--backend <name>]
mldb resume <study-result-ref>
mldb rerun <study-result-ref> [--backend <name>]
mldb cancel <study-result-ref>
mldb advance <study-result-ref>
```

`run` compiles/plans one sealed Study, creates a fresh Study Result, prints its identity as soon as it
is durable, creates/recovers the selected backend Study execution, and reconciles canonical results
until terminal.

`resume` reconnects to one existing non-terminal Study Result and its same backend Study execution;
it never creates a new formal execution. Interrupting `run`/`resume` stops local reconciliation only
and does not request backend Pipeline cancellation.
`rerun` creates a fresh Study Result from the exact immutable Plan referenced by the source execution;
it does not recompile the current mutable Study definition. `--backend` may select a new backend for
that fresh execution.

`cancel` records canonical cancellation intent and requests cancellation of the backend Study execution plus active child work.
`advance` performs exactly one idempotent canonical reconciliation pass and is intended for recovery,
automation, and debugging rather than normal interactive control.

## Monitor and investigate

```text
mldb watch [study-result-ref] [selectors]
mldb logs <study-result-ref> [--trial <trial>] [--stage <coordinate>] [--failed] [-f]
mldb doctor
```

`watch` is strictly read-only. Without a ref it refreshes the active execution summary; with a ref it
refreshes detailed progress for that execution. It MUST NOT call `advance_study`, admit backend work,
accept results, or otherwise progress the Study.

`logs` is a read-only backend projection. It may narrow by trial/stage or failed attempts and may
follow live output when the backend supports it. Logs are not canonical MLDB history.

`doctor` checks repository/root resolution, Git/source-pinning prerequisites, configured backend
reachability/capabilities, and object-store connectivity/integrity prerequisites without mutating
canonical MLDB records.
## Output and backend configuration

All applicable read/list/check commands support the structured output contract. Human-facing table
columns may evolve, but structured field semantics belong to the application API.

New executions use configured default backend when `--backend` is omitted. Resume/status/watch/
cancel/logs derive backend type from persisted Study Result and do not require the user to repeat it.
Backend endpoint, credentials, queue, worker, and UI configuration remain runtime configuration.

The CLI never auto-commits Git changes and never mutates ClearML operational entities outside the
backend actions required by MLDB execution/cancellation.
