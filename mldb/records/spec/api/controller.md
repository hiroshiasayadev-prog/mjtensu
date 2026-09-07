# Contract: Controller application interface

- **id**: `spec:mldb.api.controller`
- **status**: draft
- **date**: 2026-09-04
- **parent**: `spec:mldb.api`
- **contract_class**: `interface`

## What this is

Defines the public MLDB application operations exposed by Controller.

The interface is transport-independent. CLI, local Python callers, and future HTTP/UI adapters must preserve these operation semantics rather than implement independent MLDB domain behavior.

## Request

V1 public operations are:

| operation | purpose |
|---|---|
| `validate_definition` | Read-only validation of one reusable MLDB definition. |
| `seal_definition` | Perform the lifecycle-specific sealing gate for one sealable definition. |
| `execute_study` | Validate and launch one sealed Study as a new Study Run. |
| `get_study_run` | Read canonical Study Run state plus derived execution progress. |
| `cancel_study_run` | Request idempotent cancellation of one running Study Run. |
| `get_entity` | Read one MLDB entity or execution record through typed runtime resolution. |
| `list_entities` | List discoverable entities of one requested kind. |

### `validate_definition`

Request fields:

| field | contract |
|---|---|
| `kind` | Reusable definition kind: Task, Corpus, Architecture, Train Protocol, Evaluation Protocol, or Study. |
| `id` | Exact MLDB definition ID. |

Validation is read-only and may be applied to draft or immutable/sealed definitions where their lifecycle permits resolution.

It uses the normal runtime schema, reference, compatibility, parameter-interface, lifecycle, and integrity checks applicable to that kind.

### `seal_definition`

Request fields:

| field | contract |
|---|---|
| `kind` | One sealable definition kind: Architecture, Train Protocol, Evaluation Protocol, or Study. |
| `id` | Exact draft definition ID. |

Architecture, Train Protocol, and Evaluation Protocol sealing invokes the asset-specific pytest gate required by `spec:mldb.verification.executable_asset_tests` and persists the required implementation hash.

Study sealing applies Study validation and lifecycle rules but does not invent an executable-asset pytest requirement.

Task and Corpus are not accepted by this operation because their current lifecycle contracts do not use this sealing transition.

### `execute_study`

Request fields:

| field | contract |
|---|---|
| `study_id` | Exact sealed Study ID to execute. |

Controller performs complete Study execution validation before Study Run allocation.

After successful validation, Controller allocates one Study Run, materializes its complete immutable plan, and admits the plan-derived Training/Evaluation jobs to Queue.

The operation returns after successful plan materialization and Queue admission. It does not wait for child training or evaluation completion.

### `get_study_run`

Request fields:

| field | contract |
|---|---|
| `study_run_id` | Exact Study Run event ID. |

The operation reads the canonical Study Run and derives current progress from its immutable plan, canonical child Runs/Models, and Queue state where operational progress is useful.

### `cancel_study_run`

Request fields:

| field | contract |
|---|---|
| `study_run_id` | Exact Study Run event ID. |

Cancellation is a Study-level intent. The caller does not identify Queue jobs, attempts, Workers, or child Run IDs to cancel.

### `get_entity`

Request fields:

| field | contract |
|---|---|
| `kind` | Typed MLDB entity kind. |
| `id` | Exact entity ID. |

The operation uses canonical runtime resolution and does not scan unrelated entity directories as fallback.

### `list_entities`

Request fields:

| field | contract |
|---|---|
| `kind` | Typed MLDB entity kind to list. |

V1 listing may be limited to deterministic discovery and compact metadata needed to select an entity. Rich filtering and ranking are not required.

## Response

### Validation response

`validate_definition` returns:

| field | contract |
|---|---|
| `kind` | Validated definition kind. |
| `id` | Validated definition ID. |
| `valid` | Boolean validation result. |
| `issues` | Ordered validation issues; empty when valid. |

Each issue contains at least a stable machine-readable `code` and concise human-readable `message`. A field/reference path may be included when applicable.

A validation failure is a normal validation result and does not mutate the definition.

### Sealing response

`seal_definition` returns:

| field | contract |
|---|---|
| `kind` | Sealed definition kind. |
| `id` | Exact definition ID. |
| `status` | `sealed`. |
| `result` | `sealed` or `already_sealed`. |

A repeated sealing request for an already sealed definition may return `already_sealed` only when the persisted sealed definition still satisfies its applicable integrity contract. It must not rewrite a sealed asset to make it match current bytes.

### Study execution response

Successful `execute_study` returns:

| field | contract |
|---|---|
| `study_run_id` | Newly allocated Study Run ID. |
| `study_id` | Executed sealed Study ID. |
| `status` | Initial canonical Study Run status, normally `running`. |

This response means a complete immutable plan was established and Queue admission succeeded. It does not mean any Training or Evaluation job has completed.

If failure occurs after Study Run allocation, the definitive error response must include the allocated `study_run_id` so the resulting failed/cancelled Study Run remains observable.

### Study Run status response

`get_study_run` returns the canonical Study Run identity/status plus derived progress.

The progress view uses stable application categories rather than exposing callers to raw Queue table rows:

| progress field | meaning |
|---|---|
| `total` | Number of planned coordinates of that execution kind. |
| `satisfied` | Coordinates fully satisfied by canonical accepted child results. |
| `active` | Coordinates with one current Worker execution attempt. |
| `waiting` | Runnable or retry-delayed coordinates not currently active. |
| `blocked` | Coordinates waiting for an upstream Model dependency. |
| `unsatisfied_terminal` | Coordinates that will not receive another attempt in the current Study Run. |

A training-derived Study reports training and evaluation progress. An existing-Model Study has no planned training coordinates and may omit the training progress section or report total zero consistently.

The response may additionally expose incomplete coordinate references (`trial`, and `stage` for evaluation) and their latest child Run identity/status for diagnostics.

Derived progress is observational convenience and must not override canonical Study Run or child Run history.

### Cancellation response

`cancel_study_run` returns one of:

```text
accepted
already_terminal
```

`accepted` means Controller has recorded/propagated cancellation intent so no new work should be started for the Study Run and active attempts will receive cooperative cancellation through orchestration.

`accepted` does not require every active Worker to have stopped before the response is returned.

`already_terminal` is idempotent and must not reopen or rewrite the terminal Study Run.

### Entity responses

`get_entity` returns the canonical parsed entity/record representation appropriate to the requested kind.

`list_entities` returns a deterministic collection of entity summaries containing at least `kind` and `id`, plus lifecycle/status metadata when that entity kind has such a field.

Read operations must not mutate canonical state.

## Errors

The public application interface distinguishes validation/domain errors from transport-adapter errors.

Transport-specific HTTP status codes are outside this contract.

| error | meaning |
|---|---|
| `not_found` | Requested typed entity or Study Run does not exist. |
| `invalid_request` | Required request shape or ID/kind combination is invalid. |
| `validation_failed` | A mutating operation such as sealing or Study execution cannot proceed because its domain validation failed. |
| `unsupported_operation` | Requested operation is not defined for that entity kind, such as sealing a Task. |
| `lifecycle_conflict` | Requested mutation conflicts with immutable/terminal lifecycle state. |
| `execution_setup_failed` | Study execution failed after allocation or during plan/Queue setup; includes `study_run_id` when one was allocated. |
| `internal_failure` | Unexpected Controller/runtime failure that is not a normal MLDB validation result. |

`validate_definition` reports ordinary definition invalidity through its `valid=false` response rather than converting every validation issue into an exceptional API failure.

`execute_study` must not allocate a Study Run when complete pre-allocation Study validation fails.

A failed `execute_study` after Study Run allocation must preserve that allocated Study Run according to its lifecycle contract rather than deleting it to make the API appear atomic.

## Rules

### Single domain implementation

Controller operations must delegate to shared MLDB runtime/domain services.

CLI, HTTP handlers, UI adapters, and scripts must not maintain separate schema validation, parameter resolution, sealing, Study planning, Run lifecycle, or Queue-admission semantics.

### Public versus internal API

Normal application callers do not directly invoke:

- Worker acquire;
- Worker heartbeat;
- immutable Worker asset retrieval;
- candidate upload;
- attempt outcome reporting;
- Queue job mutation;
- Queue reconciliation internals.

Those operations remain behind `spec:mldb.orchestration`.

### Authoring boundary

V1 definition authoring remains file-based.

The lack of generic definition CRUD must not block validation, sealing, Study execution, or entity reads.

### Initial usability boundary

The Controller application interface is considered sufficient for the first real MLDB experiment workflow when repository-authored definitions can be:

```text
validate
  -> seal where required
  -> execute Study
  -> observe Study Run progress/results
  -> cancel if needed
```

Richer analytics, definition CRUD, MLflow integration, and remote-transport polish are additive later features rather than prerequisites for this workflow.

## Related specs

| ref | relation |
|---|---|
| `spec:mldb.api` | Parent public API overview. |
| `spec:mldb.runtime` | Supplies shared validation, resolution, and execution behavior. |
| `spec:mldb.verification.executable_asset_tests` | Supplies executable-definition sealing verification. |
| `spec:mldb.study` | Defines Study and Study Run semantics used by execution/status/cancellation. |
| `spec:mldb.orchestration` | Executes Study-derived work behind this public boundary. |
| `spec:mldb.orchestration.worker_api` | Internal Worker API intentionally excluded from normal application callers. |