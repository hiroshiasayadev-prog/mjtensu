# Contract: Worker API

- **id**: `spec:mldb.orchestration.worker_api`
- **status**: draft
- **date**: 2026-09-04
- **parent**: `spec:mldb.orchestration`
- **contract_class**: `interface`

## What this is

Defines the logical Controller/Worker interface for acquiring one Training or Evaluation attempt, maintaining its lease, obtaining immutable inputs, uploading candidate artifacts, and reporting the domain outcome.

The interface is pull-oriented. Worker initiates normal communication with Controller.

Exact HTTP routes, authentication, streaming protocol, and concrete JSON serialization are outside this contract.

## Request

### Common communication rule

Every logical Worker API operation distinguishes retryable communication failure from definitive Controller response and domain execution failure.

For retryable communication failure, Worker retries the same logical operation after exactly 10 seconds and continues without a retry-count limit.

The logical operation identity must remain stable across such retries where replay could otherwise duplicate Controller state.

Intentional Worker process shutdown may end the retry loop.

### Acquire work

Worker requests one assignment and supplies:

| field | contract |
|---|---|
| `worker_id` | Non-empty Worker-process identity used for operational attribution. |
| `acquire_token` | Opaque Worker-generated token for one logical acquire operation. Reused unchanged for communication retry of that operation. |
| `accepts` | Non-empty set drawn from `training`, `evaluation`. |

Worker must not request another independent assignment while it still owns an active attempt.

Controller uses the request to select compatible `ready` work, perform concrete preflight, allocate the child Run, create the Queue attempt, and issue a lease.

### Heartbeat

Worker heartbeat supplies:

| field | contract |
|---|---|
| `attempt_id` | Queue attempt identity assigned by Controller. |
| `lease_token` | Opaque current lease authorization for that attempt. |

Heartbeat continues while the attempt remains Worker-owned, including input retrieval, domain execution, candidate preparation/upload, and outcome-acceptance waiting.

### Retrieve immutable asset

Worker requests an immutable object using a Controller-provided asset descriptor.

The descriptor must identify the expected object and all integrity information required by the underlying asset contract.

Worker must not substitute an independently discovered mutable source for the assigned immutable object.

### Upload candidate artifact

Candidate upload identifies:

| field | contract |
|---|---|
| `attempt_id` | Owning attempt. |
| `lease_token` | Current lease authorization. |
| `key` | Attempt-local declared candidate key. |
| `content_identity` | Integrity identity such as SHA-256 when supplied/required by the result contract. |
| bytes | Candidate content. |

The exact transport for large bytes is deferred.

### Report attempt outcome

Worker reports one intended outcome:

```text
success
failed
cancelled
```

Every outcome request identifies the concrete attempt and lease.

Training `success` supplies the generic trained-result candidate metadata required for Controller to validate and accept canonical learned weights.

Evaluation `success` supplies the returned formal metric mapping, artifact candidate references, and explicit unavailable-output records required by the Evaluation result contracts.

`failed` supplies concise failure information describing the domain or execution-local failure.

`cancelled` reports cooperative termination rather than domain success.

Worker must not invoke `train()` or `evaluate()` again for the same attempt after reporting or preparing a `failed` domain outcome.

## Response

### Acquire response

Controller returns exactly one of:

```text
assignment
no_work
definitive_rejection
```

`no_work` is a successful API response and is not a communication failure.

An `assignment` contains:

| field | contract |
|---|---|
| `attempt_id` | Concrete Queue attempt identity. |
| `run_id` | Concrete Training Run or Evaluation Run ID. |
| `lease_token` | Opaque authorization for the active attempt. |
| `lease_until` | Current lease-expiry timestamp under the Queue contract. |
| `kind` | `training` or `evaluation`. |
| resolved input | Complete identity/value surface needed for that execution kind. |
| asset descriptors | Immutable input descriptors required by Worker. |

A repeated acquire request using an `acquire_token` whose assignment was already committed must return that same still-authorized assignment rather than create another Run.

A Training assignment resolves at least Task, Corpus, Architecture, Train Protocol, seed, complete public parameters, and immutable descriptors for the Corpus artifact plus selected Architecture and Train Protocol implementation bytes.

An Evaluation assignment resolves at least Task, Corpus, Model, the Model's validated completed Training Run lineage and selected Architecture, Evaluation Protocol, complete public parameters, and immutable descriptors for the Corpus artifact, Model weights, selected Architecture implementation, and Evaluation Protocol implementation bytes.

Assignments carry validated metadata values rather than Controller repository paths. After retrieving and verifying the exact descriptor bytes, Worker may materialize required execution files locally and construct the same runtime Handle types used by the existing `TrainContext`, `EvaluationContext`, executable loaders, and Model loader. Worker-local paths are not canonical repository locations; unavailable metadata provenance remains absent rather than synthesized.

The Corpus builder is not an assignment input or immutable descriptor. Worker execution consumes the registered immutable Corpus artifact and must not require the authoring/materialization builder merely to satisfy runtime Handle shape.

Study lineage may be returned for observability but does not grant Worker scheduling authority.

### Heartbeat response

A valid heartbeat returns current lease/control information including:

| field | contract |
|---|---|
| `lease_until` | Current or extended lease expiry. |
| `cancel_requested` | Whether Worker should cooperatively stop the attempt. |

A definitive stale/superseded-lease response ends ordinary heartbeat retry for that authorization.

Each successfully accepted heartbeat resets the active-attempt inactivity deadline to one hour from that accepted heartbeat. The assignment establishes the initial one-hour deadline until the first heartbeat is accepted.

### Asset response

Controller returns the immutable bytes corresponding to the descriptor or a definitive rejection indicating that the assignment cannot be fulfilled as requested.

Worker verifies every supplied SHA-256 and byte-count requirement before use or cache reuse. Only verified exact bytes may be materialized at Worker-local execution paths used by runtime Handles; successful transfer alone is insufficient.

### Candidate-upload response

Controller returns one of:

```text
accepted
already_present
rejected
```

`already_present` is valid only when the existing candidate for that attempt/key has the same required content identity.

A conflicting candidate for the same accepted attempt/key must be rejected rather than overwritten.

### Outcome response

Controller returns a definitive acknowledgement that describes whether the attempt outcome was:

```text
accepted
already_finalized
rejected
```

`accepted` means Controller has applied the applicable canonical Run/result contract and established the resulting child Run terminal state.

`already_finalized` supports idempotent replay after the original acknowledgement was lost. It must not create another child Run or duplicate canonical artifact/result state.

`rejected` is definitive for that reported operation, for example because the lease is stale and Controller has not already accepted that attempt result, or because the candidate violates the formal contract.

Worker success does not itself mean Queue job `satisfied`; Controller's canonical acceptance remains authoritative.

## Errors

### Retryable communication failure

The following semantic class is retryable indefinitely by Worker:

```text
no definitive valid Controller response was obtained for the logical operation
```

Examples include Controller unreachability, connection timeout/reset, lost response, and transient Controller service failure.

Required behavior:

1. preserve the current logical operation and any required local candidate/outcome state;
2. wait 10 seconds;
3. retry the same logical operation;
4. repeat without a retry-count limit until a definitive Controller response is obtained or Worker is intentionally shut down.

Communication retry must not independently change Training Run, Evaluation Run, or Queue job outcome.

### Domain execution failure

A failure of `train()`, `evaluate()`, or execution-local result preparation is not a communication failure.

Worker must:

1. stop that domain invocation;
2. preserve concise failure information;
3. report `failed` for the current attempt;
4. if reporting communication fails, retry only the outcome-report operation every 10 seconds;
5. never rerun the domain function under the same attempt/Run identity.

Controller/Queue alone decides whether the logical job receives a new attempt and new child Run ID.

### Lease loss during communication outage

Communication retry does not guarantee that Controller will keep a lease valid forever. Controller may invalidate the current authorization after one continuous hour without a successfully accepted heartbeat.

If Controller later definitively reports that the lease is stale or superseded, Worker must stop treating the attempt as authoritative and must not force candidate or outcome acceptance.

A late duplicate outcome may still receive `already_finalized` when Controller had actually accepted that exact attempt before the acknowledgement was lost.

### Integrity failure

A downloaded immutable object whose required integrity identity does not match is not a retryable communication failure merely because transfer completed.

Worker must not use the object. It reports an execution/assignment failure through the attempt outcome path unless Controller explicitly provides a different definitive recovery instruction.

### Protocol incompatibility

Unsupported Worker API version, incompatible schema, or another deterministic protocol mismatch is not subject to infinite 10-second communication retry.

Worker must surface the incompatibility as a fatal configuration/software error rather than repeatedly issuing an operation that cannot become valid without intervention.

## Rules

### Idempotent acquire

`acquire_token` exists specifically to handle this sequence:

```text
Controller allocates Run + Queue attempt
  -> Controller commits assignment
  -> response is lost
  -> Worker retries acquire with same acquire_token
  -> Controller returns same assignment
```

The same token must not create two child Runs.

Queue storage must persist enough acquisition identity to implement this replay rule.

### Idempotent result delivery

Worker retains candidate/result state until Controller gives a definitive outcome acknowledgement.

This sequence must be safe:

```text
Controller accepts result + terminalizes Run
  -> acknowledgement is lost
  -> Worker retries same outcome
  -> Controller returns already_finalized
```

A terminal Run is never reopened or duplicated to satisfy replay.

### Communication versus domain retry

| situation | retry behavior |
|---|---|
| Controller cannot be reached during acquire | Retry acquire every 10 seconds indefinitely using same acquire token. |
| Controller returns `no_work` | Acquire completed normally; future polling is a scheduler/polling concern, not communication retry. |
| Heartbeat cannot reach Controller | Retry heartbeat communication every 10 seconds; do not mark domain execution failed solely for that reason. If no heartbeat is successfully accepted for one continuous hour, Controller may expire the lease. |
| Asset transfer cannot reach Controller | Retry the same asset retrieval every 10 seconds. |
| `train()` raises | Do not call `train()` again; report failed attempt. |
| Failure report cannot reach Controller | Retry the same failure report every 10 seconds. |
| Successful result report response is lost | Retry the same result report; Controller returns accepted/already-finalized semantics. |
| Controller says lease stale | Definitive response; stop ordinary retry under that lease. |
| API/schema version incompatible | Fatal compatibility error; do not infinite-retry. |

### Cancellation

Communication failure is not cancellation.

Controller may request cancellation through a successful heartbeat/control response.

Worker should cooperatively stop the domain execution when feasible and report `cancelled`. Communication failure while reporting cancellation follows the same infinite 10-second communication retry rule.

### Local state retention

Worker must retain enough attempt-local state to retry communication without repeating completed domain work while the process remains alive.

Durable Worker-local recovery across Worker process crash is not required by this contract. Worker crash remains subject to Queue lease-loss and new-attempt semantics.

## Related specs

| ref | relation |
|---|---|
| `spec:mldb.orchestration` | Parent orchestration overview. |
| `spec:mldb.orchestration.responsibility_model` | Defines Controller/Worker authority split. |
| `spec:mldb.orchestration.job_model` | Defines Training/Evaluation work units assigned through this API. |
| `spec:mldb.orchestration.queue_lifecycle` | Defines active lease, stale-attempt, retry, and satisfaction semantics. |
| `spec:mldb.orchestration.queue_storage_format` | Persists attempts and leases used by this interface. |
| `spec:mldb.training.train_interface` | Defines Worker-side Training invocation. |
| `spec:mldb.evaluation.evaluate_interface` | Defines Worker-side Evaluation invocation and returned result shape. |
