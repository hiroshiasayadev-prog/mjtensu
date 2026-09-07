# MLDB-ADR-SCHEMA-012: Define preflight validation before Run allocation

- **status**: accepted
- **date**: 2026-09-04
- **depends_on**: MLDB-ADR-SCHEMA-005, MLDB-ADR-SCHEMA-008
- **supersedes**:
- **migrated_to_spec**: 2026-09-04

## Context

MLDB-ADR-SCHEMA-005 and MLDB-ADR-SCHEMA-008 require Training Run and Evaluation Run records to contain their exact selected inputs and complete resolved public-parameter mappings.

Those ADRs also describe launch sequences that allocate a `running` Run before asset resolution, compatibility checks, and public-parameter resolution complete.

These two requirements conflict. A request can fail because an asset ID is invalid, an asset is incompatible, an integrity check fails, or a caller supplies an unknown public parameter. In those cases MLDB cannot create a schema-valid `running` Run containing the complete resolved inputs required by the Run record contract.

Persisting a temporarily incomplete Run would also make crash recovery ambiguous because a leftover `running` record could be structurally invalid rather than representing a real execution attempt.

MLDB therefore needs an explicit preflight boundary separating invalid launch requests from recorded execution attempts.

## Decision

Training Run and Evaluation Run allocation occurs only after preflight validation succeeds.

Preflight for a direct Training Run resolves the selected Corpus, Architecture, and Train Protocol, validates required static compatibility and integrity, and resolves the complete Train Protocol public-parameter mapping.

Preflight for a direct Evaluation Run resolves the selected Model, Corpus, and Evaluation Protocol, validates required static compatibility and integrity, and resolves the complete Evaluation Protocol public-parameter mapping.

A preflight failure rejects the launch request and does not allocate a Training Run or Evaluation Run.

After preflight succeeds, MLDB allocates the Run ID, creates the Run directory and schema-valid `run.yaml` with `status: running`, records the complete selected inputs and resolved parameters, and then begins execution setup and protocol invocation.

Executable implementation loading, context construction, protocol invocation, returned-result validation, formal artifact persistence, and canonical-result serialization occur after Run allocation. Failure in those stages finalizes the allocated Run as `failed` unless cancellation semantics apply.

The conceptual Training Run flow is:

```text
launch request
  -> resolve Corpus / Architecture / Train Protocol
  -> validate static compatibility and integrity
  -> resolve complete public parameters
  -> allocate Training Run ID
  -> create schema-valid run.yaml: running
  -> load executable implementation and construct TrainContext
  -> invoke train(context)
  -> validate and materialize canonical result
  -> completed / failed / cancelled
```

The conceptual Evaluation Run flow is:

```text
launch request
  -> resolve Model / Corpus / Evaluation Protocol
  -> validate static compatibility and integrity
  -> resolve complete public parameters
  -> allocate Evaluation Run ID
  -> create schema-valid run.yaml: running
  -> load executable implementation and construct EvaluationContext
  -> invoke evaluate(context)
  -> validate and materialize formal results
  -> completed / failed / cancelled
```

This decision changes only the launch-allocation boundary described by MLDB-ADR-SCHEMA-005 and MLDB-ADR-SCHEMA-008. Their other Training Run and Evaluation Run identity, lifecycle, result, failure-isolation, and immutability decisions remain valid.

Study materialization already validates selected assets and resolves protocol defaults before persisting its immutable plan. Child Run execution still applies the Run preflight boundary before allocating a concrete child Run.

## Rationale

Allocating a Run only after preflight succeeds guarantees that every persisted Run is schema-valid from its first durable state.

The distinction also gives clear semantics to failure history. Invalid launch requests are request-validation failures rather than failed training or evaluation executions. Once a Run exists, later execution or result-materialization failures are historical Run failures and remain visible in MLDB.

Keeping executable loading after allocation preserves useful failure evidence for implementation-import failures and other execution-setup defects that occur only when a validated attempt actually begins.

The boundary avoids introducing partially populated `running` record variants solely to represent requests that never reached executable execution.

## Rejected alternatives

### Allocate the Run before preflight and allow incomplete `running` records

This would require lifecycle-dependent optionality for selected inputs and `parameters`, complicate validation, and leave ambiguous records after process interruption.

### Allocate the Run before preflight but record unresolved caller parameters

Training Run and Evaluation Run are defined to store complete resolved protocol parameters rather than caller overrides. Persisting unresolved input would weaken that invariant and require later mutation of the Run's core inputs.

### Discard every failure before protocol entrypoint invocation

Executable import, context construction, and execution setup happen after a valid launch has begun and are useful historical failures. Those stages therefore remain inside the allocated Run boundary.

## Consequences

Training Run and Evaluation Run format validators can continue requiring complete selected inputs and resolved public parameters for every persisted Run state.

Direct launch tooling must perform asset resolution, static compatibility/integrity checks, and public-parameter resolution before allocating a Run ID.

Request-validation failures do not consume Training Run or Evaluation Run IDs and do not create Run directories.

Once a Run is allocated, executable-loading failures, protocol exceptions, result validation failures, serialization/import failures, and cancellation are represented through that Run's normal lifecycle.

Runtime overview, training/evaluation flow specs, Run lifecycle specs, and public-parameter failure wording must use this same boundary.

## Evidence

The current Training Run and Evaluation Run format specs require complete resolved `parameters` for every Run, while the earlier ADR invocation sequences allocated Runs before parameter resolution. The conflict was identified during the ADR-to-spec consistency review before runtime implementation began.
