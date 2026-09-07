# Overview: MLDB evaluation

- **id**: `spec:mldb.evaluation`
- **status**: draft
- **date**: 2026-09-04
- **parent**: `spec:mldb`

## What this is

Defines post-training evaluation of an existing MLDB Model against one immutable Corpus through one reusable Evaluation Protocol.

This overview owns the evaluation-stage flow only. Protocol YAML, callable interfaces, Run fields, lifecycle, result validation, and structured artifact schemas belong to focused child specifications.

## Current contract

Evaluation separates reusable evaluation behavior from one concrete execution and its accepted formal results.

| stage | responsibility |
|---|---|
| Evaluation Protocol | Declare Task binding, public parameters, scalar metrics, formal structured artifacts, and the executable evaluation entrypoint. |
| Evaluation Run creation | After launch preflight succeeds, record the exact Model, Corpus, Evaluation Protocol, resolved parameters, start time, and optional Study lineage for one execution. |
| protocol invocation | Supply resolved Task, Corpus, Model, parameters, and Run work directory through the common evaluation interface. |
| result validation | Validate returned scalar metrics and declared structured artifacts without interpreting model-family-specific evaluation logic. |
| formal result materialization | Store accepted scalar metrics in `run.yaml` and accepted structured artifacts under the Evaluation Run `artifacts/` directory. |
| terminal history | Freeze the completed, completed-partial, failed, or cancelled Evaluation Run without creating another automatic domain entity. |

## Evaluation flow

```text
Model + Corpus + sealed Evaluation Protocol
                  |
                  v
          launch preflight
    resolve / compatibility /
      integrity / parameters
                  |
                  v
          allocate Evaluation Run
                  |
                  v
           status: running
                  |
                  v
        EvaluationContext
                  |
                  v
          evaluate(context)
                  |
                  v
         EvaluationResult
                  |
          +-------+-------+
          |               |
          v               v
   scalar metrics   structured artifacts
          |               |
          +-------+-------+
                  |
                  v
          generic validation
                  |
                  v
      accepted formal results
                  |
                  v
 completed / completed_partial / failed / cancelled
```

A launch request rejected during preflight creates no Evaluation Run. Once allocated, execution-setup, protocol, or completion-critical result failures are recorded by that Run.

Evaluation is post-training. Validation used solely by a Train Protocol to select its returned learned state remains training behavior rather than an Evaluation Run.

## Non-goals

- Define one universal classifier or detector evaluation algorithm.
- Define training-time checkpoint-selection validation.
- Treat arbitrary files in `work/` as formal outputs.
- Define a global metric ontology across unrelated Evaluation Protocols.
- Make MLflow or another visualization sink authoritative storage.
- Define queue claims, retries, leases, or worker transport.
- Define promotion or release decisions based on evaluation results.

## Topics

| title | kind | ref | summary |
|---|---|---|---|
| Evaluation Protocol format | Contract | `spec:mldb.evaluation.evaluation_protocol_format` | Versioned Evaluation Protocol YAML, lifecycle, public parameters, declared metrics, and declared artifacts. |
| Evaluation callable interface | Contract | `spec:mldb.evaluation.evaluate_interface` | `EvaluationContext`, `EvaluationResult`, and `evaluate(context)` execution boundary. |
| Evaluation Run format | Contract | `spec:mldb.evaluation.evaluation_run_format` | Persisted `run.yaml` fields, result metadata, environment, validation issues, and Study lineage. |
| Evaluation Run lifecycle | Concept | `spec:mldb.evaluation.evaluation_run_lifecycle` | Run state transitions, terminal immutability, retry behavior, and failure isolation. |
| Evaluation result validation | Reference | `spec:mldb.evaluation.result_validation` | Generic scalar and structured-artifact acceptance rules. |
| Evaluation artifact schemas | Index | `spec:mldb.evaluation.artifacts` | Versioned formal structured artifact formats supported by generic evaluation validation. |

## Related specs

| ref | relation |
|---|---|
| `spec:mldb.model` | Defines the learned Model evaluated by this stage. |
| `spec:mldb.catalog` | Defines Task and Corpus inputs. |
| `spec:mldb.runtime.public_parameters` | Defines common public-parameter resolution. |
| `spec:mldb.repository.layout` | Defines Evaluation Protocol and Evaluation Run placement. |
