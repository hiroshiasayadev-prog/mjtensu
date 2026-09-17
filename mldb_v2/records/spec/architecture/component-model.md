# Contract: Component model

- **id**: `spec:mldb.v2.architecture.component_model`
- **status**: draft
- **date**: 2026-09-17
- **parent**: `spec:mldb.v2.architecture`
- **contract_class**: `architecture`

## Required flow

```text
Repository -> typed resolver -> validation/verification -> Study compiler -> immutable Plan
                                                |
                                                v
                                      Application / Study lifecycle
                                      /                      \
                           canonical acceptance        backend Study execution
                                  |                         |
                                  v                         v
                           canonical writer       native Pipeline/controller
                                                            |
                                                            v
                                                   child step Tasks
                                                            |
                                                            v
                                                  execution harness
```

MLDB owns Study meaning, semantic gates, lineage acceptance, and canonical history. The backend owns the native execution container and physical scheduling used to realize the immutable Plan.

## Dependency direction

Domain/schema code MUST NOT import a concrete backend adapter. Generic planning, semantic readiness, StageInput construction, and result acceptance depend only on backend-neutral values/contracts. Concrete adapters depend inward on those contracts and translate them into native backend orchestration.

Repository resolution MUST NOT execute definition companion code. Executable loading happens only during verification or execution harness invocation.

Canonical writers validate complete objects before persistence. Backend response objects are never written directly into `mldb_data/`.

## Study execution boundary

The backend Study execution may contain native controller callbacks/hooks needed to bridge child completion back into MLDB result acceptance and to release canonically-gated downstream steps. Those hooks call generic MLDB acceptance/lifecycle services; they do not reimplement model-family semantics inside the adapter.

For ClearML, the native container is a Pipeline Run whose child Tasks correspond to planned training/evaluation work. The Pipeline is operational state, not an additional canonical entity.

## Prohibited core components

MLDB core has no queue database, lease manager, heartbeat service, worker registry, remote-shell transport, generic resource scheduler, or duplicate retry scheduler. Those belong to the execution backend.

A resumable MLDB Study lifecycle/reconciliation service is allowed because it protects semantic gates and canonical acceptance. It must not duplicate native backend scheduling merely to launch ready compute stages.
