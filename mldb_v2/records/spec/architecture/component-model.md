# Contract: Component model

- **id**: `spec:mldb.v2.architecture.component_model`
- **status**: draft
- **date**: 2026-09-09
- **parent**: `spec:mldb.v2.architecture`
- **contract_class**: `architecture`

## Required flow

```text
Repository -> typed resolver -> validation/verification -> Study compiler -> immutable Plan
                                                |
                                                v
                                      application Study driver
                                      /        |          \
                              readiness   backend port   result acceptance
                                           |                 |
                                           v                 v
                                  backend adapter       canonical writer
                                           |
                                           v
                                  execution harness
```

The Study driver repeatedly reconciles immutable Plan intent with canonical accepted outcomes and
backend observations. It owns semantic DAG progression, not resource scheduling.
## Dependency direction

Domain/schema code MUST NOT import a concrete backend adapter. Generic planning, readiness, and
result acceptance depend only on backend-neutral values. Concrete adapters depend inward on the
backend port and canonical contracts.

Repository resolution MUST NOT execute definition companion code. Executable loading happens only
during verification or backend execution harness invocation.

Canonical writers validate complete objects before persistence. Backend response objects are never
written directly into `mldb_data/`.

## Prohibited core components

The initial v2 core has no queue database, lease manager, heartbeat service, worker registry,
remote-shell transport, generic resource scheduler, or MLflow-style tracking database.

A resumable application driver is explicitly allowed because it advances Plan-defined semantic
dependencies and canonical acceptance rather than scheduling compute resources.
