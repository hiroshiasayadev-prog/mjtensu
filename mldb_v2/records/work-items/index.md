# MLDB v2 Work Items

- **status**: planned
- **date**: 2026-09-09

W001-W009 translate the original frozen MLDB v2 Specifications and Skeleton into implementation-sized ownership boundaries. Later Work Items may implement explicitly approved post-closure Specification amendments; those amendments must be recorded before implementation and must not silently rewrite completed Work Item evidence.

## Dependency graph

```text
W001 Foundation/repository
  -> W002 Catalog/verification
      -> W003 Study planning
      -> W004 Domain runtime
          -> W005 Backend/ClearML
W003 + W004 + W005 -> W006 Result acceptance/Study driver
W006 -> W007 Application/query API
W007 -> W008 CLI
W001..W008 -> W009 End-to-end conformance
W009 -> W010 Execution telemetry / observability
W010 -> W011 ClearML Pipeline Study execution
```

W003 and W004 may proceed in parallel after W002. Later Task records may further split work for parallel sessions, but must preserve each Work Item boundary.

## Work Items

| id | responsibility | depends on |
|---|---|---|
| MLDB-V2-WORK-001 | Foundation, canonical repository, storage/Git primitives | - |
| MLDB-V2-WORK-002 | Catalog definitions, executable verification, sealing | W001 |
| MLDB-V2-WORK-003 | Study compilation, source pinning, immutable Plan | W001, W002 |
| MLDB-V2-WORK-004 | Training/evaluation runtime, Model and artifact lifecycle | W001, W002 |
| MLDB-V2-WORK-005 | Generic backend integration and ClearML adapter | W004 |
| MLDB-V2-WORK-006 | Result acceptance and resumable Study driver | W003, W004, W005 |
| MLDB-V2-WORK-007 | Application/query API | W006 |
| MLDB-V2-WORK-008 | CLI adapter and installed `mldb` command | W007 |
| MLDB-V2-WORK-009 | End-to-end conformance and actual-backend verification | W001-W008 |
| MLDB-V2-WORK-010 | Backend-neutral execution telemetry and observability projection | W009 |
| MLDB-V2-WORK-011 | ClearML Pipeline Study execution and UI projection repair | W010 |

## Rule

Implementation lives under `mldb_v2/src/` and tests under `mldb_v2/tests/`. Canonical experiment definitions/history remain under repository-level `mldb_data/`. Work Items must not move ML execution/domain code into `tools/` or reintroduce queue/lease/heartbeat infrastructure owned by the backend.
