# MLDB v2 Tasks

- **status**: active-integration
- **date**: 2026-09-13

Task records authorize implementation work under one Work Item after the frozen Specification/Skeleton boundary.

## Completed waves

- MLDB-V2-WORK-001 completed.
- MLDB-V2-WORK-002 completed.
- MLDB-V2-WORK-003 completed; T003-01 through T003-05 completed.
- MLDB-V2-WORK-004 completed; T004-01 through T004-05 completed, including live S3-compatible transport smoke.

## Current Experiment Ready lanes

W005 and W006 are now the active convergence region.

```text
W005:
  T005-01 completed -> T005-02 completed -> T005-03 completed --+
                                      \----> T005-04 completed --+-> T005-05 verify -> W005 close

W006:
  T006-01 completed -> T006-03 completed --+
  T006-02 completed ------------------------+-> T006-04 advance_study -> T006-05 verify -> W006 close
  W005 -------------------------------------+
```
T006-04 may start before formal W005 closure using a generic fake BackendPort; W005 remains the W006 integration/completion gate. Therefore T005-05 and T006-04 are the current safe parallel fan-out.

## Milestones

Experiment Ready ends at the W005/W006 convergence: real generic backend execution plus canonical acceptance/readiness/advance sufficient to run the first real Study. W007/W008/W009 are not prerequisites for that first experiment.

Production Complete continues afterward:

```text
W006 close -> W007 Application/query API -> W008 CLI adapter -> W009 final end-to-end conformance
```

## Rule
A Task may implement only its named boundary and tests. Use focused tests plus the cheapest necessary dependent integration at Task completion. Aggregate full Wave regression, adversarial verification, and full `mldb_v2/tests` at Wave closure. Do not delay the first real experiment for W007/W008/W009 polish unless an actual experiment blocker is found.
