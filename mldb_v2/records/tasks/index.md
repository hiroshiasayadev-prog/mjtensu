# MLDB v2 Tasks

- **status**: active-integration
- **date**: 2026-09-17

Task records authorize implementation work under one Work Item after the applicable Specification boundary.

## Completed work

W001 through W010 are completed. W005/W006 established the original flat ClearML Task execution model; W010 completed backend-neutral telemetry and actual ClearML GPU/chart verification.

Their task records remain historical evidence and are not rewritten by later amendments.

## Current post-closure amendment

W011 replaces the target ClearML execution mapping without changing MLDB's scientific/canonical authority:

```text
T011-01 spec/docs amendment completed
    -> T011-02 backend Study execution / ClearML Pipeline
        -> T011-03 semantic progression over backend-owned scheduling
            -> T011-04 Pipeline UI summary / retry / cancel / logs
                -> T011-05 actual ClearML Pipeline + GPU closure
```

W011 is a focused repair prompted by actual usability of Study-level ClearML results. One MLDB Study Result should map to one ClearML Pipeline Run; child Tasks remain the detailed execution units.

## Rule

A Task may implement only its named boundary and tests. Use focused tests plus the cheapest necessary dependent integration at Task completion. Aggregate regression and actual backend verification occur at closure tasks. Historical completed evidence stays intact when a post-closure Specification amendment changes the target design.
