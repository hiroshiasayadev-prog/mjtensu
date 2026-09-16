# Contract: Application query and discovery interface

- **id**: `spec:mldb.v2.api.query_interface`
- **status**: draft
- **date**: 2026-09-09
- **parent**: `spec:mldb.v2.api`
- **contract_class**: `interface`

## Query operations

The application boundary exposes read-only query operations sufficient for CLI/UI/agent discovery:

| operation | responsibility |
|---|---|
| `list_entities` | List one canonical kind or aggregate reusable-definition view with optional filters. |
| `get_entity` | Resolve/read one exact typed canonical object. |
| `list_study_results` | List Study executions across Namespaces with status/study/time/limit filters. |
| `get_study_result` | Read one canonical Study Result and derived canonical progress. |
| `observe_study` | Add current backend observational state for one Study Result without mutation. |
| `read_backend_logs` | Read/follow backend attempt logs when backend supports the capability. |
| `diagnose` | Read-only environment/repository/backend/object-store diagnostic report. |

No query operation imports executable definition code merely to render a list.

## Filtering

Filtering is semantic, not path-search based. Namespace, kind, Study, canonical status, time window,
and result limit are applied only where the requested resource supports them.
Unsupported filter/resource combinations fail as invalid requests; they are never silently ignored.
Returned collections are deterministically ordered according to the resource contract.

## Public query shapes

`list_entities` accepts one exact canonical `EntityKind` or the aggregate selector `definitions`, plus
optional Namespace and lifecycle-status filters where meaningful, and returns the repository
`CanonicalListing` shape. `get_entity` accepts exact kind + typed ID and returns the parsed canonical
document. Unsupported filter/resource combinations are `invalid_request`.

`list_study_results` accepts optional Namespace, Study, status-set, created-at lower/upper bound, and
positive result limit and returns canonical Study Result documents in deterministic order.

Canonical progress uses one counter shape:

```yaml
planned: 12
pending: 3
completed: 7
failed: 1
cancelled: 0
skipped: 1
```

`planned` equals the sum of the five disposition counters. A Study progress value contains exactly
`training`, `evaluations`, and `total`, each using that counter shape. Existing-Model trials therefore
contribute zero planned training stages.

`get_study_result` returns exactly the canonical `StudyResult` plus this derived canonical progress.
`observe_study` returns the same two values plus ordered current backend observations keyed by the
exact planned stage identity. Those observations follow `spec:mldb.v2.backend.candidate_outcome` and
remain non-canonical.

Backend log access uses an exact request consisting of Study Result ID, optional trial, optional
Evaluation coordinate, boolean `failed_only`, and boolean `follow`. The application yields ordered
text chunks with exactly `execution_id` (opaque backend ID or null when unavailable) and `text`.
Log chunks are observational presentation data, not canonical records.

`diagnose` returns ordered checks, each with exactly non-empty `name`, `status`
(`ok|warning|error|unsupported`), and nullable bounded diagnostic. Check names are presentation-stable
within one application version but are not canonical entity identity.

## Study observation

`get_study_result` is canonical-only. `observe_study` may combine that canonical state with backend
active/terminal observation for admitted coordinates. Backend observation never changes Study Result
disposition and may disappear if the backend is lost.

A caller needing all active executions uses `list_study_results` without an explicit run ID and a
non-terminal status selection. No filesystem walk is required by the caller.

## Logs and diagnostics

Backend log access is optional backend capability. If unsupported, the application returns a stable
unsupported-capability failure rather than synthesizing logs from canonical records.

`diagnose` may probe configured services and Git/object-store prerequisites but MUST NOT enqueue work,
seal definitions, create Plans/Results, or modify canonical files.
