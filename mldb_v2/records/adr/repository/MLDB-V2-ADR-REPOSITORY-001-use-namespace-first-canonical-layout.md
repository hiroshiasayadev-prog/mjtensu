# MLDB-V2-ADR-REPOSITORY-001: Use a namespace-first canonical layout

- **status**: accepted
- **date**: 2026-09-09
- **depends_on**:
  - MLDB-V2-ADR-ARCHITECTURE-001
- **supersedes**:
- **migrated_to_spec**: `spec:mldb.v2.repository.layout`

## Context

MLDB v1 stores each entity domain at the repository root, such as
`mldb_data/architectures/` and `mldb_data/studies/`. As experiment count grows, all model families
and research concepts become mixed in the same flat directories. Experiment viewers have the same
problem when every Study becomes an isolated top-level experiment.

The project needs one coarse grouping that remains stable across Architecture, Train Protocol,
Evaluation Protocol, Study, result, and backend visualization.

## Decision

MLDB v2 uses a namespace-first canonical layout:

```text
mldb_data/<namespace>/<domain>/
```

A namespace represents one continuing experiment concept: the body of work whose architectures,
training conditions, studies, and results are normally useful to inspect and compare together.
Condition sweeps within the same concept remain in the same namespace. A materially different
experiment concept gets a different namespace.

The recognized domain names are fixed by the repository specification. Domain directories do not
define additional identity hierarchy.

Each v2 namespace root contains `namespace.yaml`. A root child without `namespace.yaml` is not a v2
namespace. This intentionally allows the existing v1 flat directories to coexist during migration.

Entity references use:

```text
<namespace>/<local-id>
```

The entity kind is supplied by the typed reference and is not repeated inside the ID.

Cross-namespace references are allowed.

Python files under `mldb_data/` are permitted only when a domain specification defines an
executable companion for a YAML definition. Such a file MUST have the exact same basename as its
YAML sibling. Standalone helpers, shared Python modules, and unrelated scripts under `mldb_data/`
are invalid.

## Rationale

Opening one namespace shows the whole research concept instead of requiring a cross-root search.
The same namespace can map directly to one ClearML Project without making each Study a separate
project.

Typed lookup keeps IDs short while preserving unambiguous resolution.

## Rejected alternatives

### Domain-first layout

`mldb_data/<domain>/<namespace>/` scatters one research concept across the repository.

### Arbitrary nested folders

Unlimited hierarchy turns organization into another taxonomy problem. MLDB v2 has exactly one
namespace segment before the fixed domain.

### Put reusable ML implementation in `tools/`

`tools/` is not a reusable domain package. MLDB executable entrypoints are definition-owned sibling
modules. Shared reusable implementation belongs in a normal project source package and may be
imported by the sibling module.

## Consequences

Repository resolution is derived from `(kind, namespace/local-id)`. Namespace names are meaningful
to humans and to backend grouping, so changing a namespace is an identity change rather than a file
move.
