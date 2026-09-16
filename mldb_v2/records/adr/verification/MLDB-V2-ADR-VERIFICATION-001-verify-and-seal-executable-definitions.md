# MLDB-V2-ADR-VERIFICATION-001: Verify and seal executable definitions

- **status**: accepted
- **date**: 2026-09-09
- **depends_on**:
  - MLDB-V2-ADR-SCHEMA-001
  - MLDB-V2-ADR-REPOSITORY-001
- **supersedes**:
- **migrated_to_spec**: `spec:mldb.v2.verification.definition_lifecycle`

## Context

Git history identifies source versions, but presence in Git does not mean a draft executable
definition has passed its domain contract. MLDB v1's validation/sealing distinction is useful even
when execution lifecycle moves to ClearML.

## Decision

Reusable definitions have a lightweight validation and sealing lifecycle.

For an executable definition:

```text
draft -> validate -> verify executable contract -> seal
```

A sealed executable definition records the SHA-256 of its exact same-basename sibling `.py`.
A sealed definition is eligible for Study planning. A draft definition is not.

Executable entrypoints live as YAML-owned sibling `.py` files in the applicable
`mldb_data/<namespace>/<domain>/` directory. Reusable experiment helpers live only under that same
namespace's `lib/` tree and are declared with exact hashes in `implementation.sources`.

The sibling may import those declared same-namespace helpers. Standard-library, third-party, and
MLDB v2 infrastructure imports remain allowed; other repository-owned Python imports are invalid.
This keeps the complete experiment implementation movable as one namespace package.

Changing executable behavior after sealing requires a new definition revision.

## Rationale

Sealing remains a cheap, explicit gate between "written" and "safe to use in formal experiments"
without reintroducing v1 Run lifecycle machinery.

## Rejected alternatives

### Git commit alone means executable-ready

A committed draft or broken entrypoint would then be indistinguishable from a verified definition.

### Put all executable code into generic tool scripts

This creates unowned, shared entrypoints and makes definition integrity ambiguous.

## Consequences

Study compilation rejects unsealed executable definitions. Verification can remain local and
backend-independent.
