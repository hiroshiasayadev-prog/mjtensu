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

Executable definition modules live only as YAML-owned sibling `.py` files in the applicable
`mldb_data/<namespace>/<domain>/` directory. There are no free-standing helper modules under
`mldb_data/`.

A sibling module may import reusable implementation from a normal project source package. `tools/`
is not a reusable ML implementation package and MUST NOT be the designated implementation
entrypoint for an MLDB definition.

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
