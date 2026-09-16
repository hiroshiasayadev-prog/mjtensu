# Contract: Executable definition integrity

- **id**: `spec:mldb.v2.verification.executable_integrity`
- **status**: draft
- **date**: 2026-09-09
- **parent**: `spec:mldb.v2.verification`
- **contract_class**: `verification`

## Owned companion

Every executable definition records `implementation.sha256` for the exact same-basename sibling
`.py`. The sibling is always part of executable identity.

Executable behavior is self-contained within its `mldb_data/<namespace>/` package. The sibling
`.py` owns the entrypoint and MAY import reusable experiment code only from the same namespace's
`lib/` tree. Python standard-library modules, third-party packages, and MLDB v2 infrastructure may
still be imported normally. Other repository-owned Python imports are invalid.

## Source entries

`implementation.sources` declares every result-affecting same-namespace helper imported from
`mldb_data/<namespace>/lib/`. Each entry is an exact repository-relative `.py` path plus lowercase
SHA-256. Entries are sorted and unique. An empty list is valid when the sibling uses no helpers.

## Seal and planning rules

Verification hashes the same-basename sibling and every declared helper before sealing. The
import scan requires all statically resolvable same-namespace `lib/` imports, including transitive
imports, to appear in `implementation.sources`; undeclared helpers are invalid. Imports of repository
Python outside that private `lib/` boundary are forbidden even if listed in `sources`.

Formal planning pins the sibling and declared helper bytes from the selected source commit. Backend
preflight revalidates those exact pins without trusting the current working tree. A behavior-changing
sibling or helper change requires a new definition revision.

This contract does not attempt to fingerprint third-party runtime environments. Environment
provenance may be recorded by the backend, while reusable project-owned behavior remains protected
by the definition and pinned source contract.
