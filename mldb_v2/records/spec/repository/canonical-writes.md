# Contract: Canonical write and recovery semantics

- **id**: `spec:mldb.v2.repository.canonical_writes`
- **status**: draft
- **date**: 2026-09-09
- **parent**: `spec:mldb.v2.repository`
- **contract_class**: `repository`

## File-level writes

Every canonical YAML/manifest mutation writes complete validated content through an atomic
file-replacement boundary. Readers must never observe a partially written canonical document.

Immutable Plan, Training Result, Model, Evaluation Result, and terminal Study Result creation is
idempotent: if the canonical path already contains the exact valid expected record, creation returns
that record; different valid content at the same identity is a lifecycle conflict and is never
overwritten.

A non-terminal Study Result may be atomically replaced only with a valid next representation allowed
by `spec:mldb.v2.results.study_result_format`.

## Immutable validation ownership

`create_immutable` is a canonical safety boundary, not a raw mapping serializer. Every concrete
canonical writer MUST be configured with a non-optional `CanonicalRecordValidator` dependency before
production immutable writes are possible. There is no implicit, pass-through, or accept-all default.

For every Study Plan, Training Result, Model, or Evaluation Result create/replay request, the writer
passes the requested canonical `kind`, typed `entity_id`, and complete proposed document to that port.
The validator owns the kind-dispatched domain validation defined by the owning format contracts. It
MUST reject at least wrong schema kind, missing or extra required structure, invalid domain-derived
identity/cross-field invariants, and invalid status-dependent payload. Validation is non-normalizing:
success authorizes persistence of the exact supplied document; it does not rewrite or repair it.

The repository writer itself owns only repository-generic safety: supported immutable kind, typed
identity/path consistency, canonical namespace/path derivation, serializability, locking, atomic
replacement, exact-content idempotence, and same-identity conflict behavior. It MUST NOT duplicate
Study Plan, Training Result, Model, or Evaluation Result format validators in the repository layer.
The configured validator is a dependency-inversion port implemented by the owning domain validators
or by a composition layer that delegates to them.

Validation MUST occur before any create or replay decision can succeed. If the canonical path already
exists, the existing document is validated through the same port for the same `kind` and `entity_id`
before exact-content equality may count as idempotent replay. An invalid existing document is never
returned as successful replay.

The persistence-time validator does not replace Study compilation or result acceptance and MUST NOT
turn the repository into a backend/execution layer. It may use read-only canonical/verification
dependencies required by the owning contract. In particular, child Training/Evaluation Results and
Models MUST remain validatable before the parent Study Result slot is updated; validation MUST NOT
require child-after-parent ordering. The child-before-parent recovery order below remains normative.

This port closes the generic immutable-write boundary only. The dedicated `create_study_result` and
`replace_nonterminal_study_result` methods retain their existing Study Result format/transition
contract and are not redesigned by this finding repair.

## Multi-record acceptance

Operations that establish multiple canonical records are recoverable rather than pretending the
filesystem provides a cross-file transaction.
For successful training acceptance:

1. ensure the deterministic Training Result record;
2. ensure the deterministic Model record;
3. only then update the Study Result training disposition to `completed`.

For Evaluation acceptance, ensure the Evaluation Result before updating the parent Study Result
slot. Failed/cancelled child Results follow the same child-before-parent rule.

If interruption occurs between these writes, the next `advance_study` pass resolves the deterministic
child identity, validates any already-present exact record, and finishes the missing parent update.
It must not delete a valid child record merely to simulate rollback.

## Git boundary

Canonical write operations do not perform Git commits. Git staging/commit remains repository
workflow outside the MLDB application mutation.
