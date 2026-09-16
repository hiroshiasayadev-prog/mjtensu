# Contract: Canonical mutation coordination

- **id**: `spec:mldb.v2.repository.mutation_coordination`
- **status**: draft
- **date**: 2026-09-09
- **parent**: `spec:mldb.v2.repository`
- **contract_class**: `repository`

## Purpose

Repeated or concurrent `advance_study` callers must not lose canonical Study Result updates.
Coordination is limited to short canonical repository mutations; it is not an execution scheduler.

## Rule

Mutations of one non-terminal Study Result and deterministic child-record reconciliation are
serialized by a repository-local lock keyed by exact Study Result ID.

The lock protects read/validate/write of canonical files only. It MUST NOT be held while waiting for
GPU work, polling backend state, uploading large artifacts, or sleeping between watch iterations.

After acquiring the lock, a caller re-resolves current canonical state before applying any prepared
backend observation so stale callers cannot overwrite newer valid dispositions.

## Backend-call boundary

Backend observation and candidate retrieval may happen before the lock is acquired. Backend
admission happens after canonical readiness is established and uses deterministic idempotent
admission identity, so two callers racing outside the lock cannot create two logical stages.

If backend admission succeeds but the caller fails before recording any local observational detail,
a later caller recovers the same logical admission through the backend port's deterministic stage
key. No canonical Queue row is required.

## Lock storage

Lock files are operational repository-local state outside `mldb_data/` canonical history. Their exact
filesystem implementation is not part of entity identity and they may be recreated after process
failure. Stale-lock recovery must use an implementation that relies on OS/process-safe locking rather
than treating the mere existence of an abandoned file as ownership.
