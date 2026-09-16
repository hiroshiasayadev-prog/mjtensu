# Contract: Artifact reference

- **id**: `spec:mldb.v2.storage.artifact_reference`
- **status**: draft
- **date**: 2026-09-09
- **parent**: `spec:mldb.v2.storage`
- **contract_class**: `value`

## Shape

Every canonical reference to large external bytes contains at least:

```yaml
uri: s3://bucket/path/to/object
bytes: 123456
sha256: <64 lowercase hex characters>
```

Domain contracts may add immutable format/media/schema metadata. URI alone is never sufficient
canonical identity.
## Rules

- `bytes` and `sha256` describe exact referenced bytes and are verified before canonical acceptance.
- A referenced object MUST NOT be overwritten in place with different bytes.
- URI is part of an immutable terminal record. Canonical terminal records are not rewritten merely
  because storage is reorganized.
- Storage migration therefore preserves the original URI through replication/redirect/compatible
  object-key retention, or produces a new higher-level canonical record that explicitly supersedes
  the old reference under a future migration contract.
- Credentials, presigned URLs, access keys, passwords, and backend-local cache paths are forbidden
  in canonical records.

ClearML artifact/model IDs may appear separately as backend provenance.
