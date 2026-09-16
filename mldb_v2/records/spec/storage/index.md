# Overview: MLDB v2 storage

- **id**: `spec:mldb.v2.storage`
- **status**: draft
- **date**: 2026-09-09
- **parent**: `spec:mldb.v2`

## What this is

Defines backend-neutral references to large immutable bytes and the manifest contract that freezes
Corpus contents.

S3-compatible object storage is the initial transport/storage class. MinIO product identity is not
part of canonical schema.

## Topics

| ref | responsibility |
|---|---|
| `spec:mldb.v2.storage.artifact_reference` | URI, byte size, digest, and immutable object semantics. |
| `spec:mldb.v2.storage.corpus_manifest` | Per-file manifest and Corpus-byte identity. |
