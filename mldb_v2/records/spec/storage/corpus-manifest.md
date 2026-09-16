# Contract: Corpus manifest

- **id**: `spec:mldb.v2.storage.corpus_manifest`
- **status**: draft
- **date**: 2026-09-09
- **parent**: `spec:mldb.v2.storage`
- **contract_class**: `format`

## Placement

For Corpus `<namespace>/<local-id>`:

```text
mldb_data/<namespace>/corpora/<local-id>.yaml
mldb_data/<namespace>/corpora/<local-id>.manifest.jsonl
```

## Entry contract

Each JSONL row identifies one immutable file relative to the Corpus `storage.root_uri` and contains:

```json
{"path":"images/000001.jpg","bytes":12345,"sha256":"...","split":"train"}
```

`path`, `bytes`, and `sha256` are mandatory. `split` is mandatory when split membership is
file-addressable through the manifest. Domain-specific metadata may be added only through a
versioned manifest schema rule.

Entries are in canonical lexical path order and paths are unique.

The SHA-256 of the exact manifest file is stored in Corpus YAML. The manifest digest therefore
commits to file membership, relative path, byte length, byte digest, and recorded split assignment.

A sealed Corpus MUST NOT reuse its ID with a different manifest digest.
