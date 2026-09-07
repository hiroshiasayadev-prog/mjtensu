# Index: MLDB catalog

- **id**: `spec:mldb.catalog`
- **status**: draft
- **date**: 2026-09-04
- **parent**: `spec:mldb`

## What this is

Navigation for reusable MLDB catalog definitions that exist before concrete training or evaluation execution.

This area owns Task, Corpus, and Architecture contracts. Training procedures, learned Models, evaluation procedures, and Study execution belong to later topics.

## Topics

| title | kind | ref | summary |
|---|---|---|---|
| Task format | Contract | `spec:mldb.catalog.task_format` | YAML contract for semantic prediction Tasks and categorical target ordering. |
| Corpus format | Contract | `spec:mldb.catalog.corpus_format` | YAML contract for immutable materialized Corpora and artifact integrity metadata. |
| Image-classification Corpus | Contract | `spec:mldb.catalog.image_classification_corpus` | SQLite sample-table contract for `mjtensu.mldb/image-classification-corpus/v1`. |
| Architecture format | Contract | `spec:mldb.catalog.architecture_format` | YAML identity, lifecycle, Task binding, interface summary, and implementation integrity for Architectures. |
| Architecture build interface | Contract | `spec:mldb.catalog.architecture_build` | PyTorch `build()` request, response, and failure boundary. |

## Related specs

| ref | relation |
|---|---|
| `spec:mldb` | Parent MLDB specification root. |
| `spec:mldb.repository.layout` | Canonical physical placement for catalog entities. |
| `spec:mldb.runtime.asset_resolution` | Resolves catalog entities through typed references. |
