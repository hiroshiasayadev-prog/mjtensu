# Reference: Executable asset verification

- **id**: `spec:mldb.v2.verification.executable_asset_tests`
- **status**: draft
- **date**: 2026-09-09
- **parent**: `spec:mldb.v2.verification`
- **contract_class**: `verification`

## Required tests

Before sealing, Architecture, Train Protocol, and Evaluation Protocol require asset-specific pytest.
Canonical placement mirrors namespace-first identity:

```text
mldb_tests/<namespace>/architectures/<local-id>/
mldb_tests/<namespace>/train_protocols/<local-id>/
mldb_tests/<namespace>/evaluation_protocols/<local-id>/
```

Test files use normal pytest discovery names. No definition stores an arbitrary test path.
## Seal gate

Sealing runs the derived test directory with pytest after schema/reference/interface validation.
Missing directory, zero collected tests, any failure/error, or inability to establish the companion
SHA-256 rejects sealing. At least one test must pass.

Tests resolve the target through the same typed MLDB resolver/executable-loader boundary used by
normal execution; they MUST NOT create a second direct-import convention for sibling modules.

Corpus builder tests are optional in v2 because a sealed Corpus is authoritative through its
verified manifest bytes, not through the builder implementation. Ordinary execution of already
sealed assets does not run pytest.
