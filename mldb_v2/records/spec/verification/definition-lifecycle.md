# Contract: Definition lifecycle

- **id**: `spec:mldb.v2.verification.definition_lifecycle`
- **status**: draft
- **date**: 2026-09-09
- **parent**: `spec:mldb.v2.verification`
- **contract_class**: `lifecycle`

## States

Task, Corpus, Architecture, Train Protocol, Evaluation Protocol, and Study use exactly `draft` and
`sealed`. Namespace metadata has no sealing lifecycle.

Draft definitions may be edited. A sealed definition is immutable under that ID and may not return
to draft; semantic or executable behavior changes require a new revision.

## Common seal gate

All sealable definitions require schema validation, ID/path consistency, typed-reference validation,
and domain-specific semantic validation. Study additionally requires complete planning validation of
its declared source/stages against referenced sealed definitions.
## Domain-specific gates

- Corpus sealing verifies the complete manifest, object sizes/digests, and builder companion hash
  when a builder exists.
- Architecture, Train Protocol, and Evaluation Protocol additionally satisfy
  `spec:mldb.v2.verification.executable_integrity`, verify callable interface, and pass the required
  asset pytest gate from `spec:mldb.v2.verification.executable_asset_tests`.
- Task and Study have no executable-companion pytest requirement.

Study planning rejects any referenced reusable definition that is not sealed or whose recorded
integrity no longer matches. Verification explicitly rejects unowned `.py` files under v2 namespace
domains.
