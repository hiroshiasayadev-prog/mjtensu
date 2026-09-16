# Reference: MLDB v2 identity and canonical serialization

- **id**: `spec:mldb.v2.common.identity`
- **status**: draft
- **date**: 2026-09-09
- **parent**: `spec:mldb.v2.common`
- **contract_class**: `reference`

## Entity references

Reusable definition references are typed `<namespace>/<local-id>` strings. Namespace and local ID
segments are ASCII lowercase kebab-case matching `[a-z0-9]+(?:-[a-z0-9]+)*`; empty segments,
underscores, uppercase letters, leading/trailing hyphens, and repeated hyphens are invalid. The kind
is supplied by the typed reference and is not embedded in the string.

Study-local trial IDs are `trial-NNNN`, starting at `trial-0001` with no gaps. Evaluation
coordinates are `eval-NNNN` within one trial, starting at `eval-0001` in canonical evaluation
expansion order.
## Canonical YAML digest form

Where a canonical MLDB record uses its own content digest for identity, the digest input is UTF-8
JSON produced from the parsed record after removing identity-only fields named by that record's
format contract, with object keys sorted lexicographically, no insignificant whitespace, JSON
booleans/null, finite JSON numbers, and LF-free single-document bytes.

The digest is lowercase SHA-256 hex. Digest-derived IDs use the first 16 hex characters only for
human-facing identity; the full digest is persisted and verified to detect collisions.

Study Plan full IDs are `<study-namespace>/<study-local-id>-plan-<sha256-prefix16>`. The plan file
basename is the local-ID portion under the source Study namespace's `study_plans/` directory.
Recompiling byte-equivalent canonical plan content yields the same Plan ID.
