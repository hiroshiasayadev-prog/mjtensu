# Contract: Executable definition integrity

- **id**: `spec:mldb.v2.verification.executable_integrity`
- **status**: draft
- **date**: 2026-09-09
- **parent**: `spec:mldb.v2.verification`
- **contract_class**: `verification`

## Owned companion

Every executable definition records `implementation.sha256` for the exact same-basename sibling
`.py`. The sibling is always part of executable identity.

A sibling may import normal project source packages. When project-owned imported source contributes
to result-affecting Architecture/Train/Evaluation behavior, every such source file MUST be declared
under `implementation.sources` as a repository-relative path plus exact SHA-256.

Third-party/site-package modules, Python standard-library modules, and MLDB infrastructure that does
not define experiment behavior are not listed as project sources.

## Source entries

Each `implementation.sources` entry is exactly:

```yaml
- path: product/recognition/models/rotated_fcos.py
  sha256: <64 lowercase hex>
```

Paths are repository-relative regular files; directories, globs, absolute paths, and `tools/`
implementation paths are invalid in this list. Entries are unique and sorted lexicographically by
path in canonical YAML.

## Seal and planning rules

Verification hashes the sibling and every declared project source before sealing. Formal planning
re-verifies those hashes against the selected source commit. Any mismatch makes the sealed
definition unusable for that Plan and requires a new definition revision when behavior changed.

An executable definition that imports result-affecting project-owned source without declaring it is
non-conforming even if its sibling hash matches.

The Plan records the executable definition ID, sibling hash, and declared project-source path/hash
set so backend preflight can verify the exact execution inputs without trusting the current working
tree.

This contract does not attempt to fingerprint third-party runtime environments. Environment
provenance may be recorded by the backend, while reusable project-owned behavior remains protected
by the definition and pinned source contract.
