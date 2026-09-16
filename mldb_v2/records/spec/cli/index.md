# Overview: MLDB v2 CLI

- **id**: `spec:mldb.v2.cli`
- **status**: draft
- **date**: 2026-09-09
- **parent**: `spec:mldb.v2`

## What this is

Defines the installed `mldb` command used by humans and automation. CLI is a thin adapter over
`spec:mldb.v2.api`; it does not own a second workflow or domain implementation.

The CLI is discovery-first: broad read/check operations do not require callers to pre-discover IDs
through repository filesystem searches.

## Topics

| ref | responsibility |
|---|---|
| `spec:mldb.v2.cli.operations` | Discover, inspect, author/check, execute, monitor, and diagnose operations. |
| `spec:mldb.v2.cli.selectors_output` | Shared selectors, bulk-scope rules, and machine-readable output. |
