# Contract: CLI selectors and output

- **id**: `spec:mldb.v2.cli.selectors_output`
- **status**: draft
- **date**: 2026-09-09
- **parent**: `spec:mldb.v2.cli`
- **contract_class**: `api`

## Common selector model

Read/check commands use the same narrowing vocabulary where meaningful:

```text
--namespace <namespace>
--status <status>
--study <study-ref>
--since <duration-or-time>
--limit <positive-int>
```

Omitting a selector means "do not narrow by that dimension". A command MUST reject selectors that
have no meaning for its resource rather than silently ignore them.

Definition kind selection uses explicit resource/kind names; typed IDs never cause a filesystem
scan to infer kind.

## Output

List/check/read commands support `-o table|wide|json|yaml`; `--json` is an alias for `-o json`.
Human table output is the default on an interactive terminal. Structured output has no decoration,
progress spinner, ANSI control sequence, or human-only banner on stdout.

JSON list output is an array in deterministic CLI sort order. JSON single-object output is one
object. YAML preserves the same semantic fields. Field names are stable API-facing names, not parsed
from rendered table headings.

## Bulk check behavior

`validate` and `verify` evaluate the complete selected scope by default and report every target;
one failure does not stop later targets. Overall command exit is non-zero when any selected target
fails. `--fail-fast` MAY stop after the first failed target but never changes validation semantics.

Selection order is deterministic: Namespace ID, definition kind in canonical kind order, then typed
ID. Broad commands therefore produce stable machine-readable output without caller-side filesystem
sorting.

## Mutation safety

A mutation may target one exact object directly. A selector resolving to multiple mutation targets
requires explicit `--all`; otherwise the command rejects the request. Bulk mutation is never implied
merely by omitting an ID.
