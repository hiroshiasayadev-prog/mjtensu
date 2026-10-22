"""One-shot migration of filesystem execution records into PostgreSQL.

Dry-run is the default.  Immutable identity conflicts fail closed.  StudyResult
replacement is available only for the controlled final cutover sync.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Mapping

from mldb_v2.src.common.ids import EntityKind, _validate_typed_reference
from mldb_v2.src.repository._yaml import _load_yaml
from mldb_v2.src.repository.execution_store import EXECUTION_RECORD_KINDS
from mldb_v2.src.repository.postgresql import PostgresExecutionRecordStore
from mldb_v2.src.repository.canonical_writes import (
    _TERMINAL_STUDY_STATUSES,
    _validate_study_result_record,
    _validate_study_result_transition,
)
from mldb_v2.src.repository.resolution import _DOMAIN_BY_KIND, _SCHEMA_BY_KIND


def _load_record(path: Path, *, root: Path, kind: EntityKind) -> dict[str, object]:
    try:
        value = _load_yaml(path.read_text(encoding="utf-8"))
    except Exception as error:
        raise RuntimeError(f"cannot load {path.relative_to(root)}") from error
    if type(value) is not dict:
        raise RuntimeError(f"record is not a mapping: {path.relative_to(root)}")
    schema = value.get("schema")
    if schema != _SCHEMA_BY_KIND[kind]:
        raise RuntimeError(
            f"schema mismatch for {path.relative_to(root)}: {schema!r}"
        )
    entity_id = _validate_typed_reference(value.get("id"))
    expected = f"{path.parents[1].name}/{path.stem}"
    if entity_id != expected:
        raise RuntimeError(
            f"identity/path mismatch for {path.relative_to(root)}: "
            f"{entity_id!r} != {expected!r}"
        )
    return value


def _records(root: Path):
    for namespace in sorted(root.iterdir(), key=lambda p: p.name):
        if not namespace.is_dir() or not (namespace / "namespace.yaml").is_file():
            continue
        for kind in sorted(EXECUTION_RECORD_KINDS, key=lambda item: item.value):
            domain = namespace / _DOMAIN_BY_KIND[kind]
            if not domain.is_dir():
                continue
            for path in sorted(domain.glob("*.yaml"), key=lambda p: p.name):
                yield kind, path, _load_record(path, root=root, kind=kind)


def _same(left: Mapping[str, object], right: Mapping[str, object]) -> bool:
    return json.dumps(left, sort_keys=True, separators=(",", ":")) == json.dumps(
        right, sort_keys=True, separators=(",", ":")
    )


def _replace_study_result(
    store: PostgresExecutionRecordStore,
    *,
    entity_id: str,
    document: Mapping[str, object],
) -> dict[str, object]:
    candidate = _validate_study_result_record(document, entity_id=entity_id)

    def migration_transition(
        current: Mapping[str, object],
        replacement: Mapping[str, object],
    ) -> None:
        validated_current = _validate_study_result_record(
            current, entity_id=entity_id
        )
        if validated_current["status"] in _TERMINAL_STUDY_STATUSES:
            raise ValueError(
                "cutover migration cannot replace a terminal StudyResult"
            )
        if not _same(replacement, candidate):
            raise ValueError("migration replacement changed unexpectedly")
        _validate_study_result_transition(validated_current, candidate)

    return store.replace(
        kind=EntityKind.STUDY_RESULT,
        entity_id=entity_id,
        replacement=candidate,
        transition_validator=migration_transition,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mldb-data-root", type=Path, required=True)
    parser.add_argument("--dsn", required=True)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--initialize-schema", action="store_true")
    parser.add_argument("--replace-study-results", action="store_true")
    args = parser.parse_args()

    root = args.mldb_data_root.resolve()
    if not root.is_dir():
        raise RuntimeError(f"mldb_data root is missing: {root}")

    store = PostgresExecutionRecordStore(args.dsn)
    if args.initialize_schema:
        store.ensure_schema()
    stats = {
        "scanned": 0,
        "missing": 0,
        "same": 0,
        "inserted": 0,
        "study_result_different": 0,
        "study_result_replaced": 0,
        "conflicts": 0,
    }

    for kind, path, document in _records(root):
        stats["scanned"] += 1
        entity_id = str(document["id"])
        try:
            existing = store.resolve(kind=kind, entity_id=entity_id)
        except FileNotFoundError:
            stats["missing"] += 1
            if args.apply:
                store.create_immutable(
                    kind=kind,
                    entity_id=entity_id,
                    document=document,
                )
                stats["inserted"] += 1
            continue

        if _same(existing, document):
            stats["same"] += 1
            continue

        if kind is EntityKind.STUDY_RESULT:
            stats["study_result_different"] += 1
            if args.apply and args.replace_study_results:
                _replace_study_result(
                    store,
                    entity_id=entity_id,
                    document=document,
                )
                stats["study_result_replaced"] += 1
                continue

        stats["conflicts"] += 1
        print(
            f"CONFLICT {kind.value} {entity_id} <- {path.relative_to(root)}"
        )

    print(json.dumps(stats, sort_keys=True))
    if stats["conflicts"]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
