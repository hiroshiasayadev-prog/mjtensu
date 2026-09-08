from __future__ import annotations

import argparse
import shutil
import sqlite3
from pathlib import Path


def materialize(source: Path, output: Path) -> None:
    source = source.resolve()
    output = output.resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, output)
    connection = sqlite3.connect(output)
    try:
        columns = {
            str(row[1])
            for row in connection.execute("PRAGMA table_info(sample)")
        }
        if "base_label" not in columns:
            raise ValueError("source sample table has no base_label column")
        if "target" not in columns:
            connection.execute("ALTER TABLE sample ADD COLUMN target TEXT")
            connection.execute("UPDATE sample SET target = base_label")
            connection.commit()
    finally:
        connection.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Materialize the immutable gray35 classifier Corpus artifact."
    )
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    materialize(args.source, args.output)
    print(args.output.resolve())


if __name__ == "__main__":
    main()
