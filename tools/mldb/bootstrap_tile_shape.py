from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from mldb.src.common.ids import CorpusId
from mldb.src.repository._local_filesystem import LocalFilesystem
from mldb.src.repository.layout import RepositoryLayout
from mldb.src.runtime.resolution import resolve_corpus

CORPUS_ID = CorpusId("gray35-jp500-seed42-v3-jp189-v1")
DEFAULT_SOURCE = Path(
    ".local/recognition/tile_classifier_datasets/gray35_jp500_seed42_v3_jp189.sqlite"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Materialize the tracked tile-shape MLDB Corpus when its SQLite artifact is absent."
    )
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--source-database", type=Path, default=DEFAULT_SOURCE)
    return parser.parse_args()


def _load_builder(path: Path):
    spec = importlib.util.spec_from_file_location("mldb_corpus_builder", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load Corpus builder: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    materialize = getattr(module, "materialize", None)
    if not callable(materialize):
        raise RuntimeError(f"Corpus builder has no materialize(source, output): {path}")
    return materialize


def main() -> None:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    layout = RepositoryLayout(repo_root)
    filesystem = LocalFilesystem()
    artifact = layout.corpus_artifact_path(CORPUS_ID)

    if not artifact.exists():
        source = args.source_database
        if not source.is_absolute():
            source = repo_root / source
        source = source.resolve()
        builder = _load_builder(layout.corpus_builder_path(CORPUS_ID))
        print(f"materializing {CORPUS_ID}: {source} -> {artifact}")
        builder(source, artifact)

    resolved = resolve_corpus(CORPUS_ID, layout, filesystem)
    print(
        f"ready: {resolved.metadata.id} "
        f"sha256={resolved.metadata.artifact.sha256} "
        f"bytes={resolved.metadata.artifact.bytes}"
    )


if __name__ == "__main__":
    main()
