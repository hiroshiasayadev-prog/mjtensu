from __future__ import annotations

import argparse
import pickle
import shutil
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Execute one MLDB assignment on a remote GPU host.")
    parser.add_argument("--package-root", type=Path, required=True)
    parser.add_argument("--assignment", type=Path, required=True)
    parser.add_argument("--assets-dir", type=Path, required=True)
    parser.add_argument("--result-dir", type=Path, required=True)
    return parser.parse_args()


def _asset_path(root: Path, key: str) -> Path:
    name = key.replace("/", "__")
    if key in {
        "architecture_implementation",
        "train_protocol_implementation",
        "model_architecture_implementation",
        "evaluation_protocol_implementation",
    }:
        name += ".py"
    return root / name


def main() -> None:
    args = parse_args()
    args.package_root = args.package_root.resolve()
    args.assignment = args.assignment.resolve()
    args.assets_dir = args.assets_dir.resolve()
    args.result_dir = args.result_dir.resolve()
    sys.path.insert(0, str(args.package_root))

    from mldb.src.orchestration.worker_api import EvaluationAssignment, TrainingAssignment
    from mldb.src.orchestration.worker_execution import (
        EvaluationExecutionFiles,
        TrainingExecutionFiles,
        execute_evaluation_attempt,
        execute_training_attempt,
    )

    with args.assignment.open("rb") as stream:
        assignment = pickle.load(stream)

    result_dir = args.result_dir
    result_dir.mkdir(parents=True, exist_ok=True)
    work_dir = result_dir / "work"
    work_dir.mkdir(parents=True, exist_ok=True)
    if isinstance(assignment, TrainingAssignment):
        weights_path = result_dir / "weights.pt"
        files = TrainingExecutionFiles(
            corpus_artifact=_asset_path(args.assets_dir, assignment.corpus_artifact.key),
            architecture_implementation=_asset_path(
                args.assets_dir, assignment.architecture_implementation.key
            ),
            train_protocol_implementation=_asset_path(
                args.assets_dir, assignment.train_protocol_implementation.key
            ),
            work_dir=work_dir,
            weights_candidate_path=weights_path,
        )
        candidate = execute_training_attempt(assignment, files)
        payload = {
            "kind": "training",
            "artifact": candidate.artifact,
            "weights_file": weights_path.name,
        }
    elif isinstance(assignment, EvaluationAssignment):
        files = EvaluationExecutionFiles(
            corpus_artifact=_asset_path(args.assets_dir, assignment.corpus_artifact.key),
            model_architecture_implementation=_asset_path(
                args.assets_dir, assignment.model_architecture_implementation.key
            ),
            model_weights=_asset_path(args.assets_dir, assignment.model_weights.key),
            evaluation_protocol_implementation=_asset_path(
                args.assets_dir, assignment.evaluation_protocol_implementation.key
            ),
            work_dir=work_dir,
        )
        result = execute_evaluation_attempt(assignment, files)
        artifact_files: dict[str, str] = {}
        artifact_dir = result_dir / "artifacts"
        artifact_dir.mkdir(exist_ok=True)
        for index, (key, source) in enumerate(result.artifacts.items()):
            target = artifact_dir / f"artifact-{index:04d}.bin"
            shutil.copyfile(source, target)
            artifact_files[key] = str(target.relative_to(result_dir))
        payload = {
            "kind": "evaluation",
            "metrics": dict(result.metrics),
            "unavailable_outputs": tuple(result.unavailable_outputs),
            "artifact_files": artifact_files,
        }
    else:
        raise TypeError(f"Unsupported assignment type: {type(assignment)!r}")

    with (result_dir / "result.pkl").open("wb") as stream:
        pickle.dump(payload, stream, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    main()
