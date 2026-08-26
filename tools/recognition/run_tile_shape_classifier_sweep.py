from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any


DEFAULT_BATCH_SIZES = (512, 1024, 2048, 4096)
DEFAULT_ROTATION_CONDITIONS = (0.0, 22.5)


def parse_args() -> argparse.Namespace:
    repository_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(
        description=(
            "Run the initial gray64 C8 sweep: no rotation augmentation versus ±22.5°, "
            "across several large batch sizes. Failed/OOM runs are recorded and the "
            "remaining conditions continue."
        )
    )
    parser.add_argument("--repository-root", type=Path, default=repository_root)
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument(
        "--output-root",
        type=Path,
        help=(
            "Defaults to <repository-root>/.local/recognition/"
            "tile_classifier_runs/gray64_c8_seed42_sweep."
        ),
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument(
        "--batch-sizes", type=int, nargs="+", default=list(DEFAULT_BATCH_SIZES)
    )
    parser.add_argument(
        "--rotation-conditions",
        type=float,
        nargs="+",
        default=list(DEFAULT_ROTATION_CONDITIONS),
    )
    parser.add_argument("--learning-rate", type=float, default=1.0e-3)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval-batch-size", type=int, default=4096)
    parser.add_argument("--checkpoint-every", type=int, default=10)
    parser.add_argument(
        "--cache-device", choices=("auto", "cuda", "cpu"), default="auto"
    )
    parser.add_argument(
        "--overwrite-runs",
        action="store_true",
        help="Pass --overwrite to every child run.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the commands without executing training.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repository_root = args.repository_root.resolve()
    database = args.database.resolve()
    output_root = (
        args.output_root.resolve()
        if args.output_root is not None
        else repository_root
        / ".local"
        / "recognition"
        / "tile_classifier_runs"
        / f"gray64_c8_seed{args.seed}_sweep"
    )
    output_root.mkdir(parents=True, exist_ok=True)

    trainer = repository_root / "tools" / "recognition" / "train_tile_shape_classifier.py"
    if not trainer.is_file():
        raise FileNotFoundError(trainer)
    if not database.is_file():
        raise FileNotFoundError(database)

    results: list[dict[str, Any]] = []
    for rotation in args.rotation_conditions:
        condition_name = "A_noaug" if abs(rotation) < 1.0e-9 else f"B_rot{format_angle(rotation)}"
        for batch_size in args.batch_sizes:
            run_name = f"{condition_name}_bs{batch_size}_seed{args.seed}"
            run_dir = output_root / run_name
            command = [
                sys.executable,
                str(trainer),
                "--repository-root",
                str(repository_root),
                "--database",
                str(database),
                "--output-dir",
                str(run_dir),
                "--model",
                "c8",
                "--epochs",
                str(args.epochs),
                "--batch-size",
                str(batch_size),
                "--eval-batch-size",
                str(args.eval_batch_size),
                "--learning-rate",
                str(args.learning_rate),
                "--weight-decay",
                str(args.weight_decay),
                "--rotation-augment-deg",
                str(rotation),
                "--seed",
                str(args.seed),
                "--checkpoint-every",
                str(args.checkpoint_every),
                "--cache-device",
                str(args.cache_device),
            ]
            if args.overwrite_runs:
                command.append("--overwrite")

            print("\n===", run_name, "===")
            print(" ".join(command))
            if args.dry_run:
                results.append(
                    {
                        "run": run_name,
                        "status": "dry_run",
                        "rotation_augment_deg": rotation,
                        "batch_size": batch_size,
                        "command": command,
                    }
                )
                continue

            started = time.perf_counter()
            process = subprocess.run(command, cwd=repository_root, check=False)
            elapsed = time.perf_counter() - started
            run_result: dict[str, Any] = {
                "run": run_name,
                "rotation_augment_deg": rotation,
                "batch_size": batch_size,
                "return_code": process.returncode,
                "elapsed_seconds": elapsed,
                "command": command,
            }
            summary_path = run_dir / "summary.json"
            if process.returncode == 0 and summary_path.is_file():
                run_result["status"] = "completed"
                run_result["summary"] = json.loads(summary_path.read_text(encoding="utf-8"))
            elif process.returncode == 75:
                run_result["status"] = "oom"
            else:
                run_result["status"] = "failed"
            results.append(run_result)
            atomic_write_json(
                output_root / "sweep_summary.json",
                build_sweep_summary(args, database, results),
            )
            if process.returncode not in (0, 75):
                raise SystemExit(
                    f"Run {run_name} failed with code {process.returncode}; "
                    "aborting the sweep because this is not a CUDA OOM."
                )

    summary = build_sweep_summary(args, database, results)
    atomic_write_json(output_root / "sweep_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def build_sweep_summary(
    args: argparse.Namespace,
    database: Path,
    results: list[dict[str, Any]],
) -> dict[str, Any]:
    ranking: list[dict[str, Any]] = []
    for result in results:
        if result.get("status") != "completed":
            continue
        summary = result["summary"]
        ranking.append(
            {
                "run": result["run"],
                "rotation_augment_deg": result["rotation_augment_deg"],
                "batch_size": result["batch_size"],
                "best_epoch": summary.get("best_epoch"),
                "best_primary_score": summary.get("best_primary_score"),
                "elapsed_seconds": result.get("elapsed_seconds"),
            }
        )
    ranking.sort(
        key=lambda item: (
            -float(item["best_primary_score"] or -1.0),
            float(item["elapsed_seconds"] or float("inf")),
        )
    )
    return {
        "status": "dry_run" if args.dry_run else "in_progress_or_completed",
        "database": str(database),
        "epochs": int(args.epochs),
        "batch_sizes": [int(value) for value in args.batch_sizes],
        "rotation_conditions": [float(value) for value in args.rotation_conditions],
        "seed": int(args.seed),
        "results": results,
        "ranking": ranking,
    }


def format_angle(value: float) -> str:
    return f"{float(value):g}".replace(".", "p").replace("-", "m")


def atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent, text=True
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as output:
            json.dump(payload, output, ensure_ascii=False, indent=2)
            output.write("\n")
        os.replace(temporary_name, path)
    except Exception:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


if __name__ == "__main__":
    main()
