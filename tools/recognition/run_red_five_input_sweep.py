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


DEFAULT_INPUT_MODES = ("rgb", "cr", "ycr")
DEFAULT_EVAL_ANGLES = (0.0, 15.0, 30.0, 45.0)


def parse_args() -> argparse.Namespace:
    repository_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(
        description=(
            "Train the same C8 red-five classifier under RGB, Cr, and Y+Cr inputs. "
            "All three runs share dataset, seed, architecture, optimization, and rotation settings."
        )
    )
    parser.add_argument("--repository-root", type=Path, default=repository_root)
    parser.add_argument(
        "--database",
        type=Path,
        help=(
            "Defaults to <repository-root>/.local/recognition/red_five_datasets/"
            "rgb64_binary_jp5000_seed42.sqlite."
        ),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        help=(
            "Defaults to <repository-root>/.local/recognition/red_five_runs/"
            "c8_rgb_cr_ycr_seed42."
        ),
    )
    parser.add_argument(
        "--input-modes",
        nargs="+",
        choices=DEFAULT_INPUT_MODES,
        default=list(DEFAULT_INPUT_MODES),
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--eval-batch-size", type=int, default=4096)
    parser.add_argument("--learning-rate", type=float, default=1.0e-3)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--rotation-augment-deg", type=float, default=22.5)
    parser.add_argument(
        "--eval-angles",
        type=float,
        nargs="+",
        default=list(DEFAULT_EVAL_ANGLES),
    )
    parser.add_argument("--angle-eval-every", type=int, default=5)
    parser.add_argument("--checkpoint-every", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--cache-device",
        choices=("auto", "cuda", "cpu"),
        default="auto",
    )
    parser.add_argument("--overwrite-runs", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repository_root = args.repository_root.resolve()
    database = (
        args.database.resolve()
        if args.database is not None
        else repository_root
        / ".local"
        / "recognition"
        / "red_five_datasets"
        / f"rgb64_binary_jp5000_seed{args.seed}.sqlite"
    )
    output_root = (
        args.output_root.resolve()
        if args.output_root is not None
        else repository_root
        / ".local"
        / "recognition"
        / "red_five_runs"
        / f"c8_rgb_cr_ycr_seed{args.seed}"
    )
    trainer = repository_root / "tools" / "recognition" / "train_red_five_classifier.py"
    if not trainer.is_file():
        raise FileNotFoundError(trainer)
    if not database.is_file():
        raise FileNotFoundError(database)
    output_root.mkdir(parents=True, exist_ok=True)

    results: list[dict[str, Any]] = []
    for input_mode in args.input_modes:
        run_name = f"c8_{input_mode}_rot{format_angle(args.rotation_augment_deg)}_seed{args.seed}"
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
            "--input-mode",
            input_mode,
            "--epochs",
            str(args.epochs),
            "--batch-size",
            str(args.batch_size),
            "--eval-batch-size",
            str(args.eval_batch_size),
            "--learning-rate",
            str(args.learning_rate),
            "--weight-decay",
            str(args.weight_decay),
            "--rotation-augment-deg",
            str(args.rotation_augment_deg),
            "--eval-angles",
            *[str(value) for value in args.eval_angles],
            "--angle-eval-every",
            str(args.angle_eval_every),
            "--checkpoint-every",
            str(args.checkpoint_every),
            "--seed",
            str(args.seed),
            "--cache-device",
            str(args.cache_device),
        ]
        if args.overwrite_runs:
            command.append("--overwrite")

        print(f"\n=== {run_name} ===")
        print(" ".join(command))
        if args.dry_run:
            results.append(
                {
                    "run": run_name,
                    "input_mode": input_mode,
                    "status": "dry_run",
                    "command": command,
                }
            )
            continue

        started = time.perf_counter()
        process = subprocess.run(command, cwd=repository_root, check=False)
        elapsed = time.perf_counter() - started
        result: dict[str, Any] = {
            "run": run_name,
            "input_mode": input_mode,
            "return_code": process.returncode,
            "elapsed_seconds": elapsed,
            "command": command,
        }
        summary_path = run_dir / "summary.json"
        if process.returncode == 0 and summary_path.is_file():
            result["status"] = "completed"
            result["summary"] = json.loads(summary_path.read_text(encoding="utf-8"))
        elif process.returncode == 75:
            result["status"] = "oom"
        else:
            result["status"] = "failed"
        results.append(result)
        atomic_write_json(
            output_root / "sweep_summary.json",
            build_sweep_summary(args, database, results),
        )
        if process.returncode not in (0, 75):
            raise SystemExit(f"Run {run_name} failed with code {process.returncode}")

    summary = build_sweep_summary(args, database, results)
    atomic_write_json(output_root / "sweep_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def build_sweep_summary(
    args: argparse.Namespace,
    database: Path,
    results: list[dict[str, Any]],
) -> dict[str, Any]:
    comparison: list[dict[str, Any]] = []
    for result in results:
        if result.get("status") != "completed":
            continue
        summary = result["summary"]
        evaluation = summary["best_evaluation"]
        manual_angles = evaluation["manual_val"]["angles"]
        jp_test_angles = evaluation["jp_test"]["angles"]
        manual_balanced = [
            float(payload["overall"]["balanced_accuracy"])
            for payload in manual_angles.values()
        ]
        manual_recall = [
            float(payload["overall"]["recall"])
            for payload in manual_angles.values()
        ]
        jp_test_balanced = [
            float(payload["overall"]["balanced_accuracy"])
            for payload in jp_test_angles.values()
        ]
        zero = manual_angles.get("0deg")
        dark_partial = None
        if zero is not None:
            dark_partial = zero.get("by_condition", {}).get(
                "brightness=dark|shadow=partial"
            )
        comparison.append(
            {
                "run": result["run"],
                "input_mode": result["input_mode"],
                "best_epoch": summary["best_epoch"],
                "best_jp_val_angle_mean_balanced_accuracy": summary["best_primary_score"],
                "jp_test_angle_mean_balanced_accuracy": mean(jp_test_balanced),
                "manual_angle_mean_balanced_accuracy": mean(manual_balanced),
                "manual_angle_mean_recall": mean(manual_recall),
                "manual_0deg": None if zero is None else zero["overall"],
                "manual_dark_partial_0deg": dark_partial,
                "elapsed_seconds": result.get("elapsed_seconds"),
            }
        )
    comparison.sort(
        key=lambda item: (
            -float(item["manual_angle_mean_balanced_accuracy"]),
            -float(item["manual_angle_mean_recall"]),
            -float(item["jp_test_angle_mean_balanced_accuracy"]),
        )
    )
    return {
        "status": "dry_run" if args.dry_run else "in_progress_or_completed",
        "database": str(database),
        "input_modes": list(args.input_modes),
        "model": "C8 rotation-equivariant classifier",
        "rotation_augment_deg": float(args.rotation_augment_deg),
        "eval_angles": [float(value) for value in args.eval_angles],
        "epochs": int(args.epochs),
        "batch_size": int(args.batch_size),
        "seed": int(args.seed),
        "comparison_note": (
            "Checkpoints are selected using JP validation only. manual_val is untouched by "
            "checkpoint selection, then used here to compare input representations."
        ),
        "results": results,
        "comparison": comparison,
    }


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


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
