from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[2]
    p = argparse.ArgumentParser(description="Retrain RGB/Cr/Y+Cr red-five C8 models from scratch with warm-light augmentation.")
    p.add_argument("--repository-root", type=Path, default=root)
    p.add_argument("--database", type=Path)
    p.add_argument("--output-root", type=Path)
    p.add_argument("--input-modes", nargs="+", choices=("rgb", "cr", "ycr"), default=["rgb", "cr", "ycr"])
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--eval-batch-size", type=int, default=4096)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--rotation-augment-deg", type=float, default=22.5)
    p.add_argument("--warm-augment-prob", type=float, default=0.50)
    p.add_argument("--warm-strength-min", type=float, default=0.10)
    p.add_argument("--warm-strength-max", type=float, default=1.00)
    p.add_argument("--warm-red-gain-max", type=float, default=1.50)
    p.add_argument("--warm-green-gain-max", type=float, default=1.08)
    p.add_argument("--warm-blue-gain-min", type=float, default=0.45)
    p.add_argument("--warm-exposure-min", type=float, default=0.65)
    p.add_argument("--warm-exposure-max", type=float, default=1.15)
    p.add_argument("--warm-database", type=Path)
    p.add_argument("--overwrite-runs", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def main() -> None:
    a = parse_args()
    root = a.repository_root.resolve()
    db = (a.database.resolve() if a.database else root / ".local/recognition/red_five_datasets" / f"rgb64_binary_jp5000_seed{a.seed}.sqlite")
    out = (a.output_root.resolve() if a.output_root else root / ".local/recognition/red_five_runs" / f"c8_rgb_cr_ycr_warmaug_seed{a.seed}")
    trainer = root / "tools/recognition/train_red_five_classifier_warm_aug.py"
    if not db.is_file():
        raise FileNotFoundError(db)
    if not trainer.is_file():
        raise FileNotFoundError(trainer)
    out.mkdir(parents=True, exist_ok=True)

    summary = []
    for mode in a.input_modes:
        run = out / f"c8_{mode}_warmaug_rot{fmt(a.rotation_augment_deg)}_seed{a.seed}"
        cmd = [
            sys.executable, str(trainer),
            "--repository-root", str(root),
            "--database", str(db),
            "--output-dir", str(run),
            "--input-mode", mode,
            "--epochs", str(a.epochs),
            "--batch-size", str(a.batch_size),
            "--eval-batch-size", str(a.eval_batch_size),
            "--rotation-augment-deg", str(a.rotation_augment_deg),
            "--seed", str(a.seed),
            "--warm-augment-prob", str(a.warm_augment_prob),
            "--warm-strength-min", str(a.warm_strength_min),
            "--warm-strength-max", str(a.warm_strength_max),
            "--warm-red-gain-max", str(a.warm_red_gain_max),
            "--warm-green-gain-max", str(a.warm_green_gain_max),
            "--warm-blue-gain-min", str(a.warm_blue_gain_min),
            "--warm-exposure-min", str(a.warm_exposure_min),
            "--warm-exposure-max", str(a.warm_exposure_max),
        ]
        if a.overwrite_runs:
            cmd.append("--overwrite")
        print("\n===", run.name, "===")
        print(" ".join(cmd))
        if a.dry_run:
            rc = None
        else:
            rc = subprocess.run(cmd, cwd=root, check=False).returncode
        summary.append({"mode": mode, "run": str(run), "return_code": rc, "command": cmd})
        if rc not in (None, 0):
            break

    warm_db = (a.warm_database.resolve() if a.warm_database else root / ".local/recognition/red_five_datasets/warm_red_five_24.sqlite")
    warm_eval_return_code = None
    if not a.dry_run and all(item["return_code"] == 0 for item in summary):
        evaluator = root / "tools/recognition/evaluate_red_five_warm_holdout.py"
        if not evaluator.is_file():
            raise FileNotFoundError(evaluator)
        if not warm_db.is_file():
            raise FileNotFoundError(warm_db)
        eval_cmd = [
            sys.executable, str(evaluator),
            "--repository-root", str(root),
            "--warm-database", str(warm_db),
            "--run-root", str(out),
        ]
        print("\n=== real warm-light holdout ===")
        print(" ".join(eval_cmd))
        warm_eval_return_code = subprocess.run(eval_cmd, cwd=root, check=False).returncode

    payload = {
        "database": str(db),
        "warm_database": str(warm_db),
        "warm_evaluation_return_code": warm_eval_return_code,
        "output_root": str(out),
        "warm_real_capture_policy": "not in training; keep as external holdout",
        "augmentation": {
            "probability": a.warm_augment_prob,
            "strength": [a.warm_strength_min, a.warm_strength_max],
            "red_gain_max": a.warm_red_gain_max,
            "green_gain_max": a.warm_green_gain_max,
            "blue_gain_min": a.warm_blue_gain_min,
            "exposure": [a.warm_exposure_min, a.warm_exposure_max],
        },
        "runs": summary,
    }
    (out / "warm_aug_sweep.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def fmt(value: float) -> str:
    return f"{float(value):g}".replace(".", "p").replace("-", "m")


if __name__ == "__main__":
    main()
