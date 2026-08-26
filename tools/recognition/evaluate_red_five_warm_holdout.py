from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path

import numpy as np
import torch

try:
    from build_red_five_classifier_dataset import preprocess_rgb_u8
    from red_five_classifier import build_model
    from train_red_five_classifier import normalize_tensor, rgb_u8_to_input_torch, binary_metrics
except ModuleNotFoundError:
    from tools.recognition.build_red_five_classifier_dataset import preprocess_rgb_u8
    from tools.recognition.red_five_classifier import build_model
    from tools.recognition.train_red_five_classifier import normalize_tensor, rgb_u8_to_input_torch, binary_metrics


def main() -> None:
    p = argparse.ArgumentParser()
    root = Path(__file__).resolve().parents[2]
    p.add_argument("--repository-root", type=Path, default=root)
    p.add_argument("--warm-database", type=Path)
    p.add_argument("--run-root", type=Path, required=True)
    a = p.parse_args()

    root = a.repository_root.resolve()
    db = (a.warm_database.resolve() if a.warm_database else root / ".local/recognition/red_five_datasets/warm_red_five_24.sqlite")
    run_root = a.run_root.resolve()
    device = torch.device("cuda")

    with sqlite3.connect(db) as c:
        c.row_factory = sqlite3.Row
        rows = c.execute("SELECT crop_id,tile_label,image_png,capture_id,brightness,shadow FROM sample ORDER BY capture_id,tile_label").fetchall()

    targets = np.asarray([1 if str(r["tile_label"]).startswith("red5") else 0 for r in rows], dtype=np.int64)
    results = []

    for run_dir in sorted(p for p in run_root.iterdir() if p.is_dir() and (p / "best.pt").is_file()):
        ckpt = torch.load(run_dir / "best.pt", map_location=device)
        cfg = ckpt["config"]
        mode = cfg["input_mode"]
        size = int(cfg.get("image_size", 64))
        model = build_model(mode, c8_fields=tuple(cfg["model"]["c8_fields"])).to(device)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()

        images = np.empty((len(rows), size, size, 3), dtype=np.uint8)
        for i, r in enumerate(rows):
            raw = preprocess_rgb_u8(bytes(r["image_png"]), image_size=size)
            images[i] = np.frombuffer(raw, dtype=np.uint8).reshape(size, size, 3)

        x = torch.from_numpy(images).to(device)
        x = rgb_u8_to_input_torch(x, input_mode=mode)
        x = normalize_tensor(x, mean=cfg["normalization"]["mean"], std=cfg["normalization"]["std"])
        with torch.inference_mode(), torch.cuda.amp.autocast():
            prob = torch.softmax(model(x), dim=1)[:, 1].float().cpu().numpy()
        pred = (prob >= 0.5).astype(np.int64)
        metrics = binary_metrics(pred, targets)
        errors = []
        for i, r in enumerate(rows):
            if pred[i] == targets[i]:
                continue
            errors.append({
                "crop_id": r["crop_id"],
                "tile_label": r["tile_label"],
                "red_probability": float(prob[i]),
                "brightness": r["brightness"],
                "shadow": r["shadow"],
                "capture_id": r["capture_id"],
            })

        normal = prob[targets == 0]
        red = prob[targets == 1]
        result = {
            "run": run_dir.name,
            "input_mode": mode,
            "best_epoch": int(ckpt["epoch"]),
            "metrics": metrics,
            "normal_max_p_red": float(normal.max()),
            "red_min_p_red": float(red.min()),
            "errors": errors,
        }
        results.append(result)
        print("=" * 72)
        print(f"{run_dir.name}: {metrics['tp'] + metrics['tn']}/{metrics['sample_count']}  FP={metrics['fp']} FN={metrics['fn']}")
        print(f"normal_max_p(red)={normal.max():.6f} red_min_p(red)={red.min():.6f}")
        for e in errors:
            print(f"ERROR {e['tile_label']:6} p(red)={e['red_probability']:.6f} brightness={e['brightness']} shadow={e['shadow']} crop={e['crop_id']}")

    out = run_root / "warm_holdout_evaluation.json"
    out.write_text(json.dumps({"warm_database": str(db), "sample_count": len(rows), "results": results}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"written: {out}")


if __name__ == "__main__":
    main()
