#!/usr/bin/env python3
"""Patch NanoDet v1.0.0 training to accept official weight-only .pth files.

NanoDet's tools/train.py treats every non-Lightning checkpoint as an old
training checkpoint and calls convert_old_model(), which requires epoch/iter.
Official release weight files contain only {"state_dict": ...}; they should be
passed directly to load_model_weight().
"""

from __future__ import annotations

import argparse
from pathlib import Path

OLD = '''        if "pytorch-lightning_version" not in ckpt:\n            warnings.warn(\n                "Warning! Old .pth checkpoint is deprecated. "\n                "Convert the checkpoint with tools/convert_old_checkpoint.py "\n            )\n            ckpt = convert_old_model(ckpt)\n'''

NEW = '''        if (\n            "pytorch-lightning_version" not in ckpt\n            and "epoch" in ckpt\n            and "iter" in ckpt\n        ):\n            warnings.warn(\n                "Warning! Old .pth checkpoint is deprecated. "\n                "Convert the checkpoint with tools/convert_old_checkpoint.py "\n            )\n            ckpt = convert_old_model(ckpt)\n        elif "state_dict" not in ckpt:\n            raise ValueError(\n                "Configured load_model file is neither a Lightning checkpoint, "\n                "an old NanoDet checkpoint, nor a weight-only state_dict file."\n            )\n'''


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "nanodet_root",
        type=Path,
        help="NanoDet repository root containing tools/train.py",
    )
    args = parser.parse_args()

    train_py = args.nanodet_root.resolve() / "tools" / "train.py"
    text = train_py.read_text(encoding="utf-8")

    if NEW in text:
        print(f"already patched: {train_py}")
        return

    if text.count(OLD) != 1:
        raise RuntimeError(
            f"Unexpected NanoDet source layout in {train_py}; "
            "refusing to patch."
        )

    train_py.write_text(text.replace(OLD, NEW, 1), encoding="utf-8")
    print(f"patched weight-only load_model support: {train_py}")


if __name__ == "__main__":
    main()
