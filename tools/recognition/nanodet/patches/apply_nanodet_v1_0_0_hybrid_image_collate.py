from __future__ import annotations

import argparse
import py_compile
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Patch NanoDet v1.0.0 so DataLoader workers stack fixed-size image "
            "tensors while variable-length ground-truth arrays remain lists."
        )
    )
    parser.add_argument(
        "nanodet_root",
        type=Path,
        help="Path to the NanoDet repository root.",
    )
    return parser.parse_args()


def patch_collate(path: Path) -> str:
    text = path.read_text(encoding="utf-8")

    import_old = "import torch\nfrom torch._six import string_classes\n"
    import_new = (
        "import torch\n"
        "from torch._six import string_classes\n\n"
        "from .batch_process import stack_batch_img\n"
    )

    body_old = '''def naive_collate(batch):
    """Only collate dict value in to a list. E.g. meta data dict and img_info
    dict will be collated."""

    elem = batch[0]
    if isinstance(elem, dict):
        return {key: naive_collate([d[key] for d in batch]) for key in elem}
    else:
        return batch
'''

    body_new = '''def naive_collate(batch):
    """Collate metadata as lists but stack image tensors inside each worker.

    Variable-length values such as ground-truth boxes and labels intentionally
    remain lists. The fixed-size image field is padded and stacked into one CPU
    tensor before it crosses the DataLoader multiprocessing queue. This reduces
    shared-storage reconstruction, pin-memory work, and host-to-device copies.
    """

    elem = batch[0]
    if isinstance(elem, dict):
        collated = {
            key: naive_collate([sample[key] for sample in batch]) for key in elem
        }
        batch_imgs = collated.get("img")
        if (
            isinstance(batch_imgs, list)
            and batch_imgs
            and all(isinstance(img, torch.Tensor) for img in batch_imgs)
        ):
            collated["img"] = stack_batch_img(batch_imgs, divisible=32)
        return collated
    return batch
'''

    if body_new in text and import_new in text:
        return "already_applied"

    if text.count(import_old) != 1:
        raise RuntimeError(
            "Expected the NanoDet v1.0.0 collate import block exactly once."
        )
    if text.count(body_old) != 1:
        raise RuntimeError(
            "Expected the NanoDet v1.0.0 naive_collate body exactly once."
        )

    text = text.replace(import_old, import_new, 1)
    text = text.replace(body_old, body_new, 1)
    path.write_text(text, encoding="utf-8", newline="\n")
    return "patched"


def patch_task(path: Path) -> str:
    text = path.read_text(encoding="utf-8")

    old = '''    def _preprocess_batch_input(self, batch):
        batch_imgs = batch["img"]
        if isinstance(batch_imgs, list):
            batch_imgs = [img.to(self.device) for img in batch_imgs]
            batch_img_tensor = stack_batch_img(batch_imgs, divisible=32)
            batch["img"] = batch_img_tensor
        return batch
'''

    new = '''    def _preprocess_batch_input(self, batch):
        batch_imgs = batch["img"]
        if isinstance(batch_imgs, list):
            # Compatibility fallback for callers that do not use naive_collate.
            batch_imgs = stack_batch_img(batch_imgs, divisible=32)
        batch["img"] = batch_imgs.to(self.device, non_blocking=True)
        return batch
'''

    if new in text:
        return "already_applied"
    if text.count(old) != 1:
        raise RuntimeError(
            "Expected the NanoDet v1.0.0 batch preprocessing block exactly once."
        )

    path.write_text(text.replace(old, new, 1), encoding="utf-8", newline="\n")
    return "patched"


def main() -> None:
    args = parse_args()
    root = args.nanodet_root.resolve()
    collate_path = root / "nanodet" / "data" / "collate.py"
    task_path = root / "nanodet" / "trainer" / "task.py"

    for path in (collate_path, task_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    collate_result = patch_collate(collate_path)
    task_result = patch_task(task_path)

    py_compile.compile(str(collate_path), doraise=True)
    py_compile.compile(str(task_path), doraise=True)

    print(
        "NanoDet hybrid image-collate patch complete:\n"
        f"  collate.py: {collate_result}\n"
        f"  task.py: {task_result}\n"
        "  image IPC storages per batch: many -> one\n"
        "  image H2D copies per batch: many -> one\n"
        "  variable-length targets: preserved as lists\n"
        "  device transfer: non_blocking"
    )


if __name__ == "__main__":
    main()
