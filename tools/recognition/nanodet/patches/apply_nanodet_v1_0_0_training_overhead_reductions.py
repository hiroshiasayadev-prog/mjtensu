from __future__ import annotations

import argparse
import py_compile
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Patch NanoDet v1.0.0 TrainingTask to make full resume-checkpoint "
            "cadence configurable and avoid duplicate CUDA scalar reads while "
            "formatting training logs."
        )
    )
    parser.add_argument(
        "nanodet_root",
        type=Path,
        help="Path to the NanoDet repository root.",
    )
    return parser.parse_args()


def replace_once(text: str, old: str, new: str, description: str) -> str:
    if new in text:
        print(f"{description}: already applied")
        return text
    if text.count(old) != 1:
        raise RuntimeError(
            f"Expected exactly one source block for {description}; refusing "
            "to patch an unexpected source layout."
        )
    print(f"{description}: patched")
    return text.replace(old, new, 1)


def main() -> None:
    args = parse_args()
    root = args.nanodet_root.resolve()
    path = root / "nanodet" / "trainer" / "task.py"
    if not path.is_file():
        raise FileNotFoundError(path)

    text = path.read_text(encoding="utf-8")

    old_logging = '''            for loss_name in loss_states:
                log_msg += "{}:{:.4f}| ".format(
                    loss_name, loss_states[loss_name].mean().item()
                )
                self.scalar_summary(
                    "Train_loss/" + loss_name,
                    "Train",
                    loss_states[loss_name].mean().item(),
                    self.global_step,
                )
'''
    new_logging = '''            for loss_name in loss_states:
                loss_value = loss_states[loss_name].mean().item()
                log_msg += "{}:{:.4f}| ".format(loss_name, loss_value)
                self.scalar_summary(
                    "Train_loss/" + loss_name,
                    "Train",
                    loss_value,
                    self.global_step,
                )
'''
    text = replace_once(
        text,
        old_logging,
        new_logging,
        "single scalar read per logged training loss",
    )

    old_checkpoint = '''    def training_epoch_end(self, outputs: List[Any]) -> None:
        self.trainer.save_checkpoint(os.path.join(self.cfg.save_dir, "model_last.ckpt"))
'''
    new_checkpoint = '''    def training_epoch_end(self, outputs: List[Any]) -> None:
        checkpoint_interval = max(
            1, int(os.environ.get("NANODET_CHECKPOINT_INTERVAL", "1"))
        )
        completed_epoch = self.current_epoch + 1
        is_final_epoch = completed_epoch >= self.cfg.schedule.total_epochs
        if completed_epoch % checkpoint_interval == 0 or is_final_epoch:
            self.trainer.save_checkpoint(
                os.path.join(self.cfg.save_dir, "model_last.ckpt")
            )
'''
    text = replace_once(
        text,
        old_checkpoint,
        new_checkpoint,
        "configurable full-checkpoint cadence",
    )

    path.write_text(text, encoding="utf-8", newline="\n")
    py_compile.compile(str(path), doraise=True)
    print(
        "NanoDet training-overhead reduction patch complete:\n"
        "  NANODET_CHECKPOINT_INTERVAL=1 retains existing behavior\n"
        "  recommended long-run value: 5\n"
        "  final epoch is always saved\n"
        "  duplicate training-log Tensor.item(): removed"
    )


if __name__ == "__main__":
    main()
