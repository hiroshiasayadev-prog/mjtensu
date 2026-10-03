from __future__ import annotations

import copy
import tarfile
from pathlib import Path

import pytorch_lightning as pl
import torch
from pytorch_lightning.loggers import Logger as LightningLoggerBase
from torch.utils.data import DataLoader

from nanodet_aabb_optimized.data.batch_process import stack_batch_img
from nanodet_aabb_optimized.data.dataset.coco import CocoDataset
from nanodet_aabb_optimized.evaluator.coco_detection import CocoDetectionEvaluator
from nanodet_aabb_optimized.trainer.task import TrainingTask
from nanodet_aabb_optimized.util import load_model_weight
from nanodet_aabb_optimized.util.yacs import CfgNode


_ARCHIVE = "nanodet-training-baseline-v1.tar"

_TRAIN_PIPELINE = {
    "perspective": 0.0,
    "scale": [0.6, 1.4],
    "stretch": [[0.8, 1.2], [0.8, 1.2]],
    "rotation": 0,
    "shear": 0,
    "translate": 0.2,
    "flip": 0.5,
    "brightness": 0.2,
    "contrast": [0.6, 1.4],
    "saturation": [0.5, 1.2],
    "normalize": [[103.53, 116.28, 123.675], [57.375, 57.12, 58.395]],
}
_VAL_PIPELINE = {
    "normalize": [[103.53, 116.28, 123.675], [57.375, 57.12, 58.395]],
}


def _collate(batch):
    """Runtime-compatible equivalent of NanoDet naive_collate.

    The fork's collate module still imports torch._six at module import time;
    only that compatibility shim is avoided here. Batch semantics are identical.
    """
    elem = batch[0]
    if isinstance(elem, dict):
        result = {key: _collate([sample[key] for sample in batch]) for key in elem}
        images = result.get("img")
        if isinstance(images, list) and images and all(isinstance(x, torch.Tensor) for x in images):
            result["img"] = stack_batch_img(images, divisible=32)
        return result
    return batch


def _extract_archive(archive: Path, destination: Path) -> Path:
    root = destination / "training-corpus"
    marker = root / ".ready"
    if marker.is_file():
        return root
    root.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, "r") as tf:
        for member in tf.getmembers():
            path = Path(member.name)
            if path.is_absolute() or ".." in path.parts:
                raise ValueError(f"unsafe corpus archive path: {member.name}")
        tf.extractall(root)
    marker.write_text("ok\n", encoding="utf-8")
    return root


def _baseline_model_cfg() -> dict:
    return {
        "weight_averager": {"name": "ExpMovingAverager", "decay": 0.9998},
        "arch": {
            "name": "NanoDetPlus",
            "detach_epoch": 10,
            "backbone": {
                "name": "ShuffleNetV2",
                "model_size": "1.0x",
                "pretrain": False,
                "out_stages": [2, 3, 4],
                "activation": "LeakyReLU",
            },
            "fpn": {
                "name": "GhostPAN",
                "in_channels": [116, 232, 464],
                "out_channels": 96,
                "kernel_size": 5,
                "num_blocks": 1,
                "use_res": False,
                "num_extra_level": 1,
                "use_depthwise": True,
                "activation": "LeakyReLU",
            },
            "head": {
                "name": "NanoDetPlusHead",
                "num_classes": 1,
                "input_channel": 96,
                "feat_channels": 96,
                "stacked_convs": 2,
                "kernel_size": 5,
                "strides": [8, 16, 32, 64],
                "activation": "LeakyReLU",
                "reg_max": 7,
                "norm_cfg": {"type": "BN"},
                "loss": {
                    "loss_qfl": {
                        "name": "QualityFocalLoss",
                        "use_sigmoid": True,
                        "beta": 2.0,
                        "loss_weight": 1.0,
                    },
                    "loss_dfl": {
                        "name": "DistributionFocalLoss",
                        "loss_weight": 0.25,
                    },
                    "loss_bbox": {"name": "GIoULoss", "loss_weight": 2.0},
                },
            },
            "aux_head": {
                "name": "SimpleConvHead",
                "num_classes": 1,
                "input_channel": 192,
                "feat_channels": 192,
                "stacked_convs": 4,
                "strides": [8, 16, 32, 64],
                "activation": "LeakyReLU",
                "reg_max": 7,
            },
        },
    }


def _cfg(root: Path, work_dir: Path, p) -> CfgNode:
    return CfgNode(
        {
            "save_dir": str(work_dir),
            "model": _baseline_model_cfg(),
            "device": {
                "gpu_ids": [0],
                "workers_per_gpu": int(p["workers"]),
                "batchsize_per_gpu": int(p["batch_size"]),
                "precision": 16 if bool(p["amp"]) else 32,
            },
            "schedule": {
                "optimizer": {
                    "name": "AdamW",
                    "lr": float(p["learning_rate"]),
                    "weight_decay": float(p["weight_decay"]),
                },
                "warmup": {
                    "name": "linear",
                    "steps": int(p["warmup_steps"]),
                    "ratio": float(p["warmup_ratio"]),
                },
                "total_epochs": int(p["epochs"]),
                "lr_schedule": {
                    "name": "CosineAnnealingLR",
                    "T_max": int(p["epochs"]),
                    "eta_min": float(p["eta_min"]),
                },
                "val_intervals": int(p["validation_interval"]),
            },
            "grad_clip": float(p["grad_clip"]),
            "evaluator": {"name": "CocoDetectionEvaluator", "save_key": "mAP"},
            "log": {"interval": int(p["log_interval"])},
            "class_names": ["mahjong_tile"],
        },
        new_allowed=True,
    )


class _TelemetryTrainingTask(TrainingTask):
    """Native NanoDet TrainingTask with telemetry-only total-loss reporting."""

    def training_step(self, batch, batch_idx):
        loss = super().training_step(batch, batch_idx)
        interval = int(self.cfg.log.interval)
        if self.global_step % interval == 0:
            self.logger.experiment.add_scalars(
                "Train_loss/total",
                {"Train": float(loss.detach().mean().cpu())},
                self.global_step,
            )
        return loss


class _TelemetryExperiment:
    def __init__(self, telemetry):
        self._telemetry = telemetry

    def add_scalars(self, tag, scalars, step):
        group = str(tag).replace("/", "_")
        for series, value in scalars.items():
            self._telemetry.report_scalar(
                group=group,
                series=str(series),
                value=float(value),
                step=int(step),
            )

    def flush(self):
        pass

    def close(self):
        pass


class _TelemetryLogger(LightningLoggerBase):
    """Bridge native NanoDet Lightning telemetry into MLDB/ClearML."""

    def __init__(self, telemetry):
        super().__init__()
        self._telemetry = telemetry
        self._experiment = _TelemetryExperiment(telemetry)

    @property
    def name(self):
        return "NanoDet-MLDB"

    @property
    def version(self):
        return "v1"

    @property
    def experiment(self):
        return self._experiment

    def info(self, message):
        print(message, flush=True)

    def log(self, message):
        print(message, flush=True)

    def log_hyperparams(self, params):
        pass

    def log_metrics(self, metrics, step):
        for name, value in metrics.items():
            self._telemetry.report_scalar(
                group="Val_metrics",
                series=str(name),
                value=float(value),
                step=int(step),
            )

    def save(self):
        pass

    def finalize(self, status):
        self._experiment.flush()


def _load_best_into(module: torch.nn.Module, best_path: Path) -> torch.nn.Module:
    payload = torch.load(best_path, map_location="cpu", weights_only=True)
    state = payload["state_dict"] if isinstance(payload, dict) and "state_dict" in payload else payload
    module.load_state_dict(state, strict=True)
    return module


def train(context):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    p = context.parameters
    pl.seed_everything(int(context.seed), workers=True)
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

    root = _extract_archive(context.corpus.root / _ARCHIVE, context.work_dir)
    cfg = _cfg(root, context.work_dir, p)

    train_dataset = CocoDataset(
        img_path=str(root / "images" / "jp_train"),
        ann_path=str(root / "annotations" / "jp_train.coco.json"),
        input_size=(320, 320),
        pipeline=_TRAIN_PIPELINE,
        keep_ratio=False,
        mode="train",
    )
    val_dataset = CocoDataset(
        img_path=str(root / "images" / "jp_valid"),
        ann_path=str(root / "annotations" / "jp_valid.coco.json"),
        input_size=(320, 320),
        pipeline=_VAL_PIPELINE,
        keep_ratio=False,
        mode="val",
    )

    workers = int(p["workers"])
    loader_kwargs = (
        {"persistent_workers": True, "prefetch_factor": 4} if workers > 0 else {}
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(p["batch_size"]),
        shuffle=True,
        num_workers=workers,
        pin_memory=True,
        collate_fn=_collate,
        drop_last=True,
        **loader_kwargs,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=int(p["batch_size"]),
        shuffle=False,
        num_workers=workers,
        pin_memory=True,
        collate_fn=_collate,
        drop_last=False,
        **loader_kwargs,
    )

    evaluator = CocoDetectionEvaluator(val_dataset)
    task = _TelemetryTrainingTask(cfg, evaluator)

    # MLDB owns the canonical architecture instance. Use it inside NanoDet's
    # native TrainingTask so EMA/optimizer/scheduler/evaluator remain upstream.
    task.model = context.model
    task.avg_model = copy.deepcopy(context.model)

    logger = _TelemetryLogger(context.telemetry)
    checkpoint = torch.load(
        root / "pretrained" / "nanodet-plus-m_320.pth",
        map_location="cpu",
        weights_only=True,
    )
    load_model_weight(task.model, checkpoint, logger)

    trainer = pl.Trainer(
        default_root_dir=str(context.work_dir),
        max_epochs=int(p["epochs"]),
        check_val_every_n_epoch=int(p["validation_interval"]),
        accelerator="gpu",
        devices=[0],
        log_every_n_steps=int(p["log_interval"]),
        num_sanity_val_steps=0,
        logger=logger,
        enable_progress_bar=False,
        benchmark=True,
        gradient_clip_val=float(p["grad_clip"]),
        precision=16 if bool(p["amp"]) else 32,
    )
    trainer.fit(task, train_loader, val_loader)

    best_path = context.work_dir / "model_best" / "nanodet_model_best.pth"
    if not best_path.is_file():
        raise RuntimeError("NanoDet native trainer did not produce model_best weights")
    return _load_best_into(context.model.cpu(), best_path)
