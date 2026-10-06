from __future__ import annotations

import copy
import tarfile
from pathlib import Path

import pytorch_lightning as pl
import torch
from pytorch_lightning.loggers import Logger as LightningLoggerBase
from torch.utils.data import ConcatDataset, DataLoader, Sampler

from nanodet_aabb_optimized.data.batch_process import stack_batch_img
from nanodet_aabb_optimized.data.dataset.coco import CocoDataset
from nanodet_aabb_optimized.evaluator.coco_detection import CocoDetectionEvaluator
from nanodet_aabb_optimized.trainer.task import TrainingTask
from nanodet_aabb_optimized.util.yacs import CfgNode


_ARCHIVE = "nanodet-training-baseline-v1.tar"
_REAL_ARCHIVE = "nanodet-real-capture-train-v1.tar"
_INIT_WEIGHTS = "jp-v2-bs24-init.pt"

_TRAIN_PIPELINE = {
    "perspective": 0.0,
    "scale": [0.8, 1.2],
    "stretch": [[0.9, 1.1], [0.9, 1.1]],
    "rotation": 0,
    "shear": 0,
    "translate": 0.08,
    "flip": 0.5,
    "brightness": 0.1,
    "contrast": [0.8, 1.2],
    "saturation": [0.8, 1.1],
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
    archives = [archive, archive.parent / _REAL_ARCHIVE]
    for current in archives:
        with tarfile.open(current, "r") as tf:
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
            "detach_epoch": 5,
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


class _DeploymentReplaySampler(Sampler[int]):
    """Expose every deployment entry plus a fresh deterministic jp_v2 subset each epoch."""

    def __init__(self, *, deployment_size: int, jp_size: int, jp_replay_images: int, seed: int) -> None:
        if deployment_size <= 0 or jp_size <= 0:
            raise ValueError("deployment and jp_v2 datasets must be non-empty")
        if jp_replay_images < 0 or jp_replay_images > jp_size:
            raise ValueError("jp_replay_images must be within the unique jp_v2 training set")
        self.deployment_size = int(deployment_size)
        self.jp_size = int(jp_size)
        self.jp_replay_images = int(jp_replay_images)
        self.seed = int(seed)
        self.epoch = 0

    def __len__(self) -> int:
        return self.deployment_size + self.jp_replay_images

    def __iter__(self):
        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch)
        deployment = torch.arange(self.deployment_size, dtype=torch.int64)
        if self.jp_replay_images:
            jp_local = torch.randperm(self.jp_size, generator=generator)[: self.jp_replay_images]
            jp = jp_local + self.deployment_size
            indices = torch.cat((deployment, jp))
        else:
            indices = deployment
        order = torch.randperm(indices.numel(), generator=generator)
        self.epoch += 1
        return iter(indices[order].tolist())


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

    jp_train = CocoDataset(
        img_path=str(root / "images" / "jp_train"),
        ann_path=str(root / "annotations" / "jp_train.coco.json"),
        input_size=(320, 320),
        pipeline=_TRAIN_PIPELINE,
        keep_ratio=False,
        mode="train",
    )
    composite_train = CocoDataset(
        img_path=str(root / "images" / "composite"),
        ann_path=str(root / "annotations" / "composite_train.coco.json"),
        input_size=(320, 320),
        pipeline=_TRAIN_PIPELINE,
        keep_ratio=False,
        mode="train",
    )
    real_train = CocoDataset(
        img_path=str(root / "images" / "real_capture"),
        ann_path=str(root / "annotations" / "real_train.coco.json"),
        input_size=(320, 320),
        pipeline=_TRAIN_PIPELINE,
        keep_ratio=False,
        mode="train",
    )
    deployment_dataset = ConcatDataset(
        [composite_train] * int(p["composite_repeat"])
        + [real_train] * int(p["real_repeat"])
    )
    train_dataset = ConcatDataset([deployment_dataset, jp_train])
    train_sampler = _DeploymentReplaySampler(
        deployment_size=len(deployment_dataset),
        jp_size=len(jp_train),
        jp_replay_images=int(p["jp_replay_images"]),
        seed=int(context.seed),
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
        shuffle=False,
        sampler=train_sampler,
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

    # Fine-tune starts from the canonical corrected-jp_v2 batch-24 Model.
    # The immutable state dict is part of the Corpus manifest so the Train Protocol
    # does not depend on mutable local paths or ClearML task state.
    initial_state = torch.load(
        context.corpus.root / _INIT_WEIGHTS,
        map_location="cpu",
        weights_only=True,
    )
    if not isinstance(initial_state, dict):
        raise ValueError("jp_v2 initialization weights are not a state dict")
    incompatible = context.model.load_state_dict(initial_state, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise ValueError("jp_v2 initialization weights do not match the NanoDet architecture")

    # MLDB owns the canonical architecture instance. Use it inside NanoDet's
    # native TrainingTask so EMA/optimizer/scheduler/evaluator remain upstream.
    task.model = context.model
    task.avg_model = copy.deepcopy(context.model)

    logger = _TelemetryLogger(context.telemetry)

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

    # Fine-tune budgets are compared at their terminal state. Selecting by jp_v2
    # validation mAP would bias checkpoint choice against deployment adaptation.
    if task.weight_averager is not None:
        final_state = task.weight_averager.state_dict()
    else:
        final_state = task.model.state_dict()
    incompatible = context.model.load_state_dict(final_state, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError("final NanoDet EMA state does not match the canonical architecture")
    return context.model.cpu()
