from __future__ import annotations

from torch import nn

from nanodet_aabb_optimized.model.arch import build_model as build_nanodet_model
from nanodet_aabb_optimized.util.yacs import CfgNode


def build_nanodet_plus_aabb(
    *,
    use_depthwise: bool,
    use_res: bool,
    num_blocks: int,
) -> nn.Module:
    model_cfg = CfgNode(
        {
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
                    "num_blocks": int(num_blocks),
                    "use_res": bool(use_res),
                    "num_extra_level": 1,
                    "use_depthwise": bool(use_depthwise),
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
                        "loss_bbox": {
                            "name": "GIoULoss",
                            "loss_weight": 2.0,
                        },
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
            }
        },
        new_allowed=True,
    )
    return build_nanodet_model(model_cfg)
