from torch import nn

from ..lib.plain_width_late_grid_v1 import build_plain_width_late_classifier


def build() -> nn.Module:
    return build_plain_width_late_classifier((20, 40, 80, 120), late_channels=None)
