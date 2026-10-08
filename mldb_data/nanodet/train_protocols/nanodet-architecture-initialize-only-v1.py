"""Return a fresh, untrained NanoDet model solely for execution-cost evaluation.

No dataset reads, GPU calls, forward/backward steps, or optimizer updates.
"""
from torch import nn

def train(context) -> nn.Module:
    model = context.model
    parameters = sum(p.numel() for p in model.parameters())
    context.telemetry.report_scalar(group="initialization", series="parameters", value=int(parameters), step=0)
    return model
