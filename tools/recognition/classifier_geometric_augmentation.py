from __future__ import annotations

import torch
import torch.nn.functional as F


def projective_augment_batch(
    images: torch.Tensor,
    *,
    max_perspective: float,
    max_shear: float,
    max_stretch: float,
    probability: float = 1.0,
) -> torch.Tensor:
    """Apply mild per-sample projective distortion to an NCHW image batch.

    The transform is intentionally small and camera-like: independent x/y stretch,
    x/y shear, and a two-axis projective denominator.  Coordinates are expressed
    in the normalized grid_sample domain, so the same strengths work for 64x64
    classifier inputs without depending on pixel dimensions.
    """
    if images.ndim != 4:
        raise ValueError(f"Expected NCHW images, got {tuple(images.shape)}")
    if not 0.0 <= probability <= 1.0:
        raise ValueError("probability must be in [0,1]")
    for name, value, limit in (
        ("max_perspective", max_perspective, 0.25),
        ("max_shear", max_shear, 0.25),
        ("max_stretch", max_stretch, 0.30),
    ):
        if not 0.0 <= float(value) <= limit:
            raise ValueError(f"{name} must be in [0,{limit}]")
    if images.shape[0] == 0 or (
        max_perspective <= 0.0 and max_shear <= 0.0 and max_stretch <= 0.0
    ):
        return images

    batch, _channels, height, width = images.shape
    device = images.device
    dtype = torch.float32

    x = torch.linspace(-1.0, 1.0, width, device=device, dtype=dtype)
    y = torch.linspace(-1.0, 1.0, height, device=device, dtype=dtype)
    grid_y, grid_x = torch.meshgrid(y, x, indexing="ij")
    base_x = grid_x.unsqueeze(0).expand(batch, -1, -1)
    base_y = grid_y.unsqueeze(0).expand(batch, -1, -1)

    def random_signed(maximum: float) -> torch.Tensor:
        if maximum <= 0.0:
            return torch.zeros((batch, 1, 1), device=device, dtype=dtype)
        return torch.empty((batch, 1, 1), device=device, dtype=dtype).uniform_(
            -maximum, maximum
        )

    stretch_x = 1.0 + random_signed(float(max_stretch))
    stretch_y = 1.0 + random_signed(float(max_stretch))
    shear_x = random_signed(float(max_shear))
    shear_y = random_signed(float(max_shear))
    perspective_x = random_signed(float(max_perspective))
    perspective_y = random_signed(float(max_perspective))

    denominator = 1.0 + perspective_x * base_x + perspective_y * base_y
    denominator = denominator.clamp_min(0.5)
    sample_x = (stretch_x * base_x + shear_x * base_y) / denominator
    sample_y = (shear_y * base_x + stretch_y * base_y) / denominator
    grid = torch.stack((sample_x, sample_y), dim=-1)

    if probability < 1.0:
        apply_mask = torch.rand((batch, 1, 1, 1), device=device) < probability
        identity = torch.stack((base_x, base_y), dim=-1)
        grid = torch.where(apply_mask, grid, identity)

    return F.grid_sample(
        images,
        grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    )
