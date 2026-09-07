from __future__ import annotations

"""Perspective-aware geometry for PRODUCT-INV-RECOGNITION-013.

The current classifier corpus already contains production-shaped 64x64 grayscale
classifier inputs.  INV-013 therefore treats perspective augmentation as an explicit
synthetic proxy, not as a claim that the original camera image can be reconstructed.

A0 keeps the historical random360 path. A1 adds anisotropic affine geometry. A2 adds
structured projective trapezoid/keystone geometry. A3 embeds the cached crop on a larger
canvas, applies the projective transform, derives an axis-aligned transformed-content
bbox, jitters that bbox, and letterboxes the recrop back to 64x64.
"""

from dataclasses import dataclass
import hashlib
import math
from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F


GEOMETRY_UNIT_COUNT = 12


@dataclass(frozen=True)
class PerspectiveAugmentationSpec:
    name: str
    use_affine: bool
    use_perspective: bool
    detector_style_recrop: bool
    scale_x_min: float = 1.0
    scale_x_max: float = 1.0
    scale_y_min: float = 1.0
    scale_y_max: float = 1.0
    shear_x_max: float = 0.0
    shear_y_max: float = 0.0
    perspective_yaw_max: float = 0.0
    perspective_pitch_max: float = 0.0
    keystone_x_max: float = 0.0
    keystone_y_max: float = 0.0
    canvas_scale: float = 1.5
    bbox_center_jitter_fraction: float = 0.0
    bbox_scale_jitter_fraction: float = 0.0


AUGMENTATION_SPECS: dict[str, PerspectiveAugmentationSpec] = {
    "a0-random360": PerspectiveAugmentationSpec(
        name="a0-random360",
        use_affine=False,
        use_perspective=False,
        detector_style_recrop=False,
    ),
    "a1-anisotropic-affine": PerspectiveAugmentationSpec(
        name="a1-anisotropic-affine",
        use_affine=True,
        use_perspective=False,
        detector_style_recrop=False,
        scale_x_min=0.75,
        scale_x_max=1.25,
        scale_y_min=0.85,
        scale_y_max=1.15,
        shear_x_max=0.12,
        shear_y_max=0.08,
    ),
    "a2-perspective": PerspectiveAugmentationSpec(
        name="a2-perspective",
        use_affine=True,
        use_perspective=True,
        detector_style_recrop=False,
        scale_x_min=0.80,
        scale_x_max=1.20,
        scale_y_min=0.88,
        scale_y_max=1.12,
        shear_x_max=0.08,
        shear_y_max=0.06,
        perspective_yaw_max=0.15,
        perspective_pitch_max=0.15,
        keystone_x_max=0.08,
        keystone_y_max=0.08,
    ),
    "a3-perspective-recrop": PerspectiveAugmentationSpec(
        name="a3-perspective-recrop",
        use_affine=True,
        use_perspective=True,
        detector_style_recrop=True,
        scale_x_min=0.80,
        scale_x_max=1.20,
        scale_y_min=0.88,
        scale_y_max=1.12,
        shear_x_max=0.08,
        shear_y_max=0.06,
        perspective_yaw_max=0.15,
        perspective_pitch_max=0.15,
        keystone_x_max=0.08,
        keystone_y_max=0.08,
        canvas_scale=1.5,
        bbox_center_jitter_fraction=0.05,
        bbox_scale_jitter_fraction=0.06,
    ),
}


@dataclass(frozen=True)
class GeometryCase:
    name: str
    augmentation: str
    angle_deg: float = 0.0
    scale_x: float = 1.0
    scale_y: float = 1.0
    shear_x: float = 0.0
    shear_y: float = 0.0
    perspective_yaw: float = 0.0
    perspective_pitch: float = 0.0
    keystone_x: float = 0.0
    keystone_y: float = 0.0
    bbox_center_x: float = 0.0
    bbox_center_y: float = 0.0
    bbox_scale_x: float = 1.0
    bbox_scale_y: float = 1.0


PERSPECTIVE_EVALUATION_CASES: tuple[GeometryCase, ...] = (
    GeometryCase("front-facing", "a0-random360"),
    GeometryCase("affine-x-compress-0p85", "a1-anisotropic-affine", scale_x=0.85),
    GeometryCase("affine-x-compress-0p75", "a1-anisotropic-affine", scale_x=0.75),
    GeometryCase("affine-y-compress-0p85", "a1-anisotropic-affine", scale_y=0.85),
    GeometryCase("perspective-yaw-left-0p10", "a2-perspective", perspective_yaw=-0.10),
    GeometryCase("perspective-yaw-right-0p10", "a2-perspective", perspective_yaw=0.10),
    GeometryCase("perspective-yaw-right-0p15", "a2-perspective", perspective_yaw=0.15),
    GeometryCase("perspective-pitch-0p10", "a2-perspective", perspective_pitch=0.10),
    GeometryCase(
        "recrop-yaw-left-0p10-angle20",
        "a3-perspective-recrop",
        angle_deg=20.0,
        perspective_yaw=-0.10,
    ),
    GeometryCase(
        "recrop-yaw-right-0p10-angle20",
        "a3-perspective-recrop",
        angle_deg=20.0,
        perspective_yaw=0.10,
    ),
    GeometryCase(
        "recrop-yaw-right-0p15-jitter",
        "a3-perspective-recrop",
        perspective_yaw=0.15,
        bbox_center_x=0.04,
        bbox_center_y=-0.03,
        bbox_scale_x=1.05,
        bbox_scale_y=0.96,
    ),
)


def deterministic_geometry_units(
    sample_ids: Sequence[str],
    *,
    seed: int,
    epoch: int,
    stream: str,
    unit_count: int = GEOMETRY_UNIT_COUNT,
) -> np.ndarray:
    """Return stable [0,1) units keyed by sample id, epoch and experiment stream."""
    if unit_count < 1:
        raise ValueError("unit_count must be positive")
    output = np.empty((len(sample_ids), unit_count), dtype=np.float32)
    prefix = f"{seed}\0{epoch}\0{stream}\0".encode("utf-8")
    denominator = float(2**64)
    for row, sample_id in enumerate(sample_ids):
        digest = hashlib.shake_256(prefix + sample_id.encode("utf-8")).digest(
            unit_count * 8
        )
        for column in range(unit_count):
            start = column * 8
            integer = int.from_bytes(digest[start : start + 8], "big")
            output[row, column] = np.float32(integer / denominator)
    return output


def apply_training_geometry(
    images: torch.Tensor,
    *,
    spec: PerspectiveAugmentationSpec,
    angles_deg: torch.Tensor,
    units: torch.Tensor,
    content_extent_x: torch.Tensor | None = None,
    content_extent_y: torch.Tensor | None = None,
) -> torch.Tensor:
    if spec.name == "a0-random360":
        return _rotate_batch_compatible(images, angles_deg)
    if units.ndim != 2 or units.shape[0] != images.shape[0] or units.shape[1] < GEOMETRY_UNIT_COUNT:
        raise ValueError(
            f"Expected geometry units [N,{GEOMETRY_UNIT_COUNT}+], got {tuple(units.shape)}"
        )
    parameters = _parameters_from_units(spec, angles_deg=angles_deg, units=units)
    return apply_geometry_parameters(
        images,
        spec=spec,
        content_extent_x=content_extent_x,
        content_extent_y=content_extent_y,
        **parameters,
    )


def apply_evaluation_case(
    images: torch.Tensor,
    case: GeometryCase,
    *,
    content_extent_x: torch.Tensor | None = None,
    content_extent_y: torch.Tensor | None = None,
) -> torch.Tensor:
    spec = AUGMENTATION_SPECS[case.augmentation]
    batch = int(images.shape[0])
    device = images.device
    dtype = images.dtype

    def full(value: float) -> torch.Tensor:
        return torch.full((batch,), float(value), device=device, dtype=dtype)

    if spec.name == "a0-random360":
        if abs(case.angle_deg) < 1.0e-12:
            return images
        return _rotate_batch_compatible(images, full(case.angle_deg))
    return apply_geometry_parameters(
        images,
        spec=spec,
        angles_deg=full(case.angle_deg),
        scale_x=full(case.scale_x),
        scale_y=full(case.scale_y),
        shear_x=full(case.shear_x),
        shear_y=full(case.shear_y),
        perspective_yaw=full(case.perspective_yaw),
        perspective_pitch=full(case.perspective_pitch),
        keystone_x=full(case.keystone_x),
        keystone_y=full(case.keystone_y),
        bbox_center_x=full(case.bbox_center_x),
        bbox_center_y=full(case.bbox_center_y),
        bbox_scale_x=full(case.bbox_scale_x),
        bbox_scale_y=full(case.bbox_scale_y),
        content_extent_x=content_extent_x,
        content_extent_y=content_extent_y,
    )


def _parameters_from_units(
    spec: PerspectiveAugmentationSpec,
    *,
    angles_deg: torch.Tensor,
    units: torch.Tensor,
) -> dict[str, torch.Tensor]:
    def lerp(column: int, minimum: float, maximum: float) -> torch.Tensor:
        values = units[:, column]
        return minimum + values * (maximum - minimum)

    def signed(column: int, maximum: float) -> torch.Tensor:
        return (units[:, column] * 2.0 - 1.0) * maximum

    bbox_scale_delta = spec.bbox_scale_jitter_fraction
    return {
        "angles_deg": angles_deg,
        "scale_x": lerp(0, spec.scale_x_min, spec.scale_x_max),
        "scale_y": lerp(1, spec.scale_y_min, spec.scale_y_max),
        "shear_x": signed(2, spec.shear_x_max),
        "shear_y": signed(3, spec.shear_y_max),
        "perspective_yaw": signed(4, spec.perspective_yaw_max),
        "perspective_pitch": signed(5, spec.perspective_pitch_max),
        "keystone_x": signed(6, spec.keystone_x_max),
        "keystone_y": signed(7, spec.keystone_y_max),
        "bbox_center_x": signed(8, spec.bbox_center_jitter_fraction),
        "bbox_center_y": signed(9, spec.bbox_center_jitter_fraction),
        "bbox_scale_x": 1.0 + signed(10, bbox_scale_delta),
        "bbox_scale_y": 1.0 + signed(11, bbox_scale_delta),
    }


def apply_geometry_parameters(
    images: torch.Tensor,
    *,
    spec: PerspectiveAugmentationSpec,
    angles_deg: torch.Tensor,
    scale_x: torch.Tensor,
    scale_y: torch.Tensor,
    shear_x: torch.Tensor,
    shear_y: torch.Tensor,
    perspective_yaw: torch.Tensor,
    perspective_pitch: torch.Tensor,
    keystone_x: torch.Tensor,
    keystone_y: torch.Tensor,
    bbox_center_x: torch.Tensor,
    bbox_center_y: torch.Tensor,
    bbox_scale_x: torch.Tensor,
    bbox_scale_y: torch.Tensor,
    content_extent_x: torch.Tensor | None = None,
    content_extent_y: torch.Tensor | None = None,
) -> torch.Tensor:
    _validate_images(images)
    batch = int(images.shape[0])
    tensors = (
        angles_deg,
        scale_x,
        scale_y,
        shear_x,
        shear_y,
        perspective_yaw,
        perspective_pitch,
        keystone_x,
        keystone_y,
        bbox_center_x,
        bbox_center_y,
        bbox_scale_x,
        bbox_scale_y,
    )
    if any(tensor.shape != (batch,) for tensor in tensors):
        raise ValueError("Geometry parameter tensors must all have shape [N]")

    if spec.detector_style_recrop:
        return _apply_detector_style_recrop(
            images,
            spec=spec,
            angles_deg=angles_deg,
            scale_x=scale_x,
            scale_y=scale_y,
            shear_x=shear_x,
            shear_y=shear_y,
            perspective_yaw=perspective_yaw,
            perspective_pitch=perspective_pitch,
            keystone_x=keystone_x,
            keystone_y=keystone_y,
            bbox_center_x=bbox_center_x,
            bbox_center_y=bbox_center_y,
            bbox_scale_x=bbox_scale_x,
            bbox_scale_y=bbox_scale_y,
            content_extent_x=content_extent_x,
            content_extent_y=content_extent_y,
        )

    source_corners = _square_corners(
        batch,
        extent=1.0,
        device=images.device,
        dtype=images.dtype,
    )
    destination = _destination_quad(
        source_corners,
        angles_deg=angles_deg,
        scale_x=scale_x,
        scale_y=scale_y,
        shear_x=shear_x,
        shear_y=shear_y,
        perspective_yaw=perspective_yaw if spec.use_perspective else torch.zeros_like(perspective_yaw),
        perspective_pitch=(
            perspective_pitch if spec.use_perspective else torch.zeros_like(perspective_pitch)
        ),
        keystone_x=keystone_x if spec.use_perspective else torch.zeros_like(keystone_x),
        keystone_y=keystone_y if spec.use_perspective else torch.zeros_like(keystone_y),
    )
    homography = solve_homography(source_corners, destination)
    return warp_homography(
        images,
        homography,
        output_height=images.shape[-2],
        output_width=images.shape[-1],
    )


def _apply_detector_style_recrop(
    images: torch.Tensor,
    *,
    spec: PerspectiveAugmentationSpec,
    angles_deg: torch.Tensor,
    scale_x: torch.Tensor,
    scale_y: torch.Tensor,
    shear_x: torch.Tensor,
    shear_y: torch.Tensor,
    perspective_yaw: torch.Tensor,
    perspective_pitch: torch.Tensor,
    keystone_x: torch.Tensor,
    keystone_y: torch.Tensor,
    bbox_center_x: torch.Tensor,
    bbox_center_y: torch.Tensor,
    bbox_scale_x: torch.Tensor,
    bbox_scale_y: torch.Tensor,
    content_extent_x: torch.Tensor | None = None,
    content_extent_y: torch.Tensor | None = None,
) -> torch.Tensor:
    height, width = int(images.shape[-2]), int(images.shape[-1])
    canvas_height = max(height + 2, int(round(height * spec.canvas_scale)))
    canvas_width = max(width + 2, int(round(width * spec.canvas_scale)))
    pad_y_total = canvas_height - height
    pad_x_total = canvas_width - width
    pad_left = pad_x_total // 2
    pad_right = pad_x_total - pad_left
    pad_top = pad_y_total // 2
    pad_bottom = pad_y_total - pad_top
    canvas = F.pad(images, (pad_left, pad_right, pad_top, pad_bottom), mode="replicate")

    batch = int(images.shape[0])
    if content_extent_x is None:
        content_extent_x = torch.ones((batch,), device=images.device, dtype=images.dtype)
    if content_extent_y is None:
        content_extent_y = torch.ones((batch,), device=images.device, dtype=images.dtype)
    if content_extent_x.shape != (batch,) or content_extent_y.shape != (batch,):
        raise ValueError("content extents must have shape [N]")
    if torch.any(content_extent_x <= 0.0) or torch.any(content_extent_x > 1.0):
        raise ValueError("content_extent_x must be in (0, 1]")
    if torch.any(content_extent_y <= 0.0) or torch.any(content_extent_y > 1.0):
        raise ValueError("content_extent_y must be in (0, 1]")

    # content_extent_* describes the actual pre-letterbox crop content inside the
    # cached 64x64 classifier image. Convert that fraction to normalized coordinates
    # on the enlarged canvas. Passing no extents intentionally preserves INV-013 v1.
    extent_x = content_extent_x * (float(width) / float(canvas_width))
    extent_y = content_extent_y * (float(height) / float(canvas_height))
    source_corners = _rectangle_corners(
        batch,
        extent_x=1.0,
        extent_y=1.0,
        device=images.device,
        dtype=images.dtype,
    )
    source_corners[:, :, 0] *= extent_x[:, None]
    source_corners[:, :, 1] *= extent_y[:, None]
    destination = _destination_quad(
        source_corners,
        angles_deg=angles_deg,
        scale_x=scale_x,
        scale_y=scale_y,
        shear_x=shear_x,
        shear_y=shear_y,
        perspective_yaw=perspective_yaw,
        perspective_pitch=perspective_pitch,
        keystone_x=keystone_x,
        keystone_y=keystone_y,
    )
    homography = solve_homography(source_corners, destination)

    minimum = destination.amin(dim=1)
    maximum = destination.amax(dim=1)
    center = (minimum + maximum) * 0.5
    size = (maximum - minimum).clamp_min(1.0e-3)
    center[:, 0] += bbox_center_x * size[:, 0]
    center[:, 1] += bbox_center_y * size[:, 1]
    size[:, 0] *= bbox_scale_x.clamp_min(0.25)
    size[:, 1] *= bbox_scale_y.clamp_min(0.25)

    return _sample_letterboxed_bbox(
        canvas,
        homography=homography,
        bbox_center=center,
        bbox_size=size,
        output_height=height,
        output_width=width,
        fill=_border_median(images),
    )


def _destination_quad(
    source_corners: torch.Tensor,
    *,
    angles_deg: torch.Tensor,
    scale_x: torch.Tensor,
    scale_y: torch.Tensor,
    shear_x: torch.Tensor,
    shear_y: torch.Tensor,
    perspective_yaw: torch.Tensor,
    perspective_pitch: torch.Tensor,
    keystone_x: torch.Tensor,
    keystone_y: torch.Tensor,
) -> torch.Tensor:
    # Corner order is TL, TR, BR, BL.  The structured deltas form bounded
    # trapezoid/keystone geometry instead of four unrelated corner perturbations.
    destination = source_corners.clone()
    extent_x = source_corners[:, :, 0].abs().amax(dim=1)
    extent_y = source_corners[:, :, 1].abs().amax(dim=1)

    yaw = perspective_yaw * extent_y
    pitch = perspective_pitch * extent_x
    key_x = keystone_x * extent_x
    key_y = keystone_y * extent_y

    destination[:, 0, 1] -= yaw
    destination[:, 1, 1] += yaw
    destination[:, 2, 1] -= yaw
    destination[:, 3, 1] += yaw

    destination[:, 0, 0] += pitch
    destination[:, 1, 0] -= pitch
    destination[:, 2, 0] += pitch
    destination[:, 3, 0] -= pitch

    destination[:, 0, 0] += key_x
    destination[:, 1, 0] += key_x
    destination[:, 2, 0] -= key_x
    destination[:, 3, 0] -= key_x

    destination[:, 0, 1] += key_y
    destination[:, 3, 1] += key_y
    destination[:, 1, 1] -= key_y
    destination[:, 2, 1] -= key_y

    radians = angles_deg * (math.pi / 180.0)
    cosine = torch.cos(radians)
    sine = torch.sin(radians)
    affine = torch.zeros(
        (source_corners.shape[0], 2, 2),
        device=source_corners.device,
        dtype=source_corners.dtype,
    )
    affine[:, 0, 0] = cosine * scale_x - sine * shear_y
    affine[:, 0, 1] = cosine * shear_x - sine * scale_y
    affine[:, 1, 0] = sine * scale_x + cosine * shear_y
    affine[:, 1, 1] = sine * shear_x + cosine * scale_y
    return torch.bmm(destination, affine.transpose(1, 2))


def solve_homography(source: torch.Tensor, destination: torch.Tensor) -> torch.Tensor:
    """Solve batched source->destination homographies with h22 fixed to one."""
    if source.shape != destination.shape or source.ndim != 3 or source.shape[1:] != (4, 2):
        raise ValueError("source and destination must both be [N,4,2]")
    x = source[:, :, 0]
    y = source[:, :, 1]
    u = destination[:, :, 0]
    v = destination[:, :, 1]
    ones = torch.ones_like(x)
    zeros = torch.zeros_like(x)
    first = torch.stack((x, y, ones, zeros, zeros, zeros, -u * x, -u * y), dim=-1)
    second = torch.stack((zeros, zeros, zeros, x, y, ones, -v * x, -v * y), dim=-1)
    matrix = torch.stack((first, second), dim=2).reshape(source.shape[0], 8, 8)
    right_hand = torch.stack((u, v), dim=2).reshape(source.shape[0], 8, 1)
    solution = torch.linalg.solve(matrix, right_hand).squeeze(-1)
    homography = torch.zeros(
        (source.shape[0], 3, 3), device=source.device, dtype=source.dtype
    )
    homography[:, 0, 0:3] = solution[:, 0:3]
    homography[:, 1, 0:3] = solution[:, 3:6]
    homography[:, 2, 0:2] = solution[:, 6:8]
    homography[:, 2, 2] = 1.0
    return homography


def transform_points(homography: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
    if homography.ndim != 3 or homography.shape[1:] != (3, 3):
        raise ValueError("homography must be [N,3,3]")
    if points.ndim != 3 or points.shape[0] != homography.shape[0] or points.shape[2] != 2:
        raise ValueError("points must be [N,P,2]")
    ones = torch.ones((*points.shape[:2], 1), device=points.device, dtype=points.dtype)
    homogeneous = torch.cat((points, ones), dim=-1)
    transformed = torch.bmm(homogeneous, homography.transpose(1, 2))
    denominator = _safe_denominator(transformed[:, :, 2:3])
    return transformed[:, :, 0:2] / denominator


def warp_homography(
    images: torch.Tensor,
    homography: torch.Tensor,
    *,
    output_height: int,
    output_width: int,
) -> torch.Tensor:
    _validate_images(images)
    if homography.shape != (images.shape[0], 3, 3):
        raise ValueError("homography must have shape [N,3,3]")
    grid = _normalized_output_grid(
        images.shape[0],
        output_height,
        output_width,
        device=images.device,
        dtype=images.dtype,
    )
    inverse = torch.linalg.inv(homography)
    source = transform_points(inverse, grid.reshape(images.shape[0], -1, 2))
    source_grid = source.reshape(images.shape[0], output_height, output_width, 2)
    return F.grid_sample(
        images,
        source_grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=False,
    )


def _sample_letterboxed_bbox(
    source: torch.Tensor,
    *,
    homography: torch.Tensor,
    bbox_center: torch.Tensor,
    bbox_size: torch.Tensor,
    output_height: int,
    output_width: int,
    fill: torch.Tensor,
) -> torch.Tensor:
    batch = int(source.shape[0])
    grid = _normalized_output_grid(
        batch,
        output_height,
        output_width,
        device=source.device,
        dtype=source.dtype,
    )
    qx = grid[:, :, :, 0]
    qy = grid[:, :, :, 1]
    width = bbox_size[:, 0].clamp_min(1.0e-4)
    height = bbox_size[:, 1].clamp_min(1.0e-4)
    ratio_x = torch.minimum(torch.ones_like(width), width / height)
    ratio_y = torch.minimum(torch.ones_like(height), height / width)

    content_mask = (
        (qx.abs() <= ratio_x[:, None, None])
        & (qy.abs() <= ratio_y[:, None, None])
    )
    local_x = qx / ratio_x[:, None, None].clamp_min(1.0e-4)
    local_y = qy / ratio_y[:, None, None].clamp_min(1.0e-4)
    warped_x = bbox_center[:, 0, None, None] + 0.5 * width[:, None, None] * local_x
    warped_y = bbox_center[:, 1, None, None] + 0.5 * height[:, None, None] * local_y
    warped = torch.stack((warped_x, warped_y), dim=-1)

    inverse = torch.linalg.inv(homography)
    source_points = transform_points(inverse, warped.reshape(batch, -1, 2))
    source_grid = source_points.reshape(batch, output_height, output_width, 2)
    sampled = F.grid_sample(
        source,
        source_grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=False,
    )
    return torch.where(
        content_mask[:, None, :, :],
        sampled,
        fill[:, :, None, None],
    )


def _border_median(images: torch.Tensor) -> torch.Tensor:
    top = images[:, :, 0, :]
    bottom = images[:, :, -1, :]
    if images.shape[-2] > 2:
        left = images[:, :, 1:-1, 0]
        right = images[:, :, 1:-1, -1]
        border = torch.cat((top, bottom, left, right), dim=-1)
    else:
        border = torch.cat((top, bottom), dim=-1)
    return border.median(dim=-1).values


def _normalized_output_grid(
    batch: int,
    height: int,
    width: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    x = (torch.arange(width, device=device, dtype=dtype) + 0.5) * (2.0 / width) - 1.0
    y = (torch.arange(height, device=device, dtype=dtype) + 0.5) * (2.0 / height) - 1.0
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    return torch.stack((xx, yy), dim=-1).unsqueeze(0).expand(batch, -1, -1, -1)


def _square_corners(
    batch: int,
    *,
    extent: float,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    return _rectangle_corners(
        batch,
        extent_x=extent,
        extent_y=extent,
        device=device,
        dtype=dtype,
    )


def _rectangle_corners(
    batch: int,
    *,
    extent_x: float,
    extent_y: float,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    corners = torch.tensor(
        [
            [-extent_x, -extent_y],
            [extent_x, -extent_y],
            [extent_x, extent_y],
            [-extent_x, extent_y],
        ],
        device=device,
        dtype=dtype,
    )
    return corners.unsqueeze(0).expand(batch, -1, -1).clone()


def _safe_denominator(value: torch.Tensor) -> torch.Tensor:
    epsilon = torch.tensor(1.0e-6, device=value.device, dtype=value.dtype)
    sign = torch.where(value < 0, -torch.ones_like(value), torch.ones_like(value))
    return torch.where(value.abs() < epsilon, sign * epsilon, value)


def _rotate_batch_compatible(images: torch.Tensor, angles_deg: torch.Tensor) -> torch.Tensor:
    """Byte-for-byte formula match for INV-007 random360's affine_grid path."""
    radians = angles_deg.to(dtype=torch.float32) * (math.pi / 180.0)
    cosine = torch.cos(radians)
    sine = torch.sin(radians)
    theta = torch.zeros((images.shape[0], 2, 3), device=images.device, dtype=torch.float32)
    theta[:, 0, 0] = cosine
    theta[:, 0, 1] = -sine
    theta[:, 1, 0] = sine
    theta[:, 1, 1] = cosine
    grid = F.affine_grid(theta, images.shape, align_corners=False)
    return F.grid_sample(
        images,
        grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=False,
    )


def _validate_images(images: torch.Tensor) -> None:
    if images.ndim != 4 or images.shape[1] != 1:
        raise ValueError(f"Expected grayscale [N,1,H,W], got {tuple(images.shape)}")
    if not images.dtype.is_floating_point:
        raise ValueError("Perspective geometry expects floating-point images")
