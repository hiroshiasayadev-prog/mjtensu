from __future__ import annotations

import pytest
import torch

from tools.recognition.classifier_geometric_augmentation import projective_augment_batch


def test_projective_augment_zero_strength_is_identity() -> None:
    images = torch.arange(2 * 1 * 8 * 8, dtype=torch.float32).reshape(2, 1, 8, 8)
    result = projective_augment_batch(
        images,
        max_perspective=0.0,
        max_shear=0.0,
        max_stretch=0.0,
        probability=1.0,
    )
    assert result is images


def test_projective_augment_probability_zero_preserves_pixels() -> None:
    images = torch.rand(3, 1, 16, 16)
    result = projective_augment_batch(
        images,
        max_perspective=0.08,
        max_shear=0.08,
        max_stretch=0.12,
        probability=0.0,
    )
    assert torch.allclose(result, images, atol=1.0e-6)


def test_projective_augment_preserves_shape_and_finite_values() -> None:
    torch.manual_seed(42)
    images = torch.rand(4, 1, 16, 16)
    result = projective_augment_batch(
        images,
        max_perspective=0.08,
        max_shear=0.08,
        max_stretch=0.12,
        probability=1.0,
    )
    assert result.shape == images.shape
    assert torch.isfinite(result).all()
    assert not torch.allclose(result, images)


def test_projective_augment_rejects_excessive_strength() -> None:
    images = torch.zeros(1, 1, 8, 8)
    with pytest.raises(ValueError):
        projective_augment_batch(
            images,
            max_perspective=0.5,
            max_shear=0.0,
            max_stretch=0.0,
        )
