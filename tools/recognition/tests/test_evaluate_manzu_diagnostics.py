from __future__ import annotations

import numpy as np

from tools.recognition.evaluate_manzu_diagnostics import (
    preletterbox_content_extent,
    select_representatives,
    true_margin,
)


def test_true_margin_uses_strongest_competitor() -> None:
    logits = np.asarray([[5.0, 4.0, 1.0], [0.0, 3.0, 4.5]], dtype=np.float32)
    targets = np.asarray([0, 1], dtype=np.int64)
    np.testing.assert_allclose(true_margin(logits, targets), [1.0, -1.5])


def test_preletterbox_extent_preserves_aspect_ratio() -> None:
    assert preletterbox_content_extent(32, 64) == (0.5, 1.0)
    assert preletterbox_content_extent(64, 32) == (1.0, 0.5)


def test_representatives_include_neighbor_failure_and_are_distinct() -> None:
    labels = ["5m", "6m", "7m"]
    rows = [
        {"base_label": label, "class_index": class_index}
        for class_index, label in enumerate(labels)
        for _ in range(3)
    ]
    baseline = np.asarray(
        [
            [0.0, 1.0, -1.0], [4.0, 0.0, 0.0], [8.0, 0.0, 0.0],
            [0.0, 0.0, 1.0], [0.0, 4.0, 0.0], [0.0, 8.0, 0.0],
            [1.0, 0.0, 0.0], [0.0, 0.0, 4.0], [0.0, 0.0, 8.0],
        ],
        dtype=np.float32,
    )
    perturbed = baseline.copy()
    perturbed[1] = [0.0, 7.0, 0.0]
    perturbed[4] = [7.0, 0.0, 0.0]
    perturbed[7] = [0.0, 7.0, 0.0]
    picks = select_representatives(
        rows,
        baseline,
        labels,
        {"front-facing": baseline, "test-view": perturbed},
    )
    assert len(picks) == 9
    for offset in (0, 3, 6):
        group = picks[offset : offset + 3]
        assert len({item["row_index"] for item in group}) == 3
        fragile = next(item for item in group if item["difficulty"] == "neighbor-fragile")
        assert fragile["neighbor_failure"]["case"] == "test-view"
