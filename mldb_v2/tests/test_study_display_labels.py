from __future__ import annotations

import pytest

from mldb_v2.src.study.display_labels import build_condition_labels


def test_condition_labels_expose_only_varying_source_parameter() -> None:
    labels = build_condition_labels(
        base_labels={
            "trial-0001": "NanoDet Plus M320",
            "trial-0002": "NanoDet Plus M320",
            "trial-0003": "NanoDet Plus M320",
        },
        conditions={
            "trial-0001": {"batch_size": 24, "jp_fraction": 0.0, "seed": 42},
            "trial-0002": {"batch_size": 24, "jp_fraction": 0.25, "seed": 42},
            "trial-0003": {"batch_size": 24, "jp_fraction": 0.5, "seed": 42},
        },
    )
    assert labels == {
        "trial-0001": "NanoDet Plus M320 | jp_fraction=0",
        "trial-0002": "NanoDet Plus M320 | jp_fraction=0.25",
        "trial-0003": "NanoDet Plus M320 | jp_fraction=0.5",
    }
    assert all("trial-" not in label for label in labels.values())


def test_condition_labels_fail_closed_when_duplicate_conditions_are_indistinguishable() -> None:
    with pytest.raises(ValueError, match="indistinguishable"):
        build_condition_labels(
            base_labels={"trial-0001": "Model A", "trial-0002": "Model A"},
            conditions={
                "trial-0001": {"batch_size": 24, "seed": 42},
                "trial-0002": {"batch_size": 24, "seed": 42},
            },
        )
