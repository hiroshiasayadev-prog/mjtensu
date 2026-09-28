from __future__ import annotations

import json
import runpy
from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]
DATA = ROOT / "mldb_data"


def test_c8_architecture_declares_historical_structure_and_owned_helper() -> None:
    definition = json.loads(
        (DATA / "tile-classifier" / "architectures" / "tile-c8-gray35-v1.yaml").read_text(
            encoding="utf-8"
        )
    )
    assert definition["task"] == "tile-classifier/tile-shape-classification-35-v1"
    assert definition["family"] == "c8-equivariant-cnn"
    assert definition["interface"]["input"]["shape"] == [1, 64, 64]
    assert definition["implementation"]["sources"][0]["path"] == (
        "mldb_data/tile-classifier/lib/c8_tile_shape_classifier_v1.py"
    )

    helper_path = DATA / "tile-classifier" / "lib" / "c8_tile_shape_classifier_v1.py"
    helper = runpy.run_path(str(helper_path))
    assert helper["C8_GROUP_SIZE"] == 8
    assert helper["C8_FIELDS"] == (8, 16, 32, 64)
    assert helper["CLASS_COUNT"] == 35


def test_c8_helper_preserves_source_topology_contract() -> None:
    source = (
        DATA / "tile-classifier" / "lib" / "c8_tile_shape_classifier_v1.py"
    ).read_text(encoding="utf-8")
    assert "gspaces.rot2dOnR2(C8_GROUP_SIZE)" in source
    assert "kernel_size = 5 if block_index == 0 else 3" in source
    assert "enn.PointwiseMaxPool(" in source
    assert "kernel_size=3" in source
    assert "stride=2" in source
    assert "padding=1" in source
    assert "self.group_pool = enn.GroupPooling(in_type)" in source
    assert "self.spatial_pool = nn.AdaptiveAvgPool2d(1)" in source
    assert "nn.Linear(hidden_channels, self.class_count)" in source
