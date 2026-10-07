from mldb_v2.src.backend._clearml_project_routing import (
    _clearml_project_for_namespace,
    _clearml_project_for_reference,
)


def test_clearml_project_routing_defaults_to_namespace_project() -> None:
    assert _clearml_project_for_namespace("tile-classifier") == "mldb/tile-classifier"
    assert _clearml_project_for_reference("tile-classifier/run-abcd") == "mldb/tile-classifier"


def test_clearml_project_routing_groups_nanodet_under_tile_detector() -> None:
    assert _clearml_project_for_namespace("nanodet") == "mldb/tile-detector"
    assert _clearml_project_for_reference("nanodet/run-abcd") == "mldb/tile-detector"
