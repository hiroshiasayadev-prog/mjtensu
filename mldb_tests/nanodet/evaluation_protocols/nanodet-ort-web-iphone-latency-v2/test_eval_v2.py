from pathlib import Path
from mldb_v2.src.evaluation.evaluate_interface import _load_evaluation_callable
ROOT=Path(__file__).resolve().parents[4]/"mldb_data"
def test_loader():
    assert callable(_load_evaluation_callable(ROOT,'nanodet/nanodet-ort-web-iphone-latency-v2'))
def test_idle_tunnel_not_rejected():
    src=(ROOT/'nanodet/evaluation_protocols/nanodet-ort-web-iphone-latency-v2.py').read_text()
    assert "health.get('tunnel_ready')" not in src
