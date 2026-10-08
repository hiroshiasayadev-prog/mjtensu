from pathlib import Path
from types import SimpleNamespace
import torch
from mldb_v2.src.training.train_interface import _load_train_callable

ROOT=Path(__file__).resolve().parents[4]/"mldb_data"
class Telemetry:
    def __init__(self): self.values=[]
    def report_scalar(self,**kw): self.values.append(kw)

def test_init_only():
    training=_load_train_callable(ROOT,'nanodet/nanodet-architecture-initialize-only-v1')
    model=torch.nn.Linear(3,4)
    tel=Telemetry()
    result=training(SimpleNamespace(model=model,telemetry=tel))
    assert result is model
    assert len(tel.values)==1
    assert tel.values[0]['series']=='parameters'
