import importlib.util
from pathlib import Path
import numpy as np
import torch
import pytest

MODULE = Path(__file__).resolve().parents[4] / 'mldb_data' / 'nanodet' / 'evaluation_protocols' / 't4-head-candidate-origin-v1.py'
spec=importlib.util.spec_from_file_location('head_origin',MODULE)
m=importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

@pytest.mark.parametrize('i,stride,row,col',[
 (0,8,0,0),(112,8,2,32),(114,8,2,34),(116,8,2,36),
 (1599,8,39,39),(1600,16,0,0),(1999,16,19,19),
 (2000,32,0,0),(2099,32,9,9),(2100,64,0,0),(2124,64,4,4)])
def test_origin_index(i,stride,row,col):
 p=m.origin(i)
 assert (p['stride'],p['row'],p['column'])==(stride,row,col)
 assert p['prior_x']==col*stride and p['prior_y']==row*stride

def test_head_decode_logit_and_box():
 arr=np.full((2125,33),-12.,dtype=np.float32)
 # center prior x256,y16, stride8: 112 index; regression logits 4 sides peak at bin 2.
 arr[112,0]=2.
 for offset in (1,9,17,25):
  arr[112,offset:offset+8]=-8
  arr[112,offset+2]=8
 decoded,scores=m.decode(arr,.35)
 assert len(decoded)==1
 assert decoded[0]['point_index']==112
 assert decoded[0]['region']=='completed-hand'
 assert scores[112] > .88

def test_nms_retains_adjacent_suppresses_same():
 one={'point_index':112,'score':.9,'box':dict(x1=10,y1=5,x2=30,y2=25),'region':'completed-hand','nms_status':'unprocessed','product_status':'unprocessed','nms_suppressed_by':None,'product_suppressed_by':None}
 twin={**one,'point_index':114,'score':.65}
 separate={**one,'point_index':116,'score':.80,'box':dict(x1=35,y1=5,x2=55,y2=25)}
 raw=[one,twin,separate]
 after=m.select_nms(raw)
 assert [d['point_index'] for d in after]==[112,116]
 assert twin['nms_suppressed_by']==112
 assert {d['point_index'] for d in m.product_filter(after)}=={112,116}
