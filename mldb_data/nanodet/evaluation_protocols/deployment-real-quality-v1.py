from __future__ import annotations

import json
import math
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader, Dataset

from mldb_v2.src.evaluation.evaluate_interface import EvaluationCandidate

INPUT_SIZE = 320
BGR_MEAN = torch.tensor((103.53, 116.28, 123.675), dtype=torch.float32).view(1, 3, 1, 1)
BGR_STD = torch.tensor((57.375, 57.12, 58.395), dtype=torch.float32).view(1, 3, 1, 1)
REGIONS = {
    "completed_hand": (7.0, 0.0, 313.0, 72.0),
    "dora_indicators": (7.0, 74.0, 313.0, 146.0),
    "melds": (74.0, 148.0, 246.0, 320.0),
}

@dataclass(frozen=True)
class Box:
    x1: float; y1: float; x2: float; y2: float
    @classmethod
    def from_xywh(cls, value: Sequence[float]) -> "Box":
        x, y, w, h = (float(v) for v in value)
        if w <= 0 or h <= 0 or not all(math.isfinite(v) for v in (x, y, w, h)):
            raise ValueError(f"invalid bbox: {value!r}")
        return cls(x, y, x + w, y + h)
    def xywh(self) -> list[float]:
        return [self.x1, self.y1, self.x2 - self.x1, self.y2 - self.y1]

@dataclass(frozen=True)
class Detection:
    box: Box; score: float; region: str

@dataclass(frozen=True)
class Sample:
    sample_id: str; source_file_name: str; image: torch.Tensor; ground_truths: tuple[Box, ...]

class RealDataset(Dataset):
    def __init__(self, path: Path, split: str, source_domain: str) -> None:
        with sqlite3.connect(path) as db:
            rows = db.execute(
                "SELECT sample_id,source_file_name,annotations_json,image_chw_u8 FROM sample WHERE split=? AND source_domain=? ORDER BY sample_id",
                (split, source_domain),
            ).fetchall()
        if not rows:
            raise ValueError(f"empty corpus selection: split={split} source_domain={source_domain}")
        self.rows = rows
    def __len__(self) -> int: return len(self.rows)
    def __getitem__(self, index: int) -> Sample:
        sample_id, source_file_name, annotations_raw, image_raw = self.rows[index]
        image = torch.from_numpy(np.frombuffer(image_raw, dtype=np.uint8).copy().reshape(3, INPUT_SIZE, INPUT_SIZE))
        annotations = json.loads(annotations_raw)
        return Sample(str(sample_id), str(source_file_name), image, tuple(Box.from_xywh(a["bbox"]) for a in annotations))

def _collate(samples: Sequence[Sample]):
    return torch.stack([s.image for s in samples]), list(samples)

def _preprocess(images: torch.Tensor, device: torch.device) -> torch.Tensor:
    bgr = images[:, [2, 1, 0]].to(device=device, dtype=torch.float32, non_blocking=device.type == "cuda")
    return (bgr - BGR_MEAN.to(device)) / BGR_STD.to(device)

def _area(b: Box) -> float: return max(0.0, b.x2-b.x1) * max(0.0, b.y2-b.y1)
def _intersection(a: Box, b: Box) -> float:
    return max(0.0, min(a.x2,b.x2)-max(a.x1,b.x1)) * max(0.0, min(a.y2,b.y2)-max(a.y1,b.y1))
def _iou(a: Box, b: Box) -> float:
    i = _intersection(a,b); u = _area(a)+_area(b)-i
    return 0.0 if u <= 0 else i/u
def _overlap_smaller(a: Box, b: Box) -> float:
    d=min(_area(a),_area(b)); return 0.0 if d <= 0 else _intersection(a,b)/d

def _region(box: Box) -> str | None:
    cx=(box.x1+box.x2)/2; cy=(box.y1+box.y2)/2
    for name,(x1,y1,x2,y2) in REGIONS.items():
        if x1 <= cx < x2 and y1 <= cy < y2: return name
    return None

def _model_spec(model: torch.nn.Module):
    head=getattr(model,"head",None); strides=tuple(int(x) for x in getattr(head,"strides",(8,16,32,64)))
    reg_max=int(getattr(head,"reg_max",7)); num_classes=int(getattr(head,"num_classes",1))
    if num_classes != 1: raise ValueError("deployment-real-quality requires one detector class")
    return strides,reg_max

def _priors(strides: Sequence[int], device: torch.device) -> torch.Tensor:
    chunks=[]
    for stride in strides:
        size=math.ceil(INPUT_SIZE/stride); coords=torch.arange(size,dtype=torch.float32,device=device)*float(stride)
        yy,xx=torch.meshgrid(coords,coords,indexing="ij")
        chunks.append(torch.stack((xx.reshape(-1),yy.reshape(-1),torch.full_like(xx.reshape(-1),float(stride))),dim=1))
    return torch.cat(chunks,dim=0)

def _nms(items: Sequence[Detection], threshold: float, maximum: int) -> list[Detection]:
    kept=[]
    for candidate in sorted(items,key=lambda d:d.score,reverse=True):
        if any(_iou(candidate.box,k.box)>threshold for k in kept): continue
        kept.append(candidate)
        if len(kept)>=maximum: break
    return kept

def _decode(output: torch.Tensor, strides: Sequence[int], reg_max: int, score_threshold: float, nms_iou: float, maximum: int):
    points=sum(math.ceil(INPUT_SIZE/s)**2 for s in strides); channels=1+4*(reg_max+1)
    if output.ndim != 3 or tuple(output.shape[1:]) != (points,channels): raise ValueError(f"unexpected detector output shape: {tuple(output.shape)}")
    priors=_priors(strides,output.device); scores=output[...,0].sigmoid()
    reg=output[...,1:].reshape(output.shape[0],points,4,reg_max+1).softmax(dim=-1)
    bins=torch.arange(reg_max+1,dtype=output.dtype,device=output.device)
    dist=(reg*bins).sum(dim=-1)*priors[None,:,2,None]
    x1=(priors[None,:,0]-dist[...,0]).clamp(0,INPUT_SIZE); y1=(priors[None,:,1]-dist[...,1]).clamp(0,INPUT_SIZE)
    x2=(priors[None,:,0]+dist[...,2]).clamp(0,INPUT_SIZE); y2=(priors[None,:,1]+dist[...,3]).clamp(0,INPUT_SIZE)
    result=[]
    for bi in range(output.shape[0]):
        ds=[]
        for idx in torch.nonzero(scores[bi]>score_threshold,as_tuple=False).squeeze(1).tolist():
            box=Box(float(x1[bi,idx]),float(y1[bi,idx]),float(x2[bi,idx]),float(y2[bi,idx])); region=_region(box)
            if region is not None and box.x2>box.x1 and box.y2>box.y1: ds.append(Detection(box,float(scores[bi,idx]),region))
        result.append(_nms(ds,nms_iou,maximum))
    return result

def _product_suppress(items: Sequence[Detection], threshold: float):
    retained=[]; bridge_removed=pair_removed=0
    for region in REGIONS:
        group=[d for d in items if d.region==region]; candidates=[]
        for candidate in group:
            smaller=[o for o in group if o is not candidate and _area(o.box)<_area(candidate.box) and _overlap_smaller(candidate.box,o.box)>=threshold]
            bridge=any(_overlap_smaller(smaller[i].box,smaller[j].box)<threshold for i in range(len(smaller)) for j in range(i+1,len(smaller)))
            if bridge: bridge_removed+=1
            else: candidates.append(candidate)
        kept=[]
        for candidate in sorted(candidates,key=lambda d:d.score,reverse=True):
            if any(_overlap_smaller(candidate.box,k.box)>=threshold for k in kept): pair_removed+=1; continue
            kept.append(candidate)
        retained.extend(kept)
    return retained,bridge_removed,pair_removed

def _match(predictions: Sequence[Detection], truths: Sequence[Box], threshold: float):
    unmatched=set(range(len(truths))); pairs=[]; false_positive=0
    for prediction in sorted(predictions,key=lambda d:d.score,reverse=True):
        choices=[(i,_iou(prediction.box,truths[i])) for i in unmatched]
        if not choices: false_positive+=1; continue
        idx,value=max(choices,key=lambda x:x[1])
        if value < threshold: false_positive+=1; continue
        unmatched.remove(idx); pairs.append((prediction,truths[idx],value))
    return pairs,false_positive,len(unmatched)

def _rate(tp:int,fp:int,fn:int):
    p=0.0 if tp+fp==0 else tp/(tp+fp); r=0.0 if tp+fn==0 else tp/(tp+fn); f=0.0 if p+r==0 else 2*p*r/(p+r)
    return p,r,f

def _percentile(values: Sequence[float], q: float) -> float:
    if not values: return 0.0
    ordered=sorted(float(v) for v in values); rank=max(1,math.ceil(q*len(ordered))); return ordered[rank-1]

def _pair_quality(pred: Box, gt: Box, iou: float):
    inter=_intersection(pred,gt); ga=_area(gt); pa=_area(pred); gw=gt.x2-gt.x1; gh=gt.y2-gt.y1
    return {
        "iou":iou,
        "gt_coverage":0.0 if ga<=0 else inter/ga,
        "crop_purity":0.0 if pa<=0 else inter/pa,
        "center_error_x":0.0 if gw<=0 else abs((pred.x1+pred.x2-gt.x1-gt.x2)/2)/gw,
        "center_error_y":0.0 if gh<=0 else abs((pred.y1+pred.y2-gt.y1-gt.y2)/2)/gh,
        "width_ratio":0.0 if gw<=0 else (pred.x2-pred.x1)/gw,
        "height_ratio":0.0 if gh<=0 else (pred.y2-pred.y1)/gh,
    }

def _quality_summary(rows: Sequence[dict[str,float]]):
    return {
        "iou_p10":_percentile([r["iou"] for r in rows],0.10),
        "gt_coverage_p10":_percentile([r["gt_coverage"] for r in rows],0.10),
        "crop_purity_p10":_percentile([r["crop_purity"] for r in rows],0.10),
        "center_error_x_p90":_percentile([r["center_error_x"] for r in rows],0.90),
        "center_error_y_p90":_percentile([r["center_error_y"] for r in rows],0.90),
        "width_ratio_median":_percentile([r["width_ratio"] for r in rows],0.50),
        "height_ratio_median":_percentile([r["height_ratio"] for r in rows],0.50),
    }

def _contact_sheet(path: Path, rows: Sequence[dict[str,Any]]) -> None:
    selected=sorted(rows,key=lambda r:(r["worst_quality"],-r["fn"],-r["fp"]))[:8]
    sheet=Image.new("RGB",(640,720),"white"); draw=ImageDraw.Draw(sheet)
    for i,row in enumerate(selected):
        image=Image.fromarray(row["image"].permute(1,2,0).numpy(),mode="RGB").resize((300,300))
        overlay=ImageDraw.Draw(image)
        for gt in row["ground_truths"]: overlay.rectangle(tuple(v*300/320 for v in (gt.x1,gt.y1,gt.x2,gt.y2)),outline="green",width=2)
        for pred in row["detections"]: overlay.rectangle(tuple(v*300/320 for v in (pred.box.x1,pred.box.y1,pred.box.x2,pred.box.y2)),outline="red",width=2)
        x=(i%2)*320+10; y=(i//2)*175+35; sheet.paste(image.resize((150,150)),(x,y)); draw.text((x,y-20),f"{Path(row['source_file_name']).name[:28]} fp={row['fp']} fn={row['fn']}",fill="black")
    sheet.save(path)

def evaluate(context):
    p=context.parameters; split=str(p["split"]); source_domain=str(p["source_domain"])
    dataset=RealDataset(context.corpus.root/"dataset.sqlite",split,source_domain)
    workers=int(p["workers"]); loader=DataLoader(dataset,batch_size=int(p["batch_size"]),shuffle=False,num_workers=workers,persistent_workers=workers>0,pin_memory=torch.cuda.is_available(),collate_fn=_collate)
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"); model=context.model.module.to(device).eval(); strides,reg_max=_model_spec(model)
    totals={r:{"tp":0,"fp":0,"fn":0} for r in REGIONS}; qualities={r:[] for r in REGIONS}; rows=[]; bridge_total=pair_total=0; clean=0
    with torch.inference_mode():
        for images,samples in loader:
            output=model(_preprocess(images,device)); output=output[0] if isinstance(output,(tuple,list)) and len(output)==1 else output
            decoded=_decode(output,strides,reg_max,float(p["score_threshold"]),float(p["nms_iou_threshold"]),int(p["max_detections"]))
            for sample,detections in zip(samples,decoded,strict=True):
                detections,bridges,pairs_removed=_product_suppress(detections,float(p["duplicate_overlap_threshold"])); bridge_total+=bridges; pair_total+=pairs_removed
                image_fp=image_fn=0; image_qualities=[]
                for region in REGIONS:
                    preds=[d for d in detections if d.region==region]; truths=[g for g in sample.ground_truths if _region(g)==region]
                    pairs,fp,fn=_match(preds,truths,float(p["match_iou_threshold"])); tp=len(pairs); totals[region]["tp"]+=tp; totals[region]["fp"]+=fp; totals[region]["fn"]+=fn; image_fp+=fp; image_fn+=fn
                    for pred,gt,iou in pairs:
                        q=_pair_quality(pred.box,gt,iou); qualities[region].append(q); image_qualities.append(q)
                if image_fp==0 and image_fn==0: clean+=1
                worst=min((min(q["gt_coverage"],q["crop_purity"]) for q in image_qualities),default=0.0)
                rows.append({"sample_id":sample.sample_id,"source_file_name":sample.source_file_name,"image":sample.image,"ground_truths":sample.ground_truths,"detections":detections,"fp":image_fp,"fn":image_fn,"worst_quality":worst})
    all_counts={k:sum(t[k] for t in totals.values()) for k in ("tp","fp","fn")}; precision,recall,f1=_rate(**all_counts); all_quality=[q for region in REGIONS for q in qualities[region]]; qs=_quality_summary(all_quality)
    metrics={"precision":precision,"recall":recall,"f1":f1,"clean_image_rate":clean/len(dataset),**qs,"merged_bridge_rejection_count":bridge_total,"pairwise_duplicate_rejection_count":pair_total}
    summary_regions={}
    for region in REGIONS:
        rp,rr,rf=_rate(**totals[region]); rq=_quality_summary(qualities[region]); summary_regions[region]={**totals[region],"precision":rp,"recall":rr,"f1":rf,**rq}
        metrics[f"{region}_precision"]=rp; metrics[f"{region}_recall"]=rr; metrics[f"{region}_f1"]=rf; metrics[f"{region}_gt_coverage_p10"]=rq["gt_coverage_p10"]; metrics[f"{region}_crop_purity_p10"]=rq["crop_purity_p10"]
    work=Path(context.work_dir); work.mkdir(parents=True,exist_ok=True); summary_path=work/'deployment-real-quality-summary.json'; details_path=work/'deployment-real-quality-per-image.jsonl'; sheet_path=work/'deployment-real-quality-worst.png'
    summary={"schema":"mjtensu.nanodet/deployment-real-quality/v1","split":split,"source_domain":source_domain,"parameters":dict(p),"overall":{**all_counts,"precision":precision,"recall":recall,"f1":f1,**qs,"clean_images":clean,"image_count":len(dataset)},"regions":summary_regions,"postprocess":{"merged_bridge_rejection_count":bridge_total,"pairwise_duplicate_rejection_count":pair_total}}
    summary_path.write_text(json.dumps(summary,indent=2,ensure_ascii=False)+'\n')
    with details_path.open('w') as h:
        for row in rows:
            h.write(json.dumps({"sample_id":row["sample_id"],"source_file_name":row["source_file_name"],"false_positive_count":row["fp"],"false_negative_count":row["fn"],"ground_truths":[g.xywh() for g in row["ground_truths"]],"predictions":[{"bbox":d.box.xywh(),"score":d.score,"region":d.region} for d in row["detections"]]},separators=(',',':'))+'\n')
    _contact_sheet(sheet_path,rows)
    return EvaluationCandidate(metrics=metrics,artifacts={"summary_json":summary_path,"per_image_details":details_path,"worst_case_contact_sheet":sheet_path})
