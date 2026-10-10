"""t4-only NanoDet head-origin diagnostic; no retraining or classifier rerun.

Captures actual per-cell head output (2125x33), product hard NMS,
and the region-specific duplicate filter. The model's raw class channel is a
logit in Torch forward and a probability in the ONNX-exported product forward.
"""
from __future__ import annotations

import csv
import json
import math
import subprocess
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import torch

from mldb_v2.src.evaluation.evaluate_interface import EvaluationCandidate

SIZE = 320
LEVELS = ((8, 40), (16, 20), (32, 10), (64, 5))
REGIONS = (
    ("completed-hand", (7, 0, 313, 72)),
    ("dora-indicators", (7, 74, 313, 146)),
    ("melds", (74, 148, 246, 320)),
)
MEAN = np.array([103.53, 116.28, 123.675], dtype=np.float32)
STD = np.array([57.375, 57.12, 58.395], dtype=np.float32)
FOCUS_FRAME = 190
FOCUS_IDS = (112, 114, 116)
CONF_THRESHOLD = .35
NMS_THRESHOLD = .60
COVERAGE_THRESHOLD = .80
COLORS = {8: (75, 225, 75), 16: (245, 190, 35), 32: (180, 60, 240), 64: (50, 170, 255)}


def origin(point: int) -> dict:
    if not 0 <= point < 2125:
        raise ValueError(f"invalid point index {point}")
    offset = 0
    for stride, side in LEVELS:
        count = side * side
        if point < offset + count:
            within = point - offset
            return {"stride": stride, "row": within // side, "column": within % side,
                    "point_index": point, "prior_x": within % side * stride,
                    "prior_y": within // side * stride}
        offset += count
    raise ValueError(point)


def overlap(a: dict, b: dict) -> tuple[float, float]:
    x = max(0., min(a["x2"], b["x2"]) - max(a["x1"], b["x1"]))
    y = max(0., min(a["y2"], b["y2"]) - max(a["y1"], b["y1"]))
    intersection = x*y
    ar1 = max(0., a["x2"]-a["x1"]) * max(0., a["y2"]-a["y1"])
    ar2 = max(0., b["x2"]-b["x1"]) * max(0., b["y2"]-b["y1"])
    union = ar1 + ar2 - intersection
    return (intersection/union if union > 0 else 0.,
            intersection/min(ar1,ar2) if min(ar1,ar2)>0 else 0.)


def decode(head: np.ndarray, threshold: float) -> tuple[list[dict], np.ndarray]:
    if head.shape != (2125, 33):
        raise ValueError(f"head is not [2125,33]: {head.shape}")
    # IMPORTANT: Torch forward outputs class logit; ONNX forward applies sigmoid.
    scores = 1. / (1. + np.exp(-np.clip(head[:,0], -80, 80)))
    results = []
    for i in np.flatnonzero(scores >= threshold):
        i = int(i)
        p = origin(i)
        dist = []
        for offset in (1,9,17,25):
            logits = head[i,offset:offset+8].astype(np.float64)
            prob = np.exp(logits-np.max(logits))
            prob /= np.sum(prob)
            dist.append(float(np.dot(np.arange(8), prob)) * p["stride"])
        x1=max(0.,p["prior_x"]-dist[0])
        y1=max(0.,p["prior_y"]-dist[1])
        x2=min(float(SIZE),p["prior_x"]+dist[2])
        y2=min(float(SIZE),p["prior_y"]+dist[3])
        if x1>=x2 or y1>=y2:
            continue
        box={"x1":x1,"y1":y1,"x2":x2,"y2":y2}
        region=next((name for name,(lx,ly,rx,ry) in REGIONS if lx <= (x1+x2)/2 < rx and ly <= (y1+y2)/2 < ry),None)
        results.append({**p,"score":float(scores[i]),"raw_logit":float(head[i,0]),
                        "box":box, "region":region, "nms_status":"unprocessed",
                        "product_status":"unprocessed", "nms_suppressed_by":None,
                        "product_suppressed_by":None})
    return results, scores


def select_nms(candidates: list[dict]) -> list[dict]:
    accepted=[]
    for a in sorted(candidates,key=lambda x:(-x["score"],x["point_index"])):
        hit=next((b for b in accepted if overlap(a["box"],b["box"])[0]>NMS_THRESHOLD),None)
        if hit:
            a["nms_status"]="suppressed"
            a["nms_suppressed_by"]=hit["point_index"]
        elif len(accepted)>=200:
            a["nms_status"]="max_detections"
        else:
            a["nms_status"]="kept"
            accepted.append(a)
    return accepted


def product_filter(accepted: list[dict]) -> list[dict]:
    """Mirror product duplicate-suppression (merged-bridge + coverage gate)."""
    selected=[]
    for name,_ in REGIONS:
        group=[a for a in accepted if a["region"]==name]
        eligible=[]
        for a in group:
            aa=a["box"]
            area=(aa["x2"]-aa["x1"])*(aa["y2"]-aa["y1"])
            smaller=[b for b in group if b is not a
                     and (b["box"]["x2"]-b["box"]["x1"])*(b["box"]["y2"]-b["box"]["y1"])<area
                     and overlap(aa,b["box"])[1]>=COVERAGE_THRESHOLD]
            bridge=any(overlap(smaller[i]["box"],smaller[j]["box"])[1]<COVERAGE_THRESHOLD
                       for i in range(len(smaller)) for j in range(i+1,len(smaller)))
            if bridge:
                a["product_status"]="bridge_rejected"
            else:
                eligible.append(a)
        kept=[]
        for a in sorted(eligible,key=lambda x:(-x["score"],x["point_index"])):
            hit=next((b for b in kept if overlap(a["box"],b["box"])[1]>=COVERAGE_THRESHOLD),None)
            if hit:
                a["product_status"]="suppressed"
                a["product_suppressed_by"]=hit["point_index"]
            else:
                a["product_status"]="kept"
                kept.append(a)
        selected+=kept
    return selected


def _canonical(frame: np.ndarray, capture: dict) -> np.ndarray:
    logical=capture["logicalCapture"]
    aspect=logical["aspectRatio"]
    rotation=int(logical["rotation"])
    h,w=frame.shape[:2]
    target=(16/9) if aspect=="16:9" else (9/16)
    if w/h>target:
        ww=round(h*target);frame=frame[:,max(0,(w-ww)//2):max(0,(w-ww)//2)+ww]
    else:
        hh=round(w/target);frame=frame[max(0,(h-hh)//2):max(0,(h-hh)//2)+hh,:]
    src_units=(16,9) if aspect=="16:9" else (9,16)
    out_units=src_units if rotation==0 else (src_units[1],src_units[0])
    units=max(1,int(min(frame.shape[1]/src_units[0],frame.shape[0]/src_units[1],1280/max(out_units))))
    tw,th=units*out_units[0],units*out_units[1]
    if rotation==0:return cv2.resize(frame,(tw,th),interpolation=cv2.INTER_LINEAR)
    pre=cv2.resize(frame,(th,tw),interpolation=cv2.INTER_LINEAR)
    return cv2.rotate(pre,cv2.ROTATE_90_COUNTERCLOCKWISE if rotation==-90 else cv2.ROTATE_90_CLOCKWISE)


def composite(frame: np.ndarray,capture:dict)->np.ndarray:
    h,w=frame.shape[:2]
    result=np.zeros((SIZE,SIZE,3),dtype=np.uint8)
    regions=capture["recognitionRegions"]
    dest={"completed-hand":(7,0,306,72),"dora-indicators":(7,74,306,72),"melds":(74,148,172,172)}
    for name, (tx,ty,tw,th) in dest.items():
        info=regions[name]
        x=max(0,min(w-1,round(info["x"]*w)))
        y=max(0,min(h-1,round(info["y"]*h)))
        x2=max(x+1,min(w-1,round((info["x"]+info["width"])*w)))
        y2=max(y+1,min(h-1,round((info["y"]+info["height"])*h)))
        result[ty:ty+th,tx:tx+tw]=cv2.resize(frame[y:y2,x:x2],(tw,th),interpolation=cv2.INTER_LINEAR)
    return result


def _probe(frame: np.ndarray) -> torch.Tensor:
    # BGR input, exactly corresponding to frontend preprocessCompositeRgba.
    raw=(frame.astype(np.float32)-MEAN)/STD
    return torch.from_numpy(raw.transpose(2,0,1).copy()).unsqueeze(0)


def _heatmap(scores: np.ndarray, comp: np.ndarray, frame_idx:int)->np.ndarray:
    panels=[]
    for stride,side in LEVELS:
        start=sum(s*s for sidx,s in LEVELS if sidx<stride)
        heat=scores[start:start+side*side].reshape(side,side)
        panel=cv2.resize(np.uint8(np.clip(heat*255,0,255)),(320,320),interpolation=cv2.INTER_NEAREST)
        panel=cv2.applyColorMap(panel,cv2.COLORMAP_TURBO)
        cv2.putText(panel,f"stride={stride}  {side}x{side}",(9,23),cv2.FONT_HERSHEY_SIMPLEX,.65,(255,255,255),2,cv2.LINE_AA)
        panels.append(panel)
    pic=np.vstack([np.hstack(panels[:2]),np.hstack(panels[2:])])
    canvas=cv2.resize(comp,(640,640),interpolation=cv2.INTER_NEAREST)
    overlay=np.hstack([canvas,pic])
    cv2.putText(overlay,f"t4 frame={frame_idx}  focus: original class scores",(12,28),cv2.FONT_HERSHEY_SIMPLEX,.70,(255,255,255),2,cv2.LINE_AA)
    return overlay


def _render_frame(comp:np.ndarray,raw:list[dict],current:int)->np.ndarray:
    picture=cv2.resize(comp,(640,640),interpolation=cv2.INTER_NEAREST)
    for a in raw:
        if a["region"]!="completed-hand":continue
        if a["nms_status"]!="kept":continue
        color=COLORS[a["stride"]]
        box=a["box"]
        x1,y1,x2,y2=(round(box[k]*2) for k in ("x1","y1","x2","y2"))
        cv2.rectangle(picture,(x1,y1),(x2,y2),color,2,cv2.LINE_AA)
        cv2.putText(picture,f'{a["point_index"]} s{a["stride"]} {a["score"]:.2f}',(x1,max(16,y1-3)),cv2.FONT_HERSHEY_SIMPLEX,.42,color,1,cv2.LINE_AA)
        if current==FOCUS_FRAME and a["point_index"] in FOCUS_IDS:
            cv2.rectangle(picture,(x1,y1),(x2,y2),(0,0,255),3,cv2.LINE_AA)
    cv2.rectangle(picture,(0,600),(640,640),(0,0,0),-1)
    cv2.putText(picture,f't4 / frame {current} / {current/10:.1f}s / head output before product NMS',(7,622),cv2.FONT_HERSHEY_SIMPLEX,.45,(255,255,255),1,cv2.LINE_AA)
    return picture


def evaluate(context):
    if context.model is None or context.model.module is None:
        raise ValueError("existing NanoDet Model must be attached")
    path=Path(context.corpus.root)/"t4"/"video.mp4"
    capture_path=Path(context.corpus.root)/"t4"/"capture.json"
    if not path.is_file() or not capture_path.is_file():
        raise FileNotFoundError("t4 video/capture missing from canonical recognition corpus")
    capture=json.loads(capture_path.read_text())
    assert capture["logicalCapture"]["rotation"]==-90
    output_dir=Path(context.work_dir)/"head-origin-t4"
    output_dir.mkdir(parents=True,exist_ok=True)
    cap=cv2.VideoCapture(str(path))
    if not cap.isOpened():raise RuntimeError("cannot read t4 video")
    orientation=cv2.CAP_PROP_ORIENTATION_AUTO
    cap.set(orientation,0)
    rotation=int(round(cap.get(cv2.CAP_PROP_ORIENTATION_META)))%360
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model=context.model.module.to(device).eval()
    expected=tuple(int(i) for i in model.head.strides)
    if expected!=(8,16,32,64):raise ValueError(f"strides changed: {expected}")
    writer_path=output_dir/"head-candidate-origin-t4.mp4"
    video=None
    # Basic MP4V diagnostic; ffmpeg converts to H.264 when installed.
    raw_path=output_dir/"source-overlay.mp4"
    jsonl_path=output_dir/"candidates.jsonl"
    summary_path=output_dir/"summary.csv"
    focus_path=output_dir/"focus-frame-190.json"
    heat_path=output_dir/"stride-score-heatmap-frame-190.png"
    focus_image=output_dir/"head-origin-frame-190.png"
    counts=Counter()
    levels=Counter()
    focus=None
    video_frames=0
    try:
        with jsonl_path.open("w") as sink,torch.inference_mode():
            for frame_idx in range(299):
                # Original browser takes a 100ms sample from the 30fps source.
                source_frame=3*frame_idx
                if frame_idx:
                    if not cap.grab() or not cap.grab():
                        raise RuntimeError(f"cannot skip source frames to {source_frame}")
                ok,frame=cap.read()
                if not ok:raise RuntimeError(f"cannot read source frame {source_frame}")
                if rotation==90:frame=cv2.rotate(frame,cv2.ROTATE_90_CLOCKWISE)
                elif rotation==180:frame=cv2.rotate(frame,cv2.ROTATE_180)
                elif rotation==270:frame=cv2.rotate(frame,cv2.ROTATE_90_COUNTERCLOCKWISE)
                presented=_canonical(frame,capture)
                comp=composite(presented,capture)
                input_tensor=_probe(comp).to(device)
                output=model(input_tensor)
                if isinstance(output,(list,tuple)):output=output[0]
                raw_head=output[0].detach().float().cpu().numpy()
                raw,scores=decode(raw_head,CONF_THRESHOLD)
                nms=select_nms(raw)
                product=product_filter(nms)
                hand=[a for a in raw if a["region"]=="completed-hand"]
                counts.update({"frames":1,"raw_all":len(raw),"raw_completed":len(hand),
                               "nms_completed":sum(a["nms_status"]=="kept" for a in hand),
                               "product_completed":sum(a["product_status"]=="kept" for a in hand)})
                levels.update({str(a["stride"]):1 for a in hand})
                for a in hand:
                    sink.write(json.dumps({"frame":frame_idx,"source_frame":source_frame,"video_time_sec":frame_idx/10,**a},separators=(",",":"))+"\n")
                if video is None:
                    video=cv2.VideoWriter(str(raw_path),cv2.VideoWriter_fourcc(*"mp4v"),10,(640,640))
                    if not video.isOpened():raise RuntimeError("OpenCV mp4v encoder missing")
                img=_render_frame(comp,raw,frame_idx)
                video.write(img)
                video_frames+=1
                if frame_idx==FOCUS_FRAME:
                    cv2.imwrite(str(heat_path),_heatmap(scores,comp,frame_idx))
                    cv2.imwrite(str(focus_image),img)
                    focus={
                      "frame":frame_idx,"source_frame":source_frame,
                      "score_semantics":"sigmoid(Torch raw class logit); native ONNX output is sigmoid",
                      "head_shape":list(raw_head.shape),
                      "spotlight_point_indices":list(FOCUS_IDS),
                      "spotlight":[a for a in raw if a["point_index"] in FOCUS_IDS],
                      "same-region_other_candidates":len(hand),
                      "spotlight_pairwise":[{"left":a["point_index"],"right":b["point_index"],
                                             "iou":overlap(a["box"],b["box"])[0],
                                             "coverage_over_smaller":overlap(a["box"],b["box"])[1]}
                                            for i,a in enumerate(raw) for b in raw[i+1:]
                                            if a["point_index"] in FOCUS_IDS and b["point_index"] in FOCUS_IDS],
                      "reference_scores_from_existing_fixed_classifier_trace":{"112":0.73636,"114":0.62858,"116":0.71390},
                      "replay_score_differences":{str(a["point_index"]):a["score"]-expected
                         for a in raw for index,expected in ((112,0.73636),(114,0.62858),(116,0.71390))
                         if a["point_index"]==index},
                      "reference_evaluation_id":"nanodet/run-09b6de7dfb7e4add819a4eef0047bf81-trial-0002-eval-0003",
                      "limits":"This head predicts mahjong_tile only; 發 identity comes from the prior fixed classifier trace. No classifier rerun here.",
                    }
                    focus_path.write_text(json.dumps(focus,ensure_ascii=False,indent=2)+"\n")
                if frame_idx%50==0:
                    context.telemetry.report_scalar(group="t4-head-origin",series="completed_raw_candidates",value=int(len(hand)),step=frame_idx)
    finally:
        cap.release()
        if video is not None:video.release()
    if video_frames!=299 or focus is None:
        raise RuntimeError(f"incomplete t4 replay: {video_frames}")
    # Lossless mapping of original results; do not silently pass an unexpected model.
    if [a["point_index"] for a in focus["spotlight"]] != list(FOCUS_IDS):
        raise RuntimeError(f"Expected three focus point origins absent: {focus['spotlight']!r}")
    # If available, produce browser-compatible H264 without changing frame timing.
    import shutil
    if not shutil.which("ffmpeg"):
        raise RuntimeError("ffmpeg H.264 encoder is required for a playable ClearML video")
    subprocess.run(["ffmpeg","-hide_banner","-loglevel","error","-y","-i",str(raw_path),
                    "-an","-c:v","libx264","-preset","veryfast","-crf","27",
                    "-pix_fmt","yuv420p","-movflags","+faststart",str(writer_path)],check=True)
    raw_path.unlink()
    summary=[{"stride":str(s),"raw_completed_count":int(levels.get(str(s),0))}
             for s,_ in LEVELS]
    with summary_path.open("w",newline="") as fd:
        w=csv.DictWriter(fd,fieldnames=["stride","raw_completed_count"])
        w.writeheader();w.writerows(summary)
    if any(abs(delta)>.08 for delta in focus["replay_score_differences"].values()):
        raise RuntimeError("t4 replay is materially inconsistent with prior browser head scores")
    metrics={
      "t4_frames_processed":299,
      "raw_completed_candidates":int(counts["raw_completed"]),
      "nms_completed_candidates":int(counts["nms_completed"]),
      "product_completed_candidates":int(counts["product_completed"]),
      "focus_distinct_strides":len({a["stride"] for a in focus["spotlight"]}),
      "focus_candidates_kept_after_nms":sum(a["nms_status"]=="kept" for a in focus["spotlight"]),
      "focus_candidates_kept_after_product":sum(a["product_status"]=="kept" for a in focus["spotlight"]),
    }
    return EvaluationCandidate(metrics=metrics,artifacts={
      "candidate_origins":jsonl_path,
      "focus_details":focus_path,
      "stride_counts":summary_path,
      "stride_heatmap":heat_path,
      "focus_overlay":focus_image,
      "t4_origin_video":writer_path,
    })
