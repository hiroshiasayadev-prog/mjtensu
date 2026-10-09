"""Draw immutable postprocessed source detections on already-sealed iPhone videos.

The orientation/canonical-frame routines below are copied unchanged from
recognition-functional-video-detector-v2, to keep the same presentation geometry
without importing a foreign Evaluation Protocol implementation at runtime.
"""
from __future__ import annotations
import hashlib
import json
import math
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Any
from mldb_data.nanodet.lib import nanodet_postprocess_trace_replay_v3 as replay

TAKES = ("t1","t2","t3","t4","t5")
REGION_COLORS = {"completed-hand":(230,60,230),"melds":(60,205,65),"dora-indicators":(230,160,40)}


def _open_raw_video(path: Path) -> Any:
    try:
        import cv2
    except ImportError as error:
        raise RuntimeError("opencv-python-headless is required for functional overlay rendering") from error

    source = cv2.VideoCapture(str(path))
    if not source.isOpened():
        raise RuntimeError(f"could not open functional video: {path}")
    orientation_auto = getattr(cv2, "CAP_PROP_ORIENTATION_AUTO", None)
    if orientation_auto is not None:
        source.set(orientation_auto, 0)
    return source

def _presentation_frame(frame: Any, source: Any, capture: dict[str, Any]) -> Any:
    try:
        import cv2
    except ImportError as error:
        raise RuntimeError("opencv-python-headless is required for functional overlay rendering") from error

    orientation_meta = getattr(cv2, "CAP_PROP_ORIENTATION_META", None)
    rotation = 0
    if orientation_meta is not None:
        value = float(source.get(orientation_meta))
        if math.isfinite(value):
            rotation = int(round(value)) % 360

    if rotation == 90:
        presented = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    elif rotation == 180:
        presented = cv2.rotate(frame, cv2.ROTATE_180)
    elif rotation == 270:
        presented = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    elif rotation == 0:
        presented = frame
    else:
        raise RuntimeError(f"unsupported MP4 display rotation: {rotation}")

    track = capture.get("trackSettings")
    if isinstance(track, dict):
        expected_width = track.get("width")
        expected_height = track.get("height")
        if (
            isinstance(expected_width, (int, float))
            and not isinstance(expected_width, bool)
            and isinstance(expected_height, (int, float))
            and not isinstance(expected_height, bool)
            and expected_width > 0
            and expected_height > 0
        ):
            actual_height, actual_width = presented.shape[:2]
            expected_aspect = float(expected_width) / float(expected_height)
            actual_aspect = float(actual_width) / float(actual_height)
            if not math.isclose(actual_aspect, expected_aspect, rel_tol=0.01, abs_tol=0.01):
                raise RuntimeError(
                    "decoded video presentation orientation does not match capture trackSettings: "
                    f"decoded={actual_width}x{actual_height} "
                    f"expected_aspect={float(expected_width):g}x{float(expected_height):g} "
                    f"mp4_rotation={rotation}"
                )
    return presented

def _canonical_frame(frame: Any, capture: dict[str, Any]) -> Any:
    try:
        import cv2
    except ImportError as error:
        raise RuntimeError("opencv-python-headless is required for functional overlay rendering") from error

    logical = capture["logicalCapture"]
    aspect = logical["aspectRatio"]
    rotation = int(logical["rotation"])
    source_h, source_w = frame.shape[:2]
    target_aspect = 16.0 / 9.0 if aspect == "16:9" else 9.0 / 16.0
    source_aspect = source_w / source_h
    if source_aspect > target_aspect:
        crop_w = int(round(source_h * target_aspect))
        x = max(0, (source_w - crop_w) // 2)
        cropped = frame[:, x : x + crop_w]
    else:
        crop_h = int(round(source_w / target_aspect))
        y = max(0, (source_h - crop_h) // 2)
        cropped = frame[y : y + crop_h, :]

    source_units = (16, 9) if aspect == "16:9" else (9, 16)
    output_units = source_units if rotation == 0 else (source_units[1], source_units[0])
    units = max(
        1,
        int(
            min(
                cropped.shape[1] / source_units[0],
                cropped.shape[0] / source_units[1],
                1280 / max(output_units),
            )
        ),
    )
    target_w = units * output_units[0]
    target_h = units * output_units[1]
    if rotation == 0:
        return cv2.resize(cropped, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

    pre_rotated = cv2.resize(
        cropped,
        (target_h, target_w),
        interpolation=cv2.INTER_LINEAR,
    )
    if rotation == 90:
        return cv2.rotate(pre_rotated, cv2.ROTATE_90_CLOCKWISE)
    if rotation == -90:
        return cv2.rotate(pre_rotated, cv2.ROTATE_90_COUNTERCLOCKWISE)
    raise ValueError(f"unsupported capture rotation: {rotation}")

def _recognition_region_rects(
    capture: dict[str, Any],
    width: int,
    height: int,
) -> list[tuple[str, tuple[int, int, int, int]]]:
    regions = capture.get("recognitionRegions")
    if not isinstance(regions, dict):
        raise RuntimeError("capture metadata has no recognitionRegions")

    result: list[tuple[str, tuple[int, int, int, int]]] = []
    for name in ("dora-indicators", "completed-hand", "melds"):
        region = regions.get(name)
        if not isinstance(region, dict):
            raise RuntimeError(f"capture metadata is missing recognition region: {name}")
        try:
            x = float(region["x"])
            y = float(region["y"])
            region_width = float(region["width"])
            region_height = float(region["height"])
        except (KeyError, TypeError, ValueError) as error:
            raise RuntimeError(f"malformed recognition region: {name}") from error
        x0 = max(0, min(width - 1, int(round(x * width))))
        y0 = max(0, min(height - 1, int(round(y * height))))
        x1 = max(x0 + 1, min(width - 1, int(round((x + region_width) * width))))
        y1 = max(y0 + 1, min(height - 1, int(round((y + region_height) * height))))
        result.append((name, (x0, y0, x1, y1)))
    return result

def _draw_recognition_regions(frame: Any, capture: dict[str, Any]) -> None:
    try:
        import cv2
    except ImportError as error:
        raise RuntimeError("opencv-python-headless is required for overlay rendering") from error

    colors = {
        "dora-indicators": (255, 140, 0),
        "completed-hand": (255, 0, 255),
        "melds": (0, 200, 0),
    }
    height, width = frame.shape[:2]
    for name, (x0, y0, x1, y1) in _recognition_region_rects(capture, width, height):
        color = colors[name]
        cv2.rectangle(frame, (x0, y0), (x1, y1), color, 2, cv2.LINE_AA)
        label_y = y0 + 20 if y0 < 24 else y0 - 6
        label_origin = (x0 + 4, max(18, label_y))
        cv2.putText(
            frame,
            name,
            label_origin,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (0, 0, 0),
            3,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            name,
            label_origin,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            color,
            1,
            cv2.LINE_AA,
        )

def _classification_label(value: object) -> str:
    if not isinstance(value, dict):
        return "?"
    if value.get("kind") != "tile":
        return "invalid"
    tile = value.get("tile")
    if not isinstance(tile, dict):
        return "?"
    kind = str(tile.get("kind", "?"))
    return kind + ("R" if tile.get("red") is True else "")

def _read_trace(trace_path: Path) -> dict[str,list[dict[str,Any]]]:
    result: dict[str,list[dict[str,Any]]] = defaultdict(list)
    for line in trace_path.open(encoding="utf-8"):
        if not line.strip():
            continue
        row=json.loads(line)
        take=row["take_id"]
        if take not in TAKES:
            raise ValueError(f"unexpected take {take!r}")
        result[take].append(row)
    if sum(map(len,result.values()))!=1497 or set(result)!=set(TAKES):
        raise ValueError("unexpected trace take/frame count")
    for take,rows in result.items():
        if any(rows[i]["source_frame"]>=rows[i+1]["source_frame"] for i in range(len(rows)-1)):
            raise ValueError(f"unordered source frames for {take}")
    return result


def _selected_detections(row: dict[str,Any], condition: str) -> list[dict[str,Any]]:
    if condition=="baseline":
        return row["detections"]
    for name,mode,value in replay.CASES:
        if name==condition:
            return [det for region in replay.REGIONS for det in replay.suppress(
                [d for d in row["detections"] if d["region"]==region],mode,value)]
    raise ValueError(f"unknown replay condition: {condition}")


def _draw_selected_overlay(frame: Any, row: dict[str,Any], capture: dict[str,Any],
                           condition: str, selected: list[dict[str,Any]]) -> Any:
    import cv2
    import numpy as np
    output=frame.copy()
    _draw_recognition_regions(output,capture)
    for det in selected:
        color=REGION_COLORS.get(det["region"],(255,255,0))
        box=det.get("sourceOrientedBox")
        if isinstance(box,dict):
            cx,cy,ww,hh=(float(box[k]) for k in ("cx","cy","width","height"))
            a=math.radians(float(box.get("angleDeg",0)))
            points=np.asarray([(
                round(cx+ux*math.cos(a)-uy*math.sin(a)),
                round(cy+ux*math.sin(a)+uy*math.cos(a))
            ) for ux,uy in ((-ww/2,-hh/2),(ww/2,-hh/2),(ww/2,hh/2),(-ww/2,hh/2))],dtype=np.int32)
            cv2.polylines(output,[points],True,color,2,cv2.LINE_AA)
            tx,ty=(int(v) for v in points[0])
        else:
            b=det["sourceBox"]
            tx,ty=round(b["x"]),round(b["y"])
            cv2.rectangle(output,(tx,ty),(round(b["x"]+b["width"]),round(b["y"]+b["height"])),color,2,cv2.LINE_AA)
        label=f'{_classification_label(det.get("classification"))} {det["confidence"]:.2f}'
        cv2.putText(output,label,(max(0,tx),max(70,ty-5)),cv2.FONT_HERSHEY_SIMPLEX,.44,(0,0,0),3,cv2.LINE_AA)
        cv2.putText(output,label,(max(0,tx),max(70,ty-5)),cv2.FONT_HERSHEY_SIMPLEX,.44,color,1,cv2.LINE_AA)
    predictions={r:replay.ordered_predictions(row,r,[d for d in selected if d["region"]==r])
                 for r in replay.REGIONS}
    hand_exact=predictions["completed-hand"]==row["detector_semantic"]["completed-hand"]["gt"]
    meld_exact=predictions["melds"]==row["detector_semantic"]["melds"]["gt"]
    removed=len(row["detections"])-len(selected)
    line1=f'{condition} | {row["take_id"]} {float(row["video_time_sec"]):.1f}s | src#{row["source_frame"]} | removed={removed}'
    line2=f'HAND exact:{"YES" if hand_exact else "NO"} | MELD exact:{"YES" if meld_exact else "NO"} | saved trace 10fps (no new inference)'
    cv2.rectangle(output,(0,0),(output.shape[1],70),(10,10,10),-1)
    cv2.putText(output,line1,(12,27),cv2.FONT_HERSHEY_SIMPLEX,.67,(255,255,255),2,cv2.LINE_AA)
    cv2.putText(output,line2,(12,55),cv2.FONT_HERSHEY_SIMPLEX,.54,(190,240,240),1,cv2.LINE_AA)
    return output


def render_condition_video(*, corpus_root: Path, trace_path: Path,
                           condition: str, destination: Path) -> dict[str,object]:
    import cv2
    if condition not in {row[0] for row in replay.CASES}:
        raise ValueError(f"unknown condition: {condition}")
    rows_by_take=_read_trace(trace_path)
    first_capture=json.loads((corpus_root/"t1"/"capture.json").read_text(encoding="utf-8"))
    cap=_open_raw_video(corpus_root/"t1"/"video.mp4")
    try:
        ok,frame=cap.read()
        if not ok:
            raise RuntimeError("cannot decode first video frame")
        canonical=_canonical_frame(_presentation_frame(frame,cap,first_capture),first_capture)
    finally:
        cap.release()
    height,width=canonical.shape[:2]
    destination.parent.mkdir(parents=True,exist_ok=True)
    encoder=subprocess.Popen([
        "ffmpeg","-hide_banner","-loglevel","error","-y","-f","rawvideo",
        "-pixel_format","bgr24","-video_size",f"{width}x{height}",
        "-framerate","10","-i","pipe:0","-an","-c:v","libx264",
        "-preset","veryfast","-crf","27","-threads","2",
        "-pix_fmt","yuv420p","-movflags","+faststart",str(destination),
    ],stdin=subprocess.PIPE,stdout=subprocess.DEVNULL,stderr=subprocess.PIPE)
    if encoder.stdin is None:
        raise RuntimeError("video encoder has no stdin")
    frames=0
    removed_total=0
    try:
        for take in TAKES:
            metadata=json.loads((corpus_root/take/"capture.json").read_text(encoding="utf-8"))
            source=_open_raw_video(corpus_root/take/"video.mp4")
            index=0
            try:
                for row in rows_by_take[take]:
                    target=int(row["source_frame"])
                    while index<target:
                        if not source.grab():
                            raise RuntimeError(f"cannot seek to frame {target} in {take}")
                        index+=1
                    ok,image=source.read()
                    if not ok:
                        raise RuntimeError(f"cannot decode source frame {take}/{target}")
                    index+=1
                    shown=_presentation_frame(image,source,metadata)
                    visual=_canonical_frame(shown,metadata)
                    if visual.shape!=(height,width,3):
                        raise RuntimeError(f"source shape mismatch: {take}/{target}")
                    detections=_selected_detections(row,condition)
                    removed_total+=len(row["detections"])-len(detections)
                    output=_draw_selected_overlay(visual,row,metadata,condition,detections)
                    encoder.stdin.write(output.tobytes())
                    frames+=1
            finally:
                source.release()
    except BaseException:
        encoder.kill()
        raise
    finally:
        encoder.stdin.close()
    returncode=encoder.wait(timeout=180)
    if returncode!=0 or not destination.is_file() or destination.stat().st_size==0:
        error=(encoder.stderr.read().decode(errors="replace") if encoder.stderr is not None else "")
        raise RuntimeError(f"video encoder failed exit={returncode} stderr={error[-900:]}")
    if frames!=1497:
        raise RuntimeError(f"incorrect encoded count {frames}")
    with destination.open("rb") as f:
        sha256=hashlib.file_digest(f,"sha256").hexdigest()
    return {"frames":frames,"fps":10,"width":width,"height":height,"removed":removed_total,
            "bytes":destination.stat().st_size,"sha256":sha256,"condition":condition}
