"""Run the trained YOLO detector over a phone video and return one VideoTrack per pothole."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

from .fuse import VideoTrack


def detect_video(
    video_path: str | Path,
    weights: str | Path,
    video_start_ms: int,
    conf: float = 0.35,
    imgsz: int = 640,
    min_frames: int = 3,
    device: int | str = 0,
    contact_y_frac: float = 1.0,
) -> list[VideoTrack]:
    """video_start_ms = epoch ms of the video's first frame (same clock as the CSV timestamps).

    contact_y_frac = how far down the frame the wheels meet the road (1.0 = bottom edge). For a
    windshield dashcam the bonnet hides the lower part of the frame, so use the bonnet line
    (~0.6). A pothole's contact time is when its box bottom first crosses this line.

    Frame time comes from the video's own presentation timestamps, so variable-frame-rate phone
    video stays aligned. Tracks seen in < min_frames frames are dropped as flicker.
    """
    import cv2  # lazy: serving env only
    from ultralytics import YOLO

    model = YOLO(str(weights))
    cap = cv2.VideoCapture(str(video_path))
    w, h = cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    seen: dict[int, list[tuple[float, float, float, float, float]]] = defaultdict(list)  # id -> (t_ms,conf,area,cx,y2)
    idx = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            t_ms = cap.get(cv2.CAP_PROP_POS_MSEC) or idx * 1000.0 / fps
            idx += 1
            r = model.track(frame, persist=True, conf=conf, imgsz=imgsz, device=device, verbose=False)[0]
            if r.boxes is None or r.boxes.id is None:
                continue
            for tid, xyxy, c in zip(r.boxes.id.int().tolist(), r.boxes.xyxy.tolist(), r.boxes.conf.tolist()):
                x1, y1, x2, y2 = xyxy
                seen[tid].append((t_ms, c, (x2 - x1) * (y2 - y1) / (w * h), (x1 + x2) / 2 / w, y2 / h))
    finally:
        cap.release()

    tracks = []
    for tid, obs in seen.items():
        if len(obs) < min_frames:
            continue
        crossed = next((o for o in obs if o[4] >= contact_y_frac), obs[-1])
        tracks.append(
            VideoTrack(
                track_id=tid,
                first_ms=int(video_start_ms + obs[0][0]),
                last_ms=int(video_start_ms + crossed[0]),
                conf=max(o[1] for o in obs),
                max_area_frac=max(o[2] for o in obs),
                cx_frac=obs[-1][3],
            )
        )
    return tracks
