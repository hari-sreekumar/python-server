"""Upload API.   uvicorn src.pothole.api:app --port 8001

Two ways to use it:

A) One shot (small files, waits for the result)
   POST /analyze                      multipart: accel_csv, [video, video_start_ms, ...]

B) Step by step (large videos, phone apps that upload in pieces)
   POST /sessions                     -> {"session_id": "..."}
   PUT  /sessions/{id}/accel          raw CSV file  (multipart field `file`)
   PUT  /sessions/{id}/video          video file    (multipart field `file`, streamed to disk)
   POST /sessions/{id}/analyze        form: video_start_ms, speed_kmh, lag_s, contact_y_frac, report
                                      -> 202, runs in the background
   GET  /sessions/{id}                -> status: created | uploaded | running | done | failed (+ result)
   DELETE /sessions/{id}              remove the uploaded files

Uploads live in data/pothole/uploads/<session_id>/ (gitignored).
"""

from __future__ import annotations

import json
import os
import re
import shutil
import tempfile
import threading
import time
import uuid
from dataclasses import asdict
from pathlib import Path

from dotenv import load_dotenv
from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile

load_dotenv()

from . import accel as accel_mod  # noqa: E402
from .fuse import DepthModel, fuse  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
WEIGHTS = Path(os.getenv("POTHOLE_WEIGHTS", ROOT / "models/pothole/yolo11s/weights/best.pt"))
DEPTH_JSON = Path(os.getenv("POTHOLE_DEPTH_MODEL", ROOT / "models/pothole/depth_model.json"))
IRI_JSON = ROOT / "models/pothole/iri_cal.json"
UPLOADS = Path(os.getenv("POTHOLE_UPLOAD_DIR", ROOT / "data/pothole/uploads"))

MAX_ACCEL_MB = int(os.getenv("POTHOLE_MAX_ACCEL_MB", "50"))
MAX_VIDEO_MB = int(os.getenv("POTHOLE_MAX_VIDEO_MB", "1500"))
VIDEO_EXT = {".mp4", ".mov", ".m4v", ".avi", ".mkv", ".webm"}
REQUIRED_COLS = {"timestamp", "accel_x", "accel_y", "accel_z"}
_ID_RE = re.compile(r"^[0-9a-f]{32}$")

_gpu_lock = threading.Lock()  # one detector run at a time: a second job waits instead of OOM-ing the GPU

app = FastAPI(title="Pothole analysis")


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "weights": WEIGHTS.exists(), "depth_calibrated": DepthModel.load(DEPTH_JSON).fitted}


# --------------------------------------------------------------------------- core pipeline
def run_analysis(
    accel_src,
    video_path: Path | None,
    video_start_ms: int | None,
    speed_kmh: float | None,
    lag_s: float,
    contact_y_frac: float,
    report: bool,
) -> dict:
    """accel_src: path or file-like CSV. Raises ValueError for bad input, RuntimeError if weights missing."""
    res = accel_mod.analyse(accel_src, speed_kmh=speed_kmh)

    tracks = []
    if video_path is not None:
        if not WEIGHTS.exists():
            raise RuntimeError(f"detector weights not found at {WEIGHTS} - train or download them first")
        from .video import detect_video

        with _gpu_lock:
            tracks = detect_video(video_path, WEIGHTS, video_start_ms, contact_y_frac=contact_y_frac)

    potholes = fuse(tracks, res.events, lag_s=lag_s, depth_model=DepthModel.load(DEPTH_JSON))
    out = {
        "survey": {"duration_s": res.duration_s, "distance_m": res.distance_m,
                   "speed_source": res.speed_source, "vertical_axis": res.vertical_axis},
        "potholes": [asdict(p) for p in potholes],
        "roughness_segments": [asdict(s) for s in res.segments],
    }
    if report:
        from .report import build_payload, retrieve_references, write_report

        try:
            out["report"] = write_report(build_payload(res, potholes, IRI_JSON.exists()), retrieve_references())
        except Exception as e:  # keep the measured results even if the LLM step fails (billing, network)
            out["report"] = None
            out["report_error"] = f"{type(e).__name__}: {e}"
    return out


# --------------------------------------------------------------------------- upload helpers
async def _save_upload(up: UploadFile, dest: Path, max_mb: int) -> int:
    """Stream to disk in 1 MB chunks (videos are too big to hold in RAM); enforce a size cap."""
    limit = max_mb * 1024 * 1024
    size = 0
    tmp = dest.with_suffix(dest.suffix + ".part")
    try:
        with open(tmp, "wb") as f:
            while chunk := await up.read(1024 * 1024):
                size += len(chunk)
                if size > limit:
                    raise HTTPException(413, f"file larger than {max_mb} MB")
                f.write(chunk)
        if size == 0:
            raise HTTPException(422, "empty file")
        os.replace(tmp, dest)
    finally:
        tmp.unlink(missing_ok=True)
    return size


def _check_accel_header(path: Path) -> None:
    with open(path, "r", encoding="utf-8-sig", errors="replace") as f:
        header = f.readline()
    cols = {c.strip().lower() for c in header.split(",")}
    if "timestamp_ms" in cols:
        cols.add("timestamp")
    missing = REQUIRED_COLS - cols
    if missing:
        raise HTTPException(422, f"CSV is missing columns: {sorted(missing)}. Expected at least {sorted(REQUIRED_COLS)}")


def _sdir(session_id: str) -> Path:
    if not _ID_RE.match(session_id):  # also blocks path traversal
        raise HTTPException(404, "unknown session")
    d = UPLOADS / session_id
    if not d.is_dir():
        raise HTTPException(404, "unknown session")
    return d


def _meta(d: Path) -> dict:
    p = d / "meta.json"
    return json.loads(p.read_text()) if p.exists() else {}


def _set_meta(d: Path, **kw) -> dict:
    m = _meta(d)
    m.update(kw)
    tmp = d / "meta.json.tmp"
    tmp.write_text(json.dumps(m, indent=1))
    os.replace(tmp, d / "meta.json")
    return m


def _video_file(d: Path) -> Path | None:
    return next((p for p in d.glob("video.*") if p.suffix != ".part"), None)


# --------------------------------------------------------------------------- sessions
@app.post("/sessions", status_code=201)
def create_session() -> dict:
    sid = uuid.uuid4().hex
    d = UPLOADS / sid
    d.mkdir(parents=True)
    _set_meta(d, status="created", created=int(time.time()))
    return {"session_id": sid}


@app.put("/sessions/{session_id}/accel")
async def upload_accel(session_id: str, file: UploadFile = File(...)) -> dict:
    d = _sdir(session_id)
    if _meta(d).get("status") == "running":
        raise HTTPException(409, "analysis is running")
    size = await _save_upload(file, d / "accel.csv", MAX_ACCEL_MB)
    try:
        _check_accel_header(d / "accel.csv")
    except HTTPException:
        (d / "accel.csv").unlink(missing_ok=True)
        raise
    _set_meta(d, status="uploaded", accel_bytes=size)
    return {"session_id": session_id, "accel_bytes": size}


@app.put("/sessions/{session_id}/video")
async def upload_video(session_id: str, file: UploadFile = File(...)) -> dict:
    d = _sdir(session_id)
    if _meta(d).get("status") == "running":
        raise HTTPException(409, "analysis is running")
    ext = Path(file.filename or "").suffix.lower() or ".mp4"
    if ext not in VIDEO_EXT:
        raise HTTPException(415, f"unsupported video type {ext}; use one of {sorted(VIDEO_EXT)}")
    for old in d.glob("video.*"):  # replacing an earlier upload
        old.unlink(missing_ok=True)
    size = await _save_upload(file, d / f"video{ext}", MAX_VIDEO_MB)
    _set_meta(d, status="uploaded", video_bytes=size, video_name=file.filename)
    return {"session_id": session_id, "video_bytes": size}


def _job(d: Path, params: dict) -> None:
    _set_meta(d, status="running", started=int(time.time()))
    try:
        result = run_analysis(d / "accel.csv", _video_file(d), **params)
        (d / "result.json").write_text(json.dumps(result))
        _set_meta(d, status="done", finished=int(time.time()), error=None)
    except Exception as e:
        _set_meta(d, status="failed", finished=int(time.time()), error=f"{type(e).__name__}: {e}")


@app.post("/sessions/{session_id}/analyze", status_code=202)
def analyze_session(
    session_id: str,
    background: BackgroundTasks,
    video_start_ms: int | None = Form(None),
    speed_kmh: float | None = Form(None),
    lag_s: float = Form(0.0),
    contact_y_frac: float = Form(1.0),
    report: bool = Form(False),
) -> dict:
    d = _sdir(session_id)
    m = _meta(d)
    if m.get("status") == "running":
        raise HTTPException(409, "analysis already running")
    if not (d / "accel.csv").exists():
        raise HTTPException(422, "upload the accelerometer CSV first (PUT /sessions/{id}/accel)")
    if _video_file(d) is not None and video_start_ms is None:
        raise HTTPException(422, "video_start_ms is required when a video was uploaded")
    (d / "result.json").unlink(missing_ok=True)
    _set_meta(d, status="running", error=None)
    background.add_task(
        _job, d,
        dict(video_start_ms=video_start_ms, speed_kmh=speed_kmh, lag_s=lag_s,
             contact_y_frac=contact_y_frac, report=report),
    )
    return {"session_id": session_id, "status": "running"}


@app.get("/sessions/{session_id}")
def get_session(session_id: str) -> dict:
    d = _sdir(session_id)
    m = _meta(d)
    out = {
        "session_id": session_id,
        "status": m.get("status"),
        "has_accel": (d / "accel.csv").exists(),
        "has_video": _video_file(d) is not None,
        "error": m.get("error"),
    }
    if m.get("status") == "done" and (d / "result.json").exists():
        out["result"] = json.loads((d / "result.json").read_text())
    return out


@app.delete("/sessions/{session_id}", status_code=204)
def delete_session(session_id: str) -> None:
    d = _sdir(session_id)
    if _meta(d).get("status") == "running":
        raise HTTPException(409, "analysis is running")
    shutil.rmtree(d, ignore_errors=True)


# --------------------------------------------------------------------------- one-shot
@app.post("/analyze")
async def analyze(
    accel_csv: UploadFile = File(...),
    video: UploadFile | None = File(None),
    video_start_ms: int | None = Form(None),
    speed_kmh: float | None = Form(None),
    lag_s: float = Form(0.0),
    contact_y_frac: float = Form(1.0),
    report: bool = Form(False),
) -> dict:
    if video is not None and video_start_ms is None:
        raise HTTPException(422, "video_start_ms is required when a video is uploaded")
    with tempfile.TemporaryDirectory() as td:
        csv_path = Path(td) / "accel.csv"
        await _save_upload(accel_csv, csv_path, MAX_ACCEL_MB)
        _check_accel_header(csv_path)
        vp = None
        if video is not None:
            ext = Path(video.filename or "").suffix.lower() or ".mp4"
            if ext not in VIDEO_EXT:
                raise HTTPException(415, f"unsupported video type {ext}")
            vp = Path(td) / f"video{ext}"
            await _save_upload(video, vp, MAX_VIDEO_MB)
        try:
            return run_analysis(csv_path, vp, video_start_ms, speed_kmh, lag_s, contact_y_frac, report)
        except ValueError as e:
            raise HTTPException(422, str(e))
        except RuntimeError as e:
            raise HTTPException(503, str(e))
