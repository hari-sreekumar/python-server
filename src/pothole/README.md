# Pothole analysis

Upload a **phone video** and an **accelerometer CSV** from the same drive. The service finds potholes in the
video (YOLO), finds bumps in the accelerometer signal, matches the two by timestamp, scores road roughness
per 100 m, optionally estimates depth, and can write a short report with an LLM (Groq) grounded in the PDFs
in `data/pdf`.

| Module | Job |
|---|---|
| `accel.py` | load CSV, auto-detect vertical axis, find bump events, roughness per segment, speed from GPS |
| `video.py` | YOLO + tracker over the video -> one track per pothole |
| `fuse.py` | match video tracks to accelerometer events (shared epoch-ms clock); `DepthModel` |
| `report.py` | LLM report (Groq) grounded in `data/pdf` via the RAG index |
| `api.py` | `POST /analyze` upload endpoint |
| `train/` | RDD2022 -> YOLO converter and training script |
| `download_weights.py` | fetch trained weights so you don't have to train |

## 1. Setup

Requires Python 3.12 and [uv](https://docs.astral.sh/uv/). The pothole stack uses its **own environment**
(the RAG chatbot env ships CPU-only PyTorch).

```bash
git clone <repo-url> && cd <repo>
uv venv .venv-pothole --python 3.12

# NVIDIA GPU (RTX 50-series needs CUDA 12.8+ wheels):
uv pip install --python .venv-pothole torch torchvision --index-url https://download.pytorch.org/whl/cu128
# No GPU? use the CPU build instead (inference only, training will be slow):
#   uv pip install --python .venv-pothole torch torchvision --index-url https://download.pytorch.org/whl/cpu

uv pip install --python .venv-pothole -r requirements-pothole.txt
cp .env.example .env        # then set GROQ_API_KEY
```

Check the GPU is visible (skip on CPU):

```bash
.venv-pothole/Scripts/python -c "import torch; print(torch.cuda.is_available())"   # Linux/macOS: .venv-pothole/bin/python
```

## 2. Get the detector weights (pick one)

**A. Use pre-trained weights** (fastest):
```bash
.venv-pothole/Scripts/python -m src.pothole.download_weights --url <release-asset-url>
```
Saved to `models/pothole/yolo11s/weights/best.pt` (override with `POTHOLE_WEIGHTS`).

**B. Train your own** on RDD2022 (CC BY 4.0). Took about 40 minutes for 60 epochs on an RTX 5070 Ti laptop (12 GB):

```bash
# 1. dataset: figshare 21431547. Per-country zips are inside one 13 GB archive; you only need a few.
#    Extract so you have  RDD2022/<Country>/train/{images,annotations/xmls}
# 2. convert VOC XML -> YOLO (D40 = pothole; add D00 D10 D20 to also learn cracks)
.venv-pothole/Scripts/python -m src.pothole.train.prepare_rdd2022 \
    --root RDD2022 --countries India Japan Czech United_States --classes D40
# 3. train (results + best.pt land in models/pothole/yolo11s/)
.venv-pothole/Scripts/python -m src.pothole.train.train --epochs 60
```
`train` prints mAP50 / mAP50-95 on the held-out validation split at the end.

Reference result (YOLO11s, 60 epochs, India+Japan+Czech+US, D40 only, 820 validation images):
precision 0.58, recall 0.47, **mAP50 0.50, mAP50-95 0.20**. Expect modest scores: RDD2022's "D40" class mixes
potholes with other surface damage and the labels are loose. On a clean, freshly tarred lane with tree
shadows (no real potholes) the default settings (`conf=0.35`, 3+ frames) produced zero detections, i.e. shadows
and roadside grass were rejected. Windows tip: if training dies silently, lower `--workers` (0-2).

## 3. Run the service

```bash
.venv-pothole/Scripts/uvicorn src.pothole.api:app --port 8001
```

```bash
curl -F accel_csv=@drive.csv -F video=@drive.mp4 -F video_start_ms=1789900440000 \
     -F contact_y_frac=0.6 -F report=true http://localhost:8001/analyze
```

| Field | Meaning |
|---|---|
| `accel_csv` | columns: `timestamp` (epoch ms), `accel_x/y/z`, `latitude`, `longitude`, `sample_rate_hz` |
| `video`, `video_start_ms` | video and the epoch-ms of its **first frame**, same clock as the CSV |
| `speed_kmh` | fallback speed when the GPS track doesn't move (a single repeated fix gives no speed) |
| `contact_y_frac` | how far down the frame the wheels meet the road. `1.0` = bottom edge; windshield dashcam with a visible bonnet: about `0.6` |
| `lag_s` | extra camera-to-wheel delay in seconds; tune per phone mount |
| `report` | `true` = also return an LLM-written report (`report_error` is set if the LLM call fails) |

### Step-by-step upload (large videos, phone apps)

`/analyze` waits for the whole run. For big videos, upload in pieces and poll:

```bash
SID=$(curl -s -X POST localhost:8001/sessions | python -c "import sys,json;print(json.load(sys.stdin)['session_id'])")
curl -X PUT  -F file=@drive.csv http://localhost:8001/sessions/$SID/accel
curl -X PUT  -F file=@drive.mp4 http://localhost:8001/sessions/$SID/video
curl -X POST -F video_start_ms=1789900440000 -F contact_y_frac=0.6 -F report=true http://localhost:8001/sessions/$SID/analyze   # 202
curl http://localhost:8001/sessions/$SID        # status: running | done | failed; "result" when done
curl -X DELETE http://localhost:8001/sessions/$SID
```

Uploads are streamed to `data/pothole/uploads/<session>/`. The CSV header is checked on upload (needs
`timestamp, accel_x, accel_y, accel_z`). Limits: `POTHOLE_MAX_VIDEO_MB` (1500), `POTHOLE_MAX_ACCEL_MB` (50).
Video types: mp4, mov, m4v, avi, mkv, webm. Only one detector run uses the GPU at a time. There is **no
authentication** yet, so keep the port on a trusted network.

Result statuses: **confirmed** (camera detection matched to an accelerometer spike), **video_only** (seen,
no spike: avoided or shallow), **accel_only** (spike with no detection, or no video supplied).

## 4. Recording tips

- Record video and accelerometer on the **same phone** (one clock). Note the video's start time.
- Keep GPS updating (about 1 Hz). Mount the phone rigidly; the vertical axis is auto-detected.
- Include a deliberate bump or clap-style event at the start to verify sync.

## 5. What is and isn't measured

- **Speed / distance**: from GPS (or the `speed_kmh` fallback).
- **Roughness**: RMS vertical acceleration and its speed-normalised form are *measured*. Converting to IRI
  (m/km) needs calibration against a reference profile (`accel.calibrate_linear`); until then values are relative.
- **Depth**: `DepthModel` refuses to output a number until fitted on at least 8 tape-measured potholes
  (`models/pothole/depth_model.json`). Camera and accelerometer alone cannot give calibrated depth.
- **Severity** is a relative band from spike size, not an engineering grade.

## 6. Tests

```bash
.venv-pothole/Scripts/python -m pytest tests -q
```

## 7. Licensing notes

- Dataset: RDD2022, CC BY 4.0 (cite the CRDDC'2022 paper).
- Detector library: Ultralytics is **AGPL-3.0**; check its terms before offering this as a hosted service.
- Don't commit personal videos/GPS traces, datasets, or weights (see `.gitignore`).
