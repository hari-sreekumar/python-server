"""Train the pothole detector (Ultralytics YOLO11) on the prepared dataset.

Needs a separate env with torch+CUDA (see src/pothole/README.md):
    python -m src.pothole.train.train --data data/pothole/yolo/data.yaml
"""

from __future__ import annotations

import argparse


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/pothole/yolo/data.yaml")
    ap.add_argument("--model", default="yolo11s.pt", help="yolo11n/s/m .pt (s fits 12GB easily)")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--project", default="models/pothole")
    ap.add_argument("--workers", type=int, default=4, help="dataloader workers (use 0 if Windows crashes)")
    ap.add_argument("--fraction", type=float, default=1.0, help="use a fraction of the data (smoke tests)")
    ap.add_argument("--name", default="yolo11s")
    a = ap.parse_args()

    import torch
    from ultralytics import YOLO

    if not torch.cuda.is_available():
        raise SystemExit(
            "CUDA not available. RTX 50-series needs a PyTorch build with CUDA >= 12.8 "
            "(pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128)."
        )
    print("GPU:", torch.cuda.get_device_name(0))
    from pathlib import Path

    project = str(Path(a.project).resolve())  # absolute: otherwise Ultralytics nests it under runs/detect/
    model = YOLO(a.model)
    model.train(
        data=a.data, epochs=a.epochs, imgsz=a.imgsz, batch=a.batch, device=0,
        project=project, name=a.name, patience=15, workers=a.workers, fraction=a.fraction,
        # road-camera specific: no vertical flip, mild colour/scale jitter
        flipud=0.0, fliplr=0.5, hsv_v=0.4, scale=0.4, mosaic=1.0,
    )
    metrics = model.val()
    print(f"mAP50={metrics.box.map50:.3f}  mAP50-95={metrics.box.map:.3f}")


if __name__ == "__main__":
    main()
