"""Fetch trained detector weights so a fresh clone can run without training.

    python -m src.pothole.download_weights                      # uses POTHOLE_WEIGHTS_URL
    python -m src.pothole.download_weights --url https://.../best.pt [--sha256 <hash>]

Host best.pt as a GitHub Release asset (or Hugging Face Hub) - not in git history.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import urllib.request
from pathlib import Path

DEFAULT_DEST = Path("models/pothole/yolo11s/weights/best.pt")


def download(url: str, dest: Path = DEFAULT_DEST, sha256: str | None = None) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".part")
    urllib.request.urlretrieve(url, tmp)
    digest = hashlib.sha256(tmp.read_bytes()).hexdigest()
    if sha256 and digest != sha256.lower():
        tmp.unlink()
        raise SystemExit(f"checksum mismatch: expected {sha256}, got {digest}")
    tmp.replace(dest)
    print(f"saved {dest} (sha256 {digest})")
    return dest


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default=os.getenv("POTHOLE_WEIGHTS_URL"))
    ap.add_argument("--dest", type=Path, default=DEFAULT_DEST)
    ap.add_argument("--sha256")
    a = ap.parse_args()
    if not a.url:
        raise SystemExit("give --url or set POTHOLE_WEIGHTS_URL")
    download(a.url, a.dest, a.sha256)
