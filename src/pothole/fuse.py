"""Fuse video detections with accelerometer events (shared epoch-ms clock) and estimate depth.

A video track = one pothole seen over several frames (tracker id). The moment the wheels hit it
is when the box reaches the bottom of the frame, so we use the track's *last* frame time (plus a
tunable camera->wheel lag) as the contact time and look for an accelerometer event around it.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from .accel import Event


@dataclass
class VideoTrack:
    track_id: int
    first_ms: int
    last_ms: int           # last frame it was visible ~ wheel contact
    conf: float            # best confidence
    max_area_frac: float   # largest box area / frame area (size cue)
    cx_frac: float         # horizontal position at last sighting (0=left,1=right)


@dataclass
class Pothole:
    status: str                       # confirmed | video_only | accel_only
    t_ms: int
    conf: float | None = None
    box_area_frac: float | None = None
    peak_to_peak_ms2: float | None = None  # vertical acceleration swing, m/s^2 (not a length)
    speed_kmh: float | None = None
    severity: str | None = None
    depth_cm: float | None = None
    depth_note: str = ""
    lat: float | None = None
    lon: float | None = None


# --------------------------------------------------------------------------- depth
@dataclass
class DepthModel:
    """depth_cm = w . [p2p, speed_ms, area_frac] + b   (least squares on measured potholes).

    No guesses: until fitted on real tape-measured depths this refuses to output a number.
    """

    w: list[float] = field(default_factory=list)
    b: float = 0.0
    n: int = 0
    rmse_cm: float | None = None

    @property
    def fitted(self) -> bool:
        return bool(self.w)

    @staticmethod
    def _x(p2p, speed_ms, area_frac):
        return np.array([p2p, speed_ms, area_frac], float)

    @classmethod
    def fit(cls, rows: list[dict]) -> "DepthModel":
        """rows: {peak_to_peak, speed_ms, area_frac, depth_cm}. Needs >= 8 (3 features + slack)."""
        if len(rows) < 8:
            raise ValueError(f"need >= 8 measured potholes to fit depth, got {len(rows)}")
        X = np.array([cls._x(r["peak_to_peak"], r["speed_ms"], r["area_frac"]) for r in rows])
        y = np.array([r["depth_cm"] for r in rows], float)
        A = np.c_[X, np.ones(len(X))]
        coef, *_ = np.linalg.lstsq(A, y, rcond=None)
        rmse = float(np.sqrt(np.mean((A @ coef - y) ** 2)))
        return cls(list(map(float, coef[:-1])), float(coef[-1]), len(rows), rmse)

    def predict(self, p2p: float, speed_ms: float, area_frac: float) -> float:
        d = float(np.dot(self.w, self._x(p2p, speed_ms, area_frac)) + self.b)
        return max(d, 0.0)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "DepthModel":
        p = Path(path)
        return cls(**json.loads(p.read_text())) if p.exists() else cls()


# --------------------------------------------------------------------------- fusion
def fuse(
    tracks: list[VideoTrack],
    events: list[Event],
    window_s: float = 1.5,
    lag_s: float = 0.0,
    depth_model: DepthModel | None = None,
) -> list[Pothole]:
    """One-to-one nearest match within +-window_s of (track.last_ms + lag)."""
    depth_model = depth_model or DepthModel()
    used: set[int] = set()
    out: list[Pothole] = []

    def depth(ev: Event, area: float | None):
        if not depth_model.fitted or area is None:
            return None, "uncalibrated: measure ~10-20 real potholes with a tape and fit DepthModel"
        return round(depth_model.predict(ev.peak_to_peak, ev.speed_ms, area), 1), (
            f"calibrated on {depth_model.n} potholes, RMSE {depth_model.rmse_cm:.1f} cm"
        )

    for tr in sorted(tracks, key=lambda t: t.last_ms):
        contact = tr.last_ms + lag_s * 1000
        cand = [(abs(e.t_ms - contact), i) for i, e in enumerate(events)
                if i not in used and abs(e.t_ms - contact) <= window_s * 1000]
        if cand:
            _, i = min(cand)
            used.add(i)
            e = events[i]
            d, note = depth(e, tr.max_area_frac)
            out.append(Pothole("confirmed", e.t_ms, tr.conf, tr.max_area_frac, e.peak_to_peak,
                               e.speed_ms * 3.6, e.severity, d, note, e.lat, e.lon))
        else:
            out.append(Pothole("video_only", int(contact), tr.conf, tr.max_area_frac,
                               depth_note="seen on camera, no accelerometer response (avoided or shallow)"))
    for i, e in enumerate(events):
        if i not in used:
            out.append(Pothole("accel_only", e.t_ms, None, None, e.peak_to_peak, e.speed_ms * 3.6,
                               e.severity, None, "bump with no detection (speed bump, joint, or missed)",
                               e.lat, e.lon))
    return sorted(out, key=lambda p: p.t_ms)
