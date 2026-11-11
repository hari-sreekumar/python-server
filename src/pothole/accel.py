"""Accelerometer processing: load phone CSV, find bump/pothole events, score roughness.

CSV columns: timestamp (epoch ms), accel_x, accel_y, accel_z, latitude, longitude, sample_rate_hz
The vertical axis is auto-detected (the axis carrying gravity), because phone mounting varies.

Honesty note: roughness (IRI) and pothole depth cannot be derived exactly from a phone
accelerometer. This module returns measured quantities (RMS, peak-to-peak, speed) and only
converts them to IRI / depth when a calibration is supplied (see ``calibrate_linear``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import IO

import numpy as np
import pandas as pd
from scipy.signal import butter, find_peaks, sosfiltfilt

G = 9.80665
REQUIRED = ["timestamp", "accel_x", "accel_y", "accel_z"]


# --------------------------------------------------------------------------- loading
def load_accel_csv(src: str | IO) -> pd.DataFrame:
    """Load + clean the CSV. Returns columns: t_ms, t (s from start), ax, ay, az, lat, lon."""
    df = pd.read_csv(src)
    if "timestamp" not in df.columns and "timestamp_ms" in df.columns:  # alias used by some exports
        df = df.rename(columns={"timestamp_ms": "timestamp"})
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"CSV missing columns: {missing}")
    df = df.sort_values("timestamp").drop_duplicates("timestamp")
    out = pd.DataFrame(
        {
            "t_ms": df["timestamp"].astype("int64").to_numpy(),
            "ax": df["accel_x"].astype(float).to_numpy(),
            "ay": df["accel_y"].astype(float).to_numpy(),
            "az": df["accel_z"].astype(float).to_numpy(),
            "lat": df["latitude"].astype(float).to_numpy() if "latitude" in df else np.nan,
            "lon": df["longitude"].astype(float).to_numpy() if "longitude" in df else np.nan,
        }
    )
    out["t"] = (out["t_ms"] - out["t_ms"].iloc[0]) / 1000.0
    return out.reset_index(drop=True)


def resample_uniform(df: pd.DataFrame, fs: float | None = None) -> tuple[pd.DataFrame, float]:
    """Linear-interpolate onto a uniform grid (phones jitter). Returns (df, fs)."""
    dt = np.diff(df["t"].to_numpy())
    if fs is None:
        fs = 1.0 / float(np.median(dt))
    grid = np.arange(0.0, df["t"].iloc[-1], 1.0 / fs)
    out = pd.DataFrame({"t": grid})
    for c in ("ax", "ay", "az"):
        out[c] = np.interp(grid, df["t"], df[c])
    out["t_ms"] = df["t_ms"].iloc[0] + (grid * 1000).round().astype("int64")
    for c in ("lat", "lon"):  # GPS is a staircase of fixes: zero-order hold, never interpolate
        s = df.set_index("t")[c].dropna()
        if len(s):
            idx = np.searchsorted(s.index.to_numpy(), grid, side="right") - 1
            out[c] = np.where(idx >= 0, s.to_numpy()[np.clip(idx, 0, None)], np.nan)
        else:
            out[c] = np.nan
    return out, fs


# --------------------------------------------------------------------------- signal
def vertical_axis(df: pd.DataFrame) -> str:
    """Axis carrying gravity = largest |mean|."""
    means = {a: abs(df[a].mean()) for a in ("ax", "ay", "az")}
    return max(means, key=means.get)


def _sos(kind: str, cutoff, fs: float, order: int = 4):
    return butter(order, cutoff, btype=kind, fs=fs, output="sos")


def dynamic_vertical(df: pd.DataFrame, fs: float, lo_hz=0.5, hi_hz=None) -> np.ndarray:
    """Vertical acceleration with gravity/orientation drift (<lo_hz) and noise (>hi_hz) removed."""
    axis = vertical_axis(df)
    x = df[axis].to_numpy()
    hi = hi_hz or min(20.0, 0.45 * fs)
    return sosfiltfilt(_sos("bandpass", [lo_hz, hi], fs), x - x.mean())


# --------------------------------------------------------------------------- speed
def speed_ms(df: pd.DataFrame, fallback_kmh: float | None = None) -> tuple[np.ndarray, str]:
    """Speed (m/s) per sample from GPS; falls back to a constant if GPS doesn't move."""
    lat, lon, t = df["lat"].to_numpy(), df["lon"].to_numpy(), df["t"].to_numpy()
    ok = ~np.isnan(lat) & ~np.isnan(lon)
    if ok.sum() >= 2:
        la, lo, tt = np.radians(lat[ok]), np.radians(lon[ok]), t[ok]
        # distinct fixes only
        keep = np.r_[True, (np.diff(la) != 0) | (np.diff(lo) != 0)]
        if keep.sum() >= 3:
            la, lo, tt = la[keep], lo[keep], tt[keep]
            d = 6371000 * np.hypot(np.diff(la), np.diff(lo) * np.cos(la[:-1]))
            v = d / np.maximum(np.diff(tt), 1e-3)
            vmid = np.interp(t, 0.5 * (tt[:-1] + tt[1:]), v)
            # light smoothing, GPS speed is noisy
            k = max(1, int(len(vmid) // 50))
            vmid = pd.Series(vmid).rolling(k, center=True, min_periods=1).mean().to_numpy()
            return vmid, "gps"
    if fallback_kmh is None:
        raise ValueError(
            "GPS track does not move (single fix) and no fallback speed given - "
            "pass speed_kmh or record continuous GPS."
        )
    return np.full(len(df), fallback_kmh / 3.6), "fallback"


# --------------------------------------------------------------------------- events
@dataclass
class Event:
    t: float            # seconds from start
    t_ms: int           # epoch ms
    peak_abs: float     # max |vertical accel| in window (m/s^2)
    peak_to_peak: float # m/s^2
    jerk: float         # max |d a / dt| (m/s^3)
    speed_ms: float
    severity: str
    lat: float | None = None
    lon: float | None = None


def detect_events(
    df: pd.DataFrame,
    fs: float,
    speed: np.ndarray,
    k: float = 6.0,
    min_abs: float = 2.5,
    min_gap_s: float = 0.6,
    win_s: float = 0.3,
) -> list[Event]:
    """Detect bumps/potholes as outliers vs a *local* roughness baseline.

    Local baseline (rolling median of |a|) matters: a rough stretch has high background
    vibration, so a fixed threshold would flag the whole stretch.
    """
    dyn = dynamic_vertical(df, fs)
    env = np.abs(dyn)
    w = int(5 * fs) | 1
    base = pd.Series(env).rolling(w, center=True, min_periods=1).median().to_numpy() * 1.4826
    thr = np.maximum(min_abs, k * base)
    peaks, _ = find_peaks(env, height=thr[: len(env)], distance=int(min_gap_s * fs))
    jerk = np.abs(np.gradient(dyn, 1 / fs))
    half = int(win_s * fs)
    events = []
    for p in peaks:
        seg = dyn[max(0, p - half) : p + half + 1]
        p2p = float(seg.max() - seg.min())
        events.append(
            Event(
                t=float(df["t"].iloc[p]),
                t_ms=int(df["t_ms"].iloc[p]),
                peak_abs=float(env[p]),
                peak_to_peak=p2p,
                jerk=float(jerk[max(0, p - half) : p + half + 1].max()),
                speed_ms=float(speed[p]),
                severity=_severity(p2p),
                lat=None if np.isnan(df["lat"].iloc[p]) else float(df["lat"].iloc[p]),
                lon=None if np.isnan(df["lon"].iloc[p]) else float(df["lon"].iloc[p]),
            )
        )
    return events


def _severity(p2p: float) -> str:
    # Relative bands only (uncalibrated): tune with your own labelled drives.
    return "high" if p2p >= 12 else "medium" if p2p >= 6 else "low"


# --------------------------------------------------------------------------- roughness
@dataclass
class Segment:
    start_m: float
    end_m: float
    t_start: float
    t_end: float
    rms_az: float          # m/s^2, vertical, band-passed
    mean_speed_ms: float
    rms_per_speed: float   # speed-normalised roughness (1/s) - comparable across speeds
    iri_m_per_km: float | None = None  # only with calibration


def roughness_segments(
    df: pd.DataFrame, fs: float, speed: np.ndarray, seg_len_m: float = 100.0
) -> list[Segment]:
    dyn = dynamic_vertical(df, fs, lo_hz=0.5, hi_hz=min(15.0, 0.45 * fs))
    dist = np.concatenate([[0.0], np.cumsum(0.5 * (speed[1:] + speed[:-1]) / fs)])
    segs = []
    n = int(dist[-1] // seg_len_m) + (1 if dist[-1] % seg_len_m > 0.5 * seg_len_m else 0)
    for i in range(max(n, 1)):
        a, b = i * seg_len_m, (i + 1) * seg_len_m
        m = (dist >= a) & (dist < b if i < n - 1 else dist <= dist[-1])
        if m.sum() < fs:  # < 1 s of data
            continue
        rms = float(np.sqrt(np.mean(dyn[m] ** 2)))
        v = float(speed[m].mean())
        ts = df["t"].to_numpy()[m]
        segs.append(Segment(a, float(min(b, dist[-1])), float(ts[0]), float(ts[-1]), rms, v, rms / max(v, 0.5)))
    return segs


# --------------------------------------------------------------------------- calibration
@dataclass
class LinearCalibration:
    """y = a * x + b, fitted from (feature, measured) pairs. Used for IRI and depth."""

    a: float
    b: float
    n: int
    r2: float
    rmse: float

    def __call__(self, x):
        return self.a * np.asarray(x) + self.b


def calibrate_linear(x, y) -> LinearCalibration:
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 3:
        raise ValueError("need >= 3 reference points to calibrate")
    A = np.c_[x, np.ones_like(x)]
    (a, b), *_ = np.linalg.lstsq(A, y, rcond=None)
    pred = a * x + b
    ss_res, ss_tot = float(((y - pred) ** 2).sum()), float(((y - y.mean()) ** 2).sum())
    return LinearCalibration(float(a), float(b), len(x), 1 - ss_res / ss_tot if ss_tot else 0.0,
                             float(np.sqrt(ss_res / len(x))))


# --------------------------------------------------------------------------- pipeline
@dataclass
class AccelResult:
    fs: float
    vertical_axis: str
    speed_source: str
    duration_s: float
    distance_m: float
    events: list[Event] = field(default_factory=list)
    segments: list[Segment] = field(default_factory=list)


def analyse(
    src: str | IO,
    speed_kmh: float | None = None,
    seg_len_m: float = 100.0,
    iri_cal: LinearCalibration | None = None,
) -> AccelResult:
    raw = load_accel_csv(src)
    df, fs = resample_uniform(raw)
    v, source = speed_ms(df, speed_kmh)
    events = detect_events(df, fs, v)
    segs = roughness_segments(df, fs, v, seg_len_m)
    if iri_cal is not None:
        for s in segs:
            s.iri_m_per_km = float(iri_cal(s.rms_per_speed))
    dist = float(np.sum(0.5 * (v[1:] + v[:-1]) / fs))
    return AccelResult(fs, vertical_axis(df), source, float(df["t"].iloc[-1]), dist, events, segs)
