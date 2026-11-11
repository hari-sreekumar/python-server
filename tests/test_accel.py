import io

import numpy as np
import pytest

from src.pothole import accel


def make_csv(seconds=40, fs=50, speed_kmh=None, bumps=(10.0, 25.0), seed=0, gps_moves=False):
    """Synthetic drive in the user's CSV format: gravity on x, smooth then rough road."""
    rng = np.random.default_rng(seed)
    n = seconds * fs
    t = np.arange(n) / fs
    noise = np.where(t < 20, 0.25, 0.9)  # smooth first half, rough second half
    ax = 9.6 + rng.normal(0, 1, n) * noise
    ay = rng.normal(0, 0.2, n)
    az = 1.8 + rng.normal(0, 0.3, n)
    for b in bumps:  # damped oscillation like a wheel dropping into a hole
        i = int(b * fs)
        k = np.arange(25)
        ax[i : i + 25] += 7 * np.exp(-k / 6) * np.cos(k * 1.1)
    lat = np.full(n, 10.5177)
    lon = np.full(n, 76.2627)
    if gps_moves:  # ~10 m/s north, a fix every second
        lat = 10.5177 + (t // 1) * 10 / 111_320
    rows = ["timestamp,accel_x,accel_y,accel_z,latitude,longitude,sample_rate_hz"]
    for i in range(n):
        rows.append(f"{1789900440000 + int(t[i]*1000)},{ax[i]:.4f},{ay[i]:.4f},{az[i]:.4f},{lat[i]:.6f},{lon[i]:.6f},{fs}")
    return io.StringIO("\n".join(rows))


def test_detects_vertical_axis_as_gravity_axis():
    raw = accel.load_accel_csv(make_csv())
    assert accel.vertical_axis(raw) == "ax"


def test_single_gps_fix_requires_fallback_speed():
    with pytest.raises(ValueError, match="does not move"):
        accel.analyse(make_csv())


def test_events_found_at_bumps_not_in_rough_stretch():
    res = accel.analyse(make_csv(), speed_kmh=36)
    times = [e.t for e in res.events]
    assert any(abs(t - 10.0) < 0.5 for t in times)
    assert any(abs(t - 25.0) < 0.5 for t in times)
    assert len(res.events) <= 4  # rough stretch alone must not trigger events
    assert res.speed_source == "fallback"


def test_rough_segment_scores_higher_than_smooth():
    res = accel.analyse(make_csv(bumps=()), speed_kmh=36, seg_len_m=100)
    assert len(res.segments) >= 3
    assert res.segments[-1].rms_az > 2 * res.segments[0].rms_az


def test_gps_speed_used_when_track_moves():
    res = accel.analyse(make_csv(gps_moves=True))
    assert res.speed_source == "gps"
    assert 8 < res.distance_m / res.duration_s < 12  # ~10 m/s


def test_calibration_recovers_linear_relation():
    x = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    cal = accel.calibrate_linear(x, 12 * x + 1)
    assert cal.a == pytest.approx(12) and cal.b == pytest.approx(1) and cal.r2 == pytest.approx(1)
