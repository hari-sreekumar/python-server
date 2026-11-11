import pytest

from src.pothole.accel import Event
from src.pothole.fuse import DepthModel, VideoTrack, fuse


def ev(t_ms, p2p=8.0, v=10.0):
    return Event(t=0, t_ms=t_ms, peak_abs=p2p / 2, peak_to_peak=p2p, jerk=50, speed_ms=v, severity="medium")


def tr(last_ms, conf=0.8, area=0.05):
    return VideoTrack(1, last_ms - 1000, last_ms, conf, area, 0.5)


def test_statuses():
    out = fuse([tr(10_000), tr(50_000)], [ev(10_400), ev(90_000)])
    assert [p.status for p in out] == ["confirmed", "video_only", "accel_only"]


def test_one_event_matches_only_one_track():
    out = fuse([tr(10_000), tr(10_200)], [ev(10_100)])
    assert sorted(p.status for p in out) == ["confirmed", "video_only"]


def test_lag_shifts_match():
    assert fuse([tr(10_000)], [ev(12_000)], window_s=1.0)[0].status == "video_only"
    assert fuse([tr(10_000)], [ev(12_000)], window_s=1.0, lag_s=1.8)[0].status == "confirmed"


def test_depth_refused_until_calibrated():
    p = fuse([tr(10_000)], [ev(10_000)])[0]
    assert p.depth_cm is None and "uncalibrated" in p.depth_note


def test_depth_fit_and_predict(tmp_path):
    rows = [{"peak_to_peak": p, "speed_ms": v, "area_frac": a, "depth_cm": 0.5 * p - 0.1 * v + 20 * a + 1}
            for p, v, a in [(4, 8, .02), (6, 10, .03), (9, 12, .05), (12, 9, .08), (5, 6, .04),
                            (8, 14, .02), (10, 7, .06), (7, 11, .05), (11, 13, .07)]]
    m = DepthModel.fit(rows)
    assert m.rmse_cm < 1e-6
    m.save(tmp_path / "d.json")
    assert DepthModel.load(tmp_path / "d.json").predict(8, 10, 0.05) == pytest.approx(0.5 * 8 - 1 + 1 + 1, abs=1e-6)
    out = fuse([tr(10_000, area=0.05)], [ev(10_000, p2p=8, v=10)], depth_model=m)[0]
    assert out.depth_cm == pytest.approx(5.0, abs=0.1)


def test_depth_fit_needs_enough_data():
    with pytest.raises(ValueError):
        DepthModel.fit([{"peak_to_peak": 1, "speed_ms": 1, "area_frac": 1, "depth_cm": 1}] * 3)
