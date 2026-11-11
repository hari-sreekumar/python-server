import pytest

pytest.importorskip("fastapi")
pytest.importorskip("multipart")
from fastapi.testclient import TestClient

from src.pothole import api
from tests.test_accel import make_csv


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(api, "UPLOADS", tmp_path / "uploads")
    api.UPLOADS.mkdir()
    return TestClient(api.app)


def csv_bytes():
    c = make_csv()
    return c.getvalue().encode() if hasattr(c, "getvalue") else c


def test_session_flow_accel_only(client):
    sid = client.post("/sessions").json()["session_id"]
    r = client.put(f"/sessions/{sid}/accel", files={"file": ("a.csv", csv_bytes())})
    assert r.status_code == 200
    assert client.post(f"/sessions/{sid}/analyze", data={"speed_kmh": 30}).status_code == 202
    got = client.get(f"/sessions/{sid}").json()  # TestClient runs background tasks before returning
    assert got["status"] == "done", got
    assert len(got["result"]["potholes"]) >= 1
    assert client.delete(f"/sessions/{sid}").status_code == 204
    assert client.get(f"/sessions/{sid}").status_code == 404


def test_bad_csv_rejected(client):
    sid = client.post("/sessions").json()["session_id"]
    r = client.put(f"/sessions/{sid}/accel", files={"file": ("a.csv", b"a,b\n1,2\n")})
    assert r.status_code == 422


def test_video_requires_start_ms_and_type_checked(client):
    sid = client.post("/sessions").json()["session_id"]
    client.put(f"/sessions/{sid}/accel", files={"file": ("a.csv", csv_bytes())})
    assert client.put(f"/sessions/{sid}/video", files={"file": ("x.exe", b"123")}).status_code == 415
    assert client.put(f"/sessions/{sid}/video", files={"file": ("x.mp4", b"123")}).status_code == 200
    assert client.post(f"/sessions/{sid}/analyze").status_code == 422


def test_analyze_before_upload_and_bad_ids(client):
    sid = client.post("/sessions").json()["session_id"]
    assert client.post(f"/sessions/{sid}/analyze").status_code == 422
    assert client.get("/sessions/..%2f..%2fetc").status_code == 404
    assert client.get("/sessions/" + "0" * 32).status_code == 404


def test_one_shot_analyze(client):
    r = client.post("/analyze", files={"accel_csv": ("a.csv", csv_bytes())}, data={"speed_kmh": 30})
    assert r.status_code == 200
    assert "roughness_segments" in r.json()
