import threading

from fastapi.testclient import TestClient

from forte_guard.api import create_app

VIDEO = {"file": ("synthetic.mov", b"synthetic")}


def blocked_pipeline(monkeypatch):
    release = threading.Event()
    monkeypatch.setattr("forte_guard.api.run_pipeline", lambda *args, **kwargs: release.wait(5))
    return release


def test_browser_queue_limit_does_not_apply_to_the_jury_key(settings, monkeypatch):
    release = blocked_pipeline(monkeypatch)
    settings.web_access = "open"
    settings.ui_max_queued = 2
    with TestClient(create_app(settings)) as client:
        client.get("/")
        assert [client.post("/api/v1/analyze", files=VIDEO).status_code for _ in range(3)] == [202, 202, 429]
        jury = client.post("/api/v1/analyze", files=VIDEO, headers={"X-API-Key": settings.api_key})
        assert jury.status_code == 202
        release.set()


def test_browser_hourly_limit(settings, monkeypatch):
    release = blocked_pipeline(monkeypatch)
    release.set()
    settings.web_access = "open"
    settings.ui_max_queued = 0
    settings.ui_max_uploads_per_hour = 2
    with TestClient(create_app(settings)) as client:
        client.get("/")
        codes = [client.post("/api/v1/analyze", files=VIDEO).status_code for _ in range(3)]
        assert codes == [202, 202, 429]
        assert "час" in client.post("/api/v1/analyze", files=VIDEO).json()["detail"]
