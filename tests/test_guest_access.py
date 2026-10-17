import threading

from fastapi.testclient import TestClient

from forte_guard.api import create_app

VIDEO = {"file": ("synthetic.mov", b"synthetic")}


def guest_app(settings, monkeypatch):
    release = threading.Event()
    monkeypatch.setattr("forte_guard.api.run_pipeline", lambda *args, **kwargs: release.wait(5))
    settings.web_access = "guest"
    return create_app(settings), release


def test_guest_uploads_without_a_key_and_sees_only_own_jobs(settings, monkeypatch):
    app, release = guest_app(settings, monkeypatch)
    key = {"X-API-Key": settings.api_key}
    with TestClient(app) as alice, TestClient(app) as bob:
        assert alice.get("/api/v1/access").json() == {"mode": "guest", "authorized": False, "guest": False}
        page = alice.get("/")
        cookie = page.headers["set-cookie"].lower()
        assert "forte_guest=" in cookie and "httponly" in cookie and "samesite=lax" in cookie
        assert settings.api_key not in page.headers["set-cookie"]
        assert alice.get("/api/v1/access").json()["guest"] is True
        bob.get("/")
        mine = alice.post("/api/v1/analyze", files=VIDEO).json()["job_id"]
        jury = alice.post("/api/v1/analyze", files=VIDEO, headers=key).json()["job_id"]

        assert [job["job_id"] for job in alice.get("/api/v1/jobs").json()["jobs"]] == [mine]
        assert bob.get("/api/v1/jobs").json()["jobs"] == []
        assert {job["job_id"] for job in alice.get("/api/v1/jobs", headers=key).json()["jobs"]} == {mine, jury}

        assert alice.get(f"/api/v1/jobs/{mine}").status_code == 200
        for path in ["", "/video", "/frames/f000001", "/artifacts/summary"]:
            assert bob.get(f"/api/v1/jobs/{mine}{path}").status_code == 404
            assert alice.get(f"/api/v1/jobs/{jury}{path}").status_code == 404
        assert alice.get(f"/api/v1/jobs/{jury}", headers=key).status_code == 200
        release.set()


def test_guest_session_does_not_open_team_endpoints(settings, monkeypatch):
    app, release = guest_app(settings, monkeypatch)
    with TestClient(app) as client:
        client.get("/")
        for path in ["/api/v1/config", "/api/v1/benchmarks", "/api/v1/readiness", "/api/v1/deployment"]:
            assert client.get(path).status_code == 401
        release.set()


def test_forged_guest_cookie_and_foreign_origin_are_rejected(settings, monkeypatch):
    app, release = guest_app(settings, monkeypatch)
    with TestClient(app) as client:
        client.cookies.set("forte_guest", "0" * 32 + ".forged")
        assert client.get("/api/v1/jobs").status_code == 401
        assert client.post("/api/v1/analyze", files=VIDEO).status_code == 401
    with TestClient(app) as client:
        client.get("/")
        response = client.post("/api/v1/analyze", files=VIDEO, headers={"Origin": "https://other.invalid"})
        assert response.status_code == 403
        release.set()


def test_guest_cookie_is_only_issued_in_guest_mode(settings):
    for mode in ["api_key", "open"]:
        settings.web_access = mode
        with TestClient(create_app(settings)) as client:
            assert "forte_guest" not in client.get("/").headers.get("set-cookie", "")
            assert "guest" not in client.get("/api/v1/access").json()


def test_dev_page_is_the_team_view_on_the_same_site(settings, monkeypatch):
    app, release = guest_app(settings, monkeypatch)
    with TestClient(app) as client:
        for path in ["/dev", "/dev/"]:
            page = client.get(path)
            assert page.status_code == 200 and "<html" in page.text
            assert "forte_guest" not in page.headers.get("set-cookie", "")
        # Without a key the team page grants nothing; with the key it sees the shared queue.
        assert client.get("/api/v1/jobs").status_code == 401
        client.post("/api/v1/analyze", files=VIDEO, headers={"X-API-Key": settings.api_key})
        assert len(client.get("/api/v1/jobs", headers={"X-API-Key": settings.api_key}).json()["jobs"]) == 1
        release.set()
