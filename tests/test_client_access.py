import pytest
from fastapi.testclient import TestClient

from forte_guard.api import create_app


def test_only_listed_clients_and_loopback_can_access_the_service(settings):
    settings.allowed_clients = ["10.70.1.2"]
    app = create_app(settings)
    for peer in ["10.70.1.2", "127.0.0.1", "::1"]:
        with TestClient(app, client=(peer, 12345)) as client:
            assert client.get("/").status_code == 200
            # Network access does not bypass API authentication.
            assert client.get("/api/v1/jobs").status_code == 401
            assert client.get("/api/v1/jobs", headers={"X-API-Key": settings.api_key}).status_code == 200
    with TestClient(app, client=("10.70.1.3", 12345)) as client:
        for path in ["/", "/health", "/assets/app.js", "/api/v1/jobs"]:
            response = client.get(
                path, headers={"X-API-Key": settings.api_key, "X-Forwarded-For": "10.70.1.2"}
            )
            assert response.status_code == 403


def test_client_access_accepts_individual_addresses_only(settings):
    settings.allowed_clients = ["10.70.0.0/16"]
    with pytest.raises(ValueError):
        settings.validate()


def test_trusted_web_access_needs_an_explicit_device_list(settings):
    settings.web_access = "trusted_clients"
    settings.allowed_clients = []
    with pytest.raises(ValueError, match="explicit"):
        settings.validate()


def test_team_devices_can_use_the_interface_without_a_key(settings):
    settings.web_access = "trusted_clients"
    settings.allowed_clients = ["10.70.1.2"]
    app = create_app(settings)
    for peer in ["10.70.1.2", "127.0.0.1", "::1"]:
        with TestClient(app, client=(peer, 12345)) as client:
            assert client.get("/api/v1/jobs").status_code == 401
            client.get("/")
            assert client.get("/api/v1/access").json()["authorized"] is True
            assert client.get("/api/v1/jobs").status_code == 200
            assert client.get("/api/v1/jobs", headers={"X-API-Key": "wrong"}).status_code == 401
    with TestClient(app, client=("10.70.1.3", 12345)) as client:
        assert client.get("/api/v1/access").status_code == 403
        assert client.get("/api/v1/jobs", headers={"X-Forwarded-For": "10.70.1.2"}).status_code == 403


def test_browser_requests_cannot_be_submitted_by_another_website(settings):
    settings.web_access = "open"
    with TestClient(create_app(settings)) as client:
        client.get("/")
        for origin in ["https://other.invalid", "null", "http://testserver:9999"]:
            response = client.post(
                "/api/v1/analyze", headers={"Origin": origin}, files={"file": ("x.txt", b"x")}
            )
            assert response.status_code == 403
        response = client.post(
            "/api/v1/analyze", headers={"Origin": "http://testserver"}, files={"file": ("x.txt", b"x")}
        )
        assert response.status_code == 415


def test_open_interface_keeps_the_jury_api_behind_the_key(settings):
    settings.web_access = "open"
    with TestClient(create_app(settings)) as client:
        assert client.get("/api/v1/access").json() == {"mode": "open", "authorized": False}
        assert client.get("/api/v1/jobs").status_code == 401
        page = client.get("/")
        cookie = page.headers["set-cookie"].lower()
        assert "httponly" in cookie and "samesite=strict" in cookie
        assert settings.api_key not in page.headers["set-cookie"] and settings.api_key not in page.text
        assert client.get("/api/v1/access").json()["authorized"] is True
        assert client.get("/api/v1/jobs").status_code == 200
    with TestClient(create_app(settings)) as client:
        client.cookies.set("forte_ui", "forged")
        assert client.get("/api/v1/jobs").status_code == 401
        assert client.get("/api/v1/jobs", headers={"X-API-Key": settings.api_key}).status_code == 200


def test_default_interface_still_asks_for_the_key(settings):
    with TestClient(create_app(settings)) as client:
        assert "set-cookie" not in client.get("/").headers
        assert client.get("/api/v1/access").json()["authorized"] is False
        assert client.get("/api/v1/jobs").status_code == 401
