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
