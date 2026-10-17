import httpx
from fastapi.testclient import TestClient

from forte_guard.api import create_app
from forte_guard.readiness import inspect_readiness


def test_readiness_is_authenticated_and_detects_missing_weights(settings):
    with TestClient(create_app(settings)) as client:
        assert client.get("/api/v1/readiness").status_code == 401
        response = client.get("/api/v1/readiness", headers={"X-API-Key": settings.api_key})
    assert response.status_code == 503
    report = response.json()
    assert report["status"] == "not_ready"
    assert not next(item for item in report["checks"] if item["name"] == "detector_weights")["ready"]
    assert settings.api_key not in response.text


def test_auto_zones_require_a_vision_model(monkeypatch, settings):
    settings.llm_enabled = True
    settings.auto_zones = True
    client_factory = httpx.Client
    transport = httpx.MockTransport(
        lambda request: httpx.Response(200, json={"capabilities": ["completion"]})
    )
    monkeypatch.setattr(
        "forte_guard.readiness.httpx.Client", lambda **kwargs: client_factory(transport=transport, **kwargs)
    )
    report = inspect_readiness(settings)
    assert next(item for item in report["checks"] if item["name"] == "local_llm")["ready"]
    assert not next(item for item in report["checks"] if item["name"] == "vision_llm")["ready"]


def test_invalid_remote_llm_configuration_is_never_probed(monkeypatch, settings):
    settings.ollama_url = "https://remote.invalid"

    def forbidden(**kwargs):
        raise AssertionError("Invalid settings must not initiate a network call")

    monkeypatch.setattr("forte_guard.readiness.httpx.Client", forbidden)
    report = inspect_readiness(settings)
    assert report["status"] == "not_ready"
    assert report["checks"][0]["name"] == "configuration"


def test_open_vocabulary_detectors_require_the_configured_vocabulary(settings):
    for detector in ("yoloe", "world"):
        settings.detector = detector
        report = inspect_readiness(settings)
        assert next(item for item in report["checks"] if item["name"] == "configuration")["ready"]
        assert not next(item for item in report["checks"] if item["name"] == "detector_vocabulary")["ready"]
    settings.detector = "yolo"
    assert all(item["name"] != "detector_vocabulary" for item in inspect_readiness(settings)["checks"])


def test_health_names_the_detector_without_exposing_paths(settings):
    settings.detector = "yolo"
    with TestClient(create_app(settings)) as client:
        health = client.get("/health").json()
    assert health["detector"] == "yolo"
    assert health["model"] == settings.model_path.name
    assert str(settings.model_path.parent) not in str(health)
