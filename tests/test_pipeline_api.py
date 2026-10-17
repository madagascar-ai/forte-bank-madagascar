import base64
import threading
import time

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from forte_guard.api import create_app
from forte_guard.pipeline import run_pipeline
from forte_guard.storage import JobStore


def test_full_pipeline_preserves_journals_and_real_evidence(demo, settings, tmp_path):
    video, objects, zones = demo
    output = tmp_path / "result"
    result = run_pipeline(video, output, settings, "synthetic", zones=zones, replay=objects)
    assert result.verdict == "violation" and len(result.violations) == 1
    for filename in [
        "object_journal",
        "action_journal",
        "candidates.rules",
        "candidates.llm",
        "candidates",
        "frames",
        "result",
    ]:
        assert (output / f"{filename}.json").is_file()
    assert result.verification_mode == "structural_only"
    evidence = result.violations[0].evidence_frames
    assert 1 <= len(evidence) <= 3
    assert any(abs(frame.timestamp_sec - result.violations[0].timecode_sec) < 0.001 for frame in evidence)
    for frame in evidence:
        image = cv2.imdecode(np.frombuffer(base64.b64decode(frame.image_base64), np.uint8), cv2.IMREAD_COLOR)
        assert image is not None


def test_both_providers_run_in_parallel_on_same_journal(monkeypatch, demo, settings, tmp_path):
    from forte_guard.reasoning.rules import rule_candidates

    barrier = threading.Barrier(2, timeout=5)
    seen = []

    def code(journal, rules, scenarios):
        seen.append(id(journal))
        barrier.wait()
        return rule_candidates(journal, rules, scenarios)

    def llm(journal, rules, scenarios, *unused):
        seen.append(id(journal))
        barrier.wait()
        items = rule_candidates(journal, rules, scenarios)
        for item in items:
            item.sources = ["llm"]
        return items, []

    monkeypatch.setattr("forte_guard.pipeline.rule_candidates", code)
    monkeypatch.setattr("forte_guard.pipeline.llm_candidates", llm)
    video, objects, zones = demo
    result = run_pipeline(video, tmp_path / "parallel", settings, "synthetic", zones=zones, replay=objects)
    assert len(seen) == 2 and seen[0] == seen[1]
    assert result.stats["rule_candidates"] == result.stats["llm_candidates"] == 1
    assert result.stats["merged_candidates"] == 1
    assert result.violations[0].sources == ["rule_engine", "llm"]


def test_authenticated_upload_poll_and_artifact_access(monkeypatch, demo, settings):
    video, objects, zones = demo
    real_pipeline = run_pipeline

    def replay_pipeline(*args, **kwargs):
        kwargs["replay"] = objects
        return real_pipeline(*args, **kwargs)

    monkeypatch.setattr("forte_guard.api.run_pipeline", replay_pipeline)
    headers = {"X-API-Key": settings.api_key}
    with TestClient(create_app(settings)) as client:
        assert client.get("/api/v1/jobs").status_code == 401
        assert client.get("/api/v1/jobs", headers={"X-API-Key": "wrong"}).status_code == 401
        assert client.get("/").status_code == 200
        assert "default-src 'self'" in client.get("/").headers["content-security-policy"]
        response = client.post(
            "/api/v1/analyze",
            headers=headers,
            files={"file": ("sample.mov", video.read_bytes(), "video/quicktime")},
            data={"zones": zones.model_dump_json()},
        )
        assert response.status_code == 202
        job = response.json()["job_id"]
        for _ in range(200):
            result = client.get(f"/api/v1/jobs/{job}", headers=headers).json()
            if result["status"] != "processing":
                break
            time.sleep(0.025)
        assert result["status"] == "done" and result["verdict"] == "violation"
        assert client.get(f"/api/v1/jobs/{job}/artifacts/action_journal").status_code == 401
        frames = client.get(f"/api/v1/jobs/{job}/artifacts/frames", headers=headers).json()
        assert "image_path" not in frames[0]
        assert (
            client.get(f"/api/v1/jobs/{job}/frames/{frames[0]['frame_id']}", headers=headers).headers[
                "content-type"
            ]
            == "image/jpeg"
        )
        assert client.get(f"/api/v1/jobs/{job}/artifacts/secret", headers=headers).status_code == 404
        assert client.get("/api/v1/jobs/invalid", headers=headers).status_code == 404


def test_invalid_uploads_fail_without_processing(settings):
    headers = {"X-API-Key": settings.api_key}
    with TestClient(create_app(settings)) as client:
        assert (
            client.post("/api/v1/analyze", headers=headers, files={"file": ("x.txt", b"a")}).status_code
            == 415
        )
        assert (
            client.post("/api/v1/analyze", headers=headers, files={"file": ("x.mov", b"")}).status_code == 422
        )
        assert (
            client.post(
                "/api/v1/analyze",
                headers=headers,
                files={"file": ("x.mov", b"a")},
                data={"zones": "not json"},
            ).status_code
            == 422
        )


def test_restart_recovery_is_explicit(tmp_path):
    store = JobStore(tmp_path)
    store.create("a" * 32, "synthetic.mov")
    assert JobStore(tmp_path).get("a" * 32)["status"] == "processing"
    assert JobStore(tmp_path, recover=True).get("a" * 32)["status"] == "failed"


def test_no_remote_ai_endpoint(settings):
    settings.ollama_url = "https://example.com"
    with pytest.raises(ValueError, match="local"):
        settings.validate()


def test_queued_job_timeout_is_reported(settings):
    settings.job_timeout_sec = 0.01
    app = create_app(settings)
    app.state.store.create("a" * 32, "synthetic.mov")
    time.sleep(0.02)
    with TestClient(app) as client:
        response = client.get("/api/v1/jobs/" + "a" * 32, headers={"X-API-Key": settings.api_key})
        assert response.json()["status"] == "failed"
