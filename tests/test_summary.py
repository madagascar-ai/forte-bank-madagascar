import json
import time

from fastapi.testclient import TestClient

from forte_guard.api import create_app
from forte_guard.contracts import ActionJournal
from forte_guard.pipeline import _build_summary, run_pipeline


def test_zero_counts_are_omitted_from_observed():
    journal = ActionJournal(duration_sec=12.0, sample_fps=5, capabilities=[])
    summary = _build_summary(journal)
    assert summary["observed"] == ["Длительность видео: 12.0 с."]


def test_notes_come_from_journal_warnings_deduped():
    journal = ActionJournal(
        duration_sec=5,
        sample_fps=5,
        capabilities=[],
        warnings=["Зоны определены автоматически.", "Зоны определены автоматически."],
    )
    summary = _build_summary(journal)
    assert summary["notes"] == ["Зоны определены автоматически."]


def test_no_violation_job_exposes_summary_but_jury_api_is_unchanged(monkeypatch, settings, tmp_path):
    from examples.make_demo import create_demo

    video, objects_path, zones = create_demo(tmp_path / "clean", disappear=True)
    real_pipeline = run_pipeline

    def replay_pipeline(*args, **kwargs):
        kwargs["replay"] = objects_path
        return real_pipeline(*args, **kwargs)

    monkeypatch.setattr("forte_guard.api.run_pipeline", replay_pipeline)
    headers = {"X-API-Key": settings.api_key}
    app = create_app(settings)
    with TestClient(app) as client:
        response = client.post(
            "/api/v1/analyze",
            headers=headers,
            files={"file": ("sample.mov", video.read_bytes(), "video/quicktime")},
            data={"zones": zones.model_dump_json()},
        )
        job = response.json()["job_id"]
        for _ in range(200):
            result = client.get(f"/api/v1/jobs/{job}", headers=headers).json()
            if result["status"] != "processing":
                break
            time.sleep(0.025)
        assert result["status"] == "done"
        assert result["verdict"] == "no_violation" and result["violations"] == []
        assert "summary" not in result

        raw_result = json.loads((app.state.store.folder(job) / "result.json").read_text())
        assert "summary" not in raw_result

        assert client.get(f"/api/v1/jobs/{job}/artifacts/summary").status_code == 401
        summary = client.get(f"/api/v1/jobs/{job}/artifacts/summary", headers=headers).json()
        assert set(summary) == {"observed", "notes"}
        assert summary["observed"]
