import json
import time

from fastapi.testclient import TestClient

from forte_guard.api import create_app
from forte_guard.contracts import ActionJournal, Rule
from forte_guard.pipeline import _build_summary, _describe_video, run_pipeline
from forte_guard.video import Deadline


def _rule(id, title, **overrides):
    return Rule(
        id=id,
        title=title,
        kind="state",
        engine={},
        document={"clause": f"пункт для {id}"},
        **overrides,
    )


def test_disabled_and_uncapable_rules_are_excluded_from_checks_performed():
    journal = ActionJournal(duration_sec=5, sample_fps=5, capabilities=["cash_transferred"])
    rules = [
        _rule("r1", "Правило A"),
        _rule("r2", "Правило B", enabled=False),
        _rule("r3", "Правило C", required_capabilities=["cash_counted"]),
    ]
    summary = _build_summary(rules, journal)
    titles = {item["title"] for item in summary["checks_performed"]}
    assert titles == {"Правило A"}
    assert "Правило B" not in summary["checks_not_performed"]
    assert "Правило C" in summary["checks_not_performed"]


def test_same_title_rules_are_merged_into_one_entry():
    journal = ActionJournal(duration_sec=5, sample_fps=5, capabilities=[])
    rules = [
        Rule(id="a", title="Личные вещи", kind="state", engine={}, document={"clause": "п.1"}),
        Rule(id="b", title="Личные вещи", kind="state", engine={}, document={"clause": "п.2"}),
    ]
    summary = _build_summary(rules, journal)
    assert len(summary["checks_performed"]) == 1
    assert summary["checks_performed"][0]["clauses"] == ["п.1", "п.2"]


def test_unsupported_actions_are_translated_and_deduped():
    journal = ActionJournal(
        duration_sec=5,
        sample_fps=5,
        capabilities=[],
        unsupported_actions=["authenticity_verified", "a_future_action", "authenticity_verified"],
    )
    summary = _build_summary([], journal)
    assert summary["checks_not_performed"].count("проверка подлинности купюр") == 1
    assert "a_future_action" in summary["checks_not_performed"]


def test_zero_counts_are_omitted_from_observed():
    journal = ActionJournal(duration_sec=12.0, sample_fps=5, capabilities=[])
    summary = _build_summary([], journal)
    assert summary["observed"] == ["Длительность видео: 12.0 с."]


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
        assert set(summary) == {
            "checks_performed",
            "checks_not_performed",
            "observed",
            "notes",
            "description",
        }
        assert summary["checks_performed"]
        assert summary["description"] is None  # settings fixture disables the LLM branch


def test_description_is_skipped_without_llm_or_facts(settings):
    assert _describe_video(settings, ["Длительность видео: 5.0 с."], Deadline(5)) is None  # llm disabled
    settings.llm_enabled = True
    assert _describe_video(settings, [], Deadline(5)) is None  # nothing to describe


def test_description_uses_only_the_given_facts(monkeypatch, settings):
    settings.llm_enabled = True
    seen = {}

    def fake_chat(settings, prompt, schema, deadline, **kwargs):
        seen["prompt"] = prompt
        return {"summary": "Клиент обслужен, деньги пересчитаны на машинке."}

    monkeypatch.setattr("forte_guard.pipeline.local_chat", fake_chat)
    facts = ["Длительность видео: 8.0 с.", "Клиентов в кадре: 1.", "Пересчётов на счётной машинке: 1."]
    description = _describe_video(settings, facts, Deadline(5))
    assert description == "Клиент обслужен, деньги пересчитаны на машинке."
    assert all(fact in seen["prompt"] for fact in facts)


def test_description_failure_is_silent(monkeypatch, settings):
    settings.llm_enabled = True

    def failing_chat(*args, **kwargs):
        raise TimeoutError("local LLM unavailable")

    monkeypatch.setattr("forte_guard.pipeline.local_chat", failing_chat)
    assert _describe_video(settings, ["Длительность видео: 5.0 с."], Deadline(5)) is None
