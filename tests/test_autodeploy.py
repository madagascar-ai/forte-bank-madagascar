import json
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from forte_guard.api import create_app
from forte_guard.storage import JobStore, MaintenanceError
from scripts.autodeploy import Agent, GitHub, activate, active_jobs, environment, set_maintenance


def test_failed_release_restores_previous_environment_before_reopening_uploads():
    old, new = {"sha": "old", "python": "old-python"}, {"sha": "new", "python": "new-python"}
    events = []

    def ready(release):
        events.append(("ready", release["python"]))
        return release == old

    with pytest.raises(RuntimeError, match="CandidateReadinessFailed"):
        activate(
            new,
            old,
            maintenance=lambda value: events.append(("maintenance", value)),
            drain=lambda: events.append("drain"),
            stop=lambda: events.append("stop"),
            start=lambda release: events.append(("start", release["python"])),
            ready=ready,
            save=lambda release: events.append(("save", release["sha"])),
        )
    assert events == [
        ("maintenance", True),
        "drain",
        "stop",
        ("start", "new-python"),
        ("ready", "new-python"),
        "stop",
        ("start", "old-python"),
        ("ready", "old-python"),
        ("save", "old"),
        ("maintenance", False),
    ]


def test_active_jobs_that_cannot_drain_leave_current_server_running():
    events = []

    def drain():
        raise TimeoutError("StillProcessing")

    with pytest.raises(TimeoutError):
        activate(
            {},
            {},
            maintenance=lambda value: events.append(value),
            drain=drain,
            stop=lambda: events.append("stopped"),
            start=lambda *_: None,
            ready=lambda *_: True,
            save=lambda *_: None,
        )
    assert events == [True, False]


def test_successful_release_is_saved_only_after_readiness():
    events = []
    activate(
        {"sha": "new"},
        None,
        maintenance=lambda value: events.append(("maintenance", value)),
        drain=lambda: events.append("drain"),
        stop=lambda: events.append("stop"),
        start=lambda *_: events.append("start"),
        ready=lambda *_: events.append("ready") or True,
        save=lambda *_: events.append("save"),
    )
    assert events.index("ready") < events.index("save") < events.index(("maintenance", False))


def test_failed_ci_never_prepares_or_deploys_code(tmp_path, monkeypatch):
    agent = Agent(tmp_path)
    agent.live = {"sha": "a" * 40}
    agent.child = SimpleNamespace(poll=lambda: None)
    agent.github = SimpleNamespace(
        head=lambda: "b" * 40,
        ci=lambda _: {
            "status": "completed",
            "conclusion": "failure",
            "html_url": "https://example.invalid/ci",
        },
    )
    monkeypatch.setattr(agent, "ensure_ollama", lambda: None)
    monkeypatch.setattr(agent, "prepare", lambda *_: pytest.fail("Failed CI must never deploy"))
    agent.tick()
    assert agent.status["state"] == "ci_failed" and agent.live["sha"] == "a" * 40


def test_latest_ci_run_supersedes_old_success():
    client = GitHub()
    base = {
        "head_sha": "a" * 40,
        "head_branch": "main",
        "event": "push",
        "repository": {"full_name": "temirrrr/forte-bank-madagascar"},
    }
    client.get = lambda _: {
        "workflow_runs": [
            {**base, "id": 1, "conclusion": "success"},
            {**base, "id": 2, "conclusion": "failure"},
            {**base, "id": 3, "event": "pull_request", "conclusion": "success"},
        ]
    }
    assert client.ci("a" * 40)["id"] == 2


def test_maintenance_and_job_acceptance_share_a_database_transaction(tmp_path):
    root = tmp_path / "runtime"
    store = JobStore(root / "data")
    set_maintenance(root, True)
    assert store.maintenance()
    with pytest.raises(MaintenanceError):
        store.create("a" * 32, "synthetic.mov")
    assert active_jobs(root) == 0
    set_maintenance(root, False)
    store.create("b" * 32, "synthetic.mov")
    assert active_jobs(root) == 1


def test_maintenance_still_blocks_an_upload_that_started_before_drain(settings, monkeypatch):
    app = create_app(settings)
    store = app.state.store
    original = store.create

    def enable_before_inserting(*args):
        set_maintenance(settings.data_dir.parent, True)
        return original(*args)

    # This fixture's data directory is <tmp>/data, the same layout used by the deployer.
    monkeypatch.setattr(store, "create", enable_before_inserting)
    with TestClient(app) as client:
        response = client.post(
            "/api/v1/analyze",
            headers={"X-API-Key": settings.api_key},
            files={"file": ("synthetic.mov", b"synthetic")},
        )
        assert response.status_code == 503 and response.headers["retry-after"] == "10"
        assert client.get("/api/v1/jobs", headers={"X-API-Key": settings.api_key}).status_code == 200
    assert store.list() == []
    assert list((settings.data_dir / "jobs").iterdir()) == []


def test_each_release_uses_shared_local_data_and_its_own_code(tmp_path):
    (tmp_path / ".env").write_text("FORTE_API_KEY=synthetic-only\nFORTE_ALLOWED_CLIENTS=10.1.2.3\n")
    release = {"sha": "a" * 40, "source": str(tmp_path / "releases" / ("a" * 40))}
    env = environment(tmp_path, release)
    assert env["FORTE_DATA_DIR"] == str(tmp_path / "data")
    assert env["PYTHONPATH"] == str(tmp_path / "releases" / ("a" * 40) / "src")
    assert env["FORTE_API_KEY"] == "synthetic-only"
    assert env["FORTE_ALLOWED_CLIENTS"] == "10.1.2.3"


def test_deployment_endpoint_exposes_version_and_never_secrets(settings, tmp_path, monkeypatch):
    status_file = tmp_path / "status.json"
    status_file.write_text(
        json.dumps({"state": "ready", "active_sha": "a" * 40, "api_key": settings.api_key})
    )
    monkeypatch.setenv("FORTE_RELEASE_SHA", "a" * 40)
    monkeypatch.setenv("FORTE_DEPLOY_STATE", str(status_file))
    with TestClient(create_app(settings)) as client:
        assert client.get("/health").json()["release"] == "a" * 40
        assert client.get("/api/v1/deployment").status_code == 401
        response = client.get("/api/v1/deployment", headers={"X-API-Key": settings.api_key})
        assert response.json()["active_sha"] == "a" * 40
        assert response.json()["managed"] is True
        assert settings.api_key not in response.text
