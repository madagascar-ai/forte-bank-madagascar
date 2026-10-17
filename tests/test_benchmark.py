import time
from concurrent.futures import ThreadPoolExecutor

import pytest
from fastapi.testclient import TestClient

from forte_guard import benchmark
from forte_guard.api import create_app
from forte_guard.benchmark import (
    BenchmarkRunner,
    Label,
    labels_url,
    normalize_name,
    parse_labels,
    parse_times,
    score_video,
    summarize,
)
from forte_guard.contracts import AnalysisResult, Violation

# Synthetic labels only; real labels and comments never appear in tests.
LABELS = (
    "наименование файла,таймстемп,нарушение / без нарушения,комментарий\n"
    "Х-1,11 секунда,нарушение,synthetic: cashier leaves\n"
    "X-2,,без нарушения,\n"
    "Х-3,1минута 5 секунда,нарушение,\n"
    "X-4,,без нарушения,synthetic: looks suspicious but fine\n"
)


def result(verdict, *times):
    return AnalysisResult(
        job_id="synthetic",
        verdict=verdict,
        violations=[
            Violation(
                type="synthetic type",
                timecode_sec=t,
                description="synthetic",
                evidence_frames=[],
                rule_id="r",
                sources=["rule_engine"],
                document={},
            )
            for t in times
        ],
    )


def test_names_and_timestamps_are_normalized():
    assert normalize_name("Х-12.mov") == normalize_name("X-12") == "X-12"
    assert parse_times("11 секунда") == [11]
    assert parse_times("1минута 5 секунда") == [65]
    assert parse_times("0:11, 1:05") == [11, 65]
    assert parse_times("") == []
    assert labels_url("https://docs.google.com/spreadsheets/d/abc_1/edit?gid=42#gid=42") == (
        "https://docs.google.com/spreadsheets/d/abc_1/export?format=csv&gid=42"
    )


def test_labels_parse_and_conflicting_duplicates_are_excluded():
    labels, warnings = parse_labels(LABELS + "X-2,,нарушение,\n")
    assert set(labels) == {"X-1", "X-3", "X-4"}
    assert labels["X-1"].violation and labels["X-1"].times == [11]
    assert not labels["X-4"].violation
    assert warnings
    with pytest.raises(ValueError):
        parse_labels("файл,тайм,нарушение,коммент\nX-9,,maybe,\n")


def test_score_outcomes_and_localization():
    positive = Label("X-1", True, [11.0])
    assert score_video(positive, result("violation", 30, 12.5), 3) == {
        "outcome": "TP",
        "time_error_sec": 1.5,
        "located": True,
        "matched_violation": 1,
    }
    assert score_video(positive, result("no_violation"), 3)["outcome"] == "FN"
    assert score_video(Label("X-2", False), result("violation", 4), 3)["outcome"] == "FP"
    assert score_video(Label("X-2", False), result("no_violation"), 3)["outcome"] == "TN"
    # A timeout must count against us, as it does for the jury.
    assert score_video(positive, None, 3)["outcome"] == "failed"


def test_summary_counts_failures_as_errors():
    items = [
        {
            "status": "done",
            "label": {"violation": True},
            "score": {"outcome": "TP", "located": True, "time_error_sec": 1.0},
            "judge": {"same_event": "yes", "type_match": True},
            "processing_sec": 40,
        },
        {
            "status": "failed",
            "label": {"violation": True},
            "score": {"outcome": "failed"},
            "processing_sec": 300,
        },
        {"status": "done", "label": {"violation": False}, "score": {"outcome": "TN"}, "processing_sec": 20},
        {"status": "pending", "label": {"violation": False}},
    ]
    s = summarize(items, 285)
    assert (s["processed"], s["TP"], s["failed"], s["TN"]) == (3, 1, 1, 1)
    assert s["recall"] == 0.5 and s["accuracy"] == 0.667
    assert (s["located"], s["judge_yes"], s["over_time_limit"]) == (1, 1, 1)


@pytest.fixture
def bench(settings, tmp_path, monkeypatch):
    videos = tmp_path / "videos"
    videos.mkdir()
    for name in ["Х-1.mov", "X-2.mov", "Х-3.mov", "X-4.mov", "X-5.mov"]:
        (videos / name).write_bytes(b"synthetic")
    (tmp_path / "labels.csv").write_text(LABELS, encoding="utf-8")
    settings.benchmark_enabled = True
    settings.benchmark_videos = videos
    settings.benchmark_labels = str(tmp_path / "labels.csv")
    answers = {
        "X-1": result("violation", 12),
        "X-2": result("violation", 3),
        "X-3": None,
        "X-4": result("no_violation"),
    }

    def fake_pipeline(source, output, settings, job_id, progress, **kwargs):
        progress("objects", 0.5)
        (output / "frames").mkdir(parents=True)
        answer = answers[normalize_name(source.name)]
        if answer is None:
            raise TimeoutError("Превышен лимит")
        return answer

    monkeypatch.setattr(benchmark.pipeline, "run_pipeline", fake_pipeline)
    return settings


def wait(runner, run_id):
    for _ in range(200):
        run = runner.get(run_id)
        if run["status"] not in {"running", "cancelling"}:
            return run
        time.sleep(0.02)
    raise AssertionError("Benchmark did not finish")


def test_full_run_scores_every_labelled_video(bench, monkeypatch):
    bench.llm_enabled = True
    judged = []
    monkeypatch.setattr(
        benchmark,
        "judge_video",
        lambda settings, label, res: (
            judged.append(label.video)
            or {"same_event": "yes", "matched_index": 0, "type_match": False, "explanation": "synthetic"}
        ),
    )
    with ThreadPoolExecutor(max_workers=1) as executor:
        runner = BenchmarkRunner(bench, executor)
        run = wait(runner, runner.start()["run_id"])
    assert run["status"] == "done"
    assert [item["video"] for item in run["items"]] == ["X-1", "X-2", "X-3", "X-4"]
    assert any("без разметки" in warning for warning in run["warnings"])
    s = run["summary"]
    assert (s["TP"], s["FP"], s["failed"], s["TN"], s["located"]) == (1, 1, 1, 1, 1)
    assert judged == ["X-1"]
    assert not (bench.data_dir / "benchmarks" / run["run_id"] / "videos" / "X-1" / "frames").exists()


def test_cancel_and_restart_recovery(bench):
    executor = ThreadPoolExecutor(max_workers=1)
    executor.submit(time.sleep, 0.2)
    runner = BenchmarkRunner(bench, executor)
    run_id = runner.start()["run_id"]
    with pytest.raises(RuntimeError):
        runner.start()
    runner.cancel(run_id)
    run = wait(runner, run_id)
    executor.shutdown()
    assert run["status"] == "cancelled"
    assert {item["status"] for item in run["items"]} == {"skipped"}

    stale = runner.get(run_id)
    stale["status"] = "running"
    benchmark.write_json(runner.path(run_id), stale)
    assert BenchmarkRunner(bench, None).get(run_id)["status"] == "interrupted"


def test_api_is_disabled_by_default_and_needs_the_key(settings, bench):
    settings_off = type(settings)(**{**settings.__dict__, "benchmark_enabled": False})
    with TestClient(create_app(settings_off)) as client:
        headers = {"X-API-Key": settings.api_key}
        assert client.get("/api/v1/benchmarks", headers=headers).json()["enabled"] is False
        assert client.post("/api/v1/benchmarks", headers=headers).status_code == 404
    with TestClient(create_app(bench)) as client:
        assert client.post("/api/v1/benchmarks").status_code == 401
        response = client.post("/api/v1/benchmarks", headers={"X-API-Key": bench.api_key})
        assert response.status_code == 202 and response.json()["videos"] == 4
        assert client.get("/api/v1/benchmarks/" + "z" * 12, headers=headers).status_code == 404


def test_judge_uses_the_judge_model_and_clamps_invented_indices(settings, monkeypatch):
    settings.judge_model = "qwen3:30b"
    calls = {}

    def fake_chat(settings, prompt, schema, deadline, **kwargs):
        calls.update(kwargs)
        return {"same_event": "yes", "matched_index": 7, "type_match": True, "explanation": "synthetic"}

    monkeypatch.setattr(benchmark, "local_chat", fake_chat)
    answer = benchmark.judge_video(settings, Label("X-1", True, [11.0], "synthetic"), result("violation", 12))
    assert calls["model"] == "qwen3:30b"
    assert answer["matched_index"] == -1
