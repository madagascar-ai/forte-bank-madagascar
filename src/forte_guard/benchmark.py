"""Offline benchmark of the full pipeline against the team's labels.

Labels are read only here, after the pipeline has produced its answer for a video; they are
never passed to the pipeline. The judge is the team's local LLM, never a cloud model.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import re
import shutil
import threading
import time
import uuid
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path

import httpx

from . import pipeline
from .contracts import AnalysisResult
from .reasoning.llm import local_chat
from .settings import Settings
from .storage import write_json
from .video import Deadline

RUN_PATTERN = re.compile(r"^[0-9a-f]{12}$")
VIDEO_SUFFIXES = {".mov", ".mp4"}


@dataclass
class Label:
    video: str
    violation: bool
    times: list[float] = field(default_factory=list)
    comment: str = ""


def normalize_name(name: str) -> str:
    # The dataset mixes Latin X and Cyrillic Х in otherwise identical file names.
    return Path(name.strip()).stem.replace("Х", "X").replace("х", "X").replace(" ", "").upper()


def parse_times(text: str) -> list[float]:
    """Accept "11 секунда", "1минута 5 секунда", "0:11", "1:05" and lists of them."""
    text = (text or "").lower().replace(",", ";")
    times = []
    for minutes, seconds in re.findall(r"(?:(\d+)\s*мин\w*\s*)?(\d+(?:\.\d+)?)\s*сек", text):
        times.append(int(minutes or 0) * 60 + float(seconds))
    for minutes, seconds in re.findall(r"(\d+):(\d{1,2}(?:\.\d+)?)", text):
        times.append(int(minutes) * 60 + float(seconds))
    if not times:
        times = [float(value) for value in re.findall(r"\d+(?:\.\d+)?", text)]
    return sorted(set(times))


def _verdict_value(value: str) -> bool:
    value = value.strip().lower()
    if value.startswith("без") or value in {"0", "false", "no", "no_violation", "нет"}:
        return False
    if value.startswith("нарушен") or value in {"1", "true", "yes", "violation", "да"}:
        return True
    raise ValueError("Unknown label value in the verdict column")


def labels_url(source: str) -> str:
    match = re.match(r"https://docs\.google\.com/spreadsheets/d/([\w-]+)", source)
    if not match:
        return source
    gid = re.search(r"gid=(\d+)", source)
    return f"https://docs.google.com/spreadsheets/d/{match[1]}/export?format=csv" + (
        f"&gid={gid[1]}" if gid else ""
    )


def read_labels_text(source: str) -> str:
    if source.startswith("https://"):
        response = httpx.get(labels_url(source), follow_redirects=True, timeout=20)
        response.raise_for_status()
        return response.content.decode("utf-8-sig")
    return Path(source).read_text(encoding="utf-8-sig")


def parse_labels(text: str) -> tuple[dict[str, Label], list[str]]:
    rows = list(csv.reader(io.StringIO(text)))
    if not rows:
        raise ValueError("Labels table is empty")
    header = [cell.strip().lower() for cell in rows[0]]

    def column(*keys, default):
        return next((i for i, h in enumerate(header) if any(key in h for key in keys)), default)

    name_i = column("файл", "name", "video", default=0)
    time_i = column("тайм", "time", default=1)
    verdict_i = column("нарушен", "label", "verdict", default=2)
    comment_i = column("коммент", "comment", default=3)
    labels: dict[str, Label] = {}
    conflicts: set[str] = set()
    warnings = []
    for row in rows[1:]:

        def cell(i, row=row):
            return row[i].strip() if i < len(row) else ""

        if not cell(name_i):
            continue
        item = Label(
            video=normalize_name(cell(name_i)),
            violation=_verdict_value(cell(verdict_i)),
            times=parse_times(cell(time_i)),
            comment=cell(comment_i),
        )
        previous = labels.get(item.video)
        if previous is None:
            labels[item.video] = item
        elif previous.violation != item.violation:
            conflicts.add(item.video)
        else:
            previous.times = sorted(set(previous.times + item.times))
            previous.comment = " ".join(filter(None, [previous.comment, item.comment]))
    for name in conflicts:
        labels.pop(name)
    if conflicts:
        warnings.append(f"Противоречивая разметка, видео исключены: {len(conflicts)}.")
    return labels, warnings


def score_video(label: Label, result: AnalysisResult | None, tolerance: float) -> dict:
    """Deterministic part of the judgement: video verdict and timecode distance."""
    if result is None:
        predicted = None
    else:
        predicted = result.verdict == "violation"
    if predicted is None:
        outcome = "failed"
    else:
        outcome = {(True, True): "TP", (True, False): "FN", (False, True): "FP", (False, False): "TN"}[
            (label.violation, predicted)
        ]
    score = {"outcome": outcome, "time_error_sec": None, "located": None, "matched_violation": None}
    if outcome == "TP" and label.times:
        error, index = min(
            (abs(v.timecode_sec - t), i) for i, v in enumerate(result.violations) for t in label.times
        )
        score.update(time_error_sec=round(error, 2), located=error <= tolerance, matched_violation=index)
    return score


JUDGE_SCHEMA = {
    "type": "object",
    "properties": {
        "same_event": {"type": "string", "enum": ["yes", "partial", "no"]},
        "matched_index": {"type": "integer"},
        "type_match": {"type": "boolean"},
        "explanation": {"type": "string"},
    },
    "required": ["same_event", "matched_index", "type_match", "explanation"],
    "additionalProperties": False,
}


def judge_video(settings: Settings, label: Label, result: AnalysisResult) -> dict:
    """Ask the local LLM whether a reported violation is the labelled one. Only for TP videos."""
    found = [
        {"index": i, "type": v.type, "timecode_sec": v.timecode_sec, "description": v.description}
        for i, v in enumerate(result.violations)
    ]
    prompt = (
        "Ты судья бенчмарка. Эталон — разметка человека: в видео есть нарушение кассовой дисциплины, "
        "известны его время и комментарий. Система выдала список найденных нарушений. Реши, описывает ли "
        "какое-то из них то же самое событие, что эталон. same_event=yes — то же действие примерно в то же "
        "время; partial — то же время или тот же участок, но описано другое или неточно; no — ничего "
        "похожего. matched_index — индекс лучшего совпадения или -1. type_match — тип нарушения системы "
        "по смыслу совпадает с нарушением из комментария. Будь строгим: совпадение только по времени без "
        "совпадения смысла — не yes. Если комментария нет, суди только по времени и отметь это. "
        "explanation — одно-два предложения по-русски.\n"
        + json.dumps(
            {"label": {"times_sec": label.times, "comment": label.comment}, "found": found},
            ensure_ascii=False,
        )
    )
    answer = local_chat(
        settings, prompt, JUDGE_SCHEMA, Deadline(90), max_seconds=60, model=settings.judge_model
    )
    if answer["matched_index"] not in range(-1, len(found)):
        answer["matched_index"] = -1
    return answer


def summarize(items: list[dict], timeout_sec: float) -> dict:
    done = [item for item in items if item["status"] in {"done", "failed"}]
    outcomes = Counter(item["score"]["outcome"] for item in done if item.get("score"))
    tp, fn, fp, tn = (outcomes[key] for key in ("TP", "FN", "FP", "TN"))
    failed = outcomes["failed"]
    positives = [item for item in done if item["label"]["violation"]]

    def ratio(a, b):
        return round(a / b, 3) if b else None

    located = [item for item in done if item.get("score", {}).get("located") is not None]
    errors = [item["score"]["time_error_sec"] for item in located]
    judged = [item["judge"] for item in done if isinstance(item.get("judge"), dict)]
    seconds = [item["processing_sec"] for item in done if item.get("processing_sec") is not None]
    return {
        "videos": len(items),
        "processed": len(done),
        "TP": tp,
        "FN": fn,
        "FP": fp,
        "TN": tn,
        "failed": failed,
        # A failed or timed-out video counts as an error, as it does for the jury.
        "accuracy": ratio(tp + tn, len(done)),
        "recall": ratio(tp, len(positives)),
        "precision": ratio(tp, tp + fp),
        "specificity": ratio(tn, len(done) - len(positives)),
        "located": sum(1 for item in located if item["score"]["located"]),
        "located_of": len(located),
        "mean_time_error_sec": round(sum(errors) / len(errors), 2) if errors else None,
        "judge_yes": sum(1 for j in judged if j.get("same_event") == "yes"),
        "judge_partial": sum(1 for j in judged if j.get("same_event") == "partial"),
        "judge_type_match": sum(1 for j in judged if j.get("type_match")),
        "judged": len(judged),
        "mean_processing_sec": round(sum(seconds) / len(seconds), 1) if seconds else None,
        "max_processing_sec": round(max(seconds), 1) if seconds else None,
        "over_time_limit": sum(1 for value in seconds if value > timeout_sec),
    }


class BenchmarkRunner:
    """One run at a time; videos are queued one by one so real uploads interleave between them."""

    def __init__(self, settings: Settings, executor):
        self.settings = settings
        self.executor = executor
        self.root = settings.data_dir.resolve() / "benchmarks"
        self.root.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()
        for path in self.root.glob("*/run.json"):
            run = json.loads(path.read_text())
            if run["status"] in {"running", "cancelling"}:
                run.update(status="interrupted", finished_at=time.time())
                for item in run["items"]:
                    if item["status"] in {"pending", "running"}:
                        item["status"] = "skipped"
                write_json(path, run)

    def path(self, run_id: str) -> Path:
        if not RUN_PATTERN.fullmatch(run_id):
            raise KeyError(run_id)
        return self.root / run_id / "run.json"

    def get(self, run_id: str) -> dict:
        path = self.path(run_id)
        if not path.is_file():
            raise KeyError(run_id)
        return json.loads(path.read_text())

    def save(self, run: dict) -> None:
        run["summary"] = summarize(run["items"], self.settings.job_timeout_sec)
        write_json(self.path(run["run_id"]), run)

    def list(self) -> list[dict]:
        runs = [json.loads(path.read_text()) for path in self.root.glob("*/run.json")]
        runs.sort(key=lambda run: run["started_at"], reverse=True)
        return [{key: value for key, value in run.items() if key != "items"} for run in runs]

    def running(self) -> str | None:
        return next(
            (run["run_id"] for run in self.list() if run["status"] in {"running", "cancelling"}), None
        )

    def start(self) -> dict:
        with self.lock:
            if self.running():
                raise RuntimeError("Бенчмарк уже выполняется")
            source = self.settings.benchmark_labels
            if not source:
                raise ValueError("Задайте FORTE_BENCHMARK_LABELS")
            text = read_labels_text(source)
            labels, warnings = parse_labels(text)
            folder = self.settings.benchmark_videos
            videos = (
                {
                    normalize_name(path.name): path
                    for path in sorted(folder.iterdir())
                    if path.suffix.lower() in VIDEO_SUFFIXES
                }
                if folder.is_dir()
                else {}
            )
            if not videos:
                raise ValueError("В FORTE_BENCHMARK_VIDEOS нет видео")
            unlabeled = len(set(videos) - set(labels))
            missing = len(set(labels) - set(videos))
            if unlabeled:
                warnings.append(f"Видео без разметки пропущены: {unlabeled}.")
            if missing:
                warnings.append(f"Размеченных видео нет в папке: {missing}.")
            names = sorted(
                set(videos) & set(labels),
                key=lambda n: [int(x) if x.isdigit() else x for x in re.split(r"(\d+)", n)],
            )
            run_id = uuid.uuid4().hex[:12]
            run = {
                "run_id": run_id,
                "status": "running",
                "started_at": time.time(),
                "finished_at": None,
                "config": {
                    "release": os.getenv("FORTE_RELEASE_SHA", "workspace"),
                    "detector": self.settings.detector,
                    "model": self.settings.model_path.name,
                    "sample_fps": self.settings.sample_fps,
                    "llm_model": self.settings.llm_model if self.settings.llm_enabled else None,
                    "zone_model": self.settings.zone_model if self.settings.auto_zones else None,
                    "judge_model": self.settings.judge_model,
                    "tolerance_sec": self.settings.benchmark_tolerance_sec,
                    # Fingerprint, not content: lets runs be compared on the same labels.
                    "labels_sha": hashlib.sha256(text.encode()).hexdigest()[:12],
                },
                "warnings": warnings,
                "items": [
                    {
                        "video": name,
                        "file": str(videos[name]),
                        "status": "pending",
                        "label": asdict(labels[name]),
                    }
                    for name in names
                ],
            }
            self.root.joinpath(run_id).mkdir()
            self.save(run)
            self.executor.submit(self.step, run_id, 0)
            return run

    def cancel(self, run_id: str) -> dict:
        with self.lock:
            run = self.get(run_id)
            if run["status"] == "running":
                run["status"] = "cancelling"
                self.save(run)
            return run

    def _finish(self, run: dict, status: str) -> None:
        for item in run["items"]:
            if item["status"] in {"pending", "running"}:
                item["status"] = "skipped"
        run.update(status=status, finished_at=time.time())
        self.save(run)

    def step(self, run_id: str, index: int) -> None:
        with self.lock:
            run = self.get(run_id)
            if run["status"] == "cancelling":
                self._finish(run, "cancelled")
                return
            if index >= len(run["items"]):
                self._finish(run, "done")
                return
            item = run["items"][index]
            item["status"] = "running"
            self.save(run)
        output = self.root / run_id / "videos" / item["video"]
        label = Label(**item["label"])
        started = time.monotonic()
        result, error, judge = None, None, None

        def progress(_stage, _value):
            if self.get(run_id)["status"] == "cancelling":
                raise TimeoutError("Бенчмарк остановлен")

        try:
            result = pipeline.run_pipeline(
                Path(item["file"]),
                output,
                self.settings,
                f"benchmark-{run_id}-{item['video']}",
                progress,
                remaining_seconds=self.settings.job_timeout_sec,
            )
        except Exception as exc:
            error = str(exc)[:300] if isinstance(exc, (ValueError, TimeoutError)) else type(exc).__name__
        finally:
            # Journals and result.json stay for inspection; thousands of sampled JPEGs do not.
            shutil.rmtree(output / "frames", ignore_errors=True)
        processing = time.monotonic() - started
        score = score_video(label, result, self.settings.benchmark_tolerance_sec)
        if score["outcome"] == "TP" and self.settings.llm_enabled:
            try:
                judge = judge_video(self.settings, label, result)
            except Exception as exc:
                judge = f"Судья недоступен ({type(exc).__name__})"
        with self.lock:
            run = self.get(run_id)
            item = run["items"][index]
            item.update(
                status="failed" if result is None else "done",
                error=error,
                processing_sec=round(processing, 1),
                score=score,
                judge=judge,
                verdict=result.verdict if result else None,
                violations=[
                    {"type": v.type, "timecode_sec": v.timecode_sec, "description": v.description}
                    for v in (result.violations if result else [])
                ],
            )
            self.save(run)
        # Re-queue instead of looping, so an upload submitted meanwhile runs before the next video.
        self.executor.submit(self.step, run_id, index + 1)
