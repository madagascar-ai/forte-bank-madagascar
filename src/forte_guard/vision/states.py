"""Object states the detector cannot tell: door/safe open-closed, laptop on-off.

The detector finds the object; the local vision model (Ollama, team machine only) reads its state from a
crop. A track's state is reused until its crop visibly changes or goes stale, and the number of model
calls per video is bounded, so the stage fits the time limit. An unreadable crop gives no state at all:
unknown is never turned into closed or off.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from pathlib import Path

import cv2
import numpy as np

from ..contracts import Frame, ObjectJournal
from ..reasoning.llm import local_chat
from ..settings import Settings
from ..video import Deadline

QUESTIONS = {
    "door": (
        ("open", "closed"),
        "This crop from an overhead CCTV camera shows the door of a bank cash room. Is the door open or "
        "closed? open: the door leaf is swung away and the doorway or a gap is visible. closed: the leaf sits "
        "flush in its frame. Answer unknown if the door is mostly hidden or you cannot tell.",
    ),
    "safe": (
        ("open", "closed"),
        "This crop from an overhead CCTV camera shows a safe in a bank cash room. Is the safe door open or "
        "closed? open: the safe door is swung away and the inside is visible. closed: the door is shut "
        "flush. Answer unknown if the safe door is hidden or you cannot tell.",
    ),
    "laptop": (
        ("on", "off"),
        "This crop from an overhead CCTV camera shows a laptop on a bank cashier's desk. Is the laptop "
        "switched on? on: the screen is lit and shows an image or text. off: the screen is dark or black, "
        "or the lid is closed. A lid that is open with a dark screen is off. Answer unknown if the screen "
        "is not visible.",
    ),
}
MARGIN = 0.12  # context around the box: a door edge or the screen border helps the model
CHANGE = 12.0  # mean gray change (0-255) in the most changed 4x4 cell that counts as a new look
STALE_SEC = 10.0  # re-check a steady object at least this often
MIN_SIDE = 24  # pixels; smaller crops are not readable
BUDGET_SEC = 30.0  # hard cap on the whole stage, on top of FORTE_STATES_MAX_CALLS
# Same context as the other local-model calls: Ollama reloads the model whenever num_ctx changes.
CONTEXT = 16384


def _crop(image: np.ndarray, box) -> np.ndarray | None:
    height, width = image.shape[:2]
    pad_x, pad_y = (box.x2 - box.x1) * MARGIN, (box.y2 - box.y1) * MARGIN
    x1, y1 = int(max(0.0, box.x1 - pad_x) * width), int(max(0.0, box.y1 - pad_y) * height)
    x2, y2 = int(min(1.0, box.x2 + pad_x) * width), int(min(1.0, box.y2 + pad_y) * height)
    if x2 - x1 < MIN_SIDE or y2 - y1 < MIN_SIDE:
        return None
    return image[y1:y2, x1:x2]


def _signature(crop: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    return cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32)


def _change(current: np.ndarray, previous: np.ndarray) -> float:
    """Mean gray change (0-255) in the most changed cell of a 4x4 grid over the crop."""
    return float(np.abs(current - previous).reshape(4, 8, 4, 8).mean(axis=(1, 3)).max())


def read_states(
    frames: list[Frame],
    objects: ObjectJournal,
    settings: Settings,
    deadline: Deadline,
    workdir: Path,
    ask: Callable[..., dict] = local_chat,
) -> list[str]:
    """Set attributes["state"] on door, safe and laptop detections in place; return journal warnings.

    Every decision goes to state_trace.json next to the crops (team debug view; never in the jury result).
    """
    if not settings.object_states:
        return []
    paths = {frame.frame_id: frame.image_path for frame in frames}
    cache: dict[str, tuple[np.ndarray, float, str | None]] = {}  # track -> (signature, time, state)
    calls = reused = known = 0
    failures: set[str] = set()
    trace: list[dict] = []
    workdir.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()

    def record(frame, detection, action, state=None, change=None, **extra):
        trace.append({
            "t": frame.timestamp_sec, "frame_id": frame.frame_id, "track_id": detection.track_id,
            "label": detection.label, "action": action, "state": state,
            "change": None if change is None else round(change, 1), **extra,
        })

    for frame in objects.frames:
        targets = [d for d in frame.detections if d.label in QUESTIONS and "state" not in d.attributes]
        if not targets or frame.frame_id not in paths:
            continue
        image = cv2.imread(paths[frame.frame_id])
        if image is None:
            continue
        for detection in targets:
            crop = _crop(image, detection.box)
            if crop is None:
                record(frame, detection, "too_small")
                continue
            signature = _signature(crop)
            cached = cache.get(detection.track_id)
            change = None if cached is None else _change(signature, cached[0])
            unchanged = change is not None and change < CHANGE
            budget_left = (
                calls < settings.states_max_calls
                and time.monotonic() - started < BUDGET_SEC
                and deadline.remaining() > 60
            )
            # A steady object keeps its state; it is re-read when it changes or, budget permitting, goes stale.
            if unchanged and (frame.timestamp_sec - cached[1] < STALE_SEC or not budget_left):
                state, reused = cached[2], reused + 1
                record(frame, detection, "reused", state, change)
            elif budget_left:
                allowed, question = QUESTIONS[detection.label]
                name = f"{frame.frame_id}_{detection.track_id}.jpg"
                cv2.imwrite(str(workdir / name), crop, [cv2.IMWRITE_JPEG_QUALITY, 90])
                schema = {
                    "type": "object",
                    "properties": {
                        "state": {"type": "string", "enum": [*allowed, "unknown"]},
                        "reason": {"type": "string"},
                    },
                    "required": ["state", "reason"],
                }
                prompt = question + " Also give a short reason, at most 12 words, naming what you see."
                calls += 1
                why = "first look" if cached is None else ("changed" if not unchanged else "stale")
                try:
                    answer = ask(settings, prompt, schema, deadline, [workdir / name], 30, CONTEXT,
                                 settings.states_model)
                    raw = answer.get("state")
                    state = raw if raw in allowed else None
                    record(frame, detection, "asked", state, change, why=why, answer=raw,
                           reason=str(answer.get("reason", ""))[:200], crop=name)
                except Exception as exc:  # a failed read leaves the state unknown, never invented
                    failures.add(type(exc).__name__)
                    state = None
                    record(frame, detection, "failed", None, change, why=why, error=type(exc).__name__, crop=name)
                cache[detection.track_id] = (signature, frame.timestamp_sec, state)
            else:
                record(frame, detection, "no_budget", None, change)
                continue  # changed look and no budget left: leave this object unknown
            if state:
                detection.attributes["state"] = state
                detection.attributes["state_source"] = "local_vlm"
                known += 1
    (workdir.parent / "state_trace.json").write_text(json.dumps({
        "model": settings.states_model, "max_calls": settings.states_max_calls, "budget_sec": BUDGET_SEC,
        "change_threshold": CHANGE, "stale_sec": STALE_SEC, "calls": calls, "reused": reused,
        "seconds": round(time.monotonic() - started, 1), "entries": trace,
    }, ensure_ascii=False))
    notes = []
    if calls or reused:
        notes.append(
            f"Состояния двери, сейфа и ноутбука определены локальной vision-моделью {settings.states_model} "
            f"по вырезанным кадрам: {calls} запросов, {reused} повторных использований, "
            f"{known} обнаружений с известным состоянием; без ответа состояние считается неизвестным."
        )
    if calls >= settings.states_max_calls or time.monotonic() - started >= BUDGET_SEC:
        notes.append("Достигнут лимит запросов к vision-модели для состояний; часть кадров без состояния.")
    if failures:
        notes.append(f"Часть запросов о состоянии не удалась ({', '.join(sorted(failures))}).")
    return notes
