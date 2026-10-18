from __future__ import annotations

import base64
import json
from pathlib import Path

import httpx

from ..contracts import ActionJournal, Box, Candidate, Frame, Rule, Zone, ZoneProfile
from ..settings import Settings
from ..video import Deadline
from .catalog import eligible


def local_chat(
    settings: Settings,
    prompt: str,
    schema: dict,
    deadline: Deadline,
    images: list[Path] | None = None,
    max_seconds: float = 80,
    context_size: int = 16384,
    model: str | None = None,
) -> dict:
    # Verify the URL here too: callers outside the web app cannot bypass locality.
    settings.validate()
    message = {"role": "user", "content": prompt}
    if images:
        message["images"] = [base64.b64encode(path.read_bytes()).decode("ascii") for path in images]
    timeout = min(max_seconds, deadline.remaining())
    with httpx.Client(trust_env=False, timeout=timeout) as client:
        response = client.post(
            settings.ollama_url.rstrip("/") + "/api/chat",
            json={
                "model": model or settings.llm_model,
                "messages": [message],
                "format": schema,
                "stream": False,
                # Hybrid reasoning models (qwen3, gemma4) would otherwise spend the budget on thinking.
                "think": False,
                "keep_alive": "10m",
                "options": {"temperature": 0, "seed": 42, "num_ctx": context_size, "num_predict": 1200},
            },
        )
        response.raise_for_status()
        result = response.json()
    if result.get("done_reason") != "stop":
        raise ValueError("Local LLM response was truncated")
    content = result["message"]["content"]
    # Some templates leak reasoning ending in </think> even with think=false; keep only the answer.
    return json.loads(content.rsplit("</think>", 1)[-1])


def auto_zones(settings: Settings, frame: Frame, deadline: Deadline) -> ZoneProfile:
    schema = {
        "type": "object",
        "properties": {
            "zones": {
                "type": "array",
                "maxItems": 4,
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {
                            "type": "string",
                            "enum": ["cashier_area", "client_area", "counter_area", "storage_area"],
                        },
                        "x1": {"type": "number", "minimum": 0, "maximum": 1},
                        "y1": {"type": "number", "minimum": 0, "maximum": 1},
                        "x2": {"type": "number", "minimum": 0, "maximum": 1},
                        "y2": {"type": "number", "minimum": 0, "maximum": 1},
                    },
                    "required": ["name", "x1", "y1", "x2", "y2"],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["zones"],
        "additionalProperties": False,
    }
    prompt = (
        "Estimate rectangular regions in this cash-counter image. Return normalized coordinates, "
        "0..1, origin top-left. cashier_area is where a worker's BODY is located; client_area is "
        "the customer side; counter_area is the horizontal cash handling surface; storage_area "
        "only if a safe/storage opening is clearly visible. Different regions must not all cover "
        "the full image. Do not identify people. Omit regions that cannot be located. Return only JSON."
    )
    result = local_chat(
        settings, prompt, schema, deadline, [Path(frame.image_path)], max_seconds=40, model=settings.zone_model
    )
    zones = [
        Zone(name=item["name"], box=Box(**{key: item[key] for key in ("x1", "y1", "x2", "y2")}))
        for item in result["zones"]
    ]
    if len({zone.name for zone in zones}) != len(zones):
        raise ValueError("Automatic profile contains duplicate zones")
    return ZoneProfile(source="auto_estimated", zones=zones)


def llm_candidates(
    journal: ActionJournal, rules: list[Rule], scenarios: dict, settings: Settings, deadline: Deadline
) -> tuple[list[Candidate], list[str]]:
    enabled = [rule for rule in rules if eligible(rule, journal)]
    if not settings.llm_enabled:
        return [], ["Языковая модель отключена настройкой; результат построен только по правилам."]
    if not enabled:
        return [], ["Для текущих возможностей распознавания нет активных правил LLM."]
    event_map = {event.event_id: event for event in journal.actions}
    state_map = {state.state_id: state for state in journal.states}
    if not event_map and not state_map:
        return [], ["Журнал пуст; LLM не запускалась."]
    compact = {
        "actions_columns": [
            "event_id",
            "type",
            "start_sec",
            "end_sec",
            "actor_ids",
            "object_ids",
            "session_id",
            "visibility",
            "confidence",
            "attributes",
        ],
        "actions": [
            [
                event.event_id,
                event.type,
                event.start_sec,
                event.end_sec,
                event.actor_ids,
                event.object_ids,
                event.session_id,
                event.visibility,
                event.confidence,
                event.attributes,
            ]
            for event in journal.actions
        ],
        "states_columns": [
            "state_id",
            "name",
            "value",
            "start_sec",
            "end_sec",
            "entity_ids",
            "session_id",
            "visibility",
            "source_event_ids",
        ],
        "states": [
            [
                state.state_id,
                state.name,
                state.value,
                state.start_sec,
                state.end_sec,
                state.entity_ids,
                state.session_id,
                state.visibility,
                state.source_event_ids,
            ]
            for state in journal.states
        ],
        "roles": journal.roles,
        "batches": [batch.model_dump() for batch in journal.batches],
        "capabilities": journal.capabilities,
        "unsupported_actions": journal.unsupported_actions,
        "coverage": journal.coverage,
        "rules": [rule.model_dump() for rule in enabled],
        "scenarios": scenarios,
    }
    payload = json.dumps(compact, ensure_ascii=False, separators=(",", ":"))
    if len(payload) > 90000:
        # Never silently truncate the timeline, especially when checking omissions.
        return [], ["Журнал превышает контекст MVP-резонера; ветка LLM пропущена, правила в коде выполнены."]

    def reference_array(ids):
        return {
            "type": "array",
            "maxItems": min(12, len(ids)),
            "items": {"type": "string", **({"enum": list(ids)} if ids else {})},
        }

    schema = {
        "type": "object",
        "properties": {
            "candidates": {
                "type": "array",
                "maxItems": 8,
                "items": {
                    "type": "object",
                    "properties": {
                        "rule_id": {"type": "string", "enum": [rule.id for rule in enabled]},
                        "event_ids": reference_array(event_map),
                        "state_ids": reference_array(state_map),
                        "reasoning": {"type": "string"},
                    },
                    "required": ["rule_id", "event_ids", "state_ids", "reasoning"],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["candidates"],
        "additionalProperties": False,
    }
    prompt = (
        "Review a cash-operation journal against the supplied rules and scenarios. Answer in Russian. "
        "Propose candidates only, using existing rule_id, event_ids and state_ids. Do not write timestamps "
        "or invent normative sources. Empty candidates is valid. Unknown/missing detections do NOT prove "
        "absence. Never infer that an unsupported action was omitted. An omission needs complete visible "
        "scenario coverage, an observed closing transition and applicable exceptions checked. Do not assume "
        "a person at a window is already being served. Contact/holding and roles are estimates. A door closed "
        "does not prove locked. Objects must belong to the same relevant batch/session. source_status needs_review "
        "means the normative mapping remains provisional. Give short reasoning, at most 35 words per candidate. "
        "Use only the provided observations and rules.\n" + payload
    )
    result = local_chat(
        settings, prompt, schema, deadline, context_size=32768 if len(payload) > 45000 else 16384
    )
    rule_map = {rule.id: rule for rule in enabled}
    candidates, warnings = [], []
    for item in result.get("candidates", []):
        try:
            rule = rule_map[item["rule_id"]]
            events = [event_map[key] for key in dict.fromkeys(item["event_ids"])]
            states = [state_map[key] for key in dict.fromkeys(item["state_ids"])]
            if not events and not states:
                raise ValueError("Candidate has no supporting references")
            if rule.kind == "omission" and (
                not journal.coverage.get("complete_observation")
                or not journal.coverage.get("scenario_start_observed")
            ):
                raise ValueError("Omission lacks coverage")
            sessions = {ref.session_id for ref in [*events, *states] if ref.session_id}
            if len(sessions) > 1 and not rule.engine.get("allow_cross_session"):
                raise ValueError("Candidate combines unrelated sessions")
            if rule.kind == "state" and states:
                start, end = max(s.start_sec for s in states), min(s.end_sec for s in states)
                anchor = start
            else:
                start, end = min(e.start_sec for e in events), max(e.end_sec for e in events)
                anchor = end if rule.kind == "omission" else start
            if start > end:
                raise ValueError("No common state interval")
            candidates.append(
                Candidate(
                    candidate_id="pending",
                    rule_id=rule.id,
                    kind=rule.kind,
                    event_ids=[e.event_id for e in events],
                    state_ids=[s.state_id for s in states],
                    actor_ids=sorted({actor for e in events for actor in e.actor_ids}),
                    object_ids=sorted({obj for e in events for obj in e.object_ids}),
                    session_id=next(iter(sessions), None),
                    start_sec=start,
                    end_sec=end,
                    timecode_sec=anchor,
                    sources=["llm"],
                    evidence_frame_ids=list(
                        dict.fromkeys(frame for ref in [*events, *states] for frame in ref.evidence_frame_ids)
                    ),
                    reasoning=str(item["reasoning"]),
                )
            )
        except (KeyError, ValueError, TypeError):
            warnings.append("Предложение языковой модели отброшено: некорректные ссылки или контекст.")
    return candidates, warnings
