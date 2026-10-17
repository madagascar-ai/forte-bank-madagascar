from __future__ import annotations

from itertools import chain

from ..contracts import ActionJournal, Candidate, Rule
from .catalog import eligible


def _candidate(rule, events, states, start, end, anchor, reason):
    return Candidate(
        candidate_id="pending",
        rule_id=rule.id,
        kind=rule.kind,
        event_ids=[event.event_id for event in events],
        state_ids=[state.state_id for state in states],
        actor_ids=sorted(set(chain.from_iterable(e.actor_ids for e in events))),
        object_ids=sorted(set(chain.from_iterable(e.object_ids for e in events))),
        session_id=next((item.session_id for item in [*events, *states] if item.session_id), None),
        start_sec=start,
        end_sec=end,
        timecode_sec=anchor,
        sources=["rule_engine"],
        evidence_frame_ids=list(
            dict.fromkeys(chain.from_iterable(item.evidence_frame_ids for item in [*events, *states]))
        ),
        reasoning=reason,
    )


def rule_candidates(journal: ActionJournal, rules: list[Rule], scenarios: dict) -> list[Candidate]:
    """Pure code; no model calls. Unknown observations never satisfy a negative condition."""
    found = []
    for rule in rules:
        if not eligible(rule, journal):
            continue
        engine = rule.engine
        kind = engine.get("type")
        if kind == "event":
            for event in journal.actions:
                if event.type == engine["event_type"]:
                    found.append(
                        _candidate(
                            rule,
                            [event],
                            [],
                            event.start_sec,
                            event.end_sec,
                            event.start_sec,
                            "В журнале найдено событие, указанное правилом.",
                        )
                    )
        elif kind == "overlap":
            intervals = [(0.0, journal.duration_sec, [], None)]
            for name, expected in engine["states"].items():
                states = [
                    state
                    for state in journal.states
                    if state.name == name
                    and type(state.value) is type(expected)
                    and state.value == expected
                    and state.visibility == "clear"
                ]
                next_intervals = []
                for start, end, used, session in intervals:
                    for state in states:
                        if session and state.session_id and session != state.session_id:
                            continue
                        left, right = max(start, state.start_sec), min(end, state.end_sec)
                        if right > left:
                            next_intervals.append((left, right, [*used, state], session or state.session_id))
                intervals = next_intervals
            for start, end, states, session in intervals:
                if end - start < engine.get("min_duration_sec", 0):
                    continue
                anchor_events = [
                    event
                    for event in journal.actions
                    if event.type == engine.get("anchor_event")
                    and abs(event.end_sec - start) <= 1 / journal.sample_fps + 0.01
                    and event.session_id == session
                ]
                if engine.get("anchor_event") and not anchor_events:
                    continue
                found.append(
                    _candidate(
                        rule,
                        anchor_events,
                        states,
                        start,
                        end,
                        start,
                        "Наблюдаемые состояния пересекаются в одном интервале обслуживания.",
                    )
                )
        elif kind == "before_event":
            # An omission requires a complete, visible scenario AND checked exceptions.
            if not journal.coverage.get("scenario_start_observed") or not journal.coverage.get(
                "complete_observation"
            ):
                continue
            if not all(journal.coverage.get(flag) for flag in engine.get("required_context_flags", [])):
                continue
            for closing in journal.actions:
                if closing.type != engine["closing_event"] or closing.visibility != "clear":
                    continue
                if any(
                    closing.attributes.get(key) != value
                    for key, value in engine.get("closing_attributes", {}).items()
                ):
                    continue
                observed = [
                    event
                    for event in journal.actions
                    if event.type == engine["required_event"]
                    and event.visibility == "clear"
                    and event.end_sec <= closing.end_sec
                    and event.session_id == closing.session_id
                    and (
                        not engine.get("same_object") or bool(set(event.object_ids) & set(closing.object_ids))
                    )
                ]
                if not observed:
                    found.append(
                        _candidate(
                            rule,
                            [closing],
                            [],
                            closing.start_sec,
                            closing.end_sec,
                            closing.end_sec,
                            "Окно обязательного шага закрыто; шаг не наблюдался при подтверждённом покрытии.",
                        )
                    )
    return found
