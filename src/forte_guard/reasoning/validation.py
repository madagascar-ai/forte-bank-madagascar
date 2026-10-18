from __future__ import annotations

from ..contracts import ActionJournal, Candidate, Frame, Rule


def merge_candidates(candidates: list[Candidate]) -> list[Candidate]:
    merged = []
    for item in sorted(candidates, key=lambda candidate: (candidate.rule_id, candidate.start_sec)):
        existing = next(
            (
                other
                for other in reversed(merged)
                if other.rule_id == item.rule_id
                and other.session_id == item.session_id
                and (
                    not other.object_ids
                    or not item.object_ids
                    or set(other.object_ids) & set(item.object_ids)
                )
                and max(other.start_sec, item.start_sec) <= min(other.end_sec, item.end_sec)
            ),
            None,
        )
        if existing is None:
            merged.append(item.model_copy(deep=True))
        else:
            for field in (
                "event_ids",
                "state_ids",
                "actor_ids",
                "object_ids",
                "sources",
                "evidence_frame_ids",
            ):
                setattr(existing, field, list(dict.fromkeys(getattr(existing, field) + getattr(item, field))))
            existing.start_sec = min(existing.start_sec, item.start_sec)
            existing.end_sec = max(existing.end_sec, item.end_sec)
            existing.timecode_sec = (
                max(existing.timecode_sec, item.timecode_sec)
                if item.kind == "omission"
                else min(existing.timecode_sec, item.timecode_sec)
            )
            if item.reasoning not in existing.reasoning:
                existing.reasoning += " " + item.reasoning
    for index, item in enumerate(merged):
        item.candidate_id = f"candidate_{index + 1:04d}"
    return merged


def validate_candidates(
    candidates: list[Candidate], journal: ActionJournal, frames: list[Frame], rules: list[Rule]
) -> tuple[list[Candidate], list[str]]:
    """Minimal structural verification only; deliberately no expensive VLM judge in MVP."""
    event_map = {event.event_id: event for event in journal.actions}
    state_map = {state.state_id: state for state in journal.states}
    frame_map = {frame.frame_id: frame for frame in frames}
    rule_map = {rule.id: rule for rule in rules if rule.enabled}
    accepted, warnings = [], []
    for item in candidates:
        valid = (
            item.rule_id in rule_map
            and item.kind == rule_map[item.rule_id].kind
            and bool(item.event_ids or item.state_ids)
            and all(key in event_map for key in item.event_ids)
            and all(key in state_map for key in item.state_ids)
            and bool(item.evidence_frame_ids)
            and all(key in frame_map for key in item.evidence_frame_ids)
            and 0 <= item.start_sec <= item.timecode_sec <= item.end_sec <= journal.duration_sec
        )
        if not valid:
            item.status, item.verification = "rejected", "invalid_references_or_interval"
            warnings.append("Одно из возможных нарушений отброшено: некорректные ссылки или интервал.")
            continue
        rule = rule_map[item.rule_id]
        refs = [*[event_map[key] for key in item.event_ids], *[state_map[key] for key in item.state_ids]]
        sessions = {ref.session_id for ref in refs if ref.session_id}
        if not rule.engine.get("allow_cross_session") and (
            len(sessions) > 1 or (sessions and item.session_id not in sessions)
        ):
            item.status, item.verification = "rejected", "inconsistent_session"
            continue
        if not set(rule.required_capabilities).issubset(journal.capabilities):
            item.status, item.verification = "rejected", "unsupported_action"
            continue
        if item.kind == "omission" and not (
            journal.coverage.get("scenario_start_observed") and journal.coverage.get("complete_observation")
        ):
            item.status, item.verification = "rejected", "insufficient_coverage"
            continue
        # Cheap consistency checks against linked observations, not a visual judge.
        engine = rule.engine
        events = [event_map[key] for key in item.event_ids]
        states = [state_map[key] for key in item.state_ids]
        supported = True
        if engine.get("type") == "overlap":
            supported = item.end_sec - item.start_sec >= engine.get("min_duration_sec", 0)
            for name, expected in engine["states"].items():
                supported = supported and any(
                    state.name == name
                    and type(state.value) is type(expected)
                    and state.value == expected
                    and state.visibility == "clear"
                    and state.start_sec <= item.start_sec + 1e-6
                    and state.end_sec >= item.end_sec - 1e-6
                    for state in states
                )
            if engine.get("anchor_event"):
                supported = supported and any(
                    event.type == engine["anchor_event"]
                    and abs(event.end_sec - item.start_sec) <= 1 / journal.sample_fps + 0.01
                    for event in events
                )
        elif engine.get("type") == "event":
            supported = any(event.type == engine["event_type"] for event in events)
        elif engine.get("type") == "without_prior":
            supported = any(event.type == engine["closing_event"] for event in events)
        elif engine.get("type") == "before_event":
            supported = all(journal.coverage.get(flag) for flag in engine.get("required_context_flags", []))
        if not supported:
            item.status, item.verification = "rejected", "observations_do_not_support_rule"
            warnings.append("Одно из возможных нарушений отброшено: наблюдения не подтверждают условия.")
            continue
        # Pick a real decoded frame, not a numeric timestamp invented by a model.
        witness = min(
            (frame_map[key] for key in item.evidence_frame_ids),
            key=lambda frame: abs(frame.timestamp_sec - item.timecode_sec),
        )
        if not item.start_sec - 1e-6 <= witness.timestamp_sec <= item.end_sec + 1e-6:
            item.status, item.verification = "rejected", "no_witness_in_interval"
            continue
        item.timecode_sec = witness.timestamp_sec
        item.status, item.verification = "accepted", "structural_only"
        accepted.append(item)
        if rule.source_status != "verified":
            warnings.append(f"Ссылка на норму для «{rule.title}» предварительная: проверьте редакцию и применимость.")
    return accepted, list(dict.fromkeys(warnings))
