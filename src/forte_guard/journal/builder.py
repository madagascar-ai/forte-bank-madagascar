from __future__ import annotations

import math
from collections import Counter, defaultdict
from itertools import pairwise

from ..contracts import Action, ActionJournal, Batch, Detection, ObjectJournal, State, ZoneProfile


def zone_of(item: Detection, profile: ZoneProfile) -> str | None:
    # A narrow counter/storage zone takes priority over a broad room area.
    matches = [zone for zone in profile.zones if zone.box.contains(item.box.center)]
    if not matches:
        return None
    return min(matches, key=lambda zone: (zone.box.x2 - zone.box.x1) * (zone.box.y2 - zone.box.y1)).name


def person_in_zone(item: Detection, profile: ZoneProfile, name: str) -> bool:
    return any(zone.name == name and zone.box.contains(item.box.center) for zone in profile.zones)


def assign_roles(objects: ObjectJournal, profile: ZoneProfile) -> dict[str, str]:
    votes = defaultdict(list)
    for frame in objects.frames:
        for item in frame.detections:
            if item.label not in {"person", "cashier", "client"}:
                continue
            if item.label != "person":
                votes[item.track_id].append(item.label)
            elif len(votes[item.track_id]) < 10:
                if person_in_zone(item, profile, "cashier_area"):
                    votes[item.track_id].append("cashier")
                elif person_in_zone(item, profile, "client_area"):
                    votes[item.track_id].append("client")
                else:
                    votes[item.track_id].append("unknown")
    roles = {}
    for track_id, values in votes.items():
        role, count = Counter(values).most_common(1)[0]
        roles[track_id] = role if count / len(values) >= 0.7 else "unknown"
    return roles


class StableValue:
    """Debounce observed transitions. Missing observations explicitly become unknown."""

    def __init__(self, required: int):
        self.required = required
        self.value = None
        self.pending = None
        self.count = 0

    def update(self, value):
        previous = self.value
        if value is None:
            self.value = None
            self.pending, self.count = None, 0
        elif value == self.value:
            self.pending, self.count = None, 0
        else:
            if value == self.pending:
                self.count += 1
            else:
                self.pending, self.count = value, 1
            if self.count >= self.required:
                self.value = value
                self.pending, self.count = None, 0
        return previous, self.value


def _contact(money: Detection, hands: list[Detection], people: list[Detection]):
    owners = set()
    for hand in hands:
        mx, my = money.box.center
        hx, hy = hand.box.center
        near = math.hypot(mx - hx, my - hy) < max(0.035, (money.box.x2 - money.box.x1) * 0.8)
        if not near:
            continue
        candidates = [
            person
            for person in people
            if (
                person.box.x1 - 0.08 <= hx <= person.box.x2 + 0.08
                and person.box.y1 - 0.08 <= hy <= person.box.y2 + 0.08
            )
        ]
        if candidates:
            person = min(candidates, key=lambda p: math.dist(p.box.center, hand.box.center))
            owners.add(person.track_id)
    return next(iter(owners)) if len(owners) == 1 else None


def build_journal(objects: ObjectJournal, profile: ZoneProfile, duration: float, fps: float) -> ActionJournal:
    roles = assign_roles(objects, profile)
    journal = ActionJournal(duration_sec=duration, sample_fps=fps, roles=roles)
    journal.warnings.extend(objects.warnings)
    if not profile.zones:
        journal.warnings.append("Зоны не определены: пространственные проверки ограничены.")
    elif profile.source.startswith("auto"):
        journal.warnings.append("Зоны оценены локальной моделью; проверьте их расположение.")
    journal.unsupported_actions = [
        "cash_counted",
        "authenticity_verified",
        "identity_verified",
        "safe_locked",
        "seal_integrity_verified",
        "cash_batches_merged",
        "cash_batch_split",
        "cash_batches_mixed_across_sessions",
    ]
    journal.capabilities = [
        "object_state_changed",
        "cashier_left_workstation",
        "cashier_returned",
        "cash_transferred",
        "cash_placed_on_counter",
        "device_used_observed",
        "document_presented",
        "cash_entered_storage_area",
    ]
    required = max(1, math.ceil(fps * 0.4))
    smoothers = defaultdict(lambda: StableValue(required))
    open_states: dict[tuple, State] = {}
    active_session: str | None = None
    sessions: dict[str, str] = {}
    service_sessions: set[str] = set()
    batches: dict[str, Batch] = {}
    holders = {}
    last_known_holder = {}
    locations = {}
    object_states = {}
    emitted_device_contacts = set()
    last_observed = {}
    frame_times = [frame.timestamp_sec for frame in objects.frames]
    journal.coverage = {
        "first_frame_sec": frame_times[0] if frame_times else None,
        "last_frame_sec": frame_times[-1] if frame_times else None,
        "sampled_frames": len(frame_times),
        "scenario_start_observed": False,
        "max_gap_sec": max((b - a for a, b in pairwise(frame_times)), default=0),
        "zone_source": profile.source,
        "complete_observation": False,
    }

    def event(
        kind,
        frame,
        actor_ids=None,
        object_ids=None,
        attributes=None,
        confidence=0.6,
        start=None,
        visibility="clear",
    ):
        action = Action(
            event_id=f"event_{len(journal.actions) + 1:06d}",
            type=kind,
            start_sec=frame.timestamp_sec if start is None else start,
            end_sec=frame.timestamp_sec,
            actor_ids=actor_ids or [],
            object_ids=object_ids or [],
            session_id=active_session,
            evidence_frame_ids=[frame.frame_id],
            confidence=confidence,
            attributes=attributes or {},
            visibility=visibility,
        )
        journal.actions.append(action)
        return action

    def state(name, value, frame, entity_ids=None, source_events=None):
        key = (name, tuple(entity_ids or []))
        previous = open_states.get(key)
        if previous and previous.value == value and previous.session_id == active_session:
            previous.end_sec = frame.timestamp_sec
            previous.evidence_frame_ids = list(
                dict.fromkeys([previous.evidence_frame_ids[0], frame.frame_id])
            )
            return
        if previous:
            # A state is only extended to the next observed timestamp, not across an unknown gap.
            previous.end_sec = frame.timestamp_sec
        item = State(
            state_id=f"state_{len(journal.states) + 1:06d}",
            name=name,
            value=value,
            start_sec=frame.timestamp_sec,
            end_sec=frame.timestamp_sec,
            entity_ids=entity_ids or [],
            session_id=active_session,
            evidence_frame_ids=[frame.frame_id],
            source_event_ids=source_events or [],
            visibility="uncertain" if value is None else "clear",
        )
        journal.states.append(item)
        open_states[key] = item

    for frame in objects.frames:
        people = [d for d in frame.detections if d.label in {"person", "cashier", "client"}]
        hands = [d for d in frame.detections if d.label == "hand"]
        money_items = [d for d in frame.detections if d.label == "money"]
        visible = {p.track_id: p for p in people}
        observed_ids = {item.track_id for item in frame.detections}
        for _key, previous in list(open_states.items()):
            if previous.entity_ids and not set(previous.entity_ids) & observed_ids:
                state(previous.name, None, frame, previous.entity_ids)
        for track_id in set(last_observed) - observed_ids:
            smoothers[(track_id, "state")].update(None)
            smoothers[(track_id, "owner")].update(None)
            holders.pop(track_id, None)
        clients = [
            p
            for p in people
            if roles.get(p.track_id) == "client" and person_in_zone(p, profile, "client_area")
        ]
        for person in clients:
            if person.track_id not in sessions:
                sessions[person.track_id] = f"session_{len(sessions) + 1:04d}"
                old_session = active_session
                active_session = sessions[person.track_id]
                event("client_observed", frame, [person.track_id], attributes={"role": "client"})
                if old_session and old_session != active_session:
                    journal.coverage.setdefault("session_transitions_observed", True)
            active_session = sessions[person.track_id]

        cashier_ids = [key for key, role in roles.items() if role == "cashier"]
        inside = [
            key
            for key in cashier_ids
            if key in visible and person_in_zone(visible[key], profile, "cashier_area")
        ]
        present = (
            True
            if inside
            else (False if cashier_ids and all(key in visible for key in cashier_ids) else None)
        )
        old, stable_present = smoothers["cashier_present"].update(present)
        transition_ids = []
        if old is True and stable_present is False:
            transition_ids = [event("cashier_left_workstation", frame, cashier_ids).event_id]
        elif old is False and stable_present is True:
            transition_ids = [event("cashier_returned", frame, cashier_ids).event_id]
        state("cashier_present", stable_present, frame, source_events=transition_ids)
        state("client_present", True if clients else None, frame)

        observed_counter_cash = [m for m in money_items if zone_of(m, profile) == "counter_area"]
        state("cash_on_counter", True if observed_counter_cash else None, frame, source_events=[])
        state("service_active", True if active_session in service_sessions else None, frame)

        for item in frame.detections:
            previous_time = last_observed.get(item.track_id)
            contiguous = previous_time is not None and frame.timestamp_sec - previous_time <= 1.5 / fps + 0.05
            last_observed[item.track_id] = frame.timestamp_sec
            if item.attributes.get("state") in {"open", "closed"}:
                value = item.attributes["state"]
                _, new_value = smoothers[(item.track_id, "state")].update(value)
                state(
                    f"{item.label}_door_state" if item.label in {"safe", "door"} else f"{item.label}_state",
                    new_value,
                    frame,
                    [item.track_id],
                )
                if new_value and new_value != object_states.get(item.track_id):
                    # First sighting reports an observed state, not an invented opening action.
                    kind = (
                        f"{item.label}_state_observed"
                        if item.track_id not in object_states or not contiguous
                        else (f"{item.label}_{'opened' if new_value == 'open' else 'closed'}")
                    )
                    event(
                        kind,
                        frame,
                        object_ids=[item.track_id],
                        attributes={"state": new_value},
                        confidence=item.confidence,
                    )
                    object_states[item.track_id] = new_value
            if item.label not in {"money", "phone", "document"}:
                continue
            owner = _contact(item, hands, people)
            if item.label in {"phone", "document"}:
                _, stable_owner = smoothers[(item.track_id, "owner")].update(owner)
                contact_key = (item.track_id, stable_owner)
                if stable_owner and contact_key not in emitted_device_contacts:
                    event(
                        "device_used_observed" if item.label == "phone" else "document_presented",
                        frame,
                        [stable_owner],
                        [item.track_id],
                        attributes={
                            "object_type": item.label,
                            "actor_role": roles.get(stable_owner, "unknown"),
                        },
                        confidence=min(item.confidence, 0.7),
                        visibility="uncertain",
                    )
                    emitted_device_contacts.add(contact_key)
                continue
            batch = batches.setdefault(item.track_id, Batch(batch_id=item.track_id))
            _, stable_owner = smoothers[(item.track_id, "owner")].update(owner)
            former = last_known_holder.get(item.track_id)
            if stable_owner and former and stable_owner != former[0] and frame.timestamp_sec - former[1] <= 4:
                from_role, to_role = roles.get(former[0], "unknown"), roles.get(stable_owner, "unknown")
                action = event(
                    "cash_transferred",
                    frame,
                    [former[0], stable_owner],
                    [item.track_id],
                    {
                        "from_actor": former[0],
                        "to_actor": stable_owner,
                        "from_role": from_role,
                        "to_role": to_role,
                    },
                    confidence=min(item.confidence, 0.7),
                    start=former[1],
                    visibility="uncertain",
                )
                action.evidence_frame_ids = list(dict.fromkeys([former[2], frame.frame_id]))
                if from_role == "client" and active_session:
                    batch.session_id = active_session
                    batch.provenance = "observed"
                    service_sessions.add(active_session)
                    state("service_active", True, frame, source_events=[action.event_id])
            if stable_owner:
                last_known_holder[item.track_id] = (stable_owner, frame.timestamp_sec, frame.frame_id)
            location = zone_of(item, profile)
            if location == "counter_area" and holders.get(item.track_id) and owner is None and contiguous:
                event(
                    "cash_placed_on_counter",
                    frame,
                    [holders[item.track_id]],
                    [item.track_id],
                    confidence=min(item.confidence, 0.6),
                    visibility="uncertain",
                )
            if (
                location == "storage_area"
                and locations.get(item.track_id) not in {None, "storage_area"}
                and contiguous
            ):
                event(
                    "cash_entered_storage_area",
                    frame,
                    object_ids=[item.track_id],
                    attributes={"storage_contents_unknown": True},
                    confidence=item.confidence,
                )
            holders[item.track_id] = stable_owner
            locations[item.track_id] = location

    journal.batches = list(batches.values())
    journal.actions.sort(key=lambda a: (a.start_sec, a.end_sec, a.event_id))
    journal.warnings.append(
        "Пересчёт, идентификация и состояние замка базовой геометрией не устанавливаются."
    )
    return journal
