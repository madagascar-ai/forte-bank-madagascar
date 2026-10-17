from __future__ import annotations

import math
from collections import Counter, defaultdict
from itertools import pairwise

from ..contracts import Action, ActionJournal, Batch, Detection, ObjectJournal, State, ZoneProfile

# The detector vocabulary evolves; the journal keeps one internal name per object kind.
LABEL_ALIASES = {
    "cash": "money",
    "banknote": "money",
    "banknotes": "money",
    "id_card": "document",
    "identity_card": "document",
    "smartphone": "phone",
    "cell phone": "phone",
    "counting_machine": "counter",
    "banknote counting machine": "counter",
}
# Counting is inferred from geometry, not from a model: cash stays inside the counting machine for
# at least COUNT_MIN_SEC, and the cashier's hand is at the machine shortly before and after.
COUNT_MIN_SEC = 1.0
COUNT_HAND_WINDOW_SEC = 2.0
# Objects that should not stay at the cashier's workstation (or only while she is there).
WORKSTATION_ITEMS = ("personal_item", "phone", "cash_bag", "stamp", "folder")
WORKSTATION_ZONES = {"cashier_area", "counter_area"}
TRANSITIONS = {"open": "opened", "closed": "closed", "on": "turned_on", "off": "turned_off"}


def _overlaps(inner: Detection, outer: Detection, margin: float = 0.03) -> bool:
    x, y = inner.box.center
    return (
        outer.box.x1 - margin <= x <= outer.box.x2 + margin
        and outer.box.y1 - margin <= y <= outer.box.y2 + margin
    )


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
    for frame in objects.frames:
        for item in frame.detections:
            item.label = LABEL_ALIASES.get(item.label, item.label)
    roles = assign_roles(objects, profile)
    journal = ActionJournal(duration_sec=duration, sample_fps=fps, roles=roles)
    journal.warnings.extend(objects.warnings)
    if not profile.zones:
        journal.warnings.append("Зоны не определены: пространственные проверки ограничены.")
    elif profile.source.startswith("auto"):
        journal.warnings.append("Зоны определены автоматически; проверьте их расположение.")
    journal.unsupported_actions = [
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
        "previous_cash_on_counter",
        "workstation_items_observed",
        "safe_key_in_lock",
        "extra_person_present",
        "cash_counted",
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
    first_seen: dict[str, float] = {}
    session_started: dict[str, float] = {}
    session_order: list[str] = []
    cashier_hand_at_counter: list[float] = []
    counting: dict | None = None  # current stay of cash inside the machine
    pending_counts: list[dict] = []
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
                session_started[active_session] = frame.timestamp_sec
                session_order.append(active_session)
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
            first_seen.setdefault(item.track_id, frame.timestamp_sec)
        # Cash that was already on the counter before the current client arrived belongs to an
        # earlier operation (Rules No. 231, p. 20). New track IDs never count as earlier cash.
        previous_cash = None
        if active_session and session_order.index(active_session) > 0:
            started = session_started[active_session]
            if any(first_seen[m.track_id] < started - 0.5 for m in observed_counter_cash):
                previous_cash = True
        state("previous_cash_on_counter", previous_cash, frame)

        for label in WORKSTATION_ITEMS:
            found = []
            for item in frame.detections:
                if item.label != label or zone_of(item, profile) not in WORKSTATION_ZONES:
                    continue
                if label == "phone" and roles.get(_contact(item, hands, people) or "") == "client":
                    continue  # a client may show a digital ID on the phone (p. 19.1, p. 31.1)
                found.append(item.track_id)
            state(f"{label}_at_workstation", True if found else None, frame)

        safes = [d for d in frame.detections if d.label == "safe"]
        keys_in_lock = [
            k for k in frame.detections if k.label == "safe_key" and any(_overlaps(k, s) for s in safes)
        ]
        state("safe_key_in_lock", True if keys_in_lock else None, frame)

        counters = [d for d in frame.detections if d.label == "counter"]
        cashier_people = [p for p in people if roles.get(p.track_id) == "cashier"]
        if counters and any(
            any(_overlaps(h, c) for c in counters) and any(_overlaps(h, p, 0.08) for p in cashier_people)
            for h in hands
        ):
            cashier_hand_at_counter.append(frame.timestamp_sec)
        cash_in_machine = [m for m in money_items if any(_overlaps(m, c, 0.0) for c in counters)]
        if cash_in_machine:
            if counting is None:
                counting = {"start": frame.timestamp_sec, "frame": frame, "objects": set()}
            counting["end"], counting["last_frame"] = frame.timestamp_sec, frame
            counting["objects"].update(m.track_id for m in cash_in_machine)
        elif counting is not None:
            if counting["end"] - counting["start"] >= COUNT_MIN_SEC:
                pending_counts.append(counting)
            counting = None
        for stay in list(pending_counts):
            before = any(
                stay["start"] - COUNT_HAND_WINDOW_SEC <= t <= stay["start"] + 0.5
                for t in cashier_hand_at_counter
            )
            after = any(
                stay["end"] - 0.5 <= t <= stay["end"] + COUNT_HAND_WINDOW_SEC for t in cashier_hand_at_counter
            )
            if before and after:
                counted = event(
                    "cash_counted",
                    stay["last_frame"],
                    [key for key, role in roles.items() if role == "cashier"],
                    sorted(stay["objects"]),
                    attributes={"method": "counting_machine"},
                    confidence=0.6,
                    start=stay["start"],
                )
                counted.evidence_frame_ids = list(
                    dict.fromkeys([stay["frame"].frame_id, stay["last_frame"].frame_id])
                )
                pending_counts.remove(stay)
            elif frame.timestamp_sec > stay["end"] + COUNT_HAND_WINDOW_SEC:
                pending_counts.remove(stay)

        others = [p for p in people if roles.get(p.track_id) != "cashier"]
        state("extra_person_present", True if len(others) >= 2 else None, frame)

        for item in frame.detections:
            previous_time = last_observed.get(item.track_id)
            contiguous = previous_time is not None and frame.timestamp_sec - previous_time <= 1.5 / fps + 0.05
            last_observed[item.track_id] = frame.timestamp_sec
            if item.attributes.get("state") in {"open", "closed", "on", "off"}:
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
                        else f"{item.label}_{TRANSITIONS.get(new_value, new_value)}"
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
        "Пересчёт на счётной машинке распознаётся приближённо, по геометрии рук и купюр; "
        "идентификация и состояние замка сейфа базовой геометрией не устанавливаются."
    )
    return journal
