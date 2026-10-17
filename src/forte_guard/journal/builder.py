from __future__ import annotations

import math
from collections import Counter, defaultdict
from itertools import pairwise

from ..contracts import Action, ActionJournal, Batch, Box, Detection, ObjectJournal, State, ZoneProfile

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
# at least COUNT_MIN_SEC, with a cashier-role person (not a hand — the fine-tuned detector has no
# hand class) near the machine for most of that dwell.
COUNT_MIN_SEC = 1.0
COUNT_PRESENCE_RATIO = 0.9
# Objects that should not stay at the cashier's workstation (or only while she is there).
WORKSTATION_ITEMS = ("personal_item", "phone", "cash_bag", "stamp", "folder")
WORKSTATION_ZONES = {"cashier_area", "counter_area"}
# Presence mode "zone": an empty workstation counts as an absence only after this wait. Vanishing in the
# middle of the area is more likely bending down or occlusion than leaving, so it waits longer.
ABSENT_MIN_SEC = 3.0
ABSENT_MIDZONE_SEC = 6.0
ZONE_EDGE_MARGIN = 0.05
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


class ZonePresence:
    """Cashier presence from the workstation itself, independent of which track id sits there.

    An empty workstation becomes an observed absence only after a wait, and only while the rest of
    the frame is still detected; a vanished track alone stays unknown.
    """

    def __init__(self, required: int, area: Box | None, exit_area: Box | None):
        self.required = required
        self.area = area
        self.exit_area = exit_area
        self.value = None
        self.pending = None
        self.count = 0
        self.seen = False
        self.last_in: float | None = None
        self.empty_since: float | None = None
        self.edge = False
        self.ids: list[str] = []

    def _near_exit(self, point: tuple[float, float]) -> bool:
        x, y = point
        a = self.area
        near_edge = min(x - a.x1, a.x2 - x, y - a.y1, a.y2 - y) <= ZONE_EDGE_MARGIN
        return near_edge or bool(self.exit_area and self.exit_area.contains(point))

    def observe(self, t: float, people: list[Detection], hands: list[Detection], walked_away: bool, alive: bool):
        if self.area is None:
            return None
        inside = [p for p in people if self.area.contains(p.box.center)]
        if inside or any(self.area.contains(h.box.center) for h in hands):
            self.seen, self.last_in, self.empty_since = True, t, None
            if inside:
                self.ids = [p.track_id for p in inside]
                self.edge = any(self._near_exit(p.box.center) for p in inside)
            return True
        if not alive or not self.seen:
            return None
        if self.empty_since is None:
            self.empty_since = t
        wait = ABSENT_MIN_SEC if self.edge else ABSENT_MIDZONE_SEC
        if walked_away or t - self.last_in >= wait:
            return False
        return None

    def update(self, observed):
        """Return (previous, current) like StableValue, but an unknown gap keeps the confirmed value."""
        previous = self.value
        if observed is None:
            self.pending, self.count = None, 0
            return previous, None
        if observed == self.value:
            self.pending, self.count = None, 0
        else:
            if observed == self.pending:
                self.count += 1
            else:
                self.pending, self.count = observed, 1
            if self.count >= self.required:
                self.value = observed
                self.pending, self.count = None, 0
        return previous, self.value


def _nearest_session(point, clients: list[Detection], sessions: dict[str, str], fallback: str | None):
    """Attribute a detection to whichever currently-visible client is physically closest.

    A single frame-global `active_session` breaks when two clients are in frame at once (a queue);
    this picks the session of the nearer one instead of whichever client happened to be last in the
    detector's output order.
    """
    if not clients:
        return fallback
    nearest = min(clients, key=lambda p: math.dist(p.box.center, point))
    return sessions.get(nearest.track_id, fallback)


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


def build_journal(
    objects: ObjectJournal, profile: ZoneProfile, duration: float, fps: float, presence: str = "track"
) -> ActionJournal:
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
    areas = {zone.name: zone.box for zone in profile.zones}
    workstation = ZonePresence(required, areas.get("cashier_area"), areas.get("exit_area"))
    open_states: dict[tuple, State] = {}
    active_session: str | None = None
    sessions: dict[str, str] = {}
    service_sessions: set[str] = set()
    batches: dict[str, Batch] = {}
    last_known_side = {}
    locations = {}
    object_states = {}
    emitted_device_contacts = set()
    last_observed = {}
    first_seen: dict[str, float] = {}
    session_started: dict[str, float] = {}
    session_order: list[str] = []
    counting: dict | None = None  # current stay of cash inside the machine
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
        session=None,
    ):
        action = Action(
            event_id=f"event_{len(journal.actions) + 1:06d}",
            type=kind,
            start_sec=frame.timestamp_sec if start is None else start,
            end_sec=frame.timestamp_sec,
            actor_ids=actor_ids or [],
            object_ids=object_ids or [],
            session_id=session if session is not None else active_session,
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
            smoothers[(track_id, "side")].update(None)
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
        transition_ids = []
        if presence == "zone":
            present = workstation.observe(
                frame.timestamp_sec,
                [p for p in people if roles.get(p.track_id) != "client"],
                hands,
                walked_away=any(key in visible for key in cashier_ids),
                alive=bool(frame.detections),
            )
            old, stable_present = workstation.update(present)
            if old is True and stable_present is False:
                transition_ids = [
                    event(
                        "cashier_left_workstation", frame, workstation.ids, start=workstation.empty_since
                    ).event_id
                ]
            elif old is False and stable_present is True:
                transition_ids = [event("cashier_returned", frame, workstation.ids).event_id]
        else:
            present = (
                True
                if inside
                else (False if cashier_ids and all(key in visible for key in cashier_ids) else None)
            )
            old, stable_present = smoothers["cashier_present"].update(present)
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
        cashier_near_machine = any(_overlaps(p, c, margin=0.05) for p in cashier_people for c in counters)
        cash_in_machine = [m for m in money_items if any(_overlaps(m, c, 0.0) for c in counters)]
        if cash_in_machine:
            if counting is None:
                counting = {"start": frame.timestamp_sec, "frame": frame, "objects": set(), "samples": 0, "near": 0}
            counting["end"], counting["last_frame"] = frame.timestamp_sec, frame
            counting["objects"].update(m.track_id for m in cash_in_machine)
            counting["samples"] += 1
            counting["near"] += int(cashier_near_machine)
        elif counting is not None:
            duration = counting["end"] - counting["start"]
            presence_ratio = counting["near"] / counting["samples"] if counting["samples"] else 0.0
            if duration >= COUNT_MIN_SEC and presence_ratio >= COUNT_PRESENCE_RATIO:
                counted = event(
                    "cash_counted",
                    counting["last_frame"],
                    [key for key, role in roles.items() if role == "cashier"],
                    sorted(counting["objects"]),
                    attributes={"method": "counting_machine"},
                    confidence=0.6,
                    start=counting["start"],
                )
                counted.evidence_frame_ids = list(
                    dict.fromkeys([counting["frame"].frame_id, counting["last_frame"].frame_id])
                )
            counting = None

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
            if item.label == "phone":
                # Kept hand-based for now: the client-phone exclusion below needs a decision with the
                # team before it's rewritten (no phone class on detector v1 either, so moot there).
                owner = _contact(item, hands, people)
                _, stable_owner = smoothers[(item.track_id, "owner")].update(owner)
                contact_key = (item.track_id, stable_owner)
                if stable_owner and contact_key not in emitted_device_contacts:
                    event(
                        "device_used_observed",
                        frame,
                        [stable_owner],
                        [item.track_id],
                        attributes={"object_type": "phone", "actor_role": roles.get(stable_owner, "unknown")},
                        confidence=min(item.confidence, 0.7),
                        visibility="uncertain",
                    )
                    emitted_device_contacts.add(contact_key)
                continue
            if item.label == "document":
                # No hand contact available: attribute to whichever client is nearest right now,
                # not to "whoever's hand is on it" — session-scoped, not person-scoped.
                nearest_session = _nearest_session(item.box.center, clients, sessions, active_session)
                contact_key = (item.track_id, nearest_session)
                if nearest_session and contact_key not in emitted_device_contacts:
                    event(
                        "document_presented",
                        frame,
                        [],
                        [item.track_id],
                        attributes={"object_type": "document"},
                        confidence=min(item.confidence, 0.7),
                        visibility="uncertain",
                        session=nearest_session,
                    )
                    emitted_device_contacts.add(contact_key)
                continue
            if item.label != "money":
                continue
            batch = batches.setdefault(item.track_id, Batch(batch_id=item.track_id))
            location = zone_of(item, profile)
            side = "client" if location == "client_area" else ("cashier" if location in {"cashier_area", "counter_area"} else None)
            _, stable_side = smoothers[(item.track_id, "side")].update(side)
            former = last_known_side.get(item.track_id)
            if stable_side and former and stable_side != former[0] and frame.timestamp_sec - former[1] <= 4:
                from_role, to_role = former[0], stable_side
                action = event(
                    "cash_transferred",
                    frame,
                    [],
                    [item.track_id],
                    {"from_role": from_role, "to_role": to_role},
                    confidence=min(item.confidence, 0.6),
                    start=former[1],
                    visibility="uncertain",
                    session=_nearest_session(item.box.center, clients, sessions, active_session),
                )
                action.evidence_frame_ids = list(dict.fromkeys([former[2], frame.frame_id]))
                if from_role == "client" and active_session:
                    batch.session_id = active_session
                    batch.provenance = "observed"
                    service_sessions.add(active_session)
                    state("service_active", True, frame, source_events=[action.event_id])
            if stable_side:
                last_known_side[item.track_id] = (stable_side, frame.timestamp_sec, frame.frame_id)
            if (
                location == "counter_area"
                and locations.get(item.track_id) not in {None, "counter_area"}
                and contiguous
            ):
                event(
                    "cash_placed_on_counter",
                    frame,
                    object_ids=[item.track_id],
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
            locations[item.track_id] = location

    journal.batches = list(batches.values())
    journal.actions.sort(key=lambda a: (a.start_sec, a.end_sec, a.event_id))
    journal.warnings.append(
        "Передача денег, пересчёт на машинке и показ документа определяются по зонам и сессии "
        "клиента, без привязки к конкретной руке — на видео с несколькими клиентами одновременно "
        "в кадре привязка может быть неточной. Потеря трека пачки денег при передаче из рук в руки "
        "не восстанавливается. Идентификация и состояние замка сейфа базовой геометрией "
        "не устанавливаются."
    )
    return journal
