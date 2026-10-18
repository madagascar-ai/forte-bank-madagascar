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
# Fallback counting signal: the detector often loses the money box entirely right at the moment
# it is fed into or pulled out of the machine (occlusion by the hand at the slot, no overlap with
# the counter box ever gets recorded). If money vanishes for a while and the cashier stays at the
# machine the whole time, infer a counting episode even without a single overlapping box. This
# runs alongside the box-overlap check above, not instead of it — whichever evidence the detector
# actually produced on a given clip is the one that fires.
GAP_COUNT_MIN_SEC = 3.0
GAP_COUNT_MAX_SEC = 20.0
GAP_COUNT_NEAR_RATIO = 0.8
# After a counting episode (either signal above), the detector sometimes re-identifies the
# retrieved cash as a quick succession of different track ids before the pile settles — e.g. a
# cashier pulling out a flagged note separately from the rest, then merging them back into one
# stack. A single long gap is normal re-identification noise; several SHORT gaps in a row right
# after counting is the distinguishing pattern. Weak geometric proxy: needs_review, not proof.
FRAGMENT_WINDOW_SEC = 10.0
FRAGMENT_GAP_MAX_SEC = 2.0
FRAGMENT_MIN_PIECES = 2
# A cash track that changes side within this window counts as handed over.
HANDOFF_WINDOW_SEC = 4.0
# Cash that first appears on the counter while a client is at it is the client's: put down from the
# client's side, it is often first seen only once it lies there. Picked up into the cashier's area within
# HANDOFF_WINDOW_SEC, it was handed over even though no client-side sample was ever seen.
# Objects that should not stay at the cashier's workstation (or only while she is there).
WORKSTATION_ITEMS = ("personal_item", "phone", "cash_bag", "stamp", "folder")
WORKSTATION_ZONES = {"cashier_area", "counter_area"}
# A client handing something across the counter (phone with a digital ID, or a paper ID the
# detector mislabels as personal_item) should not immediately count as left at the workstation.
# The hand-contact check below only fires on detectors with a "hand" class (detector-v1 has none,
# so it's otherwise dead); this zone-trajectory fallback covers client-area objects crossing over.
CLIENT_ITEM_LABELS = {"phone", "personal_item"}
CLIENT_HANDOFF_GRACE_SEC = 5.0
# A detected document counts as shown from this many samples of its track.
DOCUMENT_MIN_SAMPLES = 2
# A role needs this many samples standing in an area, or this share of the person's samples.
MIN_ROLE_VOTES = 3
MIN_ROLE_SHARE = 0.3
# Presence mode "zone": an empty workstation counts as an absence only after this wait. Vanishing in the
# middle of the area is more likely bending down or occlusion than leaving, so it waits longer.
ABSENT_MIN_SEC = 3.0
ABSENT_MIDZONE_SEC = 6.0
# A client vanishing mid-area is read a bit more readily than the cashier's own (tuned, verified)
# threshold above — missing a witnessed recount matters more than a few extra false positives, but
# this is still a deliberate wait, not instant, to stay clear of a brief occlusion.
CLIENT_ABSENT_MIDZONE_SEC = 5.0
# A clip that simply ends while the client is untracked and counting is still open never lets the
# wait above run its course — there is no more footage to confirm or refute it. Treat that as a
# departure only when both hold: she is gone for a real stretch (not one dropped sample) AND that
# stretch starts close enough to the very last frame that it looks like the clip was cut off
# mid-departure, not a long, unexplained absence earlier in an otherwise complete recording.
TAIL_UNTRACKED_MIN_SEC = 0.5
TAIL_WINDOW_SEC = 2.0
ZONE_EDGE_MARGIN = 0.05
TRANSITIONS = {"open": "opened", "closed": "closed", "on": "turned_on", "off": "turned_off"}
# A long-running state (e.g. a door staying open) kept only its first and latest frame as evidence,
# so a rule's min_duration_sec threshold had no real frame to point at near where it was crossed —
# the witness snapped back to the very first frame, making "left open" look identical to "just
# opened". Freeze a frame roughly this often so there's always one near any reasonable threshold.
STATE_EVIDENCE_INTERVAL_SEC = 1.0
# A person at the cashier's workstation who is neither the cashier nor a client is, by bank policy,
# unauthorised there regardless of who they are (a safe is present; Rules No. 14, p. 8/16.1) — no
# grace period, judged from the moment they arrive, not after some minimum stay.
TRANSACTION_EVENT_TYPES = {"document_presented", "cash_transferred", "cash_counted", "device_used_observed"}


def _overlaps(inner: Detection, outer: Detection, margin: float = 0.03) -> bool:
    x, y = inner.box.center
    return (
        outer.box.x1 - margin <= x <= outer.box.x2 + margin
        and outer.box.y1 - margin <= y <= outer.box.y2 + margin
    )


def _boxes_overlap(a: Detection, b: Detection, margin: float = 0.0) -> bool:
    # True rectangle intersection, unlike _overlaps (center-of-one-box-in-the-other). A seated
    # cashier's box includes her outstretched arm even when her torso center sits elsewhere, so
    # this is the one that actually notices her reaching for the machine.
    return (
        a.box.x1 - margin < b.box.x2
        and a.box.x2 + margin > b.box.x1
        and a.box.y1 - margin < b.box.y2
        and a.box.y2 + margin > b.box.y1
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
    samples = Counter()
    cashier_span: dict[str, list[float]] = {}  # track_id -> [first, last] timestamp seen in cashier_area
    for frame in objects.frames:
        for item in frame.detections:
            if item.label not in {"person", "cashier", "client"}:
                continue
            samples[item.track_id] += 1
            if item.label != "person":
                votes[item.track_id].append(item.label)
            elif len(votes[item.track_id]) < 10:
                # Only where the person stands counts: a client walking in through the door would
                # otherwise collect "unknown" votes on the way and never become a client.
                if person_in_zone(item, profile, "cashier_area"):
                    votes[item.track_id].append("cashier")
                    bounds = cashier_span.setdefault(item.track_id, [frame.timestamp_sec, frame.timestamp_sec])
                    bounds[1] = frame.timestamp_sec
                elif person_in_zone(item, profile, "client_area"):
                    votes[item.track_id].append("client")
    roles = {}
    cashier_votes: dict[str, int] = {}
    for track_id, values in votes.items():
        # A passer-by whose box grazes an area once is not placed there.
        if len(values) < MIN_ROLE_VOTES and len(values) < MIN_ROLE_SHARE * samples[track_id]:
            roles[track_id] = "unknown"
            continue
        role, count = Counter(values).most_common(1)[0]
        roles[track_id] = role if count / len(values) >= 0.7 else "unknown"
        if roles[track_id] == "cashier":
            cashier_votes[track_id] = count
    if len(cashier_votes) > 1:
        # Only one person works the counter at a time — but a tracker that loses and
        # re-identifies the same cashier produces several track ids that are never actually seen
        # in the zone AT THE SAME TIME as one another; those all stay "cashier". Demote a track
        # only when its time in the zone genuinely overlaps the strongest-evidence one — that is
        # the signature of a second, simultaneous person (e.g. a visitor leaning over the desk).
        keep = max(cashier_votes, key=cashier_votes.get)
        kept_start, kept_end = cashier_span[keep]
        for track_id in cashier_votes:
            if track_id == keep:
                continue
            other_start, other_end = cashier_span[track_id]
            if other_start <= kept_end and kept_start <= other_end:
                roles[track_id] = "unknown"
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

    def __init__(
        self,
        required: int,
        area: Box | None,
        exit_area: Box | None,
        min_wait: float = ABSENT_MIN_SEC,
        midzone_wait: float = ABSENT_MIDZONE_SEC,
        hold_margin: float = 0.0,
    ):
        self.required = required
        self.area = area
        self.exit_area = exit_area
        self.min_wait = min_wait
        self.midzone_wait = midzone_wait
        self.hold_margin = hold_margin
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

    def _holds(self, point: tuple[float, float]) -> bool:
        x, y = point
        a, m = self.area, self.hold_margin
        return a.x1 - m <= x <= a.x2 + m and a.y1 - m <= y <= a.y2 + m

    def observe(self, t: float, people: list[Detection], hands: list[Detection], walked_away: bool, alive: bool):
        if self.area is None:
            return None
        inside = [p for p in people if self._holds(p.box.center)]
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
        wait = self.min_wait if self.edge else self.midzone_wait
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
        "cash_retrieved_in_pieces",
        "client_left_counter",
        "client_returned_to_counter",
        "unidentified_person_no_transaction",
    ]
    required = max(1, math.ceil(fps * 0.4))
    smoothers = defaultdict(lambda: StableValue(required))
    areas = {zone.name: zone.box for zone in profile.zones}
    workstation = ZonePresence(required, areas.get("cashier_area"), areas.get("exit_area"))
    # A client sitting down or leaning over the counter can put their box centre just past the edge of
    # the client area; still in view and that close, they have not walked away.
    client_station = ZonePresence(
        required,
        areas.get("client_area"),
        areas.get("exit_area"),
        midzone_wait=CLIENT_ABSENT_MIDZONE_SEC,
        hold_margin=ZONE_EDGE_MARGIN,
    )
    open_states: dict[tuple, State] = {}
    evidence_marked_at: dict[tuple, float] = {}
    active_session: str | None = None
    sessions: dict[str, str] = {}
    service_sessions: set[str] = set()
    batches: dict[str, Batch] = {}
    last_known_side = {}
    last_money: dict[str, tuple] = {}  # last raw observation of each cash track
    counter_cash: dict[str, dict] = {}  # client cash first seen on the counter, until picked up
    document_samples: Counter = Counter()
    locations = {}
    first_location: dict[str, str | None] = {}
    object_states = {}
    emitted_device_contacts = set()
    last_observed = {}
    client_zone_seen: dict[str, float] = {}
    last_client_seen_frame = None  # the actual frame she was last detected in, anywhere
    gap_counting: dict | None = None  # current money-absence stretch near the machine
    last_money_near_machine_time: float | None = None
    fragment_window: dict | None = None  # tracks quick successive track ids right after counting
    first_seen: dict[str, float] = {}
    session_started: dict[str, float] = {}
    session_order: list[str] = []
    counting: dict | None = None  # current stay of cash inside the machine
    unidentified_episode: dict | None = None  # current stretch of a non-cashier, non-client person present
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

    def received_from_client(batch, action, frame, session=None):
        session = session or active_session
        if session:
            batch.session_id = session
            batch.provenance = "observed"
            service_sessions.add(session)
            state("service_active", True, frame, source_events=[action.event_id])

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
            ids = previous.evidence_frame_ids
            if frame.timestamp_sec - evidence_marked_at[key] >= STATE_EVIDENCE_INTERVAL_SEC:
                # Freeze the run up to now as a mark, then start a fresh "latest" pointer after it.
                ids.append(frame.frame_id)
                evidence_marked_at[key] = frame.timestamp_sec
            elif len(ids) > 1:
                ids[-1] = frame.frame_id
            else:
                ids.append(frame.frame_id)
            previous.evidence_frame_ids = list(dict.fromkeys(ids))
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
        evidence_marked_at[key] = frame.timestamp_sec

    def close_unidentified_episode():
        nonlocal unidentified_episode
        if unidentified_episode is None:
            return
        episode = unidentified_episode
        unidentified_episode = None
        # A client's box dipping into the counter for a single sample is jitter, not a visit.
        if episode["clients_only"] and episode["samples"] < required:
            return
        had_transaction = any(
            action.type in TRANSACTION_EVENT_TYPES
            and action.end_sec >= episode["start"]
            and action.start_sec <= episode["end"]
            for action in journal.actions
        )
        if not had_transaction:
            violation = event(
                "unidentified_person_no_transaction",
                episode["last_frame"],
                sorted(episode["actor_ids"]),
                [],
                confidence=0.6,
                start=episode["start"],
            )
            violation.evidence_frame_ids = list(
                dict.fromkeys([episode["frame"].frame_id, episode["last_frame"].frame_id])
            )

    for frame in objects.frames:
        people = [d for d in frame.detections if d.label in {"person", "cashier", "client"}]
        hands = [d for d in frame.detections if d.label == "hand"]
        money_items = [d for d in frame.detections if d.label == "money"]
        for item in frame.detections:
            if item.label in CLIENT_ITEM_LABELS and zone_of(item, profile) == "client_area":
                client_zone_seen[item.track_id] = frame.timestamp_sec
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

        client_ids = [key for key, role in roles.items() if role == "client"]
        client_people_here = [p for p in people if roles.get(p.track_id) == "client"]
        if client_people_here:
            last_client_seen_frame = frame
        client_transition_ids = []
        if presence == "zone":
            client_present_observed = client_station.observe(
                frame.timestamp_sec,
                client_people_here,
                hands,
                walked_away=any(key in visible for key in client_ids),
                alive=bool(frame.detections),
            )
            old_client, stable_client_present = client_station.update(client_present_observed)
            if old_client is True and stable_client_present is False:
                client_transition_ids = [
                    event(
                        "client_left_counter", frame, client_station.ids, start=client_station.empty_since
                    ).event_id
                ]
            elif old_client is False and stable_client_present is True:
                client_transition_ids = [event("client_returned_to_counter", frame, client_station.ids).event_id]
            state("client_present", stable_client_present, frame, source_events=client_transition_ids)
        else:
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
                if label in CLIENT_ITEM_LABELS:
                    hand_owner = roles.get(_contact(item, hands, people) or "")
                    last_client = client_zone_seen.get(item.track_id)
                    recently_client = (
                        last_client is not None and frame.timestamp_sec - last_client <= CLIENT_HANDOFF_GRACE_SEC
                    )
                    if hand_owner == "client" or recently_client:
                        # a client may show a digital ID on the phone (p. 19.1, p. 31.1), or hand
                        # over a paper document the detector mislabels as a personal item
                        continue
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
                fragment_window = {
                    "start": frame.timestamp_sec,
                    "deadline": frame.timestamp_sec + FRAGMENT_WINDOW_SEC,
                    "last_track": None,
                    "last_seen": frame.timestamp_sec,
                    "pieces": set(),
                    "object_ids": set(),
                }
            counting = None

        money_near_machine = any(
            zone_of(m, profile) in {"counter_area", "cashier_area"} or any(_overlaps(m, c, 0.08) for c in counters)
            for m in money_items
        )
        # The box-overlap check above (cashier_near_machine) needs the cashier's box CENTER near
        # the machine; reaching an arm out while seated doesn't move her center, only her box edge.
        cashier_reaching_machine = any(_boxes_overlap(p, c, 0.04) for p in cashier_people for c in counters)
        if money_items:
            if gap_counting is not None:
                gap_duration = frame.timestamp_sec - gap_counting["start"]
                gap_ratio = gap_counting["near"] / gap_counting["samples"] if gap_counting["samples"] else 0.0
                if (
                    money_near_machine
                    and GAP_COUNT_MIN_SEC <= gap_duration <= GAP_COUNT_MAX_SEC
                    and gap_ratio >= GAP_COUNT_NEAR_RATIO
                ):
                    event(
                        "cash_counted",
                        frame,
                        [key for key, role in roles.items() if role == "cashier"],
                        [m.track_id for m in money_items],
                        attributes={"method": "track_gap_near_machine", "inferred": True},
                        confidence=0.45,
                        start=gap_counting["start"],
                        visibility="uncertain",
                    )
                    fragment_window = {
                        "start": frame.timestamp_sec,
                        "deadline": frame.timestamp_sec + FRAGMENT_WINDOW_SEC,
                        "last_track": None,
                        "last_seen": frame.timestamp_sec,
                        "pieces": set(),
                        "object_ids": set(),
                    }
                gap_counting = None
            if money_near_machine:
                last_money_near_machine_time = frame.timestamp_sec
        elif last_money_near_machine_time is not None and counting is None:
            if gap_counting is None:
                gap_counting = {"start": last_money_near_machine_time, "samples": 0, "near": 0}
            gap_counting["samples"] += 1
            gap_counting["near"] += int(cashier_reaching_machine)
            if frame.timestamp_sec - gap_counting["start"] > GAP_COUNT_MAX_SEC:
                # Nothing came back near the machine within the window that would still read as a
                # plausible count — give up instead of leaving counting_in_progress stuck True for
                # the rest of the video on money that simply left the frame (put away, out of shot).
                gap_counting = None
                last_money_near_machine_time = None

        # True from the moment cash is first seen in the machine (or vanishes near it) until the
        # episode closes — a client leaving while this is still open skips the recount they're
        # entitled to witness (Rules No. 231, p. 20, p. 31.4).
        state("counting_in_progress", True if (counting is not None or gap_counting is not None) else None, frame)

        others = [p for p in people if roles.get(p.track_id) != "cashier"]
        state("extra_person_present", True if len(others) >= 2 else None, frame)

        # A "client"-voted track is not exempt here: the zone vote only says where someone stood
        # most of the time, not that every entry into the cashier's own restricted area was a
        # legitimate handover — close_unidentified_episode() still clears them the moment an
        # actual transaction event overlaps their presence, so a real client reaching across the
        # counter is unaffected.
        unidentified_at_workstation = [p for p in others if zone_of(p, profile) in WORKSTATION_ZONES]
        if unidentified_at_workstation:
            if unidentified_episode is None:
                unidentified_episode = {
                    "start": frame.timestamp_sec,
                    "frame": frame,
                    "actor_ids": set(),
                    "samples": 0,
                    "clients_only": True,
                }
            unidentified_episode["end"], unidentified_episode["last_frame"] = frame.timestamp_sec, frame
            unidentified_episode["samples"] += 1
            if any(roles.get(p.track_id) != "client" for p in unidentified_at_workstation):
                unidentified_episode["clients_only"] = False
            unidentified_episode["actor_ids"].update(p.track_id for p in unidentified_at_workstation)
        else:
            close_unidentified_episode()

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
                # not to "whoever's hand is on it" — session-scoped, not person-scoped. A single-sample
                # "document" is usually a misread stack or paper, and would excuse a missing ID check;
                # a real one may flicker, so its samples need not be consecutive.
                document_samples[item.track_id] += 1
                shown = document_samples[item.track_id] >= DOCUMENT_MIN_SAMPLES
                nearest_session = _nearest_session(item.box.center, clients, sessions, active_session)
                contact_key = (item.track_id, nearest_session)
                if shown and nearest_session and contact_key not in emitted_device_contacts:
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
            if (
                stable_side
                and former
                and stable_side != former[0]
                and frame.timestamp_sec - former[1] <= HANDOFF_WINDOW_SEC
            ):
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
                if from_role == "client":
                    received_from_client(batch, action, frame)
            if item.track_id not in first_location:
                first_location[item.track_id] = location
                # Cash first seen inside the counting machine is being counted, not put down by the client.
                if location == "counter_area" and clients and not any(_overlaps(item, c) for c in counters):
                    counter_cash[item.track_id] = {
                        "frame": frame,
                        "last_sec": frame.timestamp_sec,
                        "session": _nearest_session(item.box.center, clients, sessions, active_session),
                    }
            pending = counter_cash.get(item.track_id)
            if pending and location == "counter_area":
                pending["last_sec"] = frame.timestamp_sec
            elif pending and location == "cashier_area":
                del counter_cash[item.track_id]
                # The client who put it down must still be the one served: cash left from an earlier
                # client and picked up later is not a handover from the next one.
                if (
                    frame.timestamp_sec - pending["last_sec"] <= HANDOFF_WINDOW_SEC
                    and pending["session"] == active_session
                ):
                    action = event(
                        "cash_transferred",
                        frame,
                        [],
                        [item.track_id],
                        {"from_role": "client", "to_role": "cashier", "inferred_from": "counter_pickup"},
                        confidence=min(item.confidence, 0.6),
                        start=pending["frame"].timestamp_sec,
                        visibility="uncertain",
                        session=pending["session"],
                    )
                    action.evidence_frame_ids = list(dict.fromkeys([pending["frame"].frame_id, frame.frame_id]))
                    received_from_client(batch, action, frame, pending["session"])
            if stable_side:
                last_known_side[item.track_id] = (stable_side, frame.timestamp_sec, frame.frame_id)
            last_money[item.track_id] = (side, frame, item, clients)
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
            if fragment_window is not None:
                # repair_tracks() stitches a money track broken by a short gap back into one id —
                # exactly the case this is trying to catch. raw_track_id still names the original,
                # pre-stitch piece, so fragmentation is judged on that identity, not the repaired one.
                piece_id = item.attributes.get("raw_track_id", item.track_id)
                if frame.timestamp_sec > fragment_window["deadline"]:
                    fragment_window = None
                elif location in {"counter_area", "cashier_area"}:
                    if piece_id != fragment_window["last_track"]:
                        gap = frame.timestamp_sec - fragment_window["last_seen"]
                        if gap <= FRAGMENT_GAP_MAX_SEC:
                            fragment_window["pieces"].add(piece_id)
                            fragment_window["object_ids"].add(piece_id)
                            if len(fragment_window["pieces"]) >= FRAGMENT_MIN_PIECES:
                                event(
                                    "cash_retrieved_in_pieces",
                                    frame,
                                    [key for key, role in roles.items() if role == "cashier"],
                                    sorted(fragment_window["object_ids"]),
                                    attributes={"method": "track_fragmentation"},
                                    confidence=0.4,
                                    start=fragment_window["start"],
                                    visibility="uncertain",
                                )
                                fragment_window = None
                        else:
                            fragment_window = None
                    if fragment_window is not None:
                        fragment_window["object_ids"].add(piece_id)
                        fragment_window["last_track"] = piece_id
                        fragment_window["last_seen"] = frame.timestamp_sec

    # A stack handed over the counter is often seen on the client's side for a single sample before the
    # client's hand hides it, so the smoothed side never flips. The track ending there is the handover.
    for track_id, (side, frame, item, clients) in last_money.items():
        former = last_known_side.get(track_id)
        if side != "client" or not former or former[0] != "cashier":
            continue
        if frame.timestamp_sec - former[1] > HANDOFF_WINDOW_SEC:
            continue
        action = event(
            "cash_transferred",
            frame,
            [],
            [track_id],
            {"from_role": "cashier", "to_role": "client", "inferred_from": "track_end"},
            confidence=min(item.confidence, 0.6),
            start=former[1],
            visibility="uncertain",
            session=_nearest_session(item.box.center, clients, sessions, active_session),
        )
        action.evidence_frame_ids = list(dict.fromkeys([former[2], frame.frame_id]))

    if objects.frames and last_client_seen_frame is not None:
        last_frame = objects.frames[-1]
        tail_gap = last_frame.timestamp_sec - last_client_seen_frame.timestamp_sec
        cut_off_near_the_end = last_client_seen_frame.timestamp_sec >= last_frame.timestamp_sec - TAIL_WINDOW_SEC
        still_counting = counting is not None or gap_counting is not None
        if tail_gap >= TAIL_UNTRACKED_MIN_SEC and cut_off_near_the_end and still_counting:
            # Anchored at her last sighting (not the final frame) so the overlap engine's anchor
            # match lines up with where the client_present/counting_in_progress overlap begins,
            # then stretched to the last frame so the overlap has width. Built directly rather
            # than through state() / open_states, which assume frames keep arriving in order —
            # this runs once, after the loop, with a frame earlier than the one it last saw.
            transition = event(
                "client_left_counter",
                last_client_seen_frame,
                [],
                attributes={"inferred_from": "untracked_at_clip_end"},
                confidence=0.5,
                visibility="uncertain",
            )
            journal.states.append(
                State(
                    state_id=f"state_{len(journal.states) + 1:06d}",
                    name="client_present",
                    value=False,
                    start_sec=last_client_seen_frame.timestamp_sec,
                    end_sec=last_frame.timestamp_sec,
                    entity_ids=[],
                    session_id=active_session,
                    evidence_frame_ids=list(
                        dict.fromkeys([last_client_seen_frame.frame_id, last_frame.frame_id])
                    ),
                    source_event_ids=[transition.event_id],
                    visibility="clear",
                )
            )

    # After the end-of-clip transaction inference above, not before it — an episode that looks
    # transaction-free mid-loop can still be cleared by a handover only recognized once the track
    # ends with the clip.
    close_unidentified_episode()

    journal.batches = list(batches.values())
    journal.actions.sort(key=lambda a: (a.start_sec, a.end_sec, a.event_id))
    journal.warnings.append(
        "Передача денег, пересчёт на машинке и показ документа определяются по зонам и сессии "
        "клиента, без привязки к конкретной руке — на видео с несколькими клиентами одновременно "
        "в кадре привязка может быть неточной. Если пачка денег не видна во время передачи из рук "
        "в руки, передача может быть не засчитана. Личность людей и то, заперт ли сейф на замок, "
        "сервис не определяет."
    )
    return journal
