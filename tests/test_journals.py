import json

from examples.make_demo import create_demo
from forte_guard.contracts import ObjectJournal
from forte_guard.journal.builder import StableValue, assign_roles, build_journal
from forte_guard.journal.tracks import repair_tracks
from forte_guard.reasoning.catalog import load_catalog
from forte_guard.reasoning.rules import rule_candidates


def test_tracks_become_transfer_departure_and_state_overlap(demo, settings):
    _, path, profile = demo
    objects = ObjectJournal.model_validate_json(path.read_text())
    journal = build_journal(objects, profile, 8, 5)
    transfers = [event for event in journal.actions if event.type == "cash_transferred"]
    departures = [event for event in journal.actions if event.type == "cashier_left_workstation"]
    assert len(transfers) == 1 and transfers[0].object_ids == ["batch1"]
    assert transfers[0].attributes["from_role"] == "client"
    assert transfers[0].attributes["to_role"] == "cashier"
    assert len(departures) == 1 and 3 <= departures[0].start_sec <= 3.4
    assert journal.batches[0].session_id is not None
    rules, scenarios = load_catalog(settings.config_dir)
    candidates = rule_candidates(journal, rules, scenarios)
    assert len(candidates) == 1
    assert candidates[0].rule_id == "cashier_left_during_service"
    # min_duration_sec (1.0 for this rule): the anchor sits where the overlap became a violation,
    # not where the departure itself started.
    assert candidates[0].timecode_sec == departures[0].start_sec + 1.0


def test_detector_loss_is_unknown_not_observed_departure(tmp_path, settings):
    _, path, profile = create_demo(tmp_path / "occlusion", disappear=True)
    objects = ObjectJournal.model_validate_json(path.read_text())
    journal = build_journal(objects, profile, 8, 5)
    assert not any(event.type == "cashier_left_workstation" for event in journal.actions)
    assert not any(state.name == "cashier_present" and state.value is False for state in journal.states)
    assert any(state.name == "cashier_present" and state.value is None for state in journal.states)
    rules, scenarios = load_catalog(settings.config_dir)
    assert rule_candidates(journal, rules, scenarios) == []


def test_short_flicker_does_not_become_a_departure(demo):
    _, path, profile = demo
    raw = json.loads(path.read_text())
    for item in raw["frames"][7]["detections"]:
        if item["track_id"] == "cashier":
            item["box"] = {"x1": 0.43, "y1": 0.72, "x2": 0.57, "y2": 0.97}
    journal = build_journal(ObjectJournal.model_validate(raw), profile, 8, 5)
    departures = [event for event in journal.actions if event.type == "cashier_left_workstation"]
    assert len(departures) == 1 and departures[0].start_sec >= 3


def test_unknown_is_not_smoothed_into_false_certainty():
    state = StableValue(2)
    state.update(True)
    assert state.update(True)[1] is True
    assert state.update(None)[1] is None


def test_closed_door_does_not_prove_locked(demo):
    _, path, profile = demo
    journal = build_journal(ObjectJournal.model_validate_json(path.read_text()), profile, 8, 5)
    assert "safe_locked" in journal.unsupported_actions
    assert not any(event.type == "safe_locked" for event in journal.actions)


def _scene(frames):
    from forte_guard.contracts import Box, Detection, ObjectFrame, Zone, ZoneProfile

    def det(track_id, label, box):
        return Detection(
            track_id=track_id, label=label, confidence=0.9, box=Box(x1=box[0], y1=box[1], x2=box[2], y2=box[3])
        )

    objects = ObjectJournal(
        backend="synthetic-replay",
        supported_classes=["person", "hand"],
        frames=[
            ObjectFrame(
                frame_id=f"f{i:06d}",
                timestamp_sec=i / 5,
                detections=[det(*item) for item in items],
            )
            for i, items in enumerate(frames)
        ],
    )
    profile = ZoneProfile(
        source="synthetic",
        zones=[
            Zone(name="cashier_area", box=Box(x1=0, y1=0, x2=0.4, y2=1)),
            Zone(name="client_area", box=Box(x1=0.6, y1=0, x2=1, y2=1)),
        ],
    )
    return objects, profile


SEATED = [0.05, 0.1, 0.35, 0.8]
AT_EDGE = [0.3, 0.1, 0.42, 0.8]
AWAY = [0.43, 0.72, 0.57, 0.97]
CLIENT = ("client", "person", [0.65, 0.1, 0.95, 0.8])


def _departures(journal):
    return [event for event in journal.actions if event.type == "cashier_left_workstation"]


def test_zone_presence_survives_a_fragmented_cashier_track():
    # The tracker loses the cashier and re-identifies her as c5; she then walks away in view.
    frames = [[CLIENT, ("c1" if i < 10 else "c5", "person", SEATED if i < 20 else AWAY)] for i in range(40)]
    objects, profile = _scene(frames)
    assert _departures(build_journal(objects, profile, 8, 5)) == []
    departures = _departures(build_journal(objects, profile, 8, 5, presence="zone"))
    assert len(departures) == 1 and departures[0].start_sec == 4.0


def test_track_repair_lets_track_presence_follow_a_fragmented_cashier():
    frames = [[CLIENT, ("c1" if i < 10 else "c5", "person", SEATED if i < 20 else AWAY)] for i in range(40)]
    objects, profile = _scene(frames)
    departures = _departures(build_journal(repair_tracks(objects), profile, 8, 5))
    assert len(departures) == 1 and departures[0].actor_ids == ["c1"]


def test_zone_presence_reports_leaving_through_the_edge_after_the_wait():
    frames = [[CLIENT, *([("c1", "person", SEATED if i < 5 else AT_EDGE)] if i < 10 else [])] for i in range(50)]
    objects, profile = _scene(frames)
    journal = build_journal(objects, profile, 10, 5, presence="zone")
    departures = _departures(journal)
    assert len(departures) == 1 and departures[0].start_sec == 2.0
    assert departures[0].end_sec >= 1.8 + 3.0
    assert any(state.name == "cashier_present" and state.value is False for state in journal.states)


def test_zone_presence_waits_longer_when_the_cashier_vanishes_mid_area():
    frames = [[CLIENT, *([("c1", "person", SEATED)] if i < 10 else [])] for i in range(45)]
    short, profile = _scene(frames[:30])
    assert _departures(build_journal(short, profile, 6, 5, presence="zone")) == []
    long, profile = _scene(frames)
    departures = _departures(build_journal(long, profile, 9, 5, presence="zone"))
    assert len(departures) == 1 and departures[0].start_sec == 2.0


def test_zone_presence_stays_unknown_when_nothing_is_detected():
    frames = [[CLIENT, ("c1", "person", SEATED)] if i < 10 else [] for i in range(60)]
    objects, profile = _scene(frames)
    journal = build_journal(objects, profile, 12, 5, presence="zone")
    assert _departures(journal) == []
    assert not any(state.name == "cashier_present" and state.value is False for state in journal.states)


def test_zone_presence_counts_a_hand_at_the_workstation():
    hand = ("h1", "hand", [0.2, 0.5, 0.24, 0.54])
    frames = [[CLIENT, hand, *([("c1", "person", SEATED)] if i < 10 else [])] for i in range(60)]
    objects, profile = _scene(frames)
    journal = build_journal(objects, profile, 12, 5, presence="zone")
    assert _departures(journal) == []
    assert not any(state.name == "cashier_present" and state.value is False for state in journal.states)


def test_zone_presence_reports_the_return():
    frames = [[CLIENT, *([("c1", "person", SEATED)] if i < 10 or i >= 45 else [])] for i in range(55)]
    objects, profile = _scene(frames)
    journal = build_journal(objects, profile, 11, 5, presence="zone")
    assert len(_departures(journal)) == 1
    assert [event.type for event in journal.actions if event.type == "cashier_returned"] == ["cashier_returned"]


def test_zone_presence_keeps_demo_behaviour(demo, tmp_path, settings):
    _, path, profile = demo
    journal = build_journal(ObjectJournal.model_validate_json(path.read_text()), profile, 8, 5, presence="zone")
    departures = _departures(journal)
    assert len(departures) == 1 and 3 <= departures[0].start_sec <= 3.4
    _, lost, lost_profile = create_demo(tmp_path / "occlusion", disappear=True)
    journal = build_journal(ObjectJournal.model_validate_json(lost.read_text()), lost_profile, 8, 5, presence="zone")
    assert _departures(journal) == []


def test_two_simultaneous_clients_attribute_documents_to_the_right_session():
    # No hand contact is available (no "hand" class on detector v1), so document_presented is
    # session-scoped by nearest client instead. With a queue (two clients in client_area at once),
    # each document must still land on the session of the client it's actually next to.
    client_a = ("ca", "person", [0.62, 0.1, 0.72, 0.8])
    client_b = ("cb", "person", [0.85, 0.1, 0.95, 0.8])
    doc_a = ("doc_a", "document", [0.63, 0.3, 0.68, 0.34])
    doc_b = ("doc_b", "document", [0.86, 0.3, 0.91, 0.34])
    frames = [[client_a] if i < 10 else [client_a, client_b, doc_a, doc_b] for i in range(20)]
    objects, profile = _scene(frames)
    journal = build_journal(objects, profile, 4, 5)
    presented = {
        event.object_ids[0]: event.session_id for event in journal.actions if event.type == "document_presented"
    }
    assert presented.get("doc_a") is not None and presented.get("doc_b") is not None
    assert presented["doc_a"] != presented["doc_b"]


def _workstation_states(journal, label):
    return [state for state in journal.states if state.name == f"{label}_at_workstation"]


def test_personal_item_handed_across_counter_is_not_flagged_as_left_behind():
    # The detector sometimes mislabels a client's ID card as "personal_item". It should not be
    # reported as a personal item left at the workstation right after crossing from client_area.
    in_client_area = [0.7, 0.3, 0.75, 0.35]
    in_cashier_area = [0.1, 0.3, 0.15, 0.35]
    frames = [[CLIENT, ("p1", "personal_item", in_client_area if i < 10 else in_cashier_area)] for i in range(25)]
    objects, profile = _scene(frames)
    journal = build_journal(objects, profile, 5, 5)
    assert not any(state.value is True for state in _workstation_states(journal, "personal_item"))


def test_personal_item_left_at_workstation_without_a_handoff_is_still_flagged():
    # A jacket sitting at the cashier's own area the whole time, never seen on the client side,
    # is a genuine violation and must still be reported.
    in_cashier_area = [0.1, 0.3, 0.15, 0.35]
    frames = [[CLIENT, ("p1", "personal_item", in_cashier_area)] for _ in range(25)]
    objects, profile = _scene(frames)
    journal = build_journal(objects, profile, 5, 5)
    assert any(state.value is True for state in _workstation_states(journal, "personal_item"))


COUNTER = ("ctr", "counter", [0.3, 0.1, 0.5, 0.3])
# A seated cashier reaching her arm toward the machine: her box edge overlaps the counter box even
# though her box CENTER (torso) stays well to the left of it.
CASHIER_REACHING = ("cashier", "person", [0.05, 0.15, 0.32, 0.5])
# Inside cashier_area, but clearly clear of the counter box — isolates the gap/fragmentation logic
# from the pre-existing box-overlap counting method, which keys off money overlapping the counter.
MONEY_NEAR_COUNTER = [0.15, 0.15, 0.20, 0.25]


def _cash_counted(journal):
    return [event for event in journal.actions if event.type == "cash_counted"]


def _fragmented(journal):
    return [event for event in journal.actions if event.type == "cash_retrieved_in_pieces"]


def test_money_lost_near_the_machine_infers_a_count_without_box_overlap():
    # The money box never overlaps the counter box (occlusion at the slot when fed in by hand), but
    # it vanishes for a long stretch while the cashier keeps reaching for the machine — infer a
    # counting episode from the gap itself, alongside (not instead of) the box-overlap method.
    frames = (
        [[CLIENT, CASHIER_REACHING, COUNTER, ("m1", "money", MONEY_NEAR_COUNTER)] for _ in range(3)]
        + [[CLIENT, CASHIER_REACHING, COUNTER] for _ in range(20)]
        + [[CLIENT, CASHIER_REACHING, COUNTER, ("m2", "money", MONEY_NEAR_COUNTER)] for _ in range(5)]
    )
    objects, profile = _scene(frames)
    journal = build_journal(objects, profile, 6, 5)
    counted = _cash_counted(journal)
    assert len(counted) == 1 and counted[0].attributes["method"] == "track_gap_near_machine"
    assert _fragmented(journal) == []


def test_two_pieces_in_quick_succession_after_counting_is_flagged():
    # First piece settles, then a second, differently-tracked piece shows up with only a short gap
    # before merging in — e.g. a note sorted out separately then put back with the rest.
    frames = (
        [[CLIENT, CASHIER_REACHING, COUNTER, ("m1", "money", MONEY_NEAR_COUNTER)] for _ in range(3)]
        + [[CLIENT, CASHIER_REACHING, COUNTER] for _ in range(20)]
        + [[CLIENT, CASHIER_REACHING, COUNTER, ("m2", "money", MONEY_NEAR_COUNTER)] for _ in range(5)]
        + [[CLIENT, CASHIER_REACHING, COUNTER] for _ in range(2)]
        + [[CLIENT, CASHIER_REACHING, COUNTER, ("m3", "money", MONEY_NEAR_COUNTER)] for _ in range(5)]
    )
    objects, profile = _scene(frames)
    journal = build_journal(objects, profile, 7, 5)
    fragmented = _fragmented(journal)
    assert len(fragmented) == 1
    assert set(fragmented[0].object_ids) == {"m2", "m3"}


def test_a_long_gap_between_pieces_is_not_treated_as_a_split():
    # A gap long enough to be its own counting episode (not a quick hand-off) must not count as a
    # split — only short gaps right after counting are the suspicious pattern.
    frames = (
        [[CLIENT, CASHIER_REACHING, COUNTER, ("m1", "money", MONEY_NEAR_COUNTER)] for _ in range(3)]
        + [[CLIENT, CASHIER_REACHING, COUNTER] for _ in range(20)]
        + [[CLIENT, CASHIER_REACHING, COUNTER, ("m2", "money", MONEY_NEAR_COUNTER)] for _ in range(5)]
        + [[CLIENT, CASHIER_REACHING, COUNTER] for _ in range(15)]
        + [[CLIENT, CASHIER_REACHING, COUNTER, ("m3", "money", MONEY_NEAR_COUNTER)] for _ in range(5)]
    )
    objects, profile = _scene(frames)
    journal = build_journal(objects, profile, 9, 5)
    assert _fragmented(journal) == []


CASHIER = ("cashier", "person", SEATED)
STACK_AT_CASHIER = ("stack", "money", [0.28, 0.45, 0.33, 0.5])
STACK_AT_CLIENT = ("stack", "money", [0.66, 0.45, 0.71, 0.5])


def _transfers(journal):
    return [event for event in journal.actions if event.type == "cash_transferred"]


def test_stack_last_seen_on_the_client_side_is_a_handover(settings):
    # Seen once in the client's hand, then hidden by it: the smoothed side never flips.
    frames = [[CASHIER, CLIENT, STACK_AT_CASHIER] for _ in range(10)] + [[CASHIER, CLIENT, STACK_AT_CLIENT]]
    frames += [[CASHIER, CLIENT] for _ in range(10)]
    objects, profile = _scene(frames)
    journal = build_journal(objects, profile, 5, 5)
    transfers = _transfers(journal)
    assert len(transfers) == 1 and transfers[0].end_sec == 2.0
    assert transfers[0].attributes["to_role"] == "client"
    assert transfers[0].attributes["inferred_from"] == "track_end"
    rules, scenarios = load_catalog(settings.config_dir)
    found = [c for c in rule_candidates(journal, rules, scenarios) if c.rule_id == "payout_without_observed_count"]
    assert len(found) == 1 and found[0].timecode_sec == 2.0


def test_one_frame_flicker_to_the_client_side_is_not_a_handover():
    frames = [[CASHIER, CLIENT, STACK_AT_CASHIER] for _ in range(10)] + [[CASHIER, CLIENT, STACK_AT_CLIENT]]
    frames += [[CASHIER, CLIENT, STACK_AT_CASHIER] for _ in range(10)]
    objects, profile = _scene(frames)
    assert _transfers(build_journal(objects, profile, 5, 5)) == []


def test_stack_reappearing_at_the_client_long_after_is_not_a_handover():
    frames = [[CASHIER, CLIENT, STACK_AT_CASHIER] for _ in range(5)] + [[CASHIER, CLIENT] for _ in range(25)]
    frames += [[CASHIER, CLIENT, STACK_AT_CLIENT]]
    objects, profile = _scene(frames)
    assert _transfers(build_journal(objects, profile, 7, 5)) == []


STACK_ON_COUNTER = ("stack", "money", [0.48, 0.45, 0.53, 0.5])


def _with_counter(frames):
    from forte_guard.contracts import Box, Zone

    objects, profile = _scene(frames)
    profile.zones.append(Zone(name="counter_area", box=Box(x1=0.4, y1=0.4, x2=0.6, y2=0.6)))
    return objects, profile


def _id_check_candidates(journal, settings):
    rules, scenarios = load_catalog(settings.config_dir)
    return [c for c in rule_candidates(journal, rules, scenarios) if c.rule_id == "cash_taken_without_id_check"]


def _counter_then_cashier(gap=2, extra=()):
    frames = [[CASHIER, CLIENT] for _ in range(10)] + [[CASHIER, CLIENT, STACK_ON_COUNTER, *extra] for _ in range(5)]
    frames += [[CASHIER, CLIENT] for _ in range(gap)] + [[CASHIER, CLIENT, STACK_AT_CASHIER] for _ in range(5)]
    return frames


def test_client_cash_picked_up_from_the_counter_is_a_handover(settings):
    # Put down from the client's side, the stack is first seen lying on the counter, then at the cashier.
    objects, profile = _with_counter(_counter_then_cashier())
    journal = build_journal(objects, profile, 5, 5)
    transfers = _transfers(journal)
    assert len(transfers) == 1 and (transfers[0].start_sec, transfers[0].end_sec) == (2.0, 3.4)
    assert transfers[0].attributes == {"from_role": "client", "to_role": "cashier", "inferred_from": "counter_pickup"}
    found = _id_check_candidates(journal, settings)
    assert len(found) == 1 and found[0].timecode_sec == 3.4


def test_counter_cash_is_not_a_handover_without_a_client_late_or_from_the_cashier():
    alone = [[CASHIER, STACK_ON_COUNTER] for _ in range(5)] + [[CASHIER, STACK_AT_CASHIER] for _ in range(5)]
    from_cashier = [[CASHIER, CLIENT, STACK_AT_CASHIER] for _ in range(5)] + _counter_then_cashier()[10:]
    for frames in (alone, from_cashier, _counter_then_cashier(gap=25)):
        objects, profile = _with_counter(frames)
        assert _transfers(build_journal(objects, profile, 10, 5)) == []


def test_a_document_seen_for_one_sample_does_not_excuse_the_id_check(settings):
    flicker = [("doc", "document", [0.7, 0.45, 0.75, 0.5])]
    frames = _counter_then_cashier()
    frames[11] = [*frames[11], *flicker]
    objects, profile = _with_counter(frames)
    journal = build_journal(objects, profile, 5, 5)
    assert not any(event.type == "document_presented" for event in journal.actions)
    assert len(_id_check_candidates(journal, settings)) == 1
    objects, profile = _with_counter(_counter_then_cashier(extra=flicker))
    assert _id_check_candidates(build_journal(objects, profile, 5, 5), settings) == []


def test_a_client_walking_in_through_the_door_is_still_a_client():
    # The first samples are on the way in, outside both areas; the role comes from where they stand.
    walking_in = ("visitor", "person", [0.42, 0.3, 0.58, 0.6])
    standing = ("visitor", "person", [0.65, 0.1, 0.95, 0.8])
    frames = [[CASHIER, walking_in] for _ in range(4)] + [[CASHIER, standing] for _ in range(16)]
    objects, profile = _scene(frames)
    journal = build_journal(objects, profile, 4, 5)
    assert journal.roles["visitor"] == "client"
    assert [event.actor_ids for event in journal.actions if event.type == "client_observed"] == [["visitor"]]


def test_a_person_who_never_stands_in_either_area_stays_unknown():
    frames = [[CASHIER, ("passerby", "person", [0.42, 0.3, 0.58, 0.6])] for _ in range(10)]
    objects, profile = _scene(frames)
    assert build_journal(objects, profile, 2, 5).roles["passerby"] == "unknown"


def test_cash_left_by_an_earlier_client_is_not_a_handover_from_the_next(settings):
    # The stack lies on the counter while client A is there, A leaves, B arrives, then the cashier
    # takes the stack: B never handed it over.
    client_b = ("client_b", "person", [0.65, 0.1, 0.95, 0.8])
    frames = [[CASHIER, CLIENT, STACK_ON_COUNTER] for _ in range(10)] + [[CASHIER, STACK_ON_COUNTER] for _ in range(20)]
    frames += [[CASHIER, client_b, STACK_ON_COUNTER] for _ in range(5)] + [[CASHIER, client_b, STACK_AT_CASHIER] for _ in range(5)]
    objects, profile = _with_counter(frames)
    journal = build_journal(objects, profile, 8, 5)
    assert _transfers(journal) == []
    assert _id_check_candidates(journal, settings) == []


def test_cash_first_seen_in_the_counting_machine_is_not_a_handover():
    machine = ("machine", "counter", [0.44, 0.42, 0.58, 0.56])
    frames = [[CASHIER, CLIENT, machine] for _ in range(10)] + [[CASHIER, CLIENT, machine, STACK_ON_COUNTER] for _ in range(5)]
    frames += [[CASHIER, CLIENT, machine] for _ in range(2)] + [[CASHIER, CLIENT, machine, STACK_AT_CASHIER] for _ in range(5)]
    objects, profile = _with_counter(frames)
    assert _transfers(build_journal(objects, profile, 5, 5)) == []


def test_a_flickering_document_still_counts_as_shown(settings):
    # Seen on every other sample: real, just flickering. It excuses the cash pickup.
    document = ("doc", "document", [0.7, 0.45, 0.75, 0.5])
    frames = _counter_then_cashier()
    for index in (2, 4, 6, 8):
        frames[index] = [*frames[index], document]
    objects, profile = _with_counter(frames)
    journal = build_journal(objects, profile, 5, 5)
    assert any(event.type == "document_presented" for event in journal.actions)
    assert _id_check_candidates(journal, settings) == []


def test_a_passer_by_grazing_the_cashier_area_once_is_not_a_cashier():
    grazing = ("passerby", "person", [0.3, 0.3, 0.45, 0.6])  # centre inside the cashier area
    outside = ("passerby", "person", [0.42, 0.3, 0.58, 0.6])
    frames = [[CASHIER, grazing]] + [[CASHIER, outside] for _ in range(30)]
    objects, profile = _scene(frames)
    assert build_journal(objects, profile, 6, 5).roles["passerby"] == "unknown"


MONEY_IN_COUNTER = [0.35, 0.15, 0.40, 0.20]


def _client_departures(journal):
    return [event for event in journal.actions if event.type == "client_left_counter"]


def test_client_leaves_before_recount_finishes_is_flagged(settings):
    # The client is seen at the counter long enough for her role to lock in, then steps away while
    # the cash is still sitting in the counting machine (it never leaves, so counting never closes).
    frames = (
        [[CLIENT, COUNTER, ("m1", "money", MONEY_IN_COUNTER)] for _ in range(12)]
        + [[("client", "person", AWAY), COUNTER, ("m1", "money", MONEY_IN_COUNTER)] for _ in range(10)]
    )
    objects, profile = _scene(frames)
    journal = build_journal(objects, profile, 5, 5, presence="zone")
    assert len(_client_departures(journal)) == 1
    assert any(s.name == "counting_in_progress" and s.value is True for s in journal.states)
    rules, scenarios = load_catalog(settings.config_dir)
    candidates = rule_candidates(journal, rules, scenarios)
    assert any(c.rule_id == "client_left_before_count_complete" for c in candidates)


def test_client_staying_through_the_recount_is_not_flagged(settings):
    frames = [[CLIENT, COUNTER, ("m1", "money", MONEY_IN_COUNTER)] for _ in range(20)]
    objects, profile = _scene(frames)
    journal = build_journal(objects, profile, 4, 5, presence="zone")
    assert _client_departures(journal) == []
    rules, scenarios = load_catalog(settings.config_dir)
    candidates = rule_candidates(journal, rules, scenarios)
    assert not any(c.rule_id == "client_left_before_count_complete" for c in candidates)


def test_client_vanishing_mid_area_confirms_after_the_shorter_client_wait():
    # A client vanishing mid-area (not visibly walking off) is still a deliberate wait, just
    # shorter than the cashier's own tuned 6s — missing a witnessed recount matters more. A brief
    # dip that is NOT the end of the clip (she is seen again well before it ends) is neither a
    # confirmed wait-based departure nor a clip-cut-off case — ordinary flicker tolerance.
    present = [[CLIENT, COUNTER, ("m1", "money", MONEY_IN_COUNTER)] for _ in range(12)]
    absent = [[COUNTER, ("m1", "money", MONEY_IN_COUNTER)] for _ in range(30)]
    brief_dip, profile = _scene(present + absent[:10] + present[:20])
    assert _client_departures(build_journal(brief_dip, profile, 8, 5, presence="zone")) == []
    full, profile = _scene(present + absent)
    departures = _client_departures(build_journal(full, profile, 8, 5, presence="zone"))
    assert len(departures) == 1 and departures[0].start_sec == 2.4


def test_client_untracked_right_as_the_clip_cuts_off_is_flagged(settings):
    # No wait can run its course when the clip itself simply ends — there is no more footage left
    # to prove she came back. Only counts when the gap is a real stretch (not one dropped sample)
    # AND starts close to the very last frame (looks cut off, not a long unexplained absence).
    present = [[CLIENT, COUNTER, ("m1", "money", MONEY_IN_COUNTER)] for _ in range(12)]
    cut_off, profile = _scene(present + [[COUNTER, ("m1", "money", MONEY_IN_COUNTER)]] * 4)
    journal = build_journal(cut_off, profile, 4, 5, presence="zone")
    departures = _client_departures(journal)
    assert len(departures) == 1
    assert departures[0].attributes["inferred_from"] == "untracked_at_clip_end"
    # The synthetic state built for this case must still overlap counting_in_progress for the
    # existing rule to actually fire on it, not just leave a dangling, unused event behind.
    rules, scenarios = load_catalog(settings.config_dir)
    candidates = rule_candidates(journal, rules, scenarios)
    assert any(c.rule_id == "client_left_before_count_complete" for c in candidates)


def test_a_single_dropped_sample_right_at_the_clip_end_is_not_flagged():
    present = [[CLIENT, COUNTER, ("m1", "money", MONEY_IN_COUNTER)] for _ in range(12)]
    almost_full, profile = _scene([*present, [COUNTER, ("m1", "money", MONEY_IN_COUNTER)]])
    assert _client_departures(build_journal(almost_full, profile, 3, 5, presence="zone")) == []


def test_a_mid_length_absence_well_short_of_either_threshold_is_not_flagged():
    # 3.8s of absence: short of the cashier-style 5s wait, and her last sighting sits more than 2s
    # before the clip ends, so it does not look like the clip was cut off mid-departure either.
    present = [[CLIENT, COUNTER, ("m1", "money", MONEY_IN_COUNTER)] for _ in range(12)]
    absent = [[COUNTER, ("m1", "money", MONEY_IN_COUNTER)] for _ in range(19)]
    objects, profile = _scene(present + absent)
    assert _client_departures(build_journal(objects, profile, 8, 5, presence="zone")) == []


def test_counting_in_progress_gives_up_after_the_gap_cap_so_a_later_departure_is_not_flagged(settings):
    # Money seen once near the counter, then never again — put away normally, out of camera view.
    # counting_in_progress must not stay stuck True for the rest of the video just because nothing
    # ever came back to close it; a client who leaves much later must not be flagged on its account.
    seen_once = [[CLIENT, COUNTER, ("m1", "money", MONEY_IN_COUNTER)] for _ in range(12)]
    money_gone = [[CLIENT, COUNTER] for _ in range(110)]  # 22s with no money anywhere
    client_leaves_at_the_end = [[COUNTER] for _ in range(5)]  # 1s, client untracked at clip end
    objects, profile = _scene(seen_once + money_gone + client_leaves_at_the_end)
    journal = build_journal(objects, profile, 30, 5, presence="zone")
    assert not any(s.name == "counting_in_progress" and s.value is True and s.end_sec > 23 for s in journal.states)
    rules, scenarios = load_catalog(settings.config_dir)
    candidates = rule_candidates(journal, rules, scenarios)
    assert not any(c.rule_id == "client_left_before_count_complete" for c in candidates)


def test_long_open_door_reports_and_validates_near_the_threshold(settings):
    # min_duration_sec for cash_room_door_open is 4.0: a door seen open for 10s straight should
    # point at ~4s (when it became a violation), not snap back to 0s (when it merely opened) —
    # neither in the raw candidate nor after validate_candidates re-picks a witness frame.
    from forte_guard.contracts import Box, Detection, Frame, ObjectFrame
    from forte_guard.reasoning.validation import validate_candidates

    fps = 5
    count = 50  # 10 seconds of a door staying open
    door = Detection(
        track_id="door1", label="door", confidence=0.9,
        box=Box(x1=0.4, y1=0.0, x2=0.6, y2=0.5), attributes={"state": "open"},
    )
    object_frames = [
        ObjectFrame(frame_id=f"f{i:06d}", timestamp_sec=i / fps, detections=[door]) for i in range(count)
    ]
    frames = [
        Frame(
            frame_id=f"f{i:06d}", timestamp_sec=i / fps, source_frame_index=i,
            width=640, height=360, image_path=f"/tmp/f{i:06d}.jpg",
        )
        for i in range(count)
    ]
    objects, profile = _scene([])
    objects = objects.model_copy(update={"frames": object_frames})
    journal = build_journal(objects, profile, count / fps, fps)

    rules, scenarios = load_catalog(settings.config_dir)
    candidates = [c for c in rule_candidates(journal, rules, scenarios) if c.rule_id == "cash_room_door_open"]
    assert len(candidates) == 1
    # ~4s in (door opens near 0s, debounce shifts the confirmed start slightly) — not the opening.
    assert 3.9 <= candidates[0].timecode_sec <= 4.5

    accepted, _ = validate_candidates(candidates, journal, frames, rules)
    assert len(accepted) == 1
    # Near the threshold (allow one evidence-sampling interval of slack), nowhere near the opening.
    assert 3.0 <= accepted[0].timecode_sec <= 5.5


CASHIER = ("cashier", "person", SEATED)
BYSTANDER = [0.06, 0.15, 0.36, 0.85]  # also inside cashier_area, overlapping the cashier in time


def test_assign_roles_keeps_only_the_cashier_with_more_evidence_when_overlapping():
    frames = [[CASHIER, ("stranger", "person", BYSTANDER)] for _ in range(30)]
    objects, profile = _scene(frames)
    roles = assign_roles(objects, profile)
    assert roles["cashier"] == "cashier"
    assert roles["stranger"] == "unknown"


def test_unidentified_person_without_transaction_is_flagged_from_the_start():
    frames = [[CASHIER, ("stranger", "person", BYSTANDER)] for _ in range(30)]
    objects, profile = _scene(frames)
    journal = build_journal(objects, profile, 6, 5)
    events = [a for a in journal.actions if a.type == "unidentified_person_no_transaction"]
    assert len(events) == 1
    assert events[0].start_sec == 0.0
    assert events[0].actor_ids == ["stranger"]


def test_unidentified_person_is_not_flagged_once_a_transaction_occurs():
    doc = ("doc1", "document", [0.66, 0.3, 0.70, 0.34])
    frames = [[CASHIER, ("stranger", "person", BYSTANDER), CLIENT, doc] for _ in range(30)]
    objects, profile = _scene(frames)
    journal = build_journal(objects, profile, 6, 5)
    assert any(a.type == "document_presented" for a in journal.actions)
    assert not [a for a in journal.actions if a.type == "unidentified_person_no_transaction"]


def test_legitimate_client_alone_is_not_flagged_as_unidentified():
    frames = [[CASHIER, CLIENT] for _ in range(30)]
    objects, profile = _scene(frames)
    journal = build_journal(objects, profile, 6, 5)
    assert not any(a.type == "unidentified_person_no_transaction" for a in journal.actions)


COUNTER_VISIT = [0.44, 0.44, 0.56, 0.56]


def test_a_client_voted_person_who_passes_through_the_counter_without_transacting_is_flagged():
    # Mostly stands in client_area (votes "client" overall), but first steps into the restricted
    # counter_area without any transaction happening there — the client-role vote earned from
    # standing elsewhere does not excuse that visit; only an overlapping transaction event does.
    frames = [[CASHIER, ("visitor", "person", COUNTER_VISIT)] for _ in range(5)]
    frames += [[CASHIER, ("visitor", "person", [0.65, 0.1, 0.95, 0.8])] for _ in range(15)]
    objects, profile = _with_counter(frames)
    journal = build_journal(objects, profile, 4, 5)
    assert journal.roles["visitor"] == "client"
    events = [a for a in journal.actions if a.type == "unidentified_person_no_transaction"]
    assert len(events) == 1
    assert events[0].actor_ids == ["visitor"]


def test_a_client_who_transacts_during_the_counter_visit_is_not_flagged():
    # A separate, genuine client (CLIENT) is present throughout to give document_presented a
    # session to attach to — the document itself sits where "visitor" is, at the counter.
    doc = ("doc1", "document", COUNTER_VISIT)
    frames = [[CASHIER, CLIENT, ("visitor", "person", COUNTER_VISIT), doc] for _ in range(5)]
    frames += [[CASHIER, CLIENT, ("visitor", "person", [0.65, 0.1, 0.95, 0.8])] for _ in range(15)]
    objects, profile = _with_counter(frames)
    journal = build_journal(objects, profile, 4, 5)
    assert journal.roles["visitor"] == "client"
    assert any(a.type == "document_presented" for a in journal.actions)
    assert not any(a.type == "unidentified_person_no_transaction" for a in journal.actions)
