import json

from examples.make_demo import create_demo
from forte_guard.contracts import ObjectJournal
from forte_guard.journal.builder import StableValue, build_journal
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
    assert candidates[0].timecode_sec == departures[0].start_sec


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
