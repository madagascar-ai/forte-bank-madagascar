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
