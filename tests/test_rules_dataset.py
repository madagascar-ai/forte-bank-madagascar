"""Rules added from the training labels: each observation becomes one candidate of the right type."""

from examples.make_demo import create_demo
from forte_guard.contracts import ActionJournal, ObjectJournal, State
from forte_guard.journal.builder import build_journal
from forte_guard.reasoning.catalog import load_catalog
from forte_guard.reasoning.rules import rule_candidates


def run_rules(journal, settings):
    rules, scenarios = load_catalog(settings.config_dir)
    return rule_candidates(journal, rules, scenarios)


def journal_with(states, duration=20):
    items = [
        State(
            state_id=f"s{index}",
            name=name,
            value=value,
            start_sec=start,
            end_sec=end,
            evidence_frame_ids=["f000001"],
            visibility="clear",
        )
        for index, (name, value, start, end) in enumerate(states)
    ]
    return ActionJournal(duration_sec=duration, sample_fps=5, states=items, capabilities=[])


def test_cash_taken_without_document_is_reported_once(tmp_path, settings):
    _, path, profile = create_demo(tmp_path / "no_id", show_document=False)
    journal = build_journal(ObjectJournal.model_validate_json(path.read_text()), profile, 8, 5)
    found = [c for c in run_rules(journal, settings) if c.rule_id == "cash_taken_without_id_check"]
    assert len(found) == 1


def test_shown_document_prevents_id_violation(demo, settings):
    _, path, profile = demo
    journal = build_journal(ObjectJournal.model_validate_json(path.read_text()), profile, 8, 5)
    assert any(event.type == "document_presented" for event in journal.actions)
    assert not [c for c in run_rules(journal, settings) if c.rule_id == "cash_taken_without_id_check"]


def test_safe_open_briefly_is_not_a_violation(settings):
    journal = journal_with([("safe_door_state", "open", 2, 5)])
    assert not [c for c in run_rules(journal, settings) if c.rule_id == "safe_left_open"]


def test_safe_left_open_is_reported_at_the_threshold_not_the_opening(settings):
    # safe_left_open's min_duration_sec is 6.0: the violation timecode should land where the
    # overlap actually crossed that threshold (4 + 6 = 10), not at the moment the safe opened (4).
    journal = journal_with([("safe_door_state", "open", 4, 15)])
    found = [c for c in run_rules(journal, settings) if c.rule_id == "safe_left_open"]
    assert len(found) == 1 and found[0].timecode_sec == 10


def test_safe_or_door_seen_closed_again_soon_is_a_routine_operation(settings):
    # Opened for 7 s to take cash out, then clearly closed: not "left open".
    journal = journal_with([("safe_door_state", "open", 117.0, 124.2), ("safe_door_state", "closed", 124.2, 140)],
                           duration=142)
    assert not [c for c in run_rules(journal, settings) if c.rule_id == "safe_left_open"]
    journal = journal_with([("door_door_state", "open", 3, 10), ("door_door_state", "closed", 10.2, 20)])
    assert not [c for c in run_rules(journal, settings) if c.rule_id == "cash_room_door_open"]


def test_safe_never_seen_closed_or_open_too_long_is_still_reported(settings):
    # Open until the clip ends, or out of sight after opening: losing sight is not a closing.
    for states in ([("safe_door_state", "open", 4, 20)], [("safe_door_state", "open", 4, 12)]):
        assert [c for c in run_rules(journal_with(states), settings) if c.rule_id == "safe_left_open"]
    # Closed eventually, but open 40 s: that is leaving it open.
    journal = journal_with([("safe_door_state", "open", 5, 45), ("safe_door_state", "closed", 45, 60)], duration=60)
    assert [c for c in run_rules(journal, settings) if c.rule_id == "safe_left_open"]
    # A closing far later than the open interval's end does not excuse it.
    journal = journal_with([("safe_door_state", "open", 4, 12), ("safe_door_state", "closed", 16, 20)])
    assert [c for c in run_rules(journal, settings) if c.rule_id == "safe_left_open"]


def test_unattended_cash_yields_to_departure_during_service(settings):
    journal = journal_with(
        [("cash_on_counter", True, 5, 15), ("cashier_present", False, 6, 15)]
    )
    found = {c.rule_id for c in run_rules(journal, settings)}
    assert "cash_unattended" in found


AT_DESK = [0.05, 0.1, 0.35, 0.8]  # cashier's seat, inside cashier_area but away from the machine
AT_MACHINE = [0.28, 0.38, 0.44, 0.62]  # standing right at the counting machine


def counting_objects(cashier_near_machine: bool, seconds_in_machine: float = 1.4):
    from forte_guard.contracts import Box, Detection, ObjectFrame

    def det(track, label, box):
        return Detection(track_id=track, label=label, confidence=0.9, box=Box(x1=box[0], y1=box[1], x2=box[2], y2=box[3]))

    frames = []
    for index in range(30):
        t = index / 5
        cashier_box = AT_MACHINE if (cashier_near_machine and 0.4 <= t < 3.2) else AT_DESK
        items = [
            det("cashier", "person", cashier_box),
            det("client", "person", [0.65, 0.1, 0.95, 0.8]),
            det("machine", "counter", [0.30, 0.40, 0.42, 0.60]),
        ]
        if 1.0 <= t < 1.0 + seconds_in_machine:
            items.append(det("stack", "cash", [0.33, 0.45, 0.38, 0.52]))
        frames.append(ObjectFrame(frame_id=f"f{index:06d}", timestamp_sec=t, detections=items))
    return ObjectJournal(backend="synthetic", supported_classes=[], frames=frames)


def counting_profile():
    from forte_guard.contracts import Box, Zone, ZoneProfile

    return ZoneProfile(
        source="synthetic",
        zones=[
            Zone(name="cashier_area", box=Box(x1=0, y1=0, x2=0.45, y2=1)),
            Zone(name="client_area", box=Box(x1=0.6, y1=0, x2=1, y2=1)),
        ],
    )


def test_cash_in_machine_with_cashier_nearby_is_counted():
    journal = build_journal(counting_objects(cashier_near_machine=True), counting_profile(), 6, 5)
    counted = [e for e in journal.actions if e.type == "cash_counted"]
    assert len(counted) == 1 and counted[0].object_ids == ["stack"]
    assert "cash_counted" in journal.capabilities and "cash_counted" not in journal.unsupported_actions


def test_cash_near_machine_without_cashier_or_too_short_is_not_counted():
    far = build_journal(counting_objects(cashier_near_machine=False), counting_profile(), 6, 5)
    brief = build_journal(counting_objects(True, seconds_in_machine=0.4), counting_profile(), 6, 5)
    assert not [e for e in far.actions if e.type == "cash_counted"]
    assert not [e for e in brief.actions if e.type == "cash_counted"]


def test_cashier_present_in_zone_but_far_from_machine_is_not_counted():
    # Regression: the cashier sits in cashier_area the whole clip (as on a long counter) — so
    # cashier_present is True throughout — but never physically at the machine. Being "somewhere in
    # the zone" must not be enough to count as a recount; only proximity to the machine itself does.
    journal = build_journal(counting_objects(cashier_near_machine=False), counting_profile(), 6, 5)
    assert any(s.name == "cashier_present" and s.value is True for s in journal.states)
    assert not [e for e in journal.actions if e.type == "cash_counted"]


def payout_journal(counted_object: str):
    from forte_guard.contracts import Action

    def act(event_id, kind, obj, t, **attributes):
        return Action(event_id=event_id, type=kind, start_sec=t, end_sec=t, object_ids=[obj],
                      evidence_frame_ids=["f000001"], visibility="clear", confidence=0.7, attributes=attributes)

    return ActionJournal(
        duration_sec=30,
        sample_fps=5,
        capabilities=["cash_counted"],
        actions=[
            act("e1", "cash_counted", counted_object, 5),
            act("e2", "cash_transferred", "part", 12, to_role="client", from_role="cashier"),
        ],
    )


def test_part_given_to_client_without_its_own_count_is_reported(settings):
    found = [c for c in run_rules(payout_journal("whole_stack"), settings)
             if c.rule_id == "payout_without_observed_count"]
    assert len(found) == 1 and found[0].timecode_sec == 12


def test_counted_part_given_to_client_is_fine(settings):
    found = [c for c in run_rules(payout_journal("part"), settings) if c.rule_id == "payout_without_observed_count"]
    assert found == []


def two_client_payout_journal():
    from forte_guard.contracts import Action

    def act(event_id, kind, obj, t, session_id, **attributes):
        return Action(
            event_id=event_id, type=kind, start_sec=t, end_sec=t, object_ids=[obj],
            session_id=session_id, evidence_frame_ids=["f000001"], visibility="clear",
            confidence=0.7, attributes=attributes,
        )

    return ActionJournal(
        duration_sec=40,
        sample_fps=5,
        capabilities=["cash_counted"],
        actions=[
            # First client: counted, then paid out. Clean.
            act("e1", "cash_counted", "stack1", 5, "session_0001"),
            act("e2", "cash_transferred", "stack1", 8, "session_0001", to_role="client", from_role="cashier"),
            # Second client: paid out without any count for their money. A violation.
            act("e3", "cash_transferred", "stack2", 25, "session_0002", to_role="client", from_role="cashier"),
        ],
    )


def test_second_clients_missed_count_is_reported_even_if_first_was_fine(settings):
    found = [c for c in run_rules(two_client_payout_journal(), settings)
             if c.rule_id == "payout_without_observed_count"]
    assert len(found) == 1 and found[0].timecode_sec == 25
