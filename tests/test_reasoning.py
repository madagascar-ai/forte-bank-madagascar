from forte_guard.contracts import Action, ActionJournal, Candidate, Frame, Rule
from forte_guard.reasoning.catalog import load_catalog
from forte_guard.reasoning.rules import rule_candidates
from forte_guard.reasoning.validation import collapse_repeats, merge_candidates, validate_candidates


def action(event_id, kind, batch, start, end):
    return Action(
        event_id=event_id,
        type=kind,
        start_sec=start,
        end_sec=end,
        object_ids=[batch],
        session_id="session1",
        evidence_frame_ids=["f000001"],
        visibility="clear",
        attributes={"to_role": "client"},
    )


def omission_fixture(settings):
    # The strict before_event engine stays supported for full-scenario omissions.
    rule = Rule(
        id="strict_payout_count",
        title="Не наблюдался пересчёт выдаваемой части денег",
        kind="omission",
        required_capabilities=["cash_counted"],
        engine={
            "type": "before_event",
            "required_event": "cash_counted",
            "closing_event": "cash_transferred",
            "closing_attributes": {"to_role": "client"},
            "same_object": True,
            "required_context_flags": ["packaging_exceptions_checked"],
        },
        document={"title": "Правила НБРК № 231", "clause": "пункт 31 подпункт 4"},
    )
    journal = ActionJournal(
        duration_sec=10,
        sample_fps=5,
        capabilities=["cash_counted"],
        coverage={
            "scenario_start_observed": True,
            "complete_observation": True,
            "packaging_exceptions_checked": True,
        },
        actions=[
            action("count-parent", "cash_counted", "parent", 1, 2),
            action("payout-child", "cash_transferred", "child", 5, 6),
        ],
    )
    return journal, rule


def test_counting_parent_does_not_satisfy_payout_of_child(settings):
    journal, rule = omission_fixture(settings)
    found = rule_candidates(journal, [rule], {})
    assert len(found) == 1 and found[0].timecode_sec == 6


def test_counting_same_batch_satisfies_requirement(settings):
    journal, rule = omission_fixture(settings)
    journal.actions[0].object_ids = ["child"]
    assert rule_candidates(journal, [rule], {}) == []


def test_omission_requires_complete_coverage_and_supported_recognizer(settings):
    journal, rule = omission_fixture(settings)
    journal.coverage["complete_observation"] = False
    assert rule_candidates(journal, [rule], {}) == []
    journal.coverage["complete_observation"] = True
    journal.capabilities = []
    assert rule_candidates(journal, [rule], {}) == []


def test_omission_requires_checked_exceptions(settings):
    journal, rule = omission_fixture(settings)
    journal.coverage["packaging_exceptions_checked"] = False
    assert rule_candidates(journal, [rule], {}) == []


def candidate(start=1, end=2, session="s1", source="rule_engine"):
    return Candidate(
        candidate_id="test",
        rule_id="rule",
        kind="action",
        event_ids=["event"],
        start_sec=start,
        end_sec=end,
        timecode_sec=start,
        sources=[source],
        session_id=session,
        evidence_frame_ids=["f000001"],
        reasoning="Synthetic candidate",
    )


def test_merge_keeps_provenance_but_separates_repeated_episodes():
    merged = merge_candidates(
        [candidate(), candidate(source="llm"), candidate(5, 6), candidate(session="s2")]
    )
    assert len(merged) == 3
    both = next(item for item in merged if len(item.sources) == 2)
    assert both.sources == ["rule_engine", "llm"]


def test_pieces_of_one_situation_show_as_one_violation():
    # Hands covering the cash for a second split one state into pieces; a longer break or another
    # session is a separate violation.
    pieces = [candidate(1, 2), candidate(3, 4), candidate(4.2, 6), candidate(9, 10), candidate(1, 2, session="s2")]
    kept = collapse_repeats(pieces)
    assert [(item.session_id, item.start_sec, item.end_sec) for item in kept] == [
        ("s1", 1, 6),
        ("s1", 9, 10),
        ("s2", 1, 2),
    ]
    assert pieces[1].status == "rejected" and pieces[1].verification == "repeat_of_test"


def test_validator_rejects_unknown_event_and_invented_timestamp(settings):
    rules, _ = load_catalog(settings.config_dir)
    journal = ActionJournal(duration_sec=8, sample_fps=5)
    item = candidate()
    item.rule_id = rules[0].id
    item.kind = rules[0].kind
    frames = [
        Frame(
            frame_id="f000001",
            timestamp_sec=4,
            source_frame_index=40,
            width=100,
            height=100,
            image_path="unused.jpg",
        )
    ]
    accepted, warnings = validate_candidates([item], journal, frames, rules)
    assert not accepted and warnings


def test_llm_fabricated_reference_is_discarded(monkeypatch, settings):
    from forte_guard.reasoning.llm import llm_candidates
    from forte_guard.video import Deadline

    rules, scenarios = load_catalog(settings.config_dir)
    journal = ActionJournal(
        duration_sec=8, sample_fps=5, actions=[action("real", "cash_transferred", "batch", 1, 2)]
    )
    settings.llm_enabled = True
    monkeypatch.setattr(
        "forte_guard.reasoning.llm.local_chat",
        lambda *a, **kw: {
            "candidates": [
                {"rule_id": rules[0].id, "event_ids": ["invented"], "state_ids": [], "reasoning": "fake"}
            ]
        },
    )
    found, warnings = llm_candidates(journal, rules, scenarios, settings, Deadline(10))
    assert found == [] and warnings


def test_validator_does_not_accept_unknown_as_absent_cashier(demo, settings):
    from forte_guard.contracts import ObjectJournal
    from forte_guard.journal.builder import build_journal

    _, path, zones = demo
    journal = build_journal(ObjectJournal.model_validate_json(path.read_text()), zones, 8, 5)
    rules, scenarios = load_catalog(settings.config_dir)
    items = rule_candidates(journal, rules, scenarios)
    assert len(items) == 1
    frame = Frame(
        frame_id=items[0].evidence_frame_ids[-1],
        timestamp_sec=items[0].timecode_sec,
        source_frame_index=32,
        width=640,
        height=360,
        image_path="unused.jpg",
    )
    # Keep valid IDs and timing; only change the underlying observation to unknown.
    for state in journal.states:
        if state.name == "cashier_present" and state.value is False:
            state.value = None
    # Resolve all references before reaching the observation consistency check.
    frames = [frame.model_copy(update={"frame_id": key}) for key in items[0].evidence_frame_ids]
    accepted, warnings = validate_candidates(items, journal, frames, rules)
    assert not accepted and warnings
    assert items[0].verification == "observations_do_not_support_rule"
