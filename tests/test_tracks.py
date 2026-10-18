"""Track repair: one label per track among confusable classes, and tracks broken by the tracker joined."""

from forte_guard.contracts import Box, Detection, ObjectFrame, ObjectJournal, Zone, ZoneProfile
from forte_guard.journal.builder import build_journal
from forte_guard.journal.tracks import repair_tracks

CASHIER = ("cashier", "person", [0.05, 0.1, 0.35, 0.8], 0.9)
CLIENT = ("client", "person", [0.65, 0.1, 0.95, 0.8], 0.9)
PROFILE = ZoneProfile(
    source="synthetic",
    zones=[
        Zone(name="cashier_area", box=Box(x1=0, y1=0, x2=0.4, y2=1)),
        Zone(name="client_area", box=Box(x1=0.6, y1=0, x2=1, y2=1)),
    ],
)


def _objects(frames) -> ObjectJournal:
    def det(track_id, label, box, confidence):
        return Detection(
            track_id=track_id, label=label, confidence=confidence, box=Box(x1=box[0], y1=box[1], x2=box[2], y2=box[3])
        )

    return ObjectJournal(
        backend="synthetic-replay",
        supported_classes=["person", "money", "document", "bag"],
        frames=[
            ObjectFrame(frame_id=f"f{i:06d}", timestamp_sec=i / 5, detections=[det(*item) for item in items])
            for i, items in enumerate(frames)
        ],
    )


def _stack(track_id, x, label="money", confidence=0.8):
    return (track_id, label, [x, 0.45, x + 0.05, 0.5], confidence)


def _track_ids(objects, label):
    return {item.track_id for frame in objects.frames for item in frame.detections if item.label == label}


def _events(journal, kind):
    return [event for event in journal.actions if event.type == kind]


def test_cash_stack_read_as_a_document_for_a_moment_stays_cash():
    # Three confident samples of money outweigh three weak "document" samples of the same track.
    frames = [[CASHIER, CLIENT, _stack("s1", 0.7)] for _ in range(3)]
    frames += [[CASHIER, CLIENT, _stack("s1", 0.7, "document", 0.45)] for _ in range(3)]
    frames += [[CASHIER, CLIENT] for _ in range(5)]
    assert _events(build_journal(_objects(frames), PROFILE, 3, 5), "document_presented")
    objects = repair_tracks(_objects(frames))
    journal = build_journal(objects, PROFILE, 3, 5)
    assert _events(journal, "document_presented") == []
    assert _track_ids(objects, "document") == set()
    assert objects.frames[3].detections[2].attributes["raw_label"] == "document"


def test_a_track_seen_only_as_a_document_is_still_presented():
    frames = [[CASHIER, CLIENT, _stack("d1", 0.7, "document", 0.6)] for _ in range(5)]
    journal = build_journal(repair_tracks(_objects(frames)), PROFILE, 1, 5)
    assert len(_events(journal, "document_presented")) == 1


def test_cash_stack_lost_for_a_moment_keeps_its_track():
    frames = [[CASHIER, _stack("s1", 0.2)] for _ in range(5)] + [[CASHIER] for _ in range(2)]
    frames += [[CASHIER, _stack("s2", 0.3)] for _ in range(4)]
    objects = repair_tracks(_objects(frames))
    assert _track_ids(objects, "money") == {"s1"}
    assert objects.frames[7].detections[1].attributes["raw_track_id"] == "s2"


def test_tracks_are_not_joined_when_both_are_visible_too_late_or_too_far():
    overlap = [[_stack("s1", 0.2)] for _ in range(5)] + [[_stack("s1", 0.2), _stack("s2", 0.25)]]
    overlap += [[_stack("s2", 0.25)] for _ in range(3)]
    late = [[_stack("s1", 0.2)] for _ in range(5)] + [[] for _ in range(10)] + [[_stack("s2", 0.2)]]
    far = [[_stack("s1", 0.1)] for _ in range(5)] + [[_stack("s2", 0.5)] for _ in range(3)]
    for frames in (overlap, late, far):
        assert _track_ids(repair_tracks(_objects(frames)), "money") == {"s1", "s2"}


def test_two_tracks_appearing_where_one_vanished_are_not_joined():
    # A cashier change: the old track ends and two people show up next to where it was.
    frames = [[_stack("s1", 0.2)] for _ in range(5)] + [[_stack("s2", 0.25), _stack("s3", 0.15)] for _ in range(3)]
    assert _track_ids(repair_tracks(_objects(frames)), "money") == {"s1", "s2", "s3"}


def test_tracks_of_different_classes_are_not_joined():
    frames = [[_stack("s1", 0.2)] for _ in range(5)] + [[_stack("b1", 0.2, "bag")] for _ in range(3)]
    objects = repair_tracks(_objects(frames))
    assert _track_ids(objects, "money") == {"s1"} and _track_ids(objects, "bag") == {"b1"}


def test_client_lost_for_a_moment_stays_one_service_session():
    box = CLIENT[2]
    frames = [[CASHIER, ("k1", "person", box, 0.9)] for _ in range(10)] + [[CASHIER] for _ in range(3)]
    frames += [[CASHIER, ("k2", "person", box, 0.9)] for _ in range(10)]
    assert len(_events(build_journal(_objects(frames), PROFILE, 5, 5), "client_observed")) == 2
    assert len(_events(build_journal(repair_tracks(_objects(frames)), PROFILE, 5, 5), "client_observed")) == 1


def test_next_client_elsewhere_at_the_counter_is_a_new_session():
    frames = [[CASHIER, ("k1", "person", [0.65, 0.05, 0.95, 0.5], 0.9)] for _ in range(10)] + [[CASHIER]]
    frames += [[CASHIER, ("k2", "person", [0.65, 0.5, 0.95, 0.98], 0.9)] for _ in range(10)]
    assert len(_events(build_journal(repair_tracks(_objects(frames)), PROFILE, 5, 5), "client_observed")) == 2


PHANTOM = ("ghost", "person", [0.9, 0.3, 0.98, 0.38], 0.4)


def _client_walking_in(i):
    return ("k1", "person", [0.65, 0.6 - i * 0.02, 0.85, 0.85 - i * 0.02], 0.9)


def test_small_person_that_never_moves_is_set_aside():
    # Behind the glass or a false detection: a tenth of the cashier's size, standing still for 4 s.
    frames = [[CASHIER, PHANTOM, *([_client_walking_in(i)] if i >= 10 else [])] for i in range(20)]
    assert len(_events(build_journal(_objects(frames), PROFILE, 4, 5), "client_observed")) == 2
    objects = repair_tracks(_objects(frames))
    journal = build_journal(objects, PROFILE, 4, 5)
    assert [event.actor_ids for event in _events(journal, "client_observed")] == [["k1"]]
    assert _track_ids(objects, "static_person") == {"ghost"}
    assert any("отражение" in warning for warning in journal.warnings)


def test_people_that_move_or_are_full_size_are_kept():
    still_client = ("k1", "person", CLIENT[2], 0.9)
    passing_small = [("p1", "person", [0.9 - i * 0.03, 0.3, 0.98 - i * 0.03, 0.38], 0.9) for i in range(20)]
    brief_small = ("b1", "person", [0.9, 0.6, 0.98, 0.68], 0.4)
    frames = [[CASHIER, still_client, passing_small[i], *([brief_small] if i < 5 else [])] for i in range(20)]
    objects = repair_tracks(_objects(frames))
    assert _track_ids(objects, "static_person") == set()


SEATED_TO_DESK = ("cashier", "person", [0.1, 0.3, 0.3, 0.5], 0.92)
SEATED_WITH_LEGS = ("legs", "person", [0.1, 0.3, 0.3, 0.6], 0.8)


def test_second_box_on_a_seated_person_is_set_aside():
    # Seen from the front, a seated cashier gets a box down to the desk and another down to the legs
    # under it: the shorter track is the same person, not someone else at the workstation.
    frames = [[SEATED_TO_DESK, SEATED_WITH_LEGS] for _ in range(10)] + [[SEATED_TO_DESK] for _ in range(10)]
    objects = repair_tracks(_objects(frames))
    assert _track_ids(objects, "person") == {"cashier"}
    assert _track_ids(objects, "duplicate_person") == {"legs"}
    assert "legs" not in build_journal(objects, PROFILE, 4, 5).roles


def test_people_overlapping_side_by_side_are_both_kept():
    visitor = ("visitor", "person", [0.2, 0.25, 0.42, 0.6], 0.9)  # half of the cashier's box overlaps
    objects = repair_tracks(_objects([[SEATED_TO_DESK, visitor] for _ in range(10)]))
    assert _track_ids(objects, "person") == {"cashier", "visitor"}


def test_key_in_the_lock_from_the_start_that_never_moves_is_the_lock():
    # The detector reads the keyhole as a key from the first frame; a key put in later is still a key.
    lock = ("k1", "safe_key", [0.67, 0.5, 0.68, 0.51], 0.4)
    inserted = ("k2", "safe_key", [0.66, 0.49, 0.68, 0.52], 0.6)
    objects = repair_tracks(_objects([[lock] for _ in range(20)] + [[lock, inserted] for _ in range(20)]))
    assert _track_ids(objects, "safe_key") == {"k2"}
    assert _track_ids(objects, "safe_key_fixture") == {"k1"}
