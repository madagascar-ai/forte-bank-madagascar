import json

from forte_guard.contracts import Box, Detection, ObjectFrame, ObjectJournal
from forte_guard.journal.builder import assign_roles
from forte_guard.journal.zones import estimate_zones
from forte_guard.pipeline import run_pipeline


def detection(track_id, label, box):
    return Detection(track_id=track_id, label=label, confidence=0.9, box=Box(x1=box[0], y1=box[1], x2=box[2], y2=box[3]))


def journal(frames):
    return ObjectJournal(
        backend="synthetic-replay",
        supported_classes=["person", "money", "counting_machine"],
        frames=[ObjectFrame(frame_id=f"f{i:06d}", timestamp_sec=i / 5, detections=d) for i, d in enumerate(frames)],
    )


def zone(profile, name):
    return next((z.box for z in profile.zones if z.name == name), None)


def test_zones_follow_a_stationary_cashier_rotating_clients_and_resting_cash():
    frames = []
    for index in range(60):
        items = [
            detection("cashier", "person", [0.05, 0.2, 0.3, 0.9]),
            detection("machine", "counting_machine", [0.32, 0.6, 0.4, 0.68]),
        ]
        # Clients come and go as separate short tracks on the opposite side.
        if index % 20 < 15:
            items.append(detection(f"client{index // 20}", "person", [0.7, 0.15, 0.95, 0.85]))
        if 10 <= index < 30:
            items.append(detection("cash", "money", [0.48, 0.55, 0.53, 0.6]))
        frames.append(items)
    objects = journal(frames)
    profile, _ = estimate_zones(objects)
    assert profile.source == "auto_tracks"
    cashier, client, counter = (zone(profile, n) for n in ("cashier_area", "client_area", "counter_area"))
    assert cashier.contains((0.175, 0.55)) and not client.contains((0.175, 0.55))
    assert client.contains((0.825, 0.5)) and not cashier.contains((0.825, 0.5))
    assert counter.contains((0.505, 0.575))
    roles = assign_roles(objects, profile)
    assert roles["cashier"] == "cashier"
    assert {roles[f"client{i}"] for i in range(3)} == {"client"}


def test_indistinguishable_people_leave_roles_unknown_instead_of_guessing():
    frames = [
        [detection("a", "person", [0.05, 0.2, 0.3, 0.9]), detection("b", "person", [0.7, 0.2, 0.95, 0.9])]
        for _ in range(40)
    ]
    profile, notes = estimate_zones(journal(frames))
    assert zone(profile, "cashier_area") is None and zone(profile, "client_area") is None
    assert any("отличить кассира" in note for note in notes)


def test_no_observations_produce_no_zones():
    profile, notes = estimate_zones(journal([[] for _ in range(10)]))
    assert profile.zones == [] and notes


def test_pipeline_estimates_zones_without_operator_input(settings, demo, tmp_path):
    video, objects, _ = demo
    settings.auto_zones = True
    output = tmp_path / "auto"
    result = run_pipeline(video, output, settings, "synthetic", replay=objects)
    saved = json.loads((output / "zones.json").read_text())
    assert saved["source"] == "auto_tracks" and saved["zones"]
    assert "Зоны определены автоматически; проверьте их расположение." in result.warnings


def test_stateless_fixtures_from_detector_locate_storage_exit_and_cashier():
    # The detector reports "safe"/"door"/"laptop" (state stripped or absent in fine-tuned weights).
    frames = []
    for index in range(40):
        items = [
            detection("a", "person", [0.05, 0.2, 0.3, 0.9]),
            detection("b", "person", [0.7, 0.2, 0.95, 0.9]) if index % 10 < 6 else None,
            detection("safe", "safe", [0.02, 0.05, 0.12, 0.18]),
            detection("door", "door", [0.85, 0.0, 0.98, 0.15]),
            detection("laptop", "laptop", [0.2, 0.55, 0.3, 0.62]),
        ]
        frames.append([item for item in items if item])
    profile, _ = estimate_zones(journal(frames))
    assert zone(profile, "storage_area").contains((0.07, 0.11))
    assert zone(profile, "exit_area").contains((0.9, 0.07))
    assert zone(profile, "cashier_area").contains((0.175, 0.55))


def test_fine_tuned_classes_map_to_journal_vocabulary():
    root = __import__("pathlib").Path(__file__).resolve().parents[1]
    aliases = json.loads((root / "config/detector.json").read_text())["aliases"]
    trained = {
        item["name"] for item in json.loads((root / "config/training_classes.json").read_text())["classes"]
    }
    journal_names = {"person", "money", "document", "counting_machine", "safe", "door", "laptop"}
    mapped = {aliases.get(name, name) for name in trained}
    assert journal_names <= mapped
