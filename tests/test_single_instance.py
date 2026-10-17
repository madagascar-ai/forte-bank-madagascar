import json
from pathlib import Path

from forte_guard.contracts import Box, Detection
from forte_guard.vision.detector import keep_strongest

ROOT = Path(__file__).resolve().parents[1]


def detection(track: str, label: str, confidence: float) -> Detection:
    return Detection(
        track_id=track, label=label, box=Box(x1=0.1, y1=0.1, x2=0.2, y2=0.2), confidence=confidence
    )


def test_only_the_most_confident_door_survives_and_others_are_untouched():
    frame = [
        detection("track_1", "door", 0.41),
        detection("track_2", "person", 0.90),
        detection("track_3", "door", 0.87),
        detection("track_4", "door", 0.55),
        detection("track_5", "person", 0.80),
    ]
    kept = keep_strongest(frame, {"door"})
    assert [d.track_id for d in kept] == ["track_2", "track_3", "track_5"]


def test_frames_without_single_instance_labels_pass_through():
    frame = [detection("track_1", "money", 0.5), detection("track_2", "money", 0.6)]
    assert keep_strongest(frame, {"door"}) == frame


def test_door_is_configured_as_single_instance():
    config = json.loads((ROOT / "config" / "detector.json").read_text())
    assert "door" in config["single_instance"]
