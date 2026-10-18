import cv2
import numpy as np

from forte_guard.contracts import Box, Detection, Frame, ObjectFrame, ObjectJournal
from forte_guard.video import Deadline
from forte_guard.vision.states import read_states

BOX = Box(x1=0.2, y1=0.2, x2=0.6, y2=0.6)


def frame(tmp_path, index, t, shade):
    image = np.full((200, 200, 3), 60, np.uint8)
    cv2.rectangle(image, (40, 40), (120, 120), (shade, shade, shade), -1)  # inside BOX: the object's look
    path = tmp_path / f"f{index}.jpg"
    cv2.imwrite(str(path), image)
    return Frame(frame_id=f"f{index}", timestamp_sec=t, source_frame_index=index, width=200, height=200,
                 image_path=str(path))


def journal(frames, labels=("door",)):
    return ObjectJournal(
        backend="synthetic",
        supported_classes=list(labels),
        frames=[
            ObjectFrame(
                frame_id=f.frame_id,
                timestamp_sec=f.timestamp_sec,
                detections=[Detection(track_id=f"track_{label}", label=label, box=BOX, confidence=0.9)
                            for label in labels],
            )
            for f in frames
        ],
    )


class Model:
    def __init__(self, *answers):
        self.answers, self.calls = list(answers), 0

    def __call__(self, settings, prompt, schema, deadline, images, *args):
        self.calls += 1
        assert images[0].is_file() and schema["properties"]["state"]["enum"][-1] == "unknown"
        answer = self.answers.pop(0) if len(self.answers) > 1 else self.answers[0]
        if isinstance(answer, Exception):
            raise answer
        return {"state": answer}


def states(objects):
    return [[d.attributes.get("state") for d in f.detections] for f in objects.frames]


def test_disabled_stage_does_nothing(settings, tmp_path):
    frames = [frame(tmp_path, 0, 0.0, 200)]
    objects, model = journal(frames), Model("open")
    assert read_states(frames, objects, settings, Deadline(300), tmp_path / "crops", ask=model) == []
    assert model.calls == 0 and states(objects) == [[None]]


def test_state_is_reused_until_the_crop_changes_or_goes_stale(settings, tmp_path):
    settings.object_states = True
    frames = [frame(tmp_path, 0, 0.0, 200), frame(tmp_path, 1, 1.0, 200),
              frame(tmp_path, 2, 2.0, 30), frame(tmp_path, 3, 14.0, 30)]
    objects, model = journal(frames), Model("closed", "open", "open")
    notes = read_states(frames, objects, settings, Deadline(300), tmp_path / "crops", ask=model)
    assert model.calls == 3  # first look, a changed look, and a stale re-check; the repeat is reused
    assert states(objects) == [["closed"], ["closed"], ["open"], ["open"]]
    assert objects.frames[0].detections[0].attributes["state_source"] == "local_vlm"
    assert "3 запросов, 1 повторных" in notes[0]


def test_unknown_invalid_and_failed_answers_give_no_state(settings, tmp_path):
    settings.object_states = True
    for answer in ("unknown", "half-open", TimeoutError("slow")):
        frames = [frame(tmp_path, 0, 0.0, 200)]
        objects = journal(frames)
        notes = read_states(frames, objects, settings, Deadline(300), tmp_path / "crops", ask=Model(answer))
        assert states(objects) == [[None]]
    assert any("не удалась" in note for note in notes)


def test_laptop_reads_on_off_and_the_call_budget_is_respected(settings, tmp_path):
    settings.object_states = True
    settings.states_max_calls = 1
    frames = [frame(tmp_path, 0, 0.0, 200)]
    objects, model = journal(frames, labels=("laptop", "safe")), Model("on")
    notes = read_states(frames, objects, settings, Deadline(300), tmp_path / "crops", ask=model)
    assert model.calls == 1
    assert states(objects) == [["on", None]]  # the safe stays unknown once the budget is spent
    assert any("лимит" in note for note in notes)


def test_other_labels_and_tiny_boxes_are_ignored(settings, tmp_path):
    settings.object_states = True
    frames = [frame(tmp_path, 0, 0.0, 200)]
    objects, model = journal(frames, labels=("person",)), Model("open")
    tiny = Detection(track_id="track_door", label="door", box=Box(x1=0.1, y1=0.1, x2=0.15, y2=0.15),
                     confidence=0.9)
    objects.frames[0].detections.append(tiny)
    read_states(frames, objects, settings, Deadline(300), tmp_path / "crops", ask=model)
    assert model.calls == 0


def test_without_budget_a_steady_object_keeps_its_last_state(settings, tmp_path):
    settings.object_states = True
    settings.states_max_calls = 1
    frames = [frame(tmp_path, 0, 0.0, 200), frame(tmp_path, 1, 40.0, 200), frame(tmp_path, 2, 41.0, 30)]
    objects, model = journal(frames), Model("closed")
    read_states(frames, objects, settings, Deadline(300), tmp_path / "crops", ask=model)
    assert model.calls == 1
    # Stale but unchanged keeps "closed"; a changed look with no budget left stays unknown.
    assert states(objects) == [["closed"], ["closed"], [None]]


def test_every_decision_is_traced_with_the_crop_and_reason(settings, tmp_path):
    import json

    settings.object_states = True
    frames = [frame(tmp_path, 0, 0.0, 200), frame(tmp_path, 1, 1.0, 200), frame(tmp_path, 2, 2.0, 30)]
    objects = journal(frames)

    def model(settings, prompt, schema, deadline, images, *args):
        assert "reason" in schema["required"]
        return {"state": "closed", "reason": "leaf flush with the frame"}

    read_states(frames, objects, settings, Deadline(300), tmp_path / "job" / "state_crops", ask=model)
    trace = json.loads((tmp_path / "job" / "state_trace.json").read_text())
    actions = [e["action"] for e in trace["entries"]]
    assert actions == ["asked", "reused", "asked"]
    asked = trace["entries"][0]
    assert asked["why"] == "first look" and asked["reason"] == "leaf flush with the frame"
    assert (tmp_path / "job" / "state_crops" / asked["crop"]).is_file()
    assert trace["entries"][2]["why"] == "changed" and trace["entries"][2]["change"] >= 12


def test_state_crops_are_served_only_by_exact_name(settings):
    from fastapi.testclient import TestClient

    from forte_guard.api import create_app
    from forte_guard.storage import JobStore

    store = JobStore(settings.data_dir)
    store.create("d" * 32, "x.mov")
    crops = store.folder("d" * 32) / "state_crops"
    crops.mkdir(parents=True, exist_ok=True)
    (crops / "f000001_track_1.jpg").write_bytes(b"\xff\xd8\xff")
    headers = {"X-API-Key": settings.api_key}
    with TestClient(create_app(settings)) as client:
        assert client.get(f"/api/v1/jobs/{'d' * 32}/state-crops/f000001_track_1.jpg", headers=headers).status_code == 200
        assert client.get(f"/api/v1/jobs/{'d' * 32}/state-crops/..%2Fresult.json", headers=headers).status_code == 404
        assert client.get(f"/api/v1/jobs/{'d' * 32}/state-crops/f000001_track_1.jpg").status_code == 401


def test_detector_box_jitter_does_not_trigger_new_reads(settings, tmp_path):
    settings.object_states = True
    frames = [frame(tmp_path, i, i * 0.2, 200) for i in range(5)]
    objects, model = journal(frames), Model("closed")
    for i, f in enumerate(objects.frames):  # the same object, its box shifting a few pixels every frame
        f.detections[0].box = Box(x1=0.2 + 0.01 * (i % 2), y1=0.2, x2=0.6 + 0.01 * (i % 2), y2=0.6)
    read_states(frames, objects, settings, Deadline(300), tmp_path / "crops", ask=model)
    assert model.calls == 1


def test_an_object_that_keeps_changing_is_read_at_most_once_a_second(settings, tmp_path):
    settings.object_states = True
    frames = [frame(tmp_path, i, i * 0.2, 30 if i % 2 else 220) for i in range(10)]  # flickers every frame
    objects, model = journal(frames), Model("closed")
    read_states(frames, objects, settings, Deadline(300), tmp_path / "crops", ask=model)
    assert model.calls == 2  # at 0.0 s and 1.0 s over two seconds of video
