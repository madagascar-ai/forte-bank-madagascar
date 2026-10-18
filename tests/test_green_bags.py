import cv2
import numpy as np

from forte_guard.contracts import Box, Detection, Frame, ObjectFrame, ObjectJournal
from forte_guard.vision.items import relabel_green_bags

BOX = Box(x1=0.2, y1=0.2, x2=0.6, y2=0.6)


def journal(tmp_path, colour, count=5, label="personal_item"):
    frames, rows = [], []
    for index in range(count):
        image = np.full((200, 200, 3), 120, np.uint8)
        cv2.rectangle(image, (40, 40), (120, 120), colour, -1)
        path = tmp_path / f"f{index}.jpg"
        cv2.imwrite(str(path), image)
        frames.append(Frame(frame_id=f"f{index}", timestamp_sec=index * 0.2, source_frame_index=index,
                            width=200, height=200, image_path=str(path)))
        rows.append(ObjectFrame(frame_id=f"f{index}", timestamp_sec=index * 0.2, detections=[
            Detection(track_id="track_1", label=label, box=BOX, confidence=0.8)]))
    return frames, ObjectJournal(backend="synthetic", supported_classes=[label], frames=rows)


def labels(objects):
    return {d.label for f in objects.frames for d in f.detections}


def test_green_personal_item_becomes_a_cash_bag_only_when_enabled(settings, tmp_path):
    frames, objects = journal(tmp_path, (40, 160, 40))  # BGR green
    assert relabel_green_bags(frames, objects, settings) == [] and labels(objects) == {"personal_item"}
    settings.green_bags = True
    notes = relabel_green_bags(frames, objects, settings)
    assert labels(objects) == {"cash_bag"} and notes
    assert objects.frames[0].detections[0].attributes["label_source"] == "green_colour"


def test_other_colours_too_few_frames_and_other_labels_stay(settings, tmp_path):
    settings.green_bags = True
    for colour, count, label in (((40, 40, 160), 5, "personal_item"), ((40, 160, 40), 2, "personal_item"),
                                 ((40, 160, 40), 5, "phone")):
        frames, objects = journal(tmp_path, colour, count, label)
        relabel_green_bags(frames, objects, settings)
        assert labels(objects) == {label}
