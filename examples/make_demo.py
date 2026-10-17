"""Generate explicitly synthetic video + tracked objects for independent module development."""

from pathlib import Path

import cv2
import numpy as np

from forte_guard.contracts import Box, Detection, ObjectFrame, ObjectJournal, Zone, ZoneProfile
from forte_guard.storage import write_json


def create_demo(output: Path, disappear: bool = False):
    output.mkdir(parents=True, exist_ok=True)
    profile = ZoneProfile(
        source="synthetic",
        zones=[
            Zone(name="cashier_area", box=Box(x1=0, y1=0, x2=0.4, y2=1)),
            Zone(name="client_area", box=Box(x1=0.6, y1=0, x2=1, y2=1)),
            Zone(name="counter_area", box=Box(x1=0.4, y1=0.35, x2=0.6, y2=0.7)),
        ],
    )
    records = []

    def detection(track_id, label, box):
        return Detection(
            track_id=track_id,
            label=label,
            confidence=0.95,
            box=Box(x1=box[0], y1=box[1], x2=box[2], y2=box[3]),
        )

    for index in range(40):
        timestamp = index / 5
        people = [detection("client", "person", [0.65, 0.1, 0.95, 0.8])]
        if timestamp < 3:
            people.append(detection("cashier", "person", [0.05, 0.1, 0.35, 0.8]))
        elif not disappear:
            people.append(detection("cashier", "person", [0.43, 0.72, 0.57, 0.97]))
        if timestamp < 1:
            money = detection("batch1", "money", [0.68, 0.53, 0.72, 0.57])
            hands = [detection("hand_client", "hand", [0.68, 0.53, 0.72, 0.57])]
        elif timestamp < 2:
            money = detection("batch1", "money", [0.28, 0.53, 0.32, 0.57])
            hands = [detection("hand_cashier", "hand", [0.28, 0.53, 0.32, 0.57])]
        else:
            money = detection("batch1", "money", [0.48, 0.53, 0.52, 0.57])
            hands = []
        records.append(
            ObjectFrame(
                frame_id=f"f{index:06d}", timestamp_sec=timestamp, detections=[*people, money, *hands]
            )
        )
    journal = ObjectJournal(
        backend="synthetic-replay",
        supported_classes=["person", "hand", "money"],
        frames=records,
        warnings=["Синтетический пример для проверки интеграции; не результат работы YOLO."],
    )
    source = output / "video.mov"
    writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*"avc1"), 10, (640, 360))
    if not writer.isOpened():
        # Some CI/Linux OpenCV builds lack an H.264 encoder; replay tests still work.
        writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*"mp4v"), 10, (640, 360))
    if not writer.isOpened():
        raise RuntimeError("Synthetic video writer is unavailable")
    for index in range(80):
        image = np.full((360, 640, 3), 242, dtype=np.uint8)
        cv2.putText(
            image, "SYNTHETIC PIPELINE DEMO", (18, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (75, 75, 75), 2
        )
        for obj in records[min(index // 2, 39)].detections:
            box = obj.box
            color = (110, 150, 70) if obj.label == "money" else (130, 90, 180)
            cv2.rectangle(
                image,
                (int(box.x1 * 640), int(box.y1 * 360)),
                (int(box.x2 * 640), int(box.y2 * 360)),
                color,
                2,
            )
            cv2.putText(
                image,
                obj.track_id,
                (int(box.x1 * 640), max(45, int(box.y1 * 360) - 3)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                color,
                1,
            )
        writer.write(image)
    writer.release()
    write_json(output / "objects.json", journal)
    write_json(output / "zones.json", profile)
    return source, output / "objects.json", profile


if __name__ == "__main__":
    source, _, _ = create_demo(Path("data/demo"))
    print(f"Synthetic demo generated: {source}")
