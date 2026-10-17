from __future__ import annotations

import json
import os
from collections.abc import Callable
from pathlib import Path

from ..contracts import Box, Detection, Frame, ObjectFrame, ObjectJournal
from ..settings import Settings
from ..video import Deadline


def offline_environment(settings: Settings) -> None:
    os.environ["YOLO_OFFLINE"] = "true"
    os.environ["YOLO_AUTOINSTALL"] = "false"
    os.environ["YOLO_CONFIG_DIR"] = str(settings.data_dir.resolve() / "ultralytics")
    os.environ["WANDB_DISABLED"] = "true"
    os.environ["HF_HUB_OFFLINE"] = "1"


class YoloTracker:
    """One model/tracker instance per video: track IDs never leak across jobs."""

    def __init__(self, settings: Settings):
        offline_environment(settings)
        if not settings.model_path.is_file():
            raise FileNotFoundError("Веса детектора не найдены. Выполните python scripts/prepare_models.py")
        from ultralytics import YOLO, YOLOWorld
        from ultralytics import settings as yolo_settings

        yolo_settings.update({"sync": False})
        import torch

        self.settings = settings
        self.model = (YOLOWorld if settings.detector == "world" else YOLO)(str(settings.model_path.resolve()))
        self.device = settings.device
        if self.device == "auto":
            self.device = (
                "cuda:0"
                if torch.cuda.is_available()
                else ("mps" if torch.backends.mps.is_available() else "cpu")
            )
        config = json.loads((settings.config_dir / "detector.json").read_text())
        self.aliases = config.get("aliases", {})
        self.names = list(self.model.names.values())
        self.prompt_to_class = {text: label for label, text in config.get("prompts", {}).items()}
        if settings.detector == "world" and set(self.names) != set(self.prompt_to_class):
            raise ValueError("Словарь весов не совпадает с config/detector.json; повторите prepare_models.py")

    def run(self, frames: list[Frame], deadline: Deadline, progress: Callable) -> ObjectJournal:
        journal = []
        for index, frame in enumerate(frames):
            deadline.remaining()
            predictions = self.model.track(
                source=frame.image_path,
                persist=True,
                tracker="bytetrack.yaml",
                conf=self.settings.detection_confidence,
                imgsz=self.settings.image_size,
                device=self.device,
                verbose=False,
            )[0]
            observations = []
            boxes = predictions.boxes
            if boxes is not None and boxes.id is not None:
                for xyxy, label_index, confidence, track_id in zip(
                    boxes.xyxyn.cpu().tolist(),
                    boxes.cls.cpu().tolist(),
                    boxes.conf.cpu().tolist(),
                    boxes.id.cpu().tolist(),
                    strict=True,
                ):
                    raw = self.model.names[int(label_index)]
                    label = self.prompt_to_class.get(raw, self.aliases.get(raw, raw))
                    attributes = {}
                    for suffix in ("_open", "_closed"):
                        if label.endswith(suffix):
                            attributes["state"] = suffix[1:]
                            label = label.removesuffix(suffix)
                            break
                    coordinates = [max(0.0, min(1.0, float(v))) for v in xyxy]
                    if coordinates[2] <= coordinates[0] or coordinates[3] <= coordinates[1]:
                        continue
                    observations.append(
                        Detection(
                            track_id=f"track_{int(track_id)}",
                            label=label,
                            box=Box(
                                x1=coordinates[0], y1=coordinates[1], x2=coordinates[2], y2=coordinates[3]
                            ),
                            confidence=confidence,
                            attributes=attributes,
                        )
                    )
            journal.append(
                ObjectFrame(
                    frame_id=frame.frame_id, timestamp_sec=frame.timestamp_sec, detections=observations
                )
            )
            if index % 10 == 0 or index == len(frames) - 1:
                progress((index + 1) / len(frames))
        classes = [self.prompt_to_class.get(name, self.aliases.get(name, name)) for name in self.names]
        return ObjectJournal(
            backend=f"{self.settings.detector}+bytetrack",
            supported_classes=classes,
            frames=journal,
            warnings=["Детектор и оценки уверенности требуют проверки на данных банка."],
        )


class ReplayDetector:
    """Explicit test/development adapter, never a silent fallback for a missing model."""

    def __init__(self, source: Path):
        self.journal = ObjectJournal.model_validate_json(source.read_text())

    def run(self, frames: list[Frame], deadline: Deadline, progress: Callable) -> ObjectJournal:
        deadline.remaining()
        allowed = {f.frame_id for f in frames}
        if any(frame.frame_id not in allowed for frame in self.journal.frames):
            raise ValueError("Replay journal contains frame IDs absent from the video")
        progress(1.0)
        return self.journal
