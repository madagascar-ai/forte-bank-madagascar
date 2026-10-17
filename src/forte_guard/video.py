from __future__ import annotations

from pathlib import Path
from time import monotonic

import cv2

from .contracts import Frame


class Deadline:
    def __init__(self, seconds: float):
        self.end = monotonic() + seconds

    def remaining(self) -> float:
        remaining = self.end - monotonic()
        if remaining <= 0:
            raise TimeoutError("Достигнут лимит времени обработки")
        return remaining


def sample_video(
    source: Path, output: Path, fps: float, deadline: Deadline, max_duration: float = 180
) -> tuple[list[Frame], dict]:
    output.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise ValueError("Не удалось прочитать видео; проверьте формат и кодек")
    frames = []
    next_time = 0.0
    index = 0
    source_fps = capture.get(cv2.CAP_PROP_FPS)
    if not 0 < source_fps <= 1000:
        capture.release()
        raise ValueError("Не удалось определить частоту кадров видео")
    advertised_duration = capture.get(cv2.CAP_PROP_FRAME_COUNT) / source_fps
    if advertised_duration > max_duration:
        capture.release()
        raise ValueError(f"Видео длиннее поддерживаемого лимита {max_duration:g} с")
    last_time = -1.0
    timestamp_fallback = False
    try:
        while capture.grab():
            deadline.remaining()
            timestamp = capture.get(cv2.CAP_PROP_POS_MSEC) / 1000
            if timestamp <= last_time and index:
                timestamp = index / source_fps
                timestamp_fallback = True
            if timestamp > max_duration:
                raise ValueError("Декодированное видео превышает лимит длительности")
            last_time = timestamp
            if timestamp + 1e-6 >= next_time:
                ok, image = capture.retrieve()
                if not ok:
                    raise ValueError("Ошибка декодирования выбранного кадра")
                frame_id = f"f{len(frames):06d}"
                path = output / f"{frame_id}.jpg"
                if not cv2.imwrite(str(path), image, [cv2.IMWRITE_JPEG_QUALITY, 88]):
                    raise OSError("Не удалось сохранить кадр")
                height, width = image.shape[:2]
                frames.append(
                    Frame(
                        frame_id=frame_id,
                        timestamp_sec=round(timestamp, 6),
                        source_frame_index=index,
                        width=width,
                        height=height,
                        image_path=str(path),
                    )
                )
                next_time = (int(timestamp * fps + 1e-6) + 1) / fps
            index += 1
    finally:
        capture.release()
    if not frames:
        raise ValueError("Видео не содержит доступных кадров")
    duration = last_time + 1 / source_fps
    return frames, {
        "duration_sec": round(duration, 6),
        "source_fps": source_fps,
        "sample_fps": fps,
        "source_frame_count": index,
        "timestamp_fallback_used": timestamp_fallback,
    }
