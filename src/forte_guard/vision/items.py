"""Cash-collection bags are green; the detector often calls them personal items.

A personal_item track whose box centre is mostly green over several frames is relabelled cash_bag.
Pure colour statistics, no model call. Threshold from the training boxes: per box, a green share of
at least 0.2 holds for 69% of cash bags and 17% of personal items; the median over a track is steadier.
"""

from __future__ import annotations

import statistics

import cv2
import numpy as np

from ..contracts import Frame, ObjectJournal
from ..settings import Settings

LABEL = "personal_item"
GREEN_SHARE = 0.2  # median share of green pixels in the box centre that makes a track a cash bag
MIN_SAMPLES = 3  # fewer measured frames decide nothing
MAX_SAMPLES = 15  # per track, spread over its frames
CENTRE = 0.6  # measure the middle of the box: less desk and background


def green_share(image: np.ndarray, box) -> float | None:
    height, width = image.shape[:2]
    cx, cy = (box.x1 + box.x2) / 2, (box.y1 + box.y2) / 2
    w, h = (box.x2 - box.x1) * CENTRE, (box.y2 - box.y1) * CENTRE
    x1, x2 = int(max(0.0, cx - w / 2) * width), int(min(1.0, cx + w / 2) * width)
    y1, y2 = int(max(0.0, cy - h / 2) * height), int(min(1.0, cy + h / 2) * height)
    if x2 - x1 < 4 or y2 - y1 < 4:
        return None
    hsv = cv2.cvtColor(image[y1:y2, x1:x2], cv2.COLOR_BGR2HSV)
    green = (hsv[..., 0] >= 35) & (hsv[..., 0] <= 90) & (hsv[..., 1] >= 50) & (hsv[..., 2] >= 40)
    return float(green.mean())


def relabel_green_bags(frames: list[Frame], objects: ObjectJournal, settings: Settings) -> list[str]:
    """Relabel mostly-green personal_item tracks as cash_bag, in place; return journal warnings."""
    if not settings.green_bags:
        return []
    paths = {frame.frame_id: frame.image_path for frame in frames}
    sightings: dict[str, list[tuple[str, object]]] = {}
    for frame in objects.frames:
        for d in frame.detections:
            if d.label == LABEL and frame.frame_id in paths:
                sightings.setdefault(d.track_id, []).append((frame.frame_id, d.box))
    images: dict[str, np.ndarray | None] = {}
    bags = set()
    for track, seen in sightings.items():
        step = max(1, len(seen) // MAX_SAMPLES)
        shares = []
        for frame_id, box in seen[::step][:MAX_SAMPLES]:
            if frame_id not in images:
                images[frame_id] = cv2.imread(paths[frame_id])
            if images[frame_id] is not None and (share := green_share(images[frame_id], box)) is not None:
                shares.append(share)
        if len(shares) >= MIN_SAMPLES and statistics.median(shares) >= GREEN_SHARE:
            bags.add(track)
    for frame in objects.frames:
        for d in frame.detections:
            if d.track_id in bags and d.label == LABEL:
                d.label = "cash_bag"
                d.attributes["label_source"] = "green_colour"
    if not bags:
        return []
    return [f"{len(bags)} трек(а) личных вещей зелёного цвета считаются инкассаторской сумкой (по цвету)."]
