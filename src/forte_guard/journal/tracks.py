"""Track repair between the tracker and the journal.

ByteTrack (Ultralytics) matches boxes by overlap only: a track keeps its id while the detector's class
flickers (a stack of banknotes in a hand read as an ID card for a few samples), and at 5 sampled frames
per second a stack carried by hand or an occluded person comes back under a new id. Both are repaired
on the object journal before roles, zones and events are derived from it.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import dataclass

from ..contracts import ObjectJournal

# Classes the detector confuses on the same object; a track takes the label with the highest summed
# confidence within its group, so a few weak "document" samples don't outvote a confident stack.
LABEL_GROUPS = ({"money", "document"},)
# A track ending and another of the same class starting shortly after and close by is one object lost
# for a few samples: (max gap in seconds, max jump of the box center in normalized coordinates).
# A hand moves a stack across the counter within about a second; a person bending down or hidden
# behind another one stays roughly in place.
STITCH_LIMITS = {
    "money": (1.5, 0.25),
    "person": (2.0, 0.15),
    "cashier": (2.0, 0.15),
    "client": (2.0, 0.15),
}


@dataclass
class _Span:
    labels: Counter
    first_sec: float
    first_center: tuple[float, float]
    last_sec: float
    last_center: tuple[float, float]


def _detections(objects: ObjectJournal):
    for frame in objects.frames:
        for item in frame.detections:
            yield frame.timestamp_sec, item


def _group(label: str) -> int | None:
    return next((index for index, group in enumerate(LABEL_GROUPS) if label in group), None)


def _vote_labels(objects: ObjectJournal) -> None:
    scores: dict[tuple, Counter] = defaultdict(Counter)
    for _, item in _detections(objects):
        group = _group(item.label)
        if group is not None:
            scores[item.track_id, group][item.label] += item.confidence
    for _, item in _detections(objects):
        group = _group(item.label)
        if group is None:
            continue
        label = scores[item.track_id, group].most_common(1)[0][0]
        if label != item.label:
            item.attributes["raw_label"] = item.label
            item.label = label


def _spans(objects: ObjectJournal) -> dict[str, _Span]:
    spans: dict[str, _Span] = {}
    for timestamp, item in _detections(objects):
        span = spans.get(item.track_id)
        if span is None:
            spans[item.track_id] = _Span(Counter([item.label]), timestamp, item.box.center, timestamp, item.box.center)
            continue
        span.labels[item.label] += 1
        span.last_sec, span.last_center = timestamp, item.box.center
    return spans


def _predecessor(track_id: str, spans: dict[str, _Span], continued: set[str]) -> str | None:
    span = spans[track_id]
    label = span.labels.most_common(1)[0][0]
    if label not in STITCH_LIMITS:
        return None
    max_gap, max_jump = STITCH_LIMITS[label]
    options = []
    for other_id, other in spans.items():
        if other_id == track_id or other_id in continued or other.labels.most_common(1)[0][0] != label:
            continue
        gap = span.first_sec - other.last_sec
        jump = math.dist(other.last_center, span.first_center)
        # Strictly after: a track still visible when the new one appears is a different object.
        if 0 < gap <= max_gap and jump <= max_jump:
            options.append((jump, gap, other_id))
    return min(options)[2] if options else None


def _stitch(objects: ObjectJournal) -> None:
    spans = _spans(objects)
    root: dict[str, str] = {}
    continued: set[str] = set()
    for track_id in sorted(spans, key=lambda key: (spans[key].first_sec, key)):
        previous = _predecessor(track_id, spans, continued)
        if previous is not None:
            continued.add(previous)
            root[track_id] = root.get(previous, previous)
    for _, item in _detections(objects):
        if item.track_id in root:
            item.attributes["raw_track_id"] = item.track_id
            item.track_id = root[item.track_id]


def repair_tracks(objects: ObjectJournal) -> ObjectJournal:
    """Give each track one label within confusable classes, then join tracks broken by the tracker."""
    _vote_labels(objects)
    _stitch(objects)
    return objects
