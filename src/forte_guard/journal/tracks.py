"""Track repair between the tracker and the journal.

ByteTrack (Ultralytics) matches boxes by overlap only: a track keeps its id while the detector's class
flickers (a stack of banknotes in a hand read as an ID card for a few samples), and at 5 sampled frames
per second a stack carried by hand or an occluded person comes back under a new id. Both are repaired
on the object journal before zones, roles and events are derived from it, together with "people" that
never move and are far smaller than everyone else.
"""

from __future__ import annotations

import math
import statistics
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
PEOPLE = {"person", "cashier", "client"}
# A "person" that stays put for a while and is much smaller than the person seen longest (normally the
# cashier) is not someone at the counter: the cashier's reflection in the glass door when it opens, or a
# static false detection. Counted as a client, it opens service sessions and pulls the auto client area
# off the counter.
STATIC_MAX_MOVE = 0.05
STATIC_MIN_SEC = 2.0
STATIC_SIZE_RATIO = 0.5


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


def _could_continue(earlier: _Span, later: _Span) -> bool:
    label = earlier.labels.most_common(1)[0][0]
    if label not in STITCH_LIMITS or later.labels.most_common(1)[0][0] != label:
        return False
    max_gap, max_jump = STITCH_LIMITS[label]
    # Strictly after: a track still visible when the new one appears is a different object.
    gap = later.first_sec - earlier.last_sec
    return 0 < gap <= max_gap and math.dist(earlier.last_center, later.first_center) <= max_jump


def _stitch(objects: ObjectJournal) -> None:
    spans = _spans(objects)
    pairs = [(a, b) for a in spans for b in spans if a != b and _could_continue(spans[a], spans[b])]
    # Only unambiguous joins: when two people appear where one vanished (a cashier change, someone
    # stepping up to the counter), picking the nearer one swaps identities.
    successors = Counter(a for a, _ in pairs)
    predecessors = Counter(b for _, b in pairs)
    previous = {b: a for a, b in pairs if successors[a] == 1 and predecessors[b] == 1}
    root: dict[str, str] = {}
    for track_id in sorted(previous, key=lambda key: spans[key].first_sec):
        root[track_id] = root.get(previous[track_id], previous[track_id])
    for _, item in _detections(objects):
        if item.track_id in root:
            item.attributes["raw_track_id"] = item.track_id
            item.track_id = root[item.track_id]


def _size(seen: list) -> float:
    return statistics.median(math.sqrt((item.box.x2 - item.box.x1) * (item.box.y2 - item.box.y1)) for _, item in seen)


def _set_aside_static_people(objects: ObjectJournal) -> None:
    people = defaultdict(list)
    for timestamp, item in _detections(objects):
        if item.label in PEOPLE:
            people[item.track_id].append((timestamp, item))
    if not people:
        return
    reference = _size(max(people.values(), key=len))
    static = 0
    for seen in people.values():
        xs = [item.box.center[0] for _, item in seen]
        ys = [item.box.center[1] for _, item in seen]
        still = max(max(xs) - min(xs), max(ys) - min(ys)) < STATIC_MAX_MOVE
        if still and seen[-1][0] - seen[0][0] >= STATIC_MIN_SEC and _size(seen) < STATIC_SIZE_RATIO * reference:
            static += 1
            for _, item in seen:
                item.attributes["raw_label"] = item.label
                item.label = "static_person"
    if static:
        objects.warnings.append(
            f"Не учитываются неподвижные маленькие «люди» ({static}): вероятно, отражение в стекле "
            "или ложная детекция."
        )


def repair_tracks(objects: ObjectJournal) -> ObjectJournal:
    """One label per track among confusable classes, broken tracks joined, static phantom people set aside."""
    _vote_labels(objects)
    _stitch(objects)
    _set_aside_static_people(objects)
    return objects
