"""Estimate workstation zones from tracked detections so nobody has to draw them."""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from itertools import pairwise

from ..contracts import Box, Detection, ObjectJournal, Zone, ZoneProfile

PEOPLE = {"person", "cashier", "client"}
# Objects a cashier works with; proximity to them separates the cashier from a long-staying client.
# The detector strips "_open"/"_closed" into attributes, and fine-tuned weights detect these objects
# without a state, so zones match the base label.
WORK_OBJECTS = {"counting_machine", "laptop", "safe"}
STORAGE = {"safe"}
EXITS = {"door"}
# Items that rest on the counter while cash is handled; only stationary detections count.
RESTING_ITEMS = {"money", "document"}
# Without these the journal cannot assign roles or locate cash, so a fallback is worth trying.
CORE_ZONES = {"cashier_area", "counter_area"}


def _center(item: Detection) -> tuple[float, float]:
    return item.box.center


def _quantile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    return ordered[round(q * (len(ordered) - 1))]


def _spread(boxes: list[Box], pad: float, low: float, high: float, min_size: float = 0.04) -> Box | None:
    if not boxes:
        return None
    x1 = _quantile([b.x1 for b in boxes], low) - pad
    y1 = _quantile([b.y1 for b in boxes], low) - pad
    x2 = _quantile([b.x2 for b in boxes], high) + pad
    y2 = _quantile([b.y2 for b in boxes], high) + pad
    return _box(x1, y1, x2, y2, min_size)


def _box(x1: float, y1: float, x2: float, y2: float, min_size: float = 0.04) -> Box | None:
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    half_w, half_h = max(x2 - x1, min_size) / 2, max(y2 - y1, min_size) / 2
    x1, x2 = max(0.0, cx - half_w), min(1.0, cx + half_w)
    y1, y2 = max(0.0, cy - half_h), min(1.0, cy + half_h)
    if x2 - x1 < 0.02 or y2 - y1 < 0.02:
        return None
    return Box(x1=x1, y1=y1, x2=x2, y2=y2)


def _home(points: list[tuple[float, float]]) -> tuple[float, float]:
    # The most occupied 0.1 cell, not the mean: a cashier who walks away keeps their workstation.
    cells = Counter((min(int(x * 10), 9), min(int(y * 10), 9)) for x, y in points)
    (gx, gy), _ = cells.most_common(1)[0]
    anchor = ((gx + 0.5) / 10, (gy + 0.5) / 10)
    near = [p for p in points if math.dist(p, anchor) < 0.15]
    return _quantile([p[0] for p in near], 0.5), _quantile([p[1] for p in near], 0.5)


def _split(cashier: Box, client: Box, home: tuple, other: tuple) -> tuple[Box | None, Box | None]:
    """Cut both areas at the midpoint between the two homes so a centre never falls into both."""
    dx, dy = other[0] - home[0], other[1] - home[1]
    if abs(dx) >= abs(dy):
        mid = (home[0] + other[0]) / 2
        if dx > 0:
            return _box(cashier.x1, cashier.y1, min(cashier.x2, mid), cashier.y2), _box(
                max(client.x1, mid), client.y1, client.x2, client.y2
            )
        return _box(max(cashier.x1, mid), cashier.y1, cashier.x2, cashier.y2), _box(
            client.x1, client.y1, min(client.x2, mid), client.y2
        )
    mid = (home[1] + other[1]) / 2
    if dy > 0:
        return _box(cashier.x1, cashier.y1, cashier.x2, min(cashier.y2, mid)), _box(
            client.x1, max(client.y1, mid), client.x2, client.y2
        )
    return _box(cashier.x1, max(cashier.y1, mid), cashier.x2, cashier.y2), _box(
        client.x1, client.y1, client.x2, min(client.y2, mid)
    )


def estimate_zones(objects: ObjectJournal) -> tuple[ZoneProfile, list[str]]:
    """Return zones inferred from where people stay and where items rest, plus uncertainty notes.

    Zones that the evidence cannot support are omitted (unknown), never guessed as full-frame.
    """
    total = len(objects.frames)
    profile = ZoneProfile(source="auto_tracks")
    if not total:
        return profile, ["Автозоны: нет кадров с детекциями."]
    tracks: dict[tuple[str, str], list[tuple[int, Detection]]] = defaultdict(list)
    for index, frame in enumerate(objects.frames):
        for item in frame.detections:
            group = "person" if item.label in PEOPLE else item.label
            tracks[(group, item.track_id)].append((index, item))
    notes: list[str] = []
    zones: dict[str, Box] = {}

    def fixture(labels: set[str]) -> Box | None:
        boxes = [item.box for (group, _), seen in tracks.items() if group in labels for _, item in seen]
        return _spread(boxes, 0.02, 0.1, 0.9) if len(boxes) >= 3 else None

    for name, labels in (("storage_area", STORAGE), ("exit_area", EXITS)):
        if box := fixture(labels):
            zones[name] = box

    minimum = max(3, math.ceil(total * 0.05))
    people = {key[1]: seen for key, seen in tracks.items() if key[0] == "person" and len(seen) >= minimum}
    stations: list[dict] = []
    for track_id, seen in sorted(people.items(), key=lambda pair: -len(pair[1])):
        home = _home([_center(item) for _, item in seen])
        station = next((s for s in stations if math.dist(s["home"], home) < 0.15), None)
        if station is None:
            station = {"home": home, "boxes": [], "frames": set(), "tracks": set(), "labels": Counter()}
            stations.append(station)
        station["tracks"].add(track_id)
        station["labels"].update(item.label for _, item in seen if item.label != "person")
        for index, item in seen:
            if math.dist(_center(item), home) < 0.15:
                station["boxes"].append(item.box)
                station["frames"].add(index)
    work = [_center(item) for (group, _), seen in tracks.items() if group in WORK_OBJECTS for _, item in seen]

    def score(station: dict) -> float:
        coverage = len(station["frames"]) / total
        near = sum(math.dist(p, station["home"]) < 0.35 for p in work) / len(work) if work else 0.0
        labels = station["labels"]
        named = (labels["cashier"] - labels["client"]) / max(1, labels["cashier"] + labels["client"])
        return coverage + near + named

    ranked = sorted(stations, key=score, reverse=True)
    cashier = ranked[0] if ranked and len(ranked[0]["frames"]) / total >= 0.3 else None
    if cashier is None:
        notes.append("Автозоны: не найден человек, постоянно находящийся на рабочем месте.")
    elif len(ranked) > 1 and score(ranked[0]) - score(ranked[1]) < 0.15:
        notes.append("Автозоны: не удалось отличить кассира от клиента по трекам.")
        cashier = None
    client_home = None
    if cashier is not None:
        cashier_box = _spread(cashier["boxes"], 0.02, 0.1, 0.9)
        others = [
            item
            for (group, track_id), seen in tracks.items()
            if group == "person" and track_id not in cashier["tracks"]
            for _, item in seen
        ]
        client_box = _spread([item.box for item in others], 0.02, 0.05, 0.95) if len(others) >= minimum else None
        if cashier_box and client_box:
            centers = [_center(item) for item in others]
            client_home = (_quantile([p[0] for p in centers], 0.5), _quantile([p[1] for p in centers], 0.5))
            cashier_box, client_box = _split(cashier_box, client_box, cashier["home"], client_home)
        if cashier_box:
            zones["cashier_area"] = cashier_box
        if client_box:
            zones["client_area"] = client_box
        else:
            notes.append("Автозоны: клиентская сторона не наблюдалась.")

    resting = []
    for (group, _), seen in tracks.items():
        if group == "counting_machine":
            resting.extend(_center(item) for _, item in seen)
        elif group in RESTING_ITEMS:
            for (i, a), (j, b) in pairwise(seen):
                if j - i <= 2 and math.dist(_center(a), _center(b)) < 0.015:
                    resting.append(_center(b))
    if storage := zones.get("storage_area"):
        resting = [p for p in resting if not storage.contains(p)]
    if len(resting) >= 3:
        xs, ys = [p[0] for p in resting], [p[1] for p in resting]
        counter = _box(
            _quantile(xs, 0.1) - 0.05,
            _quantile(ys, 0.1) - 0.05,
            _quantile(xs, 0.9) + 0.05,
            _quantile(ys, 0.9) + 0.05,
            min_size=0.1,
        )
        if counter:
            zones["counter_area"] = counter
    elif cashier is not None and client_home is not None:
        (hx, hy), (kx, ky) = cashier["home"], client_home
        mx, my = (hx + kx) / 2, (hy + ky) / 2
        half_x, half_y = 0.15 * abs(kx - hx) + 0.06, 0.15 * abs(ky - hy) + 0.06
        if counter := _box(mx - half_x, my - half_y, mx + half_x, my + half_y):
            zones["counter_area"] = counter
            notes.append("Автозоны: стол оценён между кассиром и клиентом, лежащие деньги не наблюдались.")
    else:
        notes.append("Автозоны: поверхность стола не определена.")
    profile.zones = [Zone(name=name, box=box) for name, box in zones.items()]
    return profile, notes
