"""Training preparation: cut team videos into full-resolution frames, split by video, skip near-duplicates.

Local only. Frames stay under data/ (git-ignored); never upload them to cloud labeling services.
"""

import argparse
import csv
import json
import random
from pathlib import Path

import cv2
import numpy as np

VIDEO_SUFFIXES = {".mov", ".mp4", ".avi", ".mkv"}


def load_moments(path: Path | None) -> dict[str, list[float]]:
    """CSV with columns video,timecode_sec: moments to sample densely (violations, rare objects)."""
    moments: dict[str, list[float]] = {}
    if path:
        with path.open(newline="") as handle:
            for row in csv.DictReader(handle):
                moments.setdefault(Path(row["video"]).stem, []).append(float(row["timecode_sec"]))
    return moments


def choose_split(videos: list[Path], out: Path, val: list[str] | None, val_count: int, seed: int) -> dict:
    """Split by video, never by frame: neighbouring frames are near-copies and would leak into validation."""
    split_file = out / "split.json"
    stems = sorted(video.stem for video in videos)
    if val is None and split_file.is_file():
        split = json.loads(split_file.read_text())
        known = set(split["train"]) | set(split["val"])
        split["train"] += [stem for stem in stems if stem not in known]  # new footage goes to train
    else:
        chosen = set(val) if val is not None else set(random.Random(seed).sample(stems, min(val_count, len(stems))))
        unknown = chosen - set(stems)
        if unknown:
            raise SystemExit(f"Validation videos not found: {sorted(unknown)}")
        split = {"train": [s for s in stems if s not in chosen], "val": [s for s in stems if s in chosen]}
    out.mkdir(parents=True, exist_ok=True)
    split_file.write_text(json.dumps(split, ensure_ascii=False, indent=2))
    return split


GRID = (8, 8)  # rows, columns of the change grid


def signature(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.resize(gray, (GRID[1] * 8, GRID[0] * 8), interpolation=cv2.INTER_AREA).astype(np.float32)


def local_change(current: np.ndarray, previous: np.ndarray) -> float:
    """Largest mean difference over grid cells: a hand over the desk is local, so a global mean misses it."""
    difference = np.abs(current - previous).reshape(GRID[0], 8, GRID[1], 8)
    return float(difference.mean(axis=(1, 3)).max())


def extract(video: Path, target: Path, moments: list[float], args) -> list[dict]:
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise SystemExit(f"Cannot decode {video}")
    rows, last_time, last_signature = [], None, None
    while True:
        ok, image = capture.read()
        if not ok:
            break
        timestamp = capture.get(cv2.CAP_PROP_POS_MSEC) / 1000  # real decoded timestamp
        dense = any(abs(timestamp - moment) <= args.dense_window for moment in moments)
        interval = 1 / (args.dense_fps if dense else args.fps)
        if last_time is not None and timestamp - last_time < interval - 1e-3:
            continue
        current = signature(image)
        # Static scenes give many identical frames; dense moments are kept even when they look similar.
        if not dense and last_signature is not None and local_change(current, last_signature) < args.min_change:
            continue
        last_time, last_signature = timestamp, current
        name = f"{video.stem}_t{round(timestamp * 1000):07d}.jpg"
        cv2.imwrite(str(target / name), image, [cv2.IMWRITE_JPEG_QUALITY, 95])
        rows.append({"video": video.name, "frame": name, "timestamp_sec": round(timestamp, 3), "dense": dense})
    capture.release()
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("videos", type=Path, help="folder with the team's training videos")
    parser.add_argument("--out", type=Path, default=Path("data/training"))
    parser.add_argument("--val", nargs="*", help="video names (without extension) reserved for validation")
    parser.add_argument("--val-count", type=int, default=8, help="random validation videos when --val is absent")
    parser.add_argument("--seed", type=int, default=26)
    parser.add_argument("--fps", type=float, default=1.0)
    parser.add_argument("--moments", type=Path, help="CSV video,timecode_sec for dense sampling")
    parser.add_argument("--dense-fps", type=float, default=5.0)
    parser.add_argument("--dense-window", type=float, default=3.0, help="seconds around each moment")
    parser.add_argument(
        "--min-change", type=float, default=10.0, help="gray difference in the most changed cell that counts as new"
    )
    args = parser.parse_args()

    videos = sorted(p for p in args.videos.iterdir() if p.suffix.lower() in VIDEO_SUFFIXES)
    if not videos:
        raise SystemExit(f"No videos in {args.videos}")
    split = choose_split(videos, args.out, args.val, args.val_count, args.seed)
    moments = load_moments(args.moments)
    manifest = []
    for video in videos:
        part = "val" if video.stem in split["val"] else "train"
        target = args.out / "images" / part
        target.mkdir(parents=True, exist_ok=True)
        rows = extract(video, target, moments.get(video.stem, []), args)
        manifest += [{**row, "split": part} for row in rows]
        print(f"{part:5} {video.name}: {len(rows)} frames")
    with (args.out / "manifest.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["video", "split", "frame", "timestamp_sec", "dense"])
        writer.writeheader()
        writer.writerows(manifest)
    counts = {part: sum(row["split"] == part for row in manifest) for part in ("train", "val")}
    print(f"Done: {counts['train']} train / {counts['val']} val frames in {args.out}. Next: scripts/prelabel.py")


if __name__ == "__main__":
    main()
