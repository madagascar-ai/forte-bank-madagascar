"""Training preparation: drop near-duplicate frames before paid or slow labeling.

A frame is kept when, in at least one cell of an 8x8 grid, its mean gray level differs from the last
kept frame of the same video by >= threshold (0-255). A hand moving over the desk changes one cell,
which a whole-frame hash or embedding would average away. Static scenes measure 1-7, motion 12+.
"""

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prepare_dataset import local_change, signature


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("data/training_5fps"))
    parser.add_argument("--split", choices=["train", "val"], default="train")
    parser.add_argument("--threshold", type=float, default=10.0)
    args = parser.parse_args()

    manifest = csv.DictReader((args.dataset / "manifest.csv").open(encoding="utf-8"))
    rows = [row for row in manifest if row["split"] == args.split]
    by_video = defaultdict(list)
    for row in rows:
        by_video[row["video"]].append(row)
    kept = []
    for _, frames in sorted(by_video.items()):
        last = None
        for row in sorted(frames, key=lambda r: float(r["timestamp_sec"])):
            current = signature(cv2.imread(str(args.dataset / "images" / args.split / row["frame"])))
            if last is None or local_change(current, last) >= args.threshold:
                kept.append(Path(row["frame"]).stem)
                last = current
    target = args.dataset / f"selected_{args.split}.txt"
    target.write_text("\n".join(kept) + "\n", encoding="utf-8")
    print(f"Kept {len(kept)} of {len(rows)} {args.split} frames (threshold {args.threshold}) -> {target}")


if __name__ == "__main__":
    main()
