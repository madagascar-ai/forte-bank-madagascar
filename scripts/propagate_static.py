"""Copy fixed objects (door, safe, laptop, counter, currency detector) from a checked frame to the other frames of the same video.

Within one video the camera does not move, so a box drawn once is valid on every frame of that video.
Only unchecked frames change; checked frames are never touched. Close X-AnyLabeling before running,
then reopen it and review the copies as usual. Standard library only.
"""

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

STATIC = ("door", "safe", "laptop", "counter", "currency_detector")
FRAME = re.compile(r"^(?P<video>.+)_t(?P<ms>\d{7})$")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "images", type=Path, nargs="?", default=Path(__file__).resolve().parent / "images",
        help="package images folder (default: images next to this script)",
    )
    parser.add_argument("--classes", nargs="+", default=list(STATIC))
    args = parser.parse_args()

    videos = defaultdict(list)
    for path in sorted(args.images.glob("*.json")):
        match = FRAME.match(path.stem)
        if match:
            videos[match["video"]].append((int(match["ms"]), path))

    for video, frames in sorted(videos.items()):
        loaded = [(ms, path, json.loads(path.read_text(encoding="utf-8"))) for ms, path in frames]
        sources = [(ms, data) for ms, _, data in loaded if data.get("checked") is True]
        if not sources:
            print(f"{video}: no checked frame yet, skipped")
            continue
        changed = 0
        for ms, path, data in loaded:
            if data.get("checked") is True:
                continue
            # The nearest checked frame in time: closest to any small camera or object shift.
            _, source = min(sources, key=lambda item: abs(item[0] - ms))
            static = [dict(shape, description="copied") for shape in source["shapes"] if shape["label"] in args.classes]
            kept = [shape for shape in data["shapes"] if shape["label"] not in args.classes]
            data["shapes"] = kept + static
            path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
            changed += 1
        print(f"{video}: {', '.join(args.classes)} copied to {changed} unchecked frames from {len(sources)} checked")


if __name__ == "__main__":
    main()
