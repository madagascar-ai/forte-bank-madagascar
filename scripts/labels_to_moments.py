"""Training preparation only: turn the organisers' violation timecodes into moments.csv for dense sampling.

The service never reads open_labels.xlsx. Comments are not copied: only video name and second.
"""

import argparse
import csv
import re
from pathlib import Path

import openpyxl

ROOT = Path(__file__).resolve().parents[1]
VIDEO_SUFFIXES = {".mov", ".mp4", ".avi", ".mkv"}


def seconds(text: str) -> float:
    """'22 секунда', '1минута 8 секунда', '1:08' -> seconds."""
    value = str(text).lower()
    if match := re.fullmatch(r"\s*(\d+):(\d+(?:[.,]\d+)?)\s*", value):
        return int(match[1]) * 60 + float(match[2].replace(",", "."))
    minutes = re.search(r"(\d+)\s*мин", value)
    rest = re.sub(r"\d+\s*мин\w*", "", value)
    second = re.search(r"(\d+(?:[.,]\d+)?)", rest)
    if not minutes and not second:
        raise ValueError(f"cannot read timecode '{text}'")
    return (int(minutes[1]) * 60 if minutes else 0) + (float(second[1].replace(",", ".")) if second else 0)


def key(name: str) -> str:
    # The sheet mixes Latin X and Cyrillic Х in names; compare on one alphabet.
    return str(name).strip().upper().replace("X", "Х")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", type=Path, default=ROOT / "data" / "open_labels.xlsx")
    parser.add_argument("--videos", type=Path, default=ROOT / "videos")
    parser.add_argument("--out", type=Path, default=ROOT / "data" / "moments.csv")
    args = parser.parse_args()

    videos = {key(p.stem): p.name for p in args.videos.iterdir() if p.suffix.lower() in VIDEO_SUFFIXES}
    sheet = openpyxl.load_workbook(args.labels, read_only=True, data_only=True).worksheets[0]
    moments, missing = [], []
    for row in list(sheet.iter_rows(values_only=True))[1:]:
        name, timecode = (*row, None, None)[:2]  # trailing empty rows can be shorter
        if not name or timecode in (None, ""):
            continue
        if key(name) not in videos:
            missing.append(str(name))
            continue
        moments.append({"video": videos[key(name)], "timecode_sec": seconds(timecode)})
    with args.out.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["video", "timecode_sec"])
        writer.writeheader()
        writer.writerows(moments)
    print(f"{len(moments)} moments -> {args.out}")
    if missing:
        print(f"Not found in {args.videos}: {', '.join(missing)}")


if __name__ == "__main__":
    main()
