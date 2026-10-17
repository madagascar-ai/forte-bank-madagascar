"""Merge X-AnyLabeling JSON returned by the team into YOLO labels; only checked frames enter training.

Accepts zips or folders, can be run again as more work arrives: reviewed frames accumulate.
Pass sources oldest first: when one frame comes back several times, the later source wins.
Frames are identified by imagePath inside the JSON, so archives made on Windows with garbled
Cyrillic file names still merge. Unknown class names are reported and skipped, never guessed.
"""

import argparse
import csv
import json
import unicodedata
import zipfile
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def returned_files(sources: list[Path]):
    for source in sources:
        if source.is_dir():
            for path in sorted(source.rglob("*.json")):
                if "backup_" not in str(path.parent):
                    yield str(path), path.name, path.read_text(encoding="utf-8")
        elif zipfile.is_zipfile(source):
            with zipfile.ZipFile(source) as archive:
                for member in sorted(archive.namelist()):
                    name = Path(member).name
                    if member.endswith(".json") and not name.startswith(".") and "backup_" not in member:
                        yield f"{source}:{member}", name, archive.read(member).decode("utf-8")
        else:
            raise SystemExit(f"Not a folder or zip: {source}")


def frame_stem(data: dict, fallback: str) -> str:
    path = str(data.get("imagePath") or fallback).replace("\\", "/")
    return unicodedata.normalize("NFC", Path(path).stem)


def to_yolo(data: dict, names: list[str]) -> list[str]:
    width, height = float(data["imageWidth"]), float(data["imageHeight"])
    lines = []
    for shape in data.get("shapes", []):
        label = shape["label"].strip()
        if label not in names:
            raise ValueError(f"unknown class '{label}'")
        if shape.get("shape_type") not in {"rectangle", "polygon"}:
            raise ValueError(f"unsupported shape '{shape.get('shape_type')}' for {label}")
        xs = [min(max(point[0], 0.0), width) for point in shape["points"]]
        ys = [min(max(point[1], 0.0), height) for point in shape["points"]]
        x1, x2, y1, y2 = min(xs), max(xs), min(ys), max(ys)
        if x2 - x1 < 2 or y2 - y1 < 2:
            continue  # accidental click, not an object
        lines.append(
            f"{names.index(label)} {(x1 + x2) / 2 / width:.6f} {(y1 + y2) / 2 / height:.6f} "
            f"{(x2 - x1) / width:.6f} {(y2 - y1) / height:.6f}"
        )
    return lines


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sources", nargs="+", type=Path, help="returned zips or folders")
    parser.add_argument("--dataset", type=Path, default=ROOT / "data" / "training")
    args = parser.parse_args()

    names = (args.dataset / "classes.txt").read_text().split()
    split_of = {
        unicodedata.normalize("NFC", Path(row["frame"]).stem): row["split"]
        for row in csv.DictReader((args.dataset / "manifest.csv").open(encoding="utf-8"))
    }
    reviewed_file = args.dataset / "reviewed.json"
    reviewed = set(json.loads(reviewed_file.read_text())) if reviewed_file.is_file() else set()

    accepted: dict[str, tuple[str, list[str]]] = {}
    problems, unchecked, replaced = [], 0, 0
    for origin, name, text in returned_files(args.sources):
        try:
            data = json.loads(text)
            stem = frame_stem(data, name)
            if stem not in split_of:
                problems.append(f"{origin}: frame {stem} is not in manifest.csv")
                continue
            if data.get("checked") is not True:
                unchecked += 1
                continue
            lines = to_yolo(data, names)
        except (ValueError, KeyError, TypeError) as error:
            problems.append(f"{origin}: {error}")
            continue
        if stem in accepted and accepted[stem][1] != lines:
            replaced += 1  # a later return of the same frame supersedes the earlier one
        accepted[stem] = (origin, lines)

    boxes = Counter()
    for stem, (_, lines) in accepted.items():
        target = args.dataset / "labels" / split_of[stem] / f"{stem}.txt"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("\n".join(lines) + ("\n" if lines else ""))
        boxes.update(names[int(line.split()[0])] for line in lines)
        reviewed.add(stem)
    reviewed_file.write_text(json.dumps(sorted(reviewed), ensure_ascii=False, indent=2))

    # Training reads only reviewed frames: unchecked drafts must never pass as ground truth.
    for part in ("train", "val"):
        listed = sorted(
            str((args.dataset / "images" / part / f"{stem}.jpg").resolve())
            for stem in reviewed
            if split_of.get(stem) == part
        )
        (args.dataset / f"{part}.txt").write_text("\n".join(listed) + ("\n" if listed else ""))
    data = [f"path: {args.dataset.resolve()}", "train: train.txt", "val: val.txt", "names:"]
    data += [f"  {index}: {name}" for index, name in enumerate(names)]
    (args.dataset / "data.yaml").write_text("\n".join(data) + "\n")

    totals = Counter(split_of[stem] for stem in reviewed)
    print(f"Merged {len(accepted)} checked frames now; skipped {unchecked} unchecked.")
    if replaced:
        print(f"{replaced} frames came back more than once; the later source was used.")
    print(f"Reviewed in total: {totals['train']} train / {totals['val']} val of {len(split_of)} frames.")
    print("Boxes in this merge: " + ", ".join(f"{name} {boxes[name]}" for name in names))
    if not totals["val"]:
        print("No reviewed validation frames yet: training needs labeled frames from the held-out videos too.")
    if problems:
        print(f"\n{len(problems)} problems (files skipped):")
        print("\n".join(f"  {problem}" for problem in problems))


if __name__ == "__main__":
    main()
