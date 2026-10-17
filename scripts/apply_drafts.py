"""Replace drafts of unchecked frames with newer drafts (e.g. from the helper model). Standard library only.

Checked frames are never touched. Frames with boxes you drew yourself are skipped too, so unfinished
manual work survives. Every replaced file is backed up first. Close X-AnyLabeling before running.

Usage from the package folder:  python3 apply_drafts.py new_drafts.zip
"""

import argparse
import json
import shutil
import time
import unicodedata
import zipfile
from pathlib import Path

DRAFT_TAGS = {"draft", "copied"}


def stem_of(data: dict, fallback: str) -> str:
    path = str(data.get("imagePath") or fallback).replace("\\", "/")
    return unicodedata.normalize("NFC", Path(path).stem)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("drafts", type=Path, help="zip or folder with new draft JSON files")
    parser.add_argument("--images", type=Path, default=Path(__file__).resolve().parent / "images")
    args = parser.parse_args()

    if args.drafts.is_dir():
        incoming = [(p.name, p.read_text(encoding="utf-8")) for p in sorted(args.drafts.rglob("*.json"))]
    else:
        with zipfile.ZipFile(args.drafts) as archive:
            incoming = [(m, archive.read(m).decode("utf-8")) for m in archive.namelist() if m.endswith(".json")]
    local = {}
    for path in args.images.glob("*.json"):
        local[stem_of(json.loads(path.read_text(encoding="utf-8")), path.name)] = path

    backup = args.images.parent / f"backup_{time.strftime('%Y%m%d_%H%M%S')}"
    counts = {"replaced": 0, "checked": 0, "manual": 0, "unknown": 0}
    for name, text in incoming:
        draft = json.loads(text)
        target = local.get(stem_of(draft, name))
        if target is None:
            counts["unknown"] += 1
            continue
        current = json.loads(target.read_text(encoding="utf-8"))
        if current.get("checked") is True:
            counts["checked"] += 1
            continue
        if any(shape.get("description") not in DRAFT_TAGS for shape in current.get("shapes", [])):
            counts["manual"] += 1  # a box drawn by hand: keep the person's work
            continue
        backup.mkdir(exist_ok=True)
        shutil.copy2(target, backup / target.name)
        draft["checked"] = False
        draft["imagePath"] = current.get("imagePath", draft.get("imagePath"))
        target.write_text(json.dumps(draft, ensure_ascii=False, indent=2), encoding="utf-8")
        counts["replaced"] += 1
    print(
        f"Replaced {counts['replaced']} drafts. Kept: {counts['checked']} checked, "
        f"{counts['manual']} with your own boxes. Not in this package: {counts['unknown']}."
    )
    if counts["replaced"]:
        print(f"Previous versions saved in {backup}")


if __name__ == "__main__":
    main()
