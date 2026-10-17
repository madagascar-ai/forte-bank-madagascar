"""Training preparation: draft YOLO labels with YOLOE-26 text prompts, for human correction.

Drafts are a starting point, not ground truth. Existing label files are never overwritten unless asked,
so corrected labels survive a rerun.
"""

import argparse
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ["YOLO_CONFIG_DIR"] = str(ROOT / "data" / "ultralytics")
os.environ["YOLO_AUTOINSTALL"] = "false"

from ultralytics import YOLOE, settings  # noqa: E402 -- configure local paths before importing

settings.update({"sync": False, "weights_dir": str(ROOT / "models")})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=ROOT / "data" / "training")
    parser.add_argument("--classes", type=Path, default=ROOT / "config" / "training_classes.json")
    parser.add_argument("--weights", type=Path, default=ROOT / "models" / "yoloe-26s-seg.pt")
    parser.add_argument("--conf", type=float, default=0.15)
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--device", default=None)
    parser.add_argument("--overwrite", action="store_true", help="replace existing (possibly corrected) labels")
    args = parser.parse_args()

    classes = json.loads(args.classes.read_text())["classes"]
    names = [item["name"] for item in classes]
    prompts, prompt_class = [], []
    for index, item in enumerate(classes):
        for prompt in item["prompts"]:
            prompts.append(prompt)
            prompt_class.append(index)
    model = YOLOE(str(args.weights))
    model.set_classes(prompts)

    for part in ("train", "val"):
        images = sorted((args.dataset / "images" / part).glob("*.jpg"))
        labels = args.dataset / "labels" / part
        labels.mkdir(parents=True, exist_ok=True)
        todo = [image for image in images if args.overwrite or not (labels / f"{image.stem}.txt").exists()]
        for image, result in zip(todo, _predict(model, todo, args), strict=True):
            lines = [
                f"{prompt_class[int(cls)]} {x:.6f} {y:.6f} {w:.6f} {h:.6f}"
                for cls, (x, y, w, h) in zip(result.boxes.cls.tolist(), result.boxes.xywhn.tolist(), strict=True)
            ]
            (labels / f"{image.stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
        print(f"{part}: drafted {len(todo)} of {len(images)} images")

    (args.dataset / "classes.txt").write_text("\n".join(names) + "\n")
    # data.yaml is written by merge_labels.py and lists only human-reviewed frames.
    print("Drafts ready. Next: scripts/labeling_packages.py <names...>, then scripts/merge_labels.py")


def _predict(model, images, args, batch=16):
    for start in range(0, len(images), batch):
        chunk = [str(image) for image in images[start : start + batch]]
        yield from model.predict(chunk, conf=args.conf, imgsz=args.imgsz, device=args.device, verbose=False)


if __name__ == "__main__":
    main()
