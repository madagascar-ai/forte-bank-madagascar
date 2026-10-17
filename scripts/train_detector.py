"""Fine-tune a fixed-class detector on the corrected local dataset.

Augmentations favour robustness to a different camera angle. Validation metrics come from the held-out
videos of split.json; they say nothing about the final test angle until checked on such footage.
"""

import argparse
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ["YOLO_CONFIG_DIR"] = str(ROOT / "data" / "ultralytics")
os.environ["YOLO_AUTOINSTALL"] = "false"

from ultralytics import YOLO, settings  # noqa: E402 -- configure local paths before importing

settings.update({"sync": False, "weights_dir": str(ROOT / "models")})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "data" / "training" / "data.yaml")
    parser.add_argument("--model", default="yolo26m.pt", help="pretrained start, e.g. yolo26s.pt for a quick run")
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default=None)
    parser.add_argument("--name", default="cashdesk")
    args = parser.parse_args()
    if not args.data.is_file():
        raise SystemExit(f"{args.data} not found: run scripts/merge_labels.py on the reviewed labels first")

    model = YOLO(str(ROOT / "models" / args.model) if (ROOT / "models" / args.model).is_file() else args.model)
    model.train(
        data=str(args.data),
        imgsz=args.imgsz,
        epochs=args.epochs,
        batch=args.batch,
        device=args.device,
        patience=20,
        degrees=15,
        perspective=0.001,
        scale=0.5,
        translate=0.2,
        hsv_v=0.4,
        fliplr=0.5,
        mosaic=1.0,
        project=str(ROOT / "data" / "runs"),
        name=args.name,
        exist_ok=False,
        plots=True,
    )
    best = Path(model.trainer.best)
    print(f"Best weights: {best}")
    print(f"Use: FORTE_DETECTOR=yolo FORTE_MODEL_PATH={best}  (check per-class metrics above first)")


if __name__ == "__main__":
    main()
