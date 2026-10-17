"""Online preparation only. Runtime inference requires local files and never downloads weights."""

import argparse
import json
import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
os.environ["YOLO_CONFIG_DIR"] = str(ROOT / "data" / "ultralytics")
os.environ["YOLO_AUTOINSTALL"] = "false"
os.environ["YOLO_OFFLINE"] = "false"

from forte_guard.settings import load_env  # noqa: E402

load_env(ROOT / ".env")
# Open-vocabulary detectors whose text classes are baked into local weights.
BASES = {"yoloe": ("yoloe-26s-seg.pt", "cashier-yoloe26.pt"), "world": ("yolov8s-worldv2.pt", "cashier-world.pt")}
# Fine-tuned team weights live in a private GitHub release (derived from the dataset: never publish).
RELEASE = ("detector-v2", "detector-v2.pt")
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--detector", choices=[*sorted(BASES), "yolo"], default=os.getenv("FORTE_DETECTOR", "yolo"))
detector = parser.parse_args().detector
if detector == "yolo":
    tag, asset = RELEASE
    target = ROOT / "models" / asset
    if not target.is_file():
        target.parent.mkdir(exist_ok=True)
        subprocess.run(["gh", "release", "download", tag, "-p", asset, "-D", str(target.parent)], check=True)
    print(f"Detector weights ready: {target.relative_to(ROOT)}. Use FORTE_DETECTOR=yolo and FORTE_MODEL_PATH=models/{asset}.")
    raise SystemExit
if detector not in BASES:
    parser.error(f"FORTE_DETECTOR={detector} has no prepared open-vocabulary weights; use --detector")

from ultralytics import YOLOE, YOLOWorld, settings  # noqa: E402 -- configure local paths before importing

settings.update({"sync": False, "weights_dir": str(ROOT / "models")})
base, name = BASES[detector]
target = ROOT / "models" / name
target.parent.mkdir(exist_ok=True)
configuration = json.loads((ROOT / "config" / "detector.json").read_text())
model = (YOLOE if detector == "yoloe" else YOLOWorld)(str(ROOT / "models" / base))
# YOLOE stores the computed text embeddings in the model, so runtime needs no text encoder either.
model.set_classes(list(configuration["prompts"].values()))
model.save(str(target))
(ROOT / "models" / "vocabulary.json").write_text(json.dumps(configuration["prompts"], indent=2))
print(f"Prepared {target.relative_to(ROOT)} with a local fixed vocabulary. Runtime needs no text encoder download.")
print(f"Use FORTE_DETECTOR={detector} and FORTE_MODEL_PATH=models/{name} in .env.")
print("Prepare the local reasoning model separately: ollama pull gemma4:26b")
