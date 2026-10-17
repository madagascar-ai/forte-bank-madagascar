"""Online preparation only. Runtime inference requires local files and never downloads weights."""

import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
os.environ["YOLO_CONFIG_DIR"] = str(ROOT / "data" / "ultralytics")
os.environ["YOLO_AUTOINSTALL"] = "false"
os.environ["YOLO_OFFLINE"] = "false"

from ultralytics import YOLOWorld, settings  # noqa: E402 -- configure local paths before importing

settings.update({"sync": False, "weights_dir": str(ROOT / "models")})
target = ROOT / "models" / "cashier-world.pt"
target.parent.mkdir(exist_ok=True)
configuration = json.loads((ROOT / "config" / "detector.json").read_text())
model = YOLOWorld(str(ROOT / "models" / "yolov8s-worldv2.pt"))
model.set_classes(list(configuration["prompts"].values()))
model.save(str(target))
(ROOT / "models" / "vocabulary.json").write_text(json.dumps(configuration["prompts"], indent=2))
print("Prepared YOLO-World with a local fixed vocabulary. Runtime no longer needs a text encoder download.")
print("Prepare the local reasoning model separately: ollama pull gemma3:4b")
