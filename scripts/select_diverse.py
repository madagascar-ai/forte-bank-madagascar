"""Training preparation: pick the N most diverse frames by cosine distance of DINOv2 features (local).

Each frame becomes a vector: the global DINOv2 token plus patch tokens pooled on a 4x4 grid, so a
change in one region (a hand with cash) moves the vector, not only whole-scene changes. Selection is
k-center greedy: start anywhere, then repeatedly add the frame least similar to everything selected.
No threshold to tune, only a budget; the coverage table shows where more frames stop adding variety.
Weights download once into models/torch_hub; frames never leave the machine.
"""

import argparse
import os
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("TORCH_HOME", str(ROOT / "models" / "torch_hub"))

import cv2  # noqa: E402
import torch  # noqa: E402

SIZE = (224, 448)  # width, height: a 16x32 patch grid, divisible by the 4x4 pooling grid on MPS
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


def embed(paths: list[Path], device: str, batch: int = 32) -> np.ndarray:
    model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14").to(device).eval()
    rows, cols = SIZE[1] // 14, SIZE[0] // 14
    vectors = []
    for start in range(0, len(paths), batch):
        images = []
        for path in paths[start : start + batch]:
            image = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, SIZE, interpolation=cv2.INTER_AREA).astype(np.float32) / 255
            images.append(((image - MEAN) / STD).transpose(2, 0, 1))
        with torch.no_grad():
            features = model.forward_features(torch.from_numpy(np.stack(images)).to(device))
        cls = torch.nn.functional.normalize(features["x_norm_clstoken"], dim=-1)
        patches = features["x_norm_patchtokens"].reshape(len(images), rows, cols, -1).permute(0, 3, 1, 2)
        grid = torch.nn.functional.adaptive_avg_pool2d(patches, (4, 4)).flatten(2).transpose(1, 2)
        grid = torch.nn.functional.normalize(grid, dim=-1).flatten(1) / 4  # 16 cells -> unit norm in total
        vector = torch.nn.functional.normalize(torch.cat([cls, grid], dim=1), dim=-1)
        vectors.append(vector.cpu().numpy())
        print(f"embedded {min(start + batch, len(paths))}/{len(paths)}", end="\r", flush=True)
    print()
    return np.concatenate(vectors)


def k_center(vectors: np.ndarray, budget: int) -> tuple[list[int], list[float]]:
    """Greedy farthest-point selection; returns order and the coverage radius after each pick."""
    chosen = [0]
    distance = 1 - vectors @ vectors[0]
    radius = [float(distance.max())]
    for _ in range(budget - 1):
        index = int(distance.argmax())
        chosen.append(index)
        distance = np.minimum(distance, 1 - vectors @ vectors[index])
        radius.append(float(distance.max()))
    return chosen, radius


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", type=Path, default=ROOT / "data" / "training_5fps")
    parser.add_argument("--split", choices=["train", "val"], default="train")
    parser.add_argument("--budget", type=int, default=3000, help="frames to select")
    parser.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    args = parser.parse_args()

    paths = sorted((args.dataset / "images" / args.split).glob("*.jpg"))
    cache = args.dataset / f"embeddings_{args.split}.npy"
    if cache.is_file() and len(np.load(cache)) == len(paths):
        vectors = np.load(cache)
    else:
        vectors = embed(paths, args.device)
        np.save(cache, vectors)

    chosen, radius = k_center(vectors.astype(np.float32), min(args.budget, len(paths)))
    print("Coverage (largest cosine distance from any frame to its nearest selected frame):")
    for n in (250, 500, 1000, 1500, 2000, 2500, 3000, 4000):
        if n <= len(radius):
            print(f"  {n:5} frames -> {radius[n - 1]:.4f}  (similarity {1 - radius[n - 1]:.4f})")
    target = args.dataset / f"selected_{args.split}.txt"
    target.write_text("\n".join(paths[i].stem for i in sorted(chosen)) + "\n", encoding="utf-8")
    print(f"Selected {len(chosen)} of {len(paths)} {args.split} frames -> {target}")


if __name__ == "__main__":
    main()
