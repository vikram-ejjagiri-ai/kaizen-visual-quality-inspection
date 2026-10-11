"""Tiny synthetic dataset in the MVTec AD folder layout - for automated tests only.

Good part: a grey ring on a dark background with slight texture.
Defects:   'scratch' (bright line) and 'hole' (dark spot), each with a pixel mask.
"""
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def _part(rng, size=256):
    img = Image.new("RGB", (size, size), (25, 25, 25))
    d = ImageDraw.Draw(img)
    c, r = size // 2 + rng.integers(-4, 5), 90 + rng.integers(-3, 4)
    d.ellipse([c - r, c - r, c + r, c + r], fill=(150, 150, 150))
    d.ellipse([c - 35, c - 35, c + 35, c + 35], fill=(25, 25, 25))
    arr = np.array(img).astype(np.int16) + rng.normal(0, 6, (size, size, 1)).astype(np.int16)
    return Image.fromarray(arr.clip(0, 255).astype(np.uint8))


def make_dataset(root, category="synthetic_part", n_train=24, n_good=8, n_defect=6, seed=0):
    rng = np.random.default_rng(seed)
    base = Path(root) / category
    for sub in ["train/good", "test/good", "test/scratch", "test/hole",
                "ground_truth/scratch", "ground_truth/hole"]:
        (base / sub).mkdir(parents=True, exist_ok=True)
    for i in range(n_train):
        _part(rng).save(base / "train/good" / f"{i:03d}.png")
    for i in range(n_good):
        _part(rng).save(base / "test/good" / f"{i:03d}.png")
    for kind in ["scratch", "hole"]:
        for i in range(n_defect):
            img = _part(rng)
            mask = Image.new("L", img.size, 0)
            d, m = ImageDraw.Draw(img), ImageDraw.Draw(mask)
            x, y = rng.integers(60, 150), rng.integers(60, 150)
            if kind == "scratch":
                box = [x, y, x + 50, y + 8]
                d.line([x, y, x + 50, y + 8], fill=(240, 240, 240), width=4)
                m.line([x, y, x + 50, y + 8], fill=255, width=6)
            else:
                box = [x, y, x + 18, y + 18]
                d.ellipse(box, fill=(0, 0, 0))
                m.ellipse(box, fill=255)
            img.save(base / f"test/{kind}" / f"{i:03d}.png")
            mask.save(base / f"ground_truth/{kind}" / f"{i:03d}_mask.png")
    return base
