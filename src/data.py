"""Reading the MVTec AD folder structure.

data/mvtec_ad/<category>/
    train/good/*.png                 only defect-free parts (that's the point)
    test/good/*.png                  defect-free test parts
    test/<defect_type>/*.png         defective parts, one folder per defect type
    ground_truth/<defect_type>/*_mask.png   pixel masks of each defect
"""
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from torchvision import transforms

from src.config import CALIBRATION_SHARE, DATA_DIR, IMAGE_SIZE, RANDOM_STATE

# ImageNet statistics: the pretrained backbone expects images normalized this way
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp"}


def category_dir(category, data_dir=None):
    folder = Path(data_dir or DATA_DIR) / category
    if not (folder / "train" / "good").exists():
        raise FileNotFoundError(
            f"Could not find {folder / 'train' / 'good'}.\n"
            f"Download the '{category}' category of MVTec AD and unpack it into "
            f"{Path(data_dir or DATA_DIR)} (see README, step 2).")
    return folder


def _images(folder):
    return sorted(p for p in Path(folder).iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)


def list_images(category, data_dir=None):
    """One row per image: split, path, label (0 good / 1 defect), defect type, mask path."""
    root = category_dir(category, data_dir)
    rows = [{"split": "train", "path": str(p), "label": 0, "defect_type": "good", "mask_path": None}
            for p in _images(root / "train" / "good")]
    for defect_folder in sorted((root / "test").iterdir()):
        if not defect_folder.is_dir():
            continue
        defect = defect_folder.name
        for p in _images(defect_folder):
            mask = None
            if defect != "good":
                candidate = root / "ground_truth" / defect / f"{p.stem}_mask.png"
                mask = str(candidate) if candidate.exists() else None
            rows.append({"split": "test", "path": str(p), "label": int(defect != "good"),
                         "defect_type": defect, "mask_path": mask})
    return pd.DataFrame(rows)


def split_calibration_evaluation(test_df):
    """Split the test images in two halves, stratified by defect type.

    calibration: used to choose the decision threshold
    evaluation : used once, to report results
    """
    calib, evaluation = train_test_split(
        test_df, test_size=1 - CALIBRATION_SHARE,
        stratify=test_df["defect_type"], random_state=RANDOM_STATE)
    return calib.reset_index(drop=True), evaluation.reset_index(drop=True)


def image_transform():
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def load_mask(mask_path):
    """Binary mask resized to IMAGE_SIZE. No mask (good part) = all zeros."""
    if not isinstance(mask_path, str) or not mask_path:   # None or NaN = good part
        return np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8)
    mask = Image.open(mask_path).convert("L").resize((IMAGE_SIZE, IMAGE_SIZE), Image.NEAREST)
    return (np.array(mask) > 127).astype(np.uint8)


def load_display_image(path):
    """RGB image at IMAGE_SIZE for plots (not normalized)."""
    return np.array(Image.open(path).convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE)))


class ImageFolderDataset(Dataset):
    def __init__(self, paths):
        self.paths = list(paths)
        self.transform = image_transform()

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        image = Image.open(self.paths[i]).convert("RGB")
        return self.transform(image)


def to_tensor_batch(pil_images):
    t = image_transform()
    return torch.stack([t(img.convert("RGB")) for img in pil_images])
