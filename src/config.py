"""Central settings. Every number a reviewer could question lives here."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "mvtec_ad"          # unpack MVTec AD categories here
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# Which MVTec AD product to inspect. Change with:  set KAIZEN_CATEGORY=bottle
CATEGORY = os.environ.get("KAIZEN_CATEGORY", "metal_nut")

# Images are resized to a square. 256 keeps small defects visible and is fast.
IMAGE_SIZE = 256

# PatchCore memory bank: keep this share of all "good" patches (coreset).
CORESET_RATIO = 0.01

# Smoothing of the anomaly map (pixels), as in the PatchCore paper.
HEATMAP_SIGMA = 4

RANDOM_STATE = 42

# Business costs (see docs/01_problem_definition.md):
# a defective part shipped to a customer costs about 20x a good part
# that is wrongly sent to manual re-inspection.
COST_MISSED_DEFECT = 20
COST_FALSE_REJECT = 1

# Share of the test set used to CHOOSE the threshold (the rest is used only to
# REPORT results). MVTec AD has no defective images in its training set, so we
# need labelled examples from somewhere to set a threshold honestly.
CALIBRATION_SHARE = 0.5

# Smoke-test mode: random weights, no internet needed (used by tests only).
SMOKE_TEST = os.environ.get("KAIZEN_SMOKE_TEST") == "1"


def ensure_dirs():
    for folder in (MODELS_DIR, REPORTS_DIR, FIGURES_DIR):
        folder.mkdir(parents=True, exist_ok=True)
