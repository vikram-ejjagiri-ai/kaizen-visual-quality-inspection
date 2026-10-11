"""Automated checks. Run:  python -m pytest

Uses random network weights and a tiny synthetic dataset, so it needs no
internet and no GPU. It checks the plumbing, not the accuracy.
"""
import os

os.environ["KAIZEN_SMOKE_TEST"] = "1"

import numpy as np
import pytest
import torch

from src.data import list_images, load_mask, split_calibration_evaluation
from src.evaluation import best_threshold, decisions_summary, image_auroc
from src.patchcore import PatchCore, greedy_coreset, nearest_distance
from tests.synthetic_data import make_dataset


@pytest.fixture(scope="module")
def dataset(tmp_path_factory):
    root = tmp_path_factory.mktemp("data")
    make_dataset(root)
    return root


def test_listing_reads_mvtec_layout(dataset):
    df = list_images("synthetic_part", data_dir=dataset)
    assert (df["split"] == "train").sum() == 24
    assert set(df["defect_type"]) == {"good", "scratch", "hole"}
    defects = df[df["label"] == 1]
    assert defects["mask_path"].notna().all()           # every defect has a mask
    assert df[df["split"] == "train"]["label"].eq(0).all()  # training data is good only


def test_calibration_split_is_stratified(dataset):
    test = list_images("synthetic_part", data_dir=dataset).query("split == 'test'")
    calib, evaluation = split_calibration_evaluation(test)
    assert len(calib) + len(evaluation) == len(test)
    assert set(calib["path"]).isdisjoint(evaluation["path"])
    for defect in ["good", "scratch", "hole"]:
        assert (calib["defect_type"] == defect).any() and (evaluation["defect_type"] == defect).any()


def test_masks_are_binary(dataset):
    df = list_images("synthetic_part", data_dir=dataset)
    mask = load_mask(df[df["label"] == 1]["mask_path"].iloc[0])
    assert set(np.unique(mask)) <= {0, 1} and mask.sum() > 0
    assert load_mask(None).sum() == 0


def test_coreset_and_nearest_distance():
    torch.manual_seed(0)
    points = torch.randn(500, 32)
    keep = greedy_coreset(points, 0.1, torch.device("cpu"))
    assert len(keep) == 50 and len(set(keep.tolist())) == 50
    # A point that IS in the bank has distance ~0 to it
    assert nearest_distance(points[keep[:3]], points[keep]).max() < 1e-4


def test_threshold_prefers_catching_defects():
    labels = [0, 0, 0, 0, 1, 1]
    scores = [1.0, 1.1, 1.2, 2.5, 2.0, 3.0]
    threshold, cost = best_threshold(labels, scores)
    summary = decisions_summary(labels, scores, threshold)
    assert summary["defects_missed"] == 0          # missing a defect costs 20x
    assert summary["good_rejected"] == 1           # accept one false reject instead
    assert cost == 1


def test_patchcore_end_to_end(dataset):
    df = list_images("synthetic_part", data_dir=dataset)
    train = df[df["split"] == "train"]["path"].tolist()[:8]
    test = df[df["split"] == "test"]
    model = PatchCore(device=torch.device("cpu"), coreset_ratio=0.05).fit(train, log=lambda *_: None)
    maps, scores = model.predict(test["path"].tolist())
    assert maps.shape == (len(test), 256, 256)
    assert np.isfinite(scores).all()
    assert 0.0 <= image_auroc(test["label"], scores) <= 1.0
