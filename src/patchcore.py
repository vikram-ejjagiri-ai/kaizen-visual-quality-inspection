"""PatchCore anomaly detection (Roth et al., "Towards Total Recall in Industrial
Anomaly Detection", CVPR 2022), implemented from scratch so every step is visible.

Idea in one sentence: remember what every small patch of a GOOD part looks like;
a patch of a new part that is far from everything remembered is an anomaly.

1. fit():     describe every patch of every good training image -> memory bank
              keep a representative 10% subset (coreset) so search stays fast
2. predict(): for each patch of a new image, distance to its nearest neighbour
              in the memory bank = how unusual it is -> anomaly map
              image score = the most unusual patch (maximum of the map)
"""
import math

import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter
from torch.utils.data import DataLoader

from src.config import CORESET_RATIO, HEATMAP_SIGMA, IMAGE_SIZE, RANDOM_STATE
from src.data import ImageFolderDataset
from src.features import GlobalFeatureExtractor, PatchFeatureExtractor, best_device


def _flatten(features):
    """(B, C, H, W) -> (B*H*W, C)"""
    b, c, h, w = features.shape
    return features.permute(0, 2, 3, 1).reshape(-1, c), (h, w)


def greedy_coreset(features, ratio, device, projection_dim=128, seed=RANDOM_STATE):
    """Pick a subset of patches that covers the whole set as well as possible.

    Greedy k-center: repeatedly add the patch that is FARTHEST from everything
    already picked. A random projection to 128 dimensions makes this fast while
    keeping distances roughly intact (Johnson-Lindenstrauss).
    """
    n = features.shape[0]
    k = max(1, int(math.ceil(n * ratio)))
    gen = torch.Generator(device="cpu").manual_seed(seed)
    projection = torch.randn(features.shape[1], projection_dim, generator=gen) / math.sqrt(projection_dim)
    reduced = (features @ projection).to(device)

    selected = [int(torch.randint(n, (1,), generator=gen))]
    min_dist = torch.cdist(reduced, reduced[selected[-1]].unsqueeze(0)).squeeze(1)
    for _ in range(k - 1):
        idx = int(torch.argmax(min_dist))
        selected.append(idx)
        dist = torch.cdist(reduced, reduced[idx].unsqueeze(0)).squeeze(1)
        min_dist = torch.minimum(min_dist, dist)
    return torch.tensor(selected)


def nearest_distance(queries, bank, chunk=4096):
    """Distance from each query row to its nearest row in the bank (chunked to save memory)."""
    out = []
    for start in range(0, queries.shape[0], chunk):
        out.append(torch.cdist(queries[start:start + chunk], bank).min(dim=1).values)
    return torch.cat(out)


class PatchCore:
    def __init__(self, device=None, coreset_ratio=CORESET_RATIO):
        self.device = device or best_device()
        self.coreset_ratio = coreset_ratio
        self.extractor = PatchFeatureExtractor().to(self.device)
        self.memory_bank = None
        self.feature_shape = None

    @torch.no_grad()
    def _features(self, batch):
        return self.extractor(batch.to(self.device)).cpu()

    def fit(self, train_paths, batch_size=8, log=print):
        loader = DataLoader(ImageFolderDataset(train_paths), batch_size=batch_size)
        all_patches = []
        for batch in loader:
            flat, self.feature_shape = _flatten(self._features(batch))
            all_patches.append(flat)
        patches = torch.cat(all_patches)
        log(f"  {len(train_paths)} good images -> {patches.shape[0]:,} patch descriptions "
            f"of {patches.shape[1]} numbers each")
        keep = greedy_coreset(patches, self.coreset_ratio, self.device)
        self.memory_bank = patches[keep].to(self.device)
        log(f"  Coreset keeps {self.memory_bank.shape[0]:,} patches ({self.coreset_ratio:.0%})")
        return self

    @torch.no_grad()
    def predict_batch(self, batch):
        """Return (anomaly_maps (B, 256, 256), image_scores (B,))."""
        features = self.extractor(batch.to(self.device))
        b = features.shape[0]
        flat, (h, w) = _flatten(features)
        dist = nearest_distance(flat, self.memory_bank).reshape(b, 1, h, w)
        maps = F.interpolate(dist, size=(IMAGE_SIZE, IMAGE_SIZE), mode="bilinear",
                             align_corners=False).squeeze(1).cpu().numpy()
        maps = np.stack([gaussian_filter(m, sigma=HEATMAP_SIGMA) for m in maps])
        return maps, maps.reshape(b, -1).max(axis=1)

    def predict(self, paths, batch_size=8):
        loader = DataLoader(ImageFolderDataset(paths), batch_size=batch_size)
        maps, scores = [], []
        for batch in loader:
            m, s = self.predict_batch(batch)
            maps.append(m)
            scores.append(s)
        return np.concatenate(maps), np.concatenate(scores)

    def save(self, path):
        torch.save({"memory_bank": self.memory_bank.cpu(), "feature_shape": self.feature_shape,
                    "coreset_ratio": self.coreset_ratio}, path)

    @classmethod
    def load(cls, path, device=None):
        state = torch.load(path, map_location="cpu")
        model = cls(device=device, coreset_ratio=state["coreset_ratio"])
        model.memory_bank = state["memory_bank"].to(model.device)
        model.feature_shape = state["feature_shape"]
        return model


class GlobalKNNBaseline:
    """Simple baseline: one feature vector per image, score = distance to the nearest
    good training image. It can say "this part looks unusual" but cannot point to WHERE."""

    def __init__(self, patchcore):
        self.device = patchcore.device
        self.extractor = GlobalFeatureExtractor(patchcore.extractor).to(self.device)
        self.bank = None

    @torch.no_grad()
    def _embed(self, paths, batch_size=8):
        loader = DataLoader(ImageFolderDataset(paths), batch_size=batch_size)
        return torch.cat([self.extractor(b.to(self.device)).cpu() for b in loader])

    def fit(self, train_paths):
        self.bank = self._embed(train_paths).to(self.device)
        return self

    def predict(self, paths):
        return nearest_distance(self._embed(paths).to(self.device), self.bank).cpu().numpy()
