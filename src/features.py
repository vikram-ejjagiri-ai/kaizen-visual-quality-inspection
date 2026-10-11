"""Feature extractor: a pretrained CNN that turns an image into a grid of patch descriptions.

We never train this network. It was pretrained on ImageNet and already "knows"
edges, textures and shapes. PatchCore only compares its descriptions of new
parts with descriptions of good parts.
"""
import torch
import torch.nn.functional as F
from torch import nn
from torchvision.models import Wide_ResNet50_2_Weights, wide_resnet50_2

from src.config import RANDOM_STATE, SMOKE_TEST


class PatchFeatureExtractor(nn.Module):
    """Wide-ResNet-50 layer2 + layer3 features, locally averaged, at 1/8 resolution.

    Input : (B, 3, 256, 256) normalized images
    Output: (B, 1536, 32, 32) patch features (one 1536-number description per 8x8 patch)
    """

    def __init__(self):
        super().__init__()
        weights = None if SMOKE_TEST else Wide_ResNet50_2_Weights.IMAGENET1K_V1
        if SMOKE_TEST:
            torch.manual_seed(RANDOM_STATE)   # same random weights every time
        net = wide_resnet50_2(weights=weights)
        self.stem = nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool, net.layer1)
        self.layer2 = net.layer2
        self.layer3 = net.layer3
        # Average over a 3x3 neighbourhood: each patch also "sees" its surroundings
        self.local_avg = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)
        self.eval()

    @torch.no_grad()
    def forward(self, x):
        f2 = self.layer2(self.stem(x))     # mid-level: edges, small textures
        f3 = self.layer3(f2)               # higher level: parts, shapes
        f2 = self.local_avg(f2)
        f3 = self.local_avg(f3)
        f3 = F.interpolate(f3, size=f2.shape[-2:], mode="bilinear", align_corners=False)
        return torch.cat([f2, f3], dim=1)


class GlobalFeatureExtractor(nn.Module):
    """Baseline: one description per IMAGE (average of layer3), no spatial detail."""

    def __init__(self, patch_extractor):
        super().__init__()
        self.patch_extractor = patch_extractor

    @torch.no_grad()
    def forward(self, x):
        return self.patch_extractor(x).mean(dim=(2, 3))


def best_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
