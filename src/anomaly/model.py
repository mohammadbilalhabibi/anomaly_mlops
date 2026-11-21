"""PatchCore as torch modules.

`PatchCoreModel` maps a normalized image batch to (image scores, anomaly maps).
The whole forward pass — backbone, patch aggregation, nearest-neighbour search
against the memory bank, upsampling and Gaussian smoothing — is one module, so
evaluation in PyTorch and the exported ONNX graph compute exactly the same thing.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
import torchvision
from torch import nn

_RESNET_BLOCKS = ("layer1", "layer2", "layer3", "layer4")


class Backbone(nn.Module):
    """ResNet trunk truncated after the deepest requested layer (layer4/fc are dropped)."""

    def __init__(self, net: nn.Module, layers: tuple[str, ...]) -> None:
        super().__init__()
        unknown = set(layers) - set(_RESNET_BLOCKS)
        if unknown or not layers:
            raise ValueError(f"layers must be a non-empty subset of {_RESNET_BLOCKS}")
        self.stem = nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool)
        last = max(_RESNET_BLOCKS.index(name) for name in layers)
        self.blocks = nn.ModuleDict({n: getattr(net, n) for n in _RESNET_BLOCKS[: last + 1]})
        self.layers = tuple(sorted(layers, key=_RESNET_BLOCKS.index))

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Return the feature maps of the requested layers, shallow to deep."""
        x = self.stem(x)
        out = []
        for name, block in self.blocks.items():
            x = block(x)
            if name in self.layers:
                out.append(x)
        return out


def build_backbone(name: str, layers: tuple[str, ...], pretrained: bool = True) -> Backbone:
    """Build a frozen torchvision ResNet backbone (ImageNet weights if `pretrained`)."""
    models = torchvision.models
    if name == "wide_resnet50_2":
        weights = models.Wide_ResNet50_2_Weights.IMAGENET1K_V1 if pretrained else None
        net = models.wide_resnet50_2(weights=weights)
    elif name == "resnet18":
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        net = models.resnet18(weights=weights)
    else:
        raise ValueError(f"unsupported backbone: {name}")
    backbone = Backbone(net, layers).eval()
    for p in backbone.parameters():
        p.requires_grad_(False)
    return backbone


def adaptive_pool_matrix(in_dim: int, out_dim: int) -> torch.Tensor:
    """Matrix M with `x @ M == F.adaptive_avg_pool1d(x, out_dim)` along the last axis.

    A plain MatMul exports cleanly to ONNX, unlike adaptive pooling with
    non-divisible sizes (1536 -> 1024 channels for WideResNet-50 layer2+3).
    """
    m = torch.zeros(in_dim, out_dim)
    for j in range(out_dim):
        start = (j * in_dim) // out_dim
        end = -(-((j + 1) * in_dim) // out_dim)
        m[start:end, j] = 1.0 / (end - start)
    return m


class PatchEmbedder(nn.Module):
    """Locally aware patch features (PatchCore sec. 3.1).

    Each layer's map is average-pooled over a pool_size x pool_size neighbourhood,
    deeper maps are upsampled to the finest grid, channels are concatenated and
    reduced to `embed_dim` by adaptive average pooling.
    """

    def __init__(self, backbone: Backbone, embed_dim: int, pool_size: int, input_size: int) -> None:
        super().__init__()
        self.backbone = backbone
        self.pool = nn.AvgPool2d(pool_size, stride=1, padding=pool_size // 2)
        with torch.no_grad():
            feats = backbone(torch.zeros(1, 3, input_size, input_size))
        self.grid: tuple[int, int] = (int(feats[0].shape[-2]), int(feats[0].shape[-1]))
        in_dim = sum(int(f.shape[1]) for f in feats)
        self.register_buffer("projection", adaptive_pool_matrix(in_dim, embed_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return patch embeddings of shape (batch, grid_h * grid_w, embed_dim)."""
        feats = [self.pool(f) for f in self.backbone(x)]
        maps = [feats[0]] + [
            F.interpolate(f, size=self.grid, mode="bilinear", align_corners=False)
            for f in feats[1:]
        ]
        z = torch.cat(maps, dim=1).flatten(2).transpose(1, 2)
        return z @ self.projection


def gaussian_kernel1d(sigma: float) -> torch.Tensor:
    """Normalized 1-D Gaussian kernel truncated at 4 sigma (like scipy.ndimage)."""
    radius = int(4.0 * sigma + 0.5)
    x = torch.arange(-radius, radius + 1, dtype=torch.float32)
    k = torch.exp(-0.5 * (x / sigma) ** 2)
    return k / k.sum()


class PatchScorer(nn.Module):
    """Nearest-neighbour patch scoring against a memory bank, plus map post-processing."""

    def __init__(
        self,
        memory_bank: torch.Tensor,
        grid: tuple[int, int],
        output_size: int,
        blur_sigma: float,
    ) -> None:
        super().__init__()
        bank = memory_bank.float().contiguous()
        self.register_buffer("bank_t", bank.t().contiguous())
        self.register_buffer("bank_sq", (bank * bank).sum(dim=1))
        self.grid = grid
        self.output_size = output_size
        kernel = gaussian_kernel1d(blur_sigma) if blur_sigma > 0 else torch.ones(1)
        self.pad = kernel.numel() // 2
        self.register_buffer("kernel_h", kernel.view(1, 1, 1, -1))
        self.register_buffer("kernel_v", kernel.view(1, 1, -1, 1))

    def patch_distances(self, patches: torch.Tensor) -> torch.Tensor:
        """Euclidean distance of each patch to its nearest memory-bank entry, (batch, n)."""
        d2 = (patches * patches).sum(-1, keepdim=True) - 2.0 * (patches @ self.bank_t)
        d2 = d2 + self.bank_sq
        return d2.min(dim=-1).values.clamp_min(0.0).sqrt()

    def forward(self, patches: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return image scores (batch,) and smoothed anomaly maps (batch, H, W)."""
        dist = self.patch_distances(patches)
        scores = dist.max(dim=-1).values
        maps = dist.reshape(-1, 1, self.grid[0], self.grid[1])
        maps = F.interpolate(
            maps, size=(self.output_size, self.output_size), mode="bilinear", align_corners=False
        )
        if self.pad > 0:
            maps = F.pad(maps, (self.pad, self.pad, 0, 0), mode="reflect")
            maps = F.conv2d(maps, self.kernel_h)
            maps = F.pad(maps, (0, 0, self.pad, self.pad), mode="reflect")
            maps = F.conv2d(maps, self.kernel_v)
        return scores, maps[:, 0]


class PatchCoreModel(nn.Module):
    """Full inference graph: image batch -> (scores, anomaly maps)."""

    def __init__(self, embedder: PatchEmbedder, scorer: PatchScorer) -> None:
        super().__init__()
        self.embedder = embedder
        self.scorer = scorer

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Score a normalized (batch, 3, H, W) image tensor."""
        return self.scorer(self.embedder(x))


def num_coreset(n: int, ratio: float) -> int:
    """Number of memory-bank entries kept for `n` patches at a coreset `ratio`."""
    return max(1, min(n, math.ceil(n * ratio)))
