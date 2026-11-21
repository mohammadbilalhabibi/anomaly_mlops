"""Greedy k-center coreset subsampling of the patch memory bank (PatchCore sec. 3.2)."""

from __future__ import annotations

import math

import torch


@torch.no_grad()
def greedy_coreset(
    features: torch.Tensor, n_select: int, proj_dim: int = 128, seed: int = 0
) -> torch.Tensor:
    """Select `n_select` row indices that greedily minimise the max distance to the coreset.

    Distances are computed after a fixed-seed Gaussian random projection to
    `proj_dim` dimensions (Johnson-Lindenstrauss), which makes selection fast
    on CPU. The start point is drawn with the same seed, so results repeat.
    """
    n, dim = features.shape
    if n_select >= n:
        return torch.arange(n)
    if n_select < 1:
        raise ValueError("n_select must be >= 1")
    gen = torch.Generator().manual_seed(seed)
    x = features.float()
    if 0 < proj_dim < dim:
        proj = torch.randn(dim, proj_dim, generator=gen) / math.sqrt(proj_dim)
        x = x @ proj
    sq = (x * x).sum(dim=1)
    selected = torch.empty(n_select, dtype=torch.long)
    current = int(torch.randint(n, (1,), generator=gen))
    min_d2 = torch.full((n,), float("inf"))
    for i in range(n_select):
        selected[i] = current
        c = x[current]
        d2 = sq - 2.0 * (x @ c) + c.dot(c)
        torch.minimum(min_d2, d2, out=min_d2)
        current = int(torch.argmax(min_d2))
    return selected
