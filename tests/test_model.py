"""Tests for the PatchCore building blocks and an end-to-end synthetic run."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
F = torch.nn.functional

from anomaly.coreset import greedy_coreset  # noqa: E402
from anomaly.data import list_test  # noqa: E402
from anomaly.metrics import image_auroc  # noqa: E402
from anomaly.model import (  # noqa: E402
    PatchScorer,
    adaptive_pool_matrix,
    gaussian_kernel1d,
    num_coreset,
)
from anomaly.patchcore import (  # noqa: E402
    iter_batches,
    load_model,
    predict,
    save_model,
)
from anomaly.preprocess import load_mask  # noqa: E402

from .conftest import MODEL, PREP  # noqa: E402


@pytest.mark.parametrize(("in_dim", "out_dim"), [(1536, 1024), (64, 64), (10, 3), (6, 12)])
def test_adaptive_pool_matrix_matches_torch(in_dim: int, out_dim: int) -> None:
    x = torch.randn(5, in_dim)
    expected = F.adaptive_avg_pool1d(x[:, None, :], out_dim)[:, 0]
    torch.testing.assert_close(x @ adaptive_pool_matrix(in_dim, out_dim), expected)


def test_gaussian_kernel_is_normalized_and_symmetric() -> None:
    k = gaussian_kernel1d(4.0)
    assert k.numel() == 33
    assert torch.isclose(k.sum(), torch.tensor(1.0))
    torch.testing.assert_close(k, k.flip(0))


def test_coreset_is_deterministic_and_covers_clusters() -> None:
    gen = torch.Generator().manual_seed(0)
    centers = torch.randn(5, 32, generator=gen) * 20
    feats = torch.cat([c + torch.randn(200, 32, generator=gen) for c in centers])
    idx = greedy_coreset(feats, 5, proj_dim=16, seed=3)
    assert torch.equal(idx, greedy_coreset(feats, 5, proj_dim=16, seed=3))
    assert len(set((idx // 200).tolist())) == 5  # one point from every cluster
    assert torch.equal(greedy_coreset(feats, 2000), torch.arange(1000))


def test_num_coreset_bounds() -> None:
    assert num_coreset(1000, 0.1) == 100
    assert num_coreset(5, 0.01) == 1
    assert num_coreset(5, 2.0) == 5


def test_scorer_zero_distance_for_bank_patch() -> None:
    bank = torch.randn(20, 8)
    scorer = PatchScorer(bank, grid=(2, 2), output_size=8, blur_sigma=0.0)
    patches = torch.stack([bank[3], bank[7], bank[3], bank[0]])[None] + 0.0
    dist = scorer.patch_distances(patches)
    torch.testing.assert_close(dist, torch.zeros(1, 4), atol=1e-3, rtol=0)
    far = patches.clone()
    far[0, 2] += 10.0
    scores, maps = scorer(far)
    assert scores.shape == (1,) and maps.shape == (1, 8, 8)
    assert scores[0] > 5.0
    # The anomalous patch sits bottom-left in the 2x2 grid.
    assert maps[0, 6, 1] > maps[0, 1, 6]


def test_end_to_end_synthetic_detection_and_localisation(fake_mvtec: Path, tiny_model) -> None:
    samples = list_test(fake_mvtec, "widget")
    scores, maps = predict(tiny_model, iter_batches([s.path for s in samples], 4, PREP))
    assert scores.shape == (8,) and maps.shape == (8, PREP.crop, PREP.crop)
    labels = np.array([s.label for s in samples])
    assert image_auroc(labels, scores) >= 0.9
    # The map's peak falls inside the defect for most defect images.
    hits = 0
    for s, m in zip(samples, maps, strict=True):
        if s.label == 1:
            mask = load_mask(s.mask_path, PREP.resize, PREP.crop)
            r, c = np.unravel_index(np.argmax(m), m.shape)
            hits += int(mask[max(r - 4, 0) : r + 5, max(c - 4, 0) : c + 5].any())
    assert hits >= 3


def test_save_and_load_roundtrip(tmp_path: Path, tiny_model, fake_mvtec: Path) -> None:
    path = tmp_path / "patchcore.pt"
    bank = tiny_model.scorer.bank_t.t()
    save_model(path, bank, MODEL, PREP)
    torch.manual_seed(0)  # the random backbone must match the fixture's
    loaded, model_params, prep = load_model(path, pretrained=False)
    assert model_params == MODEL and prep == PREP
    torch.testing.assert_close(loaded.scorer.bank_t, tiny_model.scorer.bank_t)
