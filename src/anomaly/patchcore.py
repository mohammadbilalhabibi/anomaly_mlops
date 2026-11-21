"""Building, fitting, saving and loading PatchCore models."""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Sequence
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from anomaly.config import ModelParams, PreprocessParams
from anomaly.coreset import greedy_coreset
from anomaly.model import (
    PatchCoreModel,
    PatchEmbedder,
    PatchScorer,
    build_backbone,
    num_coreset,
)
from anomaly.preprocess import open_image, preprocess_image


def iter_batches(
    paths: Sequence[Path], batch_size: int, prep: PreprocessParams
) -> Iterator[np.ndarray]:
    """Yield preprocessed float32 batches of shape (b, 3, crop, crop)."""
    for start in range(0, len(paths), batch_size):
        chunk = paths[start : start + batch_size]
        yield np.stack([preprocess_image(open_image(p), prep.resize, prep.crop) for p in chunk])


def build_embedder(
    model: ModelParams, prep: PreprocessParams, pretrained: bool = True
) -> PatchEmbedder:
    """Create the frozen patch embedder (backbone + aggregation + projection)."""
    backbone = build_backbone(model.backbone, model.layers, pretrained=pretrained)
    return PatchEmbedder(backbone, model.embed_dim, model.pool_size, prep.crop).eval()


@torch.no_grad()
def extract_patches(embedder: PatchEmbedder, batches: Iterable[np.ndarray]) -> torch.Tensor:
    """Embed all batches and stack their patches into one (n_patches, dim) tensor."""
    chunks = [
        embedder(torch.from_numpy(b)).reshape(-1, embedder.projection.shape[1]) for b in batches
    ]
    return torch.cat(chunks, dim=0)


def fit_memory_bank(
    embedder: PatchEmbedder,
    batches: Iterable[np.ndarray],
    coreset_ratio: float,
    proj_dim: int,
    seed: int,
) -> tuple[torch.Tensor, int]:
    """Build the coreset memory bank from good images; returns (bank, n_patches_before)."""
    patches = extract_patches(embedder, batches)
    n_select = num_coreset(patches.shape[0], coreset_ratio)
    idx = greedy_coreset(patches, n_select, proj_dim=proj_dim, seed=seed)
    return patches[idx].clone(), int(patches.shape[0])


def assemble(
    embedder: PatchEmbedder, memory_bank: torch.Tensor, model: ModelParams, crop: int
) -> PatchCoreModel:
    """Combine an embedder and a memory bank into the full inference model."""
    scorer = PatchScorer(memory_bank, embedder.grid, crop, model.blur_sigma)
    return PatchCoreModel(embedder, scorer).eval()


@torch.no_grad()
def predict(model: PatchCoreModel, batches: Iterable[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Run the model over batches; returns (scores (n,), maps (n, H, W)) as numpy."""
    scores, maps = [], []
    for b in batches:
        s, m = model(torch.from_numpy(b))
        scores.append(s.numpy())
        maps.append(m.numpy())
    return np.concatenate(scores), np.concatenate(maps)


def save_model(
    path: Path, memory_bank: torch.Tensor, model: ModelParams, prep: PreprocessParams
) -> None:
    """Save the memory bank and the parameters needed to rebuild the model."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "memory_bank": memory_bank.contiguous(),
            "model": asdict(model),
            "preprocess": asdict(prep),
        },
        path,
    )


def load_model(
    path: Path, pretrained: bool = True
) -> tuple[PatchCoreModel, ModelParams, PreprocessParams]:
    """Rebuild a saved PatchCore model (backbone weights come from torchvision)."""
    state = torch.load(path, map_location="cpu", weights_only=True)
    model_params = ModelParams(**{**state["model"], "layers": tuple(state["model"]["layers"])})
    prep = PreprocessParams(**state["preprocess"])
    embedder = build_embedder(model_params, prep, pretrained=pretrained)
    return assemble(embedder, state["memory_bank"], model_params, prep.crop), model_params, prep
