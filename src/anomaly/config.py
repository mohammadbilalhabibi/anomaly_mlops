"""Pipeline parameters (params.yaml) and serving settings (.env)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class PreprocessParams:
    """Image geometry shared by training and serving."""

    resize: int = 256
    crop: int = 224


@dataclass(frozen=True)
class ModelParams:
    """PatchCore architecture parameters."""

    backbone: str = "wide_resnet50_2"
    layers: tuple[str, ...] = ("layer2", "layer3")
    embed_dim: int = 1024
    pool_size: int = 3
    blur_sigma: float = 4.0


@dataclass(frozen=True)
class TrainParams:
    """Memory-bank construction and threshold calibration parameters."""

    coreset_ratio: float = 0.1
    coreset_proj_dim: int = 128
    calib_fraction: float = 0.1
    threshold_percentile: float = 99.0
    batch_size: int = 8
    seed: int = 0


@dataclass(frozen=True)
class EvaluateParams:
    """Evaluation parameters."""

    batch_size: int = 4
    pro_fpr_limit: float = 0.3
    # Quality gate: ONNX variants may lose at most this much (absolute) on any metric.
    max_metric_drop: float = 0.02


@dataclass(frozen=True)
class BenchmarkParams:
    """Latency benchmark parameters."""

    warmup: int = 5
    runs: int = 50


@dataclass(frozen=True)
class Params:
    """All pipeline parameters, loaded from params.yaml."""

    data_root: Path = Path("data/mvtec_ad")
    categories: tuple[str, ...] = ("bottle",)
    preprocess: PreprocessParams = field(default_factory=PreprocessParams)
    model: ModelParams = field(default_factory=ModelParams)
    train: TrainParams = field(default_factory=TrainParams)
    evaluate: EvaluateParams = field(default_factory=EvaluateParams)
    benchmark: BenchmarkParams = field(default_factory=BenchmarkParams)


def load_params(path: Path | str = PROJECT_ROOT / "params.yaml") -> Params:
    """Load params.yaml into typed dataclasses (relative paths resolve to the project root)."""
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    data = raw.get("data", {})
    root = Path(data.get("root", "data/mvtec_ad"))
    if not root.is_absolute():
        root = PROJECT_ROOT / root
    model = dict(raw.get("model", {}))
    if "layers" in model:
        model["layers"] = tuple(model["layers"])
    return Params(
        data_root=root,
        categories=tuple(data.get("categories", ("bottle",))),
        preprocess=PreprocessParams(**raw.get("preprocess", {})),
        model=ModelParams(**model),
        train=TrainParams(**raw.get("train", {})),
        evaluate=EvaluateParams(**raw.get("evaluate", {})),
        benchmark=BenchmarkParams(**raw.get("benchmark", {})),
    )


class ServeSettings(BaseSettings):
    """Serving settings, read from environment variables or .env."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore", protected_namespaces=())

    models_dir: Path = PROJECT_ROOT / "models"
    model_variant: str = "int8"
    samples_dir: Path = PROJECT_ROOT / "data" / "mvtec_ad"
    ort_threads: int = 0
    samples_per_category: int = 6
