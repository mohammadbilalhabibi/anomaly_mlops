"""Export PatchCore to ONNX and quantize the backbone to INT8."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from pathlib import Path

import numpy as np
import torch

from anomaly.model import PatchCoreModel

OPSET = 18


def export_onnx(model: PatchCoreModel, path: Path, crop: int) -> None:
    """Export the full inference graph (image -> score, anomaly_map) with a dynamic batch axis.

    Uses the torch.export-based exporter (the default since PyTorch 2.9). The
    example batch has 2 images because torch.export specialises size-1 dims.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    example = torch.zeros(2, 3, crop, crop)
    torch.onnx.export(
        model.eval(),
        (example,),
        str(path),
        input_names=["image"],
        output_names=["score", "anomaly_map"],
        dynamic_shapes=({0: torch.export.Dim("batch")},),
        opset_version=OPSET,
        dynamo=True,
        external_data=False,
        verbose=False,
    )


class _ArrayReader:
    """ONNX Runtime calibration reader over a list of preprocessed single-image batches."""

    def __init__(self, batches: Sequence[np.ndarray]) -> None:
        self._it: Iterator[np.ndarray] = iter(batches)

    def get_next(self) -> dict[str, np.ndarray] | None:
        batch = next(self._it, None)
        return None if batch is None else {"image": batch}


def scorer_nodes(path: Path) -> list[str]:
    """Names of nodes that use the scorer's parameters (memory bank, blur kernels)."""
    import onnx

    graph = onnx.load(str(path), load_external_data=False).graph
    return [n.name for n in graph.node if any(i.startswith("scorer.") for i in n.input)]


def quantize_int8(fp32_path: Path, int8_path: Path, calib_batches: Sequence[np.ndarray]) -> None:
    """Static INT8 (QDQ) quantization of the backbone convolutions only.

    Everything after the backbone stays in fp32: quantizing the memory-bank
    MatMul would destroy the small distance differences that separate good
    from defective patches, and quantizing the Gaussian-blur convolutions
    would round the final anomaly map to 256 levels (measured: bottle pixel
    AUROC fell from 98.3% to 77.0%).
    """
    from onnxruntime.quantization import (
        CalibrationMethod,
        QuantFormat,
        QuantType,
        quantize_static,
    )
    from onnxruntime.quantization.shape_inference import quant_pre_process

    prepped = int8_path.with_name(int8_path.stem + "_prep.onnx")
    # Symbolic shape inference fails on torch.export graphs; ONNX shape inference suffices.
    quant_pre_process(str(fp32_path), str(prepped), skip_symbolic_shape=True)
    try:
        quantize_static(
            str(prepped),
            str(int8_path),
            _ArrayReader(calib_batches),
            quant_format=QuantFormat.QDQ,
            op_types_to_quantize=["Conv"],
            nodes_to_exclude=scorer_nodes(prepped),
            per_channel=True,
            activation_type=QuantType.QUInt8,
            weight_type=QuantType.QInt8,
            calibrate_method=CalibrationMethod.MinMax,
        )
    finally:
        prepped.unlink(missing_ok=True)
