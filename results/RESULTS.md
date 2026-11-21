# Measured results

**Headline (15 categories):** mean image AUROC 98.6%, pixel AUROC 97.9%, AUPRO 90.2%. The served INT8 ONNX model keeps image AUROC 98.5% / AUPRO 90.0% and passes the quality gate in 15/15 categories. INT8 inference takes 70–259 ms median per image on a laptop CPU, depending on memory-bank size.

### Detection and localisation (MVTec AD test set, %)

| Category | Image AUROC | Pixel AUROC | AUPRO (FPR≤0.3) | INT8 Image AUROC | INT8 Pixel AUROC | INT8 AUPRO | INT8 gate |
|---|---:|---:|---:|---:|---:|---:|:---:|
| bottle | 100.0 | 98.3 | 92.5 | 100.0 | 98.2 | 91.8 | pass |
| cable | 99.6 | 98.6 | 93.3 | 99.6 | 98.5 | 93.2 | pass |
| capsule | 95.8 | 98.7 | 91.1 | 95.3 | 98.6 | 90.3 | pass |
| carpet | 99.1 | 98.8 | 91.9 | 99.0 | 98.8 | 92.0 | pass |
| grid | 98.4 | 97.7 | 89.4 | 98.4 | 97.6 | 89.0 | pass |
| hazelnut | 100.0 | 98.4 | 86.3 | 100.0 | 98.4 | 86.2 | pass |
| leather | 100.0 | 99.0 | 95.8 | 100.0 | 99.1 | 95.7 | pass |
| metal_nut | 99.7 | 98.5 | 91.5 | 99.7 | 98.5 | 91.3 | pass |
| pill | 94.2 | 97.6 | 91.3 | 94.4 | 97.6 | 91.2 | pass |
| screw | 95.2 | 98.8 | 93.0 | 95.1 | 98.7 | 92.0 | pass |
| tile | 98.4 | 95.3 | 78.8 | 98.4 | 95.5 | 79.2 | pass |
| toothbrush | 99.2 | 98.7 | 87.2 | 98.9 | 98.6 | 86.9 | pass |
| transistor | 100.0 | 97.5 | 94.8 | 100.0 | 97.5 | 94.7 | pass |
| wood | 99.3 | 93.7 | 83.5 | 99.1 | 93.9 | 83.5 | pass |
| zipper | 99.5 | 98.3 | 92.8 | 99.4 | 98.3 | 92.8 | pass |
| **mean (15)** | **98.6** | **97.9** | **90.2** | **98.5** | **97.8** | **90.0** | |

PyTorch columns = reference model. ONNX fp32 differs from it by at most 3e-05 points on any metric across 15 categories. INT8 gate: every metric within the `max_metric_drop` set in `params.yaml`.

### Decisions at the label-free threshold (%)

| Category | Test images (defect) | Precision | Recall | False-positive rate |
|---|---:|---:|---:|---:|
| bottle | 83 (63) | 98.4 | 100.0 | 5.0 |
| cable | 150 (92) | 98.9 | 94.6 | 1.7 |
| capsule | 132 (109) | 98.6 | 62.4 | 4.3 |
| carpet | 117 (89) | 90.7 | 98.9 | 32.1 |
| grid | 78 (57) | 98.2 | 94.7 | 4.8 |
| hazelnut | 110 (70) | 100.0 | 100.0 | 0.0 |
| leather | 124 (92) | 94.8 | 100.0 | 15.6 |
| metal_nut | 115 (93) | 97.9 | 100.0 | 9.1 |
| pill | 167 (141) | 100.0 | 57.4 | 0.0 |
| screw | 160 (119) | 98.5 | 53.8 | 2.4 |
| tile | 117 (84) | 98.8 | 97.6 | 3.0 |
| toothbrush | 42 (30) | 93.5 | 96.7 | 16.7 |
| transistor | 100 (40) | 93.0 | 100.0 | 5.0 |
| wood | 79 (60) | 87.0 | 100.0 | 47.4 |
| zipper | 151 (119) | 99.1 | 97.5 | 3.1 |

### CPU latency per image

Measured on 11th Gen Intel(R) Core(TM) i7-1165G7 @ 2.80GHz (8 logical cores, no GPU), batch size 1, 50 runs after 5 warm-up runs, torch 2.15.0+cpu, onnxruntime 1.32.0. Times in ms per image (median / p95).

| Category | Memory bank | Preprocess | PyTorch fp32 | ONNX fp32 | ONNX INT8 |
|---|---:|---:|---:|---:|---:|
| bottle | 14,740 | 18.8 / 21.0 | 258.4 / 293.2 | 212.5 / 246.4 | 166.1 / 236.6 |
| cable | 15,837 | 34.5 / 41.7 | 266.5 / 294.6 | 208.9 / 251.9 | 157.5 / 188.8 |
| capsule | 15,445 | 31.9 / 37.9 | 265.9 / 296.4 | 230.7 / 285.4 | 155.2 / 184.9 |
| carpet | 19,757 | 30.7 / 37.1 | 298.1 / 316.2 | 241.4 / 290.6 | 187.6 / 219.4 |
| grid | 18,660 | 16.2 / 19.1 | 315.7 / 397.7 | 229.0 / 277.5 | 184.1 / 207.5 |
| hazelnut | 27,597 | 30.3 / 34.4 | 373.8 / 457.1 | 299.3 / 395.3 | 259.3 / 331.5 |
| leather | 17,327 | 29.8 / 33.7 | 294.5 / 325.0 | 232.5 / 303.0 | 168.8 / 187.6 |
| metal_nut | 15,524 | 14.8 / 18.3 | 278.3 / 312.9 | 207.9 / 247.5 | 182.0 / 219.2 |
| pill | 18,816 | 19.2 / 21.9 | 299.2 / 347.7 | 227.9 / 282.3 | 208.5 / 245.1 |
| screw | 22,580 | 16.3 / 20.2 | 382.8 / 575.4 | 257.5 / 307.4 | 207.8 / 230.1 |
| tile | 16,229 | 23.1 / 27.1 | 279.5 / 308.2 | 220.1 / 290.9 | 158.9 / 182.0 |
| toothbrush | 4,234 | 30.0 / 34.2 | 167.2 / 194.1 | 111.5 / 151.3 | 70.1 / 81.7 |
| transistor | 15,053 | 36.9 / 41.0 | 264.8 / 307.4 | 202.8 / 256.0 | 151.0 / 177.7 |
| wood | 17,405 | 39.0 / 43.8 | 298.7 / 322.7 | 216.9 / 267.4 | 172.9 / 198.3 |
| zipper | 16,935 | 14.8 / 17.5 | 275.1 / 311.9 | 216.1 / 260.3 | 171.6 / 219.3 |

End-to-end HTTP `POST /api/predict` (bottle, ONNX int8, 50 requests on localhost, incl. upload, decode, heatmap PNG encoding): median 214.2 ms, p95 255.2 ms.

### ONNX export

| Category | ONNX fp32 MB | ONNX INT8 MB | fp32 max rel. score diff | INT8 max rel. score diff |
|---|---:|---:|---:|---:|
| bottle | 166.4 | 92.2 | 2.6e-06 | 8.1e-02 |
| cable | 170.9 | 96.7 | 1.6e-06 | 4.0e-02 |
| capsule | 169.3 | 95.1 | 1.0e-05 | 2.0e-01 |
| carpet | 187.0 | 112.8 | 4.6e-06 | 4.0e-02 |
| grid | 182.5 | 108.3 | 6.5e-06 | 9.7e-03 |
| hazelnut | 219.1 | 145.0 | 2.1e-06 | 8.2e-02 |
| leather | 177.0 | 102.9 | 4.0e-06 | 2.5e-02 |
| metal_nut | 169.6 | 95.5 | 1.3e-06 | 1.5e-02 |
| pill | 183.1 | 109.0 | 2.5e-06 | 2.4e-02 |
| screw | 198.5 | 124.4 | 5.8e-06 | 4.3e-02 |
| tile | 172.5 | 98.4 | 1.4e-06 | 3.0e-02 |
| toothbrush | 123.3 | 49.2 | 3.5e-06 | 4.7e-02 |
| transistor | 167.7 | 93.5 | 2.8e-06 | 2.6e-02 |
| wood | 177.3 | 103.2 | 1.3e-06 | 6.5e-02 |
| zipper | 175.4 | 101.2 | 5.0e-06 | 2.9e-02 |

### What INT8 quantization should not touch

Pixel AUROC / AUPRO (%) on the test set for three ONNX variants:

| Category | fp32 | INT8, blur convs also quantized | INT8, backbone only (served) |
|---|---:|---:|---:|
| bottle | 98.3 / 92.5 | 77.0 / 32.6 | 98.2 / 91.8 |
| metal_nut | 98.5 / 91.5 | 82.8 / 43.6 | 98.5 / 91.3 |
| screw | 98.8 / 93.0 | 85.5 / 51.3 | 98.7 / 92.0 |
