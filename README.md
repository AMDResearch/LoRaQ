# LoRaQ

Code for **LoRaQ: Optimized Low Rank Approximation for 4-bit Quantization** (NeurIPS 2026).

Yann Bouquet, Alireza Khodamoradi, Sophie Yáng Shen, Kristof Denolf, Mathieu Salzmann  
EPFL · AMD

[Paper](https://arxiv.org/abs/2604.18117)

This repository reproduces the PixArt-Σ experiments of the paper: 4-bit post-training quantization
of every linear layer of the transformer blocks, evaluated on MJHQ-30K and sDCI.

- **LoRaQ-A** (`loraq_a`): W4A4 MXFP4 (`mxfp4e2`), rank-128 low-rank branch in MXFP4.
- **LoRaQ-B** (`loraq_b`): weights MXFP4, activations MXFP8 (`mxfp8e4`), rank-64 low-rank branch
  in MXFP8, whose input is quantized to MXFP8 as well.

## Method

Two calibration stages, the second on top of the first:

1. **Smoothing** (`smoothing`, `config/strategy/smoothing.yaml`): for each group of layers sharing
   an input, a per-channel scale s = |x|^α / |W|^β moves the activation outliers to the weights.
   (α, β) is searched on a grid by minimizing the quantized output error on calibration inputs,
   with the weights quantized as in SVDQuant (rank-32 SVD branch + quantized residual).
2. **Low-rank** (`lowrank`, `config/strategy/loraq.yaml`): a low-rank branch L R is added to every
   layer, W = (W − L R) + L R, with the residual and the factors both quantized. L and R start from
   the truncated SVD of the smoothed weight and are optimized in weight space (no calibration
   data): Adam on the quantization error of the residual, then a rotation (L r, rᵀ R) learned on
   the Stiefel manifold so that L and R quantize well.

The calibration inputs of the smoothing stage are the inputs of the transformer blocks collected
while generating 128 MS-COCO captions with the full-precision model.

## Installation

Python 3.12 and a CUDA GPU are required (tested on one NVIDIA A100 80GB).

```bash
git clone --recurse-submodules <this repository> loraq && cd loraq
pip install torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

PixArt-Σ, MJHQ-30K and sDCI are downloaded from the Hugging Face Hub on first use (set `HF_HOME` to
choose the cache location; the MJHQ-30K images take a few GB).

**TensorCast** (https://github.com/ROCm/tensorcast) is used for all quantization. It is not a pip
package: it lives in `third-parties/tcast`, pinned at commit `56c51c9`, and is imported through
`src/utils/tcast_shadow.py`. If the folder is empty (e.g. cloned without `--recurse-submodules`):

```bash
git submodule update --init third-parties/tcast
# or, without git submodules:
git clone https://github.com/ROCm/tensorcast third-parties/tcast && git -C third-parties/tcast checkout 56c51c9
```

## Reproducing the paper (PixArt-Σ)

For LoRaQ-A (replace `loraq_a` by `loraq_b` for LoRaQ-B):

```bash
# 1. Calibration, one stage at a time; +context lists the stages already calibrated
python run.py +experiment=loraq_a +context=empty operation=calibration +calibration=smoothing
python run.py +experiment=loraq_a +context=smoothing operation=calibration +calibration=lowrank
# 2. Images of the full-precision and of the quantized model (5000 MJHQ-30K prompts)
python run.py +experiment=loraq_a +context=smoothing_lowrank operation=generation dataset.max_dataset_size=5000
# 3. Metrics
python run.py +experiment=loraq_a +context=smoothing_lowrank operation=evaluation dataset.max_dataset_size=5000
```

Approximate times on one A100: smoothing 30–45 min, low-rank 6–9 min, and about 4 h per set of 5000
images (full-precision, then quantized). Calibration and generation resume where they stopped.

For sDCI, add `dataset=sdci` to the generation and evaluation commands, and
`evaluation.evaluator.batch_size=1` to the evaluation (its reference images have various sizes).

### Results

PixArt-Σ, 5000 images (from the paper):

| | MJHQ FID ↓ | MJHQ IR ↑ | MJHQ LPIPS ↓ | MJHQ PSNR ↑ | sDCI FID ↓ | sDCI IR ↑ | sDCI LPIPS ↓ | sDCI PSNR ↑ |
|---|---|---|---|---|---|---|---|---|
| FP16 | 16.6 | 0.948 | – | – | 24.8 | 0.966 | – | – |
| LoRaQ-A (W4A4) | 17.5 | 0.952 | 0.381 | 16.4 | 23.7 | 0.970 | 0.399 | 15.0 |
| LoRaQ-B (W4A8) | 16.2 | 0.939 | 0.327 | 17.0 | 22.5 | 0.980 | 0.349 | 15.8 |

Expect small deviations across GPUs and software versions.

### Outputs

All artefacts go to `output_root_dir` (default `data/pixart_sigma`); each stage nests a directory
named after its settings inside the previous one:

```
data/pixart_sigma/
├── samples.pt                                         calibration inputs
├── mjhq/fp16/eval/*.png                               full-precision images
└── Smoothing_input_..._weight_LowRankQuantizer_.../    spans.pkl, checkpoint/errors.pkl, result/best_candidates.pkl
    └── LoRaQ_128_mxfp4e2_mxfp4e2/                     results.pkl, svd_cache_rank_256.pkl
        └── input_BaseQuantizer_mxfp4e2_weight_BaseQuantizer_mxfp4e2/mjhq/eval/
            ├── *.png                                  quantized images
            └── evaluation_results.{csv,md}
```

Logs and the resolved configuration of each run are in `outputs/<model>/<dtype>/<operation>/<date>/<time>/`.

### Reproducibility notes

- Each image is generated from its own seed (a hash of the prompt name), so all methods see the same
  noise. Images also depend on the batch composition: keep `dataset.batch_size` (8) unchanged.
  `dataset.seed_offset=<n>` draws a different set of seeds (MJHQ only).

## Other models

The experiment configurations are provided for PixArt-Σ. Model configurations are also provided for
SANA (`model=sana`), FLUX.1-schnell (`model=flux_schnell`) and FLUX.1-dev (`model=flux_dev`): to
quantize them, write an experiment configuration like `config/experiment/loraq_a.yaml` with
`- override /model: <name>`. For FLUX, use `- override /calibrator@smoothing_calibrator: smoothing`
(layers that read the same input in its parallel blocks share one smoothing scale) and the
`model.input_layers_calibration` / `model.weight_layers_calibration` layer lists. The FLUX.1
repositories are gated on the Hugging Face Hub: accept their license and run
`huggingface-cli login` first.

## Citation

```bibtex
@inproceedings{bouquet2026loraq,
  title     = {LoRaQ: Optimized Low Rank Approximation for 4-bit Quantization},
  author    = {Bouquet, Yann and Khodamoradi, Alireza and Shen, Sophie Y{\'a}ng and Denolf, Kristof and Salzmann, Mathieu},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year      = {2026}
}
```

## License

MIT, Copyright (c) 2026 Advanced Micro Devices, Inc. (see `LICENSE`). Code written by Yann Bouquet.
Third-party software, data and models keep their own licenses: see `THIRD_PARTY_NOTICES.md`.
