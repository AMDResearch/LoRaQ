# Third-party notices

The code in this repository is released under the MIT license (see `LICENSE`). It uses or
refers to the following third-party software and data, which keep their own licenses.

## Software

- **TensorCast** (https://github.com/ROCm/tensorcast), MIT License, Copyright (c) 2024 AMD ROCm Software.
  Included as the git submodule `third-parties/tcast` (pinned at commit
  `56c51c955f07d71eb1b42157733d173f4b20cbc0`) and used unmodified;
  `src/utils/tcast_shadow.py` adjusts two of its behaviours at import time.
- Python dependencies listed in `requirements.txt`, under their respective licenses.

## Data

- **Calibration prompts** (`src/dataset/coco_captains_2024_prompts.yaml`): 1024 captions from
  MS-COCO Captions (annotations under CC BY 4.0, https://cocodataset.org), in the selection used
  by DeepCompressor (https://github.com/nunchaku-tech/deepcompressor, Apache License 2.0,
  `examples/diffusion/prompts/qdiff.yaml`).

## Downloaded at run time (not redistributed)

| Asset | Used for | License |
|---|---|---|
| PixArt-Σ (`PixArt-alpha/PixArt-Sigma-XL-2-1024-MS`) | model | CreativeML Open RAIL++-M |
| SANA 1600M (`Lawrence-cj/Sana_1600M_1024px_BF16_diffusers_ch5632`, BF16 conversion of `Efficient-Large-Model/Sana_1600M_1024px_diffusers`) | model (`model=sana`) | Apache License 2.0 (official checkpoint) |
| FLUX.1 [schnell] (`black-forest-labs/FLUX.1-schnell`) | model (`model=flux_schnell`) | Apache License 2.0 |
| FLUX.1 [dev] (`black-forest-labs/FLUX.1-dev`) | model (`model=flux_dev`) | FLUX.1 [dev] Non-Commercial License |
| MJHQ-30K (`playgroundai/MJHQ-30K`) | evaluation prompts and FID reference images | Playground v2.5 Community License |
| sDCI (`mit-han-lab/svdquant-datasets`, from DCI / SA-1B) | evaluation prompts and FID reference images | CC BY-NC 4.0 |
| ImageReward (`THUDM/ImageReward`) | IR metric | Apache License 2.0 |
| LPIPS weights (via torchmetrics) | LPIPS metric | BSD 2-Clause |
