# SPDX-FileCopyrightText: Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-FileContributor: Yann Bouquet
# SPDX-License-Identifier: MIT

"""Image-quality metrics, each returning a {name: value} dict.

LPIPS and PSNR compare the quantized images with the full-precision ones, IR (ImageReward)
scores both against the prompt, FID compares the quantized images with the dataset images.
"""

import os

import numpy as np
from torchmetrics.image import LearnedPerceptualImagePatchSimilarity, PeakSignalNoiseRatio
from torchmetrics.metric import Metric

from src.metrics.cleanfid import CleanFrechetInceptionDistance


class DictTorchMetric(Metric):
    """Wraps a torchmetrics metric updated with (reference, prediction) image batches."""

    def __init__(self, name: str, metric: Metric):
        super().__init__()
        self.name = name
        self._metric = metric

    def to(self, device):
        self._metric = self._metric.to(device)
        return self

    @property
    def metric(self) -> Metric:
        return self._metric

    def update(self, *args, **kwargs):
        ref, pred, *_ = args
        self._metric.update(ref, pred, **kwargs)

    def reset(self, *args, **kwargs):
        self._metric.reset(*args, **kwargs)

    def compute(self, *args, **kwargs):
        return {self.name: self._metric.compute(*args, **kwargs).cpu().item()}


class FIDDictMetric(DictTorchMetric):
    """Updated with the FID-resized predictions (fake) and dataset images (real)."""

    def update(self, *args, **kwargs):
        _, _, _, resized_pred, resized_gt, *_ = args
        self._metric.update(resized_pred, real=False)
        self._metric.update(resized_gt, real=True)


class IRDictMetric(DictTorchMetric):
    """Mean ImageReward of the quantized ("pred") and full-precision ("ref") images."""

    def __init__(self, name: str, metric, score_ref: bool = True):
        super().__init__(name, metric)
        self._score_ref = score_ref
        self._ref_results = []
        self._pred_results = []

    def update(self, *args, **kwargs):
        *_, prompts, ref_paths, pred_paths = args
        for prompt, ref_path, pred_path in zip(prompts, ref_paths, pred_paths):
            self._pred_results.append(self._metric.score(prompt, pred_path))
            if self._score_ref:
                self._ref_results.append(self._metric.score(prompt, ref_path))

    def reset(self, *args, **kwargs):
        self._ref_results = []
        self._pred_results = []

    def compute(self, *args, **kwargs):
        return {
            self.name: {
                "ref": np.mean(self._ref_results).item() if self._ref_results else 0.0,
                "pred": np.mean(self._pred_results).item() if self._pred_results else 0.0,
            }
        }


def LPIPS():
    return DictTorchMetric("LPIPS", LearnedPerceptualImagePatchSimilarity(normalize=True))


def PSNR():
    return DictTorchMetric("PSNR", PeakSignalNoiseRatio(data_range=(0, 1), reduction="elementwise_mean", dim=(1, 2, 3)))


def FID():
    return FIDDictMetric("FID", CleanFrechetInceptionDistance())


def _resolve_imagereward_cache_dir() -> str:
    candidates = []

    custom_cache = os.environ.get("IMAGEREWARD_CACHE")
    if custom_cache:
        candidates.append(custom_cache)

    xdg_cache_home = os.environ.get("XDG_CACHE_HOME")
    if xdg_cache_home:
        candidates.append(os.path.join(xdg_cache_home, "ImageReward"))

    home_dir = os.path.expanduser("~")
    if home_dir and home_dir != "~":
        candidates.append(os.path.join(home_dir, ".cache", "ImageReward"))

    candidates.append(os.path.join(os.getcwd(), ".cache", "ImageReward"))

    for cache_dir in candidates:
        try:
            os.makedirs(cache_dir, exist_ok=True)
            test_file = os.path.join(cache_dir, ".write_test")
            with open(test_file, "w", encoding="utf-8") as f:
                f.write("ok")
            os.remove(test_file)
            return cache_dir
        except OSError:
            continue

    return ""


def IR():
    import ImageReward as RM

    return IRDictMetric("IR", RM.load("ImageReward-v1.0", download_root=_resolve_imagereward_cache_dir() or None))
