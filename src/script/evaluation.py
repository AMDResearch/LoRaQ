# SPDX-FileCopyrightText: Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-FileContributor: Yann Bouquet
# SPDX-License-Identifier: MIT

import os

import torch
from hydra.utils import instantiate
from loguru import logger as log
from omegaconf import DictConfig, OmegaConf

from src.context.base import Context
from src.quantizer.base import InputQuantizer, WeightQuantizer
from src.utils.paths import fp16_dir, image_tag, quantized_dir, quantizer_name, stack_segments


@torch.inference_mode()
def main(cfg: DictConfig) -> None:
    """Score the quantized images against the full-precision references (and the dataset's
    ground-truth images for FID). Expects `operation=generation` to have run with the same config."""
    TAG = image_tag(cfg)
    DATASET = cfg.generation.dataset.name
    OUTPUT_DIR = cfg.output_root_dir

    FP16_IMAGES_OUTPUT_DIR = fp16_dir(OUTPUT_DIR, DATASET, TAG)

    i_quantizer: InputQuantizer = InputQuantizer(instantiate(cfg.input_quantizer))
    w_quantizer: WeightQuantizer = WeightQuantizer(instantiate(cfg.weight_quantizer))

    context_config = OmegaConf.to_container(cfg.context, resolve=True)
    shift_activations = cfg.model.get("shift_activations", False)
    context: Context = Context(context_config, shift_activations=shift_activations)

    quantized_output_dir = quantized_dir(
        OUTPUT_DIR,
        stack_segments(context.calibrators, shift_activations),
        quantizer_name(i_quantizer, w_quantizer),
        DATASET,
        TAG,
    )

    log.info(f"FP16 images: {FP16_IMAGES_OUTPUT_DIR}")
    log.info(f"Quantized images: {quantized_output_dir}")

    evaluator = instantiate(cfg.evaluation.evaluator)(
        reference_folder=FP16_IMAGES_OUTPUT_DIR,
        prediction_folder=quantized_output_dir,
    )
    results = evaluator()

    for directory in (cfg.hydra_run_dir, quantized_output_dir):
        evaluator.save_results(results, output_filename=os.path.join(directory, "evaluation_results"))
