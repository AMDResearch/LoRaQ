# SPDX-FileCopyrightText: Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-FileContributor: Yann Bouquet
# SPDX-License-Identifier: Apache-2.0

from hydra.utils import instantiate
from loguru import logger as log
from omegaconf import DictConfig, OmegaConf

from src.calibrator import DiTCalibrator
from src.context.base import Context

from ..utils.gpu_utils import GPUS


def main(cfg: DictConfig) -> None:
    """Calibrate one stage (`cfg.calibration.calibrator`) on top of the stages in `cfg.context`."""
    pipeline_config = OmegaConf.to_container(cfg.model.wrapper, resolve=True)
    calibrator_config = OmegaConf.to_container(cfg.calibration.calibrator, resolve=True)

    log.info("Instantiating calibrator")
    calibrator: DiTCalibrator = instantiate(calibrator_config)

    log.info("Instantiating model pipeline")
    models = []
    for device in GPUS:
        model = instantiate(pipeline_config)
        models.append(model.to(device))

    context_cfg = OmegaConf.to_container(cfg.context, resolve=True)
    context = Context(context_cfg, shift_activations=cfg.model.get("shift_activations", False))

    log.info(f"Root output directory: {cfg.output_root_dir}")
    with context(models, output_dir=cfg.output_root_dir) as ctx:
        calibrator.output_dir = ctx.output_dir
        log.info(f"Calibrating into {calibrator.output_dir}")
        calibrator.calibrate(models)
