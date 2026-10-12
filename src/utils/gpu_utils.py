# SPDX-FileCopyrightText: Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-FileContributor: Yann Bouquet
# SPDX-License-Identifier: MIT

import torch
from loguru import logger as log
from omegaconf import DictConfig

GPUS = (
    [torch.device(f"cuda:{i}") for i in range(torch.cuda.device_count())]
    if torch.cuda.is_available()
    else []
)

CPU = torch.device('cpu')


def set_device(cfg: DictConfig) -> None:
    """
    Set the CUDA device based on the configuration.
    """

    device = cfg.get("device", 0)
    if torch.cuda.is_available():
        if device < torch.cuda.device_count():
            torch.cuda.set_device(device)
            log.info(f"CUDA device set to: {device}")
        else:
            log.warning(
                f"Device {device} not available. Available devices: {torch.cuda.device_count()}. Using device 0."
            )
            torch.cuda.set_device(0)
    else:
        log.warning("CUDA is not available, using CPU.")
                
