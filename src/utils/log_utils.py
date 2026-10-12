# SPDX-FileCopyrightText: Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-FileContributor: Yann Bouquet
# SPDX-License-Identifier: MIT

import sys

import torch
from loguru import logger
from tqdm import tqdm

from .gpu_utils import CPU, GPUS


def setup_logging(root_directory: str):
    """INFO and above to stdout (main device only), DEBUG to `debug.log`, errors to `error.log`."""
    logger.remove()

    main_devices = set([GPUS[0], CPU])
    def std_sink(msg):
        if msg.record["extra"].get("device", torch.device("cpu")) in main_devices:
            if "tqdm" in msg.record["extra"]:
                tqdm.write(msg, end="")
            else:
                sys.stdout.write(msg)

    logger.add(std_sink, colorize=True, level="INFO")

    logger.add(
        f"{root_directory}/debug.log",
        colorize=False,
        level="DEBUG",
        rotation="1 MB",
    )

    logger.add(
        f"{root_directory}/error.log",
        colorize=False,
        level="ERROR",
    )
