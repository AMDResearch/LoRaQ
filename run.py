# SPDX-FileCopyrightText: Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-FileContributor: Yann Bouquet
# SPDX-License-Identifier: Apache-2.0

import logging
import os
import random
import warnings

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, open_dict

import src.utils.hydra_utils  # noqa: F401
from src.utils.gpu_utils import set_device
from src.utils.log_utils import setup_logging

os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("diffusers").setLevel(logging.ERROR)


@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg: DictConfig) -> None:
    with open_dict(cfg):
        cfg.hydra_run_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir

    setup_logging(cfg.hydra_run_dir)
    set_device(cfg)

    seed = cfg.get("seed", 0)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    cfg.script(cfg)


if __name__ == "__main__":
    main()
