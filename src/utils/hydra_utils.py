# SPDX-FileCopyrightText: Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-FileContributor: Yann Bouquet
# SPDX-License-Identifier: MIT

"""OmegaConf resolvers used by the configs. Importing this module registers them."""

import torch
from omegaconf import OmegaConf

import src.script as script

OmegaConf.register_new_resolver("torch_dtype", lambda x: getattr(torch, x, None))
OmegaConf.register_new_resolver("merge", lambda x, y: x + y)
OmegaConf.register_new_resolver("script", lambda x: getattr(script, x, None))
