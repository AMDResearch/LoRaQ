# SPDX-FileCopyrightText: Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-FileContributor: Yann Bouquet
# SPDX-License-Identifier: Apache-2.0

import math
import os
from typing import Iterable, Tuple

import datasets
from loguru import logger as log

BUILDERS = {
    "mjhq": "mjhq.py",
    "sdci": "sdci.py",
    "coco_captains_2024": "coco_captains_2024.py",
}


def get_dataset(
    name: str,
    split: str = "train",
    max_dataset_size: int = 10000,
    return_gt: bool = False,
    batch_size: int = 1,
    seed_offset: int = 0,
) -> Tuple[Iterable, int, int]:
    """Load a prompt dataset; returns (batch iterator, number of samples, number of batches).

    `max_dataset_size > 0` keeps a fixed random subset (the same for every run). `seed_offset`
    shifts the per-image noise seeds (MJHQ only); 0 reproduces the published results.
    """
    if name not in BUILDERS:
        raise ValueError(f"Unknown dataset name: {name} (available: {sorted(BUILDERS)})")
    batch_size = max(1, batch_size)
    kwargs = dict(name=name, split=split, trust_remote_code=True, max_dataset_size=max_dataset_size, return_gt=return_gt)
    if name == "mjhq":
        kwargs["seed_offset"] = seed_offset
    elif seed_offset:
        raise ValueError(f"seed_offset={seed_offset} requested but only 'mjhq' implements it.")

    path = os.path.join(os.path.dirname(__file__), BUILDERS[name])
    log.info(f"Loading dataset {name} from {path} with split {split}...")
    dataset = datasets.load_dataset(path, **kwargs)
    if not isinstance(dataset, datasets.Dataset):
        raise TypeError(f"Expected dataset to be of type datasets.Dataset, but got {type(dataset)}")
    total = len(dataset)
    return dataset.iter(batch_size=batch_size), total, math.ceil(total / batch_size)
