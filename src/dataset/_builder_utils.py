# SPDX-FileCopyrightText: Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-FileContributor: Yann Bouquet
# SPDX-License-Identifier: Apache-2.0

"""Helpers shared by the dataset builder scripts (imported relatively, so that Hugging Face
`datasets` copies this file together with the scripts)."""

import hashlib
import random
from typing import List, TypeVar

T = TypeVar("T")


def sha256_seed(text: str) -> int:
    """Noise seed of an image: the first 4 bytes of SHA-256(text) as a signed 32-bit integer.

    Seeding every image from its name makes the comparison between methods paired.
    """
    return int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:4], "big", signed=True)


def rolling_hash_seed(text: str) -> int:
    """Polynomial rolling hash of a prompt, modulo 1e9 + 7."""
    modulus = 10**9 + 7
    value = 0
    for char in text:
        value = (value * 31 + ord(char)) % modulus
    return value


def shuffled_subset(items: List[T], size: int) -> List[T]:
    """A fixed pseudo-random subset of `size` items (all items, in order, if size <= 0)."""
    if size <= 0:
        return items
    items = list(items)
    random.Random(0).shuffle(items)
    return items[:size]
