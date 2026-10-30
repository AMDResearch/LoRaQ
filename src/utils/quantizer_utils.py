# SPDX-FileCopyrightText: Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-FileContributor: Yann Bouquet
# SPDX-License-Identifier: Apache-2.0

import re
from typing import Sequence, Tuple

import torch


def _svd(input: torch.Tensor, rank: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    u, s, vt = torch.linalg.svd(input, full_matrices=False)
    return u[:, :rank], s[:rank], vt[:rank]


def low_rank_approximations(w: torch.Tensor, rank=256) -> Tuple[torch.Tensor, torch.Tensor]:
    """Truncated SVD W ~ L R with the singular values split evenly: L = U sqrt(S), R = sqrt(S) V^T."""
    u, s, vt = _svd(w.to(torch.float64), rank)
    ss = torch.diag(torch.sqrt(s))
    return u @ ss, ss @ vt


def to_be_quantized(layer_name: str, layers: Sequence[str]) -> bool:
    """Whether `layer_name` fully matches one of the regexes in `layers`."""
    return any(re.fullmatch(pattern, layer_name) for pattern in layers)
