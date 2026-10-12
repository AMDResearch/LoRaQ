# SPDX-FileCopyrightText: Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-FileContributor: Yann Bouquet
# SPDX-License-Identifier: MIT

"""Randomized Hadamard matrices of power-of-two size (Sylvester construction), used to
initialize the rotation of the low-rank branch."""

import torch


def sylvester_hadamard(n: int, dtype: torch.dtype = torch.float64) -> torch.Tensor:
    """The n x n Hadamard matrix with +-1 entries, built as H_2k = [[H_k, H_k], [H_k, -H_k]]."""
    if n < 1 or n & (n - 1):
        raise ValueError(f"Hadamard size must be a power of two, got {n}")
    h = torch.ones(1, 1, dtype=dtype)
    while h.size(0) < n:
        h = torch.cat((torch.cat((h, h), dim=1), torch.cat((h, -h), dim=1)), dim=0)
    return h


def _scale(n: int) -> torch.Tensor:
    return torch.tensor(n).sqrt()


def random_hadamard_matrix(n: int, device=None) -> torch.Tensor:
    """Randomized orthogonal Hadamard matrix diag(s) H / sqrt(n) with random signs s (float64).

    The signs are drawn from the default CPU generator.
    """
    signs = torch.randint(low=0, high=2, size=(n,)).to(torch.float64) * 2 - 1
    return (signs[:, None] * sylvester_hadamard(n) / _scale(n)).to(device)
