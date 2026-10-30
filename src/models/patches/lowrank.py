# SPDX-FileCopyrightText: Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-FileContributor: Yann Bouquet
# SPDX-License-Identifier: Apache-2.0

import torch
from torch import Tensor, nn

try:
    from typing import override
except ImportError:
    from typing_extensions import override

from src.utils.hook import Args, GenericHook, Outputs, RemovableHandles


class LowRankBranch(nn.Module, GenericHook):
    """Auxiliary branch x -> x R^T L^T added to the output of a linear layer (W ~ residual + L R)."""

    def __init__(
        self,
        l_matrix: Tensor,
        r_matrix: Tensor,
        device: torch.device = torch.device("cpu"),
        dtype=torch.float32,
    ):
        super().__init__()
        self.to(device=device, dtype=dtype)
        assert l_matrix.shape[1] == r_matrix.shape[0], (
            "Inconsistent shapes for low-rank matrices."
        )

        self.a = nn.Linear(
            r_matrix.shape[1],
            r_matrix.shape[0],
            bias=False,
            device=device,
            dtype=dtype,
        )
        self.a.weight.data.copy_(r_matrix.to(device=device, dtype=dtype))
        
        self.b = nn.Linear(
            l_matrix.shape[1],
            l_matrix.shape[0],
            bias=False,
            device=device,
            dtype=dtype,
        )
        
        self.b.weight.data.copy_(l_matrix.to(device=device, dtype=dtype))        
        self.cached_result: Tensor | None = None

    @classmethod
    @torch.no_grad()
    def from_svd(cls, weight: Tensor, rank: int, device: torch.device = torch.device("cpu"), dtype=torch.float32) -> "LowRankBranch":
        """Rank-`rank` truncated SVD of `weight` (in float64): L = U S, R = V^T."""
        u, s, vt = torch.linalg.svd(weight.double(), full_matrices=False)
        l_mat, r_mat = u[:, :rank] @ torch.diag(s[:rank]), vt[:rank]
        return cls(l_mat.to(device=device, dtype=dtype), r_mat.to(device=device, dtype=dtype), device=device, dtype=dtype)

    @property 
    def weight(self) -> Tensor:
        return self.b.weight @ self.a.weight
    
    def forward(self, x: Tensor) -> Tensor:
        return self.b(self.a(x))
    
    @override
    def _as_pre_hook(self, module: nn.Module, inputs: Args) -> None:
        assert self.cached_result is None, (
            "Cached result should be None before pre-hook."
        )
        self.cached_result = self(inputs[0])

    @override
    def _as_post_hook(self, module: nn.Module, inputs: Args, outputs: Outputs) -> Tensor:
        assert self.cached_result is not None, (
            "Cached result should be defined in post-hook."
        )
        if isinstance(outputs, tuple):
            outputs, *_ = outputs
        result = outputs + self.cached_result
        self.cached_result = None
        return result
    
    @override
    def hooked_on(self, module: nn.Module) -> RemovableHandles:
        """The branch reads the layer input (after the transforms of the previous stages) and its
        output is added first among the post-hooks."""
        handles = RemovableHandles()
        handles.append(module.register_forward_pre_hook(self._as_pre_hook, prepend=False))
        handles.append(module.register_forward_hook(self._as_post_hook, prepend=True))
        return handles
