# SPDX-FileCopyrightText: Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-FileContributor: Yann Bouquet
# SPDX-License-Identifier: MIT

"""Gradient descent on the Stiefel manifold (matrices with orthonormal rows) via the Cayley transform."""

import random

import torch
from torch.optim import Optimizer

_EPS = 1e-8


def _qr_orthonormalize(x: torch.Tensor) -> torch.Tensor:
    """Re-orthonormalize the rows of x (QR with the sign convention diag(R) > 0)."""
    q, r = torch.linalg.qr(x.t())
    q *= torch.diag(r, 0).sign().expand_as(q)
    return q.t()


def _cayley_retraction(x: torch.Tensor, skew: torch.Tensor, direction: torch.Tensor, step) -> torch.Tensor:
    """Fixed-point approximation of the Cayley transform Y = (I - s/2 W)^-1 (I + s/2 W) X."""
    y = x + step * direction
    for _ in range(5):
        y = x + step * torch.matmul(skew, 0.5 * (x + y))
    return y.t()


class CayleySGD(Optimizer):
    """SGD for parameters constrained to have orthonormal rows (e.g. rotation matrices).

    The Euclidean gradient is projected to a skew-symmetric matrix W and the parameter is
    moved along the manifold with a Cayley retraction; the step is capped by 1 / ||W||_1.
    About once every 101 steps the parameter is re-orthonormalized by QR to remove the
    numerical drift.

    This is the optimizer used by SpinQuant to learn rotations
    (Liu et al., "SpinQuant: LLM Quantization with Learned Rotations", 2024).
    # TODO(citation): add the final reference.
    """

    def __init__(self, params, lr: float):
        super().__init__(params, dict(lr=lr))

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                rows = p.size(0)
                x = p.view(rows, -1)
                x = x / x.norm(p=2, dim=1, keepdim=True).add(_EPS)
                if x.size(0) > x.size(1):
                    raise ValueError(f"CayleySGD needs at most as many rows as columns, got {tuple(x.shape)}")
                if random.randint(1, 101) == 1:
                    x = _qr_orthonormalize(x)

                grad = p.grad.view(rows, -1)
                direction = torch.zeros(grad.t().size(), device=grad.device) - grad.t()

                mx = torch.mm(direction, x)
                xmx = torch.mm(x, mx)
                xxmx = torch.mm(x.t(), xmx)
                w_hat = mx - 0.5 * xxmx
                skew = w_hat - w_hat.t()

                max_step = 1.0 / (skew.abs().sum(dim=0).max() + _EPS)
                step = min(max_step, group["lr"])

                p.copy_(_cayley_retraction(x.t(), skew, direction, step).view(p.size()))
        return loss
