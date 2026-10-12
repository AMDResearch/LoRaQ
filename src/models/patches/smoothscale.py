# SPDX-FileCopyrightText: Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-FileContributor: Yann Bouquet
# SPDX-License-Identifier: MIT

"""Activation/weight smoothing (SmoothQuant; search space of SVDQuant)."""

from typing import Union

import torch
from torch import Tensor, nn

try:
    from typing import override
except ImportError:
    from typing_extensions import override

from src.utils.hook import Args, GenericHook, RemovableHandles


def smooth_scale(a_span: Tensor, w_span: Tensor, alpha: float, beta: float) -> Tensor:
    """Per-input-channel scale s = a_span^alpha / w_span^beta (just w_span^-beta when alpha = 0).

    `a_span` and `w_span` are the per-channel maxima of |x| and |W|. Channels where s is 0 keep a
    scale of 1, and a non-finite scale falls back to the identity.
    """
    if not (0.0 <= alpha <= 1.0 and 0.0 <= beta <= 1.0):
        raise ValueError(f"alpha and beta must be in [0, 1], got ({alpha}, {beta})")
    if alpha > 0.0:
        scale = a_span.pow(alpha)
        if beta > 0.0:
            scale = scale.div(w_span.pow(beta))
    else:
        scale = w_span.pow(-beta)
    scale[scale == 0.0] = 1.0
    if not torch.isfinite(scale).all():
        scale = torch.ones_like(scale)
    return scale


class SmoothScale(GenericHook):
    """Divides the inputs of a linear layer by s and multiplies its weights by s (per input
    channel), which leaves the full-precision output unchanged. The scale is computed in float64."""

    scale_dtype = torch.float64

    def __init__(self, a_span: Tensor, w_span: Tensor, alpha: float, beta: float):
        self.alpha = alpha
        self.beta = beta
        self.w_smooth_scale = smooth_scale(a_span.to(self.scale_dtype), w_span.to(self.scale_dtype), alpha, beta)

    @override
    @torch.no_grad()
    def _as_pre_hook(self, module: nn.Module, inputs: Args) -> Union[Args, None]:
        mat, *rest = inputs
        shape = [1] * (len(mat.shape) - 1) + [-1]
        return mat.div(self.w_smooth_scale.view(shape).to(device=mat.device).to(mat.dtype)), *rest

    def scale_weights(self, weights: Tensor) -> Tensor:
        shape = [1] * (len(weights.shape) - 1) + [-1]
        return weights.to(self.scale_dtype).mul(self.w_smooth_scale.view(shape).to(weights.device)).to(weights.dtype)

    @override
    def hooked_on(self, module: nn.Module) -> RemovableHandles:
        """Scales the weights in place (not undone when the hook is removed) and divides the inputs."""
        assert isinstance(module.weight.data, Tensor), "The module's weight should be a Tensor."
        module.weight.data.copy_(self.scale_weights(module.weight.data))
        handles = RemovableHandles()
        handles.append(module.register_forward_pre_hook(self._as_pre_hook))
        return handles
