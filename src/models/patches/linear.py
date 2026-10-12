# SPDX-FileCopyrightText: Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-FileContributor: Yann Bouquet
# SPDX-License-Identifier: MIT

"""Linear layers rewritten so that their inputs can be quantized separately (used by the model wrappers)."""

import torch
import torch.nn as nn

__all__ = ["ConcatLinear", "ShiftedLinear"]


class ConcatLinear(nn.Module):
    """A linear layer over concatenated inputs, split into one linear per input chunk.

    y = sum_i x_i W_i^T + b, with x = [x_1, ..., x_k] split along the features. Used when a
    fused projection reads the concatenation of differently distributed activations (e.g.
    attention and MLP outputs in FLUX single-stream blocks), so that each chunk gets its own
    input quantization. The bias is carried by the last linear.
    """

    def __init__(self, in_features_list: list[int], out_features: int, bias: bool = True, device=None, dtype=None):
        super().__init__()
        if len(in_features_list) < 2:
            raise ValueError("ConcatLinear needs at least 2 input chunks")
        self.in_features_list = list(in_features_list)
        self.in_features = sum(in_features_list)
        self.out_features = out_features
        last = len(in_features_list) - 1
        self.linears = nn.ModuleList(
            nn.Linear(n, out_features, bias and i == last, device, dtype) for i, n in enumerate(in_features_list)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        chunks = x.split(self.in_features_list, dim=-1)
        return sum(linear(chunk.contiguous()) for linear, chunk in zip(self.linears, chunks, strict=True))

    @staticmethod
    def from_linear(linear: nn.Linear, splits: list[int]) -> "ConcatLinear":
        """Split `linear` along its input features; `splits` are the leading chunk sizes."""
        sizes = [n for n in [*splits, linear.in_features - sum(splits)] if n > 0]
        concat = ConcatLinear(
            in_features_list=sizes,
            out_features=linear.out_features,
            bias=linear.bias is not None,
            device=linear.weight.device,
            dtype=linear.weight.dtype,
        )
        start = 0
        for sub in concat.linears:
            sub.weight.data.copy_(linear.weight[:, start : start + sub.in_features])
            start += sub.in_features
        if linear.bias is not None:
            concat.linears[-1].bias.data.copy_(linear.bias)
        return concat


class ShiftedLinear(nn.Module):
    """y = (x + shift) W^T + b', with b' = b - shift W^T, i.e. the same output as the original layer.

    Shifting makes inputs such as GELU outputs non-negative, so they can use unsigned formats;
    `linear.shifted` tells the input quantizer to do so.
    """

    shift: torch.Tensor

    def __init__(self, in_features: int, out_features: int, shift: float | torch.Tensor, bias: bool = True,
                 device=None, dtype=None):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias, device, dtype)
        self.linear.shifted = True  # type: ignore[attr-defined]
        device, dtype = self.linear.weight.device, self.linear.weight.dtype
        shift = torch.as_tensor(shift, device=device, dtype=dtype).flatten()
        if shift.numel() > 1:
            if in_features % shift.numel():
                raise ValueError(f"{in_features} input features cannot be split in {shift.numel()} shift groups")
            shift = shift.view(-1, 1).expand(-1, in_features // shift.numel()).flatten()
        self.register_buffer("shift", shift)

    @property
    def in_features(self) -> int:
        return self.linear.in_features

    @property
    def out_features(self) -> int:
        return self.linear.out_features

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self.linear(input + self.shift.view([1] * (input.dim() - 1) + [-1]))

    @staticmethod
    def from_linear(linear: nn.Linear, shift: float | torch.Tensor) -> "ShiftedLinear":
        dtype = linear.weight.dtype
        shifted = ShiftedLinear(linear.in_features, linear.out_features, shift, bias=True,
                                device=linear.weight.device, dtype=dtype)
        shifted.linear.weight.data.copy_(linear.weight)
        weight, shift = linear.weight.double(), shifted.shift.double()
        correction = weight.sum(dim=1) * shift if shift.numel() == 1 else weight @ shift
        correction = correction.view(shifted.linear.bias.size())
        if linear.bias is not None:
            shifted.linear.bias.data.copy_((linear.bias.data.double() - correction).to(dtype))
        else:
            shifted.linear.bias.data.copy_(correction.to(dtype).neg_())
        return shifted
