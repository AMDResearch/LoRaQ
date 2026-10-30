# SPDX-FileCopyrightText: Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-FileContributor: Yann Bouquet
# SPDX-License-Identifier: Apache-2.0

import torch

import torch.nn as nn

class Conv1x1ToLinear(nn.Module):
    def __init__(self, conv1x1: nn.Conv2d):
        super().__init__()
        assert isinstance(conv1x1, nn.Conv2d), "Input must be nn.Conv2d"
        assert conv1x1.kernel_size == (1, 1), "Conv2d must be 1x1"
        dtype = conv1x1.weight.dtype
        device = conv1x1.weight.device
        
        self.in_channels = conv1x1.in_channels
        self.out_channels = conv1x1.out_channels
        self.linear = nn.Linear(self.in_channels, self.out_channels, bias=conv1x1.bias is not None, dtype=dtype, device=device)
        with torch.no_grad():
            self.linear.weight.copy_(conv1x1.weight.squeeze(-1).squeeze(-1))
            if conv1x1.bias is not None:
                self.linear.bias.copy_(conv1x1.bias)
        self.to(device=device, dtype=dtype)
        
    def forward(self, x):
        b, c, h, w = x.shape
        x = x.permute(0, 2, 3, 1).view(b, h*w, c)
        out = self.linear(x)
        out = out.view(b, h, w, self.out_channels).permute(0, 3, 1, 2)
        return out
