# SPDX-FileCopyrightText: Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-FileContributor: Yann Bouquet
# SPDX-License-Identifier: MIT

from typing import Any, Dict, List, Tuple, Union

from torch import Tensor, nn
from torch.utils.hooks import RemovableHandle

Args = Tuple[Any, ...]
Kwargs = Dict[str, Any]
Outputs = Union[Tuple[Tensor], Tensor]


class RemovableHandles(RemovableHandle):
    """A group of hook handles removed together."""

    def __init__(self):
        self.handles: List[RemovableHandle] = []

    def append(self, handle: RemovableHandle) -> None:
        self.handles.append(handle)

    def remove(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


class GenericHook:
    """Something that can be attached to a linear layer as forward pre/post hooks (nothing by default)."""

    def _as_pre_hook(self, module: nn.Module, inputs: Args) -> Union[Args, None]:
        pass

    def _as_post_hook(self, module: nn.Module, inputs: Args, outputs: Outputs) -> Union[Tensor, None]:
        pass

    def hooked_on(self, module: nn.Module) -> RemovableHandles:
        return RemovableHandles()
