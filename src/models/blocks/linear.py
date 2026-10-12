# SPDX-FileCopyrightText: Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-FileContributor: Yann Bouquet
# SPDX-License-Identifier: MIT

from typing import Optional, Sequence
from torch.nn import Linear, ModuleList

from .base import NamedLeaf, BaseModuleWrapper, NamedBlock

class NamedLinearLayer(NamedLeaf):
    def __init__(self, *, name: str, module: Linear, parent: Optional[BaseModuleWrapper]):
        assert isinstance(module, Linear), "Module must be an instance of nn.Linear but got {}".format(type(module))
        super().__init__(name=name, module=module, parent=parent)
    
class NamedLinearLayerList(NamedBlock):
    def __init__(self, *, name: str, named_layers: Sequence[NamedLinearLayer], parent: Optional[BaseModuleWrapper] = None):
       self.name = name
       self.module : ModuleList = ModuleList([layer.module for layer in named_layers])
       self.parent = parent
       assert all(m.is_layer for m in named_layers), "ModuleList must contain only layers"
       self._children = named_layers
    
    def clone(self) -> "NamedLinearLayerList":
        return type(self)(
            name=self.name,
            named_layers= [layer.clone() for layer in self._children],
            parent=self.parent
        )
