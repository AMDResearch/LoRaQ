# SPDX-FileCopyrightText: Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-FileContributor: Yann Bouquet
# SPDX-License-Identifier: Apache-2.0

import copy
from typing import Optional, Type, Sequence, Generator, Mapping, List, Tuple
from abc import ABC, abstractmethod

import torch
from torch import nn


def join_name(*args: str, sep: str = ".") -> str:
    """Join names with a separator."""
    return sep.join(filter(None, args))


class BaseModuleWrapper(ABC):
    def __init__(
        self, *, name: str, module: nn.Module, parent: Optional["BaseModuleWrapper"]
    ):
        self.name: str = name
        self.module: nn.Module = module
        self.parent: Optional["BaseModuleWrapper"] = parent

    @property
    @abstractmethod
    def named_layers(self) -> Generator["BaseModuleWrapper", None, None]: ...

    @property
    def is_layer(self) -> bool:
        return False

    @property
    def transformer_block(self) -> Optional["BaseModuleWrapper"]:
        return self.parent.transformer_block if self.parent else None

    @property
    def attention_block(self) -> Optional["BaseModuleWrapper"]:
        return self.parent.attention_block if self.parent else None

    @property
    def feed_forward_block(self) -> Optional["BaseModuleWrapper"]:
        return self.parent.feed_forward_block if self.parent else None

    def clone(self) -> "BaseModuleWrapper":
        return type(self)(
            name=self.name, module=copy.deepcopy(self.module), parent=self.parent
        )

    @property
    def device(self) -> torch.device:
        return self.module.parameters().__next__().device

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, BaseModuleWrapper):
            return NotImplemented
        return (
            self.name == other.name
            and self.module.parameters().__next__().device
            == other.module.parameters().__next__().device
        )
    

class NamedLeaf(BaseModuleWrapper):
    def __init__(
        self, *, name: str, module: nn.Module, parent: Optional[BaseModuleWrapper]
    ):
        assert next(module.children(), None) is None, (
            "Module must have no children here but got {} with {} children".format(
                type(module), len([c for c in module.children()])
            )
        )
        super().__init__(name=name, module=module, parent=parent)

    @property
    def is_layer(self) -> bool:
        return True

    @property
    def named_layers(self) -> Generator[BaseModuleWrapper, None, None]:
        """Generator for named layers."""
        yield self


class NamedBlock(BaseModuleWrapper):
    _attrs: Mapping[str, List[Tuple[str, Type[BaseModuleWrapper]]]] = {}

    def __init__(
        self, *, name: str, module: nn.Module, parent: Optional[BaseModuleWrapper]
    ):
        super().__init__(name=name, module=module, parent=parent)
        _children: List[BaseModuleWrapper] = []
        if self._attrs:
            for child_name, child in module.named_modules():
                if (
                    child_name in self._attrs
                ):  
                    for attr_name, block_cls in self._attrs[child_name] :
                        setattr(
                            self,
                            attr_name,
                            self._create_child(child_name, child, block_cls),
                        )
                        _children.append(getattr(self, attr_name))
        self._children: Sequence[BaseModuleWrapper] = (
            _children
        )

    def _create_child(
        self, child_name: str, child: nn.Module, block: Type[BaseModuleWrapper]
    ) -> BaseModuleWrapper:
        return block(name=join_name(self.name, child_name), module=child, parent=self)

    @property
    def children(self) -> Generator[BaseModuleWrapper, None, None]:
        """Generator for named layers."""
        yield from self._children

    @property
    def named_layers(self) -> Generator[BaseModuleWrapper, None, None]:
        """Generator for named layers."""
        for child in self.children:
            for layer in child.named_layers:
                yield layer
