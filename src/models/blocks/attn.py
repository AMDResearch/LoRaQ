# SPDX-FileCopyrightText: Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-FileContributor: Yann Bouquet
# SPDX-License-Identifier: MIT

from typing import Tuple, Dict, List, Optional, Type
from .base import BaseModuleWrapper, NamedBlock, join_name
from .linear import NamedLinearLayer, NamedLinearLayerList

try:
    from typing import override
except Exception:
    from typing_extensions import override  # type: ignore


class AttentionBlockWrapper(NamedBlock):
    q_proj: NamedLinearLayer
    k_proj: Optional[NamedLinearLayer]
    v_proj: Optional[NamedLinearLayer]
    o_proj: NamedLinearLayer
    add_q_proj: Optional[NamedLinearLayer]
    add_k_proj: Optional[NamedLinearLayer]
    add_v_proj: Optional[NamedLinearLayer]
    add_o_proj: Optional[NamedLinearLayer]

    _attrs: Dict[str, List[Tuple[str, Type[BaseModuleWrapper]]]] = {
        "to_q": [("q_proj", NamedLinearLayer)],
        "to_k": [("k_proj", NamedLinearLayer)],
        "to_v": [("v_proj", NamedLinearLayer)],
        "to_out.0": [("o_proj", NamedLinearLayer)],
        "add_q_proj": [("add_q_proj", NamedLinearLayer)],
        "add_k_proj": [("add_k_proj", NamedLinearLayer)],
        "add_v_proj": [("add_v_proj", NamedLinearLayer)],
        "to_add_out": [("add_o_proj", NamedLinearLayer)],
    }

    def __init__(
        self, *, name: str, module, parent: Optional[BaseModuleWrapper] = None
    ):
        self.k_proj: Optional[NamedLinearLayer] = None
        self.v_proj: Optional[NamedLinearLayer] = None
        
        self.add_q_proj: Optional[NamedLinearLayer] = None
        self.add_k_proj: Optional[NamedLinearLayer] = None

        self.add_v_proj: Optional[NamedLinearLayer] = None
        self.add_o_proj: Optional[NamedLinearLayer] = None

        super(AttentionBlockWrapper, self).__init__(
            name=name, module=module, parent=parent
        )
        assert self.q_proj is not None, "q_proj must be defined"
        assert self.o_proj is not None, "o_proj must be defined"

    @property
    def qkv_proj(self) -> NamedLinearLayerList:
        return NamedLinearLayerList(
            name=join_name(self.name, "qkv_proj"),
            named_layers=list(filter(None, [self.q_proj, self.k_proj, self.v_proj])),
            parent=self,
        )

    @property
    def add_qkv_proj(self) -> Optional[NamedLinearLayerList]:
        layers = list(filter(None, [self.add_q_proj, self.add_k_proj, self.add_v_proj]))
        if layers == []:
            return None
        return NamedLinearLayerList(
            name=join_name(self.name, "add_qkv_proj"), named_layers=layers, parent=self
        )

    @property
    def out_proj(self) -> Optional[NamedLinearLayer]:
        return self.o_proj

    @property
    def add_out_proj(self) -> Optional[NamedLinearLayer]:
        return self.add_o_proj

    @property
    @override
    def attention_block(self) -> Optional[BaseModuleWrapper]:
        return self

class SingleAttentionBlockWrapper(AttentionBlockWrapper):
    _attrs: Dict[str, List[Tuple[str, Type[BaseModuleWrapper]]]] = {
        "attn.to_q": [("q_proj", NamedLinearLayer)],
        "attn.to_k": [("k_proj", NamedLinearLayer)],
        "attn.to_v": [("v_proj", NamedLinearLayer)],
        "proj_out.linears.0": [("o_proj", NamedLinearLayer)]
    }
