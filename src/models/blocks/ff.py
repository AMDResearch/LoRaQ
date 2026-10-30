# SPDX-FileCopyrightText: Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-FileContributor: Yann Bouquet
# SPDX-License-Identifier: Apache-2.0

from typing import Tuple, Dict, List, Optional, Type, cast
from torch import nn 

from torch.nn import Linear
from src.models.patches.linear import ShiftedLinear

from .base import BaseModuleWrapper, NamedBlock
from .linear import NamedLinearLayer
from diffusers.models.activations import GEGLU, GELU, ApproximateGELU, FP32SiLU, LinearActivation, SwiGLU
from loguru import logger as log
from ..patches.conv import Conv1x1ToLinear

try:
    from typing import override
except Exception:
    from typing_extensions import override  # type: ignore
    
class FeedForwardBlockWrapper(NamedBlock):
    act_type: str
    up_proj: NamedLinearLayer
    down_proj: NamedLinearLayer
            
    _attrs: Dict[str, List[Tuple[str, Type[BaseModuleWrapper]]]] = {
        "net.0.proj": [("up_proj", NamedLinearLayer)],
        "net.2": [("down_proj", NamedLinearLayer)],
    }
    
    def _extract_activation_type(self) -> str:
        module = self.module.net[0] # type: ignore
        if isinstance(module, (GELU, ApproximateGELU)):
            return "gelu"
        elif isinstance(module, GEGLU):
            return "gelu_glu"
        elif isinstance(module, SwiGLU):
            return "swish_glu"
        elif isinstance(module, LinearActivation):
            return module.activation.__class__.__name__.lower()
        elif isinstance(module, FP32SiLU):
            return "silu"
        else:
            raise NotImplementedError(f"Unsupported activation module: {module.__class__.__name__}")
    
    def __init__(self, *, name: str, module: nn.Module, parent: Optional[BaseModuleWrapper]):
        super().__init__(name=name, module=module, parent=parent)
        self.act_type = self._extract_activation_type()
        
    @property
    @override
    def feed_forward_block(self) -> Optional[BaseModuleWrapper]:
        return self

    def _lowerbound(self) -> float | None:
        if self.act_type.endswith("_glu"):
            return None
        elif self.act_type == "relu":
            return 0.0
        elif self.act_type == "gelu":
            return -0.171875
        elif self.act_type == "silu" or self.act_type == "swish":
            return -0.2734375
        elif self.act_type == "mish":
            return -0.31640625
        else:
            raise NotImplementedError(f"Unsupported activation type: {self.act_type}")
    
    def shift_activations(self) -> None:
        assert self.down_proj is not None
        if self.is_shifted:
            log.debug(f"Activations already shifted in {self.name} FeedForward block")
            return
        
        if (lowerbound := self._lowerbound()) is None:
            return
        
        log.info(f"Shift activations of {self.down_proj.name}")
        shifted_down_module = ShiftedLinear.from_linear(
            self.down_proj.module, shift=-lowerbound if lowerbound is not None else 0.0
        )
        
        self.module.net[2] = shifted_down_module
        self.down_proj = NamedLinearLayer(
            name=self.down_proj.name,
            module=shifted_down_module.linear,
            parent=self,
        )
        self.act_type += "_shifted"
        self._children = [
            self.up_proj,
            self.down_proj
        ]
    
    @property
    def is_shifted(self) -> bool:
        return self.act_type.endswith("_shifted")
    
class SingleFeedForwardBlockWrapper(FeedForwardBlockWrapper):
    act_type: str     
    _attrs: Dict[str, List[Tuple[str, Type[BaseModuleWrapper]]]] = {
        "proj_mlp": [("up_proj", NamedLinearLayer)],
        "proj_out.linears.1": [("down_proj", NamedLinearLayer)],
    }
    
    @override
    def _extract_activation_type(self) -> str:
        return "gelu"
    
    @override 
    def shift_activations(self) -> None:
        assert isinstance(self.down_proj, NamedLinearLayer)
        
        if self.is_shifted:
            log.debug(f"Activations already shifted in {self.name} FeedForward block")
            return
        
        if (lowerbound := self._lowerbound()) is None:
            return
        
        module_to_shift : Linear = cast(Linear, self.down_proj.module)
        shifted_down_module = ShiftedLinear.from_linear(
            module_to_shift, shift=-lowerbound if lowerbound is not None else 0.0
        )
        
        self.module.proj_out.linears[1] = shifted_down_module # type: ignore
        
        self.down_proj = NamedLinearLayer(
            name=self.down_proj.name,
            module=shifted_down_module.linear,
            parent=self,
        )
        self._children = [
            self.up_proj,
            self.down_proj
        ]
        
        self.act_type += "_shifted"
        log.info(f"Shift activations of {self.down_proj.name} in {self.name} FeedForward block")


class ConvFeedForwardBlockWrapper(FeedForwardBlockWrapper):
    act_type: str     
    _attrs: Dict[str, List[Tuple[str, Type[BaseModuleWrapper]]]] = {
        "conv_inverted.linear": [("up_proj", NamedLinearLayer)],
        "conv_point.linear": [("down_proj", NamedLinearLayer)],
    }
    
    @override 
    def _extract_activation_type(self) -> str:
        return "silu"
    
    @override 
    def __init__(self, *, name: str, module: nn.Module, parent: Optional[BaseModuleWrapper]):
        module.conv_inverted = Conv1x1ToLinear(module.conv_inverted)  # type: ignore
        module.conv_point = Conv1x1ToLinear(module.conv_point)  # type: ignore
        super(ConvFeedForwardBlockWrapper, self).__init__(name=name, module=module, parent=parent)
    
    @override 
    def shift_activations(self) -> None:
        pass
