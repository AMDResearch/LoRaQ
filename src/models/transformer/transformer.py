# SPDX-FileCopyrightText: Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-FileContributor: Yann Bouquet
# SPDX-License-Identifier: Apache-2.0

from typing import (
    Optional,
    Type,
    Generator,
    List,
    Tuple,
    Dict,
    override,
)

from abc import ABC, abstractmethod

from src.models.blocks.ff import ConvFeedForwardBlockWrapper
import torch
from loguru import logger as log

from src.models.blocks.attn import SingleAttentionBlockWrapper
from src.models.blocks.linear import NamedLinearLayer
from src.utils.hook import Outputs

from ..blocks import (
    AttentionBlockWrapper,
    BaseModuleWrapper,
    FeedForwardBlockWrapper,
    NamedBlock,
    SingleFeedForwardBlockWrapper,
)
from ..patches.linear import ConcatLinear


class GenericTransformerBlockWrapper(NamedBlock, ABC):
    hidden_states: List[str] = ["encoder_hidden_states", "hidden_states"]
    is_parallel: bool = False

    @property
    @abstractmethod
    def attentions(self) -> Generator[AttentionBlockWrapper, None, None]: ...

    @property
    @abstractmethod
    def feed_forwards(self) -> Generator[FeedForwardBlockWrapper, None, None]: ...

    @property
    @abstractmethod
    def norms(self) -> Generator[NamedLinearLayer, None, None]: ...

    @property
    @override
    def transformer_block(self) -> Optional[BaseModuleWrapper]:
        """Returns the transformer block if it exists."""
        return self

    def extract_state_from_inputs(self, inputs, kwargs, with_const: bool = True):
        assert len(self.hidden_states) >= len(inputs), (
            "We expect all args to be hidden states"
        )

        hs_kwargs = {}
        const_kwargs = {}

        hs_kwargs = {k: v for k, v in zip(self.hidden_states, [i for i in inputs])}
        hs_kwargs.update(
            {
                k: kwargs[k] for k in self.hidden_states[len(inputs) : ] if k in kwargs
            }
        )
        if with_const:
            const_kwargs = {k: v for k, v in kwargs.items() if k not in hs_kwargs}
        return hs_kwargs, const_kwargs

    @classmethod
    def extract_inputs_from_previous_block_outputs(cls, outputs: Outputs):
        assert isinstance(outputs, Tuple) and len(outputs) == 2, (
            f"{cls.__class__.__name__} expects two outputs from the previous block, got {len(outputs)} outputs of type {type(outputs)}."
        )

        encoder_hidden_states, hidden_states = outputs

        result = {
            k: v
            for k, v in zip(cls.hidden_states, (encoder_hidden_states, hidden_states))
        }
        return result


class TransformerBlockWrapper(GenericTransformerBlockWrapper):
    attn: AttentionBlockWrapper
    ff: FeedForwardBlockWrapper
    add_ff: Optional[FeedForwardBlockWrapper] = None
    norm: Optional[NamedLinearLayer] = None
    add_norm: Optional[NamedLinearLayer] = None

    
    _attrs: Dict[str, List[Tuple[str, Type[BaseModuleWrapper]]]] = {
        "attn": [("attn", AttentionBlockWrapper)],
        "ff": [("ff", FeedForwardBlockWrapper)],
        "ff_context": [("add_ff", FeedForwardBlockWrapper)],
        "norm1.linear": [("norm", NamedLinearLayer)],
        "norm1_context.linear": [("add_norm", NamedLinearLayer)],
    }

    @property
    @override
    def attentions(self) -> Generator[AttentionBlockWrapper, None, None]:
        yield self.attn

    @property
    @override
    def feed_forwards(self) -> Generator[FeedForwardBlockWrapper, None, None]:
        yield self.ff
        if self.add_ff is not None:
            yield self.add_ff

    @property
    @override
    def norms(self) -> Generator[NamedLinearLayer, None, None]:
        if self.norm is not None:
            yield self.norm
        if self.add_norm is not None:
            yield self.add_norm

class PixartTransformerBlockWrapper(GenericTransformerBlockWrapper):
    hidden_states: List[str] = ["hidden_states"]

    attn1: AttentionBlockWrapper
    attn2: AttentionBlockWrapper
    ff: FeedForwardBlockWrapper

    _attrs: Dict[str, List[Tuple[str, Type[BaseModuleWrapper]]]] = {
        "attn1": [("attn1", AttentionBlockWrapper)],
        "attn2": [("attn2", AttentionBlockWrapper)],
        "ff": [("ff", FeedForwardBlockWrapper)],
    }

    @property
    @override
    def attentions(self) -> Generator[AttentionBlockWrapper, None, None]:
        yield self.attn1
        yield self.attn2
        
    @property
    @override
    def feed_forwards(self) -> Generator[FeedForwardBlockWrapper, None, None]:
        yield self.ff
    
    @property
    @override
    def norms(self) -> Generator[NamedLinearLayer, None, None]:
        yield from ()
        
    @override
    def extract_state_from_inputs(self, inputs, kwargs, with_const: bool = True):
        assert len(inputs) == 1, "We expect all args to be hidden states"
        const_kwargs = {}

        hs_kwargs = {k: v for k, v in zip(self.hidden_states, inputs)}
        if with_const:
            const_kwargs = {k: v for k, v in kwargs.items()}
        return hs_kwargs, const_kwargs

    @override    
    @classmethod
    def extract_inputs_from_previous_block_outputs(cls, outputs):
        assert isinstance(outputs, torch.Tensor), (
            f"{cls.__class__.__name__} expects only one output from the previous block, got {len(outputs)} outputs of type {type(outputs)}."
        )
        return {"hidden_states": outputs}

class SingleTransformerBlockWrapper(TransformerBlockWrapper):    
    hidden_states: List[str] = ["hidden_states"]

    _attrs: Dict[str, List[Tuple[str, Type[BaseModuleWrapper]]]] = {
        "": [("attn", SingleAttentionBlockWrapper), ("ff", SingleFeedForwardBlockWrapper)],
        "norm.linear": [("norm", NamedLinearLayer)],
    }

    is_parallel: bool = True

    def __init__(self, *, name: str, module, parent: Optional[BaseModuleWrapper]):
        log.info(f"Replacing fused Linear in {name} with ConcatLinear.")
        module.proj_out = ConcatLinear.from_linear(
            module.proj_out, [module.proj_out.out_features]
        )
        super(SingleTransformerBlockWrapper, self).__init__(
            name=name, module=module, parent=parent
        )

    @override
    @classmethod
    def extract_inputs_from_previous_block_outputs(cls, outputs: Outputs):
        if isinstance(outputs, torch.Tensor):
            return {"hidden_states": outputs}
        assert isinstance(outputs, Tuple), "Outputs should be a tuple or a tensor."
        
        if len(outputs) == 1:
            log.debug(f"{cls.__class__.__name__} expects only one output from the previous block.")
            hidden_states, *_ = outputs
            return {"hidden_states": hidden_states}
        else:
            assert len(outputs) == 2, (
                f"{cls.__class__.__name__} expects two outputs from the previous block, got {len(outputs)} outputs of type {type(outputs)}."
            )
            log.debug(
                f"{cls.__class__.__name__} expects two outputs from the previous block, got {len(outputs)} outputs."
            )
            
            encoder_hidden_states, hidden_states = outputs
            
            cat_hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)
            
            return {
                "hidden_states": cat_hidden_states,
            }


class SANATransformerBlockWrapper(GenericTransformerBlockWrapper):
    hidden_states: List[str] = ["hidden_states"]
    constants: List[str] = [
        "attention_mask",
        "encoder_hidden_states",
        "encoder_attention_mask",
        "timestep",
        "height",
        "width",
    ]
    
    attn1: AttentionBlockWrapper
    attn2: AttentionBlockWrapper
    ff: ConvFeedForwardBlockWrapper

    _attrs: Dict[str, List[Tuple[str, Type[BaseModuleWrapper]]]] = {
        "attn1": [("attn1", AttentionBlockWrapper)],
        "attn2": [("attn2", AttentionBlockWrapper)],
        "ff": [("ff", ConvFeedForwardBlockWrapper)],
    }
    
    @property
    @override
    def attentions(self) -> Generator[AttentionBlockWrapper, None, None]:
        yield self.attn1
        yield self.attn2
        
    @property
    @override
    def feed_forwards(self) -> Generator[FeedForwardBlockWrapper, None, None]:
        yield self.ff
    
    @property
    @override
    def norms(self) -> Generator[NamedLinearLayer, None, None]:
        yield from ()
        
    @override
    def extract_state_from_inputs(self, inputs, kwargs, with_const: bool = True):
        assert len(inputs) == 7, "We expect all args to be hidden states"
        hidden_states, *csts = inputs

        hs_kwargs = {k: v for k, v in zip(self.hidden_states, [hidden_states])}
        if with_const:
            const_kwargs = {k: v for k, v in zip(self.constants, csts)}
        return hs_kwargs, const_kwargs

    @override    
    @classmethod
    def extract_inputs_from_previous_block_outputs(cls, outputs):
        assert isinstance(outputs, torch.Tensor), (
            f"{cls.__class__.__name__} expects only one output from the previous block, got {len(outputs)} outputs of type {type(outputs)}."
        )
        return {"hidden_states": outputs}


class AbstractTransformerBlockListWrapper(GenericTransformerBlockWrapper, ABC):
    @property
    @abstractmethod
    def blocks(self) -> Generator[GenericTransformerBlockWrapper, None, None]: ...

    @property
    @override
    def attentions(self) -> Generator[AttentionBlockWrapper, None, None]:
        for block in self.blocks:
            yield from block.attentions

    @property
    @override
    def feed_forwards(self) -> Generator[FeedForwardBlockWrapper, None, None]:
        for block in self.blocks:
            yield from block.feed_forwards

    @property
    @override
    def transformer_block(self) -> Optional[BaseModuleWrapper]:
        return None

    @property
    @override
    def norms(self) -> Generator[NamedLinearLayer, None, None]:
        for block in self.blocks:
            yield from block.norms

class GenericTransformerBlockListWrapper(AbstractTransformerBlockListWrapper):
    _block_wrapper_type: Type[GenericTransformerBlockWrapper]

    def __init__(self, *, name: str, module, parent: Optional[BaseModuleWrapper]):
        NamedBlock.__init__(self, name=name, module=module, parent=parent)
        _children: List[BaseModuleWrapper] = []
        for block_name, block in module.named_children():
            _children.append(
                self._create_child(block_name, block, self._block_wrapper_type)
            )
        self._children = _children

    @property
    @override
    def blocks(self) -> Generator[GenericTransformerBlockWrapper, None, None]:
        """Generator that yields all transformer blocks."""
        for child in self._children:
            if isinstance(child, self._block_wrapper_type):
                yield child

class TransformerBlockListWrapper(GenericTransformerBlockListWrapper):
    _block_wrapper_type = TransformerBlockWrapper

class PixartTransformerBlockListWrapper(GenericTransformerBlockListWrapper):
    _block_wrapper_type = PixartTransformerBlockWrapper

class SingleTransformerBlockListWrapper(GenericTransformerBlockListWrapper):
    _block_wrapper_type = SingleTransformerBlockWrapper

class SANATransformerBlockListWrapper(GenericTransformerBlockListWrapper):
    _block_wrapper_type = SANATransformerBlockWrapper
    
