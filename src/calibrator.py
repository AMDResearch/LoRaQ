# SPDX-FileCopyrightText: Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-FileContributor: Yann Bouquet
# SPDX-License-Identifier: Apache-2.0

import typing as tp

from src.models.blocks import BaseModuleWrapper, NamedLinearLayerList
from src.models.blocks.attn import AttentionBlockWrapper
from src.models.blocks.ff import FeedForwardBlockWrapper
from src.models.blocks.linear import NamedLinearLayer
from src.models.transformer.transformer import GenericTransformerBlockWrapper
from src.utils.quantizer_utils import to_be_quantized

from .models.diffusion import GenericDiTBlock
from .strategy.base import Strategy
from .utils.hook import RemovableHandles


class DiTCalibrator:
    """Runs a calibration stage (`strategy`) on the linear layers matching `valid_layers` (regexes)."""

    def __init__(self, strategy: Strategy, valid_layers: tp.List[str]) -> None:
        self.strategy: Strategy = strategy
        self.valid_layers = valid_layers

    def _filtered_list(self, layers: NamedLinearLayerList) -> NamedLinearLayerList:
        return NamedLinearLayerList(
            name=layers.name,
            named_layers=[
                tp.cast(NamedLinearLayer, layer)
                for layer in layers.named_layers
                if layer is not None and self._layer_is_valid(layer.name) and isinstance(layer, NamedLinearLayer)
            ],
            parent=layers.parent,
        )

    def _layer_is_valid(self, layer_name: str) -> bool:
        return to_be_quantized(layer_name, self.valid_layers)

    @property
    def path_segment(self) -> str:
        return self.strategy.path_segment

    @property
    def output_dir(self) -> str:
        return self.strategy.output_dir

    @output_dir.setter
    def output_dir(self, output_dir: str) -> None:
        self.strategy.output_dir = output_dir

    def layer_extractor(
        self, block: GenericTransformerBlockWrapper
    ) -> tp.Generator[tp.Tuple[BaseModuleWrapper, BaseModuleWrapper], None, None]:
        """Yield (parent, layers) pairs of the valid layers of a transformer block."""
        for attn in block.attentions:
            yield (attn, self._filtered_list(attn.qkv_proj))

            if attn.add_qkv_proj is not None:
                yield (attn, self._filtered_list(attn.add_qkv_proj))

            o_proj = attn.o_proj
            add_o_proj = attn.add_o_proj

            if self._layer_is_valid(o_proj.name):
                yield (o_proj, o_proj)
            if add_o_proj is not None:
                if self._layer_is_valid(add_o_proj.name):
                    yield (add_o_proj, add_o_proj)

        for ff in block.feed_forwards:
            for layer in ff.named_layers:
                if self._layer_is_valid(layer.name):
                    yield (layer, layer)
        for norm in block.norms:
            if self._layer_is_valid(norm.name):
                yield (norm, norm)

    def calibrate(self, pipelines: tp.List[GenericDiTBlock]):
        self.strategy.search(pipelines, self.layer_extractor)

    def load_state_onto(self, pipelines: tp.List[GenericDiTBlock]) -> RemovableHandles:
        return self.strategy.apply(pipelines, self.layer_extractor)


class SmoothingCalibrator(DiTCalibrator):
    """Groups the layers that read the same input, so that they share one smoothing scale.

    In parallel blocks (FLUX single-stream blocks), q/k/v and the MLP up projection read the same
    normalized input and form one group; the outputs of joint attention (`to_out` and
    `to_add_out`) form another one.
    """

    def _layer_extractor_parallel(
        self, transformer_block: GenericTransformerBlockWrapper
    ) -> tp.Generator[tp.Tuple[BaseModuleWrapper, BaseModuleWrapper], None, None]:
        
        attentions = list([attn for attn in transformer_block.attentions])
        assert len(attentions) == 1, "Parallel block with multiple attentions not supported"
        attn = tp.cast(AttentionBlockWrapper, attentions[0])
        
        feed_forwards = list([ff for ff in transformer_block.feed_forwards])
        assert len(feed_forwards) == 1, "Parallel block with multiple feed-forwards"
        ff = tp.cast(FeedForwardBlockWrapper, feed_forwards[0])
        
        assert attn is not None
        assert ff is not None
        
        qkv_proj_up_proj : tp.List[NamedLinearLayer]= [
            tp.cast(NamedLinearLayer, layer)
            for layer in attn.qkv_proj.named_layers
            if layer is not None and self._layer_is_valid(layer.name) and isinstance(layer, NamedLinearLayer)
        ] + ([tp.cast(NamedLinearLayer, ff.up_proj)] if self._layer_is_valid(ff.up_proj.name) else [])

        qkv_proj_up_proj_list = NamedLinearLayerList(
            name = f"{attn.qkv_proj.name}+{ff.up_proj.name}",
            named_layers= qkv_proj_up_proj,
            parent = attn.qkv_proj.parent,
        )
        
        yield (attn, qkv_proj_up_proj_list)
            
        if attn.add_qkv_proj is not None:
            yield (attn, self._filtered_list(attn.add_qkv_proj))

        o_proj = attn.o_proj
        add_o_proj = attn.add_o_proj

        if add_o_proj is not None:
            if self._layer_is_valid(o_proj.name) and self._layer_is_valid(add_o_proj.name):
                merged_o_proj = NamedLinearLayerList(
                    name = f"{o_proj.name}+{add_o_proj.name}",
                    named_layers= [o_proj, add_o_proj],
                    parent = o_proj.parent,
                )
                yield (attn, merged_o_proj)
            else:
                if self._layer_is_valid(o_proj.name):
                    yield (o_proj, o_proj)
                if self._layer_is_valid(add_o_proj.name):
                    yield (add_o_proj, add_o_proj)
                    
        elif self._layer_is_valid(o_proj.name):
            yield (o_proj, o_proj)

        if self._layer_is_valid(ff.down_proj.name):
            yield (ff.down_proj, ff.down_proj)
                    
        for norm in transformer_block.norms:
            if self._layer_is_valid(norm.name):
                yield (norm, norm)
    
    def _layer_extractor_sequential(
        self, transformer_block: GenericTransformerBlockWrapper
    ) -> tp.Generator[tp.Tuple[BaseModuleWrapper, BaseModuleWrapper], None, None]:
        
        for attn in transformer_block.attentions:
            yield (attn, self._filtered_list(attn.qkv_proj))
                
            if attn.add_qkv_proj is not None:
                yield (attn, self._filtered_list(attn.add_qkv_proj))

            o_proj = attn.o_proj
            add_o_proj = attn.add_o_proj

            if add_o_proj is not None:
                if self._layer_is_valid(o_proj.name) and self._layer_is_valid(add_o_proj.name):
                    merged_o_proj = NamedLinearLayerList(
                        name = f"{o_proj.name}+{add_o_proj.name}",
                        named_layers= [o_proj, add_o_proj],
                        parent = o_proj.parent,
                    )
                    yield (attn, merged_o_proj)
            
            elif self._layer_is_valid(o_proj.name):
                yield (o_proj, o_proj)

            
        for ff in transformer_block.feed_forwards:
            for layer in ff.named_layers:
                if self._layer_is_valid(layer.name):
                    yield (layer, layer)
                        
        for norm in transformer_block.norms:
            if self._layer_is_valid(norm.name):
                yield (norm, norm)

    @tp.override
    def layer_extractor(
        self, block: GenericTransformerBlockWrapper
    ) -> tp.Generator[tp.Tuple[BaseModuleWrapper, BaseModuleWrapper], None, None]:
        if block.is_parallel:
            for eval_block, children in self._layer_extractor_parallel(block):
                yield (eval_block, children)
        else:
            for eval_block, children in self._layer_extractor_sequential(block):
                yield (eval_block, children)
