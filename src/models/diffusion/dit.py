# SPDX-FileCopyrightText: Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-FileContributor: Yann Bouquet
# SPDX-License-Identifier: MIT

from typing import Generator

from src.models.blocks.linear import NamedLinearLayer
from src.models.transformer.transformer import SANATransformerBlockListWrapper

try:
    from typing import override
except Exception:
    from typing_extensions import override  # type: ignore
from ..blocks import (
    AttentionBlockWrapper,
    FeedForwardBlockWrapper,
    NamedBlock,
)
from ..transformer import (
    AbstractTransformerBlockListWrapper,
    GenericTransformerBlockListWrapper,
    GenericTransformerBlockWrapper,
    PixartTransformerBlockListWrapper,
    SingleTransformerBlockListWrapper,
    TransformerBlockListWrapper,
)


class GenericDiTBlock(AbstractTransformerBlockListWrapper):
    transformer_blocks: GenericTransformerBlockListWrapper

    def __init__(self, model):
        NamedBlock.__init__(self, name="", module=model, parent=None)

    @property
    @override
    def blocks(self) -> Generator[GenericTransformerBlockWrapper, None, None]:
        yield from self.transformer_blocks.blocks

    @property
    @override
    def attentions(self) -> Generator[AttentionBlockWrapper, None, None]:
        yield from self.transformer_blocks.attentions

    @property
    @override
    def feed_forwards(self) -> Generator[FeedForwardBlockWrapper, None, None]:
        yield from self.transformer_blocks.feed_forwards

    @property
    @override
    def norms(self) -> Generator[NamedLinearLayer, None, None]:
        yield from self.transformer_blocks.norms

class DiTBlock(GenericDiTBlock):
    _attrs = {
        "transformer_blocks": [("transformer_blocks", TransformerBlockListWrapper)],
    }


class PixartDiTBlock(GenericDiTBlock):
    _attrs = {
        "transformer_blocks": [("transformer_blocks", PixartTransformerBlockListWrapper)],
    }


class SANADiTBlock(GenericDiTBlock):
    _attrs = {
        "transformer_blocks": [("transformer_blocks", SANATransformerBlockListWrapper)],
    }

class FluxModel(GenericDiTBlock):
    transformer_blocks: TransformerBlockListWrapper
    single_transformer_blocks: SingleTransformerBlockListWrapper

    _attrs = {
        "transformer_blocks": [("transformer_blocks", TransformerBlockListWrapper)],
        "single_transformer_blocks": [(
            "single_transformer_blocks",
            SingleTransformerBlockListWrapper,
        )],
    }

    @property
    @override
    def blocks(self) -> Generator[GenericTransformerBlockWrapper, None, None]:
        yield from self.transformer_blocks.blocks
        yield from self.single_transformer_blocks.blocks

    @property
    @override
    def attentions(self) -> Generator[AttentionBlockWrapper, None, None]:
        yield from self.transformer_blocks.attentions
        yield from self.single_transformer_blocks.attentions

    @property
    @override
    def feed_forwards(self) -> Generator[FeedForwardBlockWrapper, None, None]:
        yield from self.transformer_blocks.feed_forwards
        yield from self.single_transformer_blocks.feed_forwards
    
    @property
    @override
    def norms(self) -> Generator[NamedLinearLayer, None, None]:
        yield from self.transformer_blocks.norms
        yield from self.single_transformer_blocks.norms
