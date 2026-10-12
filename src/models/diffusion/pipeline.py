# SPDX-FileCopyrightText: Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-FileContributor: Yann Bouquet
# SPDX-License-Identifier: MIT

from typing import Generator, Type

from src.models.blocks.linear import NamedLinearLayer

from ..blocks import AttentionBlockWrapper, BaseModuleWrapper, FeedForwardBlockWrapper
from ..transformer import GenericTransformerBlockWrapper
from .dit import GenericDiTBlock
from diffusers import DiffusionPipeline

try:
    from typing import override
except Exception:
    from typing_extensions import override  # type: ignore


class ModelWrapperFactory:
    _model_registry = {}

    @classmethod
    def register_model(cls, model_name: str, model_cls: Type[GenericDiTBlock]):
        cls._model_registry[model_name] = model_cls

    @classmethod
    def wrap_model(cls, model) -> GenericDiTBlock:
        name = model.__class__.__name__
        wrapper_cls = cls._model_registry.get(name, None)
        if wrapper_cls is None:
            raise ValueError(f"No model wrapper registered for {name}.")
        return wrapper_cls(model)




class DiTPipelineWrapper(DiffusionPipeline, GenericDiTBlock):
    def __init__(self, pipeline):
        self.pipeline = pipeline
        
        self._model_wrapper: GenericDiTBlock = ModelWrapperFactory.wrap_model(
            pipeline.transformer
        )

    def __call__(self, *args, **kwargs):
        return self.pipeline(*args, **kwargs)

    def to(self, device=None, dtype=None, **kwargs):
        self.pipeline.to(device=device, dtype=dtype, **kwargs)
        return self

    @property
    @override
    def children(self) -> Generator[BaseModuleWrapper, None, None]:
        return self._model_wrapper.children

    @property
    def transformer(self) -> GenericDiTBlock:
        return self._model_wrapper

    @property
    @override
    def blocks(self) -> Generator[GenericTransformerBlockWrapper, None, None]:
        yield from self._model_wrapper.blocks

    @property
    @override
    def attentions(self) -> Generator[AttentionBlockWrapper, None, None]:
        yield from self._model_wrapper.attentions

    @property
    @override
    def feed_forwards(self) -> Generator[FeedForwardBlockWrapper, None, None]:
        yield from self._model_wrapper.feed_forwards

    @property
    @override
    def norms(self) -> Generator[NamedLinearLayer, None, None]:
        yield from self._model_wrapper.norms
        
    def save_pretrained(self, output_dir, **kwargs):
        self.pipeline.transformer.save_pretrained(output_dir, **kwargs)

    def from_pretrained(self, pretrained_model_name_or_path, **kwargs):
        self.pipeline.transformer.from_pretrained(
            pretrained_model_name_or_path, **kwargs
        )
