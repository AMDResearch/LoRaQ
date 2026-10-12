# SPDX-FileCopyrightText: Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-FileContributor: Yann Bouquet
# SPDX-License-Identifier: MIT

from .dit import DiTBlock, FluxModel, GenericDiTBlock, PixartDiTBlock, SANADiTBlock
from .pipeline import DiTPipelineWrapper, ModelWrapperFactory

__all__ = [
    'DiTPipelineWrapper',
    'GenericDiTBlock'
]

ModelWrapperFactory.register_model('SD3Transformer2DModel', DiTBlock)
ModelWrapperFactory.register_model('PixArtTransformer2DModel', PixartDiTBlock)
ModelWrapperFactory.register_model('FluxTransformer2DModel', FluxModel)
ModelWrapperFactory.register_model('SanaTransformer2DModel', SANADiTBlock)
