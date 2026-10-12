# SPDX-FileCopyrightText: Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-FileContributor: Yann Bouquet
# SPDX-License-Identifier: MIT

from typing import Self, Union

import torch
from loguru import logger as log
from torch import Tensor, nn

try:
    from typing import override
except ImportError:
    from typing_extensions import override

from src.models.patches.lowrank import LowRankBranch
from src.quantizer.dtypes import parse_datatype
from src.utils.hook import Args, GenericHook, RemovableHandles
from src.utils.tcast_shadow import tcast


class BaseQuantizer(GenericHook):
    """Fake-quantization (quantize-dequantize) to a TensorCast datatype, e.g. "mxfp4e2".

    Not thread-safe: some TensorCast casts keep a state depending on the input tensor, so
    each GPU worker uses its own clone.
    """

    def _get_unsigned_dtype(self, dtype: str) -> tcast.DataType | None:
        """Unsigned twin of an integer datatype, used for shifted (non-negative) activations."""
        if "int" in dtype:
            if "uint" in dtype:
                log.warning(f"Dtype {dtype} is already unsigned.")
                return None
            if "sint" in dtype:
                unsigned_name = dtype.replace("sint", "uint")
            else:
                unsigned_name = dtype.replace("int", "uint")
            try:
                return parse_datatype(unsigned_name)
            except ValueError as e:
                log.error(f"Could not find unsigned dtype {unsigned_name} for {dtype} : {e}")
                return None
        log.warning(f"Dtype {dtype} is not an integer type, cannot get unsigned dtype.")
        return None

    def __init__(self, dtype: str) -> None:
        self.dtype: tcast.DataType = parse_datatype(dtype)
        self.unsigned_dtype: tcast.DataType | None = self._get_unsigned_dtype(dtype)
        self.name = f"{self.__class__.__name__}_{self.dtype.name}"

    def __call__(self, x: Tensor) -> Tensor:
        return tcast.cast(x, self.dtype).tensor

    def quantize_modules(self, *modules: nn.Module) -> RemovableHandles:
        """Quantizes the weights of linear modules in place."""
        assert all(isinstance(m, nn.Linear) for m in modules), "BaseQuantizer can only be applied to nn.Linear modules."
        with torch.no_grad():
            for module in modules:
                module.weight.copy_(self(module.weight).to(module.weight.dtype))
        return RemovableHandles()

    def clone(self) -> Self:
        return self.__class__(self.dtype.name)


class LowRankQuantizer(BaseQuantizer):
    """Weight quantization with a high-precision low-rank branch (as in SVDQuant): the rank-`rank`
    SVD of the weight is kept in a branch and only the residual is quantized."""

    def __init__(self, dtype: str, rank: int) -> None:
        super().__init__(dtype)
        self.name = f"{self.name}_lr_{rank}"
        self.rank = rank

    @override
    def hooked_on(self, module: nn.Module) -> RemovableHandles:
        assert isinstance(module, nn.Linear), "LowRankQuantizer can only be hooked on nn.Linear modules."
        device = next(module.parameters()).device
        dtype = next(module.parameters()).dtype
        with torch.no_grad():
            branch = LowRankBranch.from_svd(module.weight, self.rank, device=device, dtype=dtype)
            residual = module.weight.data - branch.weight.data
            module.weight.data.copy_(self(residual).to(module.weight.data.dtype))
        return branch.hooked_on(module)

    @override
    def quantize_modules(self, *modules: nn.Module) -> RemovableHandles:
        handles = RemovableHandles()
        for module in modules:
            handles.append(self.hooked_on(module))
        return handles

    @override
    def clone(self) -> Self:
        return self.__class__(self.dtype.name, self.rank)


class WeightQuantizer(BaseQuantizer):
    """Wraps a quantizer applied to weights (prefixes its name with `weight_`)."""

    def __init__(self, base: BaseQuantizer):
        self.dtype = base.dtype
        self.name = f"weight_{base.name}"
        self.quantizer = base

    def __call__(self, x: Tensor) -> Tensor:
        return self.quantizer(x)

    def quantize_modules(self, *modules: nn.Module) -> RemovableHandles:
        return self.quantizer.quantize_modules(*modules)

    @override
    def clone(self) -> Self:
        return self.__class__(self.quantizer.clone())


class IOQuantizerDecorator(WeightQuantizer):
    @override
    def hooked_on(self, module: nn.Module) -> RemovableHandles:
        return self.quantizer.hooked_on(module)


class InputQuantizer(IOQuantizerDecorator):
    """Quantizes the inputs of the layers it is hooked on (forward pre-hook)."""

    def __init__(self, base: BaseQuantizer) -> None:
        super().__init__(base)
        self.name = f"input_{base.name}"

    @override
    def _as_pre_hook(self, module: nn.Module, inputs: Args) -> Union[Args, None]:
        if getattr(module, "shifted", False) and self.quantizer.unsigned_dtype is not None:
            return tuple(tcast.cast(x, self.quantizer.unsigned_dtype).tensor if isinstance(x, Tensor) else x for x in inputs)
        return tuple(self(x) if isinstance(x, Tensor) else x for x in inputs)

    @override
    def hooked_on(self, module: nn.Module) -> RemovableHandles:
        handles = super().hooked_on(module)
        handles.append(module.register_forward_pre_hook(self._as_pre_hook))
        return handles
