# SPDX-FileCopyrightText: Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-FileContributor: Yann Bouquet
# SPDX-License-Identifier: Apache-2.0

"""TensorCast (https://github.com/ROCm/tensorcast) with the modifications this code needs.

This is the only module that imports `tcast`; use `from src.utils.tcast_shadow import tcast`.

TensorCast is used unmodified from upstream (git submodule `third-parties/tcast`, pinned at
`PINNED_COMMIT`). Two behaviours are changed at import time ("shadowed"):

1. `Cast.cast` is wrapped in `torch.inference_mode` upstream. Calibration back-propagates
   through quantized tensors (straight-through estimators and quantization-error losses),
   and inference tensors cannot be saved for backward, so the wrapper is removed.
2. `Cast._get_scales` computes the scales from a detached copy of the input: scales and
   zero points are constants of the quantization grid, no gradient flows through them.
"""

import importlib
import inspect
import sys
from pathlib import Path

import torch
from loguru import logger as log

PINNED_COMMIT = "56c51c955f07d71eb1b42157733d173f4b20cbc0"
SUBMODULE_ROOT = Path(__file__).resolve().parents[2] / "third-parties" / "tcast"


def _import_tcast():
    if "tcast" in sys.modules:
        return sys.modules["tcast"]
    if (SUBMODULE_ROOT / "tcast" / "__init__.py").is_file():
        sys.path.insert(0, str(SUBMODULE_ROOT))
        try:
            return importlib.import_module("tcast")
        finally:
            sys.path.remove(str(SUBMODULE_ROOT))
    try:
        module = importlib.import_module("tcast")
    except ImportError as e:
        raise ImportError(
            "TensorCast not found. Run `git submodule update --init third-parties/tcast` "
            f"(or clone https://github.com/ROCm/tensorcast at {PINNED_COMMIT} into {SUBMODULE_ROOT})."
        ) from e
    log.warning(f"Using the installed TensorCast ({module.__file__}); this code is tested with commit {PINNED_COMMIT}.")
    return module


def _check_signature(fn, expected):
    params = list(inspect.signature(fn).parameters)
    if params != expected:
        raise RuntimeError(f"Unsupported TensorCast version: {fn.__qualname__}{params}, expected {expected}")


def _is_inference_mode_wrapper(fn) -> bool:
    return hasattr(fn, "__wrapped__") and any(
        isinstance(getattr(cell.cell_contents, "__self__", None), torch.inference_mode)
        or isinstance(cell.cell_contents, torch.inference_mode)
        for cell in (fn.__closure__ or ())
    )


def _patch(tcast) -> None:
    cast_cls = tcast.Cast
    if getattr(cast_cls, "_shadow_applied", False):
        return

    cast_fn = cast_cls.__dict__["cast"].__func__
    if _is_inference_mode_wrapper(cast_fn):
        cast_fn = cast_fn.__wrapped__
        cast_cls.cast = classmethod(cast_fn)
    _check_signature(cast_fn, ["cls", "x", "dtype", "castmode", "roundmode", "scalemode", "compmode",
                               "scaledata", "better", "select", "noreshape"])

    get_scales = cast_cls.__dict__["_get_scales"].__func__
    _check_signature(get_scales, ["cls", "x", "dtype", "noreshape"])

    def _get_scales_detached(cls, x, dtype, noreshape=False):
        return get_scales(cls, x.detach(), dtype, noreshape)

    cast_cls._get_scales = classmethod(_get_scales_detached)
    cast_cls._shadow_applied = True


def _self_test(tcast) -> None:
    x = torch.linspace(-4.0, 4.0, 64, requires_grad=True)
    dtype = tcast.DataType("e2m1fnuz", "e8m0_t32", "mxfp4e2")
    y = tcast.cast(x, dtype).tensor
    if y.is_inference():
        raise RuntimeError("TensorCast shadow failed: Cast.cast still runs in inference mode")
    if tcast.Cast._get_scales(x, dtype).scale.requires_grad:
        raise RuntimeError("TensorCast shadow failed: scales are not detached")


def _warn_if_commit_differs() -> None:
    git = SUBMODULE_ROOT / ".git"
    try:
        if git.is_file():
            git = (SUBMODULE_ROOT / git.read_text().split(":", 1)[1].strip()).resolve()
        head = (git / "HEAD").read_text().strip()
        if head.startswith("ref:"):
            head = (git / head.split(" ", 1)[1]).read_text().strip()
    except (OSError, IndexError):
        return
    if head != PINNED_COMMIT:
        log.warning(f"TensorCast checkout is at {head}, this code is tested with {PINNED_COMMIT}.")


tcast = _import_tcast()
_patch(tcast)
_self_test(tcast)
_warn_if_commit_differs()

DataType = tcast.DataType

__all__ = ["tcast", "DataType", "PINNED_COMMIT"]
