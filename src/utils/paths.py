# SPDX-FileCopyrightText: Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-FileContributor: Yann Bouquet
# SPDX-License-Identifier: MIT

"""Directory layout of the artefacts written by calibration, generation and evaluation.

Every stage of a transform stack nests its own directory inside the previous one, so
the path of a stack is the fold of the stage names over the output root:

    <root>/[shifted_acts/]<stage_0>/.../<stage_n>/                  calibration results
    <root>/[shifted_acts/]<stage_0>/.../<stage_n>/<quant>/<ds>/<tag>  quantized images
    <root>/<ds>/fp16/<tag>                                           full-precision references

The full-precision references do not depend on the stack and are shared by all of them.
"""

import os
import typing as tp

SHIFTED_ACTS_SEGMENT = "shifted_acts"


def stack_segments(calibrators: tp.Sequence[tp.Any], shift_activations: bool = False) -> tp.List[str]:
    """Directory components contributed by a stack of calibrators, in order."""
    segments = [SHIFTED_ACTS_SEGMENT] if shift_activations else []
    segments += [c.path_segment for c in calibrators]
    return segments


def stack_path(root: str, calibrators: tp.Sequence[tp.Any], shift_activations: bool = False) -> str:
    return os.path.join(root, *stack_segments(calibrators, shift_activations))


def quantizer_name(input_quantizer: tp.Any, weight_quantizer: tp.Any) -> str:
    return f"{input_quantizer.name}_{weight_quantizer.name}"


def image_tag(cfg: tp.Any) -> str:
    """Leaf directory of an image set: ``eval`` or ``debug``, suffixed by the seed offset if any."""
    tag = "debug" if cfg.generation.get("debug", False) else "eval"
    seed_offset = int(cfg.generation.dataset.get("seed_offset", 0))
    return f"{tag}_seed{seed_offset}" if seed_offset else tag


def fp16_dir(root: str, dataset: str, tag: str) -> str:
    return os.path.join(root, dataset, "fp16", tag)


def quantized_dir(root: str, segments: tp.Sequence[str], quant_name: str, dataset: str, tag: str) -> str:
    return os.path.join(root, *segments, quant_name, dataset, tag)


def existing_images(directory: str) -> tp.Set[str]:
    """Names of the images already generated in ``directory`` (empty if it does not exist)."""
    if not os.path.isdir(directory):
        return set()
    return {name for name in os.listdir(directory) if name.endswith(".png")}
