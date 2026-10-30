# SPDX-FileCopyrightText: Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-FileContributor: Yann Bouquet
# SPDX-License-Identifier: Apache-2.0

"""TensorCast datatypes selectable by name in the configs (`base_dtype`, `dtype: ...`).

Each function returns a fresh `DataType`; `mxfp4e2` (OCP MXFP4: e2m1 elements, shared e8m0
scale per 32 values) is the format of the paper.
"""

import torch

from src.utils.tcast_shadow import DataType

def float32(): return DataType(torch.float32)  # type: ignore
def float16(): return DataType(torch.float16)  # type: ignore
def bfloat16(): return DataType(torch.bfloat16)  # type: ignore

def e5m2(): return DataType("e5m2")
def e5m2fnuz(): return DataType("e5m2fnuz")
def e5m2b16fnuz(): return DataType("e5m2b16fnuz")
def binary8p3(): return DataType("e5m2b16fnuz")
def e4m3fnuz(): return DataType("e4m3fnuz")
def e4m3fn(): return DataType("e4m3fn")
def e4m3b8fnuz(): return DataType("e4m3b8fnuz")
def binary8p4(): return DataType("e4m3b8fnuz")
def binary8p5(): return DataType("e3m4b4fnuz")
def e3m3fnuz(): return DataType("e3m3fnuz")
def e3m2fnuz(): return DataType("e3m2fnuz")
def e2m3fnuz(): return DataType("e2m3fnuz")
def e2m1fnuz(): return DataType("e2m1fnuz")

def uint16_ff(): return DataType("uint16", "float16_float16", "uint16_ff")
def uint16_bb(): return DataType("uint16", "bfloat16_bfloat16", "uint16_bb")
def int16_f(): return DataType("int16", "float16", "int16_f")
def int16_b(): return DataType("int16", "bfloat16", "int16_b")
def int16_e(): return DataType("int16", "e8m0", "int16_e")
def uint8_ff(): return DataType("uint16", "float16_float16", "uint16_ff")
def uint8_bb(): return DataType("uint16", "bfloat16_bfloat16", "uint16_bb")
def uint8_fi(): return DataType("uint16", "float16_int8", "uint16_fi")
def uint8_bi(): return DataType("uint16", "bfloat16_int8", "uint16_bi")
def int8_f(): return DataType("int8", "float16", "int8_f")
def int8_b(): return DataType("int8", "bfloat16", "int8_b")
def int8_e(): return DataType("int8", "e8m0", "int8_e")
def e5m2_e(): return DataType("e5m2", "e8m0", "e5m2_e")
def e5m2z_e(): return DataType("e5m2fnuz", "e8m0", "e5m2z_e")
def e4m3_e(): return DataType("e4m3fn", "e8m0", "e4m3_e")
def e4m3z_e(): return DataType("e4m3fnuz", "e8m0", "e4m3z_e")
def e3m2_e(): return DataType("e3m2fnuz", "e8m0", "e3m2_e")
def e2m3_e(): return DataType("e2m3fnuz", "e8m0", "e2m3_e")

def mxfp8e5(): return DataType("e5m2", "e8m0_t32", "mxfp8e5")
def mxfp8e4(): return DataType("e4m3fn", "e8m0_t32", "mxfp8e4")
def mxfp6e3(): return DataType("e3m2fnuz", "e8m0_t32", "mxfp6e3")
def mxfp6e2(): return DataType("e2m3fnuz", "e8m0_t32", "mxfp6e2")
def mxfp4e2(): return DataType("e2m1fnuz", "e8m0_t32", "mxfp4e2")
def mxint8(): return DataType("int8", "e8m0_t32", "mxint8")
def mxint4(): return DataType("int4", "e8m0_t32", "mxint4")

def bfp16(): return DataType("int8", "e8m0_t8", "bfp16")

def uint4_ff32(): return DataType("uint4", "float16_float16_t32", "uint4_ff32")
def uint4_bb32(): return DataType("uint4", "bfloat16_bfloat16_t32", "uint4_bb32")
def uint4_fi32(): return DataType("uint4", "float16_int8_t32", "uint4_fi32")
def uint4_bi32(): return DataType("uint4", "bfloat16_int8_t32", "uint4_bi32")
def int4_f32(): return DataType("int4", "float16_t32", "int4_f32")
def int4_b32(): return DataType("int4", "bfloat16_t32", "int4_b32")
def int4_f64(): return DataType("int4", "float16_t64", "int4_f64")
def int4_b64(): return DataType("int4", "bfloat16_t64", "int4_b64")

def e1m6b1fnuz(): return DataType("e1m6b1fnuz", "e8m0_t8", "e1m6_e32")
def e1m2b1fnuz(): return DataType("e1m2b1fnuz", "e8m0_t8", "e1m2_e32")

def sint4(): return DataType("int4", "float32_t64", "sint4")
def uint4(): return DataType("uint4", "float32_t64_float32", "uint4")

def auint4(): return DataType("uint4", "float32_int8_t0", "auint4")
def auint4_f(): return DataType("uint4", "float32_float32_t0", "auint4_f")

def sint4_ch(): return DataType("int4", "float32_t0", "sint4_ch")
def uint4_ch(): return DataType("uint4", "float32_int8_t0", "uint4_ch")


def parse_datatype(name: str) -> DataType:
    """The datatype called `name` in this module, e.g. "mxfp4e2"."""
    factory = globals().get(name)
    if not callable(factory) or name.startswith("_") or name in ("parse_datatype", "DataType"):
        raise ValueError(f"Unsupported dtype: {name}")
    return factory()
