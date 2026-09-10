# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

from typing import Any

import torch
import torchair
from torchair._ge_concrete_graph.fx2ge_converter import register_fx_node_ge_converter
from torchair.ge import attr
from torchair.ge._ge_graph import Tensor

_QUANT_MODE_TO_INT = {"mxfp8_bf16": 2, "mxfp4_bf16": 4}
_VALID_QUANT_MODES = tuple(_QUANT_MODE_TO_INT)


def _resolve_quant_mode(quant_mode: str) -> int:
    key = quant_mode.strip().lower() if isinstance(quant_mode, str) else quant_mode
    if key not in _QUANT_MODE_TO_INT:
        valid = ", ".join(_VALID_QUANT_MODES)
        raise ValueError(f"quant_mode should be one of [{valid}], but got {quant_mode!r}")
    return _QUANT_MODE_TO_INT[key]


def _validate_mode_and_group_size(quant_mode: str, quant_group_size: int) -> None:
    key = quant_mode.strip().lower() if isinstance(quant_mode, str) else quant_mode
    valid = ((key == "mxfp8_bf16" and quant_group_size == 32) or
             (key == "mxfp4_bf16" and quant_group_size in (16, 32)))
    if not valid:
        raise ValueError(
            "supported quant_mode/quant_group_size combinations are "
            "(mxfp8_bf16, 32), (mxfp4_bf16, 32), and "
            f"(mxfp4_bf16, 16), but got mode {quant_mode!r} and "
            f"group size {quant_group_size}"
        )


@register_fx_node_ge_converter(torch.ops.custom.kv_compress_epilog_v2.default)
def convert_kv_compress_epilog_v2(
    cache: Tensor,
    x: Tensor,
    slot_mapping: Tensor,
    *,
    quant_group_size: int = 32,
    quant_mode: str = "mxfp8_bf16",
    round_scale: bool = True,
    x_scale: float = 1.0,
    meta_outputs: Any = None,
):
    _validate_mode_and_group_size(quant_mode, quant_group_size)
    quant_mode_int = _resolve_quant_mode(quant_mode)
    return torchair.ge.custom_op(
        "KvCompressEpilogV2",
        inputs={"cache": cache, "x": x, "slot_mapping": slot_mapping},
        attrs={
            "quant_group_size": attr.Int(quant_group_size),
            "quant_mode": attr.Int(quant_mode_int),
            "round_scale": attr.Bool(round_scale),
            "x_scale": attr.Float(x_scale),
        },
        outputs=["cache"],
    )
