from typing import Any

import torch
import torchair
from torchair._ge_concrete_graph.fx2ge_converter import register_fx_node_ge_converter
from torchair.ge import attr
from torchair.ge._ge_graph import Tensor

# 为自定义算子注册 converter，用于 torch.compile / GE 图模式成图。
# meta_outputs 形参名为固定写法，用于 ge 节点输出 dtype/shape 推导。


@register_fx_node_ge_converter(torch.ops.custom.grouped_situ_mx_quant.default)
def convert_grouped_situ_mx_quant(
    x: Tensor,
    expert_tokens: Tensor,
    *,
    beta: float = 1.0,
    alpha: float = 1.0,
    high_precision: bool = False,
    meta_outputs: Any = None,
):
    out = torchair.ge.custom_op(
        "GroupedSituMxQuant",
        inputs={"x": x, "expert_tokens": expert_tokens},
        attrs={
            "beta": attr.Float(beta),
            "alpha": attr.Float(alpha),
            "high_precision": attr.Bool(high_precision),
        },
        outputs=["y", "mxscale"],
    )
    return [out[0], out[1]]
