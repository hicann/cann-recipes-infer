from typing import Any

import torch
import torchair
from torchair._ge_concrete_graph.fx2ge_converter import register_fx_node_ge_converter
from torchair.ge import attr
from torchair.ge._ge_graph import Tensor


@register_fx_node_ge_converter(torch.ops.custom.npu_situ_and_mul_sparse.default)
def convert_npu_situ_and_mul_sparse(
    x: Tensor,
    expert_tokens: Tensor,
    *,
    beta: float = 1.0,
    alpha: float = 1.0,
    high_precision: bool = False,
    meta_outputs: Any = None,
):
    return torchair.ge.custom_op(
        "SituAndMulSparse",
        inputs={"x": x, "expert_tokens": expert_tokens},
        attrs={
            "beta": attr.Float(beta),
            "alpha": attr.Float(alpha),
            "high_precision": attr.Bool(high_precision),
        },
        outputs=["y"],
    )
