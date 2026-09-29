# coding=utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from tile_kernels.mhc import mhc_post_fwd

from ..registry import register_op_impl
from .mhc import hc_post_ascendc


@register_op_impl(op_type="hc_post", func_key="hc_post_tilelang")
def hc_post_tilelang(x, residual, post, comb, is_prefill=False):
    if not is_prefill:
        return hc_post_ascendc(x, residual, post, comb)
    mhc = post.shape[-1]
    return mhc_post_fwd(
        x.unsqueeze(0).contiguous(),
        residual.unsqueeze(0).contiguous(),
        post.reshape(1, -1, mhc, 1).contiguous(),
        comb.reshape(1, -1, mhc, mhc).contiguous(),
    )[0]
