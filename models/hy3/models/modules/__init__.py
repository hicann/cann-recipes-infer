# Copyright (c) 2026 Huawei Technologies Co., Ltd.
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

"""Reusable HYV3 model components."""

from .common import (
    HYV3RMSNorm,
    HYV3RotaryEmbedding,
    build_pad_aware_prefill_metadata,
    ensure_qkv_fused_kscale_registered,
    equal_all_to_all,
    is_sequence_parallel_enabled,
    quantize_sequence_parallel_transport,
)

__all__ = [
    "HYV3RMSNorm",
    "HYV3RotaryEmbedding",
    "build_pad_aware_prefill_metadata",
    "ensure_qkv_fused_kscale_registered",
    "equal_all_to_all",
    "is_sequence_parallel_enabled",
    "quantize_sequence_parallel_transport",
]
