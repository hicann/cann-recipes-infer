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

from executor.core.config import InferenceConfig


def check_model_settings(infer_config: InferenceConfig):
    mc = infer_config.model_config
    exe_mode = mc.exe_mode
    moe_chunk_max_len = mc.custom_params.get("moe_chunk_max_len", 65536)
    enable_multi_streams = mc.custom_params.get("enable_multi_streams", False)
    if exe_mode not in ["eager", "npugraph_ex"]:
        raise ValueError(f"{exe_mode=} is not supported; only 'eager' and 'npugraph_ex' are allowed.")
    if moe_chunk_max_len <= 0:
        raise ValueError(f"{moe_chunk_max_len=} should be a positive integer.")
    if exe_mode == "eager" and mc.enable_cache_compile:
        raise ValueError(f"{exe_mode=} does not support cache compile!")


def check_parallel_settings(infer_config: InferenceConfig):
    pc, sc = infer_config.parallel_config, infer_config.scheduler_config
    # world_size divisibility by each tp_size is validated by the framework's ParallelConfig._validate
    if pc.attn_tp_size != pc.o_proj_tp_size:
        raise ValueError(
            f"attn_tp_size must equal o_proj_tp_size, but got "
            f"{pc.attn_tp_size=} {pc.o_proj_tp_size=}")
    if pc.attn_dp_size and sc.batch_size % pc.attn_dp_size != 0:
        raise ValueError(f"batch_size must be divisible by attn_dp_size, got "
                         f"{sc.batch_size=} {pc.attn_dp_size=}")
    # The layer stack is written for a batch sharded across attn_tp and rebuilt by the
    # all_gather at each attention entry, so this rank's share must divide evenly.
    if sc.batch_size_per_dp_rank % pc.attn_tp_size != 0:
        raise ValueError(
            f"batch_size_per_dp_rank must be divisible by attn_tp_size, got "
            f"{sc.batch_size_per_dp_rank=} {pc.attn_tp_size=}")


def check_cache_settings(infer_config: InferenceConfig):
    """
    batch_size_per_rank sizes the KV cache / GLA state / block_table; pa_max_length sizes
    one request's region.
    """
    from .modeling_bailing_moe_v2_5 import get_pa_max_length

    if infer_config.scheduler_config.batch_size_per_dp_rank < 1:
        raise ValueError(
            f"batch_size_per_dp_rank={infer_config.scheduler_config.batch_size_per_dp_rank} must be >= 1")
    pa_max_length = get_pa_max_length(infer_config)
    input_max_len = infer_config.data_config.input_truncated_len
    if pa_max_length < input_max_len:
        raise ValueError(f"{pa_max_length=} must be >= {input_max_len=}")


def check_vars(infer_config: InferenceConfig):
    check_parallel_settings(infer_config)
    check_model_settings(infer_config)
    check_cache_settings(infer_config)
