# coding=utf-8
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

"""Unified inference configuration.

This module contains all configuration classes for the inference framework:
- DataConfig: Data and sequence configuration
- ModelConfig: Model-specific configuration
- ParallelConfig: Parallel execution configuration
- SchedulerConfig: Request scheduler configuration
- InferenceConfig: Unified configuration container
"""
import os
import math
from dataclasses import dataclass, field
from typing import Any


@dataclass
class DataConfig:
    """Data configuration for inference.

    Attributes:
        dataset: Dataset name (e.g., "default", "LongBench") (default: "default")
        dataset_path: Path to the dataset (default: "")
    """
    dataset: str = "default"
    dataset_path: str = ""
    input_truncated_len: int = 256

    @classmethod
    def from_dict(cls, data_config_dict: dict) -> "DataConfig":
        """Create DataConfig from YAML-parsed dictionary."""
        return cls(
            dataset=data_config_dict.get("dataset", "default"),
            dataset_path=data_config_dict.get("dataset_path", ""),
            input_truncated_len=data_config_dict.get("input_truncated_len", 256)
        )


@dataclass
class ModelConfig:
    """Model-specific configuration for inference.

    Attributes:
        model_name: Name of the model (default: "model")
        model_path: Path to the model weights (default: "")
        output_path: Path to save the output, log, profiling, graph cache etc. (default: "")
        dtype: Data type for model weights and computation (default: "bfloat16")
        with_ckpt: Whether to load checkpoint (default: True)
        next_n: Number of the speculative steps (default: 0)

        exe_mode: Execution mode (eager, ge_graph, npugraph_ex) (default: "eager")
        enable_cache_compile: Enable cache compilation (default: False)
        enable_static_kernel: Enable static kernel acceleration for npugraph_ex inference (default: False)
        force_eplb: Whether to enable force expert load balancing for MoE models (default: False)

        enable_profiler: Enable profiler (default: False)
        enable_weight_nz: Whether to enable NZ format for weights (default: True)
    """
    model_name: str = "model"
    model_path: str = ""
    output_path: str = ""
    dtype: str = "bfloat16"
    with_ckpt: bool = True
    next_n: int = 0
    platform_version: str = "A3"

    exe_mode: str = "eager"
    enable_cache_compile: bool = False
    enable_static_kernel: bool = False
    force_eplb: bool = False

    enable_profiler: bool = False

    enable_weight_nz: bool = True

    custom_params: dict = field(default_factory=dict)

    @classmethod
    def from_dict(cls, model_config_dict: dict) -> "ModelConfig":
        """Create ModelConfig from YAML-parsed dictionary."""
        model_config = cls(
            model_name=model_config_dict.get("model_name", "model"),
            model_path=model_config_dict.get("model_path", ""),
            output_path=model_config_dict.get("output_path", ""),
            dtype=model_config_dict.get("dtype", "bfloat16"),
            with_ckpt=model_config_dict.get("with_ckpt", True),
            next_n=model_config_dict.get("next_n", 0),
            platform_version=model_config_dict.get("platform_version", "A3"),
            exe_mode=model_config_dict.get("exe_mode", "eager"),
            enable_cache_compile=model_config_dict.get("enable_cache_compile", False),
            enable_static_kernel=model_config_dict.get("enable_static_kernel", False),
            force_eplb=model_config_dict.get("force_eplb", False),
            enable_profiler=model_config_dict.get("enable_profiler", False),
            enable_weight_nz=model_config_dict.get("enable_weight_nz", True),
            custom_params=model_config_dict.get("custom_params", {}),
        )
        model_config._validate()
        return model_config

    def _validate(self):
        """Validate model configuration consistency."""
        if self.exe_mode not in ["eager", "ge_graph", "npugraph_ex"]:
            raise ValueError(
                f"exe_mode={self.exe_mode} is not supported, expected one of ['eager', 'ge_graph', 'npugraph_ex']"
            )
        if self.enable_static_kernel and self.exe_mode != "npugraph_ex":
            raise ValueError("enable_static_kernel only supports exe_mode='npugraph_ex'")

        if self.exe_mode == "npugraph_ex" and os.getenv("TASK_QUEUE_ENABLE", "2") != "1":
            os.environ["TASK_QUEUE_ENABLE"] = "1"  # npugraph_ex only supports TASK_QUEUE_ENABLE 0 or 1
        else:
            os.environ["TASK_QUEUE_ENABLE"] = "2"  # 2: default value, opt host perf in eager



@dataclass
class ParallelConfig:
    """Unified parallel configuration for distributed inference.

    Attributes:
        world_size: Total number of processes (default: 1)
        global_rank: Global rank of this process (default: 0)
        local_rank: Local rank on the node (default: 0)
        attn_tp_size: Tensor parallelism size for attention (default: 1)
        moe_tp_size: Tensor parallelism size for MoE (default: 1)
        embed_tp_size: Tensor parallelism size for embedding (default: 1)
        lmhead_tp_size: Tensor parallelism size for LM head (default: 1)
        dense_tp_size: Tensor parallelism size for dense layers (default: 1)
        o_proj_tp_size: Tensor parallelism size for output projection (default: 1)
        moe_ep_size: Expert parallelism size for MoE (default: 1)
        cp_size: context parallelism size (default: 1)
        kvp_size: KV parallelism size (default: 1)
    """
    # Basic info
    world_size: int = 1
    global_rank: int = 0
    local_rank: int = 0

    attn_tp_size: int = 1
    attn_dp_size: int = 1  # This will be calculated based on world_size and attn_tp_size
    embed_tp_size: int = 1
    embed_dp_size: int = 1  # This will be calculated based on world_size and embed_tp_size
    moe_tp_size: int = 1
    moe_ep_size: int = 1  # This will be calculated based on world_size and moe_tp_size
    lmhead_tp_size: int = 1
    dense_tp_size: int = 1
    o_proj_tp_size: int = 1

    # Other parallelism
    cp_size: int = 1
    kvp_size: int = 1

    @classmethod
    def from_dict(cls, parallel_config_dict: dict, global_rank: int, local_rank: int) -> "ParallelConfig":
        """Create ParallelConfig from YAML-parsed dictionary."""
        parallel_config = cls(
            global_rank=global_rank,
            local_rank=local_rank,
            world_size=parallel_config_dict.get("world_size", 1),
            attn_tp_size=parallel_config_dict.get("attn_tp_size", 1),
            moe_tp_size=parallel_config_dict.get("moe_tp_size", 1),
            embed_tp_size=parallel_config_dict.get("embed_tp_size", 1),
            lmhead_tp_size=parallel_config_dict.get("lmhead_tp_size", 1),
            dense_tp_size=parallel_config_dict.get("dense_tp_size", 1),
            o_proj_tp_size=parallel_config_dict.get("o_proj_tp_size", 1),
            cp_size=parallel_config_dict.get("cp_size", 1),
            kvp_size=parallel_config_dict.get("kvp_size", 1),
        )
        parallel_config._validate()

        parallel_config.attn_dp_size = (
            parallel_config.world_size
            // (parallel_config.attn_tp_size * parallel_config.cp_size)
        )
        parallel_config.moe_ep_size = parallel_config.world_size // parallel_config.moe_tp_size
        parallel_config.embed_dp_size = parallel_config.world_size // parallel_config.embed_tp_size
        return parallel_config

    def _validate(self):
        """Validate parallel configuration consistency."""
        if self.world_size <= 0:
            raise ValueError(f"world_size must be positive, got {self.world_size}")

        if self.world_size % self.attn_tp_size != 0:
            raise ValueError(f"world_size={self.world_size} not divisible by attn_tp_size={self.attn_tp_size}")
        if self.world_size % self.moe_tp_size != 0:
            raise ValueError(f"world_size={self.world_size} not divisible by moe_tp_size={self.moe_tp_size}")
        if self.world_size % self.embed_tp_size != 0:
            raise ValueError(f"world_size={self.world_size} not divisible by embed_tp_size={self.embed_tp_size}")
        if self.world_size % self.lmhead_tp_size != 0:
            raise ValueError(f"world_size={self.world_size} not divisible by lmhead_tp_size={self.lmhead_tp_size}")
        if self.world_size % self.dense_tp_size != 0:
            raise ValueError(f"world_size={self.world_size} not divisible by dense_tp_size={self.dense_tp_size}")
        if self.world_size % self.o_proj_tp_size != 0:
            raise ValueError(f"world_size={self.world_size} not divisible by o_proj_tp_size={self.o_proj_tp_size}")
        if self.world_size % self.cp_size != 0:
            raise ValueError(f"world_size={self.world_size} not divisible by cp_size={self.cp_size}")
        attn_group_size = self.attn_tp_size * self.cp_size
        if self.world_size % attn_group_size != 0:
            raise ValueError(
                f"world_size={self.world_size} not divisible by "
                f"attn_tp_size*cp_size={attn_group_size}"
            )
        if self.world_size % self.kvp_size != 0:
            raise ValueError(f"world_size={self.world_size} not divisible by kvp_size={self.kvp_size}")

    def validate_model_config(self, hf_config):
        """Validate against model-specific configuration."""
        if hasattr(hf_config, "num_experts"):
            num_experts = getattr(hf_config, "num_experts", None)
            if num_experts is not None and self.moe_ep_size > 0:
                if num_experts % self.moe_ep_size != 0:
                    raise ValueError(f"num_experts={num_experts} not divisible by moe_ep_size={self.moe_ep_size}")

        if hasattr(hf_config, "num_attention_heads"):
            num_heads = getattr(hf_config, "num_attention_heads", None)
            if num_heads is not None and self.attn_tp_size > 0:
                if num_heads % self.attn_tp_size != 0:
                    raise ValueError(
                        f"num_attention_heads={num_heads} not divisible by attn_tp_size={self.attn_tp_size}"
                    )


@dataclass
class SchedulerConfig:
    """Configuration for the request scheduler.

    Attributes:
        batch_size: Global batch size across all ranks (default: 1)
        max_new_tokens: Maximum number of tokens to generate per request (default: 32)
        max_prefill_tokens: Maximum packed prompt tokens per prefill batch (default: 0, disabled)
        batch_size_per_dp_rank: Batch size per rank for distributed inference (default: 1)
        mem_fraction_static: Fraction of device memory reserved for static allocation (default: 0.85)
        block_size: Number of tokens contained in one KV cache block (default: 128)
        num_reserved_decode_tokens: Per-in-flight-request KV reservation used by
            online PD decode admission control.  Each request in the prealloc /
            transfer / running pipeline reserves this many tokens of KV space
            for its near-future decode steps; new requests are not admitted if
            granting their input + reservation would push the pool below this
            margin.  Smaller value = higher throughput but more retraction
            churn under bursty load.
    """
    batch_size: int = 1
    max_new_tokens: int = 32
    max_prefill_tokens: int = 0
    batch_size_per_dp_rank: int = 1  # This will be calculated based on batch_size and parallel config
    mem_fraction_static: float = 0.85
    block_size: int = 128
    num_reserved_decode_tokens: int = 64

    @classmethod
    def from_dict(cls, scheduler_config_dict: dict) -> "SchedulerConfig":
        """Create SchedulerConfig from YAML-parsed dictionary."""
        return cls(
            batch_size=scheduler_config_dict.get("batch_size", 1),
            max_new_tokens=scheduler_config_dict.get("max_new_tokens", 32),
            max_prefill_tokens=scheduler_config_dict.get("max_prefill_tokens", 0),
            mem_fraction_static=scheduler_config_dict.get("mem_fraction_static", 0.85),
            block_size=scheduler_config_dict.get("block_size", 128),
            num_reserved_decode_tokens=scheduler_config_dict.get(
                "num_reserved_decode_tokens", 64
            ),
        )


@dataclass
class DisaggConfig:
    """Configuration for PD disaggregation runtime."""

    disaggregation_mode: str = "NONE"
    bootstrap_host: str = "0.0.0.0"
    bootstrap_port: int = 18800
    # MemFabric config store address, tcp://<P primary IP>:<port>. All ranks
    # (prefill + decode, all instances) must agree on this single URL.
    store_url: str = ""
    # True only on the PREFILL instance-0 leader node — the sole node that
    # should run create_config_store. Set explicitly by server.py based on
    # node_index/role; combining with local_rank==0 inside the engine picks
    # exactly one worker out of the whole service.
    is_store_creator_node: bool = False
    # Local IP advertised to PD peers as this rank's contact address. Sourced
    # from the launch config (--ips[node_index]) so the auto-detection magic
    # never sees ambiguous multi-NIC hosts.
    local_ip: str = ""


@dataclass
class InferenceConfig:
    """Unified inference configuration loaded from YAML with structured access."""
    # Nested configuration objects
    model_config: ModelConfig
    data_config: DataConfig
    parallel_config: ParallelConfig
    scheduler_config: SchedulerConfig
    disagg_config: DisaggConfig = field(default_factory=DisaggConfig)

    @classmethod
    def from_dict(
        cls,
        yaml_dict: dict,
        global_rank: int,
        local_rank: int,
        disagg_config: "DisaggConfig | None" = None,
    ) -> "InferenceConfig":
        """Create InferenceConfig from YAML-parsed dictionary."""

        infer_config = cls(
            model_config=ModelConfig.from_dict(yaml_dict.get("model_config", {})),
            data_config=DataConfig.from_dict(yaml_dict.get("data_config", {})),
            parallel_config=ParallelConfig.from_dict(
                yaml_dict.get("parallel_config", {}),
                global_rank=global_rank,
                local_rank=local_rank
            ),
            scheduler_config=SchedulerConfig.from_dict(yaml_dict.get("scheduler_config", {})),
            disagg_config=disagg_config or DisaggConfig(),
        )

        infer_config.scheduler_config.batch_size_per_dp_rank = \
            infer_config.scheduler_config.batch_size // infer_config.parallel_config.attn_dp_size

        if infer_config.scheduler_config.max_prefill_tokens == 0:
            infer_config.scheduler_config.max_prefill_tokens = \
                infer_config.data_config.input_truncated_len * infer_config.scheduler_config.batch_size_per_dp_rank

        return infer_config

    def validate(self, hf_config=None):
        """Validate all configuration sections."""
        self.parallel_config.validate_model_config(hf_config)
