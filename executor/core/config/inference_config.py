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

    @classmethod
    def from_dict(cls, data_config_dict: dict) -> "DataConfig":
        """Create DataConfig from YAML-parsed dictionary."""
        return cls(
            dataset=data_config_dict.get("dataset", "default"),
            dataset_path=data_config_dict.get("dataset_path", ""),
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

        exe_mode: Execution mode (eager, ge_graph, acl_graph) (default: "eager")
        enable_cache_compile: Enable cache compilation (default: False)

        micro_batch_mode: Micro batch mode (default: 0)
        perfect_eplb: Whether to enable perfect expert load balancing for MoE models (default: False)

        enable_profiler: Enable profiler (default: False)
        packed_sequence: Whether input sequences are packed (batch+seq merged) (default: True)
        enable_weight_nz: Whether to enable NZ format for weights (default: True)
    """
    model_name: str = "model"
    model_path: str = ""
    output_path: str = ""
    dtype: str = "bfloat16"
    with_ckpt: bool = True
    next_n: int = 0

    exe_mode: str = "eager"
    enable_cache_compile: bool = False

    micro_batch_mode: int = 0
    perfect_eplb: bool = False

    enable_profiler: bool = False

    packed_sequence: bool = True
    enable_weight_nz: bool = True

    @classmethod
    def from_dict(cls, model_config_dict: dict) -> "ModelConfig":
        """Create ModelConfig from YAML-parsed dictionary."""
        return cls(
            model_name=model_config_dict.get("model_name", "model"),
            model_path=model_config_dict.get("model_path", ""),
            output_path=model_config_dict.get("output_path", ""),
            dtype=model_config_dict.get("dtype", "bfloat16"),
            with_ckpt=model_config_dict.get("with_ckpt", True),
            next_n=model_config_dict.get("next_n", 0),
            exe_mode=model_config_dict.get("exe_mode", "eager"),
            enable_cache_compile=model_config_dict.get("enable_cache_compile", False),
            micro_batch_mode=model_config_dict.get("micro_batch_mode", 0),
            perfect_eplb=model_config_dict.get("perfect_eplb", False),
            enable_profiler=model_config_dict.get("enable_profiler", False),
            packed_sequence=model_config_dict.get("packed_sequence", True),
            enable_weight_nz=model_config_dict.get("enable_weight_nz", True),
        )


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

        parallel_config.attn_dp_size = parallel_config.world_size // parallel_config.attn_tp_size
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
        input_max_len: Maximum input sequence length (default: 1024)
        max_new_tokens: Maximum number of tokens to generate per request (default: 32)
        batch_size_per_dp_rank: Batch size per rank for distributed inference (default: 1)
    """
    batch_size: int = 1
    input_max_len: int = 1024
    max_new_tokens: int = 32
    batch_size_per_dp_rank: int = 1  # This will be calculated based on batch_size and parallel config

    @classmethod
    def from_dict(cls, scheduler_config_dict: dict) -> "SchedulerConfig":
        """Create SchedulerConfig from YAML-parsed dictionary."""
        return cls(
            batch_size=scheduler_config_dict.get("batch_size", 1),
            input_max_len=scheduler_config_dict.get("input_max_len", 1024),
            max_new_tokens=scheduler_config_dict.get("max_new_tokens", 32),
        )


@dataclass
class InferenceConfig:
    """Unified inference configuration (replaces runner_settings dictionary).

    This class encapsulates all configuration from YAML file with structured access.
    """
    # Nested configuration objects
    model_config: ModelConfig
    data_config: DataConfig
    parallel_config: ParallelConfig
    scheduler_config: SchedulerConfig

    @classmethod
    def from_dict(cls, yaml_dict: dict, global_rank: int, local_rank: int) -> "InferenceConfig":
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
        )
    
        infer_config.scheduler_config.batch_size_per_dp_rank = \
            infer_config.scheduler_config.batch_size // infer_config.parallel_config.attn_dp_size
        return infer_config

    def validate(self, hf_config=None):
        """Validate all configuration sections."""
        self.parallel_config.validate_model_config(hf_config)
