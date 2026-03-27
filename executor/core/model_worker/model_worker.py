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

"""Model Worker for managing model loading and inference operations.

This module provides the ModelWorker class that handles low-level model operations
for a single model (either main model or MTP model), including:
- Model loading
- Weight processing and KV cache initialization
- Model inference execution
- Model compilation for graph mode
"""

import os
import logging
import time
from typing import Dict, Optional, Tuple

import torch
import torch_npu

from executor.core.config import InferenceConfig
from executor.utils.forward_metadata import set_forward_metadata, get_forward_metadata
from executor.model_loader.default_loader import DefaultModelLoader
from executor.model_loader.dummy_loader import DummyModelLoader
from module.quantization import (QUANTIZATION_METHODS, get_quant_config)

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)


class ModelWorker:
    """Worker class for managing a single model's loading and inference operations.

    This class encapsulates all low-level model operations that directly
    interact with the model, providing a clean separation between the
    execution engine orchestration and the actual model work.

    Each ModelWorker instance is responsible for one model.

    Member Variables:
        Config:
            infer_config: Inference configuration containing all runtime settings.
            hf_config: HuggingFace model configuration loaded from pretrained.
            model_name: Name of the model.
            model_path: Path to the model weights.
            exe_mode: Execution mode (eager, npugraph_ex, etc.).
            dtype: Data type for tensors (default: bfloat16).

        Model Components:
            device: NPU device for computation.
            model: Loaded inference model instance.
            model_compiled: Compiled model for graph mode (if enabled).
            comm_manager: Communication manager for distributed operations.
            quantization: Quantization method name (if applicable).

        Parallel Settings:
            global_rank: Global rank for distributed training.
            attn_tp_size: Attention tensor parallelism size.
            attn_dp_size: Attention data parallelism size.
            moe_ep_size: Mixture of Experts expert parallelism size.
            moe_tp_size: Mixture of Experts tensor parallelism size.

        MoE Load Balance:
            force_eplb: Flag for force expert per-token load balance.
            prefill_topk_list: Cached expert indices for prefill phase.
            decode_topk_list: Cached expert indices for decode phase.

        Runtime Flags:
            use_pretrained_model: Whether to load pretrained weights.
            enable_cache_compile: Whether to enable compiled graph cache.
    """

    def __init__(self, infer_config: InferenceConfig, device):
        """Initialize ModelWorker with inference configuration and device."""
        # Config
        self.infer_config = infer_config
        self.model_name = self.infer_config.model_config.model_name
        self.model_path = self.infer_config.model_config.model_path
        self.exe_mode = self.infer_config.model_config.exe_mode
        self.enable_static_kernel = self.infer_config.model_config.enable_static_kernel
        self.use_pretrained_model = self.infer_config.model_config.with_ckpt
        self.enable_cache_compile = self.infer_config.model_config.enable_cache_compile

        # Model Components
        self.device = device
        self.model = None
        self.model_compiled: Optional[torch.nn.Module] = None
        self.kv_cache: Optional[torch.Tensor] = None
        self.hf_config = None
        self.comm_manager = None
        self.dtype = torch.bfloat16
        self.quant_cache_dtype = None
        self.quantization = None

        # MoE Load Balance
        self.force_eplb = self.infer_config.model_config.force_eplb
        if self.force_eplb:
            self.prefill_topk_list = None
            self.decode_topk_list = None

        # Parallel Settings
        self.global_rank = self.infer_config.parallel_config.global_rank
        self.attn_tp_size = self.infer_config.parallel_config.attn_tp_size
        self.attn_dp_size = self.infer_config.parallel_config.attn_dp_size
        self.moe_ep_size = self.infer_config.parallel_config.moe_ep_size
        self.moe_tp_size = self.infer_config.parallel_config.moe_tp_size

    def load_model(
        self,
        model_cls,
        config_cls,
        comm_manager,
    ):
        """Load model from weights, process weights, and initialize KV cache."""
        # Select appropriate loader based on configuration
        if self.use_pretrained_model:
            logging.info("Try to load pretrained model in path: %s", self.model_path)
            loader = DefaultModelLoader()
        else:
            loader = DummyModelLoader()

        # Load HuggingFace config
        self.hf_config = config_cls.from_pretrained(
            self.model_path,
            low_cpu_mem_usage=True,
            ignore_mismatched_sizes=True,
            runner_settings=self.infer_config
        )

        # Handle quantization configuration
        self._verify_quantization()
        if self.quantization is not None:
            self.hf_config.quant_config = get_quant_config(self.hf_config, self.quantization, self.model_path)
            # Set dtype to int8 if KV cache quant mode is explicitly set to "int8"
            if self.hf_config.quant_config.kv_cache_quant_mode is not None and \
                self.hf_config.quant_config.kv_cache_quant_mode == "int8":
                self.quant_cache_dtype = torch.int8

        # Validate configuration
        self.infer_config.validate(self.hf_config)

        # Load model
        self.model = loader.load_model(
            config=self.hf_config,
            model_cls=model_cls,
            runner_settings=self.infer_config,
            model_path=self.model_path,
            comm_manager=comm_manager
        )

        # Check model settings
        if hasattr(self.model, "check_model_settings"):
            self.model.check_model_settings()

        self.model.to(self.device)

        # Process weights after loading (transpose for NPU)
        self._process_weights_after_loading()

        # Initialize KV cache
        self._init_kvcache()

    def _process_weights_after_loading(self):
        """Process weights after loading (transpose for NPU format)."""
        if hasattr(self.model, "process_weights_after_loading"):
            self.model.process_weights_after_loading()
            self.model.to(self.device)

    def _init_kvcache(self):
        """Initialize pre-allocated KV cache tensors."""
        batch_size = self.infer_config.scheduler_config.batch_size_per_dp_rank
        cache_seq_len = self.infer_config.scheduler_config.input_max_len + \
            self.infer_config.scheduler_config.max_new_tokens

        # Bind kv_cache to model modules
        if hasattr(self.model, "init_cache"):
            self.model.init_cache(self.device)
        else:
            for module in self.model.modules():
                # Standard KV cache
                if hasattr(module, "k_cache") and hasattr(module, "v_cache"):
                    module.k_cache = torch.empty((batch_size, cache_seq_len, *module.cache_unit), dtype=self.dtype,
                                                device=self.device)
                    module.v_cache = torch.empty((batch_size, cache_seq_len, *module.cache_unit), dtype=self.dtype,
                                                device=self.device)

    # Copied from vllm.config._parse_quant_hf_config
    def _parse_quant_hf_config(self):
        """Parse quantization configuration from HuggingFace config."""
        quant_cfg = getattr(self.hf_config, "quantization_config", None)
        if quant_cfg is None:
            # compressed-tensors uses a "compression_config" key
            quant_cfg = getattr(self.hf_config, "compression_config", None)
        return quant_cfg

    # Adapted from vllm.config._verify_quantization
    def _verify_quantization(self) -> None:
        """Verify quantization configuration is supported."""
        supported_quantization = QUANTIZATION_METHODS

        # Parse quantization method from the HF model config, if available.
        quant_cfg = self._parse_quant_hf_config()

        if quant_cfg is not None and quant_cfg:
            quant_method = quant_cfg.get("quant_method", "").lower()
            quant_method = quant_method.replace("compressed_tensors", "compressed-tensors")
            self.quantization = quant_method

        # Verify quantization is supported
        if self.quantization is not None:
            if self.quantization not in supported_quantization:
                raise ValueError(
                    f"Unknown quantization method: {self.quantization}. Must "
                    f"be one of {supported_quantization}.")

    def inference(self, model_inputs: Dict, is_prefill: bool, is_mtp: bool = False) -> Tuple[torch.Tensor, float]:
        """Execute model inference and log timing information."""
        # Generate expert indices for force EPLB if enabled
        if self.force_eplb:
            batch_size, seq_len = model_inputs["input_ids"].shape
            if is_prefill:
                self.prefill_topk_list = self.gen_force_eplb_topk_idx(is_prefill, batch_size, seq_len)
            elif self.decode_topk_list is None:
                # Only generate once for decode phase
                self.decode_topk_list = self.gen_force_eplb_topk_idx(is_prefill, batch_size, seq_len)
            model_inputs.update({"cur_topk_list": self.prefill_topk_list if is_prefill else self.decode_topk_list})

        # Synchronize and start timing
        torch.npu.synchronize()
        start_time = time.time()

        # Execute model forward
        with torch.no_grad():
            if self.exe_mode in ["ge_graph", "npugraph_ex"] and not is_prefill:
                # Use compiled model for decode phase
                output = self.model_compiled(**model_inputs)
            else:
                # Use eager execution for prefill or non-graph mode
                output = self.model(**model_inputs)

        # Synchronize and calculate timing
        torch.npu.synchronize()
        end_time = time.time()
        inference_time = end_time - start_time

        # Determine stage name for logging
        model = "Main" if not is_mtp else "MTP"
        if not is_prefill:
            stage = "decode"
        else:
            if model_inputs.get("cycle_idx"):
                stage = "prefill round " + str(model_inputs.get("cycle_idx"))
            else:
                stage = "prefill"
        logging.info(f"[{model}] Inference time ({stage}): {inference_time*1000:.2f} ms")

        return output, inference_time

    def compile_model(self):
        """Compile model forward for graph mode."""
        from executor.utils.graph_utils import compile_model_forward

        logging.info("The final model structure is: \n %s", self.model)
        if self.exe_mode in ["ge_graph", "npugraph_ex"]:
            logging.info("Try to compile model")
            self.model_compiled = compile_model_forward(
                self.model.forward,
                exe_mode=self.exe_mode,
                enable_cache_compile=self.enable_cache_compile,
                cache_dir=os.path.join(self.infer_config.model_config.output_path, "cache_compile"),
                enable_static_kernel=self.enable_static_kernel,
            )
        else:
            self.model_compiled = None

    def gen_force_eplb_topk_idx(
        self,
        is_prefill: bool,
        batch_size: int,
        seq_len: int,
    ) -> Optional[torch.Tensor]:
        """Generate expert indices for force EPLB (Expert Per-Token Load Balance).

        This function uses round-robin allocation to evenly distribute tokens
        across experts, ensuring perfect load balance. The allocation pattern
        depends on the phase (prefill/decode) and parallel configuration.
        """
        # Validate model has required MoE attributes
        if not hasattr(self.model, "num_experts"):
            raise AttributeError(
                f"Model {self.model.__class__.__name__} must configure 'num_experts' attribute for EPLB. "
                f"Please add 'self.num_experts = <value>' in the model's __init__ method."
            )

        if not hasattr(self.model, "num_experts_per_tok"):
            raise AttributeError(
                f"Model {self.model.__class__.__name__} must configure 'num_experts_per_tok' attribute for EPLB. "
                f"Please add 'self.num_experts_per_tok = <value>' in the model's __init__ method."
            )

        num_experts = self.model.num_experts
        num_experts_per_tok = self.model.num_experts_per_tok
        experts_per_rank = num_experts // self.moe_ep_size

        if is_prefill:
            # Prefill phase: allocate experts with rank-aware distribution
            tokens_per_rank_prefill = (batch_size * seq_len + self.attn_tp_size - 1) // self.attn_tp_size \
                if self.moe_ep_size != 1 else batch_size * seq_len * self.attn_dp_size
            step_prefill = tokens_per_rank_prefill * num_experts_per_tok
            cur_topk_list_prefill = [
                (i + self.global_rank) % num_experts for i in range(step_prefill)]
            cur_topk_list = torch.tensor(cur_topk_list_prefill, dtype=torch.int).view(tokens_per_rank_prefill, -1).npu()
        else:
            # Decode phase: allocation depends on MoE tensor parallelism
            if self.moe_tp_size > 1:
                # MoE TP mode: contiguous expert ranges per EP rank
                expanded_tokens = batch_size * num_experts_per_tok * seq_len
                cur_topk_list_decode = []
                for offset in range(self.moe_ep_size):
                    expert_start = offset * experts_per_rank
                    expert_end = expert_start + expanded_tokens
                    cur_topk_list_decode = cur_topk_list_decode + [i for i in range(expert_start, expert_end)]
                cur_topk_list = torch.tensor(cur_topk_list_decode, dtype=torch.int).view(batch_size * seq_len, -1).npu()
            else:
                # Non-TP mode: round-robin allocation across EP ranks
                expanded_tokens = batch_size * num_experts_per_tok * seq_len
                step_gap = num_experts // self.moe_ep_size
                expanded_offset = expanded_tokens * self.global_rank + self.global_rank

                cur_topk_list_decode = []
                # Allocate experts using round-robin algorithm
                for idx in range(expanded_tokens):
                    col = (expanded_offset + idx) % self.moe_ep_size
                    row = (expanded_offset + idx) // self.moe_ep_size % step_gap
                    expert_idx = row + col * step_gap
                    cur_topk_list_decode.append(expert_idx)
                cur_topk_list = torch.tensor(cur_topk_list_decode, dtype=torch.int).view(batch_size * seq_len, -1).npu()
        return cur_topk_list
