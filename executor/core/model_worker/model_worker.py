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

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)


class ModelWorker:
    """Worker class for managing a single model's loading and inference operations.

    This class encapsulates all low-level model operations that directly
    interact with the model, providing a clean separation between the
    execution engine orchestration and the actual model work.

    Each ModelWorker instance is responsible for one model.

    Attributes:
        device: NPU device for computation.
        model: Inference model instance.
        model_compiled: Compiled model for graph mode (if enabled).
        kv_cache: KV cache tensors for this model.
        hf_config: HuggingFace model configuration.
        infer_config: Inference configuration.
        exe_mode: Execution mode (eager, acl_graph, etc.).
        model_path: Path to the model weights.
        local_rank: Local rank for distributed training.
        global_rank: Global rank for distributed training.
        world_size: Total number of processes.
        comm_manager: Communication manager for distributed operations.
    """

    def __init__(self, infer_config: InferenceConfig, device):
        """Initialize ModelWorker.

        Args:
            infer_config: Inference configuration.
            device: NPU device for computation.
        """
        self.infer_config = infer_config
        self.model_name = self.infer_config.model_config.model_name
        self.model_path = self.infer_config.model_config.model_path
        self.exe_mode = self.infer_config.model_config.exe_mode
        self.use_pretrained_model = self.infer_config.model_config.with_ckpt
        self.enable_cache_compile = self.infer_config.model_config.enable_cache_compile
        # Model components
        self.device = device
        self.model = None
        self.model_compiled: Optional[torch.nn.Module] = None
        self.kv_cache: Optional[torch.Tensor] = None
        self.hf_config = None
        self.comm_manager = None
        self.dtype = torch.bfloat16

    def load_model(
        self,
        model_cls,
        config_cls,
        comm_manager,
    ):

        if self.use_pretrained_model:
            logging.info("Try to load pretrained model in path: %s", self.model_path)
            loader = DefaultModelLoader()
        else:
            loader = DummyModelLoader()

        self.hf_config = config_cls.from_pretrained(
            self.model_path,
            low_cpu_mem_usage=True,
            ignore_mismatched_sizes=True,
            runner_settings=self.infer_config
        )

        # Validate config
        self.infer_config.validate(self.hf_config)

        # Load model
        self.model = loader.load_model(
            config=self.hf_config,
            model_cls=model_cls,
            runner_settings=self.infer_config,
            model_path=self.model_path,
            comm_manager=comm_manager
        )

        # Process weights after loading (transpose weights for NPU)
        self._process_weights_after_loading()

        # Initialize KV cache
        self._init_kvcache()

    def _process_weights_after_loading(self):
        """Process weights after loading (transpose for NPU format)."""
        self.model.to(self.device)
        if hasattr(self.model, "process_weights_after_loading"):
            self.model.process_weights_after_loading()

    def _init_kvcache(self):
        """Initialize pre-allocated KV cache tensors."""
        batch_size = self.infer_config.scheduler_config.batch_size_per_dp_rank
        cache_seq_len = self.infer_config.scheduler_config.input_max_len + \
            self.infer_config.scheduler_config.max_new_tokens
        # For TP+DP mode, KV cache needs to store all_gathered data
        # batch_size_per_dp_rank is already the per-rank batch size after DP split
        # No need to multiply by attn_tp_size since input will be split in model forward

        # Bind kv_cache to model modules
        for module in self.model.modules():
            if hasattr(module, "k_cache") and hasattr(module, "v_cache"):
                module.k_cache = torch.empty((batch_size, cache_seq_len, *module.cache_unit), dtype=self.dtype,
                                             device=self.device)
                module.v_cache = torch.empty((batch_size, cache_seq_len, *module.cache_unit), dtype=self.dtype,
                                             device=self.device)

    def inference(self, model_inputs: Dict, is_prefill: bool) -> Tuple[torch.Tensor, float]:
        """Execute model inference.

        Args:
            model_inputs: Dictionary containing model inputs.
            is_prefill: Whether in prefill phase.

        Returns:
            Tuple of (logits, inference_time).
        """
        torch.npu.synchronize()
        start_time = time.time()

        with torch.no_grad():
            if "graph" in self.exe_mode and not is_prefill:
                logits = self.model_compiled(**model_inputs)
            else:
                logits = self.model(**model_inputs)

        torch.npu.synchronize()
        end_time = time.time()
        inference_time = end_time - start_time

        stage = "prefill" if is_prefill else "decode"
        logging.info(f"Inference time ({stage}): {inference_time*1000:.2f} ms")

        return logits, inference_time

    def compile_model(self):
        """Compile model forward for graph mode."""
        from executor.utils.graph_utils import compile_model_forward

        logging.info("The final model structure is: \n %s", self.model)
        if "graph" in self.exe_mode:
            logging.info("Try to compile model")
            self.model_compiled = compile_model_forward(
                self.model.forward,
                exe_mode=self.exe_mode,
                enable_cache_compile=self.enable_cache_compile,
                cache_dir=os.path.join(self.infer_config.model_config.output_path, "cache_compile"),
            )
        else:
            self.model_compiled = None
