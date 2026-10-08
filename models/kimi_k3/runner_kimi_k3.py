# coding=utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

import logging
import os

import torch
from transformers import GenerationConfig

from executor.model_runner import ModelRunner
from executor.utils import override
from models.configuration_kimi_k3 import KimiLinearConfig
from models.dspark_registry import (
    DRAFT_MODEL_DSPARK_GQA,
    get_draft_model_type,
    get_dspark_config_class,
    get_dspark_model_class,
)
from models.modeling_kimi_k3 import KimiLinearForCausalLM
from models.weight_loader import get_model_loader, init_model_with_online_split_weight
from utils.tokenizer import KimiK3Tokenizer


class KimiK3Runner(ModelRunner):
    """Legacy model loader plus Kimi K3-specific offline setup."""

    def __init__(self, runner_settings):
        super().__init__(runner_settings)
        model_config = runner_settings.get("model_config", {})
        data_config = runner_settings.get("data_config", {})
        self.enable_static_kernel = runner_settings.get("model_config", {}).get(
            "enable_static_kernel", False
        )
        self.enable_cache_compile = runner_settings.get("model_config", {}).get(
            "enable_cache_compile", False
        )
        self.enable_superkernel = model_config.get("custom_params", {}).get(
            "enable_superkernel", False
        )
        self.prefill_mini_batch_size = model_config.get(
            "prefill_mini_batch_size", 0
        )
        self.prefill_chunk_size = model_config.get("prefill_chunk_size", 0)
        self.batch_size_per_rank = data_config.get(
            "batch_size_per_rank", data_config.get("batch_size", 1)
        )
        effective_prefill_batch = (
            self.prefill_mini_batch_size
            if self.prefill_mini_batch_size > 0
            else self.batch_size_per_rank
        )
        self.prefill_cycles = self.batch_size_per_rank // effective_prefill_batch

    @override
    def init_model(self):
        self.use_pretrained_model = self.runner_settings.get("model_config", {}).get(
            "with_ckpt", True
        )
        super().init_model(KimiLinearForCausalLM, KimiLinearConfig)
        self.hf_generation_config = self._load_generation_config()

    @override
    def _init_model_with_online_splited_weight(self, model, config, **kwargs):
        init_model_with_online_split_weight(self, model, config, **kwargs)

    def _load_generation_config(self):
        try:
            return GenerationConfig.from_pretrained(
                self.model_path,
                trust_remote_code=True,
            )
        except OSError:
            logging.debug("generation_config.json not found under %s", self.model_path)
            return None
        except Exception as error:
            logging.warning(
                "Failed to load generation_config.json from %s: %s; "
                "falling back to model and tokenizer EOS settings",
                self.model_path,
                error,
            )
            return None

    @override
    def _process_weight_after_loading(self):
        self.to_device()
        self.model.check_model_settings()
        self.model.process_weights_after_loading()

    @override
    def init_tokenizer(self):
        self.tokenizer = KimiK3Tokenizer.from_pretrained(
            self.model_path,
            padding_side="right",
            truncation_side="right",
            trust_remote_code=True,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

    @override
    def graph_compile(self):
        torch._dynamo.config.inline_inbuilt_nn_modules = False
        if self.execute_mode == "npugraph_ex":
            import torchair as tng
            import torchair.ge_concrete_graph.ge_converter.experimental.patch_for_hcom_allreduce

            tng.patch_for_hcom()
            options = {
                "frozen_parameter": True,
                "static_kernel_compile": self.enable_static_kernel,
                "unsafe_skip_npugraph_capture_validation": True,
            }
            if self.enable_superkernel:
                options["super_kernel_optimize"] = True
                options["super_kernel_optimize_options"] = {
                    "dcci_disable_on_kernel": [".*"],
                    "auto_op_parallel": 1
                }
            if self.enable_cache_compile:
                cache_dir = os.path.join(
                    os.path.dirname(os.path.abspath(__file__)),
                    "compile_cache",
                    self.model_name,
                )
                if self.enable_superkernel:
                    # Keep fused and unfused graph artifacts separate.
                    cache_dir = os.path.join(cache_dir, "superkernel")
                self.model.decode = torch.npu.npugraph_ex.inference.cache_compile(
                    self.model.decode,
                    cache_dir=cache_dir,
                    dynamic=False,
                    options=options,
                )
            else:
                self.model.decode = torch.compile(
                    self.model.decode,
                    dynamic=False,
                    fullgraph=True,
                    backend="npugraph_ex",
                    options=options,
                )
            return

        import torchair as tng
        from torchair.configs.compiler_config import CompilerConfig

        compiler_config = CompilerConfig()
        compiler_config.experimental_config.frozen_parameter = True
        compiler_config.experimental_config.tiling_schedule_optimize = True
        compiler_config.experimental_config.topology_sorting_strategy = "StableRDFS"
        backend = tng.get_npu_backend(compiler_config=compiler_config)
        self.model.decode = torch.compile(
            self.model.decode, dynamic=False, fullgraph=True, backend=backend
        )


class KimiK3DSparkRunner:
    """Load and compile the BF16 DSpark proposal model beside the legacy runner."""

    def __init__(self, runner_settings, main_runner: KimiK3Runner):
        self.runner_settings = runner_settings
        self.main_runner = main_runner
        self.draft_model_type = get_draft_model_type(runner_settings)
        self.model_name = f"{main_runner.model_name}_dspark"
        if self.draft_model_type != DRAFT_MODEL_DSPARK_GQA:
            self.model_name += "_mla"
        self.model_path = runner_settings["draft_model_path"]
        self.device = main_runner.device
        self.execute_mode = runner_settings.get("exe_mode", "eager")
        model_config = runner_settings.get("model_config", {})
        self.enable_static_kernel = model_config.get(
            "enable_static_kernel", False
        )
        self.enable_cache_compile = model_config.get("enable_cache_compile", False)
        self.model = None

    def init_model(self):
        config_cls = get_dspark_config_class(self.draft_model_type)
        model_cls = get_dspark_model_class(self.draft_model_type)
        logging.info(
            "Loading DSpark model type %s from %s",
            self.draft_model_type,
            self.model_path,
        )
        config = config_cls.from_pretrained(
            self.model_path,
            low_cpu_mem_usage=True,
            ignore_mismatched_sizes=True,
        )
        loader = get_model_loader(with_ckpt=True)
        self.model = loader.load_model(
            config=config,
            model_cls=model_cls,
            runner_settings=self.runner_settings,
            model_path=self.model_path,
            comm_manager=self.main_runner.model.comm_manager,
        )
        self.model.to(self.device)
        self.model.set_shared_target_modules(self.main_runner.model)
        self.model.check_model_settings()
        self.model.process_weights_after_loading()
        if "graph" in self.execute_mode:
            self.graph_compile()

    def graph_compile(self):
        torch._dynamo.config.inline_inbuilt_nn_modules = False
        if self.execute_mode == "npugraph_ex":
            import torchair as tng
            import torchair.ge_concrete_graph.ge_converter.experimental.patch_for_hcom_allreduce

            tng.patch_for_hcom()
            options = {
                "frozen_parameter": True,
                "static_kernel_compile": self.enable_static_kernel,
                "unsafe_skip_npugraph_capture_validation": True,
            }
            if self.enable_cache_compile:
                cache_dir = os.path.join(
                    os.path.dirname(os.path.abspath(__file__)),
                    "compile_cache",
                    self.model_name,
                )
                self.model.forward_spec_decode = torch.npu.npugraph_ex.inference.cache_compile(
                    self.model.forward_spec_decode,
                    cache_dir=cache_dir,
                    dynamic=False,
                    options=options,
                )
            else:
                self.model.forward_spec_decode = torch.compile(
                    self.model.forward_spec_decode,
                    dynamic=False,
                    fullgraph=True,
                    backend="npugraph_ex",
                    options=options,
                )
            return

        import torchair as tng
        from torchair.configs.compiler_config import CompilerConfig

        compiler_config = CompilerConfig()
        compiler_config.experimental_config.frozen_parameter = True
        compiler_config.experimental_config.tiling_schedule_optimize = True
        backend = tng.get_npu_backend(compiler_config=compiler_config)
        self.model.forward_spec_decode = torch.compile(
            self.model.forward_spec_decode,
            dynamic=False,
            fullgraph=True,
            backend=backend,
        )
