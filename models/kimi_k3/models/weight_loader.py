# coding=utf-8
# Adapted from executor/model_loader/default_loader.py and weight_utils.py.
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-License-Identifier: Apache-2.0

import logging
import re
import time

from safetensors.torch import safe_open
from tqdm.auto import tqdm

from executor.model_loader.default_loader import DefaultModelLoader
from executor.model_loader.dummy_loader import DummyModelLoader
from module.quantization import get_quant_config


_EXPERT_WEIGHT_PATTERN = re.compile(r"(?:^|\.)experts\.(\d+)\.")


def get_model_loader(with_ckpt: bool):
    """Select checkpoint loading or dummy weights for K3."""
    return KimiK3ModelLoader() if with_ckpt else DummyModelLoader()


def init_model_with_online_split_weight(runner, model, config, **kwargs):
    """Keep the legacy initialization flow with K3's filtered loader."""
    if config is None:
        raise ValueError("config cannot be None")
    logging.info("Initializing model from %s (with_ckpt=%s)", runner.model_path, runner.use_pretrained_model)
    loader = get_model_loader(with_ckpt=runner.use_pretrained_model)
    runner.hf_config = config.from_pretrained(
        runner.model_path,
        low_cpu_mem_usage=True,
        ignore_mismatched_sizes=True,
        runner_settings=runner.runner_settings,
    )
    runner._verify_quantization()
    if runner.quantization is not None:
        runner.hf_config.quant_config = get_quant_config(
            runner.hf_config, runner.quantization, runner.model_path
        )
    runner.check_model_cfg()
    runner.update_model_cfg()
    runner.model = loader.load_model(
        config=runner.hf_config,
        model_cls=model,
        runner_settings=runner.runner_settings,
        model_path=runner.model_path,
        **kwargs,
    )


def is_local_expert_weight(name: str, expert_start: int, expert_end: int) -> bool:
    """Keep non-expert weights and experts owned by this EP rank."""
    match = _EXPERT_WEIGHT_PATTERN.search(name)
    return match is None or expert_start <= int(match.group(1)) < expert_end


class KimiK3ModelLoader(DefaultModelLoader):
    """Filter K3 checkpoint tensors before reading them from safetensors."""

    def get_all_weights(self, model_path, model):
        weight_filter = getattr(model, "should_load_weight", None)
        primary_weights = self.Source(
            model_path,
            None,
            prefix="",
            fall_back_to_pt=getattr(model, "fall_back_to_pt_during_load", True),
            allow_patterns_overrides=getattr(model, "allow_patterns_overrides", None),
        )
        yield from self._get_weights_iterator(primary_weights, weight_filter)
        for source in getattr(model, "secondary_weights", ()):
            yield from self._get_weights_iterator(source, weight_filter)

    def _get_weights_iterator(self, source, weight_filter=None):
        if weight_filter is None:
            yield from super()._get_weights_iterator(source)
            return

        _, weight_files, _ = self._prepare_weights(
            source.model_or_path,
            source.revision,
            source.fall_back_to_pt,
            source.allow_patterns_overrides,
        )
        if self.counter_before_loading_weights == 0.0:
            self.counter_before_loading_weights = time.perf_counter()
        for weight_file in tqdm(
            weight_files,
            desc="Loading safetensors checkpoint shards",
            bar_format="{desc}: {percentage:3.0f}% Completed | {n_fmt}/{total_fmt} "
                       "[{elapsed}<{remaining}, {rate_fmt}]\n",
        ):
            with safe_open(weight_file, framework="pt") as checkpoint:
                for name in checkpoint.keys():
                    if weight_filter(name):
                        yield source.prefix + name, checkpoint.get_tensor(name)
