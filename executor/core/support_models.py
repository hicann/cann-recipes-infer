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


"""Lazy registry mapping model name to (ForCausalLM, [ModelMTP,] Config) classes."""

import importlib

_specs: dict[str, list[tuple[str, str]]] = {
    "deepseek_r1": [
        ("models.deepseek_r1.models.modeling_deepseek", "DeepseekV3ForCausalLM"),
        ("models.deepseek_r1.models.modeling_deepseek", "DeepseekV3ModelMTP"),
        ("models.deepseek_r1.models.configuration_deepseek", "DeepseekV3Config"),
    ],
    "deepseek_v3_2_exp": [
        ("models.deepseek_v3_2_exp.models.modeling_deepseek", "DeepseekV3ForCausalLM"),
        ("models.deepseek_v3_2_exp.models.modeling_deepseek", "DeepseekV3ModelMTP"),
        ("models.deepseek_v3_2_exp.models.configuration_deepseek", "DeepseekV3IndexConfig"),
    ],
    "deepseek_v2_lite": [
        ("models.deepseek_r1.models.modeling_deepseek", "DeepseekV3ForCausalLM"),
        ("models.deepseek_r1.models.configuration_deepseek", "DeepseekV3Config"),
    ],
    "gemma_4": [
        ("models.gemma_4.models.modeling_gemma4", "Gemma4ForCausalLM"),
        ("models.gemma_4.models.configuration_gemma4", "Gemma4TextConfig"),
    ],
    "gpt_oss": [
        ("models.gpt_oss.models.modeling_gpt_oss", "GptOssForCausalLM"),
        ("models.gpt_oss.models.configuration_gpt_oss", "GptOssConfig"),
    ],
    "hy3_preview": [
        ("models.hy3_preview.models.modeling_hy_v3", "HYV3ForCausalLM"),
        ("models.hy3_preview.models.configuration_hy_v3", "HYV3Config"),
    ],
    "kimi_k2": [
        ("models.deepseek_r1.models.modeling_deepseek", "DeepseekV3ForCausalLM"),
        ("models.deepseek_r1.models.configuration_deepseek", "DeepseekV3Config"),
    ],
    "kimi_k2_thinking": [
        ("models.kimi_k2_thinking.models.modeling_deepseek", "DeepseekV3ForCausalLM"),
        ("models.kimi_k2_thinking.models.modeling_deepseek", "DeepseekV3ModelMTP"),
        ("models.kimi_k2_thinking.models.configuration_deepseek", "DeepseekV3Config"),
    ],
    "longcat_flash_lite": [
        ("models.longcat_flash_lite.models.modeling_longcat_flash_lite", "LongcatFlashNgramForCausalLM"),
        ("models.longcat_flash_lite.models.configuration_longcat_flash_lite", "LongcatFlashNgramConfig"),
    ],
    "qwen25_7b_instruct": [
        ("models.qwen.models.modeling_qwen", "QwenForCausalLM"),
        ("models.qwen.models.configuration_qwen", "Qwen2Config"),
    ],
    "qwen3_8b": [
        ("models.qwen.models.modeling_qwen", "QwenForCausalLM"),
        ("models.qwen.models.configuration_qwen", "Qwen3Config"),
    ],
    "qwen3_moe": [
        ("models.qwen3_moe.models.modeling_qwen3_moe", "Qwen3MoeForCausalLM"),
        ("models.qwen3_moe.models.configuration_qwen3_moe", "Qwen3MoeConfig"),
    ],
    "step3p7_flash": [
        ("models.step3p7_flash.models.modeling_step3p7", "Step3p5ForCausalLM"),
        ("models.step3p7_flash.models.configuration_step3p7", "Step3p7TextConfig"),
    ],
    "qwen3.5": [
        ("models.qwen3_5.models.modeling_qwen3_5_moe", "Qwen3_5MoeForCausalLM"),
        ("models.qwen3_5.models.configuration_qwen3_5_moe", "Qwen3_5MoeTextConfig"),
    ],
    "longcat_flash": [
        ("models.longcat_flash.models.modeling_longcat_flash", "LongcatFlashForCausalLM"),
        ("models.longcat_flash.models.modeling_longcat_flash", "LongcatFlashModelMTP"),
        ("models.longcat_flash.models.configuration_longcat_flash", "LongcatFlashConfig"),
    ],
    "longcat_flash_ffn": [
        ("models.longcat_flash.models.ffn", "FFNForCausalLM"),
        ("models.longcat_flash.models.configuration_longcat_flash", "LongcatFlashConfig"),
    ],
}


def load_model_classes(name: str) -> tuple:
    if name not in _specs:
        raise ValueError(f"Unsupported model: {name}")
    try:
        return tuple(getattr(importlib.import_module(m), a) for m, a in _specs[name])
    except Exception as e:
        raise ImportError(f"failed to load model '{name}': {e}") from e
