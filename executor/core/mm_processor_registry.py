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

"""Lazy registry for optional model-specific multimodal processors."""

import importlib


_MM_PROCESSOR_SPECS: dict[str, tuple[str, str]] = {
    "deepseek_v4_1":(
        "models.deepseek_v4_1.utils.image_processor",
        "DeepseekV41ImageProcessor",
    ),
}


def get_mm_processor(
    model_name: str,
    model_path: str,
    tokenizer,
    input_truncated_len: int | None = None,
):
    spec = _MM_PROCESSOR_SPECS.get(model_name)
    if spec is None:
        return None
    module_name, class_name = spec
    processor_cls = getattr(importlib.import_module(module_name), class_name)
    processor = processor_cls(
        model_path=model_path,
        tokenizer=tokenizer,
    )
    processor.input_truncated_len = input_truncated_len
    return processor
