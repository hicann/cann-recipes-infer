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

from models.gpt_oss.models.modeling_gpt_oss import GptOssForCausalLM
from models.gpt_oss.models.configuration_gpt_oss import GptOssConfig
from models.qwen3_moe.models.modeling_qwen3_moe import Qwen3MoeForCausalLM
from models.qwen3_moe.models.configuration_qwen3_moe import Qwen3MoeConfig

model_dict = {
    "gpt-oss": (GptOssForCausalLM, GptOssConfig),
    "qwen3-moe": (Qwen3MoeForCausalLM, Qwen3MoeConfig),
}
