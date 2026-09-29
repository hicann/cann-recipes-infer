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
# GSM8K test (1319), AISBench gsm8k_gen_0_shot_cot_chat_prompt ('... "answer:$ANSWER" ...
# Let's think step by step.'). Max output 4096. Numeric match on the last "answer:"/"answer is"
# number (dsv41_eval/math_tasks.py).
_base_ = []  # keep mmengine in plain-python config mode (read_base would make every import lazy)
import os
from dsv41_eval.model import dsv41_models, limit_datasets
from dsv41_eval.math_tasks import Gsm8kNumericEvaluator, gsm8k_answer

from ais_bench.benchmark.configs.datasets.gsm8k.gsm8k_gen_0_shot_cot_chat_prompt import gsm8k_datasets

_d = dict(gsm8k_datasets[0])
_d["path"] = os.path.join(os.environ.get("AISBENCH_ROOT", ""), "ais_bench/datasets/gsm8k")
_d["eval_cfg"] = dict(_d["eval_cfg"], evaluator=dict(type=Gsm8kNumericEvaluator),
                      pred_postprocessor=dict(type=gsm8k_answer))

datasets = limit_datasets([_d])
models = dsv41_models(max_out_len=4096)
del gsm8k_datasets, _d, Gsm8kNumericEvaluator, gsm8k_answer
