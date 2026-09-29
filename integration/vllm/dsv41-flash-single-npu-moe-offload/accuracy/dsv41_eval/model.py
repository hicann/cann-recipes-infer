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
"""Shared AISBench service-model config for the DSv4.1 vLLM OpenAI server.

Every knob is an environment variable so the run_<dataset>.sh scripts stay one-liners:
  HOST (127.0.0.1)  PORT (8238)  MODEL_NAME (dsv41)
  CONCURRENCY (128)      -> AISBench batch_size = max in-flight requests
  MAX_OUT_LEN            -> per-dataset default passed by the config, env overrides
  TEMPERATURE (0.0)      -> greedy by default (deterministic A/B between MXFP8 and HiF8)
  THINKING (0)           -> 0: chat mode (chat_template_kwargs thinking=False)
                            1: thinking mode, REASONING_EFFORT (high) low|high|xhigh|max|1-100
  STREAM (0)             -> 1 streams (per-token latency in AISBench details), 0 plain JSON
  RETRY (2)
The per-request client timeout is AISBENCH_REQUEST_TIMEOUT (seconds), read by the patched
ais_bench/benchmark/global_consts.py (see setup_env.sh).
"""
import os

from ais_bench.benchmark.models import VLLMCustomAPIChat
from ais_bench.benchmark.utils.postprocess.model_postprocessors import extract_non_reasoning_content


def _env(name, default):
    v = os.environ.get(name)
    return default if v is None or v == "" else v


def dsv41_models(max_out_len, abbr_suffix=""):
    thinking = _env("THINKING", "0") == "1"
    gen = dict(
        temperature=float(_env("TEMPERATURE", "0.0")),
        top_p=float(_env("TOP_P", "1.0")),
        ignore_eos=False,
        chat_template_kwargs=dict(thinking=thinking),
    )
    if thinking:
        effort = _env("REASONING_EFFORT", "high")
        gen["chat_template_kwargs"]["reasoning_effort"] = int(effort) if effort.isdigit() else effort
    if _env("SEED", ""):
        gen["seed"] = int(os.environ["SEED"])
    tag = _env("TAG", "dsv41")
    return [
        dict(
            attr="service",
            type=VLLMCustomAPIChat,
            abbr=f"{tag}{abbr_suffix}",
            path="",  # tokenizer not needed client-side (no client truncation)
            model=_env("MODEL_NAME", "dsv41"),
            stream=_env("STREAM", "0") == "1",
            request_rate=0,
            use_timestamp=False,
            retry=int(_env("RETRY", "2")),
            api_key="",
            host_ip=_env("HOST", "127.0.0.1"),
            host_port=int(_env("PORT", "8238")),
            url="",
            max_out_len=int(_env("MAX_OUT_LEN", str(max_out_len))),
            batch_size=int(_env("CONCURRENCY", "128")),
            trust_remote_code=False,
            generation_kwargs=gen,
            # strips any <think>...</think> / reasoning text before answer extraction
            pred_postprocessor=dict(type=extract_non_reasoning_content),
        )
    ]


def limit_datasets(datasets):
    """NUM_SAMPLES=n keeps the first n items of every dataset (smoke tests)."""
    n = _env("NUM_SAMPLES", "")
    if not n:
        return datasets
    out = []
    for d in datasets:
        d = dict(d)
        d["reader_cfg"] = dict(d.get("reader_cfg", {}), test_range=f"[0:{int(n)}]")
        out.append(d)
    return out
