# coding=utf-8
# Adapted from
# https://github.com/modelscope/evalscope/blob/main/examples/custom/custom_eval_model.py
# Copyright (c) 2022-2023 Alibaba ModelScope. All Rights Reserved.
# Modifications Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
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

import logging
import json
import argparse
import threading
from pathlib import Path
from typing import List, Optional, Dict, Any

import requests
from evalscope import run_task
from evalscope.config import TaskConfig

from evalscope.api.model import ModelAPI, GenerateConfig, ModelOutput
from evalscope.api.messages import ChatMessage
from evalscope.api.tool import ToolChoice, ToolInfo
from transformers import AutoTokenizer

# 配置日志格式
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 执行中间截断，并为服务端的 ChatTemplate 模板膨胀预留安全余量，
SAFETY_MARGIN = 4 * 1024

SPEC_COUNT_FIELDS = (
    "spec_num_accepted_tokens",
    "spec_num_draft_tokens",
    "spec_num_forward_ct",
)


class SpecStatsAccumulator:
    """Aggregate raw speculative counters for one benchmark run."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.num_requests_total = 0
        self.num_requests_with_spec = 0
        self._counts = dict.fromkeys(SPEC_COUNT_FIELDS, 0)

    def add(self, usage: Dict[str, Any]) -> None:
        with self._lock:
            self.num_requests_total += 1
            if "spec_num_accepted_tokens" not in usage:
                return
            self.num_requests_with_spec += 1
            for field in SPEC_COUNT_FIELDS:
                value = usage.get(field, 0) or 0
                try:
                    self._counts[field] += int(value)
                except (TypeError, ValueError) as exc:
                    raise ValueError(f"Invalid speculative usage field {field!r}: {value!r}") from exc

    def report(self) -> Dict[str, Any]:
        with self._lock:
            accepted = self._counts["spec_num_accepted_tokens"]
            draft = self._counts["spec_num_draft_tokens"]
            forward_ct = self._counts["spec_num_forward_ct"]
            return {
                "num_requests_total": self.num_requests_total,
                "num_requests_with_spec": self.num_requests_with_spec,
                **self._counts,
                "acceptance_rate": accepted / draft if draft else 0.0,
                "mean_accept_length": 1 + accepted / forward_ct if forward_ct else 1.0,
            }


class MyCustomModel(ModelAPI):
    """自定义模型实现"""

    def __init__(
            self,
            model_name: str,
            max_prefill_tokens: int,
            max_tokens: int,
            timeout: int,
            base_url: Optional[str] = None,
            api_key: Optional[str] = None,
            pretrained_model_name_or_path: Optional[str] = None,
            config: Optional[GenerateConfig] = None,
            **model_args: Dict[str, Any],
    ) -> None:
        if config is None:
            config = GenerateConfig()
        super().__init__(model_name, base_url, api_key, config)
        self.model_args = model_args
        self.max_prefill_tokens = max_prefill_tokens
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.model_name = model_name
        self.spec_stats = SpecStatsAccumulator()
        self.tokenizer = (
            AutoTokenizer.from_pretrained(pretrained_model_name_or_path, trust_remote_code=True)
            if pretrained_model_name_or_path else None
        )

    def generate(
            self,
            input: List[ChatMessage],  # pylint: disable=redefined-builtin
            tools: List[ToolInfo],
            tool_choice: ToolChoice,
            config: GenerateConfig,
    ) -> ModelOutput:
        # 3. 实现模型推理逻辑

        # 3.1 处理输入消息
        input_text = self._process_messages(input)

        # 3.2 调用模型
        response, usage = self._call_model(input_text, config)
        self.spec_stats.add(usage)

        # 3.3 返回标准化输出
        output = ModelOutput.from_content(
            model=self.model_name,
            content=response,
        )
        spec_meta = {field: usage[field] for field in SPEC_COUNT_FIELDS if field in usage}
        if spec_meta:
            output.metadata = {"spec_decoding": spec_meta}
        return output

    def _process_messages(self, messages: List[ChatMessage]) -> str:
        """将聊天消息转换为文本"""
        text_parts = []
        for message in messages:
            role = getattr(message, 'role', 'user')
            content = getattr(message, 'content', str(message))
            content = self.middle_truncation(content)
            text_parts.append(f"{role}: {content}")
        return "\n".join(text_parts)

    def middle_truncation(self, prompt: str) -> str:
        if self.tokenizer is None:
            return prompt
        input_ids = self.tokenizer.encode(prompt)
        if len(input_ids) <= self.max_prefill_tokens:
            return prompt

        global_budget = self.max_prefill_tokens - SAFETY_MARGIN
        input_ids = input_ids[:global_budget // 2] + input_ids[-global_budget // 2:]
        prompt = self.tokenizer.decode(input_ids, skip_special_tokens=True)
        return prompt

    def _call_model(self, input_text: str, config: GenerateConfig):
        """调用的模型进行推理"""
        output_text = ""
        try:
            output_text, usage = self.request_ai(input_text)
        except requests.exceptions.RequestException as e:
            logger.error(f"网络请求失败 (URL: {self.base_url}): {e}")
            # 向上抛出异常，让 EvalScope 框架识别为错误样本而不是得到一个空答案
            raise RuntimeError(f"Model API request failed due to network issue: {e}") from e
        except (json.JSONDecodeError, KeyError, IndexError) as e:
            logger.error(f"接口返回格式异常，解析失败: {e}")
            raise RuntimeError(f"Model API response parsing failed: {e}") from e

        return output_text, usage

    def request_ai(self, content: str):
        payload = json.dumps({
            "model": self.model_name,
            "messages": [
                {
                    "role": "user",
                    "content": content
                }
            ],
            "max_tokens": self.max_tokens
        })

        headers = {
            "Content-Type": "application/json"
        }
        response = requests.request("POST", self.base_url, headers=headers, data=payload, timeout=self.timeout)
        response.raise_for_status()  # 如果 HTTP 状态码不是 200，直接抛出异常
        body = response.json()
        return body["choices"][0]["message"]["content"], body.get("usage", {}) or {}


def print_spec_report(stats: Dict[str, Any]) -> None:
    """Print speculative decoding counters and acceptance metrics."""
    logger.info("%s", "=" * 60)
    logger.info("Speculative Decoding acceptance summary (this benchmark)")
    logger.info("%s", "=" * 60)
    logger.info("total requests                  : %s", stats["num_requests_total"])
    logger.info("requests with spec statistics   : %s", stats["num_requests_with_spec"])
    logger.info("accepted draft tokens           : %s", stats["spec_num_accepted_tokens"])
    logger.info("draft tokens                    : %s", stats["spec_num_draft_tokens"])
    logger.info("verification rounds             : %s", stats["spec_num_forward_ct"])
    logger.info(
        "acceptance rate (accepted/draft): %.4f (%.2f%%)",
        stats["acceptance_rate"],
        stats["acceptance_rate"] * 100,
    )
    logger.info("mean acceptance length         : %.4f", stats["mean_accept_length"])
    logger.info("%s", "=" * 60)


def main(args):
    # 创建模型实例
    custom_model = MyCustomModel(
        model_name=args.model_name,
        base_url=args.base_url,
        pretrained_model_name_or_path=args.model_path,
        max_prefill_tokens=args.max_prefill_tokens,
        max_tokens=args.max_tokens,
        timeout=args.timeout,
    )

    # 配置评测任务
    task_config = TaskConfig(
        model=custom_model,
        datasets=[args.datasets],
        generation_config={
            "max_tokens": args.max_tokens,
            "batch_size": args.batch_size,
            "timeout": args.timeout
        },
        eval_batch_size=args.batch_size,
        limit=args.limit,
    )

    # 运行评测
    if args.dataset_local_path:
        task_config.dataset_args = {args.datasets: {"local_path": args.dataset_local_path}}
    logger.info("Starting evaluation, datasets=%s", args.datasets)
    run_task(task_cfg=task_config)
    stats = custom_model.spec_stats.report()
    if stats["num_requests_with_spec"] > 0:
        print_spec_report(stats)
        # run_task updates work_dir to the output directory for this run.
        report_path = Path(task_config.work_dir) / "spec_stats.json"
        with report_path.open("w", encoding="utf-8") as report_file:
            json.dump(stats, report_file, ensure_ascii=False, indent=2)
        logger.info("Speculative decoding report saved to %s", report_path)


if __name__ == '__main__':
    # 1. 定义参数解析器
    parser = argparse.ArgumentParser(description="通过在线接口运行大模型 EvalScope 评测")
    parser.add_argument(
        "--datasets",
        type=str,
        default="longbench_v2",
        help="EvalScope dataset, e.g. longbench_v2, math_500, humaneval, or mmlu",
    )
    parser.add_argument(
        "--dataset-local-path",
        type=str,
        default="",
        help="Optional local dataset path passed to EvalScope",
    )

    # 模型相关参数
    parser.add_argument("--model-name", type=str, default='Qwen/Qwen3-235B-A22B', help="模型名称")
    parser.add_argument("--base-url", type=str, default="http://localhost:8000/v1/chat/completions",
                        help="模型 API 地址")
    parser.add_argument("--model-path", type=str, default="/data/models/Qwen3-235B-A22B", help="本地模型权重路径")

    # 显存/Token 配置参数
    parser.add_argument("--max-prefill-tokens", type=int, default=102400, help="最大 Prefill Token 数")
    parser.add_argument("--max-tokens", type=int, default=8192, help="生成最大 Token 数")

    # 评测与生成配置参数
    parser.add_argument("--batch-size", type=int, default=2, help="生成与评测的 Batch Size")
    parser.add_argument("--timeout", type=int, default=1200, help="请求超时时间(秒)")
    parser.add_argument("--limit", type=int, default=None, help="限制评测的数据条数（测试用，默认不限制）")

    main(parser.parse_args())
