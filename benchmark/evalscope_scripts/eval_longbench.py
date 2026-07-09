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
        self.tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path,
            trust_remote_code=True
        )

    def generate(
            self,
            input: List[ChatMessage],
            tools: List[ToolInfo],
            tool_choice: ToolChoice,
            config: GenerateConfig,
    ) -> ModelOutput:
        # 3. 实现模型推理逻辑

        # 3.1 处理输入消息
        input_text = self._process_messages(input)

        # 3.2 调用模型
        response = self._call_model(input_text, config)

        # 3.3 返回标准化输出
        return ModelOutput.from_content(
            model=self.model_name,
            content=response
        )

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
        input_ids = self.tokenizer.encode(prompt)
        if len(input_ids) <= self.max_prefill_tokens:
            return prompt

        global_budget = self.max_prefill_tokens - SAFETY_MARGIN
        input_ids = input_ids[:global_budget // 2] + input_ids[-global_budget // 2:]
        prompt = self.tokenizer.decode(input_ids, skip_special_tokens=True)
        return prompt

    def _call_model(self, input_text: str, config: GenerateConfig) -> str:
        """调用的模型进行推理"""
        output_text = ""
        try:
            output_text = self.request_ai(input_text)
        except requests.exceptions.RequestException as e:
            logger.error(f"网络请求失败 (URL: {self.base_url}): {e}")
            # 向上抛出异常，让 EvalScope 框架识别为错误样本而不是得到一个空答案
            raise RuntimeError(f"Model API request failed due to network issue: {e}") from e
        except (json.JSONDecodeError, KeyError, IndexError) as e:
            logger.error(f"接口返回格式异常，解析失败: {e}")
            raise RuntimeError(f"Model API response parsing failed: {e}") from e

        return output_text

    def request_ai(self, content: str) -> str:
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
        return response.json()["choices"][0]["message"]["content"]


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
        datasets=['longbench_v2'],
        generation_config={
            "max_tokens": args.max_tokens,
            "batch_size": args.batch_size,
            "timeout": args.timeout
        },
        eval_batch_size=args.batch_size,
        limit=args.limit,
    )

    # 运行评测
    run_task(task_cfg=task_config)


if __name__ == '__main__':
    # 1. 定义参数解析器
    parser = argparse.ArgumentParser(description="运行大模型 LongBench 评测脚本")

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
