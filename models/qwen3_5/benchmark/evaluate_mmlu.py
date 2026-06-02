# coding=utf-8
# Adapted from
# https://github.com/hendrycks/test/blob/master/evaluate.py
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Copyright (c) 2020 Dan Hendrycks
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

import argparse
import csv
import json
import os
import logging
from pathlib import Path

import yaml

from executor.core import InferenceConfig, OfflineInference

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)
CHOICES = ("A", "B", "C", "D")


def parse_args():
    parser = argparse.ArgumentParser(description="Qwen3.5 MMLU benchmark")
    parser.add_argument("--yaml_file_path", type=str, required=True, help="Inference yaml path")
    parser.add_argument("--mmlu-dir", type=str, default=os.getenv("MMLU_DIR", ""), help="MMLU data directory")
    parser.add_argument("--split", type=str, default=os.getenv("MMLU_SPLIT", "test"), choices=["test", "val"])
    parser.add_argument("--ntrain", type=int, default=int(os.getenv("MMLU_NTRAIN", "5")))
    parser.add_argument("--max-subjects", type=int, default=int(os.getenv("MMLU_MAX_SUBJECTS", "0")))
    parser.add_argument("--max-prompts", type=int, default=int(os.getenv("MMLU_MAX_PROMPTS", "0")))
    parser.add_argument(
        "--max-examples-per-subject",
        type=int,
        default=int(os.getenv("MMLU_MAX_EXAMPLES_PER_SUBJECT", "0")),
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default=os.getenv("MMLU_OUTPUT_PATH", ""),
        help="Optional json output path for rank0",
    )
    return parser.parse_args()


def read_csv_rows(path: Path) -> list[list[str]]:
    with path.open("r", encoding="utf-8") as f:
        return [row for row in csv.reader(f)]


def build_prompt(dev_rows: list[list[str]], test_row: list[str], subject: str, ntrain: int) -> str:
    prompt = f"The following are multiple choice questions about {subject.replace('_', ' ')}.\n\n"
    for row in dev_rows[:ntrain]:
        q, a, b, c, d, ans = row
        prompt += f"Question: {q}\nA. {a}\nB. {b}\nC. {c}\nD. {d}\nAnswer: {ans}\n\n"

    q, a, b, c, d, _ = test_row
    prompt += (
        f"Question: {q}\nA. {a}\nB. {b}\nC. {c}\nD. {d}\n"
        "Answer: Reply with only one capital letter: A, B, C, or D."
    )
    return prompt


def render_chat_prompt(prompt: str, tokenizer) -> str:
    return [{"role": "user", "content": prompt}]


def build_rendered_prompt_for_eval(
    dev_rows: list[list[str]],
    test_row: list[str],
    subject: str,
    ntrain: int,
    tokenizer,
    input_max_len: int,
):
    effective_ntrain = ntrain
    prompt = build_prompt(dev_rows, test_row, subject, effective_ntrain)
    chat_messages = render_chat_prompt(prompt, tokenizer)
    rendered_prompt = tokenizer.apply_chat_template(
        chat_messages,
        tokenize=False,
        add_generation_prompt=True
    )
    rendered_len = len(tokenizer(rendered_prompt, add_special_tokens=False).input_ids)

    if input_max_len > 0 and rendered_len > input_max_len and effective_ntrain > 2:
        effective_ntrain = 2
        prompt = build_prompt(dev_rows, test_row, subject, effective_ntrain)
        chat_messages = render_chat_prompt(prompt, tokenizer)
        rendered_prompt = tokenizer.apply_chat_template(
            chat_messages,
            tokenize=False,
            add_generation_prompt=True
        )
        rendered_len = len(tokenizer(rendered_prompt, add_special_tokens=False).input_ids)

    return chat_messages, effective_ntrain, rendered_len


def extract_choice(text: str) -> str:
    stripped = text.strip().upper()
    if stripped in CHOICES:
        return stripped
    for choice in CHOICES:
        if choice in stripped:
            return choice
    return stripped[:1]


def load_config(yaml_file_path: str, global_rank: int, local_rank: int) -> InferenceConfig:
    with open(yaml_file_path, "r", encoding="utf-8") as f:
        yaml_dict = yaml.safe_load(f)
    config = InferenceConfig.from_dict(yaml_dict, global_rank=global_rank, local_rank=local_rank)
    if config.model_config.output_path == "":
        config.model_config.output_path = os.path.dirname(yaml_file_path)
    if (
        config.scheduler_config.batch_size_per_dp_rank == 0
        and config.parallel_config.attn_tp_size == 1
    ):
        non_attn_tp_sizes = (
            config.parallel_config.moe_tp_size,
            config.parallel_config.embed_tp_size,
            config.parallel_config.lmhead_tp_size,
            config.parallel_config.dense_tp_size,
            config.parallel_config.o_proj_tp_size,
        )
        if any(size > 1 for size in non_attn_tp_sizes) or config.parallel_config.moe_ep_size > 1:
            config.scheduler_config.batch_size_per_dp_rank = config.scheduler_config.batch_size
    return config


def main():
    args = parse_args()
    local_rank = int(os.getenv("LOCAL_RANK", "0"))
    rank_offset = int(os.getenv("RANK_OFFSET", "0"))
    global_rank = local_rank + rank_offset

    config = load_config(args.yaml_file_path, global_rank=global_rank, local_rank=local_rank)
    llm = OfflineInference(config)

    data_dir = Path(args.mmlu_dir)
    test_dir = data_dir / args.split
    dev_dir = data_dir / "dev"
    subject_files = sorted(test_dir.glob(f"*_{args.split}.csv"))
    if args.max_subjects > 0:
        subject_files = subject_files[:args.max_subjects]

    total_correct = 0
    total_num = 0
    details = []
    case_id = 0
    per_subject = {}

    if global_rank == 0:
        logger.info("开始 MMLU 测试...")

    for subject_file in subject_files:
        if args.max_prompts > 0 and total_num >= args.max_prompts:
            break
        subject = subject_file.name[: -len(f"_{args.split}.csv")]
        dev_rows = read_csv_rows(dev_dir / f"{subject}_dev.csv")
        test_rows = read_csv_rows(subject_file)
        if args.max_examples_per_subject > 0:
            test_rows = test_rows[: args.max_examples_per_subject]

        subject_correct = 0
        subject_total = 0
        for test_row in test_rows:
            if args.max_prompts > 0 and total_num >= args.max_prompts:
                break
            case_id += 1
            rendered_prompt, effective_ntrain, rendered_len = build_rendered_prompt_for_eval(
                dev_rows,
                test_row,
                subject,
                args.ntrain,
                llm.engine.tokenizer,
                config.data_config.input_truncated_len,
            )
            result = llm.generate([rendered_prompt])[0][0]
            pred = extract_choice(result.output_text)
            ans = test_row[5].strip()
            correct = pred == ans

            if global_rank == 0:
                logger.info(f"pred[{case_id}]: {pred}=={ans}")
                details.append(
                    {
                        "case_id": case_id,
                        "subject": subject,
                        "question": test_row[0],
                        "prediction": pred,
                        "answer": ans,
                        "correct": correct,
                        "ntrain": effective_ntrain,
                        "rendered_prompt_tokens": rendered_len,
                        "output_text": result.output_text,
                    }
                )

            subject_correct += int(correct)
            subject_total += 1
            total_correct += int(correct)
            total_num += 1

        if global_rank == 0 and subject_total > 0:
            subject_acc = subject_correct / subject_total
            per_subject[subject] = {
                "correct": subject_correct,
                "total": subject_total,
                "accuracy": subject_acc,
            }
            logger.info(f"{subject:25} | acc: {subject_acc:.4f}")

    if global_rank != 0:
        return

    summary = {
        "total": total_num,
        "correct": total_correct,
        "accuracy": total_correct / total_num if total_num else 0.0,
        "per_subject": per_subject,
    }

    logger.info("\n" + "=" * 60)
    logger.info("MMLU 测试完成")
    logger.info(f"总准确率: {summary['accuracy']:.4f}")
    logger.info(f"正确: {total_correct} / 总数: {total_num}")
    logger.info("=" * 60)

    if args.output_path:
        output_path = Path(args.output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            json.dump({"summary": summary, "details": details}, f, ensure_ascii=False, indent=2)
        logger.info(f"Saved results to {output_path}")


if __name__ == "__main__":
    main()
