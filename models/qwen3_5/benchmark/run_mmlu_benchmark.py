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

import argparse
import logging
import os
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)


def run(cmd: list[str], cwd: Path, env: dict[str, str] | None = None) -> None:
    logger.info("Running: " + " ".join(cmd))
    subprocess.run(cmd, cwd=cwd, env=env, check=True)


def parse_args():
    parser = argparse.ArgumentParser(description="Run MMLU benchmark for qwen3_5")
    parser.add_argument("--mmlu-dir", default=None, required=True)
    parser.add_argument("--work-dir", default="models/qwen3_5/benchmark")
    parser.add_argument("--yaml-path", default="models/qwen3_5/benchmark/qwen3_5_35b_mmlu_test.yaml")
    parser.add_argument("--ntrain", type=int, default=5)
    parser.add_argument("--split", default="test", choices=["test", "val"])
    parser.add_argument("--max-subjects", type=int, default=0)
    parser.add_argument("--max-prompts", type=int, default=0)
    parser.add_argument("--max-examples-per-subject", type=int, default=0)
    parser.add_argument(
        "--visible-devices",
        default=os.environ.get("ASCEND_RT_VISIBLE_DEVICES", "0,1"),
        help="ASCEND_RT_VISIBLE_DEVICES used when launching qwen3_5",
    )
    parser.add_argument(
        "--output-name",
        default="mmlu_eval_results.json",
        help="Rank0 result json file name under work-dir",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[3]
    work_dir = (repo_root / args.work_dir).resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    yaml_path = (repo_root / args.yaml_path).resolve()
    output_path = work_dir / args.output_name

    launch_env = os.environ.copy()
    launch_env["ASCEND_RT_VISIBLE_DEVICES"] = args.visible_devices
    launch_env["BENCHMARK_ENTRY"] = str((repo_root / "models/qwen3_5/benchmark/evaluate_mmlu.py").resolve())
    launch_env["MMLU_DIR"] = args.mmlu_dir
    launch_env["MMLU_SPLIT"] = args.split
    launch_env["MMLU_NTRAIN"] = str(args.ntrain)
    launch_env["MMLU_MAX_SUBJECTS"] = str(args.max_subjects)
    launch_env["MMLU_MAX_PROMPTS"] = str(args.max_prompts)
    launch_env["MMLU_MAX_EXAMPLES_PER_SUBJECT"] = str(args.max_examples_per_subject)
    launch_env["MMLU_OUTPUT_PATH"] = str(output_path)

    run(
        [
            "bash",
            "models/qwen3_5/benchmark/run_infer.sh",
            str(yaml_path),
        ],
        cwd=repo_root,
        env=launch_env,
    )

    logger.info(f"Results: {output_path}")


if __name__ == "__main__":
    main()
