# Adapted from
# https://github.com/Tencent-Hunyuan/HunyuanImage-3.0/blob/main/run_image_gen.py,
# Copyright (c) Huawei Technologies Co., Ltd. 2025-2026.
#
# 2025 Tencent. All Rights Reserved. The trademark rights of Tencent Hunyuan are owned by Tencent or its affiliate.
#
# Licensed under the TENCENT HUNYUAN COMMUNITY LICENSE AGREEMENT (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://github.com/Tencent-Hunyuan/HunyuanImage-3.0/blob/main/LICENSE
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import argparse
import os
import sys
from pathlib import Path
import time
import threading
import logging
from loguru import logger
import torch
import torch_npu
from torch_npu.contrib import transfer_to_npu
import torch.distributed as dist
from hunyuan_image_3.hunyuan import HunyuanImage3ForCausalMM
import model_adaptor


local_rank = int(os.environ.get('LOCAL_RANK', '0'))


class OverwriteHandler(logging.StreamHandler):
    """A log processor that supports overwriting output on the same line"""
    def __init__(self, stream=None):
        super().__init__(stream)
        self._last_length = 0

    def emit(self, record):
        try:
            msg = self.format(record)
            # When new message is short, add spaces to avoid the tail of the previous message remaining on the terminal
            output = '\r' + msg.ljust(self._last_length)
            self.stream.write(output)
            self.stream.flush()
            self._last_length = len(msg)
        except Exception:
            self.handleError(record)


def parse_args():
    parser = argparse.ArgumentParser("Commandline arguments for running HunyuanImage-3 locally")
    parser.add_argument("--prompt", type=str, required=True, help="Prompt to run")
    parser.add_argument("--model-id", type=str, default="./HunyuanImage-3", help="Path to the model")
    parser.add_argument("--attn-impl", type=str, default="npu", choices=["sdpa", "flash_attention_2", "npu"],
                        help="Attention implementation. 'flash_attention_2' requires flash attention to be installed.")
    parser.add_argument("--moe-impl", type=str, default="eager", choices=["eager", "flashinfer", "npu_grouped_matmul"],
                        help="MoE implementation. 'flashinfer' requires FlashInfer to be installed.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed. Use None for random seed.")
    parser.add_argument("--diff-infer-steps", type=int, default=50, help="Number of inference steps.")
    parser.add_argument("--image-size", type=str, default="auto",
                        help="'auto' means image size is determined by the model. Alternatively, it can be in the "
                             "format of 'HxW' or 'H:W', which will be aligned to the set of preset sizes.")
    parser.add_argument("--use-system-prompt", type=str,
                        choices=["None", "dynamic", "en_vanilla", "en_recaption", "en_think_recaption", "custom"],
                        help="Use system prompt. 'None' means no system prompt; 'dynamic' means the system prompt is "
                             "determined by --bot-task; 'en_vanilla', 'en_recaption', 'en_think_recaption' are "
                             "three predefined system prompts; 'custom' means using the custom system prompt. When "
                             "using 'custom', --system-prompt must be provided. Default to load from the model "
                             "generation config.")
    parser.add_argument("--system-prompt", type=str, help="Custom system prompt. Used when --use-system-prompt "
                                                          "is 'custom'.")
    parser.add_argument("--bot-task", type=str, choices=["image", "auto", "think", "recaption"],
                        help="Type of task for the model. 'image' for direct image generation; 'auto' for text "
                             "generation; 'think' for think->re-write->image; 'recaption' for re-write->image."
                             "Default to load from the model generation config.")
    parser.add_argument("--save", type=str, default="image.png", help="Path to save the generated image")
    parser.add_argument("--verbose", type=int, default=0, help="Verbose level")
    parser.add_argument("--rewrite", type=int, default=0, help="Whether to rewrite the prompt with DeepSeek")
    parser.add_argument("--sys-deepseek-prompt", type=str, choices=["universal", "text_rendering"],
                        default="universal", help="System prompt for rewriting the prompt")

    parser.add_argument("--reproduce", action="store_true", help="Whether to reproduce the results")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--moe-tp", action="store_true", help="Use TP for MoE mudule")
    group.add_argument("--moe-ep", action="store_true", help="Use EP for MoE mudule")

    return parser.parse_args()


def set_reproducibility(enable, global_seed=None, benchmark=None):
    if enable:
        # Configure the seed for reproducibility
        import random
        random.seed(global_seed)
        # Seed the RNG for Numpy
        import numpy as np
        np.random.seed(global_seed)
        # Seed the RNG for all devices (both CPU and CUDA)
        torch.manual_seed(global_seed)
    # Set following debug environment variable
    # See the link for details: https://docs.nvidia.com/cuda/cublas/index.html#results-reproducibility
    if enable:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    # Cudnn benchmarking
    torch.backends.cudnn.benchmark = (not enable) if benchmark is None else benchmark
    # Use deterministic algorithms in PyTorch
    torch.backends.cudnn.deterministic = enable
    torch.use_deterministic_algorithms(enable)


def setup_distributed():
    world_size = int(os.environ['WORLD_SIZE'])
    rank = int(os.environ['RANK'])
    if not dist.is_initialized():
        dist.init_process_group(backend='hccl',
                                init_method=f"env://",
                                world_size=world_size,
                                rank=rank)
    torch.npu.set_device(local_rank)


def move_model_to_device(model, device, log_interval=5):
    handler = OverwriteHandler(sys.stdout)
    handler.setFormatter(logging.Formatter('%(message)s'))
    progress_logger = logging.getLogger('progress')
    progress_logger.setLevel(logging.INFO)
    progress_logger.addHandler(handler)
    progress_logger.propagate = False

    finished = threading.Event()
    started_at = time.monotonic()

    def report_progress():
        while not finished.wait(log_interval):
            elapsed = time.monotonic() - started_at
            if local_rank == 0:
                progress_logger.info(
                    f"Still loading weights and transferring model to {device}; elapsed: {elapsed:.0f}s"
                )

    reporter = threading.Thread(
        target=report_progress,
        name="model-transfer-progress",
        daemon=True,
    )
    reporter.start()

    try:
        logger.info(f"Loading weights and transferring model to {device}...")
        model = model.to(device)
        torch.npu.synchronize()
        elapsed = time.monotonic() - started_at
        logger.info(f"Model transferred to {device} in {elapsed:.2f}s")
        return model
    finally:
        finished.set()
        reporter.join()


def main(args):
    if args.reproduce:
        set_reproducibility(args.reproduce, global_seed=args.seed)

    if not args.prompt:
        raise ValueError("Prompt is required")
    if not Path(args.model_id).exists():
        raise ValueError(f"Model path {args.model_id} does not exist")

    kwargs = dict(
        torch_dtype="auto",
        moe_impl=args.moe_impl,
        moe_tp=True if not args.moe_ep else False
    )

    setup_distributed()

    # get weight path for this rank
    if os.environ.get("CFG_PARALLEL") == "1":
        tp_size = int(os.environ["WORLD_SIZE"]) // 2
        weight_index = local_rank % tp_size
    else:
        weight_index = local_rank
    model_id = os.path.join(args.model_id, f"rank-{weight_index:02d}")

    model = HunyuanImage3ForCausalMM.from_pretrained(model_id, **kwargs)
    model.load_tokenizer(args.model_id)

    # Load weights from host to device
    model = move_model_to_device(model, torch.device(f"npu:{local_rank}"))

    for k in range(4):
        image = model.generate_image(
            prompt=args.prompt,
            attn_implementation=args.attn_impl,
            seed=args.seed,
            image_size=args.image_size,
            use_system_prompt=args.use_system_prompt,
            system_prompt=args.system_prompt,
            bot_task=args.bot_task,
            diff_infer_steps=args.diff_infer_steps,
            verbose=args.verbose,
            stream=True,
            idx_round=k
        )

    if local_rank == 0:
        Path(args.save).parent.mkdir(parents=True, exist_ok=True)
        image.save(args.save)
        logger.info(f"Image saved to {args.save}")


if __name__ == "__main__":
    args = parse_args()
    main(args)
