# Adapted from  
# https://github.com/Tencent-Hunyuan/HunyuanImage-3.0,
# Copyright (c) Huawei Technologies Co., Ltd. 2025.
# Copyright (C) 2025 THL A29 Limited, a Tencent company. All rights reserved.
#
# This code is based on Tencent-Hunyuan's HunyuanImage-3.0 library and the
# HunyuanImage-3.0 implementations in this library. It has been modified from
# its original forms to accommodate minor architectural differences compared
# to HunyuanImage-3.0 used by Tencent-Hunyuan team that trained the model.
# ================================================================================
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
# ================================================================================

import argparse
import os
from pathlib import Path
import torch
import torch_npu
from torch_npu.contrib import transfer_to_npu
from loguru import logger
from hunyuan_image_3.hunyuan import HunyuanImage3ForCausalMM
import torch.distributed as dist
import model_adaptor


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
    local_rank = int(os.environ['LOCAL_RANK'])
    torch.npu.set_device(local_rank)


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
    )

    setup_distributed()

    # get weight path for this rank
    local_rank = int(os.environ["LOCAL_RANK"])
    if os.environ.get("CFG_PARALLEL") == "1":
        tp_size = int(os.environ["WORLD_SIZE"]) // 2
        weight_index = local_rank % tp_size
    else:
        weight_index = local_rank
    model_id = os.path.join(args.model_id, f"rank-{weight_index:02d}")

    model = HunyuanImage3ForCausalMM.from_pretrained(model_id, **kwargs)
    model.load_tokenizer(args.model_id)
    model = model.to(torch.device(f"npu:{local_rank}"))

    for k in range(2):
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
