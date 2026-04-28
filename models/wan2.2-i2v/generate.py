# coding=utf-8
# Adapted from
# https://github.com/Wan-Video/Wan2.2/blob/main/generate.py
# Copyright (c) Huawei Technologies Co., Ltd. 2025 - 2026. All rights reserved.
# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
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
import sys
import warnings
from datetime import datetime

warnings.filterwarnings('ignore')

import random
import time

import torch
import torch_npu
torch_npu.npu.set_compile_mode(jit_compile=False)
torch.npu.config.allow_internal_format = False
from torch_npu.contrib import transfer_to_npu

import torch.distributed as dist
from PIL import Image

import wan
from wan.configs import MAX_AREA_CONFIGS, SIZE_CONFIGS, SUPPORTED_SIZES, WAN_CONFIGS
from wan.distributed.util import init_distributed_group
from wan.utils.prompt_extend import DashScopePromptExpander, QwenPromptExpander
from wan.utils.utils import save_video, str2bool
from wan.distributed.parallel_mgr import ParallelConfig, init_parallel_env, finalize_parallel_env
from wan.cache import first_block_forward
import wan.cache.cache_block
from module.dit_cache.cache_method import cache_manager, NoCache

EXAMPLE_PROMPT = {
    "t2v-A14B": {
        "prompt":
            "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage.",
    },
    "i2v-A14B": {
        "prompt":
            "Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard. The fluffy-furred feline gazes directly at the camera with a relaxed expression. Blurred beach scenery forms the background featuring crystal-clear waters, distant green hills, and a blue sky dotted with white clouds. The cat assumes a naturally relaxed posture, as if savoring the sea breeze and warm sunlight. A close-up shot highlights the feline's intricate details and the refreshing atmosphere of the seaside.",
        "image":
            "examples/i2v_input.JPG",
    },
    "ti2v-5B": {
        "prompt":
            "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage.",
    },
}


def _validate_args(args):
    # Basic check
    if args.ckpt_dir is None:
        raise ValueError("Please specify the checkpoint directory.")
    if args.task not in WAN_CONFIGS:
        raise ValueError(f"Unsupport task: {args.task}")
    if args.task not in EXAMPLE_PROMPT:
        raise ValueError(f"Unsupport task: {args.task}")

    if args.prompt is None:
        args.prompt = EXAMPLE_PROMPT[args.task]["prompt"]
    if args.image is None and "image" in EXAMPLE_PROMPT[args.task]:
        args.image = EXAMPLE_PROMPT[args.task]["image"]

    if args.task == "i2v-A14B":
        if args.image is None:
            raise ValueError("Please specify the image path for i2v.")

    cfg = WAN_CONFIGS[args.task]

    if args.sample_steps is None:
        args.sample_steps = cfg.sample_steps

    if args.sample_shift is None:
        args.sample_shift = cfg.sample_shift

    if args.sample_guide_scale is None:
        args.sample_guide_scale = cfg.sample_guide_scale

    if args.frame_num is None:
        args.frame_num = cfg.frame_num

    args.base_seed = args.base_seed if args.base_seed >= 0 else random.randint(
        0, sys.maxsize)
    # Size check
    if args.size not in SUPPORTED_SIZES[args.task]:
        supported = ', '.join(SUPPORTED_SIZES[args.task])
        raise ValueError(
            f"Unsupport size {args.size} for task {args.task}, "
            f"supported sizes are: {supported}"
        )


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Generate a image or video from a text prompt or image using Wan"
    )
    parser.add_argument(
        "--task",
        type=str,
        default="t2v-A14B",
        choices=list(WAN_CONFIGS.keys()),
        help="The task to run.")
    parser.add_argument(
        "--size",
        type=str,
        default="1280*720",
        choices=list(SIZE_CONFIGS.keys()),
        help=(
            "The area (width*height) of the generated video. "
            "For the I2V task, the aspect ratio of the output video "
            "will follow that of the input image."
        )
    )
    parser.add_argument(
        "--frame_num",
        type=int,
        default=None,
        help="How many frames of video are generated. The number should be 4n+1"
    )
    parser.add_argument(
        "--ckpt_dir",
        type=str,
        default=None,
        help="The path to the checkpoint directory.")
    parser.add_argument(
        "--offload_model",
        type=str2bool,
        default=None,
        help="Whether to offload the model to CPU after each model forward, reducing GPU memory usage."
    )
    parser.add_argument(
        "--cfg_size",
        type=int,
        default=1,
        help="The size of the cfg parallelism in DiT.")
    parser.add_argument(
        "--ulysses_size",
        type=int,
        default=1,
        help="The size of the ulysses parallelism in DiT.")
    parser.add_argument(
        "--ring_size",
        type=int,
        default=1,
        help="The size of the ring attention parallelism in DiT.")
    parser.add_argument(
        "--tp_size",
        type=int,
        default=1,
        help="The size of the tensor parallelism in DiT.")
    parser.add_argument(
        "--vae_parallel",
        action="store_true",
        default=False,
        help="Whether to use parallel for vae.")
    parser.add_argument(
        "--t5_fsdp",
        action="store_true",
        default=False,
        help="Whether to use FSDP for T5.")
    parser.add_argument(
        "--t5_cpu",
        action="store_true",
        default=False,
        help="Whether to place T5 model on CPU.")
    parser.add_argument(
        "--dit_fsdp",
        action="store_true",
        default=False,
        help="Whether to use FSDP for DiT.")
    parser.add_argument(
        "--save_file",
        type=str,
        default=None,
        help="The file to save the generated video to.")
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="The prompt to generate the video from.")
    parser.add_argument(
        "--use_prompt_extend",
        action="store_true",
        default=False,
        help="Whether to use prompt extend.")
    parser.add_argument(
        "--prompt_extend_method",
        type=str,
        default="local_qwen",
        choices=["dashscope", "local_qwen"],
        help="The prompt extend method to use.")
    parser.add_argument(
        "--prompt_extend_model",
        type=str,
        default=None,
        help="The prompt extend model to use.")
    parser.add_argument(
        "--prompt_extend_target_lang",
        type=str,
        default="zh",
        choices=["zh", "en"],
        help="The target language of prompt extend.")
    parser.add_argument(
        "--base_seed",
        type=int,
        default=-1,
        help="The seed to use for generating the video.")
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="The image to generate the video from.")
    parser.add_argument(
        "--sample_solver",
        type=str,
        default='unipc',
        choices=['unipc', 'dpm++'],
        help="The solver used to sample.")
    parser.add_argument(
        "--sample_steps", type=int, default=None, help="The sampling steps.")
    parser.add_argument(
        "--sample_shift",
        type=float,
        default=None,
        help="Sampling shift factor for flow matching schedulers.")
    parser.add_argument(
        "--sample_guide_scale",
        type=float,
        default=None,
        help="Classifier free guidance scale.")
    parser.add_argument(
        "--convert_model_dtype",
        action="store_true",
        default=False,
        help="Whether to convert model paramerters dtype.")
    parser.add_argument(
        "--quant_mode",
        type=int,
        default=0,
        choices=[0, 1, 2, 3],
        help="Quantization mode: " \
        "0: Do not use quantized model for inference, " \
        "1: Export calibration data, " \
        "2: Export quantized model, " \
        "3: Use quantized model for inference.")

    parser.add_argument(
        "--quant_data_dir",
        type=str,
        default="./output/quant_data",
        help="Path for calibration data or weight export.")

    parser.add_argument(
        "--cache_config",
        type=str,
        default="./config/14b_single.yaml",
        help="Path to cache config YAML (reads the dit_cache section). Overridden by infer.sh via mm_function.sh.")
    args = parser.parse_args()
    _validate_args(args)

    return args


def _init_logging(rank):
    # logging
    if rank == 0:
        # set format
        logging.basicConfig(
            level=logging.INFO,
            format="[%(asctime)s] %(levelname)s: %(message)s",
            handlers=[logging.StreamHandler(stream=sys.stdout)])
    else:
        logging.basicConfig(level=logging.ERROR)


def generate(args):
    rank = int(os.getenv("RANK", 0))
    world_size = int(os.getenv("WORLD_SIZE", 1))
    local_rank = int(os.getenv("LOCAL_RANK", 0))
    device = local_rank
    _init_logging(rank)
    stream = torch.npu.Stream()
    if args.offload_model is None:
        args.offload_model = False if world_size > 1 else True
        logging.info(
            f"offload_model is not specified, set to {args.offload_model}.")
    if world_size > 1:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(
            backend="hccl",
            init_method="env://",
            rank=rank,
            world_size=world_size)
    else:
        if args.t5_fsdp or args.dit_fsdp:
            raise ValueError(
                "t5_fsdp and dit_fsdp are not supported in non-distributed environments."
            )
        context_parallel_enabled = (
            args.cfg_size > 1 
            or args.ulysses_size > 1 
            or args.ring_size > 1 
            or args.tp_size > 1
        )
        if context_parallel_enabled:
            raise ValueError(
                "Context parallel (cfg_size, ulysses_size, ring_size, tp_size > 1) "
                "is not supported in non-distributed environments."
            )
        if args.vae_parallel:
            raise ValueError(
                "VAE parallel is not supported in non-distributed environments."
            )
    
    if args.tp_size > 1:
        raise NotImplementedError("Tensor Parallel is not supported now")


    if args.cfg_size > 1 or args.ulysses_size > 1 or args.ring_size > 1 or args.tp_size > 1:
        product = args.cfg_size * args.ulysses_size * args.ring_size * args.tp_size
        if product != world_size:
            raise ValueError(
                f"The product of cfg_size, ulysses_size, ring_size and tp_size ({product}) "
                f"must equal world_size ({world_size})."
            )
        sp_degree = args.ulysses_size * args.ring_size
        parallel_config = ParallelConfig(
            sp_degree=sp_degree,
            ulysses_degree=args.ulysses_size,
            ring_degree=args.ring_size,
            tp_degree=args.tp_size,
            use_cfg_parallel=(args.cfg_size == 2),
            world_size=world_size,
        )
        init_parallel_env(parallel_config)

    if args.tp_size > 1 and args.dit_fsdp:
        logging.info("DiT using Tensor Parallel, disabled dit_fsdp")
        args.dit_fsdp = False

    if args.use_prompt_extend:
        if args.prompt_extend_method == "dashscope":
            prompt_expander = DashScopePromptExpander(
                model_name=args.prompt_extend_model,
                task=args.task,
                is_vl=args.image is not None)
        elif args.prompt_extend_method == "local_qwen":
            prompt_expander = QwenPromptExpander(
                model_name=args.prompt_extend_model,
                task=args.task,
                is_vl=args.image is not None,
                device=rank)
        else:
            raise NotImplementedError(
                f"Unsupport prompt_extend_method: {args.prompt_extend_method}")

    cfg = WAN_CONFIGS[args.task]
    if args.ulysses_size > 1:
        if cfg.num_heads % args.ulysses_size != 0:
            raise ValueError(
                f"`cfg.num_heads={cfg.num_heads}` cannot be divided evenly by `args.ulysses_size={args.ulysses_size}`."
            )

    logging.info(f"Generation job args: {args}")
    logging.info(f"Generation model config: {cfg}")

    if dist.is_initialized():
        base_seed = [args.base_seed] if rank == 0 else [None]
        dist.broadcast_object_list(base_seed, src=0)
        args.base_seed = base_seed[0]

    logging.info(f"Input prompt: {args.prompt}")
    img = None
    if args.image is not None:
        img = Image.open(args.image).convert("RGB")
        logging.info(f"Input image: {args.image}")

    # prompt extend
    if args.use_prompt_extend:
        logging.info("Extending prompt ...")
        if rank == 0:
            prompt_output = prompt_expander(
                args.prompt,
                image=img,
                tar_lang=args.prompt_extend_target_lang,
                seed=args.base_seed)
            if prompt_output.status == False:
                logging.info(
                    f"Extending prompt failed: {prompt_output.message}")
                logging.info("Falling back to original prompt.")
                input_prompt = args.prompt
            else:
                input_prompt = prompt_output.prompt
            input_prompt = [input_prompt]
        else:
            input_prompt = [None]
        if dist.is_initialized():
            dist.broadcast_object_list(input_prompt, src=0)
        args.prompt = input_prompt[0]
        logging.info(f"Extended prompt: {args.prompt}")

    if "t2v" in args.task:
        logging.info("Creating WanT2V pipeline.")
        wan_t2v = wan.WanT2V(
            config=cfg,
            checkpoint_dir=args.ckpt_dir,
            device_id=device,
            rank=rank,
            t5_fsdp=args.t5_fsdp,
            dit_fsdp=args.dit_fsdp,
            use_sp=(args.ulysses_size > 1 or args.ring_size > 1),
            t5_cpu=args.t5_cpu,
            convert_model_dtype=args.convert_model_dtype,
            use_vae_parallel=args.vae_parallel,
            quant_mode=args.quant_mode
        )

        transformer_low = wan_t2v.low_noise_model
        transformer_high = wan_t2v.high_noise_model

        if args.tp_size > 1:
            logging.info("Initializing Tensor Parallel ...")
            applicator = TensorParallelApplicator(args.tp_size, device_map="cpu")
            applicator.apply_to_model(transformer_low)
            applicator.apply_to_model(transformer_high)
        # wan_t2v.low_noise_model.to("npu")
        # wan_t2v.high_noise_model.to("npu")

        if args.quant_mode == 2:
            logging.info(f"quantize weights saved, will be return")
            return

        logging.info("Warm up 2 steps ...")
        video = wan_t2v.generate(
            args.prompt,
            size=SIZE_CONFIGS[args.size],
            frame_num=args.frame_num,
            shift=args.sample_shift,
            sample_solver=args.sample_solver,
            sampling_steps=2,
            guide_scale=args.sample_guide_scale,
            seed=args.base_seed,
            offload_model=args.offload_model)

        logging.info(f"Generating video ...")
        stream.synchronize()
        begin = time.time()
        video = wan_t2v.generate(
            args.prompt,
            size=SIZE_CONFIGS[args.size],
            frame_num=args.frame_num,
            shift=args.sample_shift,
            sample_solver=args.sample_solver,
            sampling_steps=args.sample_steps,
            guide_scale=args.sample_guide_scale,
            seed=args.base_seed,
            offload_model=args.offload_model)
        stream.synchronize()
        end = time.time()
        logging.info(f"Generating video used time {end - begin: .4f}s")

    elif "ti2v" in args.task:
        logging.info("Creating WanTI2V pipeline.")
        wan_ti2v = wan.WanTI2V(
            config=cfg,
            checkpoint_dir=args.ckpt_dir,
            device_id=device,
            rank=rank,
            t5_fsdp=args.t5_fsdp,
            dit_fsdp=args.dit_fsdp,
            use_sp=(args.ulysses_size > 1),
            t5_cpu=args.t5_cpu,
            convert_model_dtype=args.convert_model_dtype,
            use_vae_parallel=args.vae_parallel,
            quant_mode=args.quant_mode
        )

        transformer = wan_ti2v.model
        
        if args.tp_size > 1:
            logging.info("Initializing Tensor Parallel ...")
            applicator = TensorParallelApplicator(args.tp_size, device_map="cpu")
            applicator.apply_to_model(transformer)
        # wan_ti2v.model.to("npu")
        
        if args.quant_mode == 2:
            logging.info(f"quantize weights saved, will be return")
            return

        logging.info("Warm up 2 steps ...")
        video = wan_ti2v.generate(
            args.prompt,
            img=img,
            size=SIZE_CONFIGS[args.size],
            max_area=MAX_AREA_CONFIGS[args.size],
            frame_num=args.frame_num,
            shift=args.sample_shift,
            sample_solver=args.sample_solver,
            sampling_steps=2,
            guide_scale=args.sample_guide_scale,
            seed=args.base_seed,
            offload_model=args.offload_model)

        logging.info(f"Generating video ...")
        stream.synchronize()
        begin = time.time()
        video = wan_ti2v.generate(
            args.prompt,
            img=img,
            size=SIZE_CONFIGS[args.size],
            max_area=MAX_AREA_CONFIGS[args.size],
            frame_num=args.frame_num,
            shift=args.sample_shift,
            sample_solver=args.sample_solver,
            sampling_steps=args.sample_steps,
            guide_scale=args.sample_guide_scale,
            seed=args.base_seed,
            offload_model=args.offload_model)
        stream.synchronize()
        end = time.time()
        logging.info(f"Generating video used time {end - begin: .4f}s")
    else:
        logging.info("Creating WanI2V pipeline.")
        wan_i2v = wan.WanI2V(
            config=cfg,
            checkpoint_dir=args.ckpt_dir,
            device_id=device,
            rank=rank,
            t5_fsdp=args.t5_fsdp,
            dit_fsdp=args.dit_fsdp,
            use_sp=(args.ulysses_size > 1 or args.ring_size > 1),
            t5_cpu=args.t5_cpu,
            convert_model_dtype=args.convert_model_dtype,
            use_vae_parallel=args.vae_parallel,
        )
        cache_manager.from_config(
                config_path=args.cache_config,
                cache_params={"num_steps": args.sample_steps})
        if world_size > 1:
            logging.info("Cannot enable both Multi-NPU and DIT-Cache. DIT-Cache has been disabled")
            logging.info("Now apply Dit-Cache method: NoCache!")
            cache_manager.cache_method = NoCache()
        else:
            cache_block = wan_i2v.high_noise_model.blocks[0]
            cache_block.forward = first_block_forward.__get__(cache_block, type(cache_block))
            cache_block = wan_i2v.low_noise_model.blocks[0]
            cache_block.forward = first_block_forward.__get__(cache_block, type(cache_block))

        transformer_low = wan_i2v.low_noise_model
        transformer_high = wan_i2v.high_noise_model
        if args.tp_size > 1:
            logging.info("Initializing Tensor Parallel ...")
            applicator = TensorParallelApplicator(args.tp_size, device_map="cpu")
            applicator.apply_to_model(transformer_low)
            applicator.apply_to_model(transformer_high)
        # wan_i2v.low_noise_model.to("npu")
        # wan_i2v.high_noise_model.to("npu")

        if args.quant_mode == 2:
            logging.info(f"quantize weights saved, will be return")
            return

        logging.info("Warm up 2 steps ...")
        video = wan_i2v.generate(
            args.prompt,
            img,
            max_area=MAX_AREA_CONFIGS[args.size],
            frame_num=args.frame_num,
            shift=args.sample_shift,
            sample_solver=args.sample_solver,
            sampling_steps=2,
            guide_scale=args.sample_guide_scale,
            seed=args.base_seed,
            offload_model=args.offload_model)

        logging.info("Generating video ...")
        stream.synchronize()
        begin = time.time()
        video = wan_i2v.generate(
            args.prompt,
            img,
            max_area=MAX_AREA_CONFIGS[args.size],
            frame_num=args.frame_num,
            shift=args.sample_shift,
            sample_solver=args.sample_solver,
            sampling_steps=args.sample_steps,
            guide_scale=args.sample_guide_scale,
            seed=args.base_seed,
            offload_model=args.offload_model)
        stream.synchronize()
        end = time.time()
        logging.info(f"Generating video used time {end - begin: .4f}s")

    if rank == 0:
        if args.save_file is None:
            formatted_time = datetime.now().strftime("%Y%m%d_%H%M%S")
            formatted_prompt = args.prompt.replace(" ", "_").replace("/",
                                                                     "_")[:50]
            suffix = '.mp4'
            args.save_file = f"{args.task}_{args.size.replace('*','x') if sys.platform=='win32' else args.size}_{args.cfg_size}_{args.ulysses_size}_{args.ring_size}_{args.tp_size}_{formatted_prompt}_{formatted_time}" + suffix

        logging.info(f"Saving generated video to {args.save_file}")
        save_video(
            tensor=video[None],
            save_file=args.save_file,
            fps=cfg.sample_fps,
            nrow=1,
            normalize=True,
            value_range=(-1, 1))
    del video
    cache_manager.cache_method.print_statistics()
    finalize_parallel_env()
    logging.info("Finished.")


if __name__ == "__main__":
    args = _parse_args()
    generate(args)
