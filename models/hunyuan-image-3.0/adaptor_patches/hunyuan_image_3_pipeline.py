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
#
# Copyright 2024 The HuggingFace Team. All rights reserved.
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
# ================================================================================

import os
import math
from typing import Any, Callable, Dict, List
from typing import Optional, Tuple, Union

import torch
import torch_npu
from diffusers.callbacks import MultiPipelineCallbacks, PipelineCallback
from diffusers.image_processor import VaeImageProcessor
from diffusers.schedulers.scheduling_utils import SchedulerMixin
from diffusers.utils import logging
from hunyuan_image_3.hunyuan_image_3_pipeline import retrieve_timesteps
from hunyuan_image_3.hunyuan_image_3_pipeline import (
    ClassifierFreeGuidance,
    HunyuanImage3Text2ImagePipeline,
    HunyuanImage3Text2ImagePipelineOutput
)
from module.vae_patch_parallel import set_vae_patch_parallel, VAE_patch_parallel

local_rank = int(os.environ['LOCAL_RANK'])
logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


def init_vae_parallel(self):
    import torch.distributed as dist

    if not dist.is_initialized():
        logger.warning("Distributed environment not initialized, skipping VAE parallelism.")
        self.use_vae_parallel = False
        return

    world_size = dist.get_world_size()
    if world_size == 1:
        logger.warning("Single-card operation, skip VAE parallelism")
        self.use_vae_parallel = False
        return

    if world_size == 2:
        h_split, w_split = 1, 2
    elif world_size == 4:
        h_split, w_split = 2, 2
    elif world_size == 8:
        h_split, w_split = 4, 2
    else:
        h_split = int(math.sqrt(world_size))
        w_split = world_size // h_split

    all_pp_group_ranks = [[i for i in range(world_size)]]

    try:
        set_vae_patch_parallel(
            self.vae,
            h_split=h_split,
            w_split=w_split,
            all_pp_group_ranks=all_pp_group_ranks,
            decoder_decode="decoder.forward"
        )
        logger.info(f"VAE parallelism is enabled: {h_split}x{w_split} Grid, {world_size} NPUs")
    except Exception as e:
        logger.warning(f"VAE parallel initialization failed: {e}")
        self.use_vae_parallel = False


def text2_image_pipeline_init(
    self,
    model,
    scheduler: SchedulerMixin,
    vae,
    progress_bar_config: Dict[str, Any] = None,
    use_vae_parallel: bool = None,
):
    super(HunyuanImage3Text2ImagePipeline, self).__init__()

    # ==========================================================================================
    if progress_bar_config is None:
        progress_bar_config = {}
    if not hasattr(self, '_progress_bar_config'):
        self._progress_bar_config = {}
    self._progress_bar_config.update(progress_bar_config)
    # ==========================================================================================

    self.register_modules(
        model=model,
        scheduler=scheduler,
        vae=vae,
    )

    # should be a tuple or a list corresponding to the size of latents (batch_size, channel, *size)
    # if None, will be treated as a tuple of 1
    self.latent_scale_factor = self.model.config.vae_downsample_factor
    self.image_processor = VaeImageProcessor(vae_scale_factor=self.latent_scale_factor)

    # Must start with APG_mode_
    self.cfg_operator = ClassifierFreeGuidance()

    # Parallel initialization of VAEs
    if use_vae_parallel is None:
        env_value = os.environ.get('USE_VAE_PARALLEL', '0')
        self.use_vae_parallel = env_value.lower() in ['1', 'true', 'yes', 'on']
    else:
        self.use_vae_parallel = use_vae_parallel

    if self.use_vae_parallel:
        self.init_vae_parallel()


@torch.no_grad()
def text2_image_pipeline_call(
    self,
    batch_size: int,
    image_size: List[int],
    num_inference_steps: int = 50,
    timesteps: List[int] = None,
    sigmas: List[float] = None,
    guidance_scale: float = 7.5,
    generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
    latents: Optional[torch.Tensor] = None,
    output_type: Optional[str] = "pil",
    return_dict: bool = True,
    guidance_rescale: float = 0.0,
    callback_on_step_end: Optional[
        Union[Callable[[int, int, Dict], None], PipelineCallback, MultiPipelineCallbacks]
    ] = None,
    callback_on_step_end_tensor_inputs: List[str] = ["latents"],
    model_kwargs: Dict[str, Any] = None,
    **kwargs,
):
    r"""
    The call function to the pipeline for generation.

    Args:
        prompt (`str` or `List[str]`):
            The text to guide image generation.
        image_size (`Tuple[int]` or `List[int]`):
            The size (height, width) of the generated image.
        num_inference_steps (`int`, *optional*, defaults to 50):
            The number of denoising steps. More denoising steps usually lead to a higher quality image at the
            expense of slower inference.
        timesteps (`List[int]`, *optional*):
            Custom timesteps to use for the denoising process with schedulers which support a `timesteps` argument
            in their `set_timesteps` method. If not defined, the default behavior when `num_inference_steps` is
            passed will be used. Must be in descending order.
        sigmas (`List[float]`, *optional*):
            Custom sigmas to use for the denoising process with schedulers which support a `sigmas` argument in
            their `set_timesteps` method. If not defined, the default behavior when `num_inference_steps` is passed
            will be used.
        guidance_scale (`float`, *optional*, defaults to 7.5):
            A higher guidance scale value encourages the model to generate samples closely linked to the
            `condition` at the expense of lower sample quality. Guidance scale is enabled when `guidance_scale > 1`.
        generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
            A [`torch.Generator`](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make
            generation deterministic.
        latents (`torch.Tensor`, *optional*):
            Pre-generated noisy latents sampled from a Gaussian distribution, to be used as inputs for sample
            generation. Can be used to tweak the same generation with different conditions. If not provided,
            a latents tensor is generated by sampling using the supplied random `generator`.
        output_type (`str`, *optional*, defaults to `"pil"`):
            The output format of the generated sample.
        return_dict (`bool`, *optional*, defaults to `True`):
            Whether or not to return a [`~DiffusionPipelineOutput`] instead of a
            plain tuple.
        guidance_rescale (`float`, *optional*, defaults to 0.0):
            Guidance rescale factor from [Common Diffusion Noise Schedules and Sample Steps are
            Flawed](https://arxiv.org/pdf/2305.08891.pdf). Guidance rescale factor should fix overexposure when
            using zero terminal SNR.
        callback_on_step_end (`Callable`, `PipelineCallback`, `MultiPipelineCallbacks`, *optional*):
            A function or a subclass of `PipelineCallback` or `MultiPipelineCallbacks` that is called at the end of
            each denoising step during the inference. with the following arguments: `callback_on_step_end(self:
            DiffusionPipeline, step: int, timestep: int, callback_kwargs: Dict)`. `callback_kwargs` will include a
            list of all tensors as specified by `callback_on_step_end_tensor_inputs`.
        callback_on_step_end_tensor_inputs (`List`, *optional*):
            The list of tensor inputs for the `callback_on_step_end` function. The tensors specified in the list
            will be passed as `callback_kwargs` argument. You will only be able to include variables listed in the
            `._callback_tensor_inputs` attribute of your pipeline class.

    Examples:

    Returns:
        [`~DiffusionPipelineOutput`] or `tuple`:
            If `return_dict` is `True`, [`~DiffusionPipelineOutput`] is returned,
            otherwise a `tuple` is returned where the first element is a list with the generated samples.
    """

    callback_steps = kwargs.pop("callback_steps", None)
    pbar_steps = kwargs.pop("pbar_steps", None)

    if isinstance(callback_on_step_end, (PipelineCallback, MultiPipelineCallbacks)):
        callback_on_step_end_tensor_inputs = callback_on_step_end.tensor_inputs

    self._guidance_scale = guidance_scale
    self._guidance_rescale = guidance_rescale

    cfg_factor = 1 + self.do_classifier_free_guidance
    if self.model.hccl_comm_dict["cfg_parallel_size"] > 1:
        # If enable CFG parallel, cond and uncond inputs are deployed separately on two ranks, so the
        # batch size no longer needs to be multiplied by 2, cfg_factor should be 1
        cfg_factor = 1

    # Define call parameters
    device = self._execution_device

    # Prepare timesteps
    timesteps, num_inference_steps = retrieve_timesteps(
        self.scheduler, num_inference_steps, device, timesteps, sigmas,
    )

    # Prepare latent variables
    latents = self.prepare_latents(
        batch_size=batch_size,
        latent_channel=self.model.config.vae["latent_channels"],
        image_size=image_size,
        dtype=torch.bfloat16,
        device=device,
        generator=generator,
        latents=latents,
    )

    # Prepare extra step kwargs.
    _scheduler_step_extra_kwargs = self.prepare_extra_func_kwargs(
        self.scheduler.step, {"generator": generator}
    )

    # Prepare model kwargs
    input_ids = model_kwargs.pop("input_ids")
    attention_mask = self.model._prepare_attention_mask_for_generation(     # noqa
        input_ids, self.model.generation_config, model_kwargs=model_kwargs,
    )
    model_kwargs["attention_mask"] = attention_mask.to(latents.device)

    # Sampling loop
    num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
    self._num_timesteps = len(timesteps)

    idx_round = model_kwargs.get("idx_round", -1)
    enable_prof = False if idx_round > 1 else False
    if enable_prof:
        experimental_config = torch_npu.profiler._ExperimentalConfig(
            export_type=torch_npu.profiler.ExportType.Text,
            profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
            msprof_tx=False,
            aic_metrics=torch_npu.profiler.AiCMetrics.AiCoreNone,
            l2_cache=False,
            op_attr=False,
            data_simplification=False,
            record_op_args=False,
            gc_detect_threshold=None
        )

        prof = torch_npu.profiler.profile(
            activities=[
                torch_npu.profiler.ProfilerActivity.CPU,
                torch_npu.profiler.ProfilerActivity.NPU
            ],
            schedule=torch_npu.profiler.schedule(wait=0, warmup=0, active=2, repeat=1, skip_first=0),
            on_trace_ready=torch_npu.profiler.tensorboard_trace_handler("./prof/"),
            record_shapes=False,
            profile_memory=False,
            with_stack=False,
            with_modules=False,
            with_flops=False,
            experimental_config=experimental_config
        )
        prof.start()

    with self.progress_bar(total=num_inference_steps) as progress_bar:
        for i, t in enumerate(timesteps):
            # expand the latents if we are doing classifier free guidance
            latent_model_input = torch.cat([latents] * cfg_factor)
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

            t_expand = t.repeat(latent_model_input.shape[0])

            model_inputs = self.model.prepare_inputs_for_generation(
                input_ids,
                images=latent_model_input,
                timestep=t_expand,
                **model_kwargs,
            )

            with torch.autocast(device_type="npu", dtype=torch.bfloat16, enabled=True):
                model_output = self.model(**model_inputs, first_step=(i == 0))
                pred = model_output["diffusion_prediction"]
            pred = pred.to(dtype=torch.float32)

            # perform guidance
            if self.do_classifier_free_guidance:
                if self.model.hccl_comm_dict["cfg_parallel_size"] > 1:
                    cfg_parallel_group = self.model.hccl_comm_dict.get("cfg_parallel_group")
                    cfg_rank = torch.distributed.get_rank(cfg_parallel_group)
                    cfg_preds_list = [torch.empty_like(pred) for _ in range(2)]
                    torch.distributed.all_gather(cfg_preds_list, pred, group=cfg_parallel_group)
                    pred_cond, pred_uncond = cfg_preds_list
                else:
                    pred_cond, pred_uncond = pred.chunk(2)
                pred = self.cfg_operator(pred_cond, pred_uncond, self.guidance_scale, step=i)

            if self.do_classifier_free_guidance and self.guidance_rescale > 0.0:
                # Based on 3.4. in https://arxiv.org/pdf/2305.08891.pdf
                pred = rescale_noise_cfg(pred, pred_cond, guidance_rescale=self.guidance_rescale)

            # compute the previous noisy sample x_t -> x_t-1
            latents = self.scheduler.step(pred, t, latents, **_scheduler_step_extra_kwargs, return_dict=False)[0]

            if i != len(timesteps) - 1:
                model_kwargs = self.model._update_model_kwargs_for_generation(  # noqa
                    model_output,
                    model_kwargs,
                )
                if input_ids.shape[1] != model_kwargs["position_ids"].shape[1]:
                    input_ids = torch.gather(input_ids, 1, index=model_kwargs["position_ids"])

            if callback_on_step_end is not None:
                callback_kwargs = {}
                for k in callback_on_step_end_tensor_inputs:
                    callback_kwargs[k] = locals()[k]
                callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                latents = callback_outputs.pop("latents", latents)

            if enable_prof:
                prof.step()

            # call the callback, if provided
            if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                progress_bar.set_description(f"Rank:{local_rank}")
                progress_bar.update()

    if enable_prof:
        prof.stop()

    if hasattr(self.vae.config, 'scaling_factor') and self.vae.config.scaling_factor:
        latents = latents / self.vae.config.scaling_factor
    if hasattr(self.vae.config, 'shift_factor') and self.vae.config.shift_factor:
        latents = latents + self.vae.config.shift_factor

    if hasattr(self.vae, "ffactor_temporal"):
        latents = latents.unsqueeze(2)

    if self.use_vae_parallel:
        with VAE_patch_parallel():
            with torch.autocast(device_type="npu", dtype=torch.float16, enabled=True):
                image = self.vae.decode(latents, return_dict=False, generator=generator)[0]
    else:
        with torch.autocast(device_type="npu", dtype=torch.float16, enabled=True):
            image = self.vae.decode(latents, return_dict=False, generator=generator)[0]

    # b c t h w
    if hasattr(self.vae, "ffactor_temporal"):
        assert image.shape[2] == 1, "image should have shape [B, C, T, H, W] and T should be 1"
        image = image.squeeze(2)

    do_denormalize = [True] * image.shape[0]
    image = self.image_processor.postprocess(image, output_type=output_type, do_denormalize=do_denormalize)

    if not return_dict:
        return (image,)

    return HunyuanImage3Text2ImagePipelineOutput(samples=image)


HunyuanImage3Text2ImagePipeline.init_vae_parallel = init_vae_parallel
HunyuanImage3Text2ImagePipeline.__init__ = text2_image_pipeline_init
HunyuanImage3Text2ImagePipeline.__call__ = text2_image_pipeline_call