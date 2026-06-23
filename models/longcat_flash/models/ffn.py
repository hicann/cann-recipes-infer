# coding=utf-8
# Adapted from
# https://huggingface.co/meituan-longcat/LongCat-Flash-Chat/blob/main/modeling_longcat_flash.py
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# Copyright (c) 2025 Meituan
#
# MIT License:
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from typing import Iterable, Optional, Tuple, Set
import torch
import npugraph_ex
from torch import nn
import torch.distributed as dist

from transformers.generation import GenerationMixin
from transformers.modeling_layers import GradientCheckpointingLayer
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import logging

from executor.utils import superkernel_scope, npu_prefetch
from executor.utils.stream_utils import npu_stream_switch
from module.fuse_moe_gmm import FusedMoEGMM
from executor.model_loader.weight_utils import default_weight_loader
from executor.core.config import InferenceConfig, CommManager
from module.quantization import QuantizeMethodBase
from module.quantization.compressed_tensors.compressed_tensors_moe_gmm import CompressedTensorW8A8Int8MoEGMMMethod

from .configuration_longcat_flash import LongcatFlashConfig
from .modeling_longcat_flash import LongcatFlashMoE, _calc_longcat_moe_mc2_hccl_buffer_size

logger = logging.get_logger(__name__)


def _build_afd_subgroups(world_size, group_size, group_stride=1):
    """Subgroup partition for one AFD group: the same group_num x group_size
    layout is built independently inside the FFN half [0, N/2) and the
    attention half [N/2, N)."""
    ffn_world_size = world_size // 2
    if ffn_world_size % group_size != 0:
        raise ValueError(
            f"AFD half world_size={ffn_world_size} must be divisible by group_size={group_size}."
        )
    group_num = ffn_world_size // group_size
    subgroups = []
    for base_rank in (0, ffn_world_size):
        if group_stride == 1:
            start_ranks = [base_rank + group_id * group_size for group_id in range(group_num)]
        else:
            start_ranks = [base_rank + group_id for group_id in range(group_num)]
        subgroups.extend([
            [start_rank + rank_idx * group_stride for rank_idx in range(group_size)]
            for start_rank in start_ranks
        ])
    return subgroups


def init_afd_parallel_comm_group(
    comm_manager: CommManager,
    infer_config: InferenceConfig,
    config: LongcatFlashConfig,
):
    """Register the AFD business comm groups.

    Called by both the FFN-role model (here) and the attention-role model (which
    imports this lazily). Every rank registers the same groups with the full
    two-half subgroup partition, because the new CommManager creates each
    subgroup collectively across the whole world.
    """
    parallel_config = infer_config.parallel_config
    world_size = parallel_config.world_size
    if world_size % 2 != 0:
        raise ValueError(f"AFD is only supported when world_size % 2 == 0, but now {world_size=}.")
    comm_manager.register_group(
        name="attn_tp_group",
        subgroups=_build_afd_subgroups(world_size, parallel_config.attn_tp_size),
    )
    comm_manager.register_group(
        name="o_proj_tp_group",
        subgroups=_build_afd_subgroups(world_size, parallel_config.o_proj_tp_size),
    )
    comm_manager.register_group(
        name="embed_tp_group",
        subgroups=_build_afd_subgroups(world_size, parallel_config.embed_tp_size),
    )
    comm_manager.register_group(
        name="lmhead_tp_group",
        subgroups=_build_afd_subgroups(world_size, parallel_config.lmhead_tp_size),
    )
    comm_manager.register_group(
        name="dense_tp_group",
        subgroups=_build_afd_subgroups(world_size, parallel_config.dense_tp_size),
    )
    comm_manager.register_group(
        name="moe_tp_group",
        subgroups=_build_afd_subgroups(world_size, parallel_config.moe_tp_size),
    )
    comm_manager.register_group(
        name="moe_ep_group",
        subgroups=_build_afd_subgroups(
            world_size, parallel_config.moe_ep_size, group_stride=parallel_config.moe_tp_size),
        return_name=True,
    )
    moe_ep_mc2_buffer_size = None
    if parallel_config.moe_ep_size > 1 and parallel_config.moe_tp_size == 1:
        moe_ep_mc2_buffer_size = _calc_longcat_moe_mc2_hccl_buffer_size(
            infer_config, config, world_size // 2
        )
    comm_manager.register_group(
        name="moe_ep_group_mc2",
        subgroups=_build_afd_subgroups(
            world_size, parallel_config.moe_ep_size, group_stride=parallel_config.moe_tp_size),
        return_name=True,
        allow_physical_reuse=False,
        hccl_buffer_size=moe_ep_mc2_buffer_size,
    )


class LongcatFlashFFN(LongcatFlashMoE):
    """
    moe module.
    """
    def __init__(self, config, infer_config: InferenceConfig, comm_manager: CommManager, layer_idx, prefix, **kwargs):
        super().__init__(config, infer_config, comm_manager, layer_idx, prefix, **kwargs)
        self.layer_idx = layer_idx
        # ensure recv/send comm tags do not overlap. Attn send tag value should equal to FFN recv tag.
        self.recv_tag = layer_idx
        self.send_tag = layer_idx + config.num_hidden_layers
        custom_params = self.infer_config.model_config.custom_params
        self.enable_prefetch = custom_params.get("enable_prefetch", False)
        self.npugraph_prefetch_stream = None
        self.ffn_world_size = (
            self.infer_config.parallel_config.world_size // 2
            if custom_params.get("enable_afd", False)
            else 0
        )
        self.global_rank = dist.get_rank()

        dtype_bit = 1 if self.gmm_quant_mode == "w8a8int8" else 2
        self.gmm1_prefetch_size = self.hidden_size * self.intermediate_size * 2 * dtype_bit // \
                                self.moe_tp_size * self.experts.experts_per_rank // 2
        self.gmm2_prefetch_size = self.hidden_size * self.intermediate_size * dtype_bit // \
                                self.moe_tp_size * self.experts.experts_per_rank

    def forward(self, hidden_states, is_prefill=False, cur_topk_list=None):
        recv_tensor = torch.empty_like(hidden_states)
        dist.recv(recv_tensor, src=self.global_rank + self.ffn_world_size, tag=self.recv_tag)

        on_stream = self.npugraph_prefetch_stream is not None and not is_prefill
        if on_stream:
            topk_indices, topk_weights, router_logits = self.router(
                recv_tensor,
                prefetch_stream=self.npugraph_prefetch_stream,
                prefetch_weight=self.experts.w2_weight.data,
                prefetch_size=self.gmm2_prefetch_size,
                enable_prefetch=self.enable_prefetch)
        else:
            topk_indices, topk_weights, router_logits = self.router(recv_tensor)
            if not is_prefill:
                npu_prefetch(self.enable_prefetch, self.experts.w2_weight.data, \
                            router_logits, self.gmm2_prefetch_size, 0)
        if self.force_eplb:
            topk_indices = cur_topk_list
        topk_indices = topk_indices.to(torch.int32)

        if is_prefill:
            result = self.moe_infer_double_routing(recv_tensor, topk_indices, topk_weights)
            dist.send(result, dst=self.global_rank + self.ffn_world_size, tag=self.send_tag)
            return result
        else:
            if self.npugraph_prefetch_stream is not None:
                result, gmm2_out, gmm2_event = self.moe_infer_dispatch_combine(
                    recv_tensor, topk_indices, topk_weights, record_gmm2_event=True)
            else:
                result, gmm2_out = self.moe_infer_dispatch_combine(
                    recv_tensor, topk_indices, topk_weights)
                gmm2_event = None
            dist.send(result, dst=self.global_rank + self.ffn_world_size, tag=self.send_tag)
            return result, gmm2_out, gmm2_event


class FFNDecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: LongcatFlashConfig, infer_config: InferenceConfig, comm_manager: CommManager,
                 layer_idx: int, prefix: str, **kwargs):
        super().__init__()
        self.layer_idx = layer_idx
        self.infer_config = infer_config
        self.comm_manager = comm_manager
        self.mlp = LongcatFlashFFN(
            config, self.infer_config, self.comm_manager, layer_idx, prefix=f"{prefix}.mlp", **kwargs)

    def forward(self, hidden_states, is_prefill=False, cur_topk_list=None):
        return self.mlp(hidden_states, is_prefill, cur_topk_list=cur_topk_list)


class FFNModel(PreTrainedModel):
    def __init__(self, config: LongcatFlashConfig, infer_config: InferenceConfig, comm_manager: CommManager,
                 prefix: str, **kwargs):
        super().__init__(config)
        self.config = config
        self.global_rank = dist.get_rank()
        self.infer_config = infer_config
        self.comm_manager = comm_manager
        custom_params = self.infer_config.model_config.custom_params
        self.enable_superkernel = custom_params.get("enable_superkernel", False)
        self.enable_prefetch = custom_params.get("enable_prefetch", False)
        enable_multi_streams = custom_params.get("enable_multi_streams", 0)
        enable_npugraph_ex = self.infer_config.model_config.exe_mode == "npugraph_ex"
        self.moe_layer_num = config.num_hidden_layers

        self.layers = nn.ModuleList(
            [
                FFNDecoderLayer(config, self.infer_config, self.comm_manager, layer_idx, prefix, **kwargs)
                for layer_idx in range(self.moe_layer_num)
            ]
        )
        self.npugraph_prefetch_stream = None
        if enable_npugraph_ex and enable_multi_streams and self.enable_prefetch:
            self.npugraph_prefetch_stream = torch.npu.Stream()
            for layer in self.layers:
                layer.mlp.npugraph_prefetch_stream = self.npugraph_prefetch_stream
        self.gradient_checkpointing = False

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        hidden_states: Optional[torch.Tensor],
        is_prefill: Optional[bool] = False,
        cur_topk_list: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        for i in range(self.moe_layer_num):
            if is_prefill:
                hidden_states = self.layers[i](hidden_states, is_prefill, cur_topk_list=cur_topk_list)
            else:
                with superkernel_scope(self.enable_superkernel, f"scope_{i}_moe", ""):
                    hidden_states, gmm2_out, gmm2_event = self.layers[
                        i](hidden_states, is_prefill, cur_topk_list=cur_topk_list)
                if i < self.moe_layer_num - 1:
                    on_stream = self.npugraph_prefetch_stream is not None
                    with npu_stream_switch(on_stream, self.npugraph_prefetch_stream):
                        if on_stream and gmm2_event is not None:
                            self.npugraph_prefetch_stream.wait_event(gmm2_event)
                        npu_prefetch(self.enable_prefetch, self.layers[i + 1].mlp.router.classifier.weight.data, \
                                     gmm2_out, self.layers[i + 1].mlp.router.prefetch_size, 0)
                        npu_prefetch(self.enable_prefetch, self.layers[i + 1].mlp.experts.w13_weight.data, \
                                     gmm2_out, self.layers[i + 1].mlp.gmm1_prefetch_size, 0)

        if self.npugraph_prefetch_stream is not None and not is_prefill:
            with npu_stream_switch(True, self.npugraph_prefetch_stream):
                prefetch_done_event = torch.npu.current_stream().record_event()
            torch.npu.current_stream().wait_event(prefetch_done_event)

        return hidden_states


class FFNForCausalLM(PreTrainedModel, GenerationMixin):
    def __init__(self, config, infer_config: InferenceConfig, comm_manager: CommManager, is_mtp=False,
                 prefix: str = ""):
        super().__init__(config)
        self.config = config
        self.infer_config = infer_config
        self.comm_manager = comm_manager
        self.num_experts_per_tok = config.moe_topk
        self.enable_online_split_weight = True
        self.enable_weight_nz = self.infer_config.model_config.enable_weight_nz
        self.moe_ep_size = self.infer_config.parallel_config.moe_ep_size
        self.num_experts = (
            config.n_routed_experts
            if config.zero_expert_num is None
            else config.n_routed_experts + config.zero_expert_num
        )
        self.get_parallel_settings()
        init_afd_parallel_comm_group(self.comm_manager, self.infer_config, self.config)

        self.model = FFNModel(config, self.infer_config, self.comm_manager, prefix)
        # Initialize weights and apply final processing
        self.post_init()

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    def prefill(
        self,
        **kwargs
    ):
        logits = self.forward(
            is_prefill=True,
            **kwargs
        )
        return logits

    def decode(
        self,
        **kwargs
    ):
        logits = self.forward(
            is_prefill=False,
            **kwargs
        )
        return logits

    def get_parallel_settings(self):
        self.attn_dp_size = self.infer_config.parallel_config.attn_dp_size
        self.moe_ep_size = self.infer_config.parallel_config.moe_ep_size
        self.moe_tp_size = self.infer_config.parallel_config.moe_tp_size

    def build_afd_inputs(self, num_tokens, is_prefill, dtype, device):
        """Dummy inputs for an AFD FFN-only rank; the real hidden states arrive via dist.recv."""
        parallel_config = self.infer_config.parallel_config
        hidden_num_tokens = num_tokens
        if is_prefill and parallel_config.attn_tp_size > 1 and parallel_config.moe_ep_size > 1:
            hidden_num_tokens = (num_tokens + parallel_config.attn_tp_size - 1) // parallel_config.attn_tp_size
        hidden_states = torch.ones((hidden_num_tokens, self.config.hidden_size), dtype=dtype, device=device)
        return {
            "input_ids": torch.zeros(num_tokens, dtype=torch.long, device=device),
            "hidden_states": hidden_states,
            "is_prefill": is_prefill,
        }

    def process_weights_after_loading(self):
        if not self.enable_online_split_weight:
            if hasattr(self, "scale_dtype_adapter"):
                self.scale_dtype_adapter()
            if hasattr(self, "cast_format"):
                self.cast_format()
            return

        for module_name, module in self.named_modules():
            quant_method = getattr(module, "quant_method", None)
            scales_dtype = {}
            if "gate_up_proj" in module_name:
                scales_dtype['scale_dtype'] = torch.float
            if "down_proj" in module_name:
                scales_dtype['smooth_scale_dtype'] = torch.float

            if isinstance(quant_method, QuantizeMethodBase):
                quant_method.process_weights_after_loading(
                    module, is_nz=self.enable_weight_nz, scales_dtype=scales_dtype)
            if isinstance(quant_method, CompressedTensorW8A8Int8MoEGMMMethod):
                if self.moe_ep_size > 1:
                    group = self.comm_manager.get_group("moe_ep_group")
                    all_experts_smooth_scale = module.smooth_scale_1.data.new_empty(
                        module.smooth_scale_1.data.shape[0] * self.moe_ep_size,
                        module.smooth_scale_1.data.shape[1],
                    )
                    dist.all_gather_into_tensor(all_experts_smooth_scale, module.smooth_scale_1.data, group=group)
                    module.smooth_scale_1.data = all_experts_smooth_scale

    def forward(
        self,
        hidden_states: [torch.Tensor],
        is_prefill: Optional[bool] = False,
        cur_topk_list: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        return self.model(hidden_states=hidden_states,
                          is_prefill=is_prefill,
                          cur_topk_list=cur_topk_list,
                          **kwargs)

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]) -> Set[str]:
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("merge_up_gate_proj", "gate_proj", 0),
            ("merge_up_gate_proj", "up_proj", 1),
        ]

        # (param_name, weight_name, expert_id, shard_id)
        expert_params_mapping = FusedMoEGMM.make_expert_params_mapping(
            ckpt_gate_proj_name="gate_proj",
            ckpt_down_proj_name="down_proj",
            ckpt_up_proj_name="up_proj",
            num_experts=self.config.n_routed_experts)

        params_dict = dict(self.named_parameters())
        loaded_params: Set[str] = set()
        for name, loaded_weight in weights:
            if self.config.architectures[0] == 'LongcatFlashForCausalLM' and self.config.num_nextn_predict_layers > 0:
                mtp_prefix = "model.mtp"
                if name.startswith(mtp_prefix):
                    continue

            for (param_name, weight_name, shard_id) in stacked_params_mapping:
                # Skip non-stacked layers and experts (experts handled below).
                if weight_name not in name:
                    continue

                if (("mlp.experts." in name) and name not in params_dict):
                    continue
                name = name.replace(weight_name, param_name)
                if name.endswith(".bias") and name not in params_dict:
                    continue

                if name not in params_dict:
                    continue
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                for mapping in expert_params_mapping:
                    param_name, weight_name, expert_id, shard_id = mapping
                    if weight_name not in name:
                        continue
                    name = name.replace(weight_name, param_name)

                    if name not in params_dict:
                        continue
                    param = params_dict[name]
                    weight_loader = param.weight_loader
                    weight_loader(param,
                                  loaded_weight,
                                  name,
                                  shard_id=shard_id,
                                  expert_id=expert_id)
                    break
                else:
                    if name.endswith(".bias") and name not in params_dict:
                        continue
                    if name not in params_dict:
                        continue
                    param = params_dict[name]
                    weight_loader = getattr(param, "weight_loader",
                                            default_weight_loader)
                    weight_loader(param, loaded_weight)
            loaded_params.add(name)
        return loaded_params
