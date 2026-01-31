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

from typing import Callable, Iterable, Optional, Tuple, List, Dict, Set
import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist
import torch_npu
import torchair as tng

from transformers.generation import GenerationMixin
from transformers.modeling_layers import GradientCheckpointingLayer
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import auto_docstring, can_return_tuple, logging

from executor.utils import superkernel_scope, npu_prefetch
from module.fuse_moe_gmm import FusedMoEGMM
from executor.model_loader.weight_utils import default_weight_loader
from executor.utils import (init_comm_group, get_default_group)

from .configuration_longcat_flash import LongcatFlashConfig
from .modeling_longcat_flash import LongcatFlashMoE

logger = logging.get_logger(__name__)


class LongcatFlashFFN(LongcatFlashMoE):
    """
    moe module.
    """
    def __init__(self, config, runner_settings, layer_idx, prefix, **kwargs):
        super().__init__(config, runner_settings, layer_idx, prefix, **kwargs)
        self.layer_idx = layer_idx
        # ensure recv/send comm tags do not overlap. Attn send tag value should equal to FFN recv tag.
        self.recv_tag = layer_idx
        self.send_tag = layer_idx + config.num_hidden_layers
        self.enable_prefetch = self.runner_settings.get("model_config").get("enable_prefetch", False)
        self.ffn_world_size = self.runner_settings.get("ffn_world_size", 0)
        self.global_rank = dist.get_rank()

        dtype_bit = 1 if self.gmm_quant_mode == "w8a8" else 2
        self.gmm1_prefetch_size = self.hidden_size * self.intermediate_size * 2 * dtype_bit // \
                                self.moe_tp_size * self.experts.experts_per_rank // 2
        self.gmm2_prefetch_size = self.hidden_size * self.intermediate_size * dtype_bit // \
                                self.moe_tp_size * self.experts.experts_per_rank

    def forward(self, hidden_states, is_prefill=False, cur_topk_list=None):
        recv_tensor = torch.empty_like(hidden_states)
        dist.recv(recv_tensor, src=self.global_rank + self.ffn_world_size, tag=self.recv_tag)

        topk_indices, topk_weights, router_logits = self.router(recv_tensor)
        if not is_prefill:
            npu_prefetch(self.enable_prefetch, self.experts.w2_weight.data, \
                        router_logits, self.gmm2_prefetch_size, 0)
        if self.perfect_eplb:
            topk_indices = cur_topk_list
        topk_indices = topk_indices.to(torch.int32)

        if is_prefill:
            result = self.moe_infer_double_routing(recv_tensor, topk_indices, topk_weights)
            dist.send(result, dst=self.global_rank + self.ffn_world_size, tag=self.send_tag)
            return result
        else:
            result, gmm2_out = self.moe_infer_dispatch_combine(recv_tensor, topk_indices, topk_weights)
            dist.send(result, dst=self.global_rank + self.ffn_world_size, tag=self.send_tag)
            return result, gmm2_out


class FFNDecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: LongcatFlashConfig, runner_settings: Dict, layer_idx: int, prefix: str, **kwargs):
        super().__init__()
        self.layer_idx = layer_idx
        self.runner_settings = runner_settings
        self.mlp = LongcatFlashFFN(config, runner_settings, layer_idx, prefix=f"{prefix}.mlp", **kwargs)

    def forward(self, hidden_states, is_prefill=False, cur_topk_list=None):
        return self.mlp(hidden_states, is_prefill, cur_topk_list=cur_topk_list)


class FFNModel(PreTrainedModel):
    def __init__(self, config: LongcatFlashConfig, runner_settings: Dict, prefix: str, **kwargs):
        super().__init__(config)
        self.config = config
        self.global_rank = dist.get_rank()
        self.runner_settings = runner_settings
        self.enable_superkernel = runner_settings.get("model_config").get("enable_superkernel", False)
        self.enable_prefetch = runner_settings.get("model_config").get("enable_prefetch", False)
        self.moe_layer_num = config.num_hidden_layers

        self.hccl_comm_dict = kwargs.get("hccl_comm_dict", None)
        self.layers = nn.ModuleList(
            [FFNDecoderLayer(config, runner_settings, layer_idx, prefix, **kwargs)\
                                      for layer_idx in range(self.moe_layer_num)]
        )
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
                    hidden_states, gmm2_out = self.layers[i](hidden_states, is_prefill, cur_topk_list=cur_topk_list)
                if i < self.moe_layer_num - 1:
                    npu_prefetch(self.enable_prefetch, self.layers[i + 1].mlp.router.classifier.weight.data, \
                                 gmm2_out, self.layers[i + 1].mlp.router.prefetch_size, 0)
                    npu_prefetch(self.enable_prefetch, self.layers[i + 1].mlp.experts.w13_weight.data, \
                                 gmm2_out, self.layers[i + 1].mlp.gmm1_prefetch_size, 0)

        return hidden_states


@auto_docstring
class FFNForCausalLM(PreTrainedModel, GenerationMixin):
    def __init__(self, config, runner_settings, is_mtp=False, prefix: str = ""):
        super().__init__(config)
        self.config = config
        self.runner_settings = runner_settings
        self.top_k = config.moe_topk
        self.perfect_eplb = runner_settings.get("model_config").get("perfect_eplb", False)
        self.moe_ep_size = runner_settings.get("parallel_config").get("moe_ep_size", 1)

        self.num_experts = (
            config.n_routed_experts
            if config.zero_expert_num is None
            else config.n_routed_experts + config.zero_expert_num
        )
        self.experts_per_rank = self.num_experts // self.moe_ep_size
        self.get_parallel_settings()
        kwargs = {}
        default_pg = get_default_group()
        if default_pg is not None:
            if dist.get_world_size() > 1:
                self.hccl_comm_dict = self.init_parallel_comm_group()
                kwargs.update({"hccl_comm_dict": self.hccl_comm_dict})

        self.model = FFNModel(config, runner_settings, prefix, **kwargs)
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
        self.attn_dp_size = self.runner_settings.get("parallel_config").get("attn_dp_size", 1)
        self.moe_ep_size = self.runner_settings.get("parallel_config").get("moe_ep_size", 1)
        self.moe_tp_size = self.runner_settings.get("parallel_config").get("moe_tp_size", 1)

    def init_parallel_comm_group(self):
        """
        In AFD scenario, the FFN module is deployed before the ATTN module.
        """
        world_size = dist.get_world_size()
        global_rank = dist.get_rank()

        moe_tp_group = init_comm_group(
            global_rank=global_rank, group_num=world_size // self.moe_tp_size, world_size=world_size,
            group_stride=1, group_name="moe_tp_group")

        moe_ep_group, moe_ep_group_name = init_comm_group(
            global_rank=global_rank, group_num=world_size // self.moe_ep_size, world_size=world_size,
            group_stride=self.moe_tp_size, group_name="moe_ep_group", return_name=True)

        hccl_comm_dict = {
            "moe_tp_group": moe_tp_group, "moe_ep_group": moe_ep_group,
            "moe_ep_group_name": moe_ep_group_name,
        }
        return hccl_comm_dict

    @can_return_tuple
    @auto_docstring
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

    def gen_cur_topk_idx(
        self,
        is_prefill,
        batch_size,
        seq_len
    ):
        if not self.perfect_eplb:
            return None

        global_rank = dist.get_rank()
        if is_prefill:
            if self.moe_ep_size != 1:
                tokens_per_rank_prefill = batch_size * seq_len
            else:
                tokens_per_rank_prefill = batch_size * seq_len * self.attn_dp_size
            step_prefill = tokens_per_rank_prefill * self.top_k
            cur_topk_list_prefill = [
                (i + global_rank) % self.num_experts for i in range(step_prefill)]
            cur_topk_list = torch.Tensor(cur_topk_list_prefill).int().view(tokens_per_rank_prefill, -1).npu()
        else:
            if self.moe_tp_size > 1:
                tokens_per_rank_decode = batch_size * self.top_k * seq_len
                cur_topk_list_decode = []
                for offset in range(self.moe_ep_size):
                    expert_start = offset * self.experts_per_rank
                    expert_end = expert_start + tokens_per_rank_decode
                    cur_topk_list_decode = cur_topk_list_decode + [i for i in range(expert_start, expert_end)]
                cur_topk_list = torch.Tensor(cur_topk_list_decode).int().view(batch_size * seq_len, -1).npu()
            else:
                step_decode = batch_size * self.top_k * seq_len
                step_gap = self.num_experts // self.moe_ep_size if step_decode < self.num_experts else 1
                cur_topk_list_decode = [
                    ((i + global_rank // step_gap * step_gap) * step_gap +
                    global_rank % step_gap) % self.num_experts for i in range(step_decode)
                ]
                cur_topk_list = torch.Tensor(cur_topk_list_decode).int().view(batch_size * seq_len, -1).npu()
        return cur_topk_list

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