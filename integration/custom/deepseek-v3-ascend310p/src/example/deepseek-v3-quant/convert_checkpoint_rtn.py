# coding=utf-8
# Copyright (c) 2025 QINGMAO INTELLIGENCE TECHNOLOGY (BEIJING) CO., LTD. and Huawei Technologies Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import time
from collections import defaultdict
from typing import Dict, List, Literal
import gc
from functools import lru_cache
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from glob import glob
import inspect
import json
import os
from pathlib import Path
import shutil
from tqdm import tqdm
from safetensors.torch import load_file, save_file
import torch


torch.set_grad_enabled(False)


timers = defaultdict(float)

class Timer:
    """Context manager for measuring the execution time of a code block."""

    def __init__(self, name: str):
        self.name = name

    def __enter__(self):
        self.start = time.perf_counter()

    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed = time.perf_counter() - self.start
        timers[self.name] += elapsed

def main(args: argparse.Namespace) -> None:
    start_main = time.perf_counter()

    with open(os.path.join(args.model_name_or_path, "config.json")) as f:
        cfg = json.load(f)
    num_attention_heads = cfg["num_attention_heads"]
    qk_nope_head_dim = cfg["qk_nope_head_dim"]
    qk_rope_head_dim = cfg["qk_rope_head_dim"]
    v_head_dim = cfg["v_head_dim"]
    kv_lora_rank = cfg["kv_lora_rank"]
    num_hidden_layers = cfg["num_hidden_layers"]

    model_index_file = os.path.join(args.model_name_or_path, "model.safetensors.index.json")
    with open(model_index_file, "r") as f:
        model_index = json.load(f)
    weight_map = model_index["weight_map"]

    # Cache for loaded safetensors files
    loaded_files: Dict[str, Dict[str, torch.Tensor]] = {}
    mi_weight: List[Dict[str, torch.Tensor]] = [{} for _ in range(args.world_size)]

    def get_tensor(tensor_name):
        try:
            file_name = weight_map[tensor_name]
            if file_name not in loaded_files:
                file_path = os.path.join(args.model_name_or_path, file_name)
                data = load_file(file_path)
                for k, v in data.items():
                    if ".mlp.gate.e_score_correction_bias" in k or ".mlp.gate.weight" in k:
                        # convert torch.float
                        data[k] = v.to(torch.float)
                    elif "norm" in k or "embed_tokens" in k or "lm_head" in k:
                        # convert torch.float16
                        data[k] = v.to(torch.float16)

                # Memory management: keep only the 5 most recently used files
                while len(loaded_files) >= 5:
                    oldest_file = next(iter(loaded_files))
                    del loaded_files[oldest_file]
                loaded_files[file_name] = data
            return loaded_files[file_name][tensor_name]
        except:
            return torch.ones(1)

    def transpose(weight, size):
        n0 = weight.shape[0]
        n1 = weight.shape[1]
        weight = weight.view(n0, n1 // size, size)
        weight = weight.permute(1, 0, 2)

        return weight.contiguous().view(n0, n1)

    def transpose1(weight, size):
        n0 = weight.shape[0]
        n1 = weight.shape[1]
        n2 = weight.shape[2]

        weight = weight.view(n0, n1, n2 // size, size)
        weight = weight.permute(0, 2, 1, 3)

        return weight.contiguous().view(n0, n1, n2)

    # embedding_weight, norm_weight in every rank
    for i in range(args.world_size):
        mi_weight[i].update(
            {
                "model.embed_tokens.weight": get_tensor("model.embed_tokens.weight"),
                "model.norm.weight": get_tensor("model.norm.weight"),
            }
        )
    # lmhead_weight: world_size/tp_size goroup，each group has tp_size weight with out_channel split
    for tp_rank, lmhead_weight in enumerate(get_tensor("lm_head.weight").chunk(args.tp_size, 0)):
        for rank in range(args.world_size):
            if rank % args.tp_size == tp_rank:
                mi_weight[rank].update(
                    {
                        "lm_head.weight": transpose(lmhead_weight, 16).contiguous(),
                    }
                )

    # process weight in layers
    for layer_id in tqdm(range(num_hidden_layers), desc="转换权重"):
        # rmsnor weight
        input_norm_weight = get_tensor(f"model.layers.{layer_id}.input_layernorm.weight")
        post_norm_weight = get_tensor(f"model.layers.{layer_id}.post_attention_layernorm.weight")
        for rank in range(args.world_size):
            mi_weight[rank].update(
                {
                    f"model.layers.{layer_id}.input_layernorm.weight": input_norm_weight,
                    f"model.layers.{layer_id}.post_attention_layernorm.weight": post_norm_weight,
                }
            )

        # ===================================
        # QKV down proj weight without TP
        # ===================================
        # W_DQ: (q_lora_rank, hidden_size) = (1536, 7168)
        q_a_proj_weight = get_tensor(f"model.layers.{layer_id}.self_attn.q_a_proj.weight")
        q_a_proj_a_scale = get_tensor(f"model.layers.{layer_id}.self_attn.q_a_proj.a_scale")
        q_a_proj_smooth_factor = get_tensor(
            f"model.layers.{layer_id}.self_attn.q_a_proj.smooth_factor"
        )
        q_a_proj_w_scale = get_tensor(f"model.layers.{layer_id}.self_attn.q_a_proj.w_scale")
        # gamma_DQ: (q_lora_rank) = (1536)
        q_a_layernorm_weight = get_tensor(f"model.layers.{layer_id}.self_attn.q_a_layernorm.weight")
        # quant_scale, dequant_scale
        q_a_proj_dequant_scale = q_a_proj_w_scale

        # concat: (W_DKV, W_KR): (kv_lora_rank + qk_rope_head_dim, hidden_size) = (512 + 64, 7168) = (576, 7168)
        # W_DKV: (kv_lora_rank, hidden_size) = (512, 7168)
        # W_KR: (qk_rope_head_dim, hidden_size) = (64, 7168)
        kv_a_proj_weight = get_tensor(
            f"model.layers.{layer_id}.self_attn.kv_a_proj_with_mqa.weight"
        )
        kv_a_proj_a_scale = get_tensor(
            f"model.layers.{layer_id}.self_attn.kv_a_proj_with_mqa.a_scale"
        )
        kv_a_proj_w_scale = get_tensor(
            f"model.layers.{layer_id}.self_attn.kv_a_proj_with_mqa.w_scale"
        )
        # gamma_DKV: (kv_lora_rank) = (512)
        kv_a_layernorm_weight = get_tensor(
            f"model.layers.{layer_id}.self_attn.kv_a_layernorm.weight"
        )

        # quant_scale, dquant_scale
        kv_a_proj_dequant_scale =  kv_a_proj_w_scale

        # concat
        qkv_down_proj_weight = torch.cat([q_a_proj_weight, kv_a_proj_weight], dim=0).contiguous()
        print(q_a_proj_dequant_scale.shape)
        print(kv_a_proj_dequant_scale.shape)
        qkv_down_proj_dequant_scale = (
            torch.cat([q_a_proj_dequant_scale, kv_a_proj_dequant_scale], dim=0)
            .view(1, -1)
            .contiguous()
        )

        for rank in range(args.world_size):
            mi_weight[rank].update(
                {
                    f"model.layers.{layer_id}.self_attn.qkv_down_proj.weight": transpose(qkv_down_proj_weight, 32),
                    f"model.layers.{layer_id}.self_attn.q_a_layernorm.weight": q_a_layernorm_weight,
                    f"model.layers.{layer_id}.self_attn.kv_a_layernorm.weight": kv_a_layernorm_weight,
                    f"model.layers.{layer_id}.self_attn.qkv_down_proj.dequant_scale": qkv_down_proj_dequant_scale,
                }
            )

        # ===================================
        # Q up proj weight with channel split
        # ===================================
        # W_UQ: (nheads * qk_nope_head_dim, q_lora_rank) = (128 * 128, 1536) = (16384, 1536)
        q_nope_weight = get_tensor(f"model.layers.{layer_id}.self_attn.q_b1_proj.weight")
        # W_QR: (nheads * qk_rope_head_dim, q_lora_rank) = (128 * 64, 1536) = (8192, 1536)
        q_rope_weight = get_tensor(f"model.layers.{layer_id}.self_attn.q_b2_proj.weight")
        q_nope_proj_a_scale = get_tensor(f"model.layers.{layer_id}.self_attn.q_b1_proj.a_scale")
        q_nope_proj_smooth_factor = get_tensor(
            f"model.layers.{layer_id}.self_attn.q_b1_proj.smooth_factor"
        )
        q_nope_proj_w_scale = get_tensor(f"model.layers.{layer_id}.self_attn.q_b1_proj.w_scale")
        q_rope_proj_a_scale = get_tensor(f"model.layers.{layer_id}.self_attn.q_b2_proj.a_scale")
        q_rope_proj_smooth_factor = get_tensor(
            f"model.layers.{layer_id}.self_attn.q_b2_proj.smooth_factor"
        )
        q_rope_proj_w_scale = get_tensor(f"model.layers.{layer_id}.self_attn.q_b2_proj.w_scale")
        q_nope_proj_dequant_scale = (q_nope_proj_w_scale).view(1, -1)
        q_rope_proj_dequant_scale = (q_rope_proj_w_scale).view(1, -1)
        for tp_rank, (w_uq, w_qr, w_uq_dequant_scale, w_qr_dequant_scale) in enumerate(
            zip(
                q_nope_weight.chunk(args.tp_size, 0),
                q_rope_weight.chunk(args.tp_size, 0),
                q_nope_proj_dequant_scale.chunk(args.tp_size, 1),
                q_rope_proj_dequant_scale.chunk(args.tp_size, 1),
            )
        ):
            mi_w_uq = w_uq.contiguous()
            mi_w_qr = w_qr.contiguous()
            mi_w_uq_dequant_scale = w_uq_dequant_scale.contiguous()
            mi_w_qr_dequant_scale = w_qr_dequant_scale.contiguous()
            for rank in range(args.world_size):
                if rank % args.tp_size == tp_rank:
                    mi_weight[rank].update(
                        {
                            f"model.layers.{layer_id}.self_attn.q_nope_proj.weight": transpose(mi_w_uq, 32),
                            f"model.layers.{layer_id}.self_attn.q_rope_proj.weight": transpose(mi_w_qr, 32),
                            f"model.layers.{layer_id}.self_attn.q_nope_proj.dequant_scale": mi_w_uq_dequant_scale,
                            f"model.layers.{layer_id}.self_attn.q_rope_proj.dequant_scale": mi_w_qr_dequant_scale,
                        }
                    )
        # ===================================
        # process KV up proj weight with channel split
        # ===================================
        # W_UK: (nheads, kv_lora_rank=512, qk_nope_head_dim=128)
        k_b_proj_weight = torch.stack(
            [
                get_tensor(f"model.layers.{layer_id}.self_attn.kv_b2_proj.{head_idx}.weight")
                for head_idx in range(num_attention_heads)
            ],
            dim=0,
        ).transpose(1, 2)
        # W_UV: (nheads, v_head_dim=128, kv_lora_rank=128)
        v_b_proj_weight = torch.stack(
            [
                get_tensor(f"model.layers.{layer_id}.self_attn.kv_b1_proj.{head_idx}.weight")
                for head_idx in range(num_attention_heads)
            ],
            dim=0,
        ).transpose(1, 2)

        k_b_proj_a_scale = [
            get_tensor(f"model.layers.{layer_id}.self_attn.kv_b2_proj.{head_idx}.a_scale")
            for head_idx in range(num_attention_heads)
        ]
        k_b_proj_smooth_factor = [
            get_tensor(f"model.layers.{layer_id}.self_attn.kv_b2_proj.{head_idx}.smooth_factor")
            for head_idx in range(num_attention_heads)
        ]  # nhead * (1, 128)
        k_b_proj_w_scale = [
            get_tensor(f"model.layers.{layer_id}.self_attn.kv_b2_proj.{head_idx}.w_scale")
            for head_idx in range(num_attention_heads)
        ]  # nhead * (kv_lora_rank=512, 1)
        k_b_proj_dequant_scale = torch.stack(
            [
                k_b_proj_w_scale[head_idx]
                for head_idx in range(num_attention_heads)
            ],
            dim=0,
        ).transpose(1, 2)  # batch_dequantize: (nhead, 1, kv_lora_rank=512)

        v_b_proj_a_scale = [
            get_tensor(f"model.layers.{layer_id}.self_attn.kv_b1_proj.{head_idx}.a_scale")
            for head_idx in range(num_attention_heads)
        ]
        v_b_proj_smooth_factor = [
            get_tensor(f"model.layers.{layer_id}.self_attn.kv_b1_proj.{head_idx}.smooth_factor")
            for head_idx in range(num_attention_heads)
        ]  # nhead * (1, 512)
        v_b_proj_w_scale = [
            get_tensor(f"model.layers.{layer_id}.self_attn.kv_b1_proj.{head_idx}.w_scale")
            for head_idx in range(num_attention_heads)
        ]  # nhead * (v_head_dim=128, 1)

        v_b_proj_dequant_scale = torch.stack(
            [
                v_b_proj_w_scale[head_idx]
                for head_idx in range(num_attention_heads)
            ],
            dim=0,
        ).transpose(1, 2)  # batch_dequantize: (nhead, 1, v_head_dim=128)
        for tp_rank, (
            w_uk,
            w_uv,
            w_uk_dequant_scale,
            w_uv_dequant_scale,
        ) in enumerate(
            zip(
                k_b_proj_weight.chunk(args.tp_size, 0),
                v_b_proj_weight.chunk(args.tp_size, 0),
                k_b_proj_dequant_scale.chunk(args.tp_size, 0),
                v_b_proj_dequant_scale.chunk(args.tp_size, 0),
            )
        ):
            mi_w_uk = w_uk.contiguous()
            mi_w_uv = w_uv.contiguous()
            mi_w_uk_dequant_scale = w_uk_dequant_scale.contiguous()
            mi_w_uv_dequant_scale = w_uv_dequant_scale.contiguous()
            for rank in range(args.world_size):
                if rank % args.tp_size == tp_rank:
                    mi_weight[rank].update(
                        {
                            f"model.layers.{layer_id}.self_attn.k_b_proj.weight": mi_w_uk,
                            f"model.layers.{layer_id}.self_attn.v_b_proj.weight": mi_w_uv,
                            f"model.layers.{layer_id}.self_attn.k_b_proj.dequant_scale": mi_w_uk_dequant_scale,
                            f"model.layers.{layer_id}.self_attn.v_b_proj.dequant_scale": mi_w_uv_dequant_scale,
                        }
                    )

        # W_O: (hidden_size, nhead* v_head_dim) = (7168, 16384)
        o_proj_weight = get_tensor(f"model.layers.{layer_id}.self_attn.o_proj.weight")
        o_proj_a_scale = get_tensor(f"model.layers.{layer_id}.self_attn.o_proj.a_scale")
        o_proj_smooth_factor = get_tensor(f"model.layers.{layer_id}.self_attn.o_proj.smooth_factor")
        o_proj_w_scale = get_tensor(f"model.layers.{layer_id}.self_attn.o_proj.w_scale")
        # o_proj_quant_scale = o_proj_a_scale * o_proj_smooth_factor
        o_proj_dequant_scale = o_proj_w_scale

        for tp_rank, wo in enumerate(o_proj_weight.chunk(args.tp_size, 1)):
            mi_wo = wo.contiguous()
            mi_o_proj_dequant_scale = o_proj_dequant_scale.contiguous()
            # mi_w_o_quant_scale = w_o_quant_scale.contiguous()
            for rank in range(args.world_size):
                if rank % args.tp_size == tp_rank:
                    mi_weight[rank].update(
                        {
                            f"model.layers.{layer_id}.self_attn.o_proj.weight": transpose(mi_wo, 32),
                            f"model.layers.{layer_id}.self_attn.o_proj.dequant_scale": mi_o_proj_dequant_scale,
                        }
                    )

        if layer_id < 3:
            gate_proj_weight = get_tensor(f"model.layers.{layer_id}.mlp.gate_proj.weight")
            up_proj_weight = get_tensor(f"model.layers.{layer_id}.mlp.up_proj.weight")
            down_proj_weight = get_tensor(f"model.layers.{layer_id}.mlp.down_proj.weight")

            ffn1_proj_a_scale = get_tensor(f"model.layers.{layer_id}.mlp.gate_proj.a_scale")
            ffn1_proj_smooth_factor = get_tensor(
                f"model.layers.{layer_id}.mlp.gate_proj.smooth_factor"
            )
            gate_proj_w_scale = get_tensor(f"model.layers.{layer_id}.mlp.gate_proj.w_scale")
            up_proj_w_scale = get_tensor(f"model.layers.{layer_id}.mlp.up_proj.w_scale")

            gate_proj_dequant_scale = (gate_proj_w_scale).view(1, -1)
            up_proj_dequant_scale = (up_proj_w_scale).view(1, -1)

            down_proj_a_scale = get_tensor(f"model.layers.{layer_id}.mlp.down_proj.a_scale")
            down_proj_smooth_factor = get_tensor(
                f"model.layers.{layer_id}.mlp.down_proj.smooth_factor"
            )
            down_proj_w_scale = get_tensor(f"model.layers.{layer_id}.mlp.down_proj.w_scale")
            ffn2_proj_dequant_scale = (down_proj_w_scale).view(1, -1)

            for tp_rank, (
                gw,
                uw,
                dw,
                gw_dequant_scale,
                uw_dequant_scale,
            ) in enumerate(
                zip(
                    gate_proj_weight.chunk(args.tp_size, 0),
                    up_proj_weight.chunk(args.tp_size, 0),
                    down_proj_weight.chunk(args.tp_size, 1),
                    gate_proj_dequant_scale.chunk(args.tp_size, 1),
                    up_proj_dequant_scale.chunk(args.tp_size, 1),
                )
            ):
                mi_ffn1_proj_weight = torch.cat([gw, uw], dim=0).contiguous()
                mi_ffn2_proj_weight = dw.contiguous()
                mi_ffn1_proj_dequant_scale = torch.cat(
                    [gw_dequant_scale, uw_dequant_scale], dim=1
                ).contiguous()
                mi_ffn2_proj_dequant_scale = ffn2_proj_dequant_scale.contiguous()

                for rank in range(args.world_size):
                    if rank % args.tp_size == tp_rank:
                        mi_weight[rank].update(
                            {
                                f"model.layers.{layer_id}.ffn1_proj.weight": transpose(mi_ffn1_proj_weight, 32),
                                f"model.layers.{layer_id}.ffn2_proj.weight": transpose(mi_ffn2_proj_weight, 32),
                                f"model.layers.{layer_id}.ffn1_proj.dequant_scale": mi_ffn1_proj_dequant_scale,
                                f"model.layers.{layer_id}.ffn2_proj.dequant_scale": mi_ffn2_proj_dequant_scale,
                            }
                        )
        else:
            # shared expert
            shared_experts_gate_proj_weight = get_tensor(
                f"model.layers.{layer_id}.mlp.shared_experts.gate_proj.weight"
            )
            shared_experts_up_proj_weight = get_tensor(
                f"model.layers.{layer_id}.mlp.shared_experts.up_proj.weight"
            )
            shared_experts_down_proj_weight = get_tensor(
                f"model.layers.{layer_id}.mlp.shared_experts.down_proj.weight"
            )
            mi_ffn1_proj_weight = torch.cat(
                [shared_experts_gate_proj_weight, shared_experts_up_proj_weight], dim=0
            ).contiguous()
            mi_ffn2_proj_weight = shared_experts_down_proj_weight.contiguous()

            shared_experts_ffn1_proj_a_scale = get_tensor(
                f"model.layers.{layer_id}.mlp.shared_experts.gate_proj.a_scale"
            )
            shared_experts_ffn1_proj_smooth_factor = get_tensor(
                f"model.layers.{layer_id}.mlp.shared_experts.gate_proj.smooth_factor"
            )
            shared_experts_gate_proj_w_scale = get_tensor(
                f"model.layers.{layer_id}.mlp.shared_experts.gate_proj.w_scale"
            )
            shared_experts_up_proj_w_scale = get_tensor(
                f"model.layers.{layer_id}.mlp.shared_experts.up_proj.w_scale"
            )
            shared_experts_ffn1_proj_w_scale = torch.cat(
                [shared_experts_gate_proj_w_scale, shared_experts_up_proj_w_scale], dim=0
            )

            shared_experts_ffn1_proj_dequant_scale = (
                (shared_experts_ffn1_proj_w_scale)
                .view(1, -1)
                .contiguous()
            )

            shared_experts_down_proj_a_scale = get_tensor(
                f"model.layers.{layer_id}.mlp.shared_experts.down_proj.a_scale"
            )
            shared_experts_down_proj_smooth_factor = get_tensor(
                f"model.layers.{layer_id}.mlp.shared_experts.down_proj.smooth_factor"
            )
            shared_experts_down_proj_w_scale = get_tensor(
                f"model.layers.{layer_id}.mlp.shared_experts.down_proj.w_scale"
            )

            shared_experts_down_proj_dequant_scale = (
                (shared_experts_down_proj_w_scale)
                .view(1, -1)
                .contiguous()
            )

            for rank in range(args.world_size):
                mi_weight[rank].update(
                    {
                        f"model.layers.{layer_id}.ffn1_proj.weight": transpose(mi_ffn1_proj_weight,32),
                        f"model.layers.{layer_id}.ffn2_proj.weight": transpose(mi_ffn2_proj_weight,32),
                        f"model.layers.{layer_id}.ffn1_proj.dequant_scale": shared_experts_ffn1_proj_dequant_scale,
                        f"model.layers.{layer_id}.ffn2_proj.dequant_scale": shared_experts_down_proj_dequant_scale,
                    }
                )

            # moe gate
            gate_weight = get_tensor(f"model.layers.{layer_id}.mlp.gate.weight").float()
            gate_bias = get_tensor(f"model.layers.{layer_id}.mlp.gate.e_score_correction_bias")
            mi_gate_weight = gate_weight
            for rank in range(args.world_size):
                mi_weight[rank].update(
                    {
                        f"model.layers.{layer_id}.mlp.gate.weight": mi_gate_weight.contiguous(),
                        f"model.layers.{layer_id}.mlp.gate.bias": gate_bias,
                    }
                )

            # routed expert
            experts_ffn1_proj = []
            experts_ffn1_proj_dequant_scale = []
            experts_ffn2_proj = []
            experts_ffn2_proj_dequant_scale = []

            expert_ffn1_proj_a_scale = get_tensor(
                f"model.layers.{layer_id}.mlp.experts.0.gate_proj.a_scale"
            )
            expert_ffn1_proj_smooth_factor = get_tensor(
                f"model.layers.{layer_id}.mlp.experts.0.gate_proj.smooth_factor"
            )  # (1, hidden_size) = (1, 7168)

            for expert_id in range(cfg["n_routed_experts"]):
                cur_expert_gate_proj_weight = get_tensor(
                    f"model.layers.{layer_id}.mlp.experts.{expert_id}.gate_proj.weight"
                )  # (moe_intermediate_size, hidden_size) = (2048, 7168)
                cur_expert_up_proj_weight = get_tensor(
                    f"model.layers.{layer_id}.mlp.experts.{expert_id}.up_proj.weight"
                )  # (moe_intermediate_size, hidden_size) = (2048, 7168)
                cur_ffn1_weight = torch.cat(
                    [cur_expert_gate_proj_weight, cur_expert_up_proj_weight], dim=0
                )
                experts_ffn1_proj.append(cur_ffn1_weight)

                cur_expert_gate_proj_w_scale = get_tensor(
                    f"model.layers.{layer_id}.mlp.experts.{expert_id}.gate_proj.w_scale"
                )  # (moe_intermediate_size, 1) = (2048, 1)
                cur_expert_up_proj_w_scale = get_tensor(
                    f"model.layers.{layer_id}.mlp.experts.{expert_id}.up_proj.w_scale"
                )
                cur_expert_ffn1_proj_w_scale = torch.cat(
                    [cur_expert_gate_proj_w_scale, cur_expert_up_proj_w_scale], dim=0
                )
                cur_expert_ffn1_proj_dequant_scale = (
                    cur_expert_ffn1_proj_w_scale
                ).view(1, -1)
                experts_ffn1_proj_dequant_scale.append(cur_expert_ffn1_proj_dequant_scale)

                cur_expert_down_proj_weight = get_tensor(
                    f"model.layers.{layer_id}.mlp.experts.{expert_id}.down_proj.weight"
                )  # (hidden_size, moe_intermediate_size) = (7168, 2048)
                experts_ffn2_proj.append(cur_expert_down_proj_weight)

                cur_expert_ffn2_proj_a_scale = get_tensor(
                    f"model.layers.{layer_id}.mlp.experts.{expert_id}.down_proj.a_scale"
                )
                cur_expert_ffn2_proj_smooth_factor = get_tensor(
                    f"model.layers.{layer_id}.mlp.experts.{expert_id}.down_proj.smooth_factor"
                )  # (1, moe_intermediate_size) = (1, 2048)
                cur_expert_ffn2_proj_w_scale = get_tensor(
                    f"model.layers.{layer_id}.mlp.experts.{expert_id}.down_proj.w_scale"
                )  # (hidden_size, 1) = (7168, 1)

                cur_experts_ffn2_proj_dequant_scale = (
                    cur_expert_ffn2_proj_w_scale
                ).view(1, -1)
                experts_ffn2_proj_dequant_scale.append(cur_experts_ffn2_proj_dequant_scale)

            experts_ffn1_proj = torch.stack(experts_ffn1_proj, dim=0)  # (num_experts, 7168, 4096)
            experts_ffn2_proj = torch.stack(experts_ffn2_proj, dim=0)  # (num_experts, 2048, 7168)
            experts_ffn1_proj_dequant_scale = torch.stack(
                experts_ffn1_proj_dequant_scale, dim=0
            )  # (num_experts, 1, 4096)

            experts_ffn2_proj_dequant_scale = torch.stack(
                experts_ffn2_proj_dequant_scale, dim=0
            )  # (num_experts, 1, 7168)

            for ep_rank, (
                w_eu,
                w_ed,
                w_eu_dequant_scale,
                w_ed_dequant_scale,
            ) in enumerate(
                zip(
                    experts_ffn1_proj.chunk(args.ep_size, 0),
                    experts_ffn2_proj.chunk(args.ep_size, 0),
                    experts_ffn1_proj_dequant_scale.chunk(args.ep_size, 0),
                    experts_ffn2_proj_dequant_scale.chunk(args.ep_size, 0),
                )
            ):
                mi_w_eu = w_eu.contiguous()
                mi_w_ed = w_ed.contiguous()
                mi_w_eu_dequant_scale = w_eu_dequant_scale.contiguous()
                mi_w_ed_dequant_scale = w_ed_dequant_scale.contiguous()
                for rank in range(args.world_size):
                    if rank % args.ep_size == ep_rank:
                        mi_weight[rank].update(
                            {
                                f"model.layers.{layer_id}.experts.ffn1_proj.weight": transpose1(mi_w_eu, 32),
                                f"model.layers.{layer_id}.experts.ffn2_proj.weight": transpose1(mi_w_ed, 32),
                                f"model.layers.{layer_id}.experts.ffn1_proj.dequant_scale": mi_w_eu_dequant_scale,
                                f"model.layers.{layer_id}.experts.ffn2_proj.dequant_scale": mi_w_ed_dequant_scale,
                            }
                        )

    # save converted weight
    def save_weight(rank: int):
        save_file(mi_weight[rank], output_dir / f"mi_{rank}.safetensors")

    with Timer("io_load"):
        for rank in tqdm(range(args.world_size)):
            save_weight(rank)

    print("weight convert [DONE]!")
    total_main = time.perf_counter() - start_main
    print("\n===== time cost =====")
    for name, t in timers.items():
        print(f"{name}: {t:.3f}s")
    print(f"total: {total_main:.3f}s")
    print("weight convert [DONE]!")


def setup_output_dir(model_path: str, save_path: str) -> Path:
    model_name = Path(model_path.rstrip("/")).name
    output_dir = Path(save_path) / model_name
    output_dir.mkdir(parents=True, exist_ok=True)

    for ext in ["*.json", "tokenizer.*"]:
        for src in Path(model_path).glob(ext):
            if src == Path(model_path) / "config.json":
                shutil.copy2(src, output_dir)
                with open(Path(output_dir) / "config.json", "r", encoding="utf-8") as f:
                    config = json.load(f)
                if "rope_scaling" in config:
                    rope_scaling = config.pop("rope_scaling")
                    config.update(rope_scaling)
                    with open(Path(output_dir) / "config.json", "w", encoding="utf-8") as f:
                        json.dump(config, f, indent=4, ensure_ascii=False)
                if "model_type" in config:
                    config["model_type"] = (
                        "deepseek_v3-w8a8"  # modify "model_type" to "deepseek_v3-w8a8"
                    )
                    with open(Path(output_dir) / "config.json", "w", encoding="utf-8") as f:
                        json.dump(config, f, indent=4, ensure_ascii=False)

            else:
                shutil.copy2(src, output_dir)

    return output_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="deepseek_v3 model weight convert")
    parser.add_argument("--model_name_or_path", type=str, required=True, help="model weight path")
    parser.add_argument(
        "--bformat",
        type=str,
        choices=["Nd", "Nz", "Zn"],
        default="Nd",
        help="Linear weight format, default:'Nd'",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default="./.cache",
        help="save path for model",
    )
    parser.add_argument("--world_size", type=int, default=1, help="total rank size")
    parser.add_argument("--tp_size", type=int, default=1, help="N-way tensor parallelism size")
    parser.add_argument("--ep_size", type=int, default=1, help="N-way pipeline parallelism size")
    parser.add_argument("--pp_size", type=int, default=1, help="N-way pipeline parallelism size")
    args = parser.parse_args()

    # check world_size, tp_size, ep_size, pp_size
    assert args.world_size % args.tp_size == 0
    assert args.world_size % args.ep_size == 0
    assert args.world_size % args.pp_size == 0
    assert args.pp_size == 1, "Only support pp_size==1"

    output_dir = setup_output_dir(args.model_name_or_path, args.save_path)

    main(args)
