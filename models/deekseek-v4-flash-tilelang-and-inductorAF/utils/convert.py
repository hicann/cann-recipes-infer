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

import os
import shutil
from argparse import ArgumentParser
from glob import glob
from tqdm import tqdm, trange

import torch
from safetensors.torch import safe_open, save_file


# TP size, default TP size is 16, attn TP size is 4
attn_tp_size = 4

# In DeepSeek Model, use tensor parallel(TP) as MP
mapping = {
    "embed_tokens": ("embed", 0),
    "input_layernorm": ("attn_norm", None),
    "post_attention_layernorm": ("ffn_norm", None),
    "q_proj": ("wq", 0),
    "q_a_proj": ("wq_a", None),
    "q_a_layernorm": ("q_norm", None),
    "q_b_proj": ("wq_b", 0),
    "kv_a_proj_with_mqa": ("wkv_a", None),
    "kv_a_layernorm": ("kv_norm", None),
    "kv_b_proj": ("wkv_b", 0),
    "o_proj": ("wo", 1),
    "gate_proj": ("w1", 0),
    "down_proj": ("w2", 1),
    "up_proj": ("w3", 0),
    "lm_head": ("head", 0),
    "embed": ("embed", 0),
    "wq_b": ("wq_b", 0),
    "wo_a": ("wo_a", 0),
    "wo_b": ("wo_b", 1),
    "head": ("head", 0),
    "attn_sink": ("attn_sink", 0),
    "weights_proj": ("weights_proj", 0),
}

mp4_list = ["wq_b", "wo_a", "wo_b", "weights_proj", "attn_sink"]  # mp4


def main(hf_ckpt_path, save_path, n_experts, mp_):
    """
    Converts and saves model checkpoint files into a specified format.

    Args:
        hf_ckpt_path (str): Path to the directory containing the input checkpoint files.
        save_path (str): Path to the directory where the converted checkpoint files will be saved.
        n_experts (int): Total number of experts in the model.
        mp (int): Model parallelism factor.

    Returns:
        None
    """
    mp_out = mp_
    torch.set_num_threads(8)
    n_local_experts = n_experts // mp_out
    state_dicts = [{} for _ in range(mp_out)]

    for file_path in tqdm(glob(os.path.join(hf_ckpt_path, "*.safetensors"))):
        with safe_open(file_path, framework="pt", device="cpu") as f:
            for name in f.keys():
                if "model.layers.61" in name or "mtp" in name:
                    continue
                param: torch.Tensor = f.get_tensor(name)
                if name.startswith("model."):
                    name = name[len("model."):]
                name = name.replace("self_attn", "attn")
                name = name.replace("mlp", "ffn")
                name = name.replace("weight_scale_inv", "scale")
                name = name.replace("e_score_correction_bias", "bias")
                if any(x in name for x in ["hc", "attn_sink", "tie2eid", "ape"]):
                    key = name.split(".")[-1]
                else:
                    key = name.split(".")[-2]
                if key in mapping:
                    new_key, dim = mapping[key]
                else:
                    new_key, dim = key, None
                name = name.replace(key, new_key)
                for i_out in range(mp_out):
                    new_param = param
                    i = i_out
                    mp = mp_out
                    for mp8_item in mp4_list:
                        if mp8_item in name:
                            mp = attn_tp_size
                            i = i_out // attn_tp_size
                    if "experts" in name and "shared_experts" not in name:
                        idx = int(name.split(".")[-3])
                        if (
                            idx < i * n_local_experts
                            or idx >= (i + 1) * n_local_experts
                        ):
                            continue
                    elif dim is not None:
                        if param.size(dim) % mp != 0:
                            raise ValueError(
                                f"Dimension {dim} of parameter {name} with \
                                size {param.size()} is not divisible by {mp}"
                            )
                        shard_size = param.size(dim) // mp
                        new_param = param.narrow(
                            dim, i * shard_size, shard_size
                        ).contiguous()
                    state_dicts[i_out][name] = new_param

    os.makedirs(save_path, exist_ok=True)

    for i in trange(mp_out):
        save_file(
            state_dicts[i], os.path.join(save_path, f"model{i}-mp{mp_out}.safetensors")
        )

    for file_path in glob(os.path.join(hf_ckpt_path, "*token*")):
        new_file_path = os.path.join(save_path, os.path.basename(file_path))
        shutil.copyfile(file_path, new_file_path)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--hf-ckpt-path", type=str, required=True)
    parser.add_argument("--save-path", type=str, required=True)
    parser.add_argument("--n-experts", type=int, required=True)
    parser.add_argument("--model-parallel", type=int, required=True)
    args = parser.parse_args()
    if args.n_experts % args.model_parallel != 0:
        raise ValueError(
            "Number of experts must be divisible by model parallelism"
        )
    main(args.hf_ckpt_path, args.save_path, args.n_experts, args.model_parallel)
