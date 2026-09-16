# coding=utf-8
# GLM-5.3-Flash weight converter: FP8 (e4m3, 128x128 block) -> Hybrid
# HiF8-MXFP8-MXFP4.
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


import json
import os
import shutil
from argparse import ArgumentParser
from glob import glob

import torch
import torch_npu
from safetensors import safe_open
from safetensors.torch import load_file, save_file
from tqdm import tqdm

from mx_quantize import quantize_mx, pack_uint4, f32_to_f4_unpacked

FP8_BLOCK = 128        # source checkpoint weight_block_size
MX_BLOCK_K = 32        # MX shared-exponent block along the input dim
MX_PACK_FACTOR = 2     # two e2m1 values per uint8

NUM_BITS_4 = 4
NUM_BITS_8 = 8

# Attention projections that the source checkpoint quantizes (DSA layers only).
DSA_QUANT_PROJ = ("q_a_proj", "q_b_proj", "kv_a_proj_with_mqa", "o_proj")
MLP_PROJ = ("gate_proj", "up_proj", "down_proj")


# ============================================================================
# FP8 source dequant
# ============================================================================

def fp8_block_dequant(weight: torch.Tensor, scale_inv: torch.Tensor,
                      block_size: int = FP8_BLOCK) -> torch.Tensor:
    """Dequantize a 128x128 block-quantized FP8 tensor to the default dtype.

    w[i, j] = q[i, j] * scale_inv[i // block, j // block]; the scale grid is
    expanded by repeat_interleave and trimmed, so shapes that are not a whole
    multiple of `block_size` work unchanged.
    """
    if weight.dim() != 2:
        raise ValueError(f"expected a 2D weight, got shape {tuple(weight.shape)}")
    out_dim, in_dim = weight.shape
    scale_m, scale_n = scale_inv.shape
    if scale_m != (out_dim + block_size - 1) // block_size or \
            scale_n != (in_dim + block_size - 1) // block_size:
        raise ValueError(
            f"scale_inv {tuple(scale_inv.shape)} does not match weight "
            f"{tuple(weight.shape)} at block_size={block_size}")

    scale = scale_inv.to(torch.float32)
    scale = scale.repeat_interleave(block_size, dim=0)[:out_dim]
    scale = scale.repeat_interleave(block_size, dim=1)[:, :in_dim]
    return (weight.to(torch.float32) * scale).to(torch.get_default_dtype())


# ============================================================================
# Quantizer: MXFP4 (e2m1 elements, E8M0 shared exponent per 32 along K)
# ============================================================================

def quantize_mxfp4(weight: torch.Tensor, block_size: int = MX_BLOCK_K):
    """MXFP4 weight quantization (OCP MX: FP4 e2m1 + E8M0 block scale).

    Each group of `block_size` values along the input dim (K) shares one
    power-of-two scale; elements are then encoded as e2m1 and packed two per
    byte, which is the layout `MxFp4LinearMethod` / `W4A4MxFp4MoEGMMMethod`
    load (they apply the transpose / reshape_mx_scale themselves).

    Args:
        weight: 2D [N, K] BF16/FP32 weight (already FP8-dequantized).
        block_size: MX block along K (32 for OCP MXFP4; the NPU kernels
            hard-code 32, so anything else is rejected).

    Returns:
        (q_weight, scale)
          q_weight: uint8 [N, K // 2]     packed e2m1 nibbles (low = even index)
          scale:    uint8 [N, K // 32]    E8M0 biased exponent
    """
    if weight.dim() != 2:
        raise ValueError(f"mxfp4 quant expects a 2D weight, got {tuple(weight.shape)}")
    if block_size != MX_BLOCK_K:
        raise ValueError(f"the NPU MXFP4 kernels require block_size == {MX_BLOCK_K}, "
                         f"got {block_size}")
    in_dim = weight.shape[1]
    if in_dim % block_size != 0:
        raise ValueError(
            f"MXFP4 needs the input dim divisible by {block_size}, got K={in_dim}")
    if in_dim % MX_PACK_FACTOR != 0:
        raise ValueError(f"MXFP4 needs an even input dim for packing, got K={in_dim}")

    # quantize_mx returns e2m1-representable fp32 values plus the biased
    # (E8M0) shared exponent per block; real_quant keeps them separate.
    values, scale = quantize_mx(
        weight.to(torch.float32), NUM_BITS_4, block_size=block_size,
        axes=[-1], real_quant=True)
    q_weight = pack_uint4(f32_to_f4_unpacked(values.float()))
    return q_weight.contiguous(), scale.to(torch.uint8).contiguous()


# ============================================================================
# Quantizer: MXFP8 (e4m3 elements, E8M0 shared exponent per 32 along K)
# ============================================================================

def quantize_mxfp8(weight: torch.Tensor, block_size: int = MX_BLOCK_K):
    """MXFP8 weight quantization (OCP MX: FP8 e4m3 + E8M0 block scale).

    Same MX machinery as `quantize_mxfp4`, 8-bit elements instead of 4, and no
    nibble packing -- the layout `MxFp8LinearMethod` loads.

    Args:
        weight: 2D [N, K] BF16/FP32 weight (already FP8-dequantized).
        block_size: MX block along K (32 for OCP MX; the NPU kernels hard-code
            it, so anything else is rejected).

    Returns:
        (q_weight, scale)
          q_weight: float8_e4m3fn [N, K]
          scale:    uint8         [N, K // 32]   E8M0 biased exponent
    """
    if weight.dim() != 2:
        raise ValueError(f"mxfp8 quant expects a 2D weight, got {tuple(weight.shape)}")
    if block_size != MX_BLOCK_K:
        raise ValueError(f"the NPU MX kernels require block_size == {MX_BLOCK_K}, "
                         f"got {block_size}")
    in_dim = weight.shape[1]
    if in_dim % block_size != 0:
        raise ValueError(
            f"MXFP8 needs the input dim divisible by {block_size}, got K={in_dim}")

    values, scale = quantize_mx(
        weight.to(torch.float32), NUM_BITS_8, block_size=block_size,
        axes=[-1], real_quant=True)
    return values.contiguous(), scale.to(torch.uint8).contiguous()


# ============================================================================
# Quantizer: HiF8 (W8A8, per-output-channel, tapered-precision float)
# ============================================================================

def quantize_hif8(weight: torch.Tensor):
    """Per-output-channel HiF8 weight quantization via npu_dynamic_quant.

    The same call models/deepseek_v4/utils/convert_model.py:hif8_weight_quant
    and models/pangu_7b make, so the output matches those recipes byte for
    byte. The operator owns the scale (rowMax(abs(x)) / DTYPE_MAX) and needs a
    SoC where it supports hifloat8 -- 910B/A2 rejects it.

    Args:
        weight: 2D [N, K] BF16/FP32 weight (already FP8-dequantized).

    Returns:
        (q_weight, scale)
          q_weight: uint8   [N, K]   raw HiF8 bytes
          scale:    float32 [N, 1]   such that  w ~= hif8_decode(q) * scale
    """
    if weight.dim() != 2:
        raise ValueError(f"hif8 quant expects a 2D weight, got {tuple(weight.shape)}")

    q, scale = torch_npu.npu_dynamic_quant(
        weight.npu(), dst_type=torch_npu.hifloat8)
    q = q.cpu()
    # the op hands back a hifloat8-typed tensor; store the raw bytes
    if q.dtype != torch.uint8:
        if q.element_size() != 1:
            raise ValueError(f"unexpected hif8 weight dtype {q.dtype}")
        q = q.view(torch.uint8)
    return q.contiguous(), scale.cpu().unsqueeze(-1).to(torch.float32)


# ============================================================================
# GLM-5.3 module classification
# ============================================================================

class Glm53Layout:
    """Decides, from config.json, which tensors may be quantized and how."""

    def __init__(self, config: dict):
        text_config = config.get("text_config", config)
        self.num_hidden_layers = text_config["num_hidden_layers"]
        self.num_nextn_predict_layers = text_config.get("num_nextn_predict_layers", 0)
        self.num_layers = self.num_hidden_layers + self.num_nextn_predict_layers
        self.n_routed_experts = text_config["n_routed_experts"]
        self.first_k_dense_replace = text_config.get("first_k_dense_replace", 3)
        layer_types = text_config.get("layer_types")
        if layer_types is None:
            layer_types = ["linear_attention" if i % 4 != 3 else "deepseek_sparse_attention"
                           for i in range(self.num_hidden_layers)]
        self.layer_types = layer_types

    def is_kda_layer(self, layer_idx: int) -> bool:
        """The MTP layer (idx == num_hidden_layers) is a DSA layer."""
        if layer_idx >= self.num_hidden_layers:
            return False
        return self.layer_types[layer_idx] == "linear_attention"

    @staticmethod
    def _parse(name: str):
        """('model.language_model.layers.3.mlp.experts.0.gate_proj.weight')
        -> (3, 'mlp.experts.0.gate_proj')  |  (None, None) for non-layer names."""
        runtime = name
        if runtime.startswith("model.language_model."):
            runtime = "model." + runtime[len("model.language_model."):]
        if not runtime.startswith("model.layers."):
            return None, None, runtime
        rest = runtime[len("model.layers."):]
        idx_str, _, suffix = rest.partition(".")
        if not idx_str.isdigit():
            return None, None, runtime
        module = suffix.rsplit(".", 1)[0]  # strip the ".weight" / ".bias" leaf
        return int(idx_str), module, runtime

    def classify(self, tensor_name: str) -> str:
        """Return 'expert', 'shared_expert', 'linear' or 'skip' for a
        `*.weight` tensor.

        'expert'        -> routed MoE expert projection (FusedMoEGMM at runtime)
        'shared_expert' -> mlp.shared_experts.* (a *Linear* at runtime --
                           MergedColumnParallelLinear / RowParallelLinear, not
                           part of the GMM -- but it follows the routed experts'
                           scheme in hif8 mode, so it needs its own bucket)
        'linear'        -> any other quantizable nn.Linear projection
        'skip'          -> keep BF16
        """
        if not tensor_name.endswith(".weight"):
            return "skip"
        layer_idx, module, runtime = self._parse(tensor_name)
        if layer_idx is None:
            return "skip"  # embed_tokens / norm / lm_head / visual tower
        if layer_idx >= self.num_layers:
            return "skip"

        # routed experts: mlp.experts.{i}.{gate,up,down}_proj
        if module.startswith("mlp.experts."):
            leaf = module.rsplit(".", 1)[-1]
            return "expert" if leaf in MLP_PROJ else "skip"

        # shared expert + dense MLP
        if module.startswith("mlp.shared_experts."):
            leaf = module.rsplit(".", 1)[-1]
            return "shared_expert" if leaf in MLP_PROJ else "skip"
        if module.startswith("mlp."):
            leaf = module.rsplit(".", 1)[-1]
            # mlp.gate (router) stays BF16 — it is fp32 at runtime
            return "linear" if leaf in MLP_PROJ else "skip"

        # attention: only DSA layers, and only the four projections the source
        # checkpoint itself quantizes (kv_b_proj is absorbed; indexer is BF16)
        if module.startswith("self_attn."):
            if self.is_kda_layer(layer_idx):
                return "skip"
            leaf = module[len("self_attn."):]
            return "linear" if leaf in DSA_QUANT_PROJ else "skip"

        return "skip"

    def ignore_list(self) -> list:
        """Runtime module names that must stay unquantized, i.e. everything the
        recipe builds as a Linear/GMM but this converter left in BF16.

        These are RUNTIME names, not tensor names: the recipe fuses
        `mlp.gate_proj` + `mlp.up_proj` into one MergedColumnParallelLinear
        called `gate_up_proj`, so anything listed here has to use the fused
        spelling (compressed_tensors.py calls should_ignore_layer() without a
        fused_mapping, so nothing un-fuses it for us). Nothing fused ends up in
        this list today -- both dense MLP and shared experts are quantized --
        but that is the spelling to use if one ever does.
        """
        ignore = ["lm_head"]
        for i in range(self.num_layers):
            if self.is_kda_layer(i):
                for proj in ("q_proj", "k_proj", "v_proj", "o_proj",
                             "b_proj", "f_a_proj", "f_b_proj", "g_a_proj", "g_b_proj"):
                    ignore.append(f"model.layers.{i}.self_attn.{proj}")
            else:
                ignore.append(f"model.layers.{i}.self_attn.kv_b_proj")
                for proj in ("wq_b", "wk", "weights_proj"):
                    ignore.append(f"model.layers.{i}.self_attn.indexer.{proj}")
            if i >= self.first_k_dense_replace:
                ignore.append(f"model.layers.{i}.mlp.gate")
            if i >= self.num_hidden_layers:  # MTP extras
                ignore.append(f"model.layers.{i}.eh_proj")
                ignore.append(f"model.layers.{i}.shared_head.head")
        return ignore


# ============================================================================
# quantization_config emission
# ============================================================================

def _mxfp4_group(targets):
    """MXFP4 weights (FP4 e2m1 in groups of 32) + dynamic MXFP8 activations.

    Activations are declared MXFP8 (W4A8), matching the shipped Kimi-K3 MXFP4
    checkpoint (configuration_kimi_k3._MXFP8_DYNAMIC_ACTIVATIONS); the runtime
    resolves this to W4A8MxFp4MoEGMMMethod.

    NOTE: the *weights on disk are identical* for W4A8 and W4A4 — both runtime
    methods load the same uint8-packed [N, K/2] tensor plus its E8M0 scale and
    only differ in the in-memory transform they apply afterwards. Switching to
    W4A4 (W4A4MxFp4MoEGMMMethod) is a config.json edit — set this group's
    input_activations.num_bits to 4 — and needs no re-conversion.
    """
    return {
        "targets": targets,
        "weights": {"num_bits": NUM_BITS_4, "type": "float", "symmetric": True,
                    "strategy": "group", "group_size": MX_BLOCK_K, "dynamic": False,
                    "actorder": None, "block_structure": None,
                    "observer": "minmax", "observer_kwargs": {}},
        "input_activations": {"num_bits": NUM_BITS_8, "type": "float", "symmetric": True,
                              "strategy": "group", "group_size": MX_BLOCK_K,
                              "dynamic": True, "actorder": None,
                              "block_structure": None,
                              "observer": "minmax", "observer_kwargs": {}},
        "output_activations": None,
    }


def _mxfp8_group(targets):
    """MXFP8 weights (e4m3 in groups of 32) + dynamic MXFP8 activations (W8A8).

    What the routed experts' W4A8 group pairs with on the Linear side, and the
    same split models/deepseek_v4 and models/bailing_2_5 ship: 4-bit only for
    the routed experts, MX 8-bit for every other quantizable Linear. Resolved
    by is_dynamic_group_w8a8_mxfp8 -> MxFp8LinearMethod, whose create_weights
    asks for float8_e4m3fn [N, K] plus uint8 [N, ceil(K / 32)].
    """
    group = _mxfp4_group(targets)
    group["weights"]["num_bits"] = NUM_BITS_8
    group["input_activations"]["num_bits"] = NUM_BITS_8
    return group


def _hif8_group(targets):
    """HiF8 weights (per-output-channel) + dynamic per-tensor HiF8 activations
    (matches CompressedTensorsConfig.is_dynamic_token_w8a8_hifloat8, which
    insists on input strategy "tensor" -- the runtime casts activations to
    hifloat8 without scaling them, see CompressedTensorsW8A8Hif8LinearMethod).

    Linear only: there is no HiF8 FusedMoEGMM method in module/quantization,
    so routed experts can never carry this scheme.
    """
    return {
        "targets": targets,
        "weights": {"num_bits": NUM_BITS_8, "type": "float", "symmetric": True,
                    "strategy": "channel", "dynamic": False, "group_size": None,
                    "actorder": None, "block_structure": None,
                    "observer": "minmax", "observer_kwargs": {}},
        "input_activations": {"num_bits": NUM_BITS_8, "type": "float", "symmetric": True,
                              "strategy": "tensor", "dynamic": True, "group_size": None,
                              "actorder": None, "block_structure": None,
                              "observer": "memoryless", "observer_kwargs": {}},
        "output_activations": None,
    }


SHARED_EXPERT_TARGET = r"re:.*\.mlp\.shared_experts\..*"


def generate_quant_config(ignore: list) -> dict:
    """Build the `quantization_config` the runtime reads back.

    Target groups follow the framework's resolution order: FusedMoEGMM picks
    the "MoEGMM" group (get_moe_target), every nn.Linear falls through to the
    "Linear" group.

    Both groups are always emitted, even when one of them has every member in
    `ignore`: CompressedTensorsConfig.from_config() looks up "Linear" and
    get_moe_target() looks up "MoEGMM" unconditionally, so a config missing
    either one raises before a single weight is loaded.
    """
    #   HiF8  W8A8 -> attention + dense-MLP Linears
    #   MXFP8 W8A8 -> shared experts
    #   MXFP4 W4A8 -> routed experts (FusedMoEGMM)
    config_groups = {
        "group_0": _mxfp8_group([SHARED_EXPERT_TARGET]),
        "group_1": _hif8_group(["Linear"]),
        "group_2": _mxfp4_group(["MoEGMM"]),
    }
    return {
        "quant_method": "compressed-tensors",
        "format": "float-quantized",
        "quantization_status": "compressed",
        "global_compression_ratio": 1,
        "config_groups": config_groups,
        "ignore": ignore,
        "kv_cache_scheme": None,
    }


# ============================================================================
# conversion driver
# ============================================================================

def copy_aux_files(src, target):
    for root, _, files in os.walk(src):
        for file in files:
            if file.endswith((".py", ".json", ".jinja", ".txt", ".md")):
                if file == "model.safetensors.index.json":
                    continue
                src_path = os.path.join(root, file)
                rel_dir = os.path.relpath(root, src)
                dst_dir = os.path.join(target, rel_dir)
                os.makedirs(dst_dir, exist_ok=True)
                shutil.copy2(src_path, os.path.join(dst_dir, file))


def main(input_path, output_path, resume=False):
    torch.set_default_dtype(torch.bfloat16)
    os.makedirs(output_path, exist_ok=True)

    with open(os.path.join(input_path, "config.json"), "r") as f:
        config = json.load(f)
    index_file = os.path.join(input_path, "model.safetensors.index.json")
    with open(index_file, "r") as f:
        weight_map = json.load(f)["weight_map"]

    layout = Glm53Layout(config)
    config["quantization_config"] = generate_quant_config(layout.ignore_list())

    loaded_files = {}

    def get_tensor(name):
        file_name = weight_map[name]
        if file_name not in loaded_files:
            loaded_files[file_name] = load_file(
                os.path.join(input_path, file_name), device="cpu")
        return loaded_files[file_name][name]

    def quantize(kind, weight):
        """kind: 'expert' | 'shared_expert' | 'linear' -> (q_weight, scale).

        Routed experts MXFP4 W4A8, shared experts MXFP8 W8A8, rest HiF8."""
        if kind == "expert":
            return quantize_mxfp4(weight)
        if kind == "shared_expert":
            return quantize_mxfp8(weight)
        return quantize_hif8(weight)

    new_weight_map = {}
    stats = {"expert": 0, "shared_expert": 0, "linear": 0, "skip": 0}
    safetensor_files = sorted(glob(os.path.join(input_path, "*.safetensors")))
    if not safetensor_files:
        raise FileNotFoundError(f"no *.safetensors found under {input_path}")

    resumed = 0
    for safetensor_file in tqdm(safetensor_files, desc="convert->hif8"):
        file_name = os.path.basename(safetensor_file)

        # --resume: an output shard that is already there is taken as done.
        # Its tensor names still have to reach new_weight_map, or the index
        # written at the end would not cover them. Only the header is read.
        # A shard left truncated by the previous crash raises here (safe_open
        # checks the file length) -- delete it and rerun.
        done_file = os.path.join(output_path, file_name)
        if resume and os.path.isfile(done_file):
            with safe_open(done_file, framework="pt") as done:
                for key in done.keys():
                    new_weight_map[key] = file_name
                    if not key.endswith(".weight_scale"):
                        stats[layout.classify(key)] += 1
            resumed += 1
            continue

        current = load_file(safetensor_file, device="cpu")
        loaded_files[file_name] = current

        new_state_dict = {}
        for name, tensor in current.items():
            if name.endswith("_scale_inv"):
                continue  # consumed with its weight below

            # 1) FP8 -> BF16
            if tensor.dtype == torch.float8_e4m3fn:
                scale_inv_name = f"{name}_scale_inv"
                if scale_inv_name not in weight_map:
                    raise KeyError(f"missing {scale_inv_name} for FP8 tensor {name}")
                tensor = fp8_block_dequant(tensor, get_tensor(scale_inv_name))

            # 2) re-quantize the modules this recipe quantizes
            kind = layout.classify(name)
            result = quantize(kind, tensor) if kind != "skip" else None
            stats[kind if result is not None else "skip"] += 1

            if result is None:
                new_state_dict[name] = tensor
                new_weight_map[name] = file_name
            else:
                q_weight, scale = result
                scale_name = name[: -len(".weight")] + ".weight_scale"
                new_state_dict[name] = q_weight
                new_state_dict[scale_name] = scale
                new_weight_map[name] = file_name
                new_weight_map[scale_name] = file_name

        save_file(new_state_dict, os.path.join(output_path, file_name),
                  metadata={"format": "pt"})

        # keep only the 2 most recent shards resident
        while len(loaded_files) > 2:
            del loaded_files[next(iter(loaded_files))]

    copy_aux_files(input_path, output_path)
    with open(os.path.join(output_path, "model.safetensors.index.json"), "w") as f:
        json.dump({"metadata": {}, "weight_map": new_weight_map}, f, indent=2)
    with open(os.path.join(output_path, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    print(f"done: quantized {stats['expert']} expert + "
          f"{stats['shared_expert']} shared-expert + {stats['linear']} linear "
          f"tensors, kept {stats['skip']} in bf16 -> {output_path}"
          + (f" ({resumed}/{len(safetensor_files)} shards reused)" if resumed else ""))


if __name__ == "__main__":
    parser = ArgumentParser(
        description="GLM-5.3 FP8 -> Hybrid HiF8-MXFP8-MXFP4 converter")
    parser.add_argument("--input_hf_path", type=str, required=True)
    parser.add_argument("--output_hf_path", type=str, required=True)
    parser.add_argument("--resume", action="store_true",
                        help="reuse output shards from an interrupted run when "
                             "they are complete and hold the expected tensors")
    args = parser.parse_args()
    main(args.input_hf_path, args.output_hf_path, resume=args.resume)
