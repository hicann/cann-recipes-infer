# coding=utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Adapted from HuggingFace Transformers v5.5.0 Gemma4 (Apache-2.0).
# Copyright 2026 the HuggingFace Team. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Gemma-4-E2B packed offline decoder with framework-owned shared paged KV."""
from typing import Iterable, Tuple
import torch
from torch import nn
from torch.nn import functional as F
import torch_npu
from executor.core.kv_cache.cache_info import CacheEntry, LayerCacheInfo, ModelCacheInfo
from executor.utils.forward_metadata import get_forward_metadata


class Gemma4RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6, with_scale=True):
        super().__init__()
        self.eps = eps
        self.with_scale = with_scale
        if with_scale:
            self.weight = nn.Parameter(torch.ones(dim))
        else:
            # Keep the unit scale on the same device/dtype as the input without
            # adding it to the checkpoint contract. This matches the mainline
            # Gemma-4 implementation used by the repository.
            self.register_buffer("weight", torch.ones(dim), persistent=False)

    def forward(self, x):
        # Keep CPU contract tests and lightweight construction checks runnable;
        # the production path on NPU uses the mainline fused operator.
        if x.device.type != "npu":
            u = x.float()
            y = u * torch.rsqrt(u.square().mean(-1, keepdim=True) + self.eps)
            return (y * self.weight.float()).to(x.dtype)
        return torch_npu.npu_rms_norm(x, self.weight, self.eps)[0]


class CheckpointLinear(nn.Linear):
    """Allocate parameters once; strict checkpoint loading initializes their values."""

    def reset_parameters(self):
        # Avoid generating billions of random values immediately overwritten by the checkpoint.
        pass


class ScaledEmbedding(nn.Embedding):
    def __init__(self, vocab, dim, padding_idx, scale):
        super().__init__(vocab, dim, padding_idx)
        self.register_buffer("embed_scale", torch.tensor(scale), persistent=False)

    def reset_parameters(self):
        # The language checkpoint supplies the complete embedding, including the padding row.
        pass

    def forward(self, ids):
        return super().forward(ids) * self.embed_scale.to(self.weight.dtype)


class Gemma4RotaryEmbedding(nn.Module):
    """Dual RoPE: sliding (full head_dim) and full (partial_rotary_factor)."""

    def __init__(self, config, device=None, max_position_embeddings=4096):
        super().__init__()
        self.config = config
        self.max_position_embeddings = max_position_embeddings

        sliding_params = config.rope_parameters["sliding_attention"]
        sliding_theta = sliding_params["rope_theta"]
        sliding_dim = config.head_dim
        inv_freq_sliding = 1.0 / (
            sliding_theta
            ** (
                torch.arange(0, sliding_dim, 2, dtype=torch.float32, device=device)
                / sliding_dim
            )
        )

        full_params = config.rope_parameters["full_attention"]
        full_theta = full_params["rope_theta"]
        partial_rotary_factor = full_params.get("partial_rotary_factor", 0.25)
        full_dim = config.global_head_dim
        rotary_dim = int(full_dim * partial_rotary_factor)
        self.rotary_dim_full = rotary_dim
        # Gemma4 full attention uses "proportional" RoPE.  The number of
        # rotated channels is controlled by partial_rotary_factor, but the
        # frequency exponent is normalized by the complete global head
        # dimension (matching transformers' proportional RoPE definition).
        full_factor = full_params.get("factor", 1.0)
        inv_freq_full = (
            1.0
            / (
                full_theta
                ** (
                    torch.arange(0, rotary_dim, 2, dtype=torch.float32, device=device)
                    / full_dim
                )
            )
            / full_factor
        )
        # Precompute cos/sin to avoid .item() in graph mode.
        cos_sliding, sin_sliding = self._compute_cos_sin(
            max_position_embeddings, inv_freq_sliding, device, torch.bfloat16
        )
        cos_full, sin_full = self._compute_cos_sin(
            max_position_embeddings, inv_freq_full, device, torch.bfloat16
        )
        self.register_buffer("cos_sliding", cos_sliding, persistent=False)
        self.register_buffer("sin_sliding", sin_sliding, persistent=False)
        self.register_buffer("cos_full", cos_full, persistent=False)
        self.register_buffer("sin_full", sin_full, persistent=False)

    @staticmethod
    def _compute_cos_sin(seq_len, inv_freq, device, dtype, scaling=1.0):
        # Use the same batched matmul as Transformers.  `outer` is
        # mathematically identical, but on A3 it can round a frequency product
        # differently before sin/cos on NPU (observed at position 53, channel 43).
        positions = torch.arange(seq_len, device=device, dtype=torch.float32)
        inv = inv_freq.to(device) if device else inv_freq
        freqs = (
            inv[None, :, None].float()
            @ positions[None, None, :].float()
        ).transpose(1, 2).squeeze(0)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = (emb.cos() * scaling).to(dtype)
        sin = (emb.sin() * scaling).to(dtype)
        return cos, sin

    def forward(self, position_ids, layer_type="sliding_attention"):
        """Returns (cos, sin) shaped [T, 1, D_rope] for TND broadcast over heads."""
        if layer_type == "sliding_attention":
            cos, sin = self.cos_sliding, self.sin_sliding
        else:
            cos, sin = self.cos_full, self.sin_full

        cos_out = cos.index_select(0, position_ids).unsqueeze(1)
        sin_out = sin.index_select(0, position_ids).unsqueeze(1)
        return cos_out, sin_out


def apply_rope(x, cos_sin, full=False):
    cos, sin = cos_sin
    if full:
        pair = cos.shape[-1] // 2
        half = x.shape[-1] // 2
        part = torch.cat((x[..., :pair], x[..., half:half + pair]), dim=-1)
    else:
        part = x
    a, b = part.chunk(2, dim=-1)
    rotated = part * cos + torch.cat((-b, a), dim=-1) * sin
    if full:
        return torch.cat(
            (
                rotated[..., :pair],
                x[..., pair:half],
                rotated[..., pair:],
                x[..., half + pair:],
            ),
            dim=-1,
        )
    return rotated


class Gemma4E2BMLP(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        shared = layer_idx >= config.num_hidden_layers - config.num_kv_shared_layers
        width = config.intermediate_size * (
            2 if shared and config.use_double_wide_mlp else 1
        )
        self.gate_proj = CheckpointLinear(config.hidden_size, width, bias=False)
        self.up_proj = CheckpointLinear(config.hidden_size, width, bias=False)
        self.down_proj = CheckpointLinear(width, config.hidden_size, bias=False)

    def forward(self, x):
        return self.down_proj(
            F.gelu(self.gate_proj(x), approximate="tanh") * self.up_proj(x)
        )


class Gemma4E2BAttention(nn.Module):
    # Explicit BF16 matmul + FP32 softmax matches the Transformers reference for
    # short prompts. Keep the quadratic workspace bounded; production-length
    # Prefill continues to use the fused attention operator.
    PRECISE_PREFILL_MAX_TOKENS = 64

    def __init__(self, config, infer_config, layer_idx):
        super().__init__()
        self.layer_idx = layer_idx
        self.source_idx = config.kv_source_indices[layer_idx]
        self.is_source = self.source_idx == layer_idx
        self.is_sliding = config.layer_types[layer_idx] == "sliding_attention"
        self.attn_type = "SlidingWindow" if self.is_sliding else "FullAttention"
        self.head_dim = config.head_dim if self.is_sliding else config.global_head_dim
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.sliding_window = config.sliding_window
        self.block_size = infer_config.scheduler_config.block_size
        self.exe_mode = infer_config.model_config.exe_mode
        h, d = config.hidden_size, self.head_dim
        self.q_proj = CheckpointLinear(
            h, self.num_heads * d, bias=config.attention_bias
        )
        self.k_proj = CheckpointLinear(
            h, self.num_kv_heads * d, bias=config.attention_bias
        )
        self.v_proj = CheckpointLinear(
            h, self.num_kv_heads * d, bias=config.attention_bias
        )
        self.o_proj = CheckpointLinear(
            self.num_heads * d, h, bias=config.attention_bias
        )
        self.q_norm = Gemma4RMSNorm(d, config.rms_norm_eps)
        self.k_norm = Gemma4RMSNorm(d, config.rms_norm_eps)
        self.v_norm = Gemma4RMSNorm(d, config.rms_norm_eps, with_scale=False)
        self.k_cache = torch.empty(0)
        self.v_cache = torch.empty(0)
        self._consumers = []
        self.cache_entries = [
            CacheEntry(
                cache_name=name,
                attn_type=self.attn_type,
                dim=d,
                num_head=self.num_kv_heads,
                dtype=torch.bfloat16,
                needs_block=self.is_source,
                block_size=self.block_size,
                sliding_window=self.sliding_window if self.is_sliding else None,
                tensor_setter=lambda tensor, name=name: self.bind_cache(name, tensor),
            )
            for name in ("k_cache", "v_cache")
        ]

    def bind_cache(self, name, tensor):
        setattr(self, name, tensor)
        for entry in self.cache_entries:
            if entry.cache_name == name:
                entry.tensor = tensor
        for consumer in self._consumers:
            consumer.bind_cache(name, tensor)

    def forward(self, x, cos_sin, meta, source_kv=None):
        t = x.shape[0]
        q = apply_rope(
            self.q_norm(self.q_proj(x).view(t, self.num_heads, self.head_dim)),
            cos_sin,
            full=not self.is_sliding,
        )
        if self.is_source:
            k = apply_rope(
                self.k_norm(self.k_proj(x).view(t, self.num_kv_heads, self.head_dim)),
                cos_sin,
                full=not self.is_sliding,
            )
            v = self.v_norm(self.v_proj(x).view(t, self.num_kv_heads, self.head_dim))
        else:
            if source_kv is None:
                raise RuntimeError("shared attention requires its source layer K/V")
            k, v = source_kv
        if not self.k_cache.numel() or not self.v_cache.numel():
            raise RuntimeError(
                "framework must bind physical source caches before forward"
            )
        if self.is_source:
            slots = meta.slot_mapping[self.attn_type].view(-1, 1)
            torch_npu.npu_scatter_nd_update_(
                self.k_cache.view(-1, self.num_kv_heads, self.head_dim), slots, k
            )
            torch_npu.npu_scatter_nd_update_(
                self.v_cache.view(-1, self.num_kv_heads, self.head_dim), slots, v
            )
        out = self.attend(q, k, v, meta)
        return self.o_proj(out.reshape(t, self.num_heads * self.head_dim)), (k, v)

    def attend(self, q, k, v, meta):
        prefill = meta.is_prefill
        if prefill and q.shape[0] <= self.PRECISE_PREFILL_MAX_TOKENS:
            q = q.transpose(0, 1).unsqueeze(0)
            k = k.transpose(0, 1).unsqueeze(0)
            v = v.transpose(0, 1).unsqueeze(0)
            groups = self.num_heads // self.num_kv_heads
            k = k.repeat_interleave(groups, dim=1)
            v = v.repeat_interleave(groups, dim=1)
            scores = torch.matmul(q, k.transpose(2, 3))
            tokens = scores.shape[-1]
            causal_mask = torch.triu(
                torch.ones(tokens, tokens, device=q.device, dtype=torch.bool),
                diagonal=1,
            )
            if self.is_sliding:
                # Include the current token and at most window - 1 history tokens.
                causal_mask = causal_mask | torch.tril(
                    torch.ones_like(causal_mask), diagonal=-self.sliding_window
                )
            scores = scores.masked_fill(causal_mask, torch.finfo(scores.dtype).min)
            probabilities = F.softmax(scores, dim=-1, dtype=torch.float32).to(q.dtype)
            return (
                torch.matmul(probabilities, v)
                .squeeze(0)
                .transpose(0, 1)
                .contiguous()
            )
        graph = self.exe_mode == "npugraph_ex" and not prefill
        qlens = meta.actual_seq_lengths_list_q if graph else meta.actual_seq_lengths_q
        kvlens = (
            meta.actual_seq_lengths_list_kv if graph else meta.actual_seq_lengths_kv
        )
        args = dict(
            num_query_heads=self.num_heads,
            num_key_value_heads=self.num_kv_heads,
            softmax_scale=1.0,
            next_tokens=0,
        )
        if self.is_sliding:
            args.update(
                input_layout="TND",
                sparse_mode=4,
                pre_tokens=self.sliding_window - 1,
                atten_mask=meta.attention_mask,
                actual_seq_qlen=(
                    meta.actual_seq_lengths_cu_list_q
                    if graph
                    else meta.actual_seq_lengths_cu_q
                ),
                actual_seq_kvlen=(meta.actual_seq_lengths_cu_q if prefill else kvlens),
            )
        elif prefill:
            q = q.transpose(0, 1).unsqueeze(0).contiguous()
            k = k.transpose(0, 1).unsqueeze(0).contiguous()
            v = v.transpose(0, 1).unsqueeze(0).contiguous()
            args.update(
                input_layout="BNSD",
                sparse_mode=3,
                pre_tokens=2147483647,
                atten_mask=meta.attention_mask,
                actual_seq_qlen=qlens,
                actual_seq_kvlen=qlens,
            )
        else:
            q = q.reshape(1, 1, -1)
            args.update(
                input_layout="BSH",
                sparse_mode=0,
                pre_tokens=2147483647,
                actual_seq_qlen=qlens,
                actual_seq_kvlen=kvlens,
            )
        if not prefill:
            k = self.k_cache.view(*self.k_cache.shape[:2], -1)
            v = self.v_cache.view(*self.v_cache.shape[:2], -1)
            args.update(
                block_table=meta.block_table[self.attn_type], block_size=self.block_size
            )
        out, _ = torch.ops.npu.npu_fused_infer_attention_score_v2(q, k, v, **args)
        if not self.is_sliding and prefill:
            out = out.squeeze(0).transpose(0, 1)
        return out


class Gemma4E2BDecoderLayer(nn.Module):
    def __init__(self, config, infer_config, idx):
        super().__init__()
        self.self_attn = Gemma4E2BAttention(config, infer_config, idx)
        self.mlp = Gemma4E2BMLP(config, idx)
        for name in (
            "input_layernorm",
            "post_attention_layernorm",
            "pre_feedforward_layernorm",
            "post_feedforward_layernorm",
            "post_per_layer_input_norm",
        ):
            setattr(self, name, Gemma4RMSNorm(config.hidden_size, config.rms_norm_eps))
        self.per_layer_input_gate = CheckpointLinear(
            config.hidden_size, config.hidden_size_per_layer_input, bias=False
        )
        self.per_layer_projection = CheckpointLinear(
            config.hidden_size_per_layer_input, config.hidden_size, bias=False
        )
        self.register_buffer("layer_scalar", torch.ones(1))

    def forward(self, x, layer_input, cos_sin, meta, source_kv=None):
        attn, kv = self.self_attn(self.input_layernorm(x), cos_sin, meta, source_kv)
        x = x + self.post_attention_layernorm(attn)
        x = x + self.post_feedforward_layernorm(
            self.mlp(self.pre_feedforward_layernorm(x))
        )
        z = F.gelu(self.per_layer_input_gate(x), approximate="tanh") * layer_input
        x = x + self.post_per_layer_input_norm(self.per_layer_projection(z))
        return x * self.layer_scalar, kv


class Gemma4E2BTextModel(nn.Module):
    def __init__(self, config, infer_config):
        super().__init__()
        self.config = config
        h, d, layers = (
            config.hidden_size,
            config.hidden_size_per_layer_input,
            config.num_hidden_layers,
        )
        self.embed_tokens = ScaledEmbedding(
            config.vocab_size, h, config.pad_token_id, h**0.5
        )
        self.embed_tokens_per_layer = ScaledEmbedding(
            config.vocab_size_per_layer_input, layers * d, config.pad_token_id, d**0.5
        )
        self.per_layer_model_projection = CheckpointLinear(h, layers * d, bias=False)
        self.per_layer_projection_norm = Gemma4RMSNorm(d, config.rms_norm_eps)
        self.layers = nn.ModuleList(
            [Gemma4E2BDecoderLayer(config, infer_config, i) for i in range(layers)]
        )
        self.norm = Gemma4RMSNorm(h, config.rms_norm_eps)
        limit = (
            infer_config.data_config.input_truncated_len
            + infer_config.scheduler_config.max_new_tokens
        )
        if limit > config.max_position_embeddings:
            raise ValueError("configured total context exceeds max_position_embeddings")
        self.rotary_emb = Gemma4RotaryEmbedding(config, max_position_embeddings=limit)
        for i, layer in enumerate(self.layers):
            source = config.kv_source_indices[i]
            if source != i:
                self.layers[source].self_attn._consumers.append(layer.self_attn)

    def per_layer_inputs(self, ids, embeds):
        c = self.config
        shape = (ids.shape[0], c.num_hidden_layers, c.hidden_size_per_layer_input)
        dedicated = self.embed_tokens_per_layer(ids).reshape(shape)
        projected = (
            self.per_layer_model_projection(embeds) * c.hidden_size**-0.5
        ).reshape(shape)
        return (self.per_layer_projection_norm(projected) + dedicated) * 2**-0.5

    def forward(self, ids, positions, meta):
        x = self.embed_tokens(ids)
        layer_inputs = self.per_layer_inputs(ids, x)
        cos_sin = {
            kind: self.rotary_emb(positions, kind)
            for kind in set(self.config.layer_types)
        }
        shared = {}
        needed_sources = set(
            self.config.kv_source_indices[
                self.config.num_hidden_layers - self.config.num_kv_shared_layers:
            ]
        )
        for i, layer in enumerate(self.layers):
            source = self.config.kv_source_indices[i]
            x, kv = layer(
                x,
                layer_inputs[:, i, :],
                cos_sin[self.config.layer_types[i]],
                meta,
                shared.get(source),
            )
            if source == i and i in needed_sources:
                shared[i] = kv
        return self.norm(x)


class Gemma4E2BForCausalLM(nn.Module):
    def __init__(self, config, infer_config, comm_manager=None, prefix=""):
        super().__init__()
        config.validate_e2b()
        self.config = config
        self.infer_config = infer_config
        if getattr(infer_config.model_config, "dtype", "bfloat16") != "bfloat16":
            raise ValueError("Gemma-4-E2B supports BF16 only")
        if getattr(infer_config.model_config, "next_n", 0):
            raise ValueError("Gemma-4-E2B does not support speculative decoding")
        disagg = getattr(infer_config, "disagg_config", None)
        if disagg is not None and getattr(disagg, "disaggregation_mode", None) in (
            "PREFILL",
            "DECODE",
        ):
            raise ValueError("Gemma-4-E2B supports Offline only")
        if getattr(config, "quantization_config", None):
            raise ValueError("Gemma-4-E2B does not support quantized checkpoints")
        pc = infer_config.parallel_config
        if any(
            getattr(pc, n, 1) != 1
            for n in (
                "world_size",
                "attn_tp_size",
                "attn_dp_size",
                "embed_tp_size",
                "lmhead_tp_size",
                "dense_tp_size",
                "moe_tp_size",
                "moe_ep_size",
            )
        ):
            raise ValueError("Gemma-4-E2B supports one logical device, 1TP only")
        if infer_config.scheduler_config.batch_size != 1:
            raise ValueError("Gemma-4-E2B supports batch=1 only")
        if infer_config.model_config.exe_mode not in ("eager", "npugraph_ex"):
            raise ValueError("Gemma-4-E2B supports eager and npugraph_ex only")
        if comm_manager is not None:
            for name in ("attn_tp_group", "embed_tp_group", "lmhead_tp_group"):
                comm_manager.register_group(name=name, group_num=1, group_size=1)
        lengths = {
            "input_truncated_len": infer_config.data_config.input_truncated_len,
            "max_new_tokens": infer_config.scheduler_config.max_new_tokens,
            "block_size": infer_config.scheduler_config.block_size,
        }
        if any(type(value) is not int or value <= 0 for value in lengths.values()):
            raise ValueError(
                "input_truncated_len, max_new_tokens and block_size must be positive integers"
            )
        if (
            lengths["input_truncated_len"] + lengths["max_new_tokens"]
            > config.max_position_embeddings
        ):
            raise ValueError("configured total context exceeds max_position_embeddings")
        self.model = Gemma4E2BTextModel(config, infer_config)
        self.lm_head = CheckpointLinear(
            config.hidden_size, config.vocab_size, bias=False
        )
        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight
        self.weight_audit = {}
        self.weights_ready = False

    def forward(self, input_ids, position_ids=None, forward_metadata=None, **kwargs):
        if not self.weights_ready:
            raise RuntimeError(
                "a complete language checkpoint must be loaded before inference"
            )
        meta = (
            forward_metadata if forward_metadata is not None else get_forward_metadata()
        )
        if position_ids is None or meta is None:
            raise ValueError("packed positions and framework metadata are required")
        if meta.actual_seq_lengths_cu_q.shape[0] != 1:
            raise ValueError("Gemma-4-E2B requires exactly one request")
        x = self.model(input_ids, position_ids, meta)
        if meta.is_prefill:
            x = torch.index_select(x, 0, meta.actual_seq_lengths_cu_q - 1)
        logits = self.lm_head(x)
        cap = self.config.final_logit_softcapping
        if cap is not None:
            logits = torch.tanh(logits / cap) * cap
        return logits.unsqueeze(1).float()

    def get_cache_info(self):
        layers = [
            LayerCacheInfo(layer_idx=i, caches=list(layer.self_attn.cache_entries))
            for i, layer in enumerate(self.model.layers)
        ]
        return ModelCacheInfo(num_layers=len(layers), layer_infos=layers)

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]) -> set:
        self.weights_ready = False
        params = dict(self.named_parameters())
        targets = {**params, **dict(self.named_buffers())}
        required = set(params) | {n for n in targets if n.endswith("layer_scalar")}
        loaded, skipped = set(), []
        prefixes = (
            "model.vision_tower.",
            "model.audio_tower.",
            "model.embed_vision.",
            "model.embed_audio.",
            "model.multimodal_projector.",
        )
        with torch.no_grad():
            for original, weight in weights:
                if original.startswith(prefixes):
                    skipped.append(original)
                    continue
                name = original
                for prefix in ("model.language_model.", "language_model.model."):
                    if name.startswith(prefix):
                        name = "model." + name[len(prefix):]
                        break
                if name == "lm_head.weight" and self.config.tie_word_embeddings:
                    name = "model.embed_tokens.weight"
                if name not in targets:
                    raise ValueError(f"unmatched language weight: {original} -> {name}")
                if targets[name].shape != weight.shape:
                    raise ValueError(
                        f"weight shape mismatch: {original}: {weight.shape} != {targets[name].shape}"
                    )
                if name in loaded:
                    # Explicitly verify a separately stored tied head instead of replacing it silently.
                    if not torch.equal(
                        targets[name].detach().cpu(), weight.detach().cpu()
                    ):
                        raise ValueError(
                            f"conflicting duplicate/tied weight: {original}"
                        )
                else:
                    targets[name].copy_(weight)
                loaded.add(name)
        missing = sorted(required - loaded)
        self.weight_audit = {
            "loaded": sorted(loaded),
            "skipped_multimodal": skipped,
            "missing": missing,
        }
        if missing:
            raise ValueError(f"missing required language weights/buffers: {missing}")
        self.weights_ready = True
        return loaded
