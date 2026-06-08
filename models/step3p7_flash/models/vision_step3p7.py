# coding=utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
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
"""Standalone Step-3.7-Flash vision tower for NPU."""

import math
from typing import Iterable, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

try:  # NPU is only available on the remote DevSpace; keep importable on CPU.
    import torch_npu  # noqa: F401
    _HAS_NPU = True
except Exception:  # pragma: no cover - local (no NPU) fallback
    _HAS_NPU = False


# ---------------------------------------------------------------------------
# Vision config (read straight from config.json vision_config; defaults mirror
# .original_ref/configuration_step3p7.py:StepRoboticsVisionEncoderConfig).
# ---------------------------------------------------------------------------
class VisionConfig:
    """Lightweight config holder for the vision tower.

    Field names and defaults are taken verbatim from
    ``.original_ref/configuration_step3p7.py``. ``rope_theta`` and the other
    rope_* kwargs are NOT present as explicit config fields there, so the
    original ``getattr(config, "rope_theta", 10000)`` defaults apply.
    """

    def __init__(
        self,
        width: int = 1536,
        layers: int = 47,
        heads: int = 16,
        num_channels: int = 3,
        image_size: int = 728,
        patch_size: int = 14,
        mlp_ratio: float = 8960 / 1536,
        layer_norm_eps: float = 1e-5,
        hidden_act: str = "quick_gelu",
        use_cls_token: bool = False,
        use_ln_pre: bool = True,
        use_ln_post: bool = False,
        use_abs_posemb: bool = True,
        use_rope2d: bool = True,
        ls_init_value: Optional[float] = 0.1,
        # rope_* are getattr-defaults in the HF vision_encoder.py:
        rope_theta: Union[int, float] = 10000,
        rope_max_freq: int = 10,
        rope_num_freqs: int = 1,
        rope_theta_rescale_factor: float = 1.0,
        # top-level multimodal config fields used by projector/_process glue:
        text_hidden_size: int = 4096,
        projector_bias: bool = False,
        **kwargs,
    ) -> None:
        self.width = width
        self.layers = layers
        self.heads = heads
        self.num_channels = num_channels
        self.image_size = image_size
        self.patch_size = patch_size
        self.mlp_ratio = mlp_ratio
        self.layer_norm_eps = layer_norm_eps
        self.hidden_act = hidden_act
        self.use_cls_token = use_cls_token
        self.use_ln_pre = use_ln_pre
        self.use_ln_post = use_ln_post
        self.use_abs_posemb = use_abs_posemb
        self.use_rope2d = use_rope2d
        self.ls_init_value = ls_init_value
        self.rope_theta = rope_theta
        self.rope_max_freq = rope_max_freq
        self.rope_num_freqs = rope_num_freqs
        self.rope_theta_rescale_factor = rope_theta_rescale_factor
        self.text_hidden_size = text_hidden_size
        self.projector_bias = projector_bias

    @classmethod
    def from_hf_config(cls, full_config: dict) -> "VisionConfig":
        """Build from the on-disk top-level config.json dict.

        ``full_config`` is the parsed ``config.json`` (model_type=step3p7).
        Vision fields live under ``vision_config``; the projector dims come from
        top-level ``text_config.hidden_size`` and ``projector_bias``.
        """
        vc = dict(full_config.get("vision_config", {}))
        text_hidden = full_config.get("text_config", {}).get("hidden_size", 4096)
        vc.setdefault("text_hidden_size", text_hidden)
        vc.setdefault("projector_bias", full_config.get("projector_bias", False))
        return cls(**vc)


# ---------------------------------------------------------------------------
# 2D RoPE (faithful port; interleave-style, repeat_interleave(2) + reshape
# rotate_half). NOTE: this is a different convention than the text 1D RoPE
# (which is half-style / GPT-NeoX). Kept in pure torch for correctness;
# NPU op fusion is deferred (ViT runs Prefill-only, 47 small layers).
# ---------------------------------------------------------------------------
def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate last-dim halves the *interleave* way (matches HF vision_encoder)."""
    x = x.reshape(*x.shape[:-1], -1, 2)
    x1, x2 = x.unbind(dim=-1)
    x = torch.stack((-x2, x1), dim=-1)
    return x.reshape(*x.shape[:-2], -1)


def apply_rotary_emb(
    freqs: torch.Tensor,
    t: torch.Tensor,
    start_index: int = 0,
    scale: float = 1.0,
) -> torch.Tensor:
    """Apply 2D rotary embeddings to q/k. Mirrors HF apply_rotary_emb."""
    dtype = t.dtype
    rot_dim = freqs.shape[-1]
    end_index = start_index + rot_dim
    if rot_dim > t.shape[-1]:
        raise ValueError(
            f"feature dim {t.shape[-1]} too small for rot_dim {rot_dim}")
    t_left, t_mid, t_right = (
        t[..., :start_index],
        t[..., start_index:end_index],
        t[..., end_index:],
    )
    t_mid = (t_mid * freqs.cos() * scale) + (rotate_half(t_mid) * freqs.sin() * scale)
    out = torch.cat((t_left, t_mid, t_right), dim=-1)
    return out.type(dtype)


class EncoderRope2D(nn.Module):
    """Cacheable 2D rotary positional embedding (faithful HF port)."""

    def __init__(
        self,
        dim: int,
        max_grid_height: int,
        max_grid_width: int,
        use_cls_token: bool = False,
        theta: Union[int, float] = 10000,
        max_freq: int = 10,
        num_freqs: int = 1,
        theta_rescale_factor: float = 1.0,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.max_grid_height = max_grid_height
        self.max_grid_width = max_grid_width
        self.use_cls_token = use_cls_token
        self.theta = theta * theta_rescale_factor ** (dim / (dim - 2))
        self.max_freq = max_freq
        self.num_freqs = num_freqs
        cache = self._compute_2d_freqs()
        self.register_buffer("freqs_cache", cache, persistent=False)

    def _compute_inv_freq(self, base, dim: int) -> torch.Tensor:
        return 1.0 / (base ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))

    def _compute_freqs(self, t: torch.Tensor, inv_freq: torch.Tensor) -> torch.Tensor:
        freqs = torch.einsum("..., f -> ... f", t.type(inv_freq.dtype), inv_freq)
        freqs = freqs.repeat_interleave(2, dim=-1)
        return freqs

    def _compute_2d_freqs(self) -> torch.Tensor:
        grid_h_range = torch.arange(self.max_grid_height, dtype=torch.float)
        grid_w_range = torch.arange(self.max_grid_width, dtype=torch.float)
        if self.use_cls_token:
            grid_h_range += 1
            grid_w_range += 1
        inv_freq = self._compute_inv_freq(self.theta, self.dim // 2)
        freqs_h = self._compute_freqs(grid_h_range, inv_freq)[:, None].expand(
            self.max_grid_height, self.max_grid_width, -1)
        freqs_w = self._compute_freqs(grid_w_range, inv_freq)[None, :].expand(
            self.max_grid_height, self.max_grid_width, -1)
        freqs = torch.cat([freqs_w, freqs_h], dim=-1).reshape(
            self.max_grid_height * self.max_grid_width, -1)
        if self.use_cls_token:
            freqs = torch.cat([torch.zeros(1, freqs.shape[-1]), freqs], dim=0)
        freqs = freqs[None, None, ...]
        return freqs

    def forward(self, q: torch.Tensor, k: torch.Tensor,
                grid_hw: Tuple[int, int]):
        if grid_hw[0] != self.max_grid_height or grid_hw[1] != self.max_grid_width:
            rows = torch.arange(grid_hw[0], device=q.device).view(-1, 1)
            cols = torch.arange(grid_hw[1], device=q.device).view(1, -1)
            positions = (rows * self.max_grid_width + cols).reshape(-1).to(torch.long)
            if self.use_cls_token:
                positions = torch.cat(
                    [torch.zeros(1, device=q.device), positions + 1], dim=0)
            freqs = self.freqs_cache.index_select(2, positions)
        else:
            freqs = self.freqs_cache
        q = apply_rotary_emb(freqs, q)
        k = apply_rotary_emb(freqs, k)
        return q, k


# ---------------------------------------------------------------------------
# Layer-scale / MLP / Attention / Block
# ---------------------------------------------------------------------------
class EncoderLayerScale(nn.Module):
    """Per-channel residual scaling (gamma)."""

    def __init__(self, dim: int, init_values: float) -> None:
        super().__init__()
        self.gamma = nn.Parameter(torch.full((dim,), init_values))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return hidden_states * self.gamma


def quick_gelu(x: torch.Tensor) -> torch.Tensor:
    """quick_gelu = x * sigmoid(1.702 * x). Matches transformers ACT2FN."""
    return x * torch.sigmoid(1.702 * x)


class EncoderMLP(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int) -> None:
        super().__init__()
        self.c_fc = nn.Linear(hidden_size, intermediate_size, bias=True)
        self.c_proj = nn.Linear(intermediate_size, hidden_size, bias=True)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.c_proj(quick_gelu(self.c_fc(hidden_states)))


class EncoderVisionAttention(nn.Module):
    """Multi-head self attention with 2D RoPE, merged QKV + bias, non-causal."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        max_grid_height: int,
        max_grid_width: int,
        use_cls_token: bool = False,
        use_rope2d: bool = True,
        rope_theta: Union[int, float] = 10000,
        rope_max_freq: int = 10,
        rope_num_freqs: int = 1,
        rope_theta_rescale_factor: float = 1.0,
        use_fa: bool = True,
    ) -> None:
        super().__init__()
        if hidden_size % num_heads != 0:
            raise ValueError(
                f"hidden_size ({hidden_size}) must be divisible by num_heads ({num_heads}).")
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.scale = self.head_dim ** -0.5
        self.use_fa = use_fa
        # Merged QKV projection stored exactly as the HF checkpoint
        # (in_proj_weight [3H,H], in_proj_bias [3H]).
        self.in_proj_weight = nn.Parameter(torch.zeros(hidden_size * 3, hidden_size))
        self.in_proj_bias = nn.Parameter(torch.zeros(hidden_size * 3))
        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=True)

        self.rope = None
        if use_rope2d:
            self.rope = EncoderRope2D(
                dim=self.head_dim,
                max_grid_height=max_grid_height,
                max_grid_width=max_grid_width,
                use_cls_token=use_cls_token,
                theta=rope_theta,
                max_freq=rope_max_freq,
                num_freqs=rope_num_freqs,
                theta_rescale_factor=rope_theta_rescale_factor,
            )

    def forward(self, hidden_states: torch.Tensor,
                grid_hw: Tuple[int, int]) -> torch.Tensor:
        bsz, seq_len, _ = hidden_states.shape
        qkv = F.linear(hidden_states, self.in_proj_weight, self.in_proj_bias)
        q, k, v = qkv.chunk(3, dim=-1)

        # (B, N, S, Dh)
        q = q.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        if self.rope is not None:
            q, k = self.rope(q, k, grid_hw=grid_hw)
        v = v.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        if self.use_fa and _HAS_NPU:
            # NPU FA (non-causal, full bidirectional). BSH layout: collapse the
            # head axis back into hidden so we pass [B, S, N*Dh]. q/k already
            # have RoPE applied per-head, so reshape from (B,N,S,Dh)->(B,S,N*Dh).
            q_bsh = q.transpose(1, 2).reshape(bsz, seq_len, self.num_heads * self.head_dim)
            k_bsh = k.transpose(1, 2).reshape(bsz, seq_len, self.num_heads * self.head_dim)
            v_bsh = v.transpose(1, 2).reshape(bsz, seq_len, self.num_heads * self.head_dim)
            attn_output, _ = torch.ops.npu.npu_fused_infer_attention_score(
                q_bsh, k_bsh, v_bsh,
                num_heads=self.num_heads,
                num_key_value_heads=self.num_heads,
                input_layout="BSH",
                scale=self.scale,
                sparse_mode=0,
                atten_mask=None,
            )
            # attn_output is (B, S, N*Dh)
            return self.out_proj(attn_output)

        # Pure-torch SDPA fallback (CPU / correctness reference path).
        attn_output = F.scaled_dot_product_attention(
            q, k, v, is_causal=False, scale=self.scale)
        attn_output = attn_output.transpose(1, 2).reshape(
            bsz, seq_len, self.num_heads * self.head_dim)
        return self.out_proj(attn_output)


class EncoderVisionBlock(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: float,
        layer_norm_eps: float,
        ls_init_value: float,
        max_grid_height: int,
        max_grid_width: int,
        use_cls_token: bool = False,
        use_rope2d: bool = True,
        rope_kwargs: Optional[dict] = None,
        use_fa: bool = True,
    ) -> None:
        super().__init__()
        rope_kwargs = rope_kwargs or {}
        self.attn = EncoderVisionAttention(
            hidden_size, num_heads,
            max_grid_height=max_grid_height,
            max_grid_width=max_grid_width,
            use_cls_token=use_cls_token,
            use_rope2d=use_rope2d,
            use_fa=use_fa,
            **rope_kwargs,
        )
        self.ln_1 = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.ln_2 = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        intermediate = int(hidden_size * mlp_ratio)
        self.mlp = EncoderMLP(hidden_size, intermediate)
        self.ls_1 = EncoderLayerScale(hidden_size, ls_init_value)
        self.ls_2 = EncoderLayerScale(hidden_size, ls_init_value)

    def forward(self, hidden_states: torch.Tensor,
                grid_hw: Tuple[int, int]) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.ln_1(hidden_states)
        hidden_states = self.attn(hidden_states, grid_hw=grid_hw)
        hidden_states = residual + self.ls_1(hidden_states)

        residual = hidden_states
        hidden_states = self.ln_2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + self.ls_2(hidden_states)
        return hidden_states


class EncoderVisionTransformer(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        depth: int,
        num_heads: int,
        mlp_ratio: float,
        layer_norm_eps: float,
        ls_init_value: float,
        max_grid_height: int,
        max_grid_width: int,
        use_cls_token: bool = False,
        use_rope2d: bool = True,
        rope_kwargs: Optional[dict] = None,
        use_fa: bool = True,
    ) -> None:
        super().__init__()
        self.layers = depth
        rope_kwargs = rope_kwargs or {}
        self.resblocks = nn.ModuleList([
            EncoderVisionBlock(
                embed_dim, num_heads, mlp_ratio, layer_norm_eps,
                ls_init_value,
                max_grid_height=max_grid_height,
                max_grid_width=max_grid_width,
                use_cls_token=use_cls_token,
                use_rope2d=use_rope2d,
                rope_kwargs=rope_kwargs,
                use_fa=use_fa,
            )
            for _ in range(depth)
        ])

    def forward(self, hidden_states: torch.Tensor,
                grid_hw: Tuple[int, int]) -> torch.Tensor:
        for block in self.resblocks:
            hidden_states = block(hidden_states, grid_hw=grid_hw)
        return hidden_states


# ---------------------------------------------------------------------------
# Top-level vision tower (encoder + downsamplers + projector).
# Note: in the original HF code vit_downsampler1/2 live INSIDE vision_model,
# but vit_large_projector is a TOP-LEVEL parameter of Step3p7Model. We keep the
# same checkpoint key layout: downsamplers under `vision_model.`, projector at
# the module root (`vit_large_projector.weight`).
# ---------------------------------------------------------------------------
class StepRoboticsVisionEncoder(nn.Module):
    """Vision encoder: patch embed -> 47 ViT blocks -> (no ln_post)."""

    def __init__(self, config: VisionConfig, use_fa: bool = True) -> None:
        super().__init__()
        self.config = config
        self.hidden_size = config.width
        self.num_heads = config.heads
        self.num_hidden_layers = config.layers
        self.patch_size = config.patch_size
        self.image_size = config.image_size
        self.use_cls_token = config.use_cls_token
        self.use_rope2d = config.use_rope2d
        self.use_abs_posemb = config.use_abs_posemb
        self.layer_norm_eps = config.layer_norm_eps
        self.mlp_ratio = config.mlp_ratio
        self.ls_init_value = config.ls_init_value
        self.use_ln_pre = config.use_ln_pre
        self.use_ln_post = config.use_ln_post

        self.conv1 = nn.Conv2d(
            in_channels=config.num_channels,
            out_channels=self.hidden_size,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            bias=False,
        )

        self.ln_pre = (nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)
                       if self.use_ln_pre else nn.Identity())
        self.ln_post = (nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)
                        if self.use_ln_post else nn.Identity())

        grid_size = self.image_size // self.patch_size
        self.base_grid = (grid_size, grid_size)

        if self.use_cls_token:
            self.class_embedding = nn.Parameter(
                torch.randn(self.hidden_size) * (self.hidden_size ** -0.5))
        else:
            self.class_embedding = None

        if self.use_abs_posemb:
            self.posemb_grid_size = self.image_size // self.patch_size
            self.positional_embedding = nn.Parameter(
                (self.hidden_size ** -0.5) * torch.randn(
                    int(self.use_cls_token) + self.posemb_grid_size ** 2,
                    self.hidden_size,
                ))

        self.transformer = EncoderVisionTransformer(
            embed_dim=self.hidden_size,
            depth=self.num_hidden_layers,
            num_heads=self.num_heads,
            mlp_ratio=self.mlp_ratio,
            layer_norm_eps=self.layer_norm_eps,
            ls_init_value=self.ls_init_value,
            max_grid_height=self.base_grid[0],
            max_grid_width=self.base_grid[1],
            use_cls_token=self.use_cls_token,
            use_rope2d=self.use_rope2d,
            rope_kwargs={
                "rope_theta": config.rope_theta,
                "rope_max_freq": config.rope_max_freq,
                "rope_num_freqs": config.rope_num_freqs,
                "rope_theta_rescale_factor": config.rope_theta_rescale_factor,
            },
            use_fa=use_fa,
        )
        self.vit_downsampler1 = nn.Conv2d(
            self.hidden_size, self.hidden_size * 2,
            kernel_size=3, stride=2, padding=1)
        self.vit_downsampler2 = nn.Conv2d(
            self.hidden_size * 2, self.hidden_size * 4,
            kernel_size=3, stride=2, padding=1)

    def sample_abs_posemb(self, grid_h: int, grid_w: int) -> torch.Tensor:
        if self.posemb_grid_size == grid_h and self.posemb_grid_size == grid_w:
            return self.positional_embedding[None, ...]

        pos_embed = self.positional_embedding
        if self.use_cls_token:
            cls_token_embed, pos_embed = pos_embed[:1], pos_embed[1:]
        pos_embed = (pos_embed.reshape(1, self.posemb_grid_size,
                                       self.posemb_grid_size, -1)
                     .permute(0, 3, 1, 2).contiguous())
        pos_embed = F.interpolate(pos_embed, size=(grid_h, grid_w),
                                  mode="bilinear", align_corners=False)
        pos_embed = pos_embed.permute(0, 2, 3, 1).reshape(-1, self.hidden_size)
        if self.use_cls_token:
            pos_embed = torch.cat([cls_token_embed, pos_embed], dim=0)
        return pos_embed[None, ...]

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        bsz, _, height, width = pixel_values.shape
        grid_h, grid_w = height // self.patch_size, width // self.patch_size

        hidden_state = self.conv1(pixel_values)            # (B, D, Gh, Gw)
        hidden_state = hidden_state.flatten(2).transpose(1, 2)  # (B, Gh*Gw, D)

        if self.use_cls_token:
            cls_token = self.class_embedding.view(1, 1, -1).expand(bsz, -1, -1)
            hidden_state = torch.cat([cls_token, hidden_state], dim=1)

        if self.use_abs_posemb:
            pos_emb = self.sample_abs_posemb(grid_h, grid_w)
            hidden_state = hidden_state + pos_emb
        hidden_state = self.ln_pre(hidden_state)
        hidden_state = self.transformer(hidden_state, grid_hw=(grid_h, grid_w))

        if self.use_ln_post:
            hidden_state = self.ln_post(hidden_state)
        if self.use_cls_token:
            hidden_state = hidden_state[:, 1:, :]
        return hidden_state


class Step3p7VisionTower(nn.Module):
    """Vision tower + projector glue producing text-space image embeds.

    Keeps the original checkpoint key layout: encoder/downsampler weights under
    ``vision_model.``, projector at root (``vit_large_projector.weight``).
    """

    def __init__(self, config: VisionConfig, use_fa: bool = True) -> None:
        super().__init__()
        self.config = config
        self.vision_model = StepRoboticsVisionEncoder(config, use_fa=use_fa)
        # 6144 (= width*4) -> text hidden (4096); bias=False per config.
        self.vit_large_projector = nn.Linear(
            config.width * 4, config.text_hidden_size, bias=config.projector_bias)

    def _process_image_features(self, image_features: torch.Tensor) -> torch.Tensor:
        # image_features: (batch, num_patches, width) ; num_patches = grid*grid (e.g. 2704)
        batch, num_patches = image_features.shape[:2]
        grid = int(num_patches ** 0.5)
        image_features = image_features.permute(0, 2, 1).view(batch, -1, grid, grid)
        image_features = self.vision_model.vit_downsampler1(image_features)
        image_features = self.vision_model.vit_downsampler2(image_features)
        batch, _, grid, _ = image_features.shape
        image_features = image_features.view(batch, -1, grid * grid).permute(0, 2, 1)
        image_features = self.vit_large_projector(image_features)
        return image_features  # (batch, 169, text_hidden)

    @torch.no_grad()
    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """pixel_values (B,3,728,728) -> image_embeds (B,169,4096)."""
        feats = self.vision_model(pixel_values)         # (B, 2704, 1536)
        embeds = self._process_image_features(feats)    # (B, 169, 4096)
        return embeds

    # -----------------------------------------------------------------------
    # Weight loading: maps the 667 checkpoint keys
    #   vision_model.*  (299 in model-vit-00001 + most resblocks in 00002)
    #   vit_large_projector.weight (in 00001)
    # straight onto this module's parameters (same names, no transform).
    # -----------------------------------------------------------------------
    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]) -> set:
        params = dict(self.named_parameters())
        loaded = set()
        for name, w in weights:
            if name not in params:
                # Only vision_model.* / vit_large_projector.* are expected; skip
                # anything else (e.g. text/MTP keys if the iterator is broad).
                continue
            p = params[name]
            if p.shape != w.shape:
                raise ValueError(
                    f"vision weight shape mismatch for {name}: "
                    f"param {tuple(p.shape)} vs ckpt {tuple(w.shape)}")
            p.data.copy_(w.to(p.dtype).to(p.device))
            loaded.add(name)
        return loaded

    def expected_weight_names(self) -> set:
        return set(self.named_parameters().keys()) if hasattr(
            self.named_parameters(), "keys") else {n for n, _ in self.named_parameters()}


__all__ = [
    "VisionConfig",
    "StepRoboticsVisionEncoder",
    "Step3p7VisionTower",
    "quick_gelu",
    "rotate_half",
    "apply_rotary_emb",
    "EncoderRope2D",
]
