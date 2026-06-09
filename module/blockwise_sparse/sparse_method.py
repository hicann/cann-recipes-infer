from abc import ABC, abstractmethod
import json
import math
import os
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.nn.functional as F

try:
    import torch_bsa
except ImportError:
    torch_bsa = None

import torch_npu
import yaml
from loguru import logger
from module.unified_sp.uaa import all_gather_anything, _maybe_pad_qkv_head, _maybe_unpad_qkv_head

DEFAULT_CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sparse_config.yaml")


def load_sparse_config_from_file(config_path=DEFAULT_CONFIG_PATH):
    """Load sparse config from YAML.

    Supports two layouts:
    1. Inline (new): the launch YAML has a top-level ``sparse`` section with
       ``method``, ``block_size_Q/K``, ``model`` and per-method params nested
       under ``params.<Method>``. The section is flattened to the legacy shape
       expected downstream.
    2. Flat (legacy): a standalone ``sparse_config.yaml`` whose root is the
       flat structure (``block_size_Q`` / ``block_size_K`` / ``model`` at the
       top, method keys ``TopK`` / ``SVG`` holding per-method params).
    """
    with open(config_path, 'r', encoding='utf-8') as f:
        full_cfg = yaml.safe_load(f)

    if isinstance(full_cfg, dict) and isinstance(full_cfg.get("sparse"), dict):
        sparse = full_cfg["sparse"]
        config = {
            "block_size_Q": sparse.get("block_size_Q"),
            "block_size_K": sparse.get("block_size_K"),
            "model": sparse.get("model"),
        }
        for method, params in (sparse.get("params") or {}).items():
            method_cfg = dict(params)
            method_cfg.setdefault("predictor_name", method)  # TopKPredictor/SVGPredictor legacy field
            config[method] = method_cfg
    else:
        config = full_cfg

    logger.info(config)
    _validate_config_keys(config)
    return config


def _validate_config_keys(config: dict):
    required_keys = ["block_size_Q", "block_size_K", "model"]
    missed_key = [k for k in required_keys if k not in config]
    if missed_key:
        raise ValueError(f"Missing required key(s): {','.join(missed_key)}")


def parse_sparse_time_step(value):
    """
    输入： str，如"20, 30-40, 50"
    输出： list[int]
    """
    result = []
    parts = [p.strip() for p in value.split(',')]

    for part in parts:
        if not part:
            continue

        if "-" in part:
            try:
                start, end = map(int, part.split("-"))
            except ValueError as e:
                raise ValueError(
                    f"Invalid range format: {part}"
                ) from e

            if start > end:
                raise ValueError(
                    f"Invalid range: {part} (start > end)"
                )

            result.extend(range(start, end + 1))
        else:
            try:
                result.append(int(part))
            except ValueError as e:
                raise ValueError(
                    f"Invalid integer: {part}"
                ) from e
    return sorted(set(result))


class SparsePredictorManager():
    def __init__(self) -> None:
        self.sparse_attn_mode = None
        self.config = None
        self.sparse_params = {}
    
    def from_config(self, config_path, sparse_method, sparse_params=None):
        self.config = load_sparse_config_from_file(config_path)
        self.config[sparse_method]['sparse_time_step'] = \
                parse_sparse_time_step(self.config[sparse_method]['sparse_time_step'])

        if sparse_params is not None:
            self.sparse_params.update(sparse_params)
        if sparse_method == "TopK" and self.config['model'] == "HunyuanVideo":
            self.sparse_attn_mode = HunyuanVideoTopKAdapter(self.config, self.sparse_params)
        if sparse_method == "SVG" and self.config['model'] == "HunyuanVideo":
            self.sparse_attn_mode = HunyuanVideoSVGAdapter(self.config, self.sparse_params)
        # logger.info(f"Apply sparse predictor method: {self.sparse_attn_mode.predictor_name}!")

sparse_predictor_manager = SparsePredictorManager()


def sync_and_get_time(start_time=None, use_syn=True):
    if use_syn:
        torch.npu.synchronize()
    time_stamp = time.time()
    if start_time is not None:
        time_stamp -= start_time
        return time_stamp
    return time_stamp


class BaseSparsePredictor(ABC):
    def __init__(self, sparse_config: Dict[str, Any], sparse_params: Optional[Dict[str, Any]] = None):
        self.sparse_params = sparse_params or {}
        self.sparse_config = sparse_config

        # ---------- 通用参数 ----------
        self.block_size_q = int(sparse_config["block_size_Q"])
        self.block_size_k = int(sparse_config["block_size_K"])

        # ---------- 网络结构参数 ----------
        self.double_stream_layers = self.sparse_params.get("double_stream_layers")
        self.single_stream_layers = self.sparse_params.get("single_stream_layers")
        self.attn_layers = self.sparse_params.get("attn_layers")
        self.total_steps = self.sparse_params.get("num_steps")
        self.device = self.sparse_params.get("device")

        # ---------- 层/步计数器 ----------
        self.total_layers_per_step = (
            self.double_stream_layers + self.single_stream_layers
            if self.double_stream_layers is not None and self.single_stream_layers is not None
            else self.attn_layers
        )
        self.step = 0
        self.layer_counter = 0
        self.index_type = torch.int32

        # ---------- 统一状态快照 ----------
        self.current = {
            "step": self.step,
            "layer": 0,
            "num_steps": self.total_steps
        }

    @staticmethod
    def _route_heads_bsnd(x: torch.Tensor, head_index: torch.Tensor) -> torch.Tensor:
        head_index = head_index.to(device=x.device, dtype=torch.long)
        return x.index_select(2, head_index).contiguous()

    @staticmethod
    def _slice_ulysses_local_heads_with_uaa(runtime_attn, q, k, v, global_head_num):
        if runtime_attn.ulysses_anything:
            h = q.shape[2]
            padded_tensors = []
            h_pad = 0
            for idx, tensor in enumerate((q, k, v)):
                padded_tensor, tensor_h_pad = _maybe_pad_qkv_head(tensor, h, runtime_attn.ulysses_world_size)
                padded_tensors.append(padded_tensor)
                if idx == 0:
                    h_pad = tensor_h_pad
            head_per_rank = (h + h_pad) // runtime_attn.ulysses_world_size
            head_start = runtime_attn.ulysses_rank * head_per_rank
            head_end = head_start + head_per_rank
            sliced_tensors = [
                tensor[:, :, head_start:head_end, :].contiguous()
                for tensor in padded_tensors
            ]
            sliced_tensors = [
                _maybe_unpad_qkv_head(
                    tensor,
                    h_pad,
                    runtime_attn.ulysses_rank,
                    runtime_attn.ulysses_world_size,
                    runtime_attn.ulysses_pg,
                ).contiguous()
                for tensor in sliced_tensors
            ]
            return tuple(sliced_tensors)

        if global_head_num <= 0 or global_head_num % runtime_attn.ulysses_world_size != 0:
            raise ValueError("global head num must be divisible by ulysses_world_size in standard ulysses.")
        head_per_rank = global_head_num // runtime_attn.ulysses_world_size
        head_start = runtime_attn.ulysses_rank * head_per_rank
        head_end = head_start + head_per_rank
        return tuple(
            tensor[:, :, head_start:head_end, :].contiguous()
            for tensor in (q, k, v)
        )

    @staticmethod
    def _ulysses_all_gather_heads_bshd(runtime_attn, x: torch.Tensor) -> torch.Tensor:
        if runtime_attn.ulysses_world_size <= 1:
            return x
        return all_gather_anything(
            tensor=x,
            dim=2,
            world_size=runtime_attn.ulysses_world_size,
            group=runtime_attn.ulysses_pg,
        ).contiguous()

    @staticmethod
    def _ulysses_all_to_all_qkv(runtime_attn, x: torch.Tensor) -> torch.Tensor:
        if runtime_attn.ulysses_anything:
            return getattr(runtime_attn, "_all_to_all_qkv_anything")(x)()
        return getattr(runtime_attn, "_all_to_all_qkv")(x)

    @staticmethod
    def _ulysses_all_to_all_o(
        runtime_attn,
        x: torch.Tensor,
        *,
        num_qo_head: int,
        q_s_local: int,
    ) -> torch.Tensor:
        if runtime_attn.ulysses_anything:
            return getattr(runtime_attn, "_all_to_all_o_anything")(
                x,
                NUM_QO_HEAD=num_qo_head,
                Q_S_LOCAL=q_s_local,
            )()
        return getattr(runtime_attn, "_all_to_all_o")(x)

    def _ulysses_all_to_all_qkv_triplet(self, runtime_attn, q, k, v):
        return (
            self._ulysses_all_to_all_qkv(runtime_attn, q),
            self._ulysses_all_to_all_qkv(runtime_attn, k),
            self._ulysses_all_to_all_qkv(runtime_attn, v),
        )

    @staticmethod
    def _move_sink_qkv_to_end(q, k, v, *, q_sink_len: int, kv_sink_len: int):
        sink_q, rest_q = q[:, :, :q_sink_len, :], q[:, :, q_sink_len:, :]
        sink_k, rest_k = k[:, :, :kv_sink_len, :], k[:, :, kv_sink_len:, :]
        sink_v, rest_v = v[:, :, :kv_sink_len, :], v[:, :, kv_sink_len:, :]
        return (
            torch.cat((rest_q, sink_q), dim=2).contiguous(),
            torch.cat((rest_k, sink_k), dim=2).contiguous(),
            torch.cat((rest_v, sink_v), dim=2).contiguous(),
        )

    @staticmethod
    def _split_img_txt_qkv(q_full, k_full, v_full, img_q_len: int, img_kv_len: int):
        return {
            "q_img": q_full[:, :img_q_len, :, :].contiguous(),
            "k_img": k_full[:, :img_kv_len, :, :].contiguous(),
            "v_img": v_full[:, :img_kv_len, :, :].contiguous(),
            "txt_q": q_full[:, img_q_len:, :, :].contiguous(),
            "txt_k": k_full[:, img_kv_len:, :, :].contiguous(),
            "txt_v": v_full[:, img_kv_len:, :, :].contiguous(),
        }

    @staticmethod
    def _ring_all_gather_seq(runtime_attn, x: torch.Tensor) -> torch.Tensor:
        b, s_local, h, d = x.shape
        gathered = torch.empty(
            (runtime_attn.ring_world_size, b, s_local, h, d),
            dtype=x.dtype,
            device=x.device,
        )
        dist.all_gather_into_tensor(gathered, x.contiguous(), group=runtime_attn.ring_pg)
        return gathered.permute(1, 0, 2, 3, 4).reshape(b, runtime_attn.ring_world_size * s_local, h, d)

    @staticmethod
    def _ring_gathered_img_to_bnsd(pre_attn_layout, gathered, txt_bnsd, rank_indices=None):
        if rank_indices is not None:
            gathered = torch.index_select(gathered, dim=0, index=rank_indices)
        _, b, _, n, d = gathered.shape
        img_bshd = gathered.permute(1, 0, 2, 3, 4).reshape(b, -1, n, d).contiguous()
        return torch.cat([pre_attn_layout(img_bshd), txt_bnsd], dim=2).contiguous()

    @staticmethod
    def _split_patched_width(total_width: int, world_size: int):
        base = int(total_width) // int(world_size)
        extra = int(total_width) % int(world_size)
        return [base + (1 if rank < extra else 0) for rank in range(int(world_size))]

    def _get_ring_local_sink_token_len(
        self,
        *,
        local_img_token_len: int,
        ring_rank: int,
        ring_world_size: int,
    ) -> int:
        sink_len = min(int(getattr(self, "sink_frame_len", 0)), int(getattr(self, "img_token_len", 0)))
        if sink_len <= 0:
            return 0
        frame_num = int(getattr(self, "frame_num", 0))
        frame_patch_h = int(getattr(self, "frame_patch_h", 0))
        frame_patch_w = int(getattr(self, "frame_patch_w", 0))
        width_splits = self._split_patched_width(frame_patch_w, ring_world_size)
        width_start = int(sum(width_splits[:ring_rank]))
        width_len = int(width_splits[ring_rank])
        frame_size = frame_patch_h * frame_patch_w
        full_frames, rem_tokens = divmod(sink_len, frame_size)
        local_sink = full_frames * frame_patch_h * width_len
        for row in range(frame_patch_h):
            row_start = row * frame_patch_w
            row_covered = max(0, min(rem_tokens - row_start, frame_patch_w))
            if row_covered <= 0:
                break
            local_sink += max(0, min(width_start + width_len, row_covered) - width_start)
        return max(0, min(int(local_sink), int(local_img_token_len)))

    @staticmethod
    def _ring_all_gather_seq_bnsd(runtime_attn, x: torch.Tensor) -> torch.Tensor:
        b, n, s_local, d = x.shape
        local_len = torch.tensor([s_local], dtype=torch.int32, device=x.device)
        gathered_lens = [torch.empty_like(local_len) for _ in range(runtime_attn.ring_world_size)]
        dist.all_gather(gathered_lens, local_len, group=runtime_attn.ring_pg)
        seq_lens = [int(item.item()) for item in gathered_lens]
        max_len = max(seq_lens)
        if s_local < max_len:
            pad = torch.zeros((b, n, max_len - s_local, d), dtype=x.dtype, device=x.device)
            x = torch.cat([x, pad], dim=2).contiguous()
        gathered = torch.empty(
            (runtime_attn.ring_world_size, b, n, max_len, d),
            dtype=x.dtype,
            device=x.device,
        )
        dist.all_gather_into_tensor(gathered, x.contiguous(), group=runtime_attn.ring_pg)
        parts = [gathered[rank, :, :, :seq_lens[rank], :] for rank in range(runtime_attn.ring_world_size)]
        return torch.cat(parts, dim=2).contiguous()

    @staticmethod
    def _merge_two_sparse_outputs(out1, lse1, out2, lse2):
        if lse1 is None or lse2 is None:
            raise ValueError("Sparse ring overlap requires LSE outputs.")

        lse1 = BaseSparsePredictor._format_sparse_lse_for_merge(lse1, out1)
        lse2 = BaseSparsePredictor._format_sparse_lse_for_merge(lse2, out2)
        merged, _ = BaseSparsePredictor._merge_sparse_outputs_with_lse(out1, lse1, out2, lse2)
        return merged.to(dtype=out1.dtype)

    @staticmethod
    def _format_sparse_lse_for_merge(lse: torch.Tensor, ref_out: torch.Tensor) -> torch.Tensor:
        b, h, s = ref_out.shape[:3]
        if lse.dim() == 4 and lse.shape == (b, h, s, 1):
            return lse.to(dtype=torch.float32)
        if lse.dim() == 3 and lse.shape == (b, h, s):
            return lse.unsqueeze(-1).to(dtype=torch.float32)
        raise ValueError(f"Unsupported sparse LSE shape: {tuple(lse.shape)}")

    @staticmethod
    def _merge_sparse_outputs_with_lse(out1, lse1, out2, lse2):
        # local/other 两路 sparse 输出共享同一套 LSE 归一化和加权逻辑。
        lses = (lse1, lse2)
        valid = [torch.isfinite(lse) for lse in lses]
        any_valid = valid[0] | valid[1]
        masked_lse = [
            torch.where(mask, lse, torch.full_like(lse, float("-inf")))
            for lse, mask in zip(lses, valid)
        ]
        max_lse = torch.maximum(masked_lse[0], masked_lse[1])
        safe_max_lse = torch.where(any_valid, max_lse, torch.zeros_like(max_lse))
        exp_lse = [
            torch.where(mask, torch.exp(lse - safe_max_lse), torch.zeros_like(lse))
            for lse, mask in zip(lses, valid)
        ]
        denom = torch.clamp_min(exp_lse[0] + exp_lse[1], 1e-30)
        weights = [
            torch.where(any_valid, exp_item / denom, torch.zeros_like(exp_item))
            for exp_item in exp_lse
        ]
        merged = weights[0] * out1.to(torch.float32)
        merged = merged + weights[1] * out2.to(torch.float32)
        merged_lse = safe_max_lse + torch.log(denom)
        merged_lse = torch.where(any_valid, merged_lse, torch.full_like(merged_lse, float("-inf")))
        return merged.to(dtype=out1.dtype), merged_lse

    def _build_ring_native_topk_sabi(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        *,
        q_dense_prefix_len: int = 0,
        q_dense_suffix_len: int = 0,
        k_mean_override: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        block_num_q = math.ceil(int(q.shape[2]) / int(self.block_size_q))
        block_num_k = (
            int(k_mean_override.shape[2])
            if k_mean_override is not None
            else math.ceil(int(k.shape[2]) / int(self.block_size_k))
        )

        if k_mean_override is None:
            sabi = self.get_sabi_v2(q, k)
        else:
            # overlap 分支复用已聚合的 KV block 均值，只重算当前 query block 的 TopK SABI。
            q_mean = self.pooling_matmul(q, self.block_size_q, block_num_q)
            attn = (q_mean @ k_mean_override.transpose(-2, -1)).softmax(dim=-1)
            sabi = torch.full(
                size=(q.shape[0], q.shape[1], block_num_q, block_num_k),
                fill_value=-1,
                dtype=self.index_type,
                device=q.device,
            )
            step_idx, layer_idx = self.get_effective_indices()
            sparsity = self._get_layer_sparsity_runtime_head_order(step_idx, layer_idx, device=attn.device)
            k_nums = ((1.0 - sparsity) * block_num_k).to(torch.int32)
            k_nums = torch.clamp(k_nums, min=1, max=block_num_k)
            max_k = int(k_nums.max().item())
            if max_k > 0:
                _, indices = torch.topk(attn, max_k, dim=-1)
                k_nums_expanded = k_nums.view(1, q.shape[1], 1, 1)
                arange_k = torch.arange(max_k, device=q.device).view(1, 1, 1, max_k)
                mask = arange_k < k_nums_expanded
                sabi[:, :, :, :max_k] = torch.where(mask, indices, -1)
        full_k = torch.arange(block_num_k, dtype=self.index_type, device=q.device).view(1, 1, 1, block_num_k)
        full_k = full_k.expand(q.shape[0], q.shape[1], -1, -1)

        prefix_blocks = math.ceil(max(0, int(q_dense_prefix_len)) / int(self.block_size_q))
        suffix_blocks = math.ceil(max(0, int(q_dense_suffix_len)) / int(self.block_size_q))
        # sink 前缀和 text 后缀需要保持 dense 语义，因此对应 query block 强制保留全部 KV block。
        if prefix_blocks > 0:
            end = min(prefix_blocks, block_num_q)
            sabi[:, :, :end, :] = full_k.expand(q.shape[0], q.shape[1], end, block_num_k)
        if suffix_blocks > 0:
            start = max(0, block_num_q - suffix_blocks)
            rows = block_num_q - start
            sabi[:, :, start:, :] = full_k.expand(q.shape[0], q.shape[1], rows, block_num_k)
        return sabi.to(device=q.device, dtype=torch.uint16).contiguous()

    def _run_ring_topk_sparse_part(
        self,
        q,
        k,
        v,
        *,
        batch_size,
        num_heads,
        scale,
        dense_prefix_len,
        dense_suffix_len,
    ):
        sabi = self._build_ring_native_topk_sabi(
            q,
            k,
            q_dense_prefix_len=dense_prefix_len,
            q_dense_suffix_len=dense_suffix_len,
        )
        return self._call_blitz_sparse_attention(
            q,
            k,
            v,
            sabi=sabi,
            actual_seq_lengths=[int(q.shape[2])] * batch_size,
            actual_seq_lengths_kv=[int(k.shape[2])] * batch_size,
            num_heads=num_heads,
            scale=scale,
            return_lse=True,
        )

    def _call_blitz_sparse_attention(
        self,
        q,
        k,
        v,
        *,
        sabi,
        actual_seq_lengths,
        actual_seq_lengths_kv,
        num_heads,
        scale,
        return_lse: bool = False,
    ):
        kwargs = dict(
            sabi=sabi,
            actual_seq_lengths=actual_seq_lengths,
            actual_seq_lengths_kv=actual_seq_lengths_kv,
            num_heads=num_heads,
            num_key_value_heads=num_heads,
            input_layout="BNSD",
            scale_value=scale,
            atten_mask=None,
            sparse_mode=0,
            block_shape=[self.block_size_q, self.block_size_k],
        )
        if return_lse:
            kwargs["softmax_lse_flag"] = True
        out = torch_bsa.blitz_sparse_attention(q.contiguous(), k.contiguous(), v.contiguous(), **kwargs)
        if isinstance(out, tuple):
            if len(out) >= 2:
                return out[0], out[1]
            return out[0], None
        return out, None

    @staticmethod
    def _take_attention_output(attn_out):
        if isinstance(attn_out, tuple):
            return attn_out[0]
        return attn_out

    @staticmethod
    def _replace_prefix_with_dense_attention(
        sparse_out: torch.Tensor,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        prefix_len: int,
        num_heads: int,
        scale: float,
    ) -> torch.Tensor:
        if prefix_len <= 0:
            return sparse_out
        dense_prefix = torch_npu.npu_fused_infer_attention_score(
            q[:, :, :prefix_len, :].contiguous(),
            k,
            v,
            num_heads=num_heads,
            input_layout="BNSD",
            scale=scale,
        )[0]
        if prefix_len < sparse_out.shape[2]:
            return torch.cat([dense_prefix, sparse_out[:, :, prefix_len:, :]], dim=2).contiguous()
        return dense_prefix.contiguous()

    def _forward_ring_topk_global(
        self,
        runtime_attn,
        block_args: dict,
        softmax_scale: Optional[float],
    ) -> torch.Tensor:
        q_img_local, k_img_local, v_img_local = (
            block_args["q_img_local"], block_args["k_img_local"], block_args["v_img_local"]
        )
        txt_q, txt_k, txt_v = block_args["txt_q"], block_args["txt_k"], block_args["txt_v"]
        k_img_global = self._ring_all_gather_seq(runtime_attn, k_img_local)
        v_img_global = self._ring_all_gather_seq(runtime_attn, v_img_local)
        # Ring 只切分序列维，TopK 构建 SABI 前需要先聚合完整 KV 序列。
        local_sink_len = self._get_ring_local_sink_token_len(
            local_img_token_len=int(q_img_local.shape[1]),
            ring_rank=int(runtime_attn.ring_rank),
            ring_world_size=int(runtime_attn.ring_world_size),
        )
        # 首帧 sink query 走 dense 回填，稀疏主体先把本地 sink 段移到尾部。
        q_img_exec = torch.cat(
            [q_img_local[:, local_sink_len:, :, :], q_img_local[:, :local_sink_len, :, :]],
            dim=1,
        ).contiguous() if local_sink_len > 0 else q_img_local
        q_full = torch.cat([q_img_exec, txt_q], dim=1).contiguous()
        k_full = torch.cat([k_img_global, txt_k], dim=1).contiguous()
        v_full = torch.cat([v_img_global, txt_v], dim=1).contiguous()
        prefix_q_len, prefix_kv_len = int(q_full.shape[1]), int(k_full.shape[1])
        out_full = self.attention(
            q=q_full,
            k=k_full,
            v=v_full,
            cu_seqlens_q=[0, prefix_q_len],
            cu_seqlens_kv=[0, prefix_kv_len],
            return_bshd=True,
            softmax_scale=softmax_scale,
            img_token_len_q=int(q_img_local.shape[1]),
            img_token_len_k=int(k_img_global.shape[1]),
            sink_frame_len_q=0,
            sink_frame_len_k=int(self.sink_frame_len),
        )
        img_out = out_full[:, :int(q_img_local.shape[1]), :, :].contiguous()
        if local_sink_len > 0:
            img_out = torch.cat([img_out[:, -local_sink_len:, :, :], img_out[:, :-local_sink_len, :, :]], dim=1)
            dense_sink = torch_npu.npu_fused_infer_attention_score(
                q_img_local[:, :local_sink_len, :, :].contiguous(),
                k_full,
                v_full,
                num_heads=int(q_img_local.shape[2]),
                input_layout="BSND",
                scale=softmax_scale if softmax_scale is not None else q_img_local.shape[-1] ** (-0.5),
            )[0]
            img_out = torch.cat([dense_sink, img_out[:, local_sink_len:, :, :]], dim=1).contiguous()
        txt_out = out_full[:, int(q_img_local.shape[1]):, :, :].contiguous()
        return torch.cat([img_out.contiguous(), txt_out], dim=1).contiguous()

    def _forward_ring_topk_overlap(
        self,
        runtime_attn,
        block_args: dict,
        softmax_scale: Optional[float],
    ) -> torch.Tensor:
        q_img_local, k_img_local, v_img_local = (
            block_args["q_img_local"], block_args["k_img_local"], block_args["v_img_local"]
        )
        txt_q, txt_k, txt_v = block_args["txt_q"], block_args["txt_k"], block_args["txt_v"]
        pre_attn_layout, post_attn_layout = MEMORY_LAYOUT["BNSD"]
        q_full = torch.cat([q_img_local, txt_q], dim=1).contiguous()
        b, _, n, d = q_full.shape
        scale = softmax_scale if softmax_scale is not None else 1.0 / math.sqrt(d)
        q1 = pre_attn_layout(q_full).contiguous()
        k_local = pre_attn_layout(k_img_local).contiguous()
        v_local = pre_attn_layout(v_img_local).contiguous()
        txt_k_bnsd = pre_attn_layout(txt_k).contiguous()
        txt_v_bnsd = pre_attn_layout(txt_v).contiguous()
        world_size = int(runtime_attn.ring_world_size)
        rank = int(runtime_attn.ring_rank)
        # overlap 版本先算 local KV sparse，同时异步聚合其它 rank 的 KV，最后用 LSE 合并两路输出。
        local_sink_len = self._get_ring_local_sink_token_len(
            local_img_token_len=int(q_img_local.shape[1]),
            ring_rank=rank,
            ring_world_size=world_size,
        )
        # Ring overlap 先用本地 KV 计算 sparse attention，同时异步聚合其他 rank 的 KV。

        k_gathered = torch.empty(
            (runtime_attn.ring_world_size, *k_img_local.shape),
            dtype=k_img_local.dtype,
            device=k_img_local.device,
        )
        v_gathered = torch.empty(
            (runtime_attn.ring_world_size, *v_img_local.shape),
            dtype=v_img_local.dtype,
            device=v_img_local.device,
        )
        k_handle = dist.all_gather_into_tensor(
            k_gathered,
            k_img_local.contiguous(),
            group=runtime_attn.ring_pg,
            async_op=True,
        )
        v_handle = dist.all_gather_into_tensor(
            v_gathered,
            v_img_local.contiguous(),
            group=runtime_attn.ring_pg,
            async_op=True,
        )

        out_local, lse_local = self._run_ring_topk_sparse_part(
            q1,
            k_local,
            v_local,
            batch_size=b,
            num_heads=n,
            scale=scale,
            dense_prefix_len=local_sink_len,
            dense_suffix_len=0,
        )
        k_handle.wait()
        v_handle.wait()

        k_full = self._ring_gathered_img_to_bnsd(pre_attn_layout, k_gathered, txt_k_bnsd)
        v_full = self._ring_gathered_img_to_bnsd(pre_attn_layout, v_gathered, txt_v_bnsd)
        other_indices = torch.tensor(
            [idx for idx in range(world_size) if idx != rank],
            dtype=torch.long,
            device=q1.device,
        )
        k_other = self._ring_gathered_img_to_bnsd(pre_attn_layout, k_gathered, txt_k_bnsd, other_indices)
        v_other = self._ring_gathered_img_to_bnsd(pre_attn_layout, v_gathered, txt_v_bnsd, other_indices)
        out_other, lse_other = self._run_ring_topk_sparse_part(
            q1,
            k_other,
            v_other,
            batch_size=b,
            num_heads=n,
            scale=scale,
            dense_prefix_len=local_sink_len,
            dense_suffix_len=int(txt_q.shape[1]),
        )
        sparse_out = self._merge_two_sparse_outputs(out_local, lse_local, out_other, lse_other)

        sparse_out = self._replace_prefix_with_dense_attention(
            sparse_out,
            q1,
            k_full,
            v_full,
            prefix_len=local_sink_len,
            num_heads=n,
            scale=scale,
        )
        return post_attn_layout(sparse_out).contiguous()

    def forward_ring_sparse(
        self,
        runtime_attn,
        block_args: dict,
        softmax_scale: Optional[float] = None,
    ) -> torch.Tensor:
        if self.ring_sparse_overlap:
            return self._forward_ring_topk_overlap(runtime_attn, block_args, softmax_scale)
        return self._forward_ring_topk_global(runtime_attn, block_args, softmax_scale)

    def _get_local_seq_remap_indices(
        self,
        device: torch.device,
        *,
        ulysses_world_size: int,
        reverse: bool = False,
    ) -> torch.Tensor:
        cache = getattr(self, "_seq_remap_index_cache", None)
        if cache is None:
            cache = {}
            self._seq_remap_index_cache = cache
        frame_num = int(getattr(self, "frame_num", 0))
        frame_patch_h = int(getattr(self, "frame_patch_h", 0))
        frame_patch_w = int(getattr(self, "frame_patch_w", 0))
        img_token_len = int(getattr(self, "img_token_len", 0))
        cache_key = (device, ulysses_world_size, frame_num, frame_patch_h, frame_patch_w)
        cached = cache.get(cache_key)
        if cached is None:
            cpu = torch.device("cpu")
            base = frame_patch_w // ulysses_world_size
            extra = frame_patch_w % ulysses_world_size
            width_splits = [base + (1 if rank < extra else 0) for rank in range(ulysses_world_size)]
            token_splits = [frame_num * frame_patch_h * w for w in width_splits]
            idx_current = torch.arange(img_token_len, dtype=torch.long, device=cpu).view(1, 1, img_token_len, 1)
            idx_split = torch.split(idx_current, token_splits, dim=2)
            idx_split = [
                chunk.view(1, 1, frame_num, frame_patch_h, width_splits[idx], 1)
                for idx, chunk in enumerate(idx_split)
            ]
            idx_canonical = torch.cat(idx_split, dim=4).reshape(1, 1, img_token_len, 1).contiguous()
            cur2can_cpu = idx_canonical.view(-1).long()
            can2cur_cpu = torch.empty_like(cur2can_cpu)
            can2cur_cpu.index_copy_(0, cur2can_cpu, torch.arange(img_token_len, dtype=torch.long, device=cpu))
            cached = (cur2can_cpu.to(device=device), can2cur_cpu.to(device=device))
            cache[cache_key] = cached
        return cached[1] if reverse else cached[0]

    def get_effective_indices(self) -> Tuple[int, int]:
        """获取当前有效的(step, layer)索引"""
        # 对于无时序模型（如VGGT），始终返回step=0
        effective_step = self.step if self.total_steps > 1 else 0
        return effective_step, self.layer_counter

    def update_layer_counter(self):
        self.layer_counter += 1
        if self.layer_counter >= self.total_layers_per_step:
            self.step += 1
            if self.step >= self.total_steps:
                self.step = 0
            self.layer_counter = 0
        self.current["step"] = self.step
        self.current["layer"] = self.layer_counter
        return self.get_effective_indices()

    def _apply_local_seq_remap_tensor(
        self,
        x: torch.Tensor,
        *,
        ulysses_world_size: int,
        reverse: bool = False,
    ) -> torch.Tensor:
        remap_index = self._get_local_seq_remap_indices(
            x.device,
            ulysses_world_size=ulysses_world_size,
            reverse=reverse,
        )
        img_token_len = int(self.img_token_len)
        if x.shape[2] < img_token_len:
            raise ValueError(
                f"Ulysses seq len ({x.shape[2]}) is smaller than img_token_len ({img_token_len})."
            )
        x_img = x[:, :, :img_token_len, :].index_select(2, remap_index)
        x_ctx = x[:, :, img_token_len:, :]
        return torch.cat((x_img, x_ctx), dim=2).contiguous() if x_ctx.shape[2] > 0 else x_img.contiguous()

    def forward_ulysses_sparse(
        self,
        runtime_attn,
        block_args: dict,
        softmax_scale: Optional[float] = None,
    ) -> torch.Tensor:
        q_img_local, k_img_local, v_img_local = (
            block_args["q_img_local"], block_args["k_img_local"], block_args["v_img_local"]
        )
        txt_q, txt_k, txt_v = block_args["txt_q"], block_args["txt_k"], block_args["txt_v"]
        route_head_perm = route_inv_head_perm = None
        if (
            getattr(self, "ulysses_head_routing_enabled", False)
            and runtime_attn.ulysses_world_size > 1
            and hasattr(self, "get_runtime_head_route")
        ):
            route_head_perm, route_inv_head_perm = self.get_runtime_head_route()
            if route_head_perm is not None:
                q_local_full = self._route_heads_bsnd(
                    torch.cat([q_img_local, txt_q], dim=1).contiguous(),
                    route_head_perm,
                )
                k_local_full = self._route_heads_bsnd(
                    torch.cat([k_img_local, txt_k], dim=1).contiguous(),
                    route_head_perm,
                )
                v_local_full = self._route_heads_bsnd(
                    torch.cat([v_img_local, txt_v], dim=1).contiguous(),
                    route_head_perm,
                )
                img_q_len_local = q_img_local.shape[1]
                img_kv_len_local = k_img_local.shape[1]
                split_args = self._split_img_txt_qkv(
                    q_local_full,
                    k_local_full,
                    v_local_full,
                    img_q_len_local,
                    img_kv_len_local,
                )
                q_img_local, k_img_local, v_img_local = split_args["q_img"], split_args["k_img"], split_args["v_img"]
                txt_q, txt_k, txt_v = split_args["txt_q"], split_args["txt_k"], split_args["txt_v"]

        q_img_global, k_img_global, v_img_global = self._ulysses_all_to_all_qkv_triplet(
            runtime_attn,
            q_img_local,
            k_img_local,
            v_img_local,
        )

        txt_q_local_h, txt_k_local_h, txt_v_local_h = self._slice_ulysses_local_heads_with_uaa(
            runtime_attn,
            txt_q, txt_k, txt_v, q_img_local.shape[2]
        )
        predictor_name = str(getattr(self, "predictor_name", ""))
        is_svg_mode = predictor_name.lower() == "svg" or "svg" in self.__class__.__name__.lower()
        attn_kwargs = {}
        img_seq_len = int(q_img_global.shape[1])
        if is_svg_mode:
            q_full = torch.cat([q_img_global, txt_q_local_h], dim=1).contiguous()
            k_full = torch.cat([k_img_global, txt_k_local_h], dim=1).contiguous()
            v_full = torch.cat([v_img_global, txt_v_local_h], dim=1).contiguous()
            attn_kwargs["joint_q_local_bnsd"] = txt_q_local_h
        else:
            h_local = q_img_global.shape[2]
            img_packed = torch.cat([q_img_global, k_img_global, v_img_global], dim=2)
            txt_packed = torch.cat([txt_q_local_h, txt_k_local_h, txt_v_local_h], dim=2)
            qkv_full = torch.cat([img_packed, txt_packed], dim=1).contiguous()
            q_full, k_full, v_full = torch.split(qkv_full, [h_local, h_local, h_local], dim=2)
        head_sabi = None
        prefix_q_len = int(q_full.shape[1])
        prefix_kv_len = int(k_full.shape[1])
        out_full = self.attention(
            q=q_full,
            k=k_full,
            v=v_full,
            head_sabi=head_sabi,
            ulysses_pg=runtime_attn.ulysses_pg,
            ulysses_rank=runtime_attn.ulysses_rank,
            ulysses_world_size=runtime_attn.ulysses_world_size,
            cu_seqlens_q=[0, prefix_q_len],
            cu_seqlens_kv=[0, prefix_kv_len],
            return_bshd=True,
            softmax_scale=softmax_scale,
            **attn_kwargs,
        )

        img_out_global = out_full[:, :img_seq_len, :, :].contiguous()
        txt_out_local_h = out_full[:, img_seq_len:, :, :].contiguous()
        if runtime_attn.ulysses_anything:
            img_out_local = self._ulysses_all_to_all_o(
                runtime_attn,
                img_out_global,
                num_qo_head=q_img_local.shape[2],
                q_s_local=q_img_local.shape[1],
            )
            txt_out = all_gather_anything(
                tensor=txt_out_local_h,
                dim=2,
                world_size=runtime_attn.ulysses_world_size,
                group=runtime_attn.ulysses_pg,
            ).contiguous()
        else:
            img_out_local = self._ulysses_all_to_all_o(
                runtime_attn,
                img_out_global,
                num_qo_head=q_img_local.shape[2],
                q_s_local=q_img_local.shape[1],
            )
            txt_out = self._ulysses_all_gather_heads_bshd(runtime_attn, txt_out_local_h)

        out = torch.cat([img_out_local, txt_out], dim=1).contiguous()
        expected_local_seq_len = int(q_img_local.shape[1]) + int(txt_q.shape[1])
        if out.shape[1] != expected_local_seq_len:
            img_expected = int(q_img_local.shape[1])
            txt_expected = int(txt_q.shape[1])
            img_actual = int(img_out_local.shape[1])
            txt_actual = int(txt_out.shape[1])
            if img_actual >= img_expected:
                img_out_local = img_out_local[:, :img_expected, :, :].contiguous()
            if txt_actual >= txt_expected:
                txt_out = txt_out[:, :txt_expected, :, :].contiguous()
            out = torch.cat([img_out_local, txt_out], dim=1).contiguous()
            if int(out.shape[1]) > expected_local_seq_len:
                out = out[:, :expected_local_seq_len, :, :].contiguous()

        if route_inv_head_perm is not None:
            out = self._route_heads_bsnd(out, route_inv_head_perm)
        return out

    def load_sparsity(self, path: Union[str, Path], step_pattern: str = "step-{}.pt", step_num: int = 50):
        path = Path(path)
        if path.is_file():
            self.load_single_file(path)
        elif path.is_dir():
            self.load_directory(path, step_pattern, step_num)
        else:
            raise FileNotFoundError(f"路径不存在: {path}")

    def load_single_file(self, file_path: Path):
        self.sparsity_dict[0] = torch.load(file_path, map_location=self.device)

    def load_directory(self, file_path: Path, step_pattern: str, step_num: int):
        for i in range(step_num):
            step_dir_path = os.path.join(file_path, f"step-{i}/")
            sparsity_pt_path = os.path.join(step_dir_path, f"sparsity_of_RE_{self.cac_threshold}_only_img.pt")
            sparsity_per_step = torch.load(sparsity_pt_path, map_location=self.device)
            self.sparsity_dict[i] = sparsity_per_step

    @staticmethod
    def padding_sabi(selected_indices_tensor: torch.Tensor, max_width: int, pad_value: int = -1):
        padded_selected_indices_tensor = [F.pad(t, (0, max_width - t.shape[1]), value=pad_value) 
                                            for t in selected_indices_tensor]
        result = torch.stack(padded_selected_indices_tensor, dim=0)
        return result

    
    def get_block_mask(self, q, sabi):
        b, h, n, _ = q.shape

        block_num_q = math.ceil(n / self.block_size_q)
        block_num_k = math.ceil(n / self.block_size_k)
        block_mask = torch.full((b, h, block_num_q, block_num_k), True, device=q.device, dtype=torch.bool)
        valid_mask = sabi > -1
        b_valid, h_valid, r_valid, _ = torch.where(valid_mask)
        col_indices = sabi[valid_mask].long()
        block_mask[b_valid, h_valid, r_valid, col_indices] = False
        return block_mask

    
    def get_token_mask(self, block_mask, n):
        mask = block_mask[:, :, :, None, :, None]
        b, h, bq, bk = block_mask.shape
        mask = mask.expand(b, h, bq, self.block_size_q, bk, self.block_size_k)
        mask = mask.reshape(b, h, bq * self.block_size_q, bk * self.block_size_k)
        return mask[:, :, :n, :n]

    def get_token_level_sparisty(self, token_mask):
        sparisty = token_mask.to(torch.float32).mean().item()
        with open(self.runtime_sparisty_path, 'a') as f:
            f.write(f"{sparisty} \n")

    def get_sparse_token_mask(self, q: torch.Tensor, k: torch.Tensor, final_sabi: torch.Tensor):
        block_mask = self.get_block_mask(q, final_sabi)
        token_mask = self.get_token_mask(block_mask, q.shape[2])
        return token_mask

    @abstractmethod
    def get_sabi(self, q: torch.Tensor, k: torch.Tensor):
        pass

    def get_must_keep_blocks_indices(self, **kwargs):
        return None

    def combined_sabi_tensor_list(self, mid_sabi_list, must_keep_indices_q, must_keep_indices_k, num_blocks_k):
        pass
    
    def combined_sabi_tensor(self, mid_sabi_list, must_keep_indices_q, must_keep_indices_k, num_blocks_k):
        return None

    


class TopKPredictor(BaseSparsePredictor):
    def __init__(self, sparse_config, sparse_params=None):
        super().__init__(sparse_config, sparse_params)
        topk_config = sparse_config['TopK'] # [uncond, cond]
        logger.info(topk_config)
        self.predictor_name = topk_config['predictor_name']

        self.sparse_time_step = topk_config['sparse_time_step']
        self.sparsity_files_path = topk_config['sparsity_files_path']
        self.cac_threshold = topk_config['CAC_threshold']
        self.ring_sparse_overlap = bool(topk_config.get("ring_sparse_overlap", False))

        self.sparsity_dict = {}
        self.load_sparsity(self.sparsity_files_path, self.total_steps)
        sample_step = next(iter(self.sparsity_dict.keys()))
        sample_layer = self.sparsity_dict[sample_step][0]
        self.head_num = int(torch.as_tensor(sample_layer).numel())
        self.ulysses_degree_for_lb = 1
        self.ulysses_head_routing_enabled = False
        self.step_layer_head_perm = {}
        self.step_layer_inv_head_perm = {}
        self._ulysses_head_split_cache = {}

    def _get_layer_sparsity_tensor(self, step_idx: int, layer_idx: int, device=None) -> torch.Tensor:
        raw_sparsity = self.sparsity_dict[step_idx][layer_idx]
        sparsity_cpu = torch.as_tensor(raw_sparsity, dtype=torch.float32, device="cpu").contiguous()
        if device is None:
            return sparsity_cpu
        if isinstance(device, torch.device) and device.type == "cpu":
            return sparsity_cpu
        if isinstance(device, str) and device.lower().startswith("cpu"):
            return sparsity_cpu
        return torch.tensor(sparsity_cpu.tolist(), dtype=torch.float32, device=device).contiguous()

    def _get_layer_sparsity_runtime_head_order(self, step_idx: int, layer_idx: int, device=None) -> torch.Tensor:
        sparsity = self._get_layer_sparsity_tensor(step_idx, layer_idx, device=device)
        if not self.ulysses_head_routing_enabled:
            return sparsity
        perm = self.step_layer_head_perm.get((int(step_idx), int(layer_idx)))
        if perm is None:
            return sparsity
        if not torch.is_tensor(perm):
            perm = torch.as_tensor(perm, dtype=torch.long)
        perm = perm.to(device=sparsity.device, dtype=torch.long)
        return sparsity.index_select(0, perm).contiguous()

    def apply_head_reorder_for_load_balance(self, ulysses_degree: int):
        self.ulysses_degree_for_lb = int(ulysses_degree)
        self.step_layer_head_perm.clear()
        self.step_layer_inv_head_perm.clear()
        if self.ulysses_degree_for_lb <= 1:
            self.ulysses_head_routing_enabled = False
            return
        for step_idx, layer_sparsity_list in self.sparsity_dict.items():
            total_layers = len(layer_sparsity_list)
            for layer_idx in range(total_layers):
                sparsity = self._get_layer_sparsity_tensor(int(step_idx), int(layer_idx), device="cpu")
                sparsity_list = sparsity.tolist()
                rank_buckets = [[] for _ in range(self.ulysses_degree_for_lb)]
                rank_costs = [0.0 for _ in range(self.ulysses_degree_for_lb)]
                ranked_heads = sorted(
                    range(len(sparsity_list)),
                    key=lambda idx: (1.0 - float(sparsity_list[idx])),
                    reverse=True,
                )
                for head_idx in ranked_heads:
                    target_rank = min(
                        range(self.ulysses_degree_for_lb),
                        key=lambda rank: (rank_costs[rank], len(rank_buckets[rank]), rank),
                    )
                    rank_buckets[target_rank].append(head_idx)
                    rank_costs[target_rank] += 1.0 - float(sparsity_list[head_idx])
                perm_list = [head_idx for bucket in rank_buckets for head_idx in bucket]
                perm = torch.tensor(perm_list, dtype=torch.long)
                inv_list = [0] * len(perm_list)
                for new_idx, old_idx in enumerate(perm_list):
                    inv_list[old_idx] = new_idx
                inv = torch.tensor(inv_list, dtype=torch.long)
                self.step_layer_head_perm[(int(step_idx), int(layer_idx))] = perm
                self.step_layer_inv_head_perm[(int(step_idx), int(layer_idx))] = inv
        self.ulysses_head_routing_enabled = True

    def get_runtime_head_route(self):
        step_idx, layer_idx = self.get_effective_indices()
        return (
            self.step_layer_head_perm.get((int(step_idx), int(layer_idx))),
            self.step_layer_inv_head_perm.get((int(step_idx), int(layer_idx))),
        )

    
    def get_block_attn(self, q: torch.Tensor, k, block_num_q, block_num_k):
        _, _, n_q, _ = q.shape
        n_k = k.shape[-2]
        q_list = []
        k_list = []
        for i in range(block_num_q):
            start = i * self.block_size_q
            end = min((i + 1) * self.block_size_q, n_q)
            q_i = q[:, :, start: end, :]
            q_i = torch.mean(q_i, dim=2, keepdim=True)
            q_list.append(q_i)
        for i in range(block_num_k):
            start = i * self.block_size_k
            end = min((i + 1) * self.block_size_k, n_k)
            k_i = k[:, :, start: end, :]
            k_i = torch.mean(k_i, dim=2, keepdim=True)
            k_list.append(k_i)
        q_mean = torch.cat(q_list, dim=2).to(q.device)
        k_mean = torch.cat(k_list, dim=2).to(k.device)
        attn = q_mean @ k_mean.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        return attn, q_list, k_list

    def pooling_matmul(self, q, block_size, block_num):
        b, h, n, d = q.shape
        if block_num <= 0:
            return q.new_zeros((b, h, 0, d))

        full_len = int(block_num) * int(block_size)
        if n < full_len:
            q = F.pad(q, (0, 0, 0, full_len - n))
        elif n > full_len:
            q = q[:, :, :full_len, :].contiguous()

        q_blocks = q.reshape(b, h, int(block_num), int(block_size), d)
        q_sum = q_blocks.sum(dim=3)
        valid_counts = torch.full((int(block_num),), int(block_size), device=q.device, dtype=q.dtype)
        tail = n - (int(block_num) - 1) * int(block_size)
        if tail < int(block_size):
            valid_counts[-1] = max(tail, 1)
        return (q_sum / valid_counts.view(1, 1, int(block_num), 1)).contiguous()

    def get_block_attn_by_matmul_v2(self, q, k, block_num_q, block_num_k):
        q_mean = self.pooling_matmul(q, self.block_size_q, block_num_q)
        k_mean = self.pooling_matmul(k, self.block_size_k, block_num_k)
        attn = q_mean @ k_mean.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        return attn

    def get_sabi(self, q: torch.Tensor, k: torch.Tensor, sparsity_override: Optional[torch.Tensor] = None):
        b, h, n_q, _ = q.shape
        n_k = k.shape[-2]
        block_num_q = math.ceil(n_q / self.block_size_q)
        block_num_k = math.ceil(n_k / self.block_size_k)
        attn, _, _ = self.get_block_attn(q, k, block_num_q, block_num_k)
        new_sabi_list = []
        if sparsity_override is None:
            step_idx, layer_idx = self.get_effective_indices()
            sparsity = self._get_layer_sparsity_runtime_head_order(step_idx, layer_idx, device=attn.device)
        else:
            sparsity = sparsity_override.to(device=attn.device, dtype=torch.float32).contiguous()
        for batch in range(b):
            for head in range(h):
                k_num = int((1 - sparsity[head]) * block_num_k)
                attn_bh = attn[batch, head]
                _, indices = torch.topk(attn_bh, k_num, dim=-1)
                new_sabi_list.append(indices)
        return new_sabi_list

    def get_sabi_v2(self, q: torch.Tensor, k: torch.Tensor, sparsity_override: Optional[torch.Tensor] = None):
        b, h, n_q, _ = q.shape
        n_k = k.shape[-2]
        block_num_q = math.ceil(n_q / self.block_size_q)
        block_num_k = math.ceil(n_k / self.block_size_k)
        attn = self.get_block_attn_by_matmul_v2(q, k, block_num_q, block_num_k)

        new_sabi_tensor = torch.full(size=(b, h, block_num_q, block_num_k),
                                        fill_value=-1, dtype=self.index_type, device=q.device)
        if sparsity_override is None:
            step_idx, layer_idx = self.get_effective_indices()
            sparsity = self._get_layer_sparsity_runtime_head_order(step_idx, layer_idx, device=attn.device)
        else:
            sparsity = sparsity_override.to(device=attn.device, dtype=torch.float32).contiguous()
        k_nums = ((1.0 - sparsity) * block_num_k).to(torch.int32)
        k_nums = torch.clamp(k_nums, min=1, max=block_num_k)
        max_k = int(k_nums.max().item())
        _, indices = torch.topk(attn, max_k, dim=-1)
        
        k_nums_expanded = k_nums.view(1, h, 1, 1)
        arange_k = torch.arange(max_k, device=q.device).view(1, 1, 1, max_k)
        mask = arange_k < k_nums_expanded
        new_sabi_tensor[:, :, :, :max_k] = torch.where(mask, indices, -1)

        return new_sabi_tensor
 
    def get_final_sabi(self, q: torch.Tensor, k: torch.Tensor, **kwargs):
        '''sabi连接must_keep的block indices，得到final_sabi'''
        sink_frame_len = kwargs["sink_frame_len"]
        img_token_len = kwargs["img_token_len"]
        sparsity_override = kwargs.get("sparsity_override")
        all_token_len = q.shape[2]
        txt_token_len = all_token_len - img_token_len
        n_q = q.shape[2]
        n_k = k.shape[-2]
        block_num_q = math.ceil(n_q / self.block_size_q)
        block_num_k = math.ceil(n_k / self.block_size_k)

        sink_txt_blocks_q, must_keep_indices_q, sink_txt_blocks_k, must_keep_indices_k = \
            self.get_must_keep_blocks_indices(token_len=all_token_len, 
                                              sink_frame_len=sink_frame_len, 
                                              txt_token_len=txt_token_len)

        mid_q, mid_k = q[:, :, : (block_num_q - sink_txt_blocks_q) * self.block_size_q, :],\
            k[:, :, : (block_num_k - sink_txt_blocks_k) * self.block_size_k, :]
        
        mid_sabi_list = self.get_sabi(mid_q, mid_k, sparsity_override=sparsity_override)
        final_sabi = self.combined_sabi_tensor_list(mid_sabi_list, must_keep_indices_q, 
                        must_keep_indices_k, block_num_k).unsqueeze(0)
        return final_sabi

    def get_final_sabi_v2(self, q: torch.Tensor, k: torch.Tensor, **kwargs):
        '''sabi连接must_keep的block indices，得到final_sabi'''
        sink_frame_len = kwargs["sink_frame_len"]
        img_token_len = kwargs["img_token_len"]
        img_token_len_q = int(kwargs.get("img_token_len_q", img_token_len))
        img_token_len_k = int(kwargs.get("img_token_len_k", img_token_len))
        sink_frame_len_q = int(kwargs.get("sink_frame_len_q", sink_frame_len))
        sink_frame_len_k = int(kwargs.get("sink_frame_len_k", sink_frame_len))
        sparsity_override = kwargs.get("sparsity_override")
        n_q = q.shape[2]
        n_k = k.shape[-2]
        txt_token_len_q = max(n_q - img_token_len_q, 0)
        txt_token_len_k = max(n_k - img_token_len_k, 0)
        block_num_q = math.ceil(n_q / self.block_size_q)
        block_num_k = math.ceil(n_k / self.block_size_k)

        sink_txt_blocks_q, must_keep_indices_q, sink_txt_blocks_k, must_keep_indices_k = \
            self.get_must_keep_blocks_indices(
                token_len_q=n_q,
                token_len_k=n_k,
                sink_frame_len=sink_frame_len,
                sink_frame_len_q=sink_frame_len_q,
                sink_frame_len_k=sink_frame_len_k,
                txt_token_len=txt_token_len_q,
                txt_token_len_q=txt_token_len_q,
                txt_token_len_k=txt_token_len_k,
            )

        mid_q, mid_k = q[:, :, : (block_num_q - sink_txt_blocks_q) * self.block_size_q, :],\
            k[:, :, : (block_num_k - sink_txt_blocks_k) * self.block_size_k, :]
        
        mid_q = mid_q.contiguous()
        mid_k = mid_k.contiguous()

        mid_sabi_tensor = self.get_sabi_v2(mid_q, mid_k, sparsity_override=sparsity_override)
        final_sabi = self.combined_sabi_tensor(mid_sabi_tensor, must_keep_indices_q, must_keep_indices_k, block_num_k)
        return final_sabi

MEMORY_LAYOUT = {
    "TND": (
        lambda x: x.view(x.shape[0] * x.shape[1], *x.shape[2:]), 
        lambda x: x,
    ),
    "BNSD": (
        lambda x: x.transpose(1, 2),
        lambda x: x.transpose(1, 2),
    ),
    "BSND": (
        lambda x: x,
        lambda x: x,
    ),
}


def get_cu_seqlens(text_mask, img_len):
    """Calculate cu_seqlens_q, cu_seqlens_kv using text_mask and img_len

    Args:
        text_mask (torch.Tensor): the mask of text
        img_len (int): the length of image

    Returns:
        torch.Tensor: the calculated cu_seqlens for flash attention
    """
    batch_size = text_mask.shape[0]
    text_len = text_mask.sum(dim=1)
    max_len = text_mask.shape[1] + img_len

    cu_seqlens = torch.zeros([2 * batch_size + 1], dtype=torch.int32, device="cuda")

    for i in range(batch_size):
        s = text_len[i] + img_len
        s1 = i * max_len + s
        s2 = (i + 1) * max_len
        cu_seqlens[2 * i + 1] = s1
        cu_seqlens[2 * i + 2] = s2

    return cu_seqlens


def get_row_indices(block):
    indices_list = []
    for row in block:
        row_false_indices = torch.where(~row)[0]
        indices_list.append(row_false_indices)
    result = torch.full((len(indices_list), block.shape[1]), -1, dtype=torch.long)
    for i, indices in enumerate(indices_list):
        if len(indices) > 0:
            result[i, :len(indices)] = indices
    return result


def block2sabi(block_mask):
    sabi_mask = get_row_indices(block_mask)

    return sabi_mask


class SVGPredictor(BaseSparsePredictor):
    def __init__(self, sparse_config, sparse_params=None):
        super().__init__(sparse_config, sparse_params)
        svg_config = sparse_config['SVG']
        logger.info(svg_config)
        
        self.sparse_time_step = svg_config['sparse_time_step']
        self.sparsity = svg_config['sparsity']
        self.context_length = svg_config['context_length']
        self.sample_mse_max_row = svg_config['sample_mse_max_row']
        self.attention_masks = []
        self.sabi_tensor = None
        self._svg_mask_cache_key = None

    def sparsity_to_width(self, sparsity, num_frame, frame_size):
        seq_len = self.context_length + num_frame * frame_size

        width = seq_len * (1 - math.sqrt(sparsity)) - self.context_length
        width_frame = width / frame_size

        return width_frame
    
    def sample_mse(self, query, key, value, context_length):
        if context_length > 0:
            key = key[:, :, :-context_length]
            value = value[:, :, :-context_length]

        mask_name = ["spatial", "temporal"]
        num_sampled_rows = 64

        _, _, seq_len, dim = query.size()
        num_sampled_rows = min(num_sampled_rows, seq_len)
        sampled_row_high = min(seq_len, self.sample_mse_max_row)
        sampled_rows = torch.randint(low=0, high=sampled_row_high, size=(num_sampled_rows,), device=query.device)
        sampled_q = query[:, :, sampled_rows, :]
        sampled_qk_scores = torch.matmul(sampled_q, key.transpose(-2, -1)) / (dim**0.5)

        sampled_attn_weights = F.softmax(sampled_qk_scores, dim=-1)
        sampled_golden_hidden_states = torch.matmul(sampled_attn_weights, value)  # (1, seq_len, dim)

        sampled_mses = {}

        # Only have Tri-diagonal and Striped
        for mask_idx, attn_mask in enumerate(self.attention_masks):
            sampled_attention_mask = attn_mask[sampled_rows, :-context_length]
            sampled_attention_scores = sampled_qk_scores.masked_fill(sampled_attention_mask == 0, float("-inf"))
            sampled_attn_weights = F.softmax(sampled_attention_scores, dim=-1)
            sampled_hidden_states = torch.matmul(sampled_attn_weights, value)
            mse = torch.mean((sampled_hidden_states - sampled_golden_hidden_states) ** 2, dim=(2, 3))
            
            sampled_mses[mask_name[mask_idx]] = mse
        del sampled_attention_mask, sampled_attention_scores, sampled_attn_weights, sampled_hidden_states
        return sampled_mses


    def get_attention_mask(self, mask_name, s, num_frame, frame_size, width_frame=1.5, block_size=512, device="NPU"):
        block_size_q, block_size_k = self.block_size_q, self.block_size_k

        if block_size % block_size_q != 0 or block_size % block_size_k != 0:
            raise ValueError("block_size must be divisible")
        
        q_num_per_block = block_size // block_size_q
        k_num_per_block = block_size // block_size_k

        context_length = s - num_frame * frame_size

        attention_mask = torch.zeros((s, s), device="cpu")

        num_block_q = math.ceil(s / block_size_q)
        num_block_k = math.ceil(s / block_size_k)

        pixel_attn_mask = torch.zeros_like(
                attention_mask[:-context_length, :-context_length], dtype=torch.bool, device="cpu"
        )
        block_thres = frame_size * width_frame
        num_block = math.ceil(num_frame * frame_size / block_size)
        block_mask = torch.full((num_block_q, num_block_k), True).to(device)
      
        if mask_name == "spatial":
            for i in range(num_block):
                for j in range(num_block):
                    if abs(i - j) < block_thres // block_size:
                        row_start = i * block_size
                        row_end = (i + 1) * block_size
                        col_start = j * block_size
                        col_end = (j + 1) * block_size

                        pixel_attn_mask[row_start: row_end, col_start: col_end] = 1

                        block_row_start = i * q_num_per_block
                        block_row_end = (i + 1) * q_num_per_block
                        block_col_start = j * k_num_per_block
                        block_col_end = (j + 1) * k_num_per_block

                        block_mask[block_row_start: block_row_end, block_col_start: block_col_end] = False

            attention_mask[:-context_length, :-context_length] = pixel_attn_mask

            attention_mask[-context_length:, :] = 1
            attention_mask[:, -context_length:] = 1

            context_blocks_q = math.ceil((context_length) / block_size_q)
            context_blocks_k = math.ceil((context_length) / block_size_k)
            block_mask[-context_blocks_q:, :] = False
            block_mask[:, -context_blocks_k:] = False
        else:
            for i in range(num_block):
                for j in range(num_block):
                    if abs(i - j) < block_thres // block_size:
                        row_start = i * block_size
                        row_end = (i + 1) * block_size
                        col_start = j * block_size
                        col_end = (j + 1) * block_size

                        pixel_attn_mask[row_start: row_end, col_start: col_end] = 1

            pixel_attn_mask = (
                pixel_attn_mask.reshape(frame_size, num_frame, frame_size, num_frame)
                .permute(1, 0, 3, 2)
                .reshape(frame_size * num_frame, frame_size * num_frame)
            )
            attention_mask[:-context_length, :-context_length] = pixel_attn_mask

            attention_mask[-context_length:, :] = 1
            attention_mask[:, -context_length:] = 1
        attention_mask = attention_mask[:self.sample_mse_max_row].to(device)
        return attention_mask, block_mask
    
    def get_sabi(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, num_frames: int, frame_size: int):
        b, h, s, _ = q.shape
        
        device = q.device
        self.device = device
        frame_width = self.sparsity_to_width(self.sparsity, num_frames, frame_size)
        context_length_cu = s - num_frames * frame_size
        cache_key = (
            int(s),
            int(num_frames),
            int(frame_size),
            str(device),
            int(self.sample_mse_max_row),
        )

        if self._svg_mask_cache_key != cache_key:
            spatial_mask, block_mask = self.get_attention_mask("spatial", s, num_frames, 
                                                                frame_size, frame_width, device=device)
            temporal_mask, _ = self.get_attention_mask("temporal", s, num_frames,
                                                                frame_size, frame_width, device=device)

            self.attention_masks = [spatial_mask, temporal_mask]
            base_sabi_tensor = block2sabi(block_mask)
            block_num_q, block_num_k = base_sabi_tensor.shape[0], base_sabi_tensor.shape[1]
            self.sabi_tensor = base_sabi_tensor.unsqueeze(0).unsqueeze(0).expand(b, h, block_num_q, block_num_k)
            self._svg_mask_cache_key = cache_key
        elif self.sabi_tensor is None or self.sabi_tensor.shape[0] != b or self.sabi_tensor.shape[1] != h:
            base_sabi_tensor = self.sabi_tensor[0, 0]
            block_num_q, block_num_k = base_sabi_tensor.shape[0], base_sabi_tensor.shape[1]
            self.sabi_tensor = base_sabi_tensor.unsqueeze(0).unsqueeze(0).expand(b, h, block_num_q, block_num_k)
        
        mse_result = self.sample_mse(q, k, v, context_length_cu)
        pattern = (mse_result["spatial"] < mse_result["temporal"]).flatten()

        return pattern
    
 
    def get_final_sabi(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, num_frames: int, frame_size: int):
        '''sabi连接must_keep的block indices，得到final_sabi'''
        sink_frame_len = 0

        all_token_len = q.shape[2]
        txt_token_len = q.shape[2] - num_frames * frame_size
        self.get_must_keep_blocks_indices(
            token_len=all_token_len,
            sink_frame_len=sink_frame_len,
            txt_token_len=txt_token_len,
        )

        pattern = self.get_sabi(q, k, v, num_frames, frame_size)
        return pattern, self.sabi_tensor


class HunyuanVideoTopKAdapter(TopKPredictor):
    def __init__(self, sparse_config, sparse_params=None):
        super().__init__(sparse_config, sparse_params)
        self.update_sparse_params(sparse_params)
        # HyVideo特定参数
        
    def update_sparse_params(self, sparse_params):
        # HyVideo特定参数
        self.sink_frame_len = sparse_params.get("sink_frame_len", 0)
        self.img_token_len = sparse_params.get("img_token_len", 0)
        self.frame_num = sparse_params.get("frame_num", 0)
        self.frame_patch_h = sparse_params.get("frame_patch_h", 0)
        self.frame_patch_w = sparse_params.get("frame_patch_w", 0)
        logger.info(f"update sparse params successfully, sink_frame_len: {self.sink_frame_len}\
                    img_token_len: {self.img_token_len}. ")

    def get_must_keep_blocks_indices(self, **kwargs):
        token_len_q = kwargs.get("token_len_q", kwargs.get("token_len"))
        token_len_k = kwargs.get("token_len_k", kwargs.get("token_len"))
        sink_frame_len = kwargs.get("sink_frame_len", 0)
        sink_frame_len_q = kwargs.get("sink_frame_len_q", sink_frame_len)
        sink_frame_len_k = kwargs.get("sink_frame_len_k", sink_frame_len)
        txt_len = kwargs.get("txt_token_len", 0)
        txt_len_q = kwargs.get("txt_token_len_q", txt_len)
        txt_len_k = kwargs.get("txt_token_len_k", txt_len)
        sink_txt_len_k = sink_frame_len_k + txt_len_k
        sink_txt_blocks_k = math.ceil(sink_txt_len_k / self.block_size_k)
        num_blocks_k = math.ceil(token_len_k / self.block_size_k)
        k_num_of_last_block = token_len_k % self.block_size_k
        multi_last_k_blocks_token_num = (sink_txt_blocks_k - 1) * self.block_size_k + k_num_of_last_block
        if multi_last_k_blocks_token_num < sink_txt_len_k:
            sink_txt_blocks_k += 1
        
        sink_txt_start_indices = num_blocks_k - sink_txt_blocks_k
        must_keep_indices_k = torch.cat([
            torch.arange(sink_txt_start_indices, num_blocks_k)
        ])

        sink_txt_len_q = sink_frame_len_q + txt_len_q
        sink_txt_blocks_q = math.ceil(sink_txt_len_q / self.block_size_q)
        num_blocks_q = math.ceil(token_len_q / self.block_size_q)
        q_num_of_last_block = token_len_q % self.block_size_q
        multi_last_q_blocks_token_num = (sink_txt_blocks_q - 1) * self.block_size_q + q_num_of_last_block
        if multi_last_q_blocks_token_num < sink_txt_len_q:
            sink_txt_blocks_q += 1
        
        sink_txt_start_indices = num_blocks_q - sink_txt_blocks_q
        must_keep_indices_q = torch.cat([
            torch.arange(sink_txt_start_indices, num_blocks_q)
        ])
        return sink_txt_blocks_q, must_keep_indices_q, sink_txt_blocks_k, must_keep_indices_k

    def combined_sabi_tensor_list(self, mid_sabi_list, must_keep_indices_q, must_keep_indices_k, num_blocks_k):
        sink_txt_suffix = must_keep_indices_k \
                            .unsqueeze(0) \
                            .expand(mid_sabi_list[0].size(0), -1)\
                            .to(mid_sabi_list[0].device)

        full_k_indices = torch.arange(num_blocks_k, device=mid_sabi_list[0].device) \
                            .view(1, 1, num_blocks_k) \
                            .expand(len(mid_sabi_list), must_keep_indices_q.shape[0], num_blocks_k)
        padded_sabi_list = []
        for mid_sabi in mid_sabi_list:
            padded_sabi = torch.cat((mid_sabi, sink_txt_suffix[:, :]), dim=1)
            padded_sabi_list.append(padded_sabi)
        padded_sabi_tensor = self.padding_sabi(padded_sabi_list, max_width=num_blocks_k)
        padded_sabi_tensor = torch.cat((padded_sabi_tensor, full_k_indices[:, :]), dim=1)
        return padded_sabi_tensor

    def move_sink_frame_to_end(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, sink_frame_len: int):
        return self._move_sink_qkv_to_end(q, k, v, q_sink_len=sink_frame_len, kv_sink_len=sink_frame_len)

    def move_sink_frame_back(self, attn_out: torch.Tensor, sink_frame_len: int):
        s_q = attn_out.shape[2]
        sink_attn_out = attn_out[:, :, s_q - sink_frame_len:, :]
        rest_attn_out = attn_out[:, :, :s_q - sink_frame_len, :]
        attn_out = torch.cat((sink_attn_out, rest_attn_out), dim=2)
        return attn_out

    def combined_sabi_tensor(self, mid_sabi_tensor, must_keep_indices_q, must_keep_indices_k, total_num_blocks_k):
        b, h, num_blocks_q, _ = mid_sabi_tensor.shape
        must_keep_k = must_keep_indices_k.shape[0]
        added_q_row_num = must_keep_indices_q.shape[0]
        sink_txt_suffix = must_keep_indices_k.view(1, 1, 1, must_keep_k) \
                            .expand(b, h, num_blocks_q, must_keep_k) \
                            .to(mid_sabi_tensor[0].device)
        full_k_indices = torch.arange(total_num_blocks_k, device=mid_sabi_tensor.device, dtype=self.index_type) \
                            .view(1, 1, 1, total_num_blocks_k) \
                            .expand(b, h, added_q_row_num, total_num_blocks_k)
        padded_sabi_tensor = torch.cat((sink_txt_suffix, mid_sabi_tensor), dim=3)[:, :, :, :total_num_blocks_k]
        padded_sabi_tensor = torch.cat((padded_sabi_tensor, full_k_indices), dim=2)
        return padded_sabi_tensor

    def _resolve_ulysses_local_sparsity_override(
        self,
        *,
        local_head_num: int,
        device: torch.device,
        ulysses_pg=None,
        ulysses_rank: int = 0,
        ulysses_world_size: int = 1,
    ) -> Optional[torch.Tensor]:
        if ulysses_world_size <= 1:
            return None

        step_idx, layer_idx = self.get_effective_indices()
        layer_sparsity = self._get_layer_sparsity_runtime_head_order(step_idx, layer_idx, device=device)
        if int(layer_sparsity.numel()) == local_head_num:
            return layer_sparsity

        head_start = ulysses_rank * local_head_num
        head_end = head_start + local_head_num
        if ulysses_pg is not None and dist.is_initialized():
            cache_key = (
                int(ulysses_world_size),
                int(ulysses_rank),
                int(local_head_num),
                str(device),
                int(layer_sparsity.numel()),
            )
            cached = self._ulysses_head_split_cache.get(cache_key)
            if cached is not None:
                head_start, head_end = cached
            else:
                local_h = torch.tensor([local_head_num], dtype=torch.int64, device=device)
                gathered_h = torch.empty((ulysses_world_size, 1), dtype=torch.int64, device=device)
                dist.all_gather_into_tensor(gathered_h, local_h, group=ulysses_pg)
                head_splits = gathered_h.view(-1).to(device="cpu", dtype=torch.long).tolist()
                head_start = int(sum(head_splits[:ulysses_rank]))
                head_end = head_start + int(head_splits[ulysses_rank])
                self._ulysses_head_split_cache[cache_key] = (head_start, head_end)
        return layer_sparsity[head_start:head_end].contiguous()

    def attention(self, q,
            k,
            v,
            drop_rate=0,
            attn_mask=None,
            causal=False,
            cu_seqlens_q=None,
            cu_seqlens_kv=None,
            max_seqlen_q=None,
            max_seqlen_kv=None,
            batch_size=1,
            head_sabi: Optional[torch.Tensor] = None,
            ulysses_pg=None,
            ulysses_rank: int = 0,
            ulysses_world_size: int = 1,
            return_bshd: bool = False,
            softmax_scale: Optional[float] = None,
            img_token_len_q: Optional[int] = None,
            img_token_len_k: Optional[int] = None,
            sink_frame_len_q: Optional[int] = None,
            sink_frame_len_k: Optional[int] = None,):
        pre_attn_layout, post_attn_layout = MEMORY_LAYOUT["BNSD"]
        b, s, n, d = q.shape
        q, k, v = pre_attn_layout(q), pre_attn_layout(k), pre_attn_layout(v)
        scale = softmax_scale if softmax_scale is not None else 1.0 / math.sqrt(d)
        if cu_seqlens_q is None or cu_seqlens_kv is None:
            if cu_seqlens_q is not None or cu_seqlens_kv is not None:
                raise ValueError("TopK sparse attention requires both cu_seqlens_q and cu_seqlens_kv, or neither.")
            x = torch_npu.npu_fused_infer_attention_score(
                q, k, v,
                num_heads=n,
                input_layout="BNSD",
                scale=scale,
            )[0]
            x = post_attn_layout(x)
            if return_bshd:
                return x
            return x.reshape(b, s, -1)
        q1 = q[:, :, :cu_seqlens_q[1], :].contiguous()
        k1 = k[:, :, :cu_seqlens_kv[1], :].contiguous()
        v1 = v[:, :, :cu_seqlens_kv[1], :].contiguous()
        if ulysses_world_size > 1:
            qkv1 = torch.cat([q1, k1, v1], dim=1).contiguous()
            qkv1 = self._apply_local_seq_remap_tensor(
                qkv1,
                ulysses_world_size=ulysses_world_size,
            )
            q1, k1, v1 = torch.split(qkv1, [n, n, n], dim=1)
        sink_frame_len_q_eff = int(self.sink_frame_len if sink_frame_len_q is None else sink_frame_len_q)
        sink_frame_len_k_eff = int(self.sink_frame_len if sink_frame_len_k is None else sink_frame_len_k)
        sink_frame_len_q_eff = max(0, min(sink_frame_len_q_eff, int(q1.shape[2])))
        sink_frame_len_k_eff = max(0, min(sink_frame_len_k_eff, int(k1.shape[2]), int(v1.shape[2])))
        sink_dense_q_len = sink_frame_len_q_eff
        q1_dense_ref = q1
        k1_dense_ref = k1
        v1_dense_ref = v1
        # 稀疏构造阶段将 sink 移到尾部，计算后再用 dense attention 回填 sink 输出。
        q1, k1, v1 = self._move_sink_qkv_to_end(
            q1, k1, v1, q_sink_len=sink_frame_len_q_eff, kv_sink_len=sink_frame_len_k_eff
        )
        sparsity_override = self._resolve_ulysses_local_sparsity_override(
            local_head_num=n,
            device=q1.device,
            ulysses_pg=ulysses_pg,
            ulysses_rank=ulysses_rank,
            ulysses_world_size=ulysses_world_size,
        )
        sabi_tensor = self.get_final_sabi_v2(
            q1,
            k1,
            sink_frame_len=self.sink_frame_len,
            sink_frame_len_q=sink_frame_len_q_eff,
            sink_frame_len_k=sink_frame_len_k_eff,
            img_token_len=self.img_token_len,
            img_token_len_q=img_token_len_q if img_token_len_q is not None else self.img_token_len,
            img_token_len_k=img_token_len_k if img_token_len_k is not None else self.img_token_len,
            sparsity_override=sparsity_override,
        )
        sabi_tensor = sabi_tensor.to(device=q1.device, dtype=torch.uint16).contiguous()
        actseqlen = [cu_seqlens_q[1]] * b
        actseqlenkv = [cu_seqlens_kv[1]] * b
        attn1, _ = torch_bsa.blitz_sparse_attention(
            q1,
            k1,
            v1,
            sabi=sabi_tensor,
            actual_seq_lengths=[cu_seqlens_q[1]] * b,
            actual_seq_lengths_kv=[cu_seqlens_kv[1]] * b,
            num_heads=n,
            num_key_value_heads=n,
            input_layout="BNSD",
            scale_value=scale,
            atten_mask=None,
            sparse_mode=0,
            softmax_lse_flag=False,
            block_shape=[self.block_size_q, self.block_size_k],
        )
        attn1 = self.move_sink_frame_back(attn1, sink_frame_len_q_eff)
        attn1 = self._replace_prefix_with_dense_attention(
            attn1,
            q1_dense_ref,
            k1_dense_ref,
            v1_dense_ref,
            prefix_len=sink_dense_q_len,
            num_heads=n,
            scale=scale,
        )
        if ulysses_world_size > 1:
            attn1 = self._apply_local_seq_remap_tensor(
                attn1,
                ulysses_world_size=ulysses_world_size,
                reverse=True,
            )
        if cu_seqlens_q[1] < s:
            attn2 = torch_npu.npu_fused_infer_attention_score(
                q[:, :, cu_seqlens_q[1]:, :],
                k[:, :, cu_seqlens_kv[1]:, :],
                v[:, :, cu_seqlens_kv[1]:, :],
                num_heads=n,
                input_layout="BNSD",
                scale=scale,
            )[0]
            x = torch.cat([attn1, attn2], dim=2)
        else:
            x = attn1
        x = post_attn_layout(x)
        if return_bshd:
            return x
        out = x.reshape(b, s, -1)
        return out


class HunyuanVideoSVGAdapter(SVGPredictor):
    def __init__(self, sparse_config, sparse_params=None):
        super().__init__(sparse_config, sparse_params)
        self.ring_svg_overlap = bool(sparse_config["SVG"].get("ring_sparse_overlap", False))
        self.ring_sample_mse_rows = int(sparse_config["SVG"].get("ring_sample_mse_rows", 64))
        self.update_sparse_params(sparse_params)
        # HyVideo特定参数
        
    def update_sparse_params(self, sparse_params):
        # HyVideo特定参数
        self.sink_frame_len = int(sparse_params.get("sink_frame_len", 0))
        self.img_token_len = int(sparse_params.get("img_token_len", 0))
        self.frame_num = int(sparse_params.get("frame_num", 0))
        self.frame_patch_h = int(sparse_params.get("frame_patch_h", 0))
        self.frame_patch_w = int(sparse_params.get("frame_patch_w", 0))
        logger.info(f"update sparse params successfully, sink_frame_len: {self.sink_frame_len}\
                    img_token_len: {self.img_token_len}, frame_num:{self.frame_num}. ")

    def forward_ring_sparse(
        self,
        runtime_attn,
        block_args: dict,
        softmax_scale: Optional[float] = None,
    ) -> torch.Tensor:
        if self.ring_svg_overlap:
            logger.info("Ring SVG overlap is not adapted yet; using the Ring SVG path.")
        return self._forward_ring_svg_global(runtime_attn, block_args, softmax_scale)

    def _ring_all_gather_canonical_img(self, runtime_attn, x: torch.Tensor) -> torch.Tensor:
        x_rank_major = self._ring_all_gather_seq_bnsd(runtime_attn, x.transpose(1, 2).contiguous())
        x_rank_major = x_rank_major.transpose(1, 2).contiguous()
        frame_num = self.frame_num
        frame_patch_h = self.frame_patch_h
        frame_patch_w = self.frame_patch_w
        width_splits = self._split_patched_width(frame_patch_w, int(runtime_attn.ring_world_size))
        token_splits = [frame_num * frame_patch_h * width for width in width_splits]
        b, _, h, d = x_rank_major.shape
        rank_parts = torch.split(x_rank_major, token_splits, dim=1)
        rank_parts = [
            part.reshape(b, frame_num, frame_patch_h, width_splits[rank], h, d)
            for rank, part in enumerate(rank_parts)
        ]
        return torch.cat(rank_parts, dim=3).reshape(b, -1, h, d).contiguous()

    def _get_ring_svg_local_metadata(
        self,
        runtime_attn,
        *,
        device: torch.device,
        img_q_len: int,
        img_kv_len: int,
        q_len: int,
        kv_len: int,
    ):
        frame_num = self.frame_num
        frame_patch_h = self.frame_patch_h
        frame_patch_w = self.frame_patch_w
        world_size = int(runtime_attn.ring_world_size)
        frame_size = frame_patch_h * frame_patch_w
        width_splits = self._split_patched_width(frame_patch_w, world_size)
        rank = int(runtime_attn.ring_rank)
        width_start, width_len = sum(width_splits[:rank]), width_splits[rank]

        local_ids = torch.arange(img_q_len, device=device, dtype=torch.long)
        local_ids = local_ids.reshape(frame_num, frame_patch_h, width_len)
        local_ids = local_ids + torch.arange(frame_num, device=device).view(-1, 1, 1) * (
            frame_size - frame_patch_h * width_len
        )
        local_ids = local_ids + torch.arange(frame_patch_h, device=device).view(1, -1, 1) * (
            frame_patch_w - width_len
        )
        local_ids = (local_ids + width_start).reshape(-1)

        # 将当前 rank 的局部宽度切片映射回全局时空坐标，再生成 spatial/temporal 两套 SABI 模板。
        block_num_q = math.ceil(int(q_len) / int(self.block_size_q))
        block_num_k = math.ceil(int(kv_len) / int(self.block_size_k))
        img_block_num_k = math.ceil(int(img_kv_len) / int(self.block_size_k))
        key_ids = torch.arange(img_kv_len, device=device, dtype=torch.long)
        key_blocks = key_ids // int(self.block_size_k)
        svg_block_size = 512
        spatial_key_groups = key_ids // svg_block_size
        temporal_key_groups = (
            (key_ids % frame_size) * frame_num + key_ids // frame_size
        ) // svg_block_size
        group_num = math.ceil(int(img_kv_len) / svg_block_size)
        relation_spatial = torch.zeros((group_num, img_block_num_k), dtype=torch.bool, device=device)
        relation_temporal = torch.zeros_like(relation_spatial)
        relation_spatial[spatial_key_groups, key_blocks] = True
        relation_temporal[temporal_key_groups, key_blocks] = True
        context_length = int(kv_len) - img_kv_len
        width_frame = ((context_length + frame_num * frame_size) * (1 - math.sqrt(self.sparsity)) - context_length)
        width_frame = width_frame / frame_size
        radius = max(1, int(frame_size * width_frame // svg_block_size))
        group_ids = torch.arange(group_num, device=device)
        nearby = (group_ids[:, None] - group_ids[None, :]).abs() < radius
        spatial_groups = (nearby.float() @ relation_spatial.float()) > 0
        temporal_groups = (nearby.float() @ relation_temporal.float()) > 0

        spatial_rows, temporal_rows = [], []
        for row in range(math.ceil(img_q_len / int(self.block_size_q))):
            row_ids = local_ids[row * int(self.block_size_q):min((row + 1) * int(self.block_size_q), img_q_len)]
            spatial_rows.append(spatial_groups.index_select(0, row_ids // svg_block_size).any(dim=0))
            temporal_ids = (
                (row_ids % frame_size) * frame_num + row_ids // frame_size
            ) // svg_block_size
            temporal_rows.append(temporal_groups.index_select(0, temporal_ids).any(dim=0))

        def build_sabi_template(img_rows):
            selected = torch.zeros((block_num_q, block_num_k), dtype=torch.bool, device=device)
            selected[:img_rows.shape[0], :img_block_num_k] = img_rows
            txt_k_start = min(block_num_k, int(img_kv_len) // int(self.block_size_k))
            selected[:, txt_k_start:] = True
            txt_q_start = min(block_num_q, int(img_q_len) // int(self.block_size_q))
            selected[txt_q_start:, :] = True
            block_ids = torch.arange(block_num_k, dtype=self.index_type, device=device)
            block_ids = block_ids.view(1, block_num_k).expand_as(selected)
            sabi = torch.where(selected, block_ids, torch.full_like(block_ids, block_num_k))
            sabi = torch.sort(sabi, dim=-1).values
            return torch.where(sabi < block_num_k, sabi, -1).to(dtype=torch.uint16).contiguous()

        metadata = {
            "frame_num": frame_num,
            "frame_size": frame_size,
            "svg_block_size": svg_block_size,
            "local_ids": local_ids,
            "nearby": nearby,
            "spatial_sabi": build_sabi_template(torch.stack(spatial_rows)),
            "temporal_sabi": build_sabi_template(torch.stack(temporal_rows)),
        }
        return metadata

    def _build_ring_svg_pattern(
        self,
        runtime_attn,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        img_q_len: int,
        img_kv_len: int,
        metadata,
    ) -> torch.Tensor:
        local_ids = metadata["local_ids"]
        frame_num = metadata["frame_num"]
        frame_size = metadata["frame_size"]
        svg_block_size = metadata["svg_block_size"]
        nearby = metadata["nearby"]
        key_global_ids = torch.arange(img_kv_len, dtype=torch.long, device=q.device)
        spatial_key_groups = key_global_ids // svg_block_size
        temporal_key_groups = (
            (key_global_ids % frame_size) * frame_num + key_global_ids // frame_size
        ) // svg_block_size

        sample_num = max(1, math.ceil(self.ring_sample_mse_rows / int(runtime_attn.ring_world_size)))
        sample_num = min(sample_num, int(img_q_len))
        sample_local = torch.randint(0, int(img_q_len), (sample_num,), device=q.device)
        sample_global = local_ids.index_select(0, sample_local)
        sampled_q = q[:, :, sample_local, :]
        image_k, image_v = k[:, :, :img_kv_len, :], v[:, :, :img_kv_len, :]
        scores = torch.matmul(sampled_q, image_k.transpose(-2, -1)) / math.sqrt(q.shape[-1])
        golden = torch.matmul(F.softmax(scores, dim=-1), image_v)
        mse_values = []
        for query_groups, key_groups in (
            (sample_global // svg_block_size, spatial_key_groups),
            (
                ((sample_global % frame_size) * frame_num + sample_global // frame_size) // svg_block_size,
                temporal_key_groups,
            ),
        ):
            allowed = nearby.index_select(0, query_groups).index_select(1, key_groups)
            sparse_hidden = torch.matmul(F.softmax(scores.masked_fill(~allowed, float("-inf")), dim=-1), image_v)
            mse_values.append(torch.mean((sparse_hidden - golden) ** 2, dim=(0, 2, 3)))
        mse = torch.stack(mse_values)
        dist.all_reduce(mse, op=dist.ReduceOp.SUM, group=runtime_attn.ring_pg)
        return mse[0] < mse[1]

    @staticmethod
    def _select_ring_svg_local_sabi(pattern: torch.Tensor, metadata, batch_size: int) -> torch.Tensor:
        sabi = torch.where(
            pattern.view(-1, 1, 1),
            metadata["spatial_sabi"],
            metadata["temporal_sabi"],
        )
        return sabi.unsqueeze(0).expand(batch_size, -1, -1, -1).contiguous()

    def _build_ring_svg_local_sabi(
        self,
        runtime_attn,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        img_q_len: int,
        img_kv_len: int,
    ) -> torch.Tensor:
        metadata = self._get_ring_svg_local_metadata(
            runtime_attn,
            device=q.device,
            img_q_len=img_q_len,
            img_kv_len=img_kv_len,
            q_len=int(q.shape[2]),
            kv_len=int(k.shape[2]),
        )
        pattern = self._build_ring_svg_pattern(
            runtime_attn,
            q,
            k,
            v,
            img_q_len=img_q_len,
            img_kv_len=img_kv_len,
            metadata=metadata,
        )
        return self._select_ring_svg_local_sabi(pattern, metadata, q.shape[0])

    def _forward_ring_svg_global(
        self,
        runtime_attn,
        block_args: dict,
        softmax_scale: Optional[float] = None,
    ) -> torch.Tensor:
        q_img_local, k_img_local, v_img_local = (
            block_args["q_img_local"], block_args["k_img_local"], block_args["v_img_local"]
        )
        txt_q, txt_k, txt_v = block_args["txt_q"], block_args["txt_k"], block_args["txt_v"]
        k_img_global = self._ring_all_gather_canonical_img(runtime_attn, k_img_local)
        v_img_global = self._ring_all_gather_canonical_img(runtime_attn, v_img_local)
        q_full = torch.cat([q_img_local, txt_q], dim=1).contiguous()
        k_full = torch.cat([k_img_global, txt_k], dim=1).contiguous()
        v_full = torch.cat([v_img_global, txt_v], dim=1).contiguous()
        pre_attn_layout, post_attn_layout = MEMORY_LAYOUT["BNSD"]
        q_exec = pre_attn_layout(q_full).contiguous()
        k_exec = pre_attn_layout(k_full).contiguous()
        v_exec = pre_attn_layout(v_full).contiguous()
        b, n, s_q, d = q_exec.shape
        s_kv = int(k_exec.shape[2])
        scale = softmax_scale if softmax_scale is not None else 1.0 / math.sqrt(d)
        sabi_tensor = self._build_ring_svg_local_sabi(
            runtime_attn,
            q_exec,
            k_exec,
            v_exec,
            img_q_len=int(q_img_local.shape[1]),
            img_kv_len=int(k_img_global.shape[1]),
        )
        out_full, _ = self._call_blitz_sparse_attention(
            q_exec,
            k_exec,
            v_exec,
            sabi=sabi_tensor.to(device=q_exec.device, dtype=torch.uint16).contiguous(),
            actual_seq_lengths=[s_q] * b,
            actual_seq_lengths_kv=[s_kv] * b,
            num_heads=n,
            scale=scale,
        )
        return post_attn_layout(out_full).contiguous()

    def get_must_keep_blocks_indices(self, **kwargs):
        token_len = kwargs["token_len"]
        sink_frame_len = kwargs['sink_frame_len']
        txt_len = kwargs['txt_token_len']
        sink_txt_len = sink_frame_len + txt_len
        sink_txt_blocks_k = math.ceil(sink_txt_len / self.block_size_k)
        num_blocks_k = math.ceil(token_len / self.block_size_k)
        k_num_of_last_block = token_len % self.block_size_k
        multi_last_k_blocks_token_num = (sink_txt_blocks_k - 1) * self.block_size_k + k_num_of_last_block
        if multi_last_k_blocks_token_num < sink_txt_len:
            sink_txt_blocks_k += 1
        
        sink_txt_start_indices = num_blocks_k - sink_txt_blocks_k
        must_keep_indices_k = torch.cat([
            torch.arange(sink_txt_start_indices, num_blocks_k)
        ])

        sink_txt_blocks_q = math.ceil(sink_txt_len / self.block_size_q)
        num_blocks_q = math.ceil(token_len / self.block_size_q)
        q_num_of_last_block = token_len % self.block_size_q
        multi_last_q_blocks_token_num = (sink_txt_blocks_q - 1) * self.block_size_q + q_num_of_last_block
        if multi_last_q_blocks_token_num < sink_txt_len:
            sink_txt_blocks_q += 1
        
        sink_txt_start_indices = num_blocks_q - sink_txt_blocks_q
        must_keep_indices_q = torch.cat([
            torch.arange(sink_txt_start_indices, num_blocks_q)
        ])
        return sink_txt_blocks_q, must_keep_indices_q, sink_txt_blocks_k, must_keep_indices_k

    def rearrange_x(self, x, frame_num, frame_size):
        _, d = x.shape
        s_wocontext = frame_num * frame_size
        x_wocontext = x[:s_wocontext]
        x_context = x[s_wocontext:]
        x_wocontext = x_wocontext.reshape(frame_num, frame_size, d) \
                        .permute(1, 0, 2) \
                        .reshape(frame_num * frame_size, -1) \
                        .contiguous()
        x = torch.cat([x_wocontext, x_context], dim=0)
        return x

    def inv_rearrange_x(self, x, frame_num, frame_size):
        _, d = x.shape
        s_wocontext = frame_num * frame_size
        x_wocontext = x[:s_wocontext]
        x_context = x[s_wocontext:]
        x_wocontext = x_wocontext.reshape(frame_size, frame_num, d) \
                        .permute(1, 0, 2) \
                        .reshape(frame_num * frame_size, -1) \
                        .contiguous()
        x = torch.cat([x_wocontext, x_context], dim=0)
        return x

    def build_sp_sabi_before_head_shard(self, block_args: dict):
        if block_args["v_local_bnsd"] is None:
            raise ValueError("SVG sparse path requires v_local_bnsd.")
        q_local_bnsd, k_local_bnsd, v_local_bnsd = (
            block_args["q_local_bnsd"], block_args["k_local_bnsd"], block_args["v_local_bnsd"]
        )
        q_local, k_local, v_local = (
            q_local_bnsd.transpose(1, 2).contiguous(),
            k_local_bnsd.transpose(1, 2).contiguous(),
            v_local_bnsd.transpose(1, 2).contiguous(),
        )
        joint_q_local = block_args.get("joint_q_local_bnsd")
        if joint_q_local is not None:
            joint_q_local = joint_q_local.transpose(1, 2).contiguous()
        q_global, k_global, v_global = q_local, k_local, v_local

        if block_args["ulysses_world_size"] > 1:
            qkv_global = torch.cat([q_global, k_global, v_global], dim=1).contiguous()
            qkv_global = self._apply_local_seq_remap_tensor(
                qkv_global,
                ulysses_world_size=block_args["ulysses_world_size"],
            )
            h = q_global.shape[1]
            q_global, k_global, v_global = torch.split(qkv_global, [h, h, h], dim=1)

        frame_size = self.img_token_len // self.frame_num
        q_meta = q_global
        pattern, sabi_tensor = self.get_final_sabi(q_meta, k_global, v_global, self.frame_num, frame_size)
        img_blocks_q = math.ceil(self.img_token_len / self.block_size_q)
        sabi_tensor = sabi_tensor[:, :, :img_blocks_q, :].contiguous()
        return {"pattern": pattern, "sabi": sabi_tensor}

    def attention(self, q,
            k,
            v,
            drop_rate=0,
            attn_mask=None,
            causal=False,
            cu_seqlens_q=None,
            cu_seqlens_kv=None,
            max_seqlen_q=None,
            max_seqlen_kv=None,
            batch_size=1,
            head_sabi: Optional[Dict[str, torch.Tensor]] = None,
            ulysses_pg=None,
            ulysses_rank: int = 0,
            ulysses_world_size: int = 1,
            return_bshd: bool = False,
            softmax_scale: Optional[float] = None,
            joint_q_local_bnsd: Optional[torch.Tensor] = None,
            ):
        pre_attn_layout, post_attn_layout = MEMORY_LAYOUT["BNSD"]

        frame_num = self.frame_num
        frame_size = (q.shape[1] - self.context_length) // frame_num

        b, s, n, d = q.shape
        q, k, v = pre_attn_layout(q), pre_attn_layout(k), pre_attn_layout(v)
        scale = softmax_scale if softmax_scale is not None else 1.0 / math.sqrt(d)
        if cu_seqlens_q is None or cu_seqlens_kv is None:
            if cu_seqlens_q is not None or cu_seqlens_kv is not None:
                raise ValueError("SVG sparse attention requires both cu_seqlens_q and cu_seqlens_kv, or neither.")
            x = torch_npu.npu_fused_infer_attention_score(
                q, k, v,
                num_heads=n,
                input_layout="BNSD",
                scale=scale,
            )[0]
        else:
            actseqlen = [cu_seqlens_q[1]] * b
            actseqlenkv = [cu_seqlens_kv[1]] * b
            q1 = q[:, :, :cu_seqlens_q[1], :].contiguous()
            k1 = k[:, :, :cu_seqlens_kv[1], :].contiguous()
            v1 = v[:, :, :cu_seqlens_kv[1], :].contiguous()
            qkv1 = torch.cat([q1, k1, v1], dim=1).contiguous()
            qkv1 = self._apply_local_seq_remap_tensor(
                qkv1,
                ulysses_world_size=ulysses_world_size,
            )
            q1, k1, v1 = torch.split(qkv1, [n, n, n], dim=1)
            if head_sabi is None:
                if ulysses_world_size <= 1:
                    pattern, sabi_tensor = self.get_final_sabi(q1, k1, v1, frame_num, frame_size)
                    head_sabi = {"pattern": pattern, "sabi": sabi_tensor}
                else:
                    if joint_q_local_bnsd is None:
                        prefix_q_len = int(cu_seqlens_q[1])
                        img_q_len = min(int(self.img_token_len), prefix_q_len)
                        joint_q_local_bnsd = q[:, :, img_q_len:prefix_q_len, :].transpose(1, 2).contiguous()
                    head_sabi = self.build_sp_sabi_before_head_shard(
                        {
                            "q_local_bnsd": q.transpose(1, 2).contiguous(),
                            "k_local_bnsd": k.transpose(1, 2).contiguous(),
                            "v_local_bnsd": v.transpose(1, 2).contiguous(),
                            "joint_q_local_bnsd": joint_q_local_bnsd,
                            "ulysses_pg": ulysses_pg,
                            "ulysses_rank": ulysses_rank,
                            "ulysses_world_size": ulysses_world_size,
                        }
                    )
            pattern = head_sabi["pattern"]
            sabi_tensor = head_sabi["sabi"]

            for h in range(n):
                if pattern[h] == False:
                    q_h = q1[0, h]
                    k_h = k1[0, h]
                    v_h = v1[0, h]
                    q1[0, h] = self.rearrange_x(q_h, frame_num, frame_size).contiguous()
                    k1[0, h] = self.rearrange_x(k_h, frame_num, frame_size).contiguous()
                    v1[0, h] = self.rearrange_x(v_h, frame_num, frame_size).contiguous()

            sabi_tensor = sabi_tensor.contiguous()
            sabi_tensor = sabi_tensor.to(q.device).to(torch.uint16)
            attn1, _ = torch_bsa.blitz_sparse_attention(
                q1, k1, v1,
                sabi=sabi_tensor,
                actual_seq_lengths=actseqlen,
                actual_seq_lengths_kv=actseqlenkv,
                num_heads=n,
                num_key_value_heads=n,
                input_layout="BNSD",
                scale_value=scale,
                atten_mask=None,
                sparse_mode=0,
                softmax_lse_flag=False,
                block_shape=[self.block_size_q, self.block_size_k],
            )
            attn1 = self._take_attention_output(attn1)
            for h in range(n):
                if pattern[h] == False:
                    attn1_h = attn1[0, h]
                    attn1[0, h] = self.inv_rearrange_x(attn1_h, frame_num, frame_size).contiguous()
            attn1 = self._apply_local_seq_remap_tensor(
                attn1,
                ulysses_world_size=ulysses_world_size,
                reverse=True,
            )
            if cu_seqlens_q[1] < s:
                attn2 = torch_npu.npu_fused_infer_attention_score(
                    q[:, :, cu_seqlens_q[1]:, :],
                    k[:, :, cu_seqlens_kv[1]:, :],
                    v[:, :, cu_seqlens_kv[1]:, :],
                    num_heads=n,
                    input_layout="BNSD",
                    scale=scale,
                )[0]
                x = torch.cat([attn1, attn2], dim=2)
            else:
                x = attn1
        x = post_attn_layout(x)
        if return_bshd:
            return x
        out = x.reshape(b, s, -1)
        return out

if __name__ == "__main__":
    sparse_predictor_manager.from_config(DEFAULT_CONFIG_PATH)
    logger.info(sparse_predictor_manager.config)
