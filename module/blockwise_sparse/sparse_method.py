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

DEFAULT_CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sparse_config.yaml")
yaml.SafeLoader.add_constructor("!env", lambda loader, node: os.path.expandvars(node.value))


def load_sparse_config_from_file(config_path=DEFAULT_CONFIG_PATH):
    with open(config_path, 'r', encoding='utf-8') as f:
        raw_content = f.read()
    expanded_content = os.path.expandvars(raw_content)
    config = yaml.safe_load(expanded_content)
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
        b, h, n, d = q.shape

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
        self.predictor_name = topk_config['predictor_name']

        self.sparsity_dict = {}
        self.load_sparsity(self.sparsity_files_path, self.total_steps)

    
    def get_block_attn(self, q: torch.Tensor, k, block_num_q, block_num_k):
        b, h, n_q, d = q.shape
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
        q_bmm = q.view(b * h * block_num, block_size, d)
        weight = torch.full((b * h * block_num, 1, block_size), 1.0 / block_size, device=q.device, dtype=q.dtype)
        q_mean = torch.bmm(weight, q_bmm).squeeze(1).view(b, h, block_num, d)
        return q_mean

    def get_block_attn_by_matmul_v2(self, q, k, block_num_q, block_num_k):
        q_mean = self.pooling_matmul(q, self.block_size_q, block_num_q)
        k_mean = self.pooling_matmul(k, self.block_size_k, block_num_k)
        attn = q_mean @ k_mean.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        return attn

    def get_sabi(self, q: torch.Tensor, k: torch.Tensor):
        b, h, n_q, d = q.shape
        n_k = k.shape[-2]
        block_num_q = math.ceil(n_q / self.block_size_q)
        block_num_k = math.ceil(n_k / self.block_size_k)
        attn, q_list, k_list = self.get_block_attn(q, k, block_num_q, block_num_k)
        new_sabi_list = []
        step_idx, layer_idx = self.get_effective_indices()
        sparsity = self.sparsity_dict[step_idx][layer_idx]
        for batch in range(b):
            for head in range(h):
                k_num = int((1 - sparsity[head]) * block_num_k)
                attn_bh = attn[batch, head]
                _, indices = torch.topk(attn_bh, k_num, dim=-1)
                new_sabi_list.append(indices)
        return new_sabi_list

    def get_sabi_v2(self, q: torch.Tensor, k: torch.Tensor):
        b, h, n_q, d = q.shape
        n_k = k.shape[-2]
        block_num_q = math.ceil(n_q / self.block_size_q)
        block_num_k = math.ceil(n_k / self.block_size_k)
        attn = self.get_block_attn_by_matmul_v2(q, k, block_num_q, block_num_k)

        new_sabi_tensor = torch.full(size=(b, h, block_num_q, block_num_k),
                                        fill_value=-1, dtype=self.index_type, device=q.device)
        step_idx, layer_idx = self.get_effective_indices()
        sparsity = self.sparsity_dict[step_idx][layer_idx]
        k_nums = ((1.0 - sparsity) * block_num_k).to(torch.int32)
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
        
        mid_sabi_list = self.get_sabi(mid_q, mid_k)
        final_sabi = self.combined_sabi_tensor_list(mid_sabi_list, must_keep_indices_q, 
                        must_keep_indices_k, block_num_k).unsqueeze(0)
        return final_sabi

    def get_final_sabi_v2(self, q: torch.Tensor, k: torch.Tensor, **kwargs):
        '''sabi连接must_keep的block indices，得到final_sabi'''
        sink_frame_len = kwargs["sink_frame_len"]
        img_token_len = kwargs["img_token_len"]
        all_token_len = q.shape[2]
        txt_token_len = all_token_len - img_token_len
        n_q = q.shape[2]
        n_k = k.shape[-2]
        block_num_q = math.ceil(n_q / self.block_size_q)
        block_num_k = math.ceil(n_k / self.block_size_k)

        sink_txt_blocks_q, must_keep_indices_q, sink_txt_blocks_k, must_keep_indices_k = \
            self.get_must_keep_blocks_indices(token_len=all_token_len, sink_frame_len=sink_frame_len, 
                                                txt_token_len=txt_token_len)

        mid_q, mid_k = q[:, :, : (block_num_q - sink_txt_blocks_q) * self.block_size_q, :],\
            k[:, :, : (block_num_k - sink_txt_blocks_k) * self.block_size_k, :]
        
        mid_q = mid_q.contiguous()
        mid_k = mid_k.contiguous()

        mid_sabi_tensor = self.get_sabi_v2(mid_q, mid_k)
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
    max_len = max(len(indices) for indices in indices_list)
    result = torch.full((len(indices_list), block.shape[1]), -1, dtype=torch.long)
    for i, indices in enumerate(indices_list):
        if len(indices) > 0:
            result[i, :len(indices)] = indices
    return result


def block2sabi(block_mask):
    sabi_list = []
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

    def sparsity_to_width(self, sparsity, num_frame, frame_size):
        seq_len = self.context_length + num_frame * frame_size
        total_elements = seq_len**2

        width = seq_len * (1 - math.sqrt(sparsity)) - self.context_length
        width_frame = width / frame_size

        return width_frame
    
    def sample_mse(self, query, key, value, context_length):

        key = key[:, :, :-context_length]
        value = value[:, :, :-context_length]

        mask_name = ["spatial", "temporal"]
        num_sampled_rows = 64

        cfg, num_heads, seq_len, dim = query.size()
        num_sampled_rows = min(num_sampled_rows, seq_len)
        sampled_rows = torch.randint(low=0, high=self.sample_mse_max_row, size=(num_sampled_rows,))
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
        b, h, s, d = q.shape
        
        device = q.device
        self.device = device
        frame_width = self.sparsity_to_width(self.sparsity, num_frames, frame_size)
        context_length_cu = s - num_frames * frame_size

        if len(self.attention_masks) == 0:
            spatial_mask, block_mask = self.get_attention_mask("spatial", s, num_frames, 
                                                                frame_size, frame_width, device=device)
            temporal_mask, _ = self.get_attention_mask("temporal", s, num_frames,
                                                                frame_size, frame_width, device=device)

            self.attention_masks = [spatial_mask, temporal_mask]
            self.sabi_tensor = block2sabi(block_mask)

            block_num_q, block_num_k = self.sabi_tensor.shape[0], self.sabi_tensor.shape[1]
            self.sabi_tensor = self.sabi_tensor.unsqueeze(0).unsqueeze(0).expand(b, h, block_num_q, block_num_k)
        
        mse_result = self.sample_mse(q, k, v, context_length_cu)
        pattern = (mse_result["spatial"] < mse_result["temporal"]).flatten()

        return pattern
    
 
    def get_final_sabi(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, num_frames: int, frame_size: int):
        '''sabi连接must_keep的block indices，得到final_sabi'''
        sink_frame_len = 0

        all_token_len = q.shape[2]
        txt_token_len = q.shape[2] - num_frames * frame_size
        b, h, n_q, d = q.shape
        n_k = k.shape[-2]
        block_num_q = math.ceil(n_q / self.block_size_q)
        block_num_k = math.ceil(n_k / self.block_size_k)

        sink_txt_blocks_q, must_keep_indices_q, sink_txt_blocks_k, must_keep_indices_k = \
            self.get_must_keep_blocks_indices(token_len=all_token_len, sink_frame_len=sink_frame_len,
                                                txt_token_len=txt_token_len)

        mid_q, mid_k = q[:, :, : (block_num_q - sink_txt_blocks_q) * self.block_size_q, :],\
            k[:, :, : (block_num_k - sink_txt_blocks_k) * self.block_size_k, :]
        
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
        logger.info(f"update sparse params successfully, sink_frame_len: {self.sink_frame_len}\
                    img_token_len: {self.img_token_len}. ")

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
        sink_q, rest_q = q[:, :, :sink_frame_len, :], q[:, :, sink_frame_len:, :]
        sink_k, rest_k = k[:, :, :sink_frame_len, :], k[:, :, sink_frame_len:, :]
        sink_v, rest_v = v[:, :, :sink_frame_len, :], v[:, :, sink_frame_len:, :]
        q = torch.cat((rest_q, sink_q), dim=2)
        k = torch.cat((rest_k, sink_k), dim=2)
        v = torch.cat((rest_v, sink_v), dim=2)
        return q, k, v

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
            batch_size=1,):
        pre_attn_layout, post_attn_layout = MEMORY_LAYOUT["BNSD"]
        b, s, n, d = q.shape
        q = pre_attn_layout(q)
        k = pre_attn_layout(k)
        v = pre_attn_layout(v)
        scale = 1.0 / math.sqrt(d)
        if cu_seqlens_q is None:
            x = torch_npu.npu_fused_infer_attention_score(
                q, k, v,
                num_heads=n,
                input_layout="BNSD",
                scale=scale,
            )[0]
        else:
            q1 = q[:, :, :cu_seqlens_q[1], :]
            k1 = k[:, :, :cu_seqlens_kv[1], :]
            v1 = v[:, :, :cu_seqlens_kv[1], :]
            q1, k1, v1 = self.move_sink_frame_to_end(q1, k1, v1, self.sink_frame_len)
            start_time = sync_and_get_time()
            sabi_tensor = self.get_final_sabi_v2(q1, k1, sink_frame_len=self.sink_frame_len,
                                                    img_token_len=self.img_token_len)
            exec_time = sync_and_get_time(start_time)
            logger.info(f"get sabi time:{exec_time*1000:.2f} ms")
            sabi_tensor = sabi_tensor.to(torch.uint16)
            actseqlen = [cu_seqlens_q[1]] * b
            actseqlenkv = [cu_seqlens_kv[1]] * b
            attn1 = torch_bsa.blitz_sparse_attention(q1,
                k1,
                v1,
                sabi=sabi_tensor,
                actual_seq_lengths=actseqlen,
                actual_seq_lengths_kv=actseqlenkv,
                num_heads=n,
                num_key_value_heads=n,
                input_layout="BNSD",
                scale_value=scale,
                atten_mask=None,
                sparse_mode=0
            )
            attn1 = self.move_sink_frame_back(attn1, self.sink_frame_len)
            attn2 = torch_npu.npu_fused_infer_attention_score(
                q[:, :, cu_seqlens_q[1]:, :],
                k[:, :, cu_seqlens_kv[1]:, :],
                v[:, :, cu_seqlens_kv[1]:, :],
                num_heads=n,
                input_layout="BNSD",
                scale=scale,
            )[0]
            x = torch.cat([attn1, attn2], dim=2)
        x = post_attn_layout(x)
        out = x.reshape(b, s, -1)
        return out


class HunyuanVideoSVGAdapter(SVGPredictor):
    def __init__(self, sparse_config, sparse_params=None):
        super().__init__(sparse_config, sparse_params)
        self.update_sparse_params(sparse_params)
        # HyVideo特定参数
        
    def update_sparse_params(self, sparse_params):
        # HyVideo特定参数
        self.sink_frame_len = sparse_params.get("sink_frame_len", 0)
        self.img_token_len = sparse_params.get("img_token_len", 0)
        self.frame_num = sparse_params.get("frame_num", 0)
        logger.info(f"update sparse params successfully, sink_frame_len: {self.sink_frame_len}\
                    img_token_len: {self.img_token_len}, frame_num:{self.frame_num}. ")
    
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
        s, d = x.shape
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
        s, d = x.shape
        s_wocontext = frame_num * frame_size
        x_wocontext = x[:s_wocontext]
        x_context = x[s_wocontext:]
        x_wocontext = x_wocontext.reshape(frame_size, frame_num, d) \
                        .permute(1, 0, 2) \
                        .reshape(frame_num * frame_size, -1) \
                        .contiguous()
        x = torch.cat([x_wocontext, x_context], dim=0)
        return x

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
            ):
        pre_attn_layout, post_attn_layout = MEMORY_LAYOUT["BNSD"]

        frame_num = self.frame_num
        frame_size = (q.shape[1] - self.context_length) // frame_num

        b, s, n, d = q.shape
        q = pre_attn_layout(q)
        k = pre_attn_layout(k)
        v = pre_attn_layout(v)
        scale = 1.0 / math.sqrt(d)

        actseqlen = [cu_seqlens_q[1]] * b
        actseqlenkv = [cu_seqlens_kv[1]] * b
        if cu_seqlens_q is None:
            x = torch_npu.npu_fused_infer_attention_score(
                q, k, v,
                num_heads=n,
                input_layout="BNSD",
                scale=scale,
            )[0]
        else:
            q1 = q[:, :, :cu_seqlens_q[1], :].contiguous()
            k1 = k[:, :, :cu_seqlens_kv[1], :].contiguous()
            v1 = v[:, :, :cu_seqlens_kv[1], :].contiguous()
            pattern, sabi_tensor = self.get_final_sabi(q1, k1, v1, frame_num, frame_size)

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
            attn1 = torch_bsa.blitz_sparse_attention(
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
            )
            attn2 = torch_npu.npu_fused_infer_attention_score(
                q[:, :, cu_seqlens_q[1]:, :],
                k[:, :, cu_seqlens_kv[1]:, :],
                v[:, :, cu_seqlens_kv[1]:, :],
                num_heads=n,
                input_layout="BNSD",
                scale=scale,
            )[0]
            for h in range(n):
                if pattern[h] == False:
                    attn1_h = attn1[0, h]
                    attn1[0, h] = self.inv_rearrange_x(attn1_h, frame_num, frame_size).contiguous()
            x = torch.cat([attn1, attn2], dim=2)
        x = post_attn_layout(x)

        out = x.reshape(b, s, -1)
        return out

if __name__ == "__main__":
    sparse_predictor_manager.from_config(DEFAULT_CONFIG_PATH)
    logger.info(sparse_predictor_manager.config)