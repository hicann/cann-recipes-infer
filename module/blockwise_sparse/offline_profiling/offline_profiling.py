import os
import math

import torch
import torch_npu
import torch.nn.functional as F
from loguru import logger

import numpy as np


def get_dense_attention_score(q, k):
    scores = q @ k.transpose(-2, -1)
    d_k = q.size(-1)
    del q, k
    scores = scores / (math.sqrt(d_k) + 1e-8)

    attn_score = F.softmax(scores, dim=-1, dtype=torch.float32)
    return attn_score


def get_per_row_min_cumulative_coverage(dense_qk, sparsity, device="npu:0"):
    batch, head, l_q, l_k = dense_qk.shape
    per_row_k = int(l_k * sparsity)
    cumultative_coverage_results = torch.empty(batch, head, device=device, dtype=dense_qk.dtype)

    for b in range(batch):
        for h in range(head):
            topk_values, _ = torch.topk(dense_qk[b, h], k=l_k - per_row_k, dim=-1, largest=False, sorted=False)
            topk_sum = torch.sum(topk_values, dim=-1)
            coverage_per_row = 1 - (topk_sum / torch.sum(dense_qk[b, h], dim=-1))
            min_coverage_per_head = torch.min(coverage_per_row)
            cumultative_coverage_results[b, h] = min_coverage_per_head
            del topk_values, topk_sum, coverage_per_row, min_coverage_per_head
    
    return cumultative_coverage_results


def remove_q_k_spec_token(q, txt_len=11):
    return q[:, :, :-txt_len, :]


'''
遍历不同的稀疏度，计算对应的累计注意力分数覆盖率
'''


def get_cumulative_coverage_of_different_sparsity(dir_path, global_layer_num, sparsity_list, image_len=10, 
                                                    txt_len=11, cu_seqlens_q=10206, img_token_lens=10200, 
                                                    frame_count=17, filter_first_frame=True, device="npu:0"):
    all_layer_sparsity_cumulative_coverage = {}
    for global_idx in range(0, global_layer_num):
        per_layer_sparsity_cumulative_coverage = {}
        qk_path = os.path.join(dir_path, f"layer-{global_idx}-qk.pt")
        qk = torch.load(qk_path, map_location=device)
        q = qk['q'].permute(0, 2, 1, 3)[:, :, :cu_seqlens_q, :]
        k = qk['k'].permute(0, 2, 1, 3)[:, :, :cu_seqlens_q, :]
        tokens_per_frame = int(q.shape[2] / frame_count)
        if filter_first_frame:
            q = q[:, :, tokens_per_frame:, :]
            k = k[:, :, tokens_per_frame:, :]
        txt_len = cu_seqlens_q - img_token_lens
        q_img = remove_q_k_spec_token(q, txt_len)
        k_img = remove_q_k_spec_token(k, txt_len)
        del q, k
        q_len = q_img.shape[2]
        dense_qk = get_dense_attention_score(q_img, k_img)
        del q_img, k_img
        for sparsity in sparsity_list:
            logger.info(f"Global layer idx:{global_idx}, Sparsity: {sparsity}.")
            coverage = get_per_row_min_cumulative_coverage(dense_qk, sparsity)
            per_layer_sparsity_cumulative_coverage[sparsity] = coverage
        del dense_qk
        all_layer_sparsity_cumulative_coverage[global_idx] = per_layer_sparsity_cumulative_coverage

    return all_layer_sparsity_cumulative_coverage


def get_sparsity_of_target_cumulative_coverage(all_layer_sparsity_cumulative_coverage, global_layer_num, head_num,
                                                sparsity_list, target_coverage=0.95, device="npu:0"):
    target_sparsity_of_target_coverage = torch.ones((global_layer_num, head_num)) * -1
    target_sparsity_of_target_coverage = target_sparsity_of_target_coverage.to(device)

    for sparsity in sparsity_list:
        for global_idx in range(0, global_layer_num):
            per_layer_sparsity_cumulative_coverage = all_layer_sparsity_cumulative_coverage[global_idx][sparsity][0]
            
            for idx, per_head_cov in enumerate(per_layer_sparsity_cumulative_coverage):
                if per_head_cov > target_coverage:
                    target_sparsity_of_target_coverage[global_idx][idx] = 1 - sparsity
    target_sparsity_of_target_coverage[torch.where(target_sparsity_of_target_coverage == -1)] = 0
    return target_sparsity_of_target_coverage


def save_expected_sparsity(dir_path, target_sparsity_expected_coverage, target_coverage=0.95):
    os.makedirs(dir_path, exist_ok=True)
    sparsity_file_path_of_expected_coverage = os.path.join(dir_path, 
                                                f"sparsity_of_RE_{target_coverage}_only_img.pt")
    torch.save(target_sparsity_expected_coverage, sparsity_file_path_of_expected_coverage)