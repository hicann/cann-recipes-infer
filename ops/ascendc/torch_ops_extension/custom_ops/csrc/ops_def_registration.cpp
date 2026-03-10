/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <torch/extension.h>
#include <torch/library.h>

// 在custom命名空间里注册add_custom和npu_sparse_flash_attention和后续的XXX算子，每次新增自定义aten ir都需先增加定义
// step1, 为新增自定义算子添加定义
TORCH_LIBRARY(custom, m) {
    m.def("npu_sparse_flash_attention(Tensor query, Tensor key, Tensor value, Tensor sparse_indices, float scale_value, int sparse_block_size, *, Tensor? block_table=None, Tensor? actual_seq_lengths_query=None, Tensor? actual_seq_lengths_kv=None, Tensor? query_rope=None, Tensor? key_rope=None, str layout_query='BSND', str layout_kv='BSND', int sparse_mode=3) -> Tensor");
    m.def("npu_sparse_flash_attention_antiquant(Tensor query, Tensor key, Tensor value, Tensor sparse_indices, float scale_value, int sparse_block_size, int key_quant_mode, int value_quant_mode, *, Tensor? key_dequant_scale=None, Tensor? value_dequant_scale=None, Tensor? block_table=None, Tensor? actual_seq_lengths_query=None, Tensor? actual_seq_lengths_kv=None, str layout_query='BSND', str layout_kv='BSND', int sparse_mode=3, int attention_mode=0, int quant_scale_repo_mode=0, int tile_size=0, int rope_head_dim=0) -> Tensor");
    m.def("npu_lightning_indexer(Tensor query, Tensor key, Tensor weights, *, Tensor? actual_seq_lengths_query=None, Tensor? actual_seq_lengths_key=None, Tensor? block_table=None, str layout_query='BSND', str layout_key='BSND', int sparse_count=2048, int sparse_mode=3) -> Tensor");
    m.def("npu_lightning_indexer_quant(Tensor query, Tensor key, Tensor weights, Tensor query_dequant_scale, Tensor key_dequant_scale, *, Tensor? actual_seq_lengths_query=None, Tensor? actual_seq_lengths_key=None, Tensor? block_table=None, int query_quant_mode=0, int key_quant_mode=0, str layout_query='BSND', str layout_key='BSND', int sparse_count=2048, int sparse_mode=3) -> Tensor");
    m.def("npu_swiglu_clip_quant(Tensor x, Tensor group_index, Tensor group_alpha, *, bool activate_left=False, int quant_mode=1, int clamp_mode=1) -> (Tensor, Tensor)");

    m.def("npu_mla_prolog_v3(Tensor token_x, Tensor weight_dq, Tensor weight_uq_qr, Tensor weight_uk,"
        "Tensor weight_dkv_kr, Tensor rmsnorm_gamma_cq, Tensor rmsnorm_gamma_ckv, Tensor rope_sin, Tensor rope_cos,"
        "Tensor(a!) kv_cache, Tensor(b!) kr_cache, *, Tensor? cache_index=None, Tensor? dequant_scale_x=None,"
        "Tensor? dequant_scale_w_dq=None, Tensor? dequant_scale_w_uq_qr=None, Tensor? dequant_scale_w_dkv_kr=None,"
        "Tensor? quant_scale_ckv=None, Tensor? quant_scale_ckr=None, Tensor? smooth_scales_cq=None,"
        "Tensor? actual_seq_len=None, Tensor? k_nope_clip_alpha=None, float rmsnorm_epsilon_cq=1e-05,"
        "float rmsnorm_epsilon_ckv=1e-05, str cache_mode='PA_BSND', bool query_norm_flag=False, int weight_quant_mode=0,"
        "int kv_cache_quant_mode=0, int query_quant_mode=0, int ckvkr_repo_mode=0, int quant_scale_repo_mode=0, int tile_size=128,"
        "float qc_qr_scale=1.0, float kc_scale=1.0) -> (Tensor, Tensor, Tensor, Tensor, Tensor)");

    m.def("npu_mla_prolog_v3_functional(Tensor token_x, Tensor weight_dq, Tensor weight_uq_qr, Tensor weight_uk,"
        "Tensor weight_dkv_kr, Tensor rmsnorm_gamma_cq, Tensor rmsnorm_gamma_ckv, Tensor rope_sin, Tensor rope_cos,"
        "Tensor kv_cache, Tensor kr_cache, *, Tensor? cache_index=None, Tensor? dequant_scale_x=None,"
        "Tensor? dequant_scale_w_dq=None, Tensor? dequant_scale_w_uq_qr=None, Tensor? dequant_scale_w_dkv_kr=None,"
        "Tensor? quant_scale_ckv=None, Tensor? quant_scale_ckr=None, Tensor? smooth_scales_cq=None,"
        "Tensor? actual_seq_len=None, Tensor? k_nope_clip_alpha=None, float rmsnorm_epsilon_cq=1e-05,"
        "float rmsnorm_epsilon_ckv=1e-05, str cache_mode='PA_BSND', bool query_norm_flag=False, int weight_quant_mode=0,"
        "int kv_cache_quant_mode=0, int query_quant_mode=0, int ckvkr_repo_mode=0, int quant_scale_repo_mode=0,"
        "int tile_size=128, float qc_qr_scale=1.0, float kc_scale=1.0) ->"
        "(Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor)");
    m.def("npu_gather_selection_kv_cache(Tensor(a!) selection_k_rope, Tensor(b!) selection_kv_cache, Tensor(c!) "
        "selection_kv_block_table, Tensor(d!) selection_kv_block_status, Tensor selection_topk_indices, Tensor full_k_rope, "
        "Tensor full_kv_cache, Tensor full_kv_block_table, Tensor full_kv_actual_seq, Tensor full_q_actual_seq, *, "
        "int selection_topk_block_size=64) -> Tensor");
    m.def("npu_gather_selection_kv_cache_functional(Tensor selection_k_rope, Tensor selection_kv_cache, "
        "Tensor selection_kv_block_table, Tensor selection_kv_block_status, Tensor selection_topk_indices, "
        "Tensor full_k_rope, Tensor full_kv_cache, Tensor full_kv_block_table, Tensor full_kv_actual_seq, "
        "Tensor full_q_actual_seq, *, int selection_topk_block_size=64) -> (Tensor, Tensor, Tensor, Tensor, Tensor)");
    }

// 通过pybind将c++接口和python接口绑定，这里绑定的是接口不是算子
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
}
