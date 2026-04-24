/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <torch/extension.h>
#include <torch/library.h>

// 在custom命名空间里注册add_custom和npu_sparse_flash_attention和后续的XXX算子，每次新增自定义aten ir都需先增加定义
// step1, 为新增自定义算子添加定义
TORCH_LIBRARY(custom, m) {
    m.def("npu_swiglu_clip_quant(Tensor x, Tensor group_index, Tensor group_alpha, *, bool activate_left=False, int quant_mode=1, int clamp_mode=1) -> (Tensor, Tensor)");
    m.def("npu_swiglu_group_quant(Tensor x, *, Tensor? topk_weight=None, Tensor? group_index=None, ScalarType dst_type, int quant_mode=1, int group_size=128, bool round_scale=False, bool ue8m0_scale=False, bool output_origin=False, int group_list_type=0, float clamp_value=0.0) -> (Tensor, Tensor, Tensor)");
    m.def("npu_hc_post(Tensor x, Tensor residual, Tensor post, Tensor comb) -> Tensor");
    m.def("indexer_compress_epilog(Tensor(a!) indexer_compress_cache, Tensor(b!) indexer_compress_scale, Tensor x, Tensor slot_mapping,  *, int quant_mode=1, bool round_scale=True) -> ()");

    m.def("inplace_partial_rotary_mul(Tensor(a!) x, Tensor r1, Tensor r2, *, str rotary_mode, int[2] partial_slice) -> ()");
    
    m.def("npu_gather_selection_kv_cache(Tensor(a!) selection_k_rope, Tensor(b!) selection_kv_cache, Tensor(c!) "
        "selection_kv_block_table, Tensor(d!) selection_kv_block_status, Tensor selection_topk_indices, Tensor full_k_rope, "
        "Tensor full_kv_cache, Tensor full_kv_block_table, Tensor full_kv_actual_seq, Tensor full_q_actual_seq, *, "
        "int selection_topk_block_size=64) -> Tensor");
    m.def("npu_gather_selection_kv_cache_functional(Tensor selection_k_rope, Tensor selection_kv_cache, "
        "Tensor selection_kv_block_table, Tensor selection_kv_block_status, Tensor selection_topk_indices, "
        "Tensor full_k_rope, Tensor full_kv_cache, Tensor full_kv_block_table, Tensor full_kv_actual_seq, "
        "Tensor full_q_actual_seq, *, int selection_topk_block_size=64) -> (Tensor, Tensor, Tensor, Tensor, Tensor)");
    m.def("npu_moe_gating_top_k(Tensor x, int k, *, Tensor? bias=None, Tensor? input_ids=None, Tensor? tid2eid=None, int k_group=1, int group_count=1, float routed_scaling_factor=1., float eps=9.9999999999999995e-21, int group_select_mode=0, int renorm=0, int norm_type=0, bool out_flag=False) -> (Tensor, Tensor, Tensor)"); 
    
    m.def("compressor(Tensor x, Tensor wkv, Tensor wgate, Tensor(a!) state_cache, "
        "Tensor ape, Tensor norm_weight, Tensor rope_sin, Tensor rope_cos, int rope_head_dim, int cmp_ratio, *,"
        "Tensor? state_block_table=None, Tensor? cu_seqlens=None, Tensor? seqused=None, Tensor? start_pos=None,"
        "int coff=1, float norm_eps=1e-6, int rotary_mode=1, int cache_mode=1) -> (Tensor)");

    m.def("scatter_nd_update_asc(Tensor(a!) var, Tensor indices, Tensor update) -> ()");
    
    m.def("npu_hc_pre_sinkhorn(Tensor mixes, Tensor rsqrt, Tensor hc_scale, Tensor hc_base, Tensor x, int hc_mult=4, int hc_sinkhorn_iters=20,"
          "float hc_eps=1e-5) -> (Tensor, Tensor, Tensor)");

    m.def("npu_hc_pre_inv_rms(Tensor x, *, float epsilon=1e-20) -> Tensor");

    m.def("npu_hc_pre(Tensor x, Tensor hc_fn, Tensor hc_scale, Tensor hc_base, *,int hc_mult=4, int hc_sinkhorn_iters=20, float norm_eps=1e-6,"
          "float hc_eps=1e-6) -> (Tensor, Tensor, Tensor)");

    m.def("npu_moe_init_routing_group_quant(Tensor x, Tensor expert_idx, Tensor? scale=None, Tensor? offset=None, int active_num=-1,"
          "int expert_capacity=-1, int expert_num=-1, int drop_pad_mode=-1, int expert_tokens_num_type=-1, bool expert_tokens_num_flag=False,"
          "int quant_mode=-1, SymInt[] active_expert_range, int row_idx_type=-1, int group_size=128) ->(Tensor, Tensor, Tensor, Tensor)");
    m.def("npu_quant_lightning_indexer(Tensor query, Tensor key, Tensor weights, Tensor query_dequant_scale, Tensor key_dequant_scale,"
        "int query_quant_mode, int key_quant_mode, *, Tensor? actual_seq_lengths_query=None, Tensor? actual_seq_lengths_key=None, "
        "Tensor? block_table=None, Tensor? metadata=None, str layout_query='BSND', str layout_key='PA_BSND', int sparse_count=2048, int sparse_mode=3, "
        "int pre_tokens=9223372036854775807, int next_tokens=9223372036854775807, int cmp_ratio=1, bool return_value=False) -> (Tensor, Tensor)");

    m.def("npu_sparse_attn_sharedkv_metadata(int num_heads_q, int num_heads_kv, int head_dim,"
        " *, Tensor? cu_seqlens_q=None, Tensor? cu_seqlens_ori_kv=None, Tensor? cu_seqlens_cmp_kv=None, Tensor? seqused_q=None, Tensor? seqused_kv=None, "
        "int batch_size=0, int max_seqlen_q=0, int max_seqlen_kv=0, int ori_topk=0, int cmp_topk=0, int cmp_ratio=-1, int ori_mask_mode=4, int cmp_mask_mode=3, int ori_win_left=127, int ori_win_right=0, "
        "str layout_q='BSND', str layout_kv='PA_ND', bool has_ori_kv=True, bool has_cmp_kv=True, str device='npu:0') -> Tensor");
    
    m.def("kv_compress_epilog(Tensor(a!) kv_compress_cache, Tensor x, Tensor slot_mapping, "
          " *, int quant_group_size=64, int quant_mode = 2, bool round_scale_flag=True) -> ()");

    m.def("npu_sparse_attn_sharedkv(Tensor q, *, Tensor? ori_kv=None, Tensor? cmp_kv=None, Tensor? ori_sparse_indices=None, Tensor? cmp_sparse_indices=None, Tensor? ori_block_table=None, "
        "Tensor? cmp_block_table=None, Tensor? cu_seqlens_q=None, Tensor? cu_seqlens_ori_kv=None, Tensor? cu_seqlens_cmp_kv=None, Tensor? seqused_q=None, Tensor? seqused_kv=None, "
        "Tensor? sinks=None, Tensor? metadata=None, float softmax_scale=0, int cmp_ratio=0, int ori_mask_mode=4, int cmp_mask_mode=3, int ori_win_left=127, int ori_win_right=0, "
        "str layout_q='BSND', str layout_kv='PA_ND', bool return_softmax_lse=False) -> (Tensor, Tensor)");
    m.def("npu_quant_lightning_indexer_metadata(int num_heads_q, int num_heads_k, int head_dim, int query_quant_mode, int key_quant_mode, *, "
        "Tensor? actual_seq_lengths_query=None, Tensor? actual_seq_lengths_key=None, int batch_size=0, int max_seqlen_q=0, "
        "int max_seqlen_k=0, str layout_query='BSND', str layout_key='BSND', int sparse_count=2048, int sparse_mode=3, "
        "int pre_tokens=9223372036854775807, int next_tokens=9223372036854775807, int cmp_ratio=1, str device='npu:0') -> Tensor");
    m.def("npu_kv_quant_sparse_attn_sharedkv(Tensor q, int kv_quant_mode, *, Tensor? ori_kv=None, Tensor? cmp_kv=None, "
          "Tensor? ori_sparse_indices=None, Tensor? cmp_sparse_indices=None, Tensor? ori_block_table=None, "
          "Tensor? cmp_block_table=None, Tensor? cu_seqlens_q=None, Tensor? cu_seqlens_ori_kv=None, "
          "Tensor? cu_seqlens_cmp_kv=None, Tensor? seqused_q=None, Tensor? seqused_kv=None, Tensor? sinks=None, "
          "Tensor? metadata=None, int tile_size=0, int rope_head_dim=0, float softmax_scale=0, "
          "int cmp_ratio=0, int ori_mask_mode=4, int cmp_mask_mode=3, int ori_win_left=127, int ori_win_right=0, "
          "str layout_q='BSND', str layout_kv='PA_ND', bool return_softmax_lse=False) -> (Tensor, Tensor)");
    m.def("npu_kv_quant_sparse_attn_sharedkv_metadata(int num_heads_q, int num_heads_kv, int head_dim, int kv_quant_mode,"
        " *, Tensor? cu_seqlens_q=None, Tensor? cu_seqlens_ori_kv=None, Tensor? cu_seqlens_cmp_kv=None, Tensor? seqused_q=None, Tensor? seqused_kv=None, "
        "int batch_size=0, int max_seqlen_q=0, int max_seqlen_kv=0, int ori_topk=0, int cmp_topk=0, int tile_size=0, int rope_head_dim=0, int cmp_ratio=-1, int ori_mask_mode=4, int cmp_mask_mode=3, int ori_win_left=127, int ori_win_right=0, "
        "str layout_q='BSND', str layout_kv='PA_ND', bool has_ori_kv=True, bool has_cmp_kv=True, str device='npu:0') -> Tensor");
    m.def("npu_rms_norm_dynamic_quant(Tensor x, Tensor gamma, *, Tensor? smooth_scale=None, Tensor? beta=None, float epsilon=1e-6) -> (Tensor, Tensor)");
    m.def("npu_dequant_swiglu_clamp_quant(Tensor x, *, Tensor? weight_scale=None, Tensor? activation_scale=None, Tensor? bias=None, Tensor? quant_scale=None, Tensor? quant_offset=None, Tensor? group_index=None, bool activate_left=False, int quant_mode=0, int? dst_type=None, int? round_mode=None, int? activate_dim=None, int swiglu_mode=0, float clamp_limit=7.0, float glu_alpha=1.702, float glu_bias=1.0) -> (Tensor, Tensor)");
    }
// 通过pybind将c++接口和python接口绑定，这里绑定的是接口不是算子
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
}