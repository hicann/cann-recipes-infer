/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <torch/extension.h>
#include <torch/library.h>

// 在custom命名空间里注册add_custom和npu_sparse_flash_attention和后续的XXX算子，每次新增自定义aten ir都需先增加定义
// step1, 为新增自定义算子添加定义
TORCH_LIBRARY(custom, m) {
    m.def("npu_sparse_flash_attention(Tensor query, Tensor key, Tensor value, Tensor sparse_indices, float scale_value, int sparse_block_size, *, Tensor? block_table=None, Tensor? actual_seq_lengths_query=None, Tensor? actual_seq_lengths_kv=None, Tensor? query_rope=None, Tensor? key_rope=None, str layout_query='BSND', str layout_kv='BSND', int sparse_mode=3) -> Tensor");
    m.def("npu_lightning_indexer(Tensor query, Tensor key, Tensor weights, *, Tensor? actual_seq_lengths_query=None, Tensor? actual_seq_lengths_key=None, Tensor? block_table=None, str layout_query='BSND', str layout_key='PA_BSND', int sparse_count=2048, int sparse_mode=3) -> Tensor");
    m.def("npu_mla_prolog_v3(Tensor token_x, Tensor weight_dq, Tensor weight_uq_qr, Tensor weight_uk,"
        "Tensor weight_dkv_kr, Tensor rmsnorm_gamma_cq, Tensor rmsnorm_gamma_ckv, Tensor rope_sin, Tensor rope_cos,"
        "Tensor cache_index, Tensor(a!) kv_cache, Tensor(b!) kr_cache, *, Tensor? dequant_scale_x=None,"
        "Tensor? dequant_scale_w_dq=None, Tensor? dequant_scale_w_uq_qr=None, Tensor? dequant_scale_w_dkv_kr=None,"
        "Tensor? quant_scale_ckv=None, Tensor? quant_scale_ckr=None, Tensor? smooth_scales_cq=None,"
        "Tensor? actual_seq_len=None, float rmsnorm_epsilon_cq=1e-05, float rmsnorm_epsilon_ckv=1e-05,"
        "str cache_mode='PA_BSND', bool query_norm_flag=False) -> (Tensor, Tensor, Tensor, Tensor, Tensor)");

    m.def("npu_mla_prolog_v3_functional(Tensor token_x, Tensor weight_dq, Tensor weight_uq_qr, Tensor weight_uk,"
        "Tensor weight_dkv_kr, Tensor rmsnorm_gamma_cq, Tensor rmsnorm_gamma_ckv, Tensor rope_sin, Tensor rope_cos,"
        "Tensor cache_index, Tensor kv_cache, Tensor kr_cache, *, Tensor? dequant_scale_x=None,"
        "Tensor? dequant_scale_w_dq=None, Tensor? dequant_scale_w_uq_qr=None, Tensor? dequant_scale_w_dkv_kr=None,"
        "Tensor? quant_scale_ckv=None, Tensor? quant_scale_ckr=None, Tensor? smooth_scales_cq=None,"
        "Tensor? actual_seq_len=None, float rmsnorm_epsilon_cq=1e-05, float rmsnorm_epsilon_ckv=1e-05,"
        "str cache_mode='PA_BSND', bool query_norm_flag=False) ->"
        "(Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor)");
    m.def("npu_lightning_indexer_quant(Tensor query, Tensor key, Tensor weights, Tensor query_dequant_scale, Tensor key_dequant_scale, *, Tensor? actual_seq_lengths_query=None, Tensor? actual_seq_lengths_key=None, Tensor? block_table=None, int query_quant_mode=0, int key_quant_mode=0, str layout_query='BSND', str layout_key='PA_BSND', int sparse_count=2048, int sparse_mode=3) -> Tensor");
    }

// 通过pybind将c++接口和python接口绑定，这里绑定的是接口不是算子
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
}
