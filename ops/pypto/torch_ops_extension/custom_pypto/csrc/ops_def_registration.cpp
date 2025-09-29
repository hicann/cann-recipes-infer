/* *
 * This program is free software, you can redistribute it and/or modify it.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.  * See LICENSE in the root of
the software repository for the full text of the License.
 */

#include <torch/extension.h>
#include <torch/library.h>
// 在custom命名空间里注册add_custom和npu_selected_flash_attention和后续的XXX算子，每次新增自定义aten ir都需先增加定义
// step1, 为新增自定义算子添加定义
TORCH_LIBRARY(custom_pypto, m) {
    m.def("add_custom(Tensor self, Tensor other) -> Tensor");
    m.def("add_custom_backward(Tensor self) -> (Tensor, Tensor)");
    m.def("npu_lightning_indexer_pto(Tensor query, Tensor key, Tensor weights, *, Tensor? actual_seq_lengths_query=None, Tensor? actual_seq_lengths_key=None,Tensor? block_table=None, str layout_query='BSND', str layout_key='PA_BSND', int sparse_count=2048, int sparse_mode=3) -> Tensor");
    m.def("npu_sparse_attention_pto(Tensor x, Tensor w_dq, Tensor w_uq_qr, Tensor w_uk, Tensor w_dkv_kr, Tensor gamma_cq, Tensor gamma_ckv, Tensor sin, Tensor cos, Tensor cache_index, Tensor kv_cache, Tensor kr_cache, Tensor block_table, Tensor act_seqs, Tensor w_idx_qb, Tensor w_idx_k, Tensor w_idx_proj, Tensor in_gamma_k, Tensor in_beta_k, Tensor index_k_cache) -> Tensor");
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
}


