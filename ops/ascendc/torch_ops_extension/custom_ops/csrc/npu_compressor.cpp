/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <iostream>
#include <vector>
#include <torch/library.h>
#include <torch/torch.h>
#include "ops_common.h"

namespace custom {
using namespace at_npu::native;

// npu tensor max size
const int DIM_1 = 1;
const int DIM_2 = 2;
const int DIM_3 = 3;
const int VALUE_0 = 0;
const int CONTINUOUS = 1;
const int CYCLE = 2;

std::vector<bool> is_contiguous_axes(const at::Tensor &tensor)
{
    auto sizes = tensor.sizes();
    auto strides = tensor.strides();
    int64_t ndim = sizes.size();

    if (ndim == 0) {
        return {};
    }
    std::vector<bool> result(ndim, false);

    std::vector<int64_t> contiguous_stride(ndim, 1);
    for (int64_t i = ndim - 2; i >= 0; i--) {
        contiguous_stride[i] = contiguous_stride[i + 1] * sizes[i + 1];
    }


    for (int64_t i = 0; i < ndim; i++) {
        result[i] = (strides[i] == contiguous_stride[i]);
    }
    return result;
}

std::tuple<at::Tensor> construct_compressor_output_tensor(const at::Tensor &x, const at::Tensor &norm_weight,
                                                          const at::Tensor &rope_sin, int64_t cmp_ratio, int64_t coff)
{
    auto x_dim = x.dim();
    at::SmallVector<int64_t, 8> cmp_kv_size;
    at::Tensor cmp_kv;
    auto cmp_s = 0;
    if (x_dim == DIM_3) {
        cmp_s = (x.size(1) + cmp_ratio - 1) / cmp_ratio;
        cmp_kv_size = {x.size(0), cmp_s, norm_weight.size(0)};
    } else {
        cmp_s = rope_sin.size(0);
        cmp_kv_size = {cmp_s, norm_weight.size(0)};
    }

    cmp_kv = at::empty(cmp_kv_size, x.options().dtype(x.dtype()));

    return std::tuple<at::Tensor>(cmp_kv);
}

// 为NPU设备实现前向接口
std::tuple<at::Tensor> compressor(const at::Tensor &x, const at::Tensor &wkv, const at::Tensor &wgate,
                                  at::Tensor &state_cache, const at::Tensor &ape, const at::Tensor &norm_weight,
                                  const at::Tensor &rope_sin, const at::Tensor &rope_cos,
                                  int64_t rope_head_dim, int64_t cmp_ratio,
                                  const c10::optional<at::Tensor> &state_block_table,
                                  const c10::optional<at::Tensor> &cu_seqlens, const c10::optional<at::Tensor> &seqused,
                                  const c10::optional<at::Tensor> &start_pos,
                                  int64_t coff, double norm_eps, int64_t rotary_mode, int64_t cache_mode)
{
    // construct the output tensor
    TORCH_CHECK(x.defined(), "Check x != nullptr failed");
    auto x_dim = x.dim();
    TORCH_CHECK(x_dim == DIM_2 || x_dim == DIM_3, "x dim num[", x_dim, "] should be 2 or 3");

    TORCH_CHECK(norm_weight.defined(), "Check norm_weight != nullptr failed");
    auto norm_weight_dim = norm_weight.dim();
    TORCH_CHECK(norm_weight_dim == DIM_1, "norm_weight dim num[", norm_weight_dim, "] should be 1");

    TORCH_CHECK(rope_sin.defined(), "Check rope_sin != nullptr failed");
    auto rope_sin_dim = rope_sin.dim();
    TORCH_CHECK(rope_sin_dim == x_dim, "rope_sin dim num[", rope_sin_dim, "] should be equal to x dim num[", x_dim,
                "]");

    TORCH_CHECK(cmp_ratio > VALUE_0, "cmp_ratio should be greater than 0");

    std::tuple<at::Tensor> output = construct_compressor_output_tensor(x, norm_weight, rope_sin, cmp_ratio, coff);
    at::Tensor cmp_kv = std::get<0>(output);
    
    TORCH_CHECK(state_cache.defined(), "Check state_cache != nullptr failed");
    auto state_cache_dim = state_cache.dim();
    TORCH_CHECK(state_cache_dim == DIM_3, "state_cache dim num[", state_cache_dim, "] should be 3");

    TORCH_CHECK(wkv.defined(), "Check wkv != nullptr failed");
    auto wkv_size = wkv.numel();
    TORCH_CHECK(wkv_size != VALUE_0, "wkv should not be empty tensor");

    TORCH_CHECK(wgate.defined(), "Check wgate != nullptr failed");
    auto wgate_size = wgate.numel();
    TORCH_CHECK(wgate_size != VALUE_0, "wgate should not be empty tensor");

    auto state_cache_size = state_cache.sizes();
    TORCH_CHECK(state_cache_size[DIM_1] != VALUE_0 && state_cache_size[DIM_2] != VALUE_0,
        "state_cache should not be empty tensor except B equal to 0");
    
    TORCH_CHECK(ape.defined(), "Check ape != nullptr failed");
    auto ape_size = ape.numel();
    TORCH_CHECK(ape_size != VALUE_0, "ape should not be empty tensor");

    auto norm_weight_size = norm_weight.numel();
    TORCH_CHECK(norm_weight_size != VALUE_0, "norm_weight should not be empty tensor");

    auto rope_sin_size = rope_sin.sizes();
    TORCH_CHECK(rope_sin_size[rope_sin_dim-1] != VALUE_0,
        "rope_sin should not be empty tensor except B or S equal to 0");

    TORCH_CHECK(rope_cos.defined(), "Check rope_cos != nullptr failed");
    auto rope_cos_size = rope_cos.sizes();
    auto rope_cos_dim = rope_cos.dim();
    TORCH_CHECK(rope_cos_dim == x_dim, "rope_cos dim num[", rope_cos_dim, "] should be equal to x dim num[", x_dim,
                "]");
    TORCH_CHECK(rope_cos_size[rope_cos_dim-1] != VALUE_0,
        "rope_cos should not be empty tensor except B or S equal to 0");

    auto contiguous_axes_result = is_contiguous_axes(state_cache);
    TORCH_CHECK(contiguous_axes_result[DIM_1] && contiguous_axes_result[DIM_2], "when cache_mode == ", cache_mode,
        ", state_cache must be contiguous on all axes except axis 0");

    int64_t state_cache_stride_dim0 = state_cache.stride(0);

    EXEC_NPU_CMD_V1(aclnnCompressor, x, wkv, wgate, state_cache, ape, norm_weight, rope_sin, rope_cos,
                    state_block_table, cu_seqlens, seqused, start_pos, rope_head_dim, cmp_ratio, coff, norm_eps,
                    rotary_mode, cache_mode, state_cache_stride_dim0, cmp_kv);

    return std::tuple<at::Tensor>(cmp_kv);
}

// 为META设备实现前向接口
std::tuple<at::Tensor>
compressor_meta(const at::Tensor &x, const at::Tensor &wkv, const at::Tensor &wgate, at::Tensor &state_cache,
                const at::Tensor &ape, const at::Tensor &norm_weight, const at::Tensor &rope_sin,
                const at::Tensor &rope_cos, int64_t rope_head_dim, int64_t cmp_ratio,
                const c10::optional<at::Tensor> &state_block_table, const c10::optional<at::Tensor> &cu_seqlens,
                const c10::optional<at::Tensor> &seqused, const c10::optional<at::Tensor> &start_pos,
                int64_t coff, double norm_eps, int64_t rotary_mode, int64_t cache_mode)
{
    // construct the output tensor
    TORCH_CHECK(x.defined(), "Check x != nullptr failed");
    auto x_dim = x.dim();
    TORCH_CHECK(x_dim == DIM_2 || x_dim == DIM_3, "x dim num[", x_dim, "] should be 2 or 3");

    TORCH_CHECK(norm_weight.defined(), "Check norm_weight != nullptr failed");
    auto norm_weight_dim = norm_weight.dim();
    TORCH_CHECK(norm_weight_dim == DIM_1, "norm_weight dim num[", norm_weight_dim, "] should be 1");

    TORCH_CHECK(rope_sin.defined(), "Check rope_sin != nullptr failed");
    auto rope_sin_dim = rope_sin.dim();
    TORCH_CHECK(rope_sin_dim == x_dim, "rope_sin dim num[", rope_sin_dim, "] should be equal to x dim num[", x_dim,
                "]");

    TORCH_CHECK(cmp_ratio > VALUE_0, "cmp_ratio should be greater than 0");

    std::tuple<at::Tensor> output = construct_compressor_output_tensor(x, norm_weight, rope_sin, cmp_ratio, coff);

    return output;
}
} // namespace custom

// 为NPU设备注册前向实现
TORCH_LIBRARY_IMPL(custom, PrivateUse1, m)
{
    m.impl("compressor", &custom::compressor);
}

// step5, 为META设备注册前向实现
TORCH_LIBRARY_IMPL(custom, Meta, m)
{
    m.impl("compressor", &custom::compressor_meta);
}
