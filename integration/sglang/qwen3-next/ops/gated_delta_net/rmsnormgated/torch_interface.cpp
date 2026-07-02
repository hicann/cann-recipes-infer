/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "op_kernel/CustVec.h"
#include "kernel_operator.h"

#include <cstdlib>

#include <ATen/Operators.h>
#include <torch/all.h>
#include <torch/library.h>

#include "acl/acl.h"
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "torch_npu/csrc/core/npu/DeviceUtils.h"
#include "torch_npu/csrc/framework/OpCommand.h"

#include "tiling/platform/platform_ascendc.h"

namespace npu_ops_transformer_ext {
namespace Mambav2Rmsnormgated {

class CubeHandler {
public:
    __aicore__ inline CubeHandler() {}
    __aicore__ inline void Init(GM_ADDR xmtx, GM_ADDR zmtx, GM_ADDR wmtx, GM_ADDR outmtx, GM_ADDR workspace, int B, int S, int D, int G, float E)
    {
        int offset = 0;
    }

    __aicore__ inline void Compute() {}

private:
    TPipe pipe;
};

class VecHandler {
public:
    __aicore__ inline VecHandler() {}
    __aicore__ inline void Init(GM_ADDR xmtx, GM_ADDR zmtx, GM_ADDR wmtx, GM_ADDR outmtx, GM_ADDR workspace, int B, int S, int D, int G, float E)
    {
        int offset = 0;
        tilingShapeCustVec(B, S, D, G, E, vec0shape);

        vec0.Init(xmtx, zmtx, wmtx, outmtx, vec0shape);
    }

    __aicore__ inline void Compute()
    {
        vec0.Compute();
    }

private:
    TPipe pipe;
    CustVec vec0;
    CustVecShapeInfo vec0shape;
};

__global__ __aicore__ void kernel_cust_rmsnormgated(GM_ADDR xmtx, GM_ADDR zmtx, GM_ADDR wmtx, GM_ADDR outmtx, GM_ADDR workspace, int B, int S, int D, int G, float E)
{
    if ASCEND_IS_AIC {
        CubeHandler cube;
        cube.Init(xmtx, zmtx, wmtx, outmtx, workspace, B, S, D, G, E);
        cube.Compute();
        mad((__cc__ float*)0, (__ca__ float*)0, (__cb__ float*)0, 32, 32, 32, 0x0, false, false, true);  // dummy instruction to make compiler happy
    }
    if ASCEND_IS_AIV {
        VecHandler vec;
        vec.Init(xmtx, zmtx, wmtx, outmtx, workspace, B, S, D, G, E);
        vec.Compute();
        copy_ubuf_to_ubuf((__ubuf__ float*)0, (__ubuf__ float*)0, 0, 1, 1, 0, 0);  // dummy instruction to make compiler happy
    }
}

at::Tensor mambav2_rmsnormgated(at::Tensor &x, at::Tensor &z, at::Tensor &w, int64_t G, double E)
{
    auto ascendcPlatform = platform_ascendc::PlatformAscendCManager::GetInstance();
    uint32_t blockDims = 1;
    const char* env_p = std::getenv("RMSNORMGATED_BLOCK_DIMS");
    if (env_p != nullptr) {
        blockDims = static_cast<uint32_t>(std::atoi(env_p));
    }
    int devidx = x.device().index();
    c10_npu::set_device(devidx);
    int B = x.size(0);
    int S = x.size(1);
    int D = x.size(2);

    int headdim = w.size(0);

    c10_npu::NPUStream stream = c10_npu::getCurrentNPUStream(devidx);
    void* aclstream = stream.stream();

    at::Tensor output = torch::empty(x.sizes(), x.options());

    // === workspace ===
    auto userWorkspaceSize = 1024;
    size_t systemWorkspaceSize = static_cast<size_t>(ascendcPlatform->GetLibApiWorkSpaceSize());
    int64_t workspaceSize = userWorkspaceSize + systemWorkspaceSize;
    auto workspaceTensor = at::empty({workspaceSize}, at::TensorOptions().dtype(at::kByte).device(x.options().device()));

    // === kernel ===
    auto acl_call = [=]() -> int {
        kernel_cust_rmsnormgated<<<blockDims, nullptr, aclstream>>>(
            (GM_ADDR)(x.storage().data()),
            (GM_ADDR)(z.storage().data()),
            (GM_ADDR)(w.storage().data()),
            (GM_ADDR)(output.storage().data()),
            (GM_ADDR)(workspaceTensor.storage().data()),
            B, S, D, G, E
        );
        return 0;
    };
    at_npu::native::OpCommand::RunOpApiV2("Mambav2Rmsnormgated", acl_call);
    return output;
}

torch::Tensor mambav2_rmsnormgated_meta(at::Tensor &x, at::Tensor &z, at::Tensor &w, int64_t G, double E)
{
    TORCH_CHECK(x.defined(), "Input tensor must be defined");
    return x;
}

TORCH_LIBRARY_IMPL(npu_ops_transformer_ext, PrivateUse1, m)
{
    m.impl("mambav2_rmsnormgated", mambav2_rmsnormgated);
}

TORCH_LIBRARY_IMPL(npu_ops_transformer_ext, Meta, m)
{
    m.impl("mambav2_rmsnormgated", TORCH_FN(mambav2_rmsnormgated_meta));
}

} // namespace Mambav2Rmsnormgated
} // namespace npu_ops_transformer_ext