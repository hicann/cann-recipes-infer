/**
 * This program is free software, you can redistribute it and/or modify it.
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
 * BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file npu_ops_def.cpp
 * \brief
 */

#define Py_LIMITED_API_VERSION 0x03080000
#include <Python.h>
#include <ATen/Operators.h>
#include <torch/all.h>
#include <torch/library.h>
#include "acl/acl.h"

#include <vector>

extern "C" {
PyObject* PyInit__C(void)
{
    static struct PyModuleDef module_def = {
        PyModuleDef_HEAD_INIT, "_C", NULL, -1, NULL,
    };
    return PyModule_Create(&module_def);
}
}

namespace npu_ops_transformer_ext {

TORCH_LIBRARY(npu_ops_transformer_ext, m)
{
    m.def("mambav2_rmsnormgated(Tensor x, Tensor z, Tensor w, int G, float E) -> Tensor");
    m.def("recurrent_gated_delta_rule(Tensor mix_qkv, Tensor state, Tensor beta, "
        "float scale, Tensor actual_seq_lengths, Tensor ssm_state_indices, "
        "int nk, int nv, "
        "Tensor? num_accepted_tokens=None, Tensor? g=None, Tensor? gk=None) -> Tensor");
}
} // namespace npu_ops_transformer_ext
