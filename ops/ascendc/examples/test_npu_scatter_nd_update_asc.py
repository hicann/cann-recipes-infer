# This program is free software, you can redistribute it and/or modify it.
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import torch
import torch_npu
import torchair
import custom_ops
import numpy as np
import torch.nn as nn

from torch_npu.testing.testcase import TestCase, run_tests

DEVICE_ID = 0
torch_npu.npu.set_device(int(DEVICE_ID))

npDtypeDict = {
    'float16':np.float16,
    'bfloat16':np.float16,
    'int8':np.int8,
    'int32':np.int32,
    'int64':np.int64,
}

torchDtypeDict = {
    'float16':torch.float16,
    'bfloat16':torch.float16,
    'int8':torch.int8,
    'int32':torch.int32,
    'int64':torch.int64,
}

def _scatter_nd_update_asc_cpu(var, indices, update, a, b, c):
    for i in range(c):
        if (indices[i][0] >= 0):
            var[indices[i][0]][:] = update[i][:]
    return var

def _run_test_single(a, b, c, varDtype, indicesDtype, compare=True):

    npVarDtype = npDtypeDict[varDtype]
    torchVarDtype = torchDtypeDict[varDtype]
    varArray = np.random.uniform(-10, 10, (a, b)).astype(npVarDtype)
    varTensor = torch.tensor(varArray).to(torchVarDtype)

    npIndicesDtype = npDtypeDict[indicesDtype]
    torchIndicesDtype = torchDtypeDict[indicesDtype]
    # indicesArray = np.random.uniform(0, (c -1), (c, 1)).astype(npIndicesDtype)
    # indicesArray = np.arange(0, c, 1).reshape(c, 1).astype(npIndicesDtype)
    indicesArray = np.random.choice(a, size=c, replace=False).reshape(c, 1).astype(npIndicesDtype)
    indicesTensor = torch.tensor(indicesArray).to(torchIndicesDtype)

    updateArray = np.random.uniform(20, 40, (c, b)).astype(npVarDtype)
    updateTensor = torch.tensor(updateArray).to(torchVarDtype)

    yArray = _scatter_nd_update_asc_cpu(varArray, indicesArray, updateArray, a, b, c)

    torch_npu.npu.set_device(int(DEVICE_ID))
    varNpu = varTensor.to("npu:%s" % DEVICE_ID)
    indicesNpu = indicesTensor.to("npu:%s" % DEVICE_ID)
    updateNpu = updateTensor.to("npu:%s" % DEVICE_ID)

    # start run custom ops
    print(f'======================== PTA eager BEGIN ========================')
    torch.ops.custom.scatter_nd_update_asc(varNpu, indicesNpu, updateNpu)

    # compare result
    if compare:
        error_num = 0
        npuOut = varNpu.reshape(a, b).cpu().numpy().astype(npVarDtype)
        cpuOut = yArray.reshape(a, b).astype(npVarDtype)

        for i in range(a):
            for j in range(b):
                if (npuOut[i][j] - cpuOut[i][j]) != 0:
                    print("row:{}, col:{}, npu={}, cpu={}".format(i, j, npuOut[i][j], cpuOut[i][j]))
                    error_num = error_num + 1
                    break
        if error_num == 0:
            print("run success")
        else:
            print("run fail error_num = {}".format(error_num))
    print(f'======================== PTA eager FINISH ========================')

def _run_test_graph(a, b, c, varDtype, indicesDtype, compare=True):

    class Network(nn.Module):
        def __init__(self):
            super(Network, self).__init__()

        def forward(self, varNpu, indicesNpu, updateNpu):
            torch.ops.custom.scatter_nd_update_asc(varNpu, indicesNpu, updateNpu)


    npVarDtype = npDtypeDict[varDtype]
    torchVarDtype = torchDtypeDict[varDtype]
    varArray = np.random.uniform(-10, 10, (a, b)).astype(npVarDtype)
    varTensor = torch.tensor(varArray).to(torchVarDtype)

    npIndicesDtype = npDtypeDict[indicesDtype]
    torchIndicesDtype = torchDtypeDict[indicesDtype]
    # indicesArray = np.random.uniform(0, (c -1), (c, 1)).astype(npIndicesDtype)
    # indicesArray = np.arange(0, c, 1).reshape(c, 1).astype(npIndicesDtype)
    indicesArray = np.random.choice(a, size=c, replace=False).reshape(c, 1).astype(npIndicesDtype)
    indicesTensor = torch.tensor(indicesArray).to(torchIndicesDtype)

    updateArray = np.random.uniform(20, 40, (c, b)).astype(npVarDtype)
    updateTensor = torch.tensor(updateArray).to(torchVarDtype)

    yArray = _scatter_nd_update_asc_cpu(varArray, indicesArray, updateArray, a, b, c)

    torch_npu.npu.set_device(int(DEVICE_ID))
    varNpu = varTensor.to("npu:%s" % DEVICE_ID)
    indicesNpu = indicesTensor.to("npu:%s" % DEVICE_ID)
    updateNpu = updateTensor.to("npu:%s" % DEVICE_ID)

    # start run custom ops
    print(f'======================== PTA graph BEGIN ========================')

    npu_mode = Network().to("npu:%s" % DEVICE_ID)
    from torchair.configs.compiler_config import CompilerConfig
    config = CompilerConfig()
    config.mode = "reduce-overhead"
    npu_backend = torchair.get_npu_backend(compiler_config=config)
    npu_mode = torch.compile(npu_mode, fullgraph=True, backend=npu_backend, dynamic=False)
    npu_mode(varNpu, indicesNpu, updateNpu)

    # compare result
    if compare:
        error_num = 0
        npuOut = varNpu.reshape(a, b).cpu().numpy().astype(npVarDtype)
        cpuOut = yArray.reshape(a, b).astype(npVarDtype)

        for i in range(a):
            for j in range(b):
                if (npuOut[i][j] - cpuOut[i][j]) != 0:
                    print("row:{}, col:{}, npu={}, cpu={}".format(i, j, npuOut[i][j], cpuOut[i][j]))
                    error_num = error_num + 1
                    break
        if error_num == 0:
            print("run success")
        else:
            print("run fail error_num = {}".format(error_num))
    print(f'======================== PTA graph FINISH ========================')

class TestScatterNdUpateAsc(TestCase):
    def test_scatter_nd_update_asc_single(self):
        cases_time = [
            (1065088, 512, 128, 'bfloat16', 'int64', False),
            (16512, 512, 128, 'bfloat16', 'int32', False),
            (278656, 1, 128, 'float16', 'int32', False),
            (278656, 128, 128, 'int8', 'int32', False),
            (278656, 512, 128, 'bfloat16', 'int32', False),
            (2304, 1, 2047, 'float16', 'int32', False),
            (2304, 128, 2047, 'int8', 'int32', False),
            (2304, 512, 2047, 'bfloat16', 'int32', False),
            (256, 512, 63, 'bfloat16', 'int32', False),
            (8448, 512, 8192, 'bfloat16', 'int64', False),
            ]
        cases_real = [
            (5088, 512, 128, 'bfloat16', 'int64'),
            (16512, 512, 128, 'bfloat16', 'int32'),
            (8656, 1, 128, 'float16', 'int32'),
            (8656, 128, 128, 'int8', 'int32'),
            (8656, 512, 128, 'bfloat16', 'int32'),
            (2304, 1, 2047, 'float16', 'int32'),
            (2304, 128, 2047, 'int8', 'int32'),
            (2304, 512, 2047, 'bfloat16', 'int32'),
            (256, 512, 63, 'bfloat16', 'int32'),
            (8448, 512, 8192, 'bfloat16', 'int64'),
            ]
        cases = cases_real
        for idx in range(len(cases)):
            one = cases[idx]
            print("***** run case_{} = {} *****".format(idx, one))
            _run_test_single(*one)
            print("***** end case_{} = {} *****".format(idx, one))

    def test_scatter_nd_update_asc_garph(self):
        torch._dynamo.config.cache_size_limit = 16
        cases_time = [
            (1065088, 512, 128, 'bfloat16', 'int64', False),
            (16512, 512, 128, 'bfloat16', 'int32', False),
            (278656, 1, 128, 'float16', 'int32', False),
            (278656, 128, 128, 'int8', 'int32', False),
            (278656, 512, 128, 'bfloat16', 'int32', False),
            (2304, 1, 2047, 'float16', 'int32', False),
            (2304, 128, 2047, 'int8', 'int32', False),
            (2304, 512, 2047, 'bfloat16', 'int32', False),
            (256, 512, 63, 'bfloat16', 'int32', False),
            (8448, 512, 8192, 'bfloat16', 'int64', False),
            ]
        cases_real = [
            (5088, 512, 128, 'bfloat16', 'int64'),
            (16512, 512, 128, 'bfloat16', 'int32'),
            (8656, 1, 128, 'float16', 'int32'),
            (8656, 128, 128, 'int8', 'int32'),
            (8656, 512, 128, 'bfloat16', 'int32'),
            (2304, 1, 2047, 'float16', 'int32'),
            (2304, 128, 2047, 'int8', 'int32'),
            (2304, 512, 2047, 'bfloat16', 'int32'),
            (256, 512, 63, 'bfloat16', 'int32'),
            (8448, 512, 8192, 'bfloat16', 'int64'),
            ]
        cases = cases_real
        for idx in range(len(cases)):
            one = cases[idx]
            print("***** run graph case_{} = {} *****".format(idx, one))
            _run_test_graph(*one)
            print("***** end graph case_{} = {} *****".format(idx, one))


if __name__ == "__main__":
    run_tests()