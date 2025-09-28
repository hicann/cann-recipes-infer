#!/bin/bash
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify.
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ======================================================================================================================

short_soc_version=$1
if [ ! -n "$short_soc_version" ]
then
    echo "compute unit is empty"
    exit 1
fi
echo "short_soc_version is $short_soc_version"

soc_version="Ascend910B1"
if [ "x$short_soc_version" = "xAscend910_93" ]
then
    soc_version="Ascend910_9391"
fi
echo "soc_version is $soc_version"

bin_param_path=$2
echo "bin_param_path is $bin_param_path"

adapter_py_file=$3
echo "adapter_py_file is $adapter_py_file"

tile_fwk_impl_path=$4
export TILE_FWK_OP_IMPL_PATH=$tile_fwk_impl_path
echo "tile_fwk_impl_path is $tile_fwk_impl_path"


tile_fwk_func_name=$5
echo "tile_fwk_func_name is $tile_fwk_func_name"

kernel_output_path=$6
echo "kernel_output_path is $kernel_output_path"
mkdir -p $kernel_output_path

compile_fatbin_by_opc()
{
    bin_param_file=$1
    opc $adapter_py_file --main_func=$tile_fwk_func_name --input_param=$bin_param_file --soc_version=$soc_version \
        --output=$kernel_output_path --simplified_key_mode=0 --op_mode=dynamic
    ls -alF $kernel_output_path
}

for bin_param_json_file in $(ls $bin_param_path/*.json)
do
    compile_fatbin_by_opc $bin_param_json_file
done
