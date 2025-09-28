#!/bin/bash
# This program is free software, you can redistribute it and/or modify.
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ======================================================================================================================

if [ -z "$ASCEND_HOME_PATH" ]; then
    if [ -z "$ASCEND_AICPU_PATH" ]; then
        echo "please set env"
        exit 1
    else 
        export ASCEND_HOME_PATH=$ASCEND_AICPU_PATH
    fi
else
    export ASCEND_HOME_PATH=$ASCEND_HOME_PATH
fi

echo "using ASCEND_HOME_PATH=$ASCEND_HOME_PATH"

BASEPATH=$(cd "$(dirname $0)"; pwd)
BUILD_PATH="${BASEPATH}/build"
OUTPUT_PATH="${BASEPATH}/output"
COMPUTE_UNIT=$1
if [ ! -n "$COMPUTE_UNIT" ]
then
    COMPUTE_UNIT="ascend910b,ascend910_93"
fi
BINARY_OUTPUT_PATH=$2
if [ ! -n "$BINARY_OUTPUT_PATH" ]
then
    BINARY_OUTPUT_PATH="${OUTPUT_PATH}/op_impl/ai_core/tbe/kernel"
fi
echo ${LD_LIBRARY_PATH}

pkg_suffix="linux.x86_64"
if cat /proc/version | grep -q "aarch64"
then
    pkg_suffix="linux.aarch64"
fi
echo "linux is $pkg_suffix"

build_ops()
{
    mkdir -p "${BUILD_PATH}"
    cd "${BUILD_PATH}"
    cmake -D BUILD_OPEN_PROJECT=True \
          -D ENABLE_BUILD_HOST=True \
          -D ENABLE_BUILD_BINARY=True \
          -D ASCEND_HOME_PATH=${ASCEND_HOME_PATH} \
          -D CMAKE_INSTALL_PREFIX=${OUTPUT_PATH} \
          -D ASCEND_COMPUTE_UNIT=${COMPUTE_UNIT} \
          -D BINARY_OUTPUT_PATH=${BINARY_OUTPUT_PATH} \
          ..
    make -j32 VERBOSE=1 && make install
}

main() {
    echo "---------------Tile fwk ops build begin----------------"
    g++ -v
    cmake --version
    env
    rm -rf ${BUILD_PATH}
    rm -rf ${OUTPUT_PATH}

    cd ${BASEPATH}
    mkdir -p ${OUTPUT_PATH}
    build_ops || { echo "Tile fwk ops build failed."; exit 1; }
    echo "---------------Tile fwk ops build successfully----------------"
    echo "---------------Begin to package nsa custom run opkg----------------"
    rm -rf ${BASEPATH}/run_pkg
    OPS_PKG_DIR=${BASEPATH}/run_pkg/packages/vendors/customize_ops/
    mkdir -p ${BASEPATH}/run_pkg/packages/vendors/customize_ops/
    cp ${BASEPATH}/scripts/help.info ${BASEPATH}/run_pkg
    cp ${BASEPATH}/scripts/install.sh ${BASEPATH}/run_pkg
    chmod -R 755 ${BASEPATH}/run_pkg/install.sh
    cp -r ${OUTPUT_PATH}/op_proto ${OPS_PKG_DIR}
    cp -r ${OUTPUT_PATH}/op_impl ${OPS_PKG_DIR}
    mkdir -p ${OPS_PKG_DIR}/op_impl/ai_core/tbe/customize_ops_impl

    MAKESELF_DIR=${ASCEND_HOME_PATH}/tools/op_project_templates/ascendc/customize/cmake/util/makeself
    cd ${BASEPATH}/run_pkg
    bash ${MAKESELF_DIR}/makeself.sh --header ${MAKESELF_DIR}/makeself-header.sh \
        --help-header ./help.info --tar-format posix  --gzip --complevel 4 --nomd5 --sha256 \
        ./ customize_ops_${pkg_suffix}.run "version:1.0" ./install.sh
    echo "---------------Finish to package nsa custom run opkg----------------"
}

main "$@"
