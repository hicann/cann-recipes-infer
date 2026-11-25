# coding=utf-8
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#!/bin/bash
SCRIPT_PATH=$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
SET_ENV_ABS_PATH="${SCRIPT_PATH}/../../../executor/scripts/set_env.sh"
SET_ENV_ABS_PATH=$(realpath "${SET_ENV_ABS_PATH}")
source ${SET_ENV_ABS_PATH}

INPUT_FP8_HF_PATH=""
OUTPUT_HF_PATH=""
QUANT_MODE=""


while [[ $# -gt 0 ]]; do
    case $1 in
        --input_fp8_hf_path)
            INPUT_FP8_HF_PATH="$2"
            shift 2
            ;;
        --output_hf_path)
            OUTPUT_HF_PATH="$2"
            shift 2
            ;;
        --quant_mode)
            QUANT_MODE="$2"
            shift 2
            ;;
        *)
            echo "Unknown Parameters: $1"
            echo "Usage: $0 --input_fp8_hf_path <input_path> --output_hf_path <output_path> --quant_mode <mode>"
            echo "Supported Quant Mode: bfloat16, w8a8c16, w8a8c8"
            exit 1
            ;;
    esac
done


if [[ -z "$INPUT_FP8_HF_PATH" ]] || [[ -z "$OUTPUT_HF_PATH" ]] || [[ -z "$QUANT_MODE" ]]; then
    echo "Usage: $0 --input_fp8_hf_path <input_path> --output_hf_path <output_path> --quant_mode <mode>"
    echo "Supported Quant Mode: bfloat16, w8a8c16, w8a8c8"
    exit 1
fi


case "${QUANT_MODE,,}" in
    bfloat16)
        echo "Convert to bfloat16 weights..."
        python utils/convert_model.py \
            --input_fp8_hf_path "$INPUT_FP8_HF_PATH" \
            --output_hf_path "$OUTPUT_HF_PATH"
        ;;
    w8a8c16)
        echo "Convert to w8a8c16 weights..."
        python utils/convert_model.py \
            --input_fp8_hf_path "$INPUT_FP8_HF_PATH" \
            --output_hf_path "$OUTPUT_HF_PATH" \
            --w8a8
        ;;
    w8a8c8)
        export QUANT_URL=https://cann-ai.obs.cn-north-4.myhuaweicloud.com/cann-quantization/DeepSeek-R1/mla_c8_scale.tar.gz
        mkdir -p ./quantization

        # Download the quantization zip file
        if ! wget --no-check-certificate -P ./quantization "$QUANT_URL"; then
            echo "Error: Failed to download quantization parameters from $QUANT_URL"
            exit 1
        fi

        # Unzip the file
        echo "Extracting quantization parameters..."
        if ! tar -zxvf "./quantization/mla_c8_scale.tar.gz" -C ./quantization; then
            echo "Error: Failed to extract ./quantization/mla_c8_scale.tar.gz"
            exit 1
        fi

        echo "Convert to w8a8c8 weights..."
        python utils/convert_model.py \
            --input_fp8_hf_path "$INPUT_FP8_HF_PATH" \
            --output_hf_path "$OUTPUT_HF_PATH" \
            --w8a8 \
            --c8 \
            --quant_param_path "./quantization/mla_c8_scale"
        ;;
    *)
        echo "Error: Unsupported Quant_mode: $QUANT_MODE"
        echo "Supported Mode: bfloat16, w8a8c16, w8a8c8"
        exit 1
        ;;
esac

echo "Output path: $OUTPUT_HF_PATH"
