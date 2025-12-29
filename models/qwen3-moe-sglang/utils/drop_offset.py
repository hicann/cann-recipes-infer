# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
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

import os
import json
from argparse import ArgumentParser
from glob import glob
from tqdm import tqdm
from safetensors.torch import load_file, save_file


def drop_offset(model_path):
    input_path = model_path
    output_path = model_path
    safetensors_files = list(glob(os.path.join(input_path, "*.safetensors")))
    safetensors_files.sort()

    new_weight_map = {}
    for safetensors_file in tqdm(safetensors_files):
        file_name = os.path.basename(safetensors_file)
        state_dict = load_file(safetensors_file, device="cpu")
        new_state_dict = {}
        for weight_name, weight in state_dict.items():
            if "weight_offset" in weight_name:
                continue
            new_state_dict[weight_name] = weight
            new_weight_map[weight_name] = file_name

        new_safetensor_file = os.path.join(output_path, file_name)
        save_file(new_state_dict, new_safetensor_file)

    model_index_file = os.path.join(input_path, "quant_model_weights.safetensors.index.json")
    with open(model_index_file, "r") as f:
        model_index = json.load(f)
    model_index["weight_map"] = new_weight_map
    with open(model_index_file, "w", encoding="utf-8") as f:
        json.dump(model_index, f, indent=2, ensure_ascii=False, sort_keys=True)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    args = parser.parse_args()

    drop_offset(args.model_path)