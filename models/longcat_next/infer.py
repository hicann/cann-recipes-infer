# coding=utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
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


import argparse
import logging
import os
import sys
from pathlib import Path

import yaml

# The offline launcher adds the repository root to PYTHONPATH in set_env.sh.
# Drop this script's directory from module lookup to avoid resolving imports
# through its nested models/ directory.
_MODEL_DIR = Path(__file__).resolve().parent
_REPOSITORY_ROOT = _MODEL_DIR.parents[1]
sys.path[:] = [entry for entry in sys.path if Path(entry or ".").resolve() != _MODEL_DIR]


def main():
    parser = argparse.ArgumentParser(description="LongCat-Next inference")
    parser.add_argument("--yaml_file_path", required=True)
    args = parser.parse_args()
    with open(args.yaml_file_path, encoding="utf-8") as stream:
        settings = yaml.safe_load(stream)
    if settings.get("data_config", {}).get("dataset", "default") != "longcat_multimodal":
        if settings.get("model_config", {}).get("custom_params", {}).get("multimodal"):
            raise ValueError("Multimodal request_file requires data_config.dataset: longcat_multimodal")
        # Normalize repository-relative dataset paths regardless of the shell's
        # current directory before handing control to the common offline entry.
        os.chdir(_REPOSITORY_ROOT)
        from executor.offline.infer import main as native_main
        native_main()
        return 0

    from models.longcat_next.models.model_infer import run_multimodal
    from executor.core.config import InferenceConfig
    from executor.utils.logging_config import setup_logging

    setup_logging()
    local_rank = int(os.getenv("LOCAL_RANK", "0"))
    global_rank = local_rank + int(os.getenv("RANK_OFFSET", "0"))
    config = InferenceConfig.from_dict(settings, global_rank=global_rank, local_rank=local_rank)
    config.model_config.output_path = os.path.join(os.getenv("WORK_DIR", "."), os.getenv("RES_PATH", ""))
    logging.getLogger(__name__).info("Inference Configuration: %s", config)
    run_multimodal(config, args.yaml_file_path)
    return 0


if __name__ == "__main__":
    main()
