# GPT-OSS Model Inference on NPU

## Overview
This example implements single-node single-batch inference on the **Atlas A2 series products** based on the [GPT-OSS](https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt_oss/modeling_gpt_oss.py) model from the Transformers library. The GPT-OSS-120B model can be deployed with 8 cards, while the GPT-OSS-20B model can be deployed on a single device.
- For detailed optimization points adopted in this example, please refer to [Optimization Practices for GPT-OSS Model Inference on Atlas A2 Series Products](../../docs/models/gpt-oss/gpt_oss_optimization.md).

The following sections describe in detail the steps to run the GPT-OSS inference example on NPU.

## Supported Hardware Models
<term>Atlas A2 Series Products</term>

## Environment Preparation

1. Install the CANN software package.

   The compilation and execution of this example depend on the CANN development kit and CANN binary operator package. The supported CANN software version is `CANN 9.0.0`.

   Download the `Ascend-cann-toolkit_${version}_linux-${arch}.run` and `Ascend-cann-A3-ops_${version}_linux-${arch}.run` packages from the [software package download page](https://www.hiascend.com/developer/download/community/result?module=cann&cann=9.0.0), and refer to the [CANN installation guide](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/900/softwareinst/instg/instg_0090.html?OS=Ubuntu&InstallType=localpack) for installation.

   - `${version}` indicates the CANN package version number, e.g., 9.0.0.
   - `${arch}` indicates the CPU architecture, e.g., aarch64 or x86_64.

2. Install Ascend Extension for PyTorch (torch_npu).

   Ascend Extension for PyTorch (torch_npu) is an adapter plugin that enables PyTorch to run on NPU. The supported version is `v26.0.0` and PyTorch `2.8.0`.

   Download the `torch_npu-2.8.0.post4-cp311-cp311-manylinux_2_28_${arch}.whl` installation package from the [software package download page](https://gitcode.com/Ascend/pytorch/releases/v26.0.0-pytorch2.8.0) and refer to the [torch_npu installation guide](https://www.hiascend.com/document/detail/zh/Pytorch/2600/configandinstg/instg/docs/zh/installation_guide/installation_via_binary_package.md) for installation.

   - `${arch}` indicates the CPU architecture, like aarch64 or x86_64.

3. Download source code of the project and install the required Python libraries.

   ```bash
   # Download the project source code (using the master branch as an example)
   git clone https://gitcode.com/cann/cann-recipes-infer.git

   # Install the required Python libraries
   cd cann-recipes-infer/models/gpt_oss
   pip3 install -r requirements.txt

   pip3 install -r requirements.txt

4. Configure the environment variables required for running the example.

   Modify the following fields in the `executor/scripts/set_env.sh` script:
   - `cann_path`: Installation path of the CANN software package, e.g., `/usr/local/Ascend/ascend-toolkit/latest`.

   > **Note:** HCCL-related configurations such as `HCCL_SOCKET_IFNAME` and `HCCL_OP_EXPANSION_MODE` can be customized in `executor/scripts/function.sh` by referring to the [collective communication documentation](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/900/maintenref/envvar/envref_07_0001.html).

## Model Weight Preparation

This example splits and adjusts the original weights of the open-source GPT-OSS models. Two original weight sets are provided:
- [GPT-OSS-20B weights](https://huggingface.co/openai/gpt-oss-20b/tree/main)
- [GPT-OSS-120B weights](https://huggingface.co/openai/gpt-oss-120b/tree/main)

Developers can choose based on their model tasks and download the original weights to a local path, e.g., `/data/models/gpt-oss-20b-bf16`.

> **Note:** The original weights are in mxfp4 format, but this inference script only supports bf16 format. Please convert them by referring to the [official code](https://github.com/openai/gpt-oss/blob/main/gpt_oss/torch/weights.py).

## Running Inference

1. Configure the weight files and YAML files to be loaded for inference execution.

   - Modify the `model_path` parameter in the YAML file.

     The `models/gpt_oss/config` directory already provides YAML examples with good performance for your reference. You can choose the corresponding YAML file based on the weights. Taking `gpt_oss_20b.yaml` as an example, modify the `model_path` parameter to point to the weight file path prepared in the [Model Weight Preparation](#model-weight-preparation) stage, e.g., `/data/models/gpt-oss-20b-bf16`.

     For more configuration details in the YAML file, see [YAML Parameter Description](../../docs/common/inference_config_guide.md).

   - Configure parameters in the `executor/scripts/infer.sh` script.

     For offline inference, set `--yaml` to the YAML file name in the `config` folder, e.g., `gpt_oss_20b.yaml`.
     For online inference, set `--mode` to `online`, `--pd-role` to `prefill` or `decode`, and you can specify the prefill/decode YAML files via `--p-yaml-name` and `--d-yaml-name`.

2. Prepare the input prompt.

   - Use the built-in prompt.

     This example already includes a built-in input prompt in `dataset/default_prompt.json`. If you directly use it, you can skip this step.

     Of course, you can also customize the prompt input in the `dataset/default_prompt.json` file.

   - Use a long-sequence prompt.

     By default, this example uses the built-in prompt. If you need to use a long-sequence prompt, perform the following steps:

     1. Modify the `dataset` parameter in the YAML file to `dataset: "LongBench"` to use the LongBench dataset as the long-sequence prompt.

     2. If your machine cannot access the internet, you need to manually download the dataset from [huggingface](https://huggingface.co/datasets/zai-org/LongBench/tree/main) to the `dataset/LongBench` directory (create it manually). The directory should contain `LongBench.py` and a `data` subdirectory, and you need to modify the dataset loading path in `LongBench.py`. If your machine has internet access, the example will automatically read the LongBench dataset online during execution, and no manual download is required.

     > **Note:** When using the LongBench dataset, the default task is text summarization. You can modify the default system prompt in the `build_dataset_input` function in `cann-recipes-infer/executor/utils/data_utils.py`.

3. Execute the unified inference script.

   The unified entry script is located at `executor/scripts/infer.sh` and is controlled by the following parameters:

   | Parameter | Meaning | Example Values |
   | --- | --- | --- |
   | `--model` | Model directory name, corresponding to a subdirectory under `models/` | `gpt_oss` |
   | `--mode` | Inference mode | `offline` / `online` |
   | `--yaml` | Offline mode: YAML file name | `gpt_oss_20b.yaml` |
   | `--pd-role` | Online mode: PD role | `prefill` / `decode` |
   | `--p-yaml-name` | Optional, online mode: prefill YAML file name; if not provided, defaults to `gpt_oss_pd/prefill.yaml` | `gpt_oss_pd/prefill.yaml` |
   | `--d-yaml-name` | Optional, online mode: decode YAML file name; if not provided, defaults to `gpt_oss_pd/decode.yaml` | `gpt_oss_pd/decode.yaml` |

   > For more configurations (IP addresses, etc.) in online mode, refer to [executor design document §5.1 Startup Methods](../../docs/design/executor_design.md#51-启动方式).

   **Method 1: Pass parameters via command line**
   ```shell
   # offline mode
   bash executor/scripts/infer.sh --model gpt_oss --yaml gpt_oss_20b.yaml
   # online mode
   bash executor/scripts/infer.sh --model gpt_oss --mode online --pd-role prefill
   # online mode (specify prefill/decode yaml)
   bash executor/scripts/infer.sh --model gpt_oss --mode online --pd-role prefill --p-yaml-name gpt_oss_pd/prefill.yaml --d-yaml-name gpt_oss_pd/decode.yaml
   ```

   To view parameter descriptions, run `bash executor/scripts/infer.sh --help`.

   **Method 2: Modify default values directly in the script and execute**
   Edit `executor/scripts/infer.sh` and change the default values of parameters such as `MODEL`, `MODE`, `YAML_FILE`, `PD_ROLE`, `P_YAML_NAME`, `D_YAML_NAME`, for example:
   ```shell
   MODEL=gpt_oss
   MODE=offline
   YAML_FILE=gpt_oss_20b.yaml
   ```
   Save the file and execute directly:
   ```shell
   bash executor/scripts/infer.sh
   ```

   > **Note:** Inference logs and results are saved under `models/gpt_oss/res/`.

   > **Important Points to Notes**
   > - Currently only supports a batch size of 1 for the prompt.
   > - Uses `eager` single-operator mode for inference by default.
   > - For the 20B model, single-device inference is supported; for the 120B model, 8-card inference (only TP splitting) is supported.
   > - The YAML file sets `enable_online_split_weight: True` by default. The model weights will be [split online](../../docs/common/online_split_weight_guide.md) to each device during loading, so offline splitting is not required.