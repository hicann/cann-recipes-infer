# LongCat-Next 模型在 NPU 上推理

## 概述

本样例基于 LongCat-Next 开源代码进行迁移，并完成对应的 NPU 优化适配，覆盖 Paged Attention 缓存管理、TP/EP 多卡并行、多模态生成图模式加速与官方生成状态机接入。

- HuggingFace: [meituan-longcat/LongCat-Next](https://huggingface.co/meituan-longcat/LongCat-Next)
- 架构: 原生多模态（文本 + 图像 + 语音）
- 任务: 文本 / 图像理解 / 图像生成 / 语音理解 / 语音合成 / 语音复刻

## 支持的产品型号

<term>Atlas A3 系列产品</term>

## 环境准备

| 组件 | 版本 |
| --- | --- |
| CANN | 9.1.0 |
| Python | 3.12 |
| PyTorch | 2.8.0 |
| torch_npu | 2.8.0 |

1. 安装 CANN 软件包：从 [软件包下载地址](https://ascend.devcloud.huaweicloud.com/artifactory/cann-run-release/software/master) 下载 `Ascend-cann-toolkit_${version}_linux-${arch}.run` 与 `Ascend-cann-A3-ops_${version}_linux-${arch}.run`，按 [CANN 安装文档](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/910/softwareinst/instg/instg_0001.html?Mode=PmIns&OS=Debian&Software=cannToolKit) 操作。
   - `${version}`：CANN 包版本号，如 `9.1.0`
   - `${arch}`：CPU 架构，如 `aarch64` / `x86_64`

2. 安装 Ascend Extension for PyTorch（torch_npu）：从 [软件包下载地址](https://gitcode.com/Ascend/pytorch/releases/v26.0.0-pytorch2.8.0) 下载 `v2.8.0.post4` whl 包，按 [torch_npu 安装文档](https://www.hiascend.com/document/detail/zh/Pytorch/2600/configandinstg/instg/docs/zh/installation_guide/installation_via_binary_package.md) 操作。

3. 下载项目源码：

```bash
git clone https://gitcode.com/cann/cann-recipes-infer.git
cd cann-recipes-infer
```

4. 配置环境并安装依赖：编辑 `executor/scripts/set_env.sh`，把 `cann_path` 改为本机 CANN 安装路径（例如 `/usr/local/Ascend/ascend-toolkit/latest`）；多节点部署时按 rank 顺序填 `IPs`，单节点忽略。该脚本内部会执行 `source $cann_path/bin/setenv.bash`，无需再手动 source CANN 环境。随后：

```bash
source executor/scripts/set_env.sh
pip3 install -r models/longcat_next/requirements.txt
```

## 权重与输入准备

从 HuggingFace 仓库（[meituan-longcat/LongCat-Next](https://huggingface.co/meituan-longcat/LongCat-Next)）下载完整权重目录到本地（例如 `/path/to/LongCat-Next`），并将 models/longcat_next/config目录下yaml文件中的 `model_config.model_path` 改为该路径。

本样例直接使用官方权重，无需转换，也无需替换权重目录中的任何文件。

## 推理执行

1. 配置 yaml：`model_config.model_path` 改为本地权重路径（必改）；

   模型特有参数位于 `model_config.custom_params.multimodal` 下，yaml 内已给出可直接使用的默认值：

   | 参数 | 说明 |
   | --- | --- |
   | `request_file` | 请求输入路径，相对 `models/longcat_next/` 解析；yaml 中默认为 `requests/text.json`，可照此命名，也可改成自己的文件名或绝对路径，格式见下文。 |
   | `head_tp_size` | 图像/语音生成头的 TP 切分数；`1` 表示单 owner 参考实现。 |
   | `head_exe_mode` | 生成头执行模式，可选 `eager`、`ge_graph` 或 `npugraph_ex`。 |
   | `enable_thinking` | 聊天模板 thinking 开关，仅文本输出任务生效。 |

   随附配置为示例，并行度由 `world_size` 与各 `tp_size` 字段决定：
   - `config/longcat_next_multimodal_tp4_ep4.yaml` — 多模态示例（4 卡 TP4 + EP4）
   - `config/longcat_next_text_tp4_ep4_eager.yaml` — 纯文本示例（框架通用流程）

   文本配置不含 `custom_params.multimodal`，不进入本样例的多模态适配，其输入方式见第 2 步末尾。


   本样例入口由 `data_config.dataset` 决定，需保持 `longcat_multimodal`；改为其他值而 yaml 内仍保留 `custom_params.multimodal` 会直接报错，删除该段后回落到框架的通用流程。

2. 准备推理输入：请求输入需使用者自行创建 JSON文件（推荐放在 `models/longcat_next/requests/` 下，如 `requests/text.json`），并把路径填进 yaml 的 `model_config.custom_params.multimodal.request_file`。先创建清单目录：`mkdir -p models/longcat_next/requests`。

 每条请求的字段如下表所示：

   | 字段 | 必填 | 说明 |
   | --- | --- | --- |
   | `id` | 是 | 请求标识，限 `[A-Za-z0-9_-]`，不超过 80 字符，同一清单内唯一；同时是结果子目录名 |
   | `task` | 是 | 任务类型，取值见下表 |
   | `prompt` | 视任务 | `text`、`image_generation`、`speech_synthesis` 必填，其余任务可省略 |
   | `system` | 否 | 系统提示词；含 `reference_audio` 的任务由代码按官方模板填充 system |
   | `enable_thinking` | 否 | 聊天模板 thinking 开关，仅文本输出任务可显式指定 |
   | `media` | 视任务 | 素材路径，键名按任务固定，相对该清单所在目录解析，也可写绝对路径 |
   | `generation` | 否 | 覆盖采样参数，见下文 |
   | `seed` | 否 | 随机种子，缺省取 `data_config.seed` |

   各任务的 `media` 键与产物：

   | task | media 键 | 产物 |
   | --- | --- | --- |
   | `text` | — | `text.txt` |
   | `image_understanding` | `image` | `text.txt` |
   | `image_generation` | — | `image*.png` |
   | `audio_to_text` | `audio` | `text.txt` |
   | `audio_to_audio` | `audio`、`reference_audio` | `audio*.wav` |
   | `speech_synthesis` | `reference_audio` | `audio*.wav` |

   JSON文件示例（假设清单位于 `requests/`；素材需自备，下例用绝对路径占位）：

   ```json
   {
     "schema_version": 1,
     "requests": [
       {
         "id": "text_001",
         "task": "text",
         "prompt": "你是一个小说家，请续写下面的故事：……",
         "generation": {"max_new_tokens": 2048, "do_sample": true, "repetition_penalty": 1.0}
       },
       {
         "id": "audio_to_audio_001",
         "task": "audio_to_audio",
         "prompt": "请用参考音频里的声音，说出下面的内容：……",
         "media": {"audio": "/path/to/your/audio.wav", "reference_audio": "/path/to/your/reference.wav"},
         "generation": {"max_new_tokens": 2048, "do_sample": true, "temperature": 0.2, "top_k": 20, "top_p": 0.85}
       }
     ]
   }
   ```

   - `media` 的键必须与任务要求的完全一致，多写或少写都会报错；素材文件不存在直接报错；路径中不能包含 `<longcat_…>` 控制标记
   - `generation` 按请求覆盖 yaml 内 `custom_params.multimodal.generation` 的同名默认值，可用的采样参数为 `max_new_tokens`、`do_sample`、`temperature`、`top_k`、`top_p`、`repetition_penalty`，也可用 `visual_generation_config` / `audio_generation_config` 覆盖生成头参数（如 `token_h`、`token_w`、`cfg_scale`）
   - 单条请求的 `max_new_tokens` 不得超过 yaml 的 `scheduler_config.max_new_tokens`（该字段同时是 KV 预留），实际运行的输出上限取清单内的最大值
   - `enable_thinking` 写在请求顶层，写进 `generation` 会报错；`image_generation`、`audio_to_audio`、`speech_synthesis` 不接受显式 `enable_thinking`
   - 清单内的请求按顺序串行执行；出现 `image_generation` 时启用 CFG（`scheduler_config.batch_size: 2`）
   - 音频输入由官方 processor 按配置重采样，生成音频输出为官方 24 kHz 配置
   - 素材需自备：本样例不随仓提供素材。官方 HuggingFace 仓库（[meituan-longcat/LongCat-Next](https://huggingface.co/meituan-longcat/LongCat-Next)）的 `assets/` 目录提供示例素材（`math1.wav`、`system_audio.wav`、`vc_zh3.wav`、`book.png` 等），按需下载即可；图像支持 PNG/JPG，音频支持 WAV 等常见格式
   - 使用文本配置（`config/longcat_next_text_*.yaml`）时输入方式不同：走框架通用流程，需在 `data_config.dataset_path` 指向的目录（默认为 `models/longcat_next/requests/`）下放置 `default_prompt.json`，内容形如 `{"text": "你的 prompt"}` 或 `{"text": ["prompt 1", "prompt 2"]}`，格式与仓库根 `dataset/default_prompt.json` 一致

3. 执行统一推理脚本。

   统一入口 `executor/scripts/infer.sh` 通过 `--model` / `--yaml` 指定模型与配置（本样例仅支持离线推理，`--mode` 默认 `offline`）：

   | 参数 | 含义 | 取值示例 |
   | --- | --- | --- |
   | `--model` | 模型目录名，对应 `models/` 下的子目录 | `longcat_next` |
   | `--yaml` | `config/` 下的 yaml 文件名 | `longcat_next_multimodal_tp4_ep4.yaml` |

   ```bash
   bash executor/scripts/infer.sh --model longcat_next --yaml longcat_next_multimodal_tp4_ep4.yaml
   ```

   如需查看参数说明，可执行 `bash executor/scripts/infer.sh --help`。

4. 查看结果：日志与结果保存在 `models/longcat_next/res/<日期>/longcat_next_<yaml 名>/` 下，`log_<rank>.log` 为各 rank 日志，`results/<请求 id>/` 为每条请求的产物，包含 `text.txt`（文本输出）、`image*.png`（图像生成）、`audio*.wav`（语音合成）与 `result.json`（请求参数、耗时与产物列表）。使用文本配置时结果为日志形式，在 `log_0.log` 中查看 `Request …: outputs:` 与 `Finished inference` 行。

   如需采集性能数据，将 yaml 内 `model_config.enable_profiler` 置为 `true` 后重新拉起，采集方式与结果目录同仓库其他样例。

## 优化点参考

本样例主干与 LongCat-Flash 同源，优化点与性能 Benchmark 参见[基于Atlas A3训练/推理集群的LongCat-Flash模型推理性能优化实践](../../docs/models/longcat_flash/longcat_flash_optimization.md)。
