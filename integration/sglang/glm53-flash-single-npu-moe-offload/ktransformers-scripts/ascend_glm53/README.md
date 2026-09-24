# ascend_glm53

GLM-5.3-Flash 跑在单个昇腾 910C die 上，路由专家 offload 到主机 DDR。
die 上是 INT8 W8A8，涵盖注意力、3 个 dense 层与常驻专家；其余专家以 MXFP4 常驻主机内存，
由 kt-kernel 的 `LLAMAFILE` MoE 计算。

[部署指南](../../README.md)中的 `02-fetch-patch.sh` 把本目录复制进
`kt-kernel/tools/ascend_glm53`，脚本在那份副本里调用。CANN、Python 环境和两份权重
均由该指南准备。

项目虚拟环境的直接依赖统一锁定在交付目录的 `requirements.txt`。`03-python-env.sh`
按该文件创建基础环境，`setup.sh deps` 再把同一文件作为 constraints 安装 SGLang 依赖，
并逐项核对锁定版本。torch_npu→torch、kt-kernel→torch 和 triton_ascend→scipy
三条运行时契约单独校验；不使用全局 `pip check`，避免把 CANN 通过 `PYTHONPATH`
暴露的厂商工具包误算成项目虚拟环境依赖。

## 配置项

这里只列出确需在进程启动前传给 runtime 的功能和资源配置。仓库、模型、构建产物等
路径由 `01-config.sh` 生成，不作为环境变量配置接口。

| 变量 | 默认值 | 含义 |
|---|---|---|
| `GLM53_PREFILL_STREAM` | `0` | 取 `1` 走流式 prefill，取 `0` 走 hybrid。标准配置是 `1` |
| `GLM53_NPU_DEVICE_ID` | `0` | 使用哪张 die，填容器内逻辑序号 |
| `GLM53_HOST` | `127.0.0.1` | 服务监听地址 |
| `GLM53_PORT` | `30013` | 服务端口 |
| `GLM53_NUM_GPU_EXPERTS` | `32` | die 上常驻的专家数 |
| `GLM53_MEM_FRACTION` | 流式 `0.95`，hybrid `0.85` | 静态显存占比 |
| `GLM53_MAX_TOTAL_TOKENS` | 流式 `40960`，hybrid 不设 | KV 池上限 |
| `GLM53_CHUNKED_PREFILL_SIZE` | 流式 `6144`，hybrid `8192` | 单次 prefill 的分块长度，须为 64 的正整数倍 |
| `GLM53_CONTEXT_LENGTH` | `32768` | 上下文长度 |
| `GLM53_MAX_RUNNING_REQUESTS` | `1` | 并发上限 |
| `GLM53_THREADPOOL_COUNT` | 自动取 NUMA 节点数 | CPU MoE 线程池个数 |
| `GLM53_CPUINFER` | 自动取线程池数乘 16，上限为核数的四分之三 | CPU MoE 工作线程总数 |
| `GLM53_KT_NUMA_NODES` | 不设 | CPU MoE 子池绑定的 NUMA 节点，写成 `0,1` |
| `GLM53_PIN_CORES` | 不设 | taskset 核号列表，须在启动时给出 |
| `GLM53_EAGER` | `0` | 取 `1` 关闭 decode 图捕获 |
| `KT_DYNAMIC_RESIDENT` | 流式下 `1` | 动态热专家，按实际路由更新常驻集合 |
| `KT_PREFILL_STREAM_THRESHOLD` | `512` | prompt 超过这个 token 数才走流式路径 |
| `KT_HOT_TAIL_TOKENS` | `512` | 从 prompt 末尾多少个 token 拟合 decode 热点集合 |
| `KT_STREAM_WARMUP` | `1` | 启动时预热流式路径 |
| `KT_SIDE_STREAM` | `1` | CPU MoE 主机回调放到第二条流上 |

检查最终启动命令使用 `./serve.sh --dry-run`，前台运行使用
`./serve.sh --foreground`；二者都是脚本选项，不新增环境变量。

`GLM53_MEM_FRACTION`、`GLM53_MAX_TOTAL_TOKENS`、`GLM53_CHUNKED_PREFILL_SIZE` 三项由
`GLM53_PREFILL_STREAM` 派生。先 source 再翻转该开关时，`glm53_env.sh` 按
`GLM53_DERIVED_FOR_STREAM` 戳记只重推自己派生过的值，显式设定的值保留。
全部 `KT_*` 在模块 import 时读取并冻结为全局，服务启动之后修改无效。
`SGLANG_MAMBA_CONV_DTYPE` 与本配方使用的四个 `SGLANG_OPT_*` 值由 `serve.sh` 固定，
属于已验证实现的一部分，不作为用户配置接口。

## 脚本

| 脚本名称 | 作用 | 调用命令 |
|---|---|---|
| `glm53_env.sh` | 解析全部环境变量，其余脚本 source 它 | `source glm53_env.sh`，查看解析结果用 `bash glm53_env.sh --show` |
| `setup.sh` | 构建入口。八个子命令各负责一段构建，每步开工前先查产物，已完成的跳过，因此可以反复重跑 | `./setup.sh all`，或 `./setup.sh {probe｜submodules｜deps｜sgl-kernel｜cann-ops｜kt-kernel｜gguf｜check}` |
| `serve.sh` | 组装启动命令并拉起服务 | `./serve.sh`，前台运行用 `./serve.sh --foreground` |
| `verify.sh` | 服务检验，对已运行的服务跑 11 项检查，只读日志与 HTTP 接口 | `./verify.sh`，交互客户端用 `./verify.sh chat` |
| `ask.sh` | 对已运行的服务发送一条真实文本请求，分开报告 TTFT 与客户端侧 decode 吞吐 | `./ask.sh --prompt-tokens 1024 --new-tokens 256` |
| `bench.sh` | 独占式 decode 吞吐测试；检查机器污染，自动起停服务并输出 JSON | `./bench.sh --name baseline` |
| `run_ppl.py` | 对运行中的服务做 teacher-forced 困惑度，窗口默认 4096 | `"$GLM53_PYTHON" run_ppl.py --limit 12 --out ppl.json` |
| `run_bench.py` | 基于 evalscope，验证 gsm8k 与 gpqa_diamond 两个数据集的精度 | `"$GLM53_ENV_ROOT/.venv-eval/bin/python" run_bench.py --dataset gsm8k` |

### Decode 吞吐测试

`ask.sh` 用于快速观察已经运行的服务。它从第一枚输出 token 之后开始计时，不把
prefill/TTFT 折入 decode token/s。参数全部通过命令行传入，完整选项见 `./ask.sh --help`。

正式记录使用 `bench.sh`。运行前先显式设置需要对照的 CPU、NUMA 和 resident-expert
配置，并确保配置端口上没有服务。脚本会拒绝替换已有服务，也会在发现其他 SGLang
进程或过高系统负载时拒绝给出可比较数字。

```bash
GLM53_THREADPOOL_COUNT=1 \
GLM53_CPUINFER=32 \
GLM53_KT_NUMA_NODES=0 \
GLM53_PIN_CORES=0-39 \
GLM53_NUM_GPU_EXPERTS=32 \
./bench.sh --name baseline \
  --prompt-tokens 630 \
  --short-tokens 64 \
  --long-tokens 256 \
  --warmup-tokens 256 \
  --repeats 4
```

上例的 CPU/NUMA 值对应 40 核、单 NUMA 的验证环境；在其他 CANNLab 规格上须按实际
拓扑选择节点和核号，baseline 与待测版本必须保持完全一致。

默认使用 PPL 同一份 WikiText 真实语料，先做一次完整 decode 预热。随后把相同 prompt
的 64/256-token 请求组成相邻测量对，交错先后顺序，并对每组 wall-time 差值取中位数，
以降低 TTFT 和单向热漂移对 decode 斜率的影响。JSON 同时记录 prompt 哈希、输出前缀
一致性、CPU/NUMA 配置、依赖版本、代码提交、机器污染、streaming 实际命中和 fallback
计数；只有 `verdict=clean` 的结果才能互相比较。`--synthetic-prompt` 只用于历史诊断，
重复文本会夸大热专家命中率。该流程与旧脚本的 8-token warmup 不同，baseline 和待测版本
必须都用当前脚本重测，不能把旧结果混入比较。

## 注意事项

- `run_ppl.py` 须用 `"$GLM53_PYTHON"` 调用，不要用 `./run_ppl.py`。shebang 会取到 PATH
  上的 python，而 transformers 只安装在项目 venv 中。
- `run_bench.py` 须用另建的 eval venv 中的 python 调用。evalscope 会改动项目 venv 里
  钉死的 torch 与 torch_npu 版本。
- `glm53_env.sh` 保留调用方设置的标准 HTTP(S) 下载代理，并把 loopback 与
  `GLM53_HOST` 加入 `NO_PROXY/no_proxy`；内置客户端也显式直连，健康检查和推理请求
  不会经过下载代理。
- `bash glm53_env.sh --show` 与 `./setup.sh probe` 会短暂打开 die 0。`npu-smi` 对 A2 与
  A3 打印同一型号名，识别型号只能依靠 `torch.npu.get_device_name(0)`。die 被其他任务
  占用时，先 `export GLM53_SOC=ascend910_93` 跳过探测。
- 流式路径中的异常均被捕获并回退 hybrid，从外部无法分辨，服务仍能应答，`verify.sh`
  前几项同样全部 PASS。判定依据只有日志，且需使用超过 512 token 的 prompt。
  `grep -c 'inline resident' $GLM53_LOG_DIR/serve.log` 应大于 0，
  `grep -cE 'streaming failed|hybrid fallback'` 应等于 0。
- `serve.sh` 中的 `--page-size 64`、`--disable-shared-experts-fusion`、
  `--disable-radix-cache` 形似调优开关，删除后服务无法启动，或启动后精度错误。
- GGUF 缺层不会报错。kt-kernel 对该层加载零个专家，模型仍能应答，但输出无意义。
  `serve.sh` 启动前与 `./setup.sh check` 均校验文件个数，期望 42 层，即第 3 层到第 44 层。
- 手工启动的服务按端口停止，`pkill -f -- "[-]-port ${GLM53_PORT:-30013}"`。方括号用于
  避免模式匹配到发起该命令的 shell 自身。`bench.sh` 将自己的服务放进独立进程会话，
  退出时只清理该会话，不按端口清理共享账号下的进程。
