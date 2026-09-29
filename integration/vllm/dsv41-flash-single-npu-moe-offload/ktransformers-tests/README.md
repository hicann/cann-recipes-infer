# kt-kernel MXFP4 CPU-offload 数值校验

纯 CPU 运行，不打开 NPU。校验第 3 步自建的 kt-kernel 在 MXFP4 CPU offload 路径上的数值正确性。

## 前置

第 1 步已 `source dsv41.env`，第 2、3 步已完成（`$KT_SITE` 下已装 kt-kernel，`$MODEL` 指向权重）。

## 运行

```bash
env -u LD_PRELOAD PYTHONPATH="$KT_SITE" python3 ktransformers-tests/dsv41_mxfp4_offload_check.py
```

## 校验内容

| 项 | 内容 |
|---|---|
| E8M0 字节回环 | 从 kt `BufferB` 读回的 scale 与 safetensors 原始字节逐字节相同；正控制翻转一个字节，必须报出且只报出一处不符 |
| 前向数值 | `KTMoEWrapper(method="MXFP4", swiglu_limit=10)` 经 `forward()` 的 pinned-buffer 路径，对照独立 torch 参考实现（E2M1 LUT × 2^(byte−127)、DSV4.1 SwiGLU clamp、sqrtsoftplus 归一化 top-k × route_scale 1.5）；`gpu_experts_mask` 只留子集在 CPU 上，GPU 侧专家贡献 0 |
| 双档阈值 | 先定 floor 再比阈值，A（对 fp32 参考）与 B（对 bf16 模拟）两档都必须成立 |
| 变异臂 | nibble 顺序、单个专家整个 w2 scale +1、去掉 clamp 三项必须被判红；单字节 scale +1 是检测下限探针，只报不判 |

## 预期

末行 `VERDICT: PASS`，退出码 0。任一档阈值不成立或变异臂没被判红时输出 `VERDICT: FAIL`。
加 `--roundtrip-only` 时不跑前向也不跑变异臂，判据只剩各层的字节回环与权重落点。

## 可调参数

| 参数 | 默认 | 含义 |
|---|---|---|
| `--layers` | `1,3` | 参与校验的层号，逗号分隔 |
| `--tokens` | `1,4,32` | 每层跑的 token 数，逗号分隔 |
| `--threads` | `32` | cpuinfer 线程数 |
| `--numa` | `1` | NUMA 节点数；`--nodes` 给了则被覆盖 |
| `--nodes` | 空 | 显式指定 NUMA 节点，逗号分隔 |
| `--tp` | `1` | `threadpool_count` |
| `--n-cpu` | `0` | 改用前 N 个专家，量带宽用；N 至少 2（`--roundtrip-only`）或 4（全量）|
| `--cpu-ids` | 空 | 显式指定留在 CPU 上的专家 id；至少 2 个（`--roundtrip-only`）或 4 个（全量）|
| `--seed` | `0` | 随机种子 |
| `--x-std` | `1.0` | 输入激活的标准差 |
| `--roundtrip-only` | 关 | 只跑 E8M0 字节回环与权重落点审计，跳过前向与变异臂 |

权重目录取自环境变量 `MODEL`，未设时用 `/workspace/models/DeepSeek-V4.1-Flash`。
`KT_MXFP4_COMPACT_SCALES` 未设时按 `1` 处理，脚本首行会打印实际取值。
