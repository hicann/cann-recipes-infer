# ktransformers 核心代码 patch 集

基线提交号 `105359b4bf70073af9fab992ad29920666acccc9`（`github.com/kvcache-ai/ktransformers` 主仓，
被 tag `v0.7.0.post4` 与 `v0.7.1` 包含）。

10 个 patch 全部只改 `kt-kernel/` 子目录下的核心代码（`cpu_backend/`、`operators/`、`python/`、
`ext_bindings.cpp`），不含测试、不含 bench、不含任何 `.sh`，不触碰子模块与构建文件。

按编号顺序应用后：

| 闸门 | 期望值 |
|---|---|
| `git write-tree` | `393d8d5d99afef42449e4e804467c668d71a40c4` |
| `git diff --stat` 末行 | `16 files changed, 2080 insertions(+), 161 deletions(-)` |
| `git apply --check` offset / fuzz | `0` |
| 相对基线新增的文件 | 1 个：`operators/zerocopy_weights.hpp` |

`guide-scripts/02-fetch.sh` 逐一断言前三项。

同一套 patch 对 upstream `main` `c40722bf04c494f2492b7eb9e86ef01a4ede45b3`（2026-09-23 实测）
同样 10/10 零 offset 零 fuzz、改动量一致。基线取 `105359b` 是因为它被上述两个 tag 包含；
`main` 相对它的两个新提交只改 `kt-kernel/pyproject.toml` 的依赖上限与 `third_party/sglang` 指针，
本流程都不经过。

## 应用方式

```bash
git init ktransformers && cd ktransformers
git remote add origin https://github.com/kvcache-ai/ktransformers.git
git fetch --depth 1 origin 105359b4bf70073af9fab992ad29920666acccc9
git checkout FETCH_HEAD
git submodule update --init --depth 1 third_party/llama.cpp third_party/pybind11
for p in <本目录>/0*.patch; do git apply "$p"; done
```

保留 commit message 与作者信息用 `git am <本目录>/0*.patch`。

## 清单

| # | 提交号 | 标题 | 改动 |
|---|---|---|---|
| 0001 | `5378c88b5b` | [feat](kt-kernel): allocate CPU MXFP4 experts only for the offloaded subset | 7 files changed, 216 insertions(+), 31 deletions(-) |
| 0002 | `b5e0f8b2c0` | [perf](kt-kernel): stripe worker binding over CCDs, compact MXFP4 E8M0 scales, runtime phase timing | 6 files changed, 247 insertions(+), 78 deletions(-) |
| 0003 | `6499021e57` | [feat](kt-kernel): let the host framework inject the graph-capture probe | 2 files changed, 48 insertions(+), 5 deletions(-) |
| 0004 | `e18fb4d92f` | [feat](kt-kernel): public CPU-only drain for framework-managed overlap | 1 file changed, 40 insertions(+) |
| 0005 | `4ff48b77ee` | [feat](kt-kernel): worker 忙等时长改成 KT_WORKER_SPIN_MS（默认仍 50 ⇒ 行为不变） | 2 files changed, 40 insertions(+), 2 deletions(-) |
| 0006 | `5ff7514244` | [feat](kt-kernel): CPUInfer.run_inline —— 图路径下不走 TaskQueue 线程握手（KT_CPUINFER_INLINE，默认关） | 5 files changed, 168 insertions(+) |
| 0007 | `b75b0abb35` | [feat](kt-kernel): KT_FULLSET_LOAD —— 驻留专家也加载 CPU 权重，使运行时可降级（默认关） | 1 file changed, 93 insertions(+) |
| 0008 | `e7255b5910` | [feat](amx): 零拷贝权重 —— w13 指向 checkpoint 页缓存，w2 保留拷贝 | 5 files changed, 1087 insertions(+), 34 deletions(-) |
| 0009 | `b7999bb148` | [fix](amx): aligned_alloc 失败不再静默 + KT_ZEROCOPY_MLOCK 严格三态 | 4 files changed, 162 insertions(+), 13 deletions(-) |
| 0010 | `7c27ddf746` | [fix](amx): 零拷贝回落改抛错；顺带精简注释 | 3 files changed, 28 insertions(+), 47 deletions(-) |

## 各 patch 提供什么，改哪些文件

文件清单由 patch 自身生成，路径相对 `kt-kernel/`。

| # | 提供 | 改动的文件 |
|---|---|---|
| 0001 | 只为 offload 子集分配 CPU MXFP4 专家内存 | `ext_bindings.cpp`<br>`operators/amx/fp4-moe.hpp`<br>`operators/amx/moe_base.hpp`<br>`operators/common.hpp`<br>`python/experts_base.py`<br>`python/utils/amx.py`<br>`python/utils/loader.py` |
| 0002 | worker 绑核按 CCD 条带化；MXFP4 E8M0 scale 紧凑排布；运行期分阶段计时 | `cpu_backend/worker_pool.cpp`<br>`ext_bindings.cpp`<br>`operators/amx/fp4-moe.hpp`<br>`operators/amx/moe_base.hpp`<br>`operators/common.hpp`<br>`python/utils/amx.py` |
| 0003 | 由宿主框架注入图捕获探针 | `python/experts.py`<br>`python/experts_base.py` |
| 0004 | 公开 CPU-only drain 接口，供框架托管 overlap | `python/experts_base.py` |
| 0005 | worker 忙等时长改为 KT_WORKER_SPIN_MS，默认 50，行为不变 | `cpu_backend/worker_pool.cpp`<br>`cpu_backend/worker_pool.h` |
| 0006 | KT_CPUINFER_INLINE：图路径下不走 TaskQueue 线程握手，默认关 | `cpu_backend/cpuinfer.h`<br>`cpu_backend/task_queue.cpp`<br>`cpu_backend/task_queue.h`<br>`ext_bindings.cpp`<br>`python/experts_base.py` |
| 0007 | KT_FULLSET_LOAD：驻留专家同时加载 CPU 权重，默认关 | `python/utils/amx.py` |
| 0008 | 零拷贝权重：w13 直接指向 checkpoint 页缓存，w2 保留拷贝 | `operators/amx/fp4-moe.hpp`<br>`operators/amx/la/amx_buffers.hpp`<br>`operators/zerocopy_weights.hpp`<br>`python/utils/amx.py`<br>`python/utils/loader.py` |
| 0009 | aligned_alloc 失败不再静默；KT_ZEROCOPY_MLOCK 严格三态 | `operators/amx/moe_base.hpp`<br>`operators/amx/sft_moe.hpp`<br>`operators/common.hpp`<br>`operators/zerocopy_weights.hpp` |
| 0010 | 零拷贝回落路径改为抛错 | `operators/amx/fp4-moe.hpp`<br>`operators/common.hpp`<br>`operators/zerocopy_weights.hpp` |

## 服务端开关与 patch 的对应

`serve_lowlatency.sh` 的 `--additional-config` 里两个开关依赖本 patch 集：

| 开关 | 依赖 | patch |
|---|---|---|
| `zerocopy_w13: true` | `KT_ZEROCOPY_WEIGHTS`、`KT_ZEROCOPY_SCOPE` | 0008 / 0009 / 0010 |
| `dynamic_resident: true` | `KT_FULLSET_LOAD` | 0007 |

两个开关对 kt-kernel 都是普通环境变量，未打 patch 的 kt-kernel 会静默忽略它们。
`_common.sh` 因此探测进程实际 import 到的 kt-kernel，而不是只检查变量是否导出。
`06-verify.sh` 的「kt_kernel 解析」一项校验服务进程映射的 `kt_kernel` 来自 `$KT_SITE`。

## 相对基线新增的文件

```
operators/zerocopy_weights.hpp
```

## License

patch 内容以 Apache 2.0 发布，见上一级目录的 `LICENSE.txt`。
被修改的 ktransformers 源码遵循其上游许可证。
