# Distributed Embedding

基于 Ascend C SIMT 和 SHMEM 的分布式哈希表容器，提供 `Init`、`Insert`、`Search` 和 `Finalize` 接口。单卡在本地哈希表完成插入和查询，多卡按 key 的哈希值分区，通过 SHMEM 访问对应 PE。

多卡模式配置 `hostTableSize > 0` 时，本卡负责的本地输入与其他 PE 发来的输入都会写入本卡 host 哈希表（需保证 host 表容量足够）。每个 PE 每轮 `Insert` 的输入数须不超过 `maxKeysPerPe`，本地暂存与远程发送共用该容量；调用前应确保所有 PE 都满足此限制。接收缓冲区在每轮发送前清空并同步，避免重放旧记录。host 插入阶段使用多核、多线程，通过对映射后的 host 桶 key 执行 SIMT CAS 原子占桶，避免不同 key 的哈希冲突覆盖。key 占桶后单独写入 value，查询须在插入完成后进行；同一轮重复 key 的不同 value 不保证写入顺序。

算子原位于 `ops/ascendc/src/distributed_embedding`，现使用独立 CMake 工程构建，入口为本目录或 `test` 目录。

## 环境

当前构建默认使用 `dav-3510`，已在 Ascend950PR_957b、CANN 9.1.0、SHMEM 1.7.0 环境验证。

测试脚本默认从 `$HOME/ascend/cann` 和 `$HOME/ascend/shmem/latest/shmem` 加载依赖，也可以显式指定：

```bash
export ASCEND_HOME_PATH=/home/luyoubing/ascend/cann
export SHMEM_ROOT=/home/luyoubing/ascend/shmem/latest/shmem
```

`SHMEM_ROOT` 指向包含 `include/shmem.h` 和 `lib/libshmem.so` 的目录。脚本自动加载指定 CANN 的环境，并设置 SHMEM 动态库搜索路径。

## 编译和测试

支持在 x86_64 和 aarch64 主机上原生编译，脚本自动选择 CANN 安装目录中对应的 `<主机架构>-linux` 编译器、头文件和库目录。

在本算子目录执行：

```bash
# 编译单卡、双卡测试，不启动设备测试。
bash test/build.sh

# 编译并运行单卡测试，默认使用设备 0。
bash test/d2d/run_rank1.sh

# 指定其他设备运行单卡测试，SHMEM rank 始终为 0。
DEVICE_ID=1 bash test/d2d/run_rank1.sh

# 编译并运行双卡测试，默认使用设备 0、1。
RANK0_DEVICE=0 RANK1_DEVICE=1 bash test/d2d/run_rank2.sh
```

构建产物默认位于 `test/build`。可用 `BUILD_DIR`、`BUILD_JOBS` 分别指定构建目录和编译并行度；用 `SHMEM_LIBRARY_DIR` 覆盖 SHMEM 库目录；用 `SHMEM_IP_PORT` 指定通信地址。双卡运行的超时由 `TEST_TIMEOUT_SECONDS` 控制（默认 180 秒），日志位于 `test/build/d2d_rank2_logs` 或 `test/build/d2h_rank2_logs`。

单卡测试检查初始化、插入、命中和未命中查询、资源释放，成功输出 `PASS: Init, Insert, Search, and Finalize succeeded`。双卡测试为每个 PE 构造本地和远端 key，检查跨 PE 插入和查询。

D2H 用例位于 `test/d2h`，将 Device 表限制为 1 个槽位，并启用 Host 对称表。插入的数据超过 Device 表容量，查询必须回退到 Host 表才能全部命中。单卡还检查未命中返回值，双卡覆盖两个 PE 的本地和远端分区。SHMEM 必须启用 HOST_SIDE 支持；当前 Ascend 950 环境使用 `bash scripts/build.sh -soc_type Ascend950 -cann` 构建 SHMEM。

```bash
bash test/d2h/run_rank1.sh
RANK0_DEVICE=0 RANK1_DEVICE=1 bash test/d2h/run_rank2.sh
# 仅编译四个目标：
bash test/build.sh d2d_rank1 d2d_rank2 d2h_rank1 d2h_rank2
```

也可以从算子目录构建库和四个测试程序：

```bash
source "$ASCEND_HOME_PATH/set_env.sh"
cmake -S . -B build \
    -DASCEND_HOME_PATH="$ASCEND_HOME_PATH" \
    -DSHMEM_ROOT="$SHMEM_ROOT"
cmake --build build -j8
```

如仅需构建算子库，可在配置时增加 `-DDISTRIBUTED_EMBEDDING_BUILD_TESTS=OFF`。

## 迁移验证

2026-09-10，使用上述 `/home/luyoubing/ascend` 环境完成验证：

- `test/build.sh`：`test_rank1`、`test_rank2` 均编译成功。
- 算子目录的独立 CMake 入口：算子库及两个测试目标均编译成功。
- `DEVICE_ID=0` 和 `DEVICE_ID=1`：分别运行单卡用例，均通过。
- 双卡用例本次仅验证编译，未执行双卡运行。

迁移保留原 host/kernel 实现，补充数学库链接、统一依赖路径，并修正单卡测试的设备编号与 SHMEM rank 映射。
