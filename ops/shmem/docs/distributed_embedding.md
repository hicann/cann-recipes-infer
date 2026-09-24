# DistEmbeddingContainer（分布式 Embedding 哈希表容器）

## 产品支持情况

| 产品                        | 是否支持 |
| --------------------------- | :------: |
| Ascend 950PR/Ascend 950DT   |    √     |

> **说明：**<br>
> 已在 Ascend950PR_957b、CANN 9.1.0、SHMEM 1.7.0 环境完成验证，默认编译架构为 `dav-3510`。

## 功能说明

`DistEmbeddingContainer` 是基于 Ascend C SIMT 编程范式和 SHMEM 对称内存通信实现的分布式哈希表容器，提供 `Init`、`Insert`、`Search`、`Finalize` 四个生命周期接口，用于分布式 Embedding 场景下的 key-value 存取。

-   **哈希分区**：key 经 MurmurHash3 计算哈希值，`hash % nPes` 决定 key 归属的 PE（卡），`hash % tableSize` 决定 key 在哈希表中的桶位置；哈希冲突通过开放寻址（线性探测）解决，多核多线程并发写入通过 SIMT CAS 原子占桶保证正确性。

-   **单卡模式**（`nPes = 1`）：所有插入和查询均在本地 Device 哈希表上完成，不涉及跨卡通信。

-   **多卡模式**（`nPes > 1`）：插入时按 key 哈希值分发，属于本卡的 key 直接写入本地 Device 哈希表，属于远端卡的 key 通过 SHMEM 远程写入（`put`）目标 PE 的接收缓冲区，随后各卡从接收缓冲区提取属于自己的记录完成插入；查询时按 key 哈希值定位目标 PE，本地直接读取，远端通过 SHMEM 远程读取（`get`）目标 PE 的哈希表。

-   **分层存储**（可选，`hostTableSize > 0`）：额外分配 Host 侧对称内存哈希表。多卡插入时，本卡负责的记录（本卡本地输入与其他 PE 发来的输入）除写入 Device 表外，还会写入本卡 Host 表；查询时先查 Device 表，未命中再查 Host 表，适用于单卡 Device 显存放不下全量 Embedding 的场景。

## 接口总览

| 接口                                                             | 功能                                             | 多卡下是否为集合操作 |
| ---------------------------------------------------------------- | ------------------------------------------------ | :------------------: |
| `DistEmbeddingContainer(...)`                                    | 构造容器，配置表容量、PE 拓扑与通信参数          |          否          |
| `Init()`                                                         | 初始化设备、Stream、SHMEM、对称内存与哈希表      |          是          |
| `Insert(keys, values, nums)`                                     | 将 key-value 对写入分布式哈希表                  |          是          |
| `Search(keys, values, nums)`                                     | 按 key 查询 value，未命中返回 `Tvalue` 最大值    |          是          |
| `Finalize()`                                                     | 释放对称内存等资源并退出 SHMEM                   |          否          |
| `GetMyPe()` / `GetNPes()` / `GetTableSize()`                     | 查询本 PE 编号、PE 总数、Device 表实际桶数       |          否          |

## 函数原型

头文件：[op_host/dist_embedding_container.h](../distributed_embedding/op_host/dist_embedding_container.h)

```cpp
template <typename Tkey, typename Tvalue>
class DistEmbeddingContainer {
public:
    // 构造函数一：不启用 Host 侧分层存储
    DistEmbeddingContainer(uint64_t tableSize, int32_t myPe, int32_t nPes,
                           int32_t deviceOffset, uint32_t maxKeysPerPe,
                           const char* ipPort);

    // 构造函数二：启用 Host 侧分层存储
    DistEmbeddingContainer(uint64_t tableSize, uint64_t hostTableSize,
                           int32_t myPe, int32_t nPes,
                           int32_t deviceOffset, uint32_t maxKeysPerPe,
                           const char* ipPort);

    ~DistEmbeddingContainer();  // 析构时自动调用 Finalize

    void Init();
    void Insert(const Tkey* keys, const Tvalue* values, size_t nums);
    void Search(const Tkey* keys, Tvalue* values, size_t nums);
    void Finalize();

    int32_t GetMyPe() const;
    int32_t GetNPes() const;
    uint64_t GetTableSize() const;
};
```

配置结构体 `DistEmbeddingOptions`（已定义，其对应的构造函数重载当前版本暂未提供实现，请使用上述两个构造函数）：

```cpp
struct DistEmbeddingOptions {
    uint64_t tableSize;       // Device 表期望容量
    int32_t myPe;             // 本 PE 编号
    int32_t nPes;             // PE 总数
    int32_t deviceOffset;     // 设备号偏移
    uint32_t maxKeysPerPe;    // 每 PE 单轮 Insert 的最大输入数
    const char* ipPort;       // SHMEM 通信地址
};
```

当前已显式实例化、可直接使用的模板组合：

-   `DistEmbeddingContainer<unsigned int, size_t>`：32 位 key、64 位 value。
-   `DistEmbeddingContainer<long long, size_t>`：64 位 key、64 位 value。

## 参数说明

> **说明：**<br>
> PE（Processing Element）指 SHMEM 通信域中的一个进程端点，通常一个进程（一张卡）对应一个 PE。`tableSize` 表示 Device 表的期望元素容量，`hostTableSize` 表示 Host 表的期望元素容量。

### 模板参数

-   **Tkey**（`typename`）：key 类型，仅支持 32 位或 64 位整型（如 `unsigned int`、`long long`）。
-   **Tvalue**（`typename`）：value 类型，大小须为 4 或 8 字节（如 `size_t`）。

### 构造参数

-   **tableSize**（`uint64_t`）：Device 侧哈希表的期望元素容量，必选参数。容器按 0.75 的负载因子预留桶位，实际分配桶数约为 `tableSize / 0.75`，可通过 `GetTableSize()` 查询。

-   **hostTableSize**（`uint64_t`）：Host 侧分层哈希表的期望元素容量。传 `0` 表示不启用 Host 分层存储；大于 `0` 时启用，实际桶数同样按 0.75 负载因子预留。

-   **myPe**（`int32_t`）：本进程的 PE 编号，取值范围为 \[0, nPes\)。

-   **nPes**（`int32_t`）：PE 总数。`nPes = 1` 为单卡模式；`nPes > 1` 为多卡分布式模式。

-   **deviceOffset**（`int32_t`）：设备号偏移，实际使用的设备号 = `myPe + deviceOffset`。例如 rank 0、rank 1 分别运行在 Device 0、Device 1 时，`deviceOffset` 传 `0`；若 rank 1 运行在 Device 5，则该进程应传 `deviceOffset = 5 - 1 = 4`。

-   **maxKeysPerPe**（`uint32_t`）：每个 PE 单轮 `Insert` 允许的最大输入条数。多卡模式下接收缓冲区按 `nPes * maxKeysPerPe` 个桶分配，本地暂存与远程发送共用该容量。

-   **ipPort**（`const char*`）：SHMEM 通信地址，格式如 `"tcp://127.0.0.1:8998"`，所有 PE 必须使用相同地址。

### Init

```cpp
void Init();
```

无参数。完成以下初始化，并在最后执行一次全 PE 同步（`aclshmem_barrier_all`）：

-   设置 Device（`myPe + deviceOffset`）并创建 Stream。
-   初始化 SHMEM 通信域（本地对称内存池 1 GB，通信超时 300 秒）。
-   分配 Device 侧哈希表、接收缓冲区（多卡）及 Host 侧哈希表（分层模式）的对称内存，并将哈希表所有桶填充为 `{unusedKey, unusedValue}`。

### Insert

-   **keys**（`const Tkey*`）：输入 key 数组，必选参数，须为 Device 侧内存地址（由 `aclrtMalloc` 分配），长度为 `nums`。

-   **values**（`const Tvalue*`）：输入 value 数组，必选参数，须为 Device 侧内存地址，长度为 `nums`。

-   **nums**（`size_t`）：本轮插入的 key-value 对数量。多卡模式下要求 `nums <= maxKeysPerPe`，且所有 PE 均须满足。

多卡模式下的执行流程（接口内部完成，含三次全 PE barrier）：

1.  清空接收缓冲区并同步，避免重放上一轮旧记录。
2.  分发（dispatch）：属于本卡的 key 直接插入本地 Device 表（分层模式下同时暂存到接收缓冲区的本卡分区）；属于远端卡的 key 通过 SHMEM 写入目标 PE 接收缓冲区。
3.  本地插入（local insert）：各卡从接收缓冲区提取属于自己的记录，插入本地 Device 表。
4.  Host 插入（分层模式）：将本卡负责的全部记录写入本卡 Host 表，通过对映射后的 Host 桶 key 执行 SIMT CAS 原子占桶，避免不同 key 的哈希冲突覆盖；key 占桶后单独写入 value。

接口为同步语义，返回前通过 `aclrtSynchronizeStream` 等待本轮插入全部完成。

### Search

-   **keys**（`const Tkey*`）：待查询的 key 数组，必选参数，须为 Device 侧内存地址，长度为 `nums`。

-   **values**（`Tvalue*`）：输出 value 数组，必选参数，须为 Device 侧内存地址，长度为 `nums`。命中的位置写入对应 value；未命中的位置写入 `Tvalue` 的最大值（`std::numeric_limits<Tvalue>::max()`）。

-   **nums**（`size_t`）：本轮查询的 key 数量。

多卡模式下按 key 哈希值定位目标 PE：本卡的 key 直接读本地 Device 表，远端的 key 通过 SHMEM 远程读取目标 PE 的 Device 表。分层模式（`hostTableSize > 0`）下为先 Device 表、后 Host 表的分层查找。接口为同步语义。

### Finalize

无参数。释放 Device/Host 对称内存与接收缓冲区、销毁 Stream、复位 Device，并调用 `aclshmem_finalize` 退出 SHMEM。析构函数会自动调用 `Finalize`，无需重复调用。

### 辅助接口

-   **GetMyPe()**（返回 `int32_t`）：返回本 PE 编号。
-   **GetNPes()**（返回 `int32_t`）：返回 PE 总数。
-   **GetTableSize()**（返回 `uint64_t`）：返回 Device 表实际分配的桶数（约为 `tableSize / 0.75`）。

## 返回值与异常说明

生命周期接口（`Init`/`Insert`/`Search`/`Finalize`）均无返回值，错误通过 C++ 异常上报：

-   `std::runtime_error`：ACL/SHMEM 调用失败、内存分配失败或流同步失败。
-   `std::invalid_argument`：多卡模式下 `Insert` 的 `nums > maxKeysPerPe`。

## 约束说明

-   该容器面向推理场景的分布式 Embedding 查表使用。
-   多卡模式下，`Init`、`Insert`、`Search` 为集合操作（内部含全 PE barrier），所有 PE 必须以相同的顺序调用，否则可能死锁。
-   每个 PE 每轮 `Insert` 的输入数须不超过 `maxKeysPerPe`，本地暂存与远程发送共用该容量；调用前应确保所有 PE 都满足此限制。
-   `tableSize` 应不小于预期写入本卡的 key 总数，Device 表写满后新 key 会被静默丢弃；建议按 0.75 负载因子估算容量。
-   分层模式下，本卡 Host 表需容纳所有路由到本卡的记录（本卡输入与其他 PE 发来的输入），`hostTableSize` 不足时新 key 会被静默丢弃。
-   保留值约束：业务 key 不能取 `Tkey` 的最大值，业务 value 不能取 `Tvalue` 的最大值，二者分别被用作空桶占位和查询未命中的哨兵值。
-   `Search` 须在对应 `Insert` 完成后调用（接口本身为同步语义，返回即已完成）。
-   同一轮 `Insert` 中重复 key 携带不同 value 时，不保证最终写入顺序。
-   `keys`/`values` 指针须为 Device 侧地址（`aclrtMalloc` 分配），不支持 Host 侧指针。
-   环境要求：CANN 9.1.0 及以上、SHMEM 1.7.0，编译架构 `dav-3510`（Ascend 950 系列）。

## 调用示例

### 单卡模式（nPes = 1）

```cpp
#include <climits>
#include "acl/acl.h"
#include "dist_embedding_container.h"

using Key = unsigned int;
using Value = size_t;

int main()
{
    aclInit(nullptr);
    const char* ipPort = "tcp://127.0.0.1:8998";

    // tableSize=1024, myPe=0, nPes=1, deviceOffset=0（使用 Device 0）, maxKeysPerPe=16
    DistEmbeddingContainer<Key, Value> container(1024, 0, 1, 0, 16, ipPort);
    container.Init();

    // 准备 Device 侧输入
    const Key insertKeys[] = {11, 22};
    const Value insertValues[] = {1011, 2022};
    void* devKeys = nullptr;
    void* devValues = nullptr;
    aclrtMalloc(&devKeys, sizeof(insertKeys), ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc(&devValues, sizeof(insertValues), ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMemcpy(devKeys, sizeof(insertKeys), insertKeys, sizeof(insertKeys), ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(devValues, sizeof(insertValues), insertValues, sizeof(insertValues), ACL_MEMCPY_HOST_TO_DEVICE);

    container.Insert(static_cast<const Key*>(devKeys),
                     static_cast<const Value*>(devValues), 2);

    // 查询：命中返回插入值，未命中（key=99）返回 Value 最大值
    const Key searchKeys[] = {11, 22, 99};
    Value searchValues[3] = {0};
    void* devSearchKeys = nullptr;
    void* devSearchValues = nullptr;
    aclrtMalloc(&devSearchKeys, sizeof(searchKeys), ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc(&devSearchValues, sizeof(searchValues), ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMemcpy(devSearchKeys, sizeof(searchKeys), searchKeys, sizeof(searchKeys), ACL_MEMCPY_HOST_TO_DEVICE);
    container.Search(static_cast<const Key*>(devSearchKeys),
                     static_cast<Value*>(devSearchValues), 3);
    aclrtMemcpy(searchValues, sizeof(searchValues), devSearchValues, sizeof(searchValues),
                ACL_MEMCPY_DEVICE_TO_HOST);
    // searchValues = {1011, 2022, SIZE_MAX}

    aclrtFree(devKeys); aclrtFree(devValues);
    aclrtFree(devSearchKeys); aclrtFree(devSearchValues);
    container.Finalize();
    aclFinalize();
    return 0;
}
```

### 多卡模式（nPes = 2）

每个 PE 对应一个进程，各进程可运行在不同 Device 上。完整代码见 [test/test_rank2.cpp](../distributed_embedding/test/test_rank2.cpp)，核心用法如下：

```cpp
const int rank = /* 从环境变量 RANK_ID 读取 */;
const int deviceId = /* 从环境变量 DEVICE_ID 读取 */;
const char* ipPort = /* 从环境变量 SHMEM_IP_PORT 读取，各 rank 保持一致 */;

// rank 0 -> Device 0，rank 1 -> Device 1：deviceOffset 传 deviceId - rank
DistEmbeddingContainer<Key, Value> container(1024, rank, 2, deviceId - rank, 32, ipPort);
container.Init();

// 每个 rank 只插入自己的输入，key 会按哈希自动路由到归属卡
container.Insert(devKeys, devValues, nums);

// 可查询任意 key，包括路由到远端卡的 key
container.Search(devSearchKeys, devSearchValues, nums);

container.Finalize();
```

### 编译与运行

```bash
cd ops/shmem/distributed_embedding

# 仅编译单卡、双卡测试
bash test/build.sh

# 编译并运行单卡测试，默认使用 Device 0
bash test/run_rank1.sh

# 编译并运行双卡测试，默认使用 Device 0、1
RANK0_DEVICE=0 RANK1_DEVICE=1 bash test/run_rank2.sh
```

常用环境变量：

| 环境变量             | 说明                                             | 默认值                        |
| -------------------- | ------------------------------------------------ | ----------------------------- |
| `ASCEND_HOME_PATH`   | CANN 安装目录                                    | `$HOME/ascend/cann`           |
| `SHMEM_ROOT`         | SHMEM 安装目录（含 include、lib）                | `$HOME/ascend/shmem/latest/shmem` |
| `SHMEM_IP_PORT`      | SHMEM 通信地址（双卡运行，各 rank 须一致）       | `tcp://127.0.0.1:8999`        |
| `DEVICE_ID`          | 单卡测试使用的设备号                             | `0`                           |
| `RANK0_DEVICE`       | 双卡测试 rank 0 使用的设备号                     | `0`                           |
| `RANK1_DEVICE`       | 双卡测试 rank 1 使用的设备号                     | `1`                           |
| `TEST_TIMEOUT_SECONDS` | 双卡测试超时时间（秒）                         | `180`                         |

单卡测试成功输出 `PASS: Init, Insert, Search, and Finalize succeeded`；双卡测试为每个 PE 构造本地和远端 key，校验跨 PE 插入与查询结果。
