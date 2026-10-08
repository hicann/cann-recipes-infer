# D2H 功能用例

`test_rank1.cpp` 和 `test_rank2.cpp` 共用 `functional_cases.h`，分别覆盖单 PE 和双 PE。
两个程序都编译 `unsigned int / long long` 两种 Key，Value 为 `size_t`（64 位）。
`D2H_KEY_BITS=32` 或 `64` 选择运行类型；两个运行脚本默认分两次进程运行两种类型，任一失败最终返回非零。

在算子目录仅编译，不运行：

```bash
bash test/build.sh d2h_rank1 d2h_rank2
```

运行入口：

```bash
bash test/d2h/run_rank1.sh
bash test/d2h/run_rank2.sh
# 仅运行 64 位 Key：
D2H_KEY_BITS=64 bash test/d2h/run_rank2.sh
```

双卡日志为 `test/build/d2h_rank2_logs/key32_rank0.log` 等四个文件，保留原有设备、地址和超时环境变量。

## Host 路径保证

每个 PE 的 Device 表固定为 1 个桶，先为各 PE 插入一条独立记录并验证 Device 命中；之后插入的新 key 均需要回退到 `HOST_SIDE` 表查询。保留的 Device 记录位于 Host 桶 32/33，避开碰撞和绕回用例的目标桶。
碰撞与绕回按 Host 表容量生成，而不是 Device 表容量。Host 桶数按容器当前的 `0.75` 负载因子计算，并选择奇数桶数，使双 PE 的两个 owner 都能命中指定桶。

单 PE 保留 `maxKeysPerPe=0`，验证直接消费输入数组、不依赖接收缓冲区的路径。双 PE 每个 owner 均覆盖本地来源和远端来源的 Host 插入，且每个 PE 查询完整 golden，因此覆盖本地和远端 Host 查询。

## 覆盖范围

| 用例标签 | 输入和检查 | 分支目标 |
|---|---|---|
| `device_resident_hit / device_resident_update_preserves_host` | 填满单槽 Device 表并更新驻留记录，检查 Host 数据仍完整 | Device 命中与 Host 回退共存 |
| `empty_table_query` | Init 后直接查询，全部返回未命中标记 | 本地/远端查询遇空桶退出 |
| `hash_collision` | 同一 PE 的 3 个不同 key 哈希到桶 7，逐轮插入；追加同桶缺失 key | 空桶 CAS、冲突重试、碰撞链命中和未命中 |
| `probe_wraparound` | 同一 PE 的 3 个 key 哈希到 Host 表最后一个桶，逐轮插入 | 探测越过表尾回到桶 0；查询同样绕回 |
| `hits_and_misses` | 混合已有和从未插入的 key | 查询命中/未命中 |
| `existing_key_update` | 大批插入后，用小批更新已有 key，再验证整个表 | CAS 返回相同 key、旧数据保留、接收缓冲区清理 |
| `duplicate_key_same_value` | 同批重复 key/value，双卡还跨源 PE 重复 | 并发重复写入，不依赖不同 value 的竞争顺序 |
| `multiple_insert_rounds` | 多轮新增、更新、重复后再次新增 | 累积结果、发送计数复位 |
| `empty_insert_preserves_table` | 所有 PE 输入数为 0，再查全表 | 零次插入循环、空接收槽、旧记录不重放 |
| `one_empty_pe` | 双卡下一方 0 条、一方有输入 | 不等长输入和同步 |
| `key_value_boundaries_round_*` | 每个边界 key 轮流写入全部边界 value | Key 哈希高位、32/64 位 CAS、Value 高位和零值 |
| `insert_size_* / query_size_*` | 见下方规模集合，逐轮验证全表和缺失 key | 非整齐尾部、线程数上限前后、`maxKeysPerPe` 等于上限 |
| `empty_search` | Search 数量为 0，输出哨兵必须保持原值；放在最后 | 零长度查询及历史除零问题回归 |

双卡每个 PE 查询完整 golden 表，碰撞组分别覆盖两个 owner；逐轮切换输入来源，使本地插入、远端发送及接收插入均被覆盖。
碰撞生成使用按容器负载因子换算后的 Host 桶数。
预期结果由 CPU 上的输入映射维护，不读取被测表作为 golden。查询输出预填充非零字节，避免遗漏写入时零值误通过。
每次查询完成后执行 PE barrier，避免下一轮更新与另一 PE 的查询重叠。

Key 边界包括：0、1、最大可用值；32 位额外包含最高位为 1 的值；64 位额外包含最小负数、-1、2^32，以及低 32 位相同但高位不同的 key。
Key 最大值是空桶标记，不作为有效 key。Value 覆盖 0、1、2^32-1、2^32、2^32+1、2^63、最大值-1；最大值保留为未命中标记。

设实际 AIV 核数为 C，输入规模为去重后的：

- 0、1、31、32、33、2047、2048、2049；
- C-1、C、C+1；
- C×2048-1、C×2048、C×2048+1。

双 PE 的 `maxKeysPerPe` 为 C×2048+1，单 PE 为 0（不使用接收缓冲区）。大批输入复用同一组 key 的前缀，避免多轮测试累积到满表。
这些用例覆盖 Device 满表后的 Host 回退，不覆盖 Host 满表、输入超限异常、所有 Host 错误分支或未接入容器接口的 kernel。

## 空查询回归

`Search` 与 `Insert` 均使用 `CalBlockDim(std::max<size_t>(nums, 1), ...)` 计算启动参数，避免零输入导致线程数为零和除零。传给 kernel 的实际输入数量仍为 0。
`empty_search` 检查调用正常返回，且输出哨兵值不变。
